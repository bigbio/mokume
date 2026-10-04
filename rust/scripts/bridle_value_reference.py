#!/usr/bin/env python
"""Reference values for the BRIDLE per-dataset value report
(`mokume_stats::batch::bridle::value`).

Generates the synthetic collection of `bridle/value_tests.rs::fixture` (same
NumPy `RandomState` draw order) and applies the B4 prototype definitions
(`lecture-figures/2026-10-bridle-x/B4-dataset-value/scripts/b4_desc.py`,
sections 1, 2d and 4) to it. The prototype blocks are copied with three
substitutions, which are the documented Rust deviations:
  * the honest LLO values `H` are the fixture's corrected values (one fit);
  * the replicate noise `v_noise` is absent: the NNLS target is
    median(e^2) / 0.4549 of the plain cross-dataset difference e = v_a - v_b
    (vn = 0), and the redundancy weight is 1 / (per-cell fitted variance);
  * agree_med / disagree_var are computed on the same differences.

Usage:
    python rust/scripts/bridle_value_reference.py \
        rust/crates/mokume-stats/tests/fixtures/bridle_value_expected.tsv
Requires numpy, pandas and scipy.
"""

import sys

import numpy as np
import pandas as pd
from scipy.optimize import nnls

SEED = 20261004
G = 300
NL = 20
# name, lines, offset, noise sd, fitted variance
PLAN = [
    ("REF", list(range(0, 16)), 0.0, 0.2, 0.04),
    ("B", list(range(5, 19)), 1.0, 0.2, 0.04),
    ("C", list(range(0, 13)), -0.5, 0.3, 0.09),
    ("DUP", list(range(5, 11)), 0.7, 0.0, 0.04),  # copy of B + 0.01 noise
    ("BAD", list(range(0, 5)), 0.3, 0.2, 1.0),  # genes reversed
    ("SINGLE", [19], 0.2, 0.2, 0.04),  # only source of L19
]
MISS = 0.1


def fixture():
    rs = np.random.RandomState(SEED)
    n = rs.standard_normal
    m = [20 + 2 * n() for _ in range(G)]
    e = [[0.6 * n() for _ in range(G)] for _ in range(NL)]
    rna = [[e[l][g] + 0.5 * n() for g in range(G)] for l in range(NL)]
    truth = [[m[g] + e[l][g] for g in range(G)] for l in range(NL)]
    rows, corr = [], {}
    for name, lines, off, sd, var in PLAN:
        off_g = [off + 0.3 * n() for _ in range(G)]
        for l in lines:
            for g in range(G):
                z = n()
                miss = rs.random_sample() < MISS
                if name == "DUP":
                    v = corr[("B", l, g)] + 0.01 * z
                elif name == "BAD":
                    v = truth[l][G - 1 - g] + sd * z
                else:
                    v = truth[l][g] + sd * z
                corr[(name, l, g)] = v
                if not miss and np.isfinite(v):
                    rows.append((name, f"L{l:02d}", f"G{g:03d}", v + off_g[g], v, var))
                elif name == "B":
                    corr[(name, l, g)] = np.nan
    L = pd.DataFrame(rows, columns=["study", "line", "gene", "value", "corr", "v_noise"])
    R = pd.DataFrame(rna, index=[f"L{l:02d}" for l in range(NL)], columns=[f"G{g:03d}" for g in range(G)])
    return L, R


def main(out):
    L, RNA = fixture()
    H = L.rename(columns={"value": "raw"}).rename(columns={"corr": "value"})
    # ---------------- 1. coverage (b4_desc.py)
    prof = L[["study", "line"]].drop_duplicates()
    nst = prof.groupby("line").study.nunique()
    gst = L[["study", "gene"]].drop_duplicates().groupby("gene").study.nunique()
    cell_st = L.groupby(["line", "gene"]).study.nunique()
    rows = []
    for s, d in L.groupby("study"):
        ls = set(d.line)
        oth = prof[(prof.study != s) & prof.line.isin(ls)]
        partners = oth.groupby("study").line.nunique()
        ug = [g for g in d.gene.unique() if gst[g] == 1]
        uc = int((cell_st.reindex(pd.MultiIndex.from_frame(d[["line", "gene"]])).values == 1).sum())
        rows.append(dict(study=s, n_lines=len(ls), n_cells=len(d), n_genes=d.gene.nunique(),
                         genes_per_profile=float(d.groupby("line").gene.size().median()),
                         uniq_lines=int(sum(nst[l] == 1 for l in ls)), anchor_lines=int(sum(nst[l] >= 2 for l in ls)),
                         anchor_partners=int((partners >= 3).sum()), partners_any=int(len(partners)),
                         bridge_lines=int(sum(nst[l] == 2 for l in ls)), uniq_genes=len(ug), uniq_gene_cells=uc))
    C = pd.DataFrame(rows).set_index("study")
    # ---------------- cross-dataset differences (substitute for the scorer pairs)
    W0 = H.pivot_table(index=["study", "line"], columns="gene", values="value", aggfunc="first")
    pairs = []
    for line, idx in W0.groupby(level=1).groups.items():
        idx = list(idx)
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                ea = W0.loc[idx[a]].values - W0.loc[idx[b]].values
                ok = np.isfinite(ea)
                sa, sb = sorted([idx[a][0], idx[b][0]])
                pairs.append(pd.DataFrame({"study_a": sa, "study_b": sb, "e": ea[ok]}))
    K = pd.concat(pairs, ignore_index=True)
    K["e2"] = K.e ** 2
    both = pd.concat([K.assign(study=K.study_a), K.assign(study=K.study_b)])
    C["pair_cells"] = both.groupby("study").size()
    C["agree_med"] = both.groupby("study").e.apply(lambda x: float(np.median(np.abs(x))))
    C["disagree_var"] = both.groupby("study").e2.median() / 0.4549
    pp = K.groupby(["study_a", "study_b"]).agg(e2=("e2", "median"), n=("e2", "size")).reset_index()
    pp = pp[pp.n >= 200]
    st = sorted(set(pp.study_a) | set(pp.study_b))
    si = {s: i for i, s in enumerate(st)}
    Xm = np.zeros((len(pp), len(st)))
    yv = (pp.e2 / 0.4549 - 0.0).values
    w = np.sqrt(np.log1p(pp.n.values))
    for r, (a, b) in enumerate(zip(pp.study_a, pp.study_b)):
        Xm[r, si[a]] = 1
        Xm[r, si[b]] = 1
    dfit, _ = nnls(Xm * w[:, None], yv * w)
    C["excess_var"] = pd.Series(dict(zip(st, dfit)))
    # ---------------- 2d. identity (b4_desc.py)
    W = W0 - W0.median()
    gg = [x for x in W.columns if x in RNA.columns]
    R = RNA[gg].astype(float)
    R = R - R.mean()
    Rv = R.values
    rpos = {l: i for i, l in enumerate(R.index.values)}
    Wv = W[gg].values
    idr = []
    for i, (s, l) in enumerate(W.index):
        x = Wv[i]
        o = np.isfinite(x)
        if o.sum() < 200 or l not in rpos:
            idr.append((s, l, np.nan, np.nan))
            continue
        xs = x[o] - x[o].mean()
        Rs = Rv[:, o]
        Rs = Rs - Rs.mean(1, keepdims=True)
        r = (Rs @ xs) / np.sqrt((Rs ** 2).sum(1) * (xs ** 2).sum() + 1e-12)
        idr.append((s, l, float(r[rpos[l]]), int((r > r[rpos[l]]).sum() + 1)))
    IDR = pd.DataFrame(idr, columns=["study", "line", "rna_r_self", "rna_rank"])
    Xp = W.values
    Mk = np.isfinite(Xp).astype(np.float64)
    X0 = np.nan_to_num(Xp)
    nn = Mk @ Mk.T
    s1 = X0 @ Mk.T
    ss = (X0 ** 2) @ Mk.T
    sxy = X0 @ X0.T
    with np.errstate(invalid="ignore", divide="ignore"):
        Rp = (sxy - s1 * s1.T / nn) / np.sqrt((ss - s1 ** 2 / nn) * (ss.T - (s1.T) ** 2 / nn))
    Rp[nn < 200] = np.nan
    pst = W.index.get_level_values(0).values
    pln = W.index.get_level_values(1).values
    idp = []
    for i in range(len(pst)):
        oth = pst != pst[i]
        same = oth & (pln == pln[i])
        ra = Rp[i, oth]
        j = np.where(oth)[0][np.nanargmax(ra)] if np.isfinite(ra).any() else None
        top = (float(np.nanmax(ra)), pln[j], float(np.nanmedian(ra))) if j is not None else (np.nan, "", np.nan)
        if not same.any():
            idp.append((pst[i], pln[i], np.nan, np.nan, *top))
            continue
        rs = np.nanmax(Rp[i, same])
        ro = Rp[i, oth & ~same]
        idp.append((pst[i], pln[i], float(np.nanmean(Rp[i, same])), int(np.nansum(ro > rs) + 1), *top))
    IDP = pd.DataFrame(idp, columns=["study", "line", "self_r", "self_rank", "best_r", "best_line", "med_r_any"])
    gmed = L.groupby(["study", "gene"]).value.median().unstack(0)
    ab = []
    for (s, l), d in L.groupby(["study", "line"]):
        ref = gmed.drop(columns=s).median(1).reindex(d.gene.values).values
        o = np.isfinite(ref)
        ab.append((s, l, float(pd.Series(d.value.values[o]).rank().corr(pd.Series(ref[o]).rank()))))
    AB = pd.DataFrame(ab, columns=["study", "line", "abund_rho"])
    ID = IDR.merge(IDP, on=["study", "line"]).merge(AB, on=["study", "line"], how="outer")
    gi = ID.groupby("study")
    C["id_rna_self"] = gi.rna_r_self.median()
    C["id_rna_rank"] = gi.rna_rank.median()
    C["id_rna_top1"] = gi.rna_rank.apply(lambda x: float((x == 1).mean()) if x.notna().any() else np.nan)
    C["id_rna_top5"] = gi.rna_rank.apply(lambda x: float((x <= 5).mean()) if x.notna().any() else np.nan)
    C["id_self_r"] = gi.self_r.median()
    C["id_rank"] = gi.self_rank.median()
    C["id_r_max_any"] = gi.best_r.median()
    C["id_r_med_any"] = gi.med_r_any.median()
    C["abund_rho"] = gi.abund_rho.median()
    C["abund_rho_min"] = gi.abund_rho.min()
    # ---------------- 4. redundancy (b4_desc.py; var = per-cell fitted variance)
    Hs = H[H.line.isin(nst[nst >= 2].index)].copy()
    Hs["w"] = 1 / Hs.v_noise
    Hs["wv"] = Hs.w * Hs.value
    g = Hs.groupby(["line", "gene"])
    Hs["Sw"] = g.w.transform("sum")
    Hs["Swv"] = g.wv.transform("sum")
    Hs["nsrc"] = g.w.transform("size")
    Hs = Hs[Hs.nsrc >= 2].copy()
    cons = Hs.Swv / Hs.Sw
    cons_wo = (Hs.Swv - Hs.wv) / (Hs.Sw - Hs.w)
    Hs["shift"] = (cons - cons_wo).abs()
    Hs["se_gain"] = 1 - np.sqrt((Hs.Sw - Hs.w) / Hs.Sw)
    Hs["wshare"] = Hs.w / Hs.Sw
    LS = Hs.groupby(["study", "line"]).agg(marg_shift=("shift", "median"), marg_se_gain=("se_gain", "median"),
                                           wshare=("wshare", "median")).reset_index()
    LS["n_other_studies"] = LS.line.map(nst) - 1
    gl = LS.groupby("study")
    C["marg_shift"] = gl.marg_shift.median()
    C["marg_se_gain"] = gl.marg_se_gain.median()
    C["wshare"] = gl.wshare.median()
    C["redund_ge3"] = gl.n_other_studies.apply(lambda x: float((x >= 3).mean()))
    C["other_src_med"] = gl.n_other_studies.median()
    P = ID.merge(LS.drop(columns="n_other_studies"), on=["study", "line"], how="left")
    with open(out, "w") as fh:
        fh.write(f"# generated by rust/scripts/bridle_value_reference.py (seed={SEED}, G={G}) from the b4_desc.py definitions\n")
        fh.write("kind\tdataset\tline\tmetric\tvalue\n")
        for s, r in C.iterrows():
            for k, v in r.items():
                fh.write(f"dataset\t{s}\t\t{k}\t{float(v)!r}\n")
        for _, r in P.iterrows():
            for k in ["abund_rho", "self_r", "self_rank", "best_r", "med_r_any", "rna_r_self", "rna_rank",
                      "marg_shift", "marg_se_gain", "wshare"]:
                fh.write(f"profile\t{r.study}\t{r.line}\t{k}\t{float(r[k])!r}\n")
            fh.write(f"profile\t{r.study}\t{r.line}\tbest_line\t{r.best_line}\n")


if __name__ == "__main__":
    main(sys.argv[1])
