#!/usr/bin/env python
"""Golden reference for the benchmark-winning BRIDLE configuration.

Generates a synthetic collection (identical draw order to
`crates/mokume-stats/tests/bridle_benchmark_golden.rs::fixture`) and runs the
Python reference of the 2026-10 benchmark arm `A3gnc_phdelta_all` on it:

  1. graphp offsets (`B0-protocol/scripts/arms_simple.py::graph("pb", 3)`,
     centring fallback `centre`), turned into the per-(dataset, gene) prior
     exactly as `B1b-headtohead/scripts/bvar_run.py gprior`;
  2. `bridle_var.fit_var(prior_A=..., stop_tol=..., min_stop=...)`
     (`B1b-headtohead/scripts/bridle_var.py`, on `bridle_py.py`);
  3. `out + c` (`bvar_run.py addc`);
  4. the label-blind plex rescale of `B1d-phdelta-allrows/scripts/
     posthoc_delta_all.py`.

`arms_simple.py` and `posthoc_delta_all.py` are scripts with cluster paths, so
their functions are copied below verbatim (only the input frame is passed in
instead of read from disk). `bridle_var.py` / `bridle_py.py` are imported from
the directory given on the command line; the only patch is
`bridle_py.DESIGN_COLS`, set to this fixture's feature names.

Usage:
    python rust/scripts/bridle_benchmark_reference.py \\
        /path/to/lecture-figures/2026-10-bridle-x/B1b-headtohead/scripts \\
        rust/crates/mokume-stats/tests/fixtures/bridle_benchmark_expected.tsv
Requires numpy, pandas and scipy.
"""

import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

SEED = 20261003
G = 200
NF = 6
NL = 44
RANK = 4
SWEEPS = 120
MIN_STOP = 40
N_PLEX = 3
REF = "REF"
DATASETS = {
    "REF": list(range(0, 25)),  # reference
    "TMT": list(range(8, 26)),  # 3 plexes x 6 profiles, plex P1 inflated x1.6
    "LFQ": list(range(20, 35)),
    "SMALL": list(range(0, 4)),  # 4 anchor samples, graph-linked, cross-fitted
    "SINGLE": [5],  # single-line dataset
    "UNL": [0, 1] + list(range(35, 44)),  # 2 shared lines: not linked -> centred
}
ORDER = ["REF", "TMT", "LFQ", "SMALL", "SINGLE", "UNL"]


def fixture():
    rs = np.random.RandomState(SEED)
    n = rs.standard_normal
    X = np.array([[n() for _ in range(NF)] for _ in range(G)])
    m = np.array([20 + 2 * n() for _ in range(G)])
    U = np.array([[n() for _ in range(3)] for _ in range(NL)])
    V = np.array([[0.4 * n() for _ in range(3)] for _ in range(G)])
    R = np.array([[0.2 * n() for _ in range(G)] for _ in range(NL)])
    theta = m[None] + U @ V.T + R
    A = {}
    for d in ORDER:
        a0 = 0.5 * n()
        beta = np.array([0.3 * n() for _ in range(NF)])
        r = np.array([0.15 * n() for _ in range(G)])
        A[d] = np.zeros(G) if d == REF else a0 + X @ beta + r
    P = np.array([[0.3 * n() for _ in range(G)] for _ in range(N_PLEX)])
    P = P - P.mean(0)
    pmiss = np.array([[rs.rand() < 0.2 for _ in range(G)] for _ in range(N_PLEX)])
    rows = []
    for d in ORDER:
        for j, li in enumerate(DATASETS[d]):
            c = 0.2 * n()
            eps = np.array([0.2 * n() for _ in range(G)])
            miss = np.array([rs.rand() < 0.1 for _ in range(G)])
            k = j // 6 if d == "TMT" else -1
            if k >= 0:
                miss = pmiss[k]
            scale = 1.6 if k == 1 else 1.0
            y = m + scale * (theta[li] - m) + A[d] + c + eps + (P[k] if k >= 0 else 0.0)
            for g in range(G):
                if not miss[g]:
                    rows.append(
                        (
                            d,
                            f"L{li:02d}",
                            f"G{g:03d}",
                            float(y[g]),
                            f"P{k}" if k >= 0 else "",
                        )
                    )
    long = pd.DataFrame(rows, columns=["ds", "cvcl", "gene", "v", "plex"])
    feats = pd.DataFrame(
        X, columns=[f"f{j}" for j in range(NF)], index=[f"G{g:03d}" for g in range(G)]
    )
    return long, feats


# ---------------- arms_simple.py (verbatim, `d` passed in) ----------------
def centre(v, batch, gene, cvcl, ds):
    df = pd.DataFrame({"v": v, "b": batch, "g": gene, "c": cvcl})
    nl = df.groupby("b").c.nunique()
    g = df.groupby(["b", "g"]).v
    mu = g.transform("mean")
    n = g.transform("count")
    ref = df[ds == REF].groupby("g").v.mean()
    allm = df.groupby(["b", "g"]).v.mean().groupby("g").mean()
    G_ = df.g.map(ref).fillna(df.g.map(allm))
    ok = (df.b.map(nl) >= 5) & (n >= 3)
    return np.where(ok, v - mu + G_, v), G_.values


def graph(d, batchcol, min_shared, niter=400):
    b = pd.Categorical(d[batchcol])
    gcat = pd.Categorical(d.gene)
    lcat = pd.Categorical(d.cvcl)
    ng = len(gcat.categories)
    bg = b.codes.astype(np.int64) * ng + gcat.codes
    lg = lcat.codes.astype(np.int64) * ng + gcat.codes
    cnt = np.bincount(lg, minlength=lg.max() + 1)
    A = cnt[lg] >= 2
    v = d.v.values
    ia = np.where(A)[0]
    bga = bg[ia]
    lga = lg[ia]
    va = v[ia]
    ub, bi = np.unique(bga, return_inverse=True)
    ul, li = np.unique(lga, return_inverse=True)
    nbc = np.bincount(bi)
    nlc = np.bincount(li)
    o = np.zeros(len(ub))
    for _ in range(niter):
        th = np.bincount(li, va - o[bi]) / nlc
        on = np.bincount(bi, va - th[li]) / nbc
        o = on
    df = pd.DataFrame({"l": lga, "n": bi})
    pr = df.merge(df, on="l")
    pr = pr[pr.n_x < pr.n_y]
    e = pr.groupby(["n_x", "n_y"]).size()
    e = e[e >= min_shared]
    Gm = sp.coo_matrix(
        (np.ones(len(e)), (e.index.get_level_values(0), e.index.get_level_values(1))),
        shape=(len(ub), len(ub)),
    )
    _, comp = connected_components(Gm, directed=False)
    gene_of = ub % ng
    bat_of = ub // ng
    refb = list(b.categories).index(REF)
    refnode = pd.Series(np.where(bat_of == refb)[0], index=gene_of[bat_of == refb])
    rn = pd.Series(gene_of).map(refnode).values
    has = ~np.isnan(rn)
    rn2 = np.where(has, rn, 0).astype(int)
    acc = has & (comp == comp[rn2])
    off = np.where(acc, o - o[rn2], np.nan)
    pos = np.searchsorted(ub, bg)
    pos = np.clip(pos, 0, len(ub) - 1)
    hit = ub[pos] == bg
    offc = np.where(hit, off[pos], np.nan)
    cen, _ = centre(v, d[batchcol].values, d.gene.values, d.cvcl.values, d.ds.values)
    return np.where(np.isfinite(offc), v - offc, cen)


# ---------------- bvar_run.py gprior (verbatim aggregation) ----------------
def gprior(inp, graphp_value):
    J = inp.copy()
    J["value"] = graphp_value
    J["off"] = J.v - J.value
    Q = J.groupby(["ds", "gene"]).off.agg(["mean", lambda x: (x.abs() > 0).any()])
    Q.columns = ["mean", "nz"]
    Q = Q[Q.nz]["mean"].unstack()
    studies = np.array(sorted(inp.ds.unique()))
    genes = np.array(sorted(inp.gene.unique()))
    return Q.reindex(index=studies, columns=genes).values.astype(float)


# ---------------- posthoc_delta_all.py (verbatim, frame passed in) ----------------
def posthoc_delta_all(A):
    A = A.copy()
    A["batch"] = np.where(A.plex.notna(), A.study + "|" + A.plex.astype(str), A.study)
    tr = np.ones(len(A), bool)
    T = A[tr & np.isfinite(A.value)]
    g = T.groupby(["batch", "gene"]).value
    S = pd.DataFrame({"n": g.size(), "mean": g.mean(), "var": g.var()}).reset_index()
    S["ss"] = S["var"] * (S.n - 1)
    P = (
        S[S.n >= 2]
        .groupby("gene")
        .agg(ss=("ss", "sum"), n=("n", "sum"), nb=("n", "size"))
    )
    P["sig2"] = P.ss / (P.n - P.nb)
    P["df"] = P.n - P.nb
    S = S.merge(P[["sig2", "df"]], left_on="gene", right_index=True, how="left")
    ok = (S.n >= 5) & (S["var"] > 1e-8) & (S.sig2 > 1e-8)
    S["lh"] = np.where(ok, np.log(S["var"] / S.sig2) + 1 / (S.n - 1) - 1 / S.df, np.nan)
    S["sv"] = 2 / (S.n - 1)
    est = {}
    for b, d in S[ok].groupby("batch"):
        if len(d) < 50:
            continue
        x, v = d.lh.values, d.sv.values
        t2 = max(float(x.var() - v.mean()), 1e-4)
        for _ in range(5):
            wq = 1 / (v + t2)
            mu = (wq * x).sum() / (wq.sum() + 1 / 0.25)
            t2 = max(float(((x - mu) ** 2 - v).mean()), 1e-4)
        est[b] = (mu, dict(zip(d.gene, (x / v + mu / t2) / (1 / v + 1 / t2))))
    for b in S.batch.unique():
        if b in est:
            continue
        st = b.split("|")[0]
        sib = [est[x][0] for x in est if "|" in b and x.startswith(st + "|")]
        est[b] = (float(np.median(sib)) if sib else 0.0, {})
    S["l"] = [est[b][1].get(gn, est[b][0]) for b, gn in zip(S.batch, S.gene)]
    S["delta"] = np.clip(np.exp(S.l / 2), 0.25, 4)
    A = A.merge(S[["batch", "gene", "mean", "delta"]], on=["batch", "gene"], how="left")
    has = A.delta.notna()
    A.loc[has, "value"] = (
        A["mean"][has] + (A.value[has] - A["mean"][has]) / A.delta[has]
    )
    mus = {b: e[0] for b, e in est.items()}
    return A, mus


def main(ref_dir, out_path):
    sys.path.insert(0, ref_dir)
    import bridle_py
    import bridle_var as bv

    bridle_py.DESIGN_COLS = [f"f{j}" for j in range(NF)]
    long, feats = fixture()
    inp = long[["ds", "cvcl", "gene", "v"]]
    plex = {(d, c): p for d, c, p in zip(long.ds, long.cvcl, long.plex) if p}

    # 1. graphp prior (batch = ds:plex, else ds)
    d = long.copy()
    d["pb"] = np.where(d.plex != "", d.ds + ":" + d.plex, d.ds)
    prior_A = gprior(inp, graph(d, "pb", 3))

    # 2-3. fit (gprior) + keep c; first without the stop rule to place the
    # tolerance between two consecutive output changes
    kw = dict(ref=REF, rank=RANK, seed=0, prior_A=prior_A, min_stop=MIN_STOP)
    probe = bv.fit_var(inp, plex, feats, sweeps=SWEEPS, **kw)
    dout = np.array([h["dout"] for h in probe["history"]])
    k = MIN_STOP + 10
    assert np.all(np.diff(dout[MIN_STOP - 1 : k + 1]) < 0), dout[MIN_STOP - 1 : k + 1]
    tol = float(np.sqrt(dout[k - 1] * dout[k]))
    res = bv.fit_var(inp, plex, feats, sweeps=SWEEPS, stop_tol=tol, **kw)
    print(
        "stop sweep",
        res["stop_sweep"],
        "tol",
        tol,
        "dout around",
        dout[k - 2 : k + 1],
        file=sys.stderr,
    )
    res["out"] = res["out"] + res["c"][:, None]
    D = bv.to_long(res)

    # 4. post-fit plex rescale (plex labels as in the fit input)
    D = D.rename(columns={"ds": "study", "cvcl": "line_label", "v": "value"})
    D["plex"] = [plex.get((s, c)) for s, c in zip(D.study, D.line_label)]
    F, mus = posthoc_delta_all(D)

    out = []
    for s, row in zip(res["studies"], prior_A):
        for gname, val in zip(res["genes"], row):
            if np.isfinite(val):
                out.append(("prior", s, "", gname, repr(float(val))))
    fit_v = D.value.values
    for idx in range(0, len(D), 5):
        out.append(
            (
                "fit",
                D.study.iat[idx],
                D.line_label.iat[idx],
                D.gene.iat[idx],
                repr(float(fit_v[idx])),
            )
        )
    for idx in range(0, len(F), 5):
        out.append(
            (
                "cell",
                F.study.iat[idx],
                F.line_label.iat[idx],
                F.gene.iat[idx],
                repr(float(F.value.iat[idx])),
            )
        )
    for b in sorted(mus):
        out.append(("mu", b, "", "", repr(float(mus[b]))))
    out.append(("n_sweeps", "", "", "", str(len(res["history"]))))
    out.append(("stop_tol", "", "", "", repr(tol)))
    out.append(("dout_last", "", "", "", repr(float(res["history"][-1]["dout"]))))
    with open(out_path, "w") as fh:
        fh.write(
            "# generated by rust/scripts/bridle_benchmark_reference.py from bridle_var.py "
            f"(gprior + addc) + posthoc_delta_all.py (seed={SEED}, rank={RANK}, "
            f"sweeps<={SWEEPS}, min_stop={MIN_STOP})\n"
        )
        fh.write("kind\tds\tline\tgene\tvalue\n")
        for row in out:
            fh.write("\t".join(row) + "\n")
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
