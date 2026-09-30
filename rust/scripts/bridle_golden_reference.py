#!/usr/bin/env python
"""Golden reference for the Rust BRIDLE port (`mokume_stats::batch::bridle`).

Generates the synthetic multi-dataset fixture (identical draw order to
`crates/mokume-stats/tests/bridle_golden.rs::fixture`), runs the Python prototype
`lim.py` (research/cellline-integration/lim_lin/lim.py, the benchmark `lim_lin`
variant: linear f, no MNAR, plex on, sample loading on) on it and writes the
expectation file consumed by the Rust golden test.

Only two things are patched so the prototype is deterministic and comparable:
  * torch.randn (initial U, V) draws from np.random.RandomState(seed + 1), the
    stream the Rust port uses;
  * gene_feats reads this fixture's raw features instead of the cluster path
    and returns float64 (the whole prototype runs in float64 here).

Usage:
    python rust/scripts/bridle_golden_reference.py /path/to/lim.py \
        rust/crates/mokume-stats/tests/fixtures/bridle_golden_expected.tsv
Requires numpy, pandas, pyarrow, scipy and torch.
"""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

SEED = 20260927
G = 240
NF = 6
NL = 80
RANK = 4
SWEEPS = 30
DATASETS = {
    "REF": list(range(0, 40)),
    "DSB": list(range(20, 50)),
    "DSC": list(range(30, 60)),  # TMT-like: 6 plexes x 5 profiles
    "DSD": list(range(0, 5)) + list(range(50, 62)),  # 5 anchor samples -> cross-fitted
    "SINGLE": [10],  # single-sample dataset, anchored to REF -> cross-fitted
    "UNB": list(range(62, 72)),  # unanchored
}
ORDER = ["REF", "DSB", "DSC", "DSD", "SINGLE", "UNB"]
N_PLEX = 6


def fixture():
    rs = np.random.RandomState(SEED)
    n = rs.standard_normal
    X = np.array([[n() for _ in range(NF)] for _ in range(G)])
    m = np.array([20 + 2 * n() for _ in range(G)])
    Lin = np.array([[0.5 * n() for _ in range(G)] for _ in range(4)])
    U = np.array([[n() for _ in range(3)] for _ in range(NL)])
    V = np.array([[0.4 * n() for _ in range(3)] for _ in range(G)])
    R = np.array([[0.2 * n() for _ in range(G)] for _ in range(NL)])
    lineage = [None if li % 10 == 9 else f"LIN{li % 4}" for li in range(NL)]
    lin_eff = np.array(
        [Lin[li % 4] if lineage[li] else np.zeros(G) for li in range(NL)]
    )
    theta = m[None] + lin_eff + U @ V.T + R
    A = {}
    for d in ORDER:
        a0 = 0.5 * n()
        beta = np.array([0.3 * n() for _ in range(NF)])
        r = np.array([0.15 * n() for _ in range(G)])
        A[d] = np.zeros(G) if d == "REF" else a0 + X @ beta + r
    P = np.array([[0.3 * n() for _ in range(G)] for _ in range(N_PLEX)])
    P = P - P.mean(0)
    pmiss = np.array([[rs.rand() < 0.2 for _ in range(G)] for _ in range(N_PLEX)])
    rows = []
    truth = []
    for d in ORDER:
        for j, li in enumerate(DATASETS[d]):
            c = 0.2 * n()
            eps = np.array([0.2 * n() for _ in range(G)])
            miss = np.array([rs.rand() < 0.1 for _ in range(G)])
            k = j // 5 if d == "DSC" else -1
            if k >= 0:
                miss = pmiss[k]
            y = theta[li] + A[d] + c + eps + (P[k] if k >= 0 else 0.0)
            line = f"L{li:02d}"
            plex = f"P{k}" if k >= 0 else ""
            truth.append((d, line, c))
            for g in range(G):
                if not miss[g]:
                    rows.append(
                        (d, line, f"G{g:03d}", float(y[g]), lineage[li] or "", plex)
                    )
    long = pd.DataFrame(rows, columns=["ds", "cvcl", "gene", "v", "lineage", "plex"])
    feats = pd.DataFrame(
        X, columns=[f"f{j}" for j in range(NF)], index=[f"G{g:03d}" for g in range(G)]
    )
    return long, feats, theta, A


def main(prototype_path, out_path):
    torch.set_default_dtype(torch.float64)
    spec = importlib.util.spec_from_file_location("lim", prototype_path)
    proto = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proto)

    long, feats, _, _ = fixture()
    Wd = long.pivot_table(
        index=["ds", "cvcl"], columns="gene", values="v", aggfunc="first"
    )
    X = Wd.values.astype(np.float64)
    genes = np.array(Wd.columns)
    ds_arr = Wd.index.get_level_values(0).values.astype(str)
    cv_arr = Wd.index.get_level_values(1).values.astype(str)
    linmap = dict(zip(long.cvcl, long.lineage))
    lin_arr = np.array([linmap[c] or None for c in cv_arr], dtype=object)

    # abundance axis as feats.py: reference median, fallback overall median
    pro = long[long.ds == "REF"].groupby("gene").v.median()
    allm = long.groupby("gene").v.median()
    ab = np.array([pro.get(g, np.nan) for g in genes])
    ab = np.where(np.isnan(ab), np.array([allm[g] for g in genes]), ab)
    tmp = tempfile.mkdtemp()
    F = feats.reindex(genes).copy()
    F["abund"] = ab
    F.to_parquet(os.path.join(tmp, "feats_raw.parquet"))
    proto.W = tmp
    orig_gene_feats = proto.gene_feats

    def gene_feats64(genes_, cfg_):
        Xg, cols, abz = orig_gene_feats(genes_, cfg_)
        return Xg.astype(np.float64), cols, abz.astype(np.float64)

    proto.gene_feats = gene_feats64
    cfg = proto.Cfg(proto.DEFAULT, mnar=False, rank=RANK, sweeps=SWEEPS, ref="REF")
    init_rs = np.random.RandomState(cfg.seed + 1)
    torch.randn = lambda *shape: torch.tensor(init_rs.standard_normal(shape))

    V, info, extra = proto.fit(X, genes, ds_arr, cv_arr, lin_arr, cfg)
    print(
        "plex:",
        info["pinfo"],
        "cf:",
        info["cf"],
        "sweeps:",
        len(info["hist"]),
        file=sys.stderr,
    )

    # expectation: every 10th observed cell (row-major) + per-dataset summaries
    out = []
    obs = np.argwhere(np.isfinite(X))
    for idx, (i, g) in enumerate(obs):
        if idx % 10 == 0:
            out.append(("cell", ds_arr[i], cv_arr[i], genes[g], repr(float(V[i, g]))))
    for d in sorted(set(ds_arr)):
        vv = V[ds_arr == d]
        vv = vv[np.isfinite(vv)]
        out.append(("ds_mean", d, "", "", repr(float(vv.mean()))))
        out.append(("ds_sd", d, "", "", repr(float(vv.std()))))
    out.append(("hold_mse_last", "", "", "", repr(float(info["hist"][-1]["hold_mse"]))))
    out.append(("n_sweeps", "", "", "", str(len(info["hist"]))))
    out.append(("tauR", "", "", "", repr(float(info["tauR"]))))
    with open(out_path, "w") as fh:
        fh.write(
            "# generated by rust/scripts/bridle_golden_reference.py from lim_lin/lim.py "
            f"(seed={SEED}, rank={RANK}, sweeps={SWEEPS}, float64)\n"
        )
        fh.write("kind\tds\tline\tgene\tvalue\n")
        for row in out:
            fh.write("\t".join(row) + "\n")
    print(f"wrote {len(out)} rows to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
