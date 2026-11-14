"""Non-ML fusion baseline: no-station model + kernel-interpolated station residuals ("residual kriging" lite).

The station-fusion network (plan group "obs") has to beat this to justify itself. For each no-station run, a
station's daily departure from the MODEL (not from POWER, so the model's own correction is not counted twice) is
de-biased exactly as in scripts/wind_obs_store.py (own train-period mean and annual harmonic), then spread with the
same Gaussian kernels and the SAME input stations per day (random 60% outside the holdout, fold f inside) and added
back to the model. Scored by scripts/wind_obs_eval.py's Fold: ISD only at the withheld fold, Mesonet never an input.

Variants: +dep25, +dep75, +blend (25 km where its kernel weight >= 0.5, else 75 km).

Fusion runs (w10o0_*) are also accepted: each fold is predicted on its own store, and the network's residuals at its
own input stations are spread and added (stacking).

Usage: python scripts/wind_obs_hybrid.py [--runs w10_L_s1337 ...] [--out obs_hybrid]   -> outputs/wind/<out>.{json,md}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
import wind_inference_maps as wim                                                 # noqa: E402
import wind_obs_eval as woe                                                       # noqa: E402
import wind_obs_store as wos                                                      # noqa: E402

OUT = REPO / "outputs" / "wind"


def main(runs, out="obs_hybrid"):
    meta = json.loads((OUT / "obs_stations.json").read_text())["stations"]
    df = pd.DataFrame(meta)
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    st = wim.store_for("w10")
    times, n = st.times, st.ds.sizes["time"]
    O = d.pivot_table(index="date", columns="key", values="ws").reindex(times).reindex(columns=df.key.values)
    O = O.values.astype("float32").T                                         # (stations, days)
    rows, cols = df.row.values, df.col.values
    tr = st.split == "train"
    ang = 2 * np.pi * st.doy / 365.25
    H = np.c_[np.ones(n), np.sin(ang), np.cos(ang)]
    folds = [woe.Fold(f, meta, isd) for f in (0, 1)]
    res = {}
    for run in runs:
        fusion = run.startswith("w10o0_")          # fusion run: predict each fold on its own store (stacking)
        A = None if fusion else wim.predict(st, run, np.arange(n))           # anomaly, all days
        for F in folds:
            if fusion:
                A = wim.predict(F.st, run, np.arange(n))
            M = A[:, rows, cols].T + st.clim[st.doy][:, rows, cols].T        # (stations, days) absolute
            D = O - M
            for s in range(len(df)):
                ok = np.isfinite(D[s]) & tr
                if ok.sum() >= 50:
                    D[s] -= H @ np.linalg.lstsq(H[ok], D[s, ok], rcond=None)[0]
                else:
                    D[s] = np.nan
            ch = wos.channels(st, df, D, fold_input=(df.fold.values == F.f))[F.te]   # (test days, 4, H, W)
            Ate = A[F.te]
            blend = np.where(ch[:, 2] >= 0.5, ch[:, 0], ch[:, 1])
            for lab, add in (("+dep25", ch[:, 0]), ("+dep75", ch[:, 1]), ("+blend", blend)):
                Man = Ate + add
                res.setdefault(f"{run}{lab}", {})[F.f] = F.score(Man + F.clim, Man)
            res.setdefault(run, {})[F.f] = F.score(Ate + F.clim, Ate)
        print(f"{run} done", flush=True)
    mean = {k: woe.mean_folds(v[0], v[1]) for k, v in res.items()}
    (OUT / f"{out}.json").write_text(json.dumps({"per_fold": res, "mean": mean}, indent=1, default=float))
    lines = ["| Variant | grid RMSE | grid bias | SSIM | r2 <100 km | sd <100 | r2 100-300 | ISD de-biased | ISD r_dep | "
             "Meso de-biased | Meso r_dep |", "|" + "---|" * 11]
    for k, m in mean.items():
        g, b, i, s = m["grid"], m["bands"], m["isd"] or {}, m["meso"] or {}
        lines.append(f"| {k} | {g['rmse']:.3f} | {g['bias']:+.3f} | {g['ssim']:.3f} | {b['lt100']['r2']:.2f} | "
                     f"{b['lt100']['sd_ratio']:.2f} | {b['100_300']['r2']:.2f} | {i.get('rmse_db', np.nan):.3f} | "
                     f"{i.get('r_dep', np.nan):.3f} | {s.get('rmse_db', np.nan):.3f} | {s.get('r_dep', np.nan):.3f} |")
    txt = "Model + kernel-interpolated station residuals, 2023, mean of both folds.\n\n" + "\n".join(lines) + "\n"
    (OUT / f"{out}.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["w10_L_s1337", "w10_L_s7", "w10_L_s42"])
    ap.add_argument("--out", default="obs_hybrid")
    a = ap.parse_args()
    main(a.runs, a.out)
