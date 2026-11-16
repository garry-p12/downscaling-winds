"""Two-stage control: does training the fusion network on 2019-22 match the 2022 corrector? (WIND_WORK_LOG 7d)

The best 2023 result so far is two-stage: the LO fusion network trained on 2019-21, then a corrector (MSE or
diffusion) trained on that network's out-of-sample 2022 errors. Plan group "obs4" retrains the same network, same
channels and seeds, on 2019-22 (DOWNSCALE_TRAIN_SPLITS=train,val). If the 2019-22 network gets close to the corrector,
the gain was simply "use the latest year"; if not, learning from out-of-sample errors is what helps.

All on the same footing as wind_diffusion_decomp.py: both fold stores, 2023 holdout land cells vs AORC (anomaly
space), withheld ISD fold and Mesonet MAE in m/s, mean of the two folds.

Usage (repo root, after the obs4 training): python scripts/wind_two_stage.py   -> outputs/wind/two_stage.{md,json}
"""
from __future__ import annotations

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

OUT = REPO / "outputs" / "wind"
DIF = OUT / "diffusion"
SEEDS = (1337, 7, 42)


def station_mae(F, X, rows):
    e = []
    for _, i, j, o in rows:
        ok = np.isfinite(o)
        if ok.sum() >= 100:
            e.append(np.mean(np.abs(X[ok, i, j] + F.clim[ok, i, j] - o[ok])))
    return float(np.mean(e))


def main():
    meta = json.loads((OUT / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    res = {}
    for f in (0, 1):
        F = woe.Fold(f, meta, isd)
        m = F.mask
        y21 = np.mean([wim.predict(F.st, f"w10o0_LO_s{s}", F.te) for s in SEEDS], 0)
        y22 = np.mean([wim.predict(F.st, f"w10o0_LO4y_s{s}", F.te) for s in SEEDS], 0)
        b22 = float(np.nanmean((y22 - F.Ta)[:, m]))
        rows = {
            "network 2019-21 (3 seeds)": y21,
            "network 2019-22 (3 seeds)": y22,
            "network 2019-22 minus its own 2023 bias (oracle)": y22 - b22,
            # the MSE corrector has one member, which is its prediction
            "network 2019-21 + MSE corrector on 2022":
                np.asarray(np.load(DIF / f"obsvaldet_w10o{f}_test.npy", mmap_mode="r")[0]),
            "network 2019-21 + diffusion on 2022 (ens. mean)":
                np.asarray(np.load(DIF / f"obsval_w10o{f}_test.npy", mmap_mode="r").mean(0)),
        }
        det_saved = np.load(DIF / f"obsval_w10o{f}_det.npy")
        res[f] = {}
        for k, X in rows.items():
            err = (X - F.Ta)[:, m]
            res[f][k] = dict(rmse=float(np.sqrt(np.nanmean(err ** 2))), bias=float(np.nanmean(err)),
                             isd_mae=station_mae(F, X, F.eval_isd), meso_mae=station_mae(F, X, F.eval_meso))
        res[f]["check: saved det vs 2019-21 seed mean, max abs diff"] = float(np.nanmax(np.abs(det_saved - y21)))
        print(f"fold {f}:", json.dumps(res[f], indent=1), flush=True)
    keys = [k for k in res[0] if isinstance(res[0][k], dict)]
    mean = {k: {s: (res[0][k][s] + res[1][k][s]) / 2 for s in res[0][k]} for k in keys}
    (OUT / "two_stage.json").write_text(json.dumps(dict(per_fold=res, mean=mean), indent=1))
    lines = ["Two-stage control, test year 2023, mean of the two fold stores (holdout land cells; ISD at the "
             "withheld fold; Mesonet never an input).\n", "| Variant | grid RMSE | grid bias | ISD MAE | Meso MAE |",
             "|---|---|---|---|---|"]
    for k in keys:
        v = mean[k]
        lines.append(f"| {k} | {v['rmse']:.3f} | {v['bias']:+.3f} | {v['isd_mae']:.3f} | {v['meso_mae']:.3f} |")
    txt = "\n".join(lines) + "\n"
    (OUT / "two_stage.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
