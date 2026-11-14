"""Year-by-year bias check: is the 2023 negative bias a drift between POWER, AORC and the stations?

Per calendar year, mean over land cells (holdout and outside separately) of POWER - AORC and model - AORC, and over ISD
stations (all 190, own cell) of ISD - AORC, ISD - POWER and ISD - model. If the model's 2023 bias is real, ISD - model
should rise in 2023 too; if ISD - AORC moves instead, AORC drifted.

Usage: python scripts/wind_bias_years.py [--run w10_L_s1337]   -> outputs/wind/bias_years.md
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

OUT = REPO / "outputs" / "wind"


def main(run):
    st = wim.store_for("w10")
    n = st.ds.sizes["time"]
    z = st._zarr
    Ta = np.asarray(z["target"][:, 0], "float32")
    T = Ta + st.clim[st.doy]
    P = np.asarray(z["input"][:, st.stored.index("coarse_tmp")], "float32") + st.clim_coarse[st.doy]
    M = wim.predict(st, run, np.arange(n)) + st.clim[st.doy]
    land = st.land & np.isfinite(Ta).all(0)
    regions = {"holdout": st.m_hold & land, "outside": ~st.m_hold & land}
    meta = pd.DataFrame(json.loads((OUT / "obs_stations.json").read_text())["stations"])
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    O = d.pivot_table(index="date", columns="key", values="ws").reindex(st.times).reindex(columns=meta.key.values)
    O = O.values.astype("float32")                                            # (days, stations)
    r, c = meta.row.values, meta.col.values
    yrs = pd.DatetimeIndex(st.times).year.values
    lines = [f"Year-by-year mean differences (m/s), model {run}.\n",
             "| year | split | POWER-AORC hold | model-AORC hold | POWER-AORC out | model-AORC out | "
             "ISD-AORC | ISD-POWER | ISD-model | n ISD station-days |", "|" + "---|" * 10]
    for y in np.unique(yrs):
        k = yrs == y
        sp = ",".join(sorted(set(np.asarray(st.split)[k])))
        row = [str(y), sp]
        for g in regions.values():
            row += [f"{np.nanmean((P[k] - T[k])[:, g]):+.3f}", f"{np.nanmean((M[k] - T[k])[:, g]):+.3f}"]
        o = O[k]
        for F in (T, P, M):
            row.append(f"{np.nanmean(o - F[k][:, r, c]):+.3f}")
        row.append(str(int(np.isfinite(o).sum())))
        lines.append("| " + " | ".join(row) + " |")
    txt = "\n".join(lines) + "\n"
    (OUT / "bias_years.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="w10_L_s1337")
    main(ap.parse_args().run)
