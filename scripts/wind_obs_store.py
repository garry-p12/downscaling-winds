"""Station-fusion wind stores: the 10 km wind store plus four channels of same-day ISD station information.

WHY (WIND_WORK_LOG.md section 7c)
    AORC's fine-scale daily wind tracks the ISD stations (r 0.88 for the daily departure from POWER) and POWER cannot
    predict it (best r2 0.22 below 100 km). The information is in the observations, so give the model the
    observations.

CHANNELS (all per day, 10 km grid)
    obs_dep25, obs_dep75   kernel-weighted mean of station departures, Gaussian sigma 25 / 75 km
    obs_w25,   obs_w75     kernel weight (a lone station gives 1 at its own cell), i.e. how much data is nearby
    A station's departure on day t is (daily mean ISD speed) - (POWER WS10M interpolated to its cell), minus that
    station's own mean departure and annual harmonic fitted on TRAIN days (2019-2021). Only the daily signal passes;
    the station's siting bias does not.

WHICH STATIONS ARE INPUTS
    Outside the holdout: on each day a random 60% of stations (seeded by day), so the network has to spread station
    information into cells without a station; the same subset in both fold stores.
    Inside the holdout: fold 0 store uses fold-A stations as inputs, fold 1 store uses fold-B. A model is scored at
    the stations of the OTHER fold (never inputs), and on the grid against AORC. Oklahoma Mesonet is never an input.

OUTPUTS
    data_store_sc/wind_obs_f{0,1}/dataset.zarr (+ norm_stats.json), outputs/wind/obs_stations.json (stations, cells,
    folds), outputs/wind/isd_domain_daily.csv (daily means, all stations)

Usage (login node, from $WORK/downscaling):
    python scripts/wind_obs_store.py --stations-only     # station list + downloads + daily means
    python scripts/wind_obs_store.py                     # everything
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
from evaluation.fields import Store                                               # noqa: E402
from evaluation.station_panel import CACHE, ISD_HISTORY                          # noqa: E402
import wind_plan                                                                  # noqa: E402

OUT = REPO / "outputs" / "wind"
SRC = "w10"
SIGMAS_KM = (25, 75)
KEEP_FRAC = 0.6
YEARS = range(2019, 2024)
OBS_CHANNELS = [f"obs_dep{s}" for s in SIGMAS_KM] + [f"obs_w{s}" for s in SIGMAS_KM]


def log(*a):
    print(*a, flush=True)


def station_list(st: Store) -> pd.DataFrame:
    """ISD stations active over 2019-2023 whose nearest 10 km cell is on land in the domain."""
    import urllib.request
    from pyproj import Transformer
    hist = CACHE / "isd-history.csv"
    if not hist.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ISD_HISTORY, hist)
    df = pd.read_csv(hist, dtype={"USAF": str, "WBAN": str}).dropna(subset=["LAT", "LON"])
    df = df[(df.LAT != 0) | (df.LON != 0)]
    df["BEGIN"] = pd.to_datetime(df.BEGIN, format="%Y%m%d", errors="coerce")
    df["END"] = pd.to_datetime(df.END, format="%Y%m%d", errors="coerce")
    df = df[(df.BEGIN <= "2019-01-01") & (df.END >= "2023-12-31") & (df.USAF != "999999")]
    tf = Transformer.from_crs("EPSG:4326", st.cfg["grid"]["target_crs"], always_xy=True)
    sx, sy = tf.transform(df.LON.values, df.LAT.values)
    gx, gy = st.ds["x"].values, st.ds["y"].values
    half = st.res_m / 2
    ok = (sx >= gx.min() - half) & (sx <= gx.max() + half) & (sy >= gy.min() - half) & (sy <= gy.max() + half)
    df, sx, sy = df[ok].copy(), sx[ok], sy[ok]
    df["col"] = [int(np.abs(gx - x).argmin()) for x in sx]
    df["row"] = [int(np.abs(gy - y).argmin()) for y in sy]
    df = df[st.land[df.row.values, df.col.values]]
    i0, i1, j0, j1 = st.box
    df["in_holdout"] = (df.row >= i0) & (df.row < i1) & (df.col >= j0) & (df.col < j1)
    df["key"] = df.USAF + "-" + df.WBAN
    return df.drop_duplicates("key").reset_index(drop=True)


def daily_means(df: pd.DataFrame) -> pd.DataFrame:
    import wind_suite
    f = OUT / "isd_domain_daily.csv"
    have = pd.read_csv(f, parse_dates=["date"]) if f.exists() else pd.DataFrame(columns=["key", "date", "ws"])
    done = set(have.key.unique())
    rows = [have]
    for n, r in enumerate(df.itertuples()):
        if r.key in done:
            continue
        parts = [wind_suite.isd_wind(r.USAF, r.WBAN, y) for y in YEARS]
        parts = [p for p in parts if p is not None and len(p)]
        if parts:
            s = pd.concat(parts)
            rows.append(pd.DataFrame({"key": r.key, "date": s.index, "ws": s.values}))
        if n % 20 == 0:
            log(f"  {n}/{len(df)} stations")
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(f, index=False)
    log(f"daily means: {out.key.nunique()} stations, {len(out)} station-days -> {f.relative_to(REPO)}")
    return out


def departures(st: Store, df: pd.DataFrame, daily: pd.DataFrame):
    """(stations, days) de-biased departure from POWER, NaN where no obs; stations without enough data dropped."""
    times = st.times
    z = st._zarr
    ic = st.stored.index("coarse_tmp")
    rows, cols = df.row.values, df.col.values
    P = np.empty((len(times), len(df)), "float32")
    for b0 in range(0, len(times), 128):
        b1 = min(b0 + 128, len(times))
        P[b0:b1] = np.asarray(z["input"][b0:b1, ic], "float32")[:, rows, cols]
    P += st.clim_coarse[st.doy][:, rows, cols]
    O = daily.pivot_table(index="date", columns="key", values="ws").reindex(times)
    O = O.reindex(columns=df.key.values).values.astype("float32")
    D = (O - P).T                                                        # (stations, days)
    tr = st.split == "train"
    ang = 2 * np.pi * st.doy / 365.25
    H = np.c_[np.ones(len(times)), np.sin(ang), np.cos(ang)]
    keep = np.zeros(len(df), bool)
    for s in range(len(df)):
        ok = np.isfinite(D[s]) & tr
        if ok.sum() < 300:
            continue
        coef = np.linalg.lstsq(H[ok], D[s, ok], rcond=None)[0]
        D[s] -= H @ coef
        keep[s] = True
    return D[keep], df[keep].reset_index(drop=True)


def channels(st: Store, df: pd.DataFrame, D: np.ndarray, fold_input: np.ndarray) -> np.ndarray:
    """(days, 4, H, W). fold_input: bool per station, which holdout stations are inputs in this store."""
    from scipy.ndimage import gaussian_filter
    n = D.shape[1]
    out = np.zeros((n, len(OBS_CHANNELS), *st.shape), "float32")
    hold = df.in_holdout.values
    for k in range(n):
        rng = np.random.default_rng(10_000 + k)
        use = np.isfinite(D[:, k]) & np.where(hold, fold_input, rng.random(len(df)) < KEEP_FRAC)
        Dp = np.zeros(st.shape, "float64"); Wp = np.zeros(st.shape, "float64")
        np.add.at(Dp, (df.row.values[use], df.col.values[use]), D[use, k])
        np.add.at(Wp, (df.row.values[use], df.col.values[use]), 1.0)
        for c, s_km in enumerate(SIGMAS_KM):
            sg = s_km / st.dx_km
            ws = gaussian_filter(Wp, sg, mode="constant")
            ds_ = gaussian_filter(Dp, sg, mode="constant")
            out[k, c] = np.where(ws > 1e-3 / (2 * np.pi * sg * sg), ds_ / np.maximum(ws, 1e-12), 0.0)
            out[k, len(SIGMAS_KM) + c] = ws * 2 * np.pi * sg * sg
    return out


def write_store(st: Store, obs: np.ndarray, dst: Path, src_zarr: str):
    import xarray as xr
    from common.normalize import Normalizer
    if dst.parent.exists():
        shutil.rmtree(dst.parent)
    dst.parent.mkdir(parents=True)
    ds = xr.open_zarr(src_zarr, consolidated=True)
    o = xr.DataArray(obs, dims=("time", "channel_in", "y", "x"),
                     coords={"time": ds.time, "channel_in": OBS_CHANNELS, "y": ds.y, "x": ds.x})
    inp = xr.concat([ds["input"], o.chunk({"time": 1})], dim="channel_in").chunk({"time": 1, "channel_in": -1})
    inp = inp.assign_coords(channel_in=np.array(list(ds.channel_in.values) + OBS_CHANNELS, dtype="<U16"))
    new = ds.drop_vars(["input", "channel_in"]).assign(input=inp)
    for v in new.variables:
        new[v].encoding = {}
    new.attrs = dict(ds.attrs, obs_channels=",".join(OBS_CHANNELS))
    new.to_zarr(str(dst), mode="w", consolidated=True)
    nz = json.loads((Path(src_zarr).parent / "norm_stats.json").read_text())
    tr = st.split == "train"
    for c, nm in enumerate(OBS_CHANNELS):
        v = obs[tr, c][:, st.land]
        nz[f"in::{nm}"] = {"mean": float(v.mean()), "std": float(v.std() + 1e-6), "log1p": False}
    (dst.parent / "norm_stats.json").write_text(json.dumps(nz, indent=2))
    Normalizer.load(str(dst.parent / "norm_stats.json"))
    log(f"wrote {dst} ({len(new.channel_in)} channels)")


def main(stations_only: bool):
    cfg, zarr = wind_plan.STORES[SRC][:2]
    st = Store(zarr, cfg_name=cfg)
    df = station_list(st)
    log(f"ISD stations active 2019-2023 on land in the domain: {len(df)} ({int(df.in_holdout.sum())} in the holdout)")
    daily = daily_means(df)
    D, df = departures(st, df, daily)
    log(f"with >= 300 train days: {len(df)} stations ({int(df.in_holdout.sum())} in the holdout)")
    rng = np.random.default_rng(0)
    fold = np.full(len(df), -1)
    h = np.where(df.in_holdout.values)[0]
    fold[h] = rng.permutation(np.arange(len(h)) % 2)
    df["fold"] = fold
    meta = [dict(key=r["key"], name=str(r["STATION NAME"]), lat=float(r["LAT"]), lon=float(r["LON"]),
                 row=int(r["row"]), col=int(r["col"]), in_holdout=bool(r["in_holdout"]), fold=int(r["fold"]),
                 n_days=int(np.isfinite(D[i]).sum())) for i, (_, r) in enumerate(df.iterrows())]
    (OUT / "obs_stations.json").write_text(json.dumps({"keep_frac": KEEP_FRAC, "sigmas_km": SIGMAS_KM,
                                                       "stations": meta}, indent=1))
    np.save(OUT / "obs_departures.npy", D)
    log(f"folds in the holdout: A {int((fold == 0).sum())}, B {int((fold == 1).sum())}")
    if stations_only:
        return
    for f in (0, 1):
        obs = channels(st, df, D, fold_input=(df.fold.values == f))
        tag = f"w10o{f}"
        write_store(st, obs, REPO / wind_plan.STORES[tag][1], zarr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations-only", action="store_true")
    main(ap.parse_args().stations_only)
