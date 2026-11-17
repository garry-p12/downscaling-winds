"""Export the fields behind the product figures of Paper 1 (the final wind pipeline), so they can be drawn off-cluster.

Fold store 0 (fold-1 ISD stations withheld), test year 2023, same stages as scripts/wind_results_figs.py.
Writes outputs/wind/paper_fields2.npz (float16 fields, roughly 15-25 MB):

  grid      lat, lon, land, hold (held-out land cells), box (i0, i1, j0, j1 of the held-out box)
  days      for d in DAYS and stage s in S0..S5:  d<d>_<s>  (absolute m/s); d<d>_T (AORC);
            d<d>_mem (3 calibrated members), d<d>_q10, d<d>_q90 (from all 25 members)
            d<d>_stn_kind (0 ISD, 1 Mesonet), d<d>_stn_i, d<d>_stn_j, d<d>_stn_obs   (withheld stations)
  year      y_rmse_<s>, y_bias_<s> per cell for every stage; y_T (AORC mean); y_E (product mean);
            y_wid (mean 80% width), y_cov (fraction of days AORC inside the 80% range), y_crps (mean CRPS)
  series    s_T, s_P, s_E (daily holdout means), s_mem (25 x 365 member holdout means)
  spectra   sp_lam (wavelength, km) and sp_<name> (power) for AORC, no-station, fusion, ens. mean, raw and calibrated members
  1 km      l1_T, l1_M, l1_P, l1_lat, l1_lon, l1_land, l1_hold on DAYS[0] (run w1_L_s1337), if that run exists

Stages: S0 POWER, S1 no-station model, S2 fusion 2019-21, S3 fusion 2019-22, S4 product (ens. mean), S5 MSE corrector.
Usage (repo root, Vista login node, about 10 min):  python scripts/wind_paper_export2.py
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
import wind_results_figs as wrf            # noqa: E402
import wind_inference_maps as wim          # noqa: E402
import wind_diffusion_maps as wdm          # noqa: E402

DAYS = ["2023-02-10", "2023-04-22", "2023-11-16"]
NAMES = wrf.STAGES + ["MSE corrector"]
h = lambda a: np.asarray(a, "float32").astype("float16")     # noqa: E731


def main():
    OUT = wrf.OUT
    meta = json.loads((OUT / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    F, X, E = wrf.load_fold(0, meta, isd)
    st, T, cl = F.st, F.Tabs, F.clim
    nm = E.shape[0]
    out = {"lat": h(st.LAT), "lon": h(st.LON), "land": np.asarray(F.land, bool), "hold": np.asarray(F.mask, bool),
           "box": np.asarray(st.box, "int32"), "names": np.array(NAMES), "days": np.array(DAYS)}
    times = pd.DatetimeIndex(st.times[F.te])
    # days
    for day in DAYS:
        k = int(np.where(times == pd.Timestamp(day))[0][0]); tg = day.replace("-", "")
        for n, s in enumerate(NAMES):
            out[f"d{tg}_S{n}"] = h(X[s][k])
        out[f"d{tg}_T"] = h(T[k])
        mem = np.asarray(E[:, k], "float32") + cl[k][None]
        out[f"d{tg}_mem"] = h(mem[:3])
        q10, q90 = np.percentile(mem, [10, 90], axis=0)
        out[f"d{tg}_q10"] = h(q10); out[f"d{tg}_q90"] = h(q90)
        rows = [(kd, i, j, float(v)) for kd, i, j, v in wdm.withheld_stations(F, k) if np.isfinite(v)]
        out[f"d{tg}_stn_kind"] = np.array([0 if r[0] == "isd" else 1 for r in rows], "int8")
        out[f"d{tg}_stn_i"] = np.array([r[1] for r in rows], "int16")
        out[f"d{tg}_stn_j"] = np.array([r[2] for r in rows], "int16")
        out[f"d{tg}_stn_obs"] = np.array([r[3] for r in rows], "float32")
        print(day, "done", flush=True)
    # test year, per cell
    with np.errstate(invalid="ignore"):
        for n, s in enumerate(NAMES):
            D = X[s] - T
            out[f"y_rmse_S{n}"] = h(np.sqrt(np.nanmean(D ** 2, 0))); out[f"y_bias_S{n}"] = h(np.nanmean(D, 0))
        out["y_T"] = h(np.nanmean(T, 0)); out["y_E"] = h(np.nanmean(X["Product (ens. mean)"], 0))
        wid = np.zeros(T.shape[1:], "float64"); cov = np.zeros_like(wid); crps = np.zeros_like(wid); cnt = np.zeros_like(wid)
        w = 2 * np.arange(1, nm + 1) - nm - 1
        w = w.reshape(-1, 1, 1)
        for k in range(T.shape[0]):
            Xk = np.sort(np.asarray(E[:, k], "float32") + cl[k][None], 0)
            q10, q90 = np.percentile(Xk, [10, 90], axis=0)
            ok = np.isfinite(T[k])
            c = np.mean(np.abs(Xk - T[k][None]), 0) - (w * Xk).sum(0) / (nm * (nm - 1))
            wid[ok] += (q90 - q10)[ok]; cov[ok] += ((T[k] >= q10) & (T[k] <= q90))[ok]; crps[ok] += c[ok]; cnt[ok] += 1
        cnt[cnt == 0] = np.nan
        out["y_wid"] = h(wid / cnt); out["y_cov"] = h(cov / cnt); out["y_crps"] = h(crps / cnt)
        # daily holdout means
        m = F.mask
        hm = lambda A: np.nanmean(A[:, m], 1).astype("float32")         # noqa: E731
        out["s_T"] = hm(T); out["s_P"] = hm(X["POWER"]); out["s_E"] = hm(X["Product (ens. mean)"])
        out["s_mem"] = np.stack([hm(np.asarray(E[i]) + cl) for i in range(nm)])
        out["s_times"] = np.array([str(t.date()) for t in times])
    print("test year done", flush=True)
    # spectra over the box (members: 5 of 25)
    box = st.box
    Eraw = np.load(wrf.DIF / f"obsval_w10o{F.f}_test.npy", mmap_mode="r")
    an = lambda A: A - cl                                                # noqa: E731
    spec = {"AORC": F.Ta, "nostation": an(X["No-station model"]), "fusion": an(X["Fusion 2019-21"]),
            "ensmean": an(X["Product (ens. mean)"]), "mem_raw": np.asarray(Eraw[:5]), "mem_cal": np.asarray(E[:5])}
    for kk, A in spec.items():
        lam, pw = wrf.spectrum(A, box)
        out["sp_lam"] = lam.astype("float32"); out[f"sp_{kk}"] = pw.astype("float32")
    print("spectra done", flush=True)
    # 1 km rung on the windy day
    try:
        s1 = wim.store_for("w1"); k1 = wim.day_index(s1, DAYS[0])
        out["l1_T"] = h(wim.truth_abs(s1, k1)); out["l1_P"] = h(wim.input_abs(s1, k1, "coarse_tmp"))
        out["l1_M"] = h(wim.predict(s1, "w1_L_s1337", np.array([k1]))[0] + s1.clim[s1.doy[k1]])
        out["l1_lat"] = h(s1.LAT); out["l1_lon"] = h(s1.LON)
        out["l1_land"] = np.asarray(s1.land, bool); out["l1_hold"] = np.asarray(s1.m_hold, bool)
        print("1 km done", flush=True)
    except Exception as e:                                              # noqa: BLE001
        print("1 km skipped:", repr(e), flush=True)
    p = OUT / "paper_fields2.npz"
    np.savez_compressed(p, **out)
    print("wrote", p, round(p.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
