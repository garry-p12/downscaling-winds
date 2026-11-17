"""Export the fields behind the wind map figures of Paper 1 into one small file, so the figures can be drawn off-cluster.

Writes outputs/wind/paper_fields.npz (float16 fields, about 5 MB):
  ladder, for g in w25, w10, w5 (run <g>_L_s1337) on DAY:
      {g}_P, {g}_T, {g}_M        interpolated POWER, AORC, model (absolute m/s)
      {g}_lat, {g}_lon, {g}_land, {g}_hold
  stations on DAY (10 km grid, holdout panel and Mesonet):
      stn_kind (0 ISD, 1 Mesonet), stn_lat, stn_lon, stn_obs, stn_P, stn_M (POWER and model at the station cell)
  test year 2023, w10_L_s1337:
      ty_Tm, ty_Pm, ty_Mm        2023 means;  ty_rP, ty_rM   per-cell RMSE against AORC
  diffusion, fold store w10o0 on DAY (tag obsvalcal3, control obsvaldet):
      df_T, df_B (station-fusion network), df_C (MSE corrector), df_E (ensemble mean), df_M3 (3 members), df_S (spread)
      df_lat, df_lon, df_land, df_hold, dstn_kind, dstn_lat, dstn_lon, dstn_obs (withheld ISD and Mesonet)

Usage (repo root on Vista, login node):  python scripts/wind_paper_export.py [--day 2023-02-10]
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
import wind_inference_maps as wim          # noqa: E402
import wind_obs_eval as woe                # noqa: E402

h = lambda a: np.asarray(a, "float32").astype("float16")     # noqa: E731


def grid(out, key, st):
    out[f"{key}_lat"] = h(st.LAT); out[f"{key}_lon"] = h(st.LON)
    out[f"{key}_land"] = np.asarray(st.land, bool); out[f"{key}_hold"] = np.asarray(st.m_hold, bool)


def main(a):
    day = a.day; out = {"day": np.array(day)}
    # ladder and the windy day
    for g in ("w25", "w10", "w5"):
        run = f"{g}_L_s1337"; st = wim.store_for(g); k = wim.day_index(st, day)
        out[f"{g}_T"] = h(wim.truth_abs(st, k))
        out[f"{g}_P"] = h(wim.input_abs(st, k, "coarse_tmp"))
        out[f"{g}_M"] = h(wim.predict(st, run, np.array([k]))[0] + st.clim[st.doy[k]])
        grid(out, g, st)
        print(g, "done", flush=True)
    st = wim.store_for("w10"); k = wim.day_index(st, day)
    P = out["w10_P"].astype("float32"); M = out["w10_M"].astype("float32")
    rows = wim.station_values(st, "w10", day)
    out["stn_kind"] = np.array([0 if kd == "isd" else 1 for kd, *_ in rows])
    out["stn_lat"] = np.array([st.LAT[i, j] for _, i, j, _ in rows], "float32")
    out["stn_lon"] = np.array([st.LON[i, j] for _, i, j, _ in rows], "float32")
    out["stn_obs"] = np.array([v for *_, v in rows], "float32")
    out["stn_P"] = np.array([P[i, j] for _, i, j, _ in rows], "float32")
    out["stn_M"] = np.array([M[i, j] for _, i, j, _ in rows], "float32")
    out["stn_hold"] = np.array([bool(st.m_hold[i, j]) for _, i, j, _ in rows])
    # test year
    te = st.i_test
    T = np.asarray(st._zarr["target"][te[0]:te[-1] + 1, 0], "float32") + st.clim[st.doy[te]]
    Pt = np.asarray(st._zarr["input"][te[0]:te[-1] + 1, st.stored.index("coarse_tmp")], "float32") + st.clim_coarse[st.doy[te]]
    Mt = wim.predict(st, "w10_L_s1337", te) + st.clim[st.doy[te]]
    with np.errstate(invalid="ignore"):
        out["ty_Tm"] = h(np.nanmean(T, 0)); out["ty_Pm"] = h(np.nanmean(Pt, 0)); out["ty_Mm"] = h(np.nanmean(Mt, 0))
        out["ty_rP"] = h(np.sqrt(np.nanmean((Pt - T) ** 2, 0))); out["ty_rM"] = h(np.sqrt(np.nanmean((Mt - T) ** 2, 0)))
    del T, Pt, Mt
    print("test year done", flush=True)
    # diffusion
    meta = json.loads((REPO / "outputs" / "wind" / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(REPO / "outputs" / "wind" / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {kk: g_.set_index("date")["ws"] for kk, g_ in d.groupby("key")}
    F = woe.Fold(0, meta, isd); s0 = F.st
    DIF = REPO / "outputs" / "wind" / "diffusion"
    E = np.load(DIF / f"{a.tag}_w10o0_test.npy", mmap_mode="r"); det = np.load(DIF / f"{a.tag}_w10o0_det.npy", mmap_mode="r")
    ctl = np.load(DIF / f"{a.control}_w10o0_test.npy", mmap_mode="r")[0]
    kk = int(np.where(s0.times[s0.i_test] == pd.Timestamp(day))[0][0]); cl = F.clim[kk]
    Mem = np.asarray(E[:, kk], "float32") + cl
    out["df_T"] = h(F.Tabs[kk]); out["df_B"] = h(det[kk] + cl); out["df_C"] = h(ctl[kk] + cl)
    out["df_E"] = h(Mem.mean(0)); out["df_M3"] = h(Mem[:3]); out["df_S"] = h(Mem.std(0, ddof=1))
    grid(out, "df", s0)
    st_ = [(0, i, j, o[kk]) for _, i, j, o in F.eval_isd] + [(1, i, j, o[kk]) for _, i, j, o in F.eval_meso]
    st_ = [r for r in st_ if np.isfinite(r[3])]
    out["dstn_kind"] = np.array([r[0] for r in st_]); out["dstn_obs"] = np.array([r[3] for r in st_], "float32")
    out["dstn_lat"] = np.array([s0.LAT[r[1], r[2]] for r in st_], "float32")
    out["dstn_lon"] = np.array([s0.LON[r[1], r[2]] for r in st_], "float32")
    p = REPO / "outputs" / "wind" / "paper_fields.npz"
    np.savez_compressed(p, **out)
    print("wrote", p, round(p.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="2023-02-10")
    ap.add_argument("--tag", default="obsvalcal3")
    ap.add_argument("--control", default="obsvaldet")
    main(ap.parse_args())
