"""Inference maps for the wind program: what each model was given, what it produced, and where it went wrong.

Sheets written to outputs/wind/maps/ (all speeds in m/s, absolute: the day-of-year climatology is added back):

  inputs_<store>_<day>.png      every channel the run sees on that day, in physical units (dynamic POWER fields
                                with their climatology restored), plus POWER 10 m direction as arrows over speed
  inference_<run>_<day>.png     POWER interpolated / AORC truth / model / correction made (model - POWER);
                                POWER error / model error (one shared diverging scale) / correction needed vs made /
                                holdout RMSE bars. ISD (circles) and Mesonet (squares) daily means drawn on the
                                speed panels with the same colour scale.
  testyear_<run>.png            2023 mean speed (AORC, POWER, model), bias (POWER, model), per-cell RMSE (POWER,
                                model) and the RMSE change model - POWER
  ladder_<day>.png              the same day on the 25, 10 and 5 km rungs (and 1 km when trained): POWER, AORC,
                                model, model error
  loro_<day>.png                the three leave-one-region-out holdouts at 10 km: AORC, model, model error

Predictions come from .field_cache when the eval suite has already produced them (same key as
evaluation.fields.predict_all). Otherwise ONLY the needed days are run here, on CPU, and nothing is written to the
cache: the cache key has no device in it, and a CPU-filled entry would silently become the field a later GPU eval
scores.

Usage (on Vista, from $WORK/downscaling; a login node is enough for 25/10/5 km):
    DOWNSCALE_MAX_WORKERS=4 python scripts/wind_inference_maps.py
    python scripts/wind_inference_maps.py --runs w10_A0_s1337 w10_A8_s1337 --days 2023-03-15
    python scripts/wind_inference_maps.py --only testyear --runs w10_unet_A8_s1337
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
from evaluation.fields import CACHE, Store                                        # noqa: E402
from evaluation.mapping import (GRAT, INK, INK2, INK3, ORANGE, SURF,            # noqa: E402
                                graticule, header, holdout_box, palettes)
import wind_plan                                                                  # noqa: E402

OUT = REPO / "outputs" / "wind" / "maps"
PLAN = {r["name"]: r for r in wind_plan.plan()}
DEFAULT_RUNS = ["w25_L_s1337", "w10_L_s1337", "w5_L_s1337"]
LORO_RUNS = ["w10west_L_s1337", "w10coast_L_s1337", "w10north_L_s1337"]

# display name, units, colormap kind, absolute (add climatology) for every stored channel
CH = {
    "coarse_tmp": ("POWER 10 m speed (interp.)", "m/s", "seq"),
    "coarse_u10m": ("POWER 10 m u (eastward)", "m/s", "div"),
    "coarse_v10m": ("POWER 10 m v (northward)", "m/s", "div"),
    "coarse_ws50m": ("POWER 50 m speed", "m/s", "seq"),
    "coarse_wind": ("POWER 2 m speed", "m/s", "seq"),
    "coarse_ws10m_max": ("POWER daily max 10 m speed", "m/s", "seq"),
    "coarse_t2m_range": ("POWER diurnal T range", "degC", "seq"),
    "dem": ("Elevation", "m", "terrain"),
    "landcover": ("Land cover class (NLCD)", "class", "cat"),
    "coastal_dist": ("Distance to coast", "km", "seq"),
    "urban_frac": ("Urban fraction", "", "frac"),
    "land_mask": ("Land mask", "", "frac"),
    "elev_std": ("Sub-grid relief (elev. SD)", "m", "seq"),
    "slope_mean": ("Mean slope", "deg", "seq"),
    "tpi": ("Topographic position index", "m", "div"),
    "northness": ("Northness", "", "div"),
    "eastness": ("Eastness", "", "div"),
}
for _lc in ("water", "developed", "barren", "forest", "shrub", "herbaceous", "cultivated", "wetland"):
    CH[f"lc_{_lc}"] = (f"{_lc.capitalize()} fraction", "", "frac")
CH.update({"obs_dep25": ("Station departure, 25 km kernel", "m/s", "div"),
           "obs_dep75": ("Station departure, 75 km kernel", "m/s", "div"),
           "obs_w25": ("Station weight, 25 km kernel", "", "seq"),
           "obs_w75": ("Station weight, 75 km kernel", "", "seq")})
SKIP_INPUT = {"doy_sin", "doy_cos", "land_mask"}          # constant in space on any one day / the mask itself


def log(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------------------- predictions
_STORES: dict[str, Store] = {}


def target_name(tag: str) -> str:
    """Name of the store's training target: w10e stores hold ERA5-Land, every other store AORC."""
    return "ERA5-Land" if tag.startswith("w10e") else "AORC"


def store_for(tag: str) -> Store:
    if tag not in _STORES:
        cfg, zarr = wind_plan.STORES[tag][:2]
        _STORES[tag] = Store(zarr, cfg_name=cfg)
    return _STORES[tag]


def cache_path(st: Store, run: str) -> Path:
    """The exact key evaluation.fields.predict_all uses (checkpoint mtime, store path, days, grid shape)."""
    path = REPO / "checkpoints" / run / "last.pt"
    key = hashlib.sha1(f"{path}:{path.stat().st_mtime_ns}:{st.path}:{st.ds.sizes['time']}:"
                       f"{st.shape[0]}x{st.shape[1]}".encode()).hexdigest()[:16]
    return CACHE / f"{run}_last_{key}.npy"


_MODELS: dict = {}


def predict(st: Store, run: str, days: np.ndarray) -> np.ndarray:
    """(len(days), H, W) prediction, ANOMALY m/s. Cache read if present; otherwise DOWNSCALE_DEVICE (default cpu), nothing written."""
    cp = cache_path(st, run)
    if cp.exists():
        return np.asarray(np.load(cp, mmap_mode="r")[days])
    import torch
    from common import load_config
    from data.build_dataset import NO_NORMALIZE
    from models.model import load_checkpoint
    torch.set_num_threads(int(os.environ.get("DOWNSCALE_MAX_WORKERS", "4")))
    device = os.environ.get("DOWNSCALE_DEVICE", "cpu")
    if run not in _MODELS:
        _MODELS[run] = load_checkpoint(REPO / "checkpoints" / run / "last.pt", device, load_config("model"))
    model, mcfg = _MODELS[run]
    names = list(mcfg.get("use_channels") or st.stored)
    idx = [st.stored.index(n) for n in names]
    out = np.empty((len(days), *st.shape), "float32")
    log(f"    {run}: no cached field, predicting {len(days)} day(s) on {device}")
    for n, k in enumerate(days):
        x = np.asarray(st._zarr["input"][int(k)])[idx].astype("float32")
        for c, nm in enumerate(names):
            if nm not in NO_NORMALIZE:
                x[c] = st.nz.transform(f"in::{nm}", x[c])
        np.nan_to_num(x, copy=False)
        with torch.no_grad():
            y = model(torch.from_numpy(x).unsqueeze(0).to(device), {"temp"})["temp"][0, 0].cpu().numpy()
        out[n] = st.nz.inverse("out::tmp", y)
    return out


def run_channels(run: str) -> list[str]:
    return list(PLAN[run]["channels"]) if run in PLAN else []


# ----------------------------------------------------------------------------------------------- field helpers
def day_index(st: Store, day: str) -> int:
    k = int(np.argmin(np.abs(st.times - pd.Timestamp(day))))
    if st.times[k] != pd.Timestamp(day):
        raise SystemExit(f"{day} is not in the store")
    return k


def truth_abs(st: Store, k: int) -> np.ndarray:
    t = np.asarray(st._zarr["target"][k, 0], "float32")
    return t + st.clim[st.doy[k]]                                   # NaN where AORC has no data stays NaN


def input_abs(st: Store, k: int, name: str) -> np.ndarray:
    v = np.asarray(st._zarr["input"][k, st.stored.index(name)], "float32")
    cl = "clim_coarse_tmp" if name == "coarse_tmp" else f"clim_{name}"
    if cl in st.ds:
        v = v + np.asarray(st.ds[cl].isel(doy=int(st.doy[k])).values, "float32")
    return v


def pick_days(st: Store) -> dict[str, str]:
    """Windy, typical and calm test days by mean AORC speed over the holdout (10 km store)."""
    te = st.i_test
    T = np.asarray(st._zarr["target"][te[0]:te[-1] + 1, 0], "float32") + st.clim[st.doy[te]]
    m = np.nanmean(T[:, st.m_hold], axis=1)
    order = np.argsort(m)
    pick = {"windy": order[int(.98 * (len(m) - 1))], "typical": order[len(m) // 2], "calm": order[int(.02 * (len(m) - 1))]}
    out = {lab: st.times[te[i]].strftime("%Y-%m-%d") for lab, i in pick.items()}
    log("days: " + ", ".join(f"{lab} {d} ({m[pick[lab]]:.2f} m/s holdout mean)" for lab, d in out.items()))
    return out


# --------------------------------------------------------------------------------------------------- stations
_STN: dict = {}


def stations(st: Store, tag: str):
    """[(kind, row, col, obs series)] for ISD (holdout panel) and Mesonet, or [] if anything is missing."""
    if tag in _STN:
        return _STN[tag]
    rows = []
    try:
        from evaluation.station_panel import load_panel
        import wind_suite
        for s in load_panel(st):
            ser = None
            for y in (2022, 2023):
                p = wind_suite.isd_wind(s["usaf"], s["wban"], y)
                if p is not None and len(p):
                    ser = p if ser is None else pd.concat([ser, p])
            if ser is not None:
                rows.append(("isd", s["row"], s["col"], ser))
        for m in wind_suite.mesonet_cells(st) or []:
            rows.append(("meso", m["row"], m["col"], m["obs"]))
    except BaseException as e:                       # noqa: BLE001 - stations are decoration, never fatal
        log(f"    stations for {tag} unavailable: {type(e).__name__}: {e}")
    _STN[tag] = rows
    return rows


def station_values(st: Store, tag: str, day: str):
    t = pd.Timestamp(day); out = []
    for kind, i, j, ser in stations(st, tag):
        v = ser.get(t, np.nan)
        if np.isfinite(v):
            out.append((kind, i, j, float(v)))
    return out


# ------------------------------------------------------------------------------------------------------ drawing
def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                         "text.color": INK, "font.size": 10})
    return plt


def cmaps():
    import matplotlib
    seq, div = palettes()
    spd = matplotlib.colormaps["viridis"].copy(); spd.set_bad("#c9c6bf")
    ter = matplotlib.colormaps["terrain"].copy(); ter.set_bad("#c9c6bf")
    frac = matplotlib.colormaps["YlGn"].copy(); frac.set_bad("#c9c6bf")
    cat = matplotlib.colormaps["tab20"].copy(); cat.set_bad("#c9c6bf")
    return dict(seq=seq, div=div, speed=spd, terrain=ter, frac=frac, cat=cat)


def panel(fig, ax, st, field, title, cmap, sub="", unit="m/s", box=True, land=True, stn=None, norm=None, **kw):
    f = np.ma.masked_invalid(field)
    if land:
        f = np.ma.masked_where(~st.land, f)
    im = ax.imshow(f, cmap=cmap, norm=norm, interpolation="nearest", **kw)
    graticule(ax, st.LAT, st.LON, fontsize=6.5)
    if box:
        holdout_box(ax, st.box, "")
    if stn:
        for kind, i, j, v in stn:
            ax.scatter(j, i, c=[v], cmap=cmap, norm=im.norm, s=26 if kind == "isd" else 20,
                       marker="o" if kind == "isd" else "s", edgecolors=INK, linewidths=.7, zorder=8)
    ax.set_title(title, fontsize=10.5, fontweight="bold", pad=17 if sub else 8)
    if sub:
        ax.text(.5, 1.012, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=7, color=INK3)
    cb = fig.colorbar(im, ax=ax, fraction=.042, pad=.02)
    cb.ax.tick_params(labelsize=7, color=GRAT, labelcolor=INK3)
    if unit:
        cb.set_label(unit, fontsize=7.5, color=INK3)
    return im


def rmse(a, b, m):
    d = (a - b)[m]; d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else np.nan


def bias(a, b, m):
    d = (a - b)[m]; d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else np.nan


def score_mask(st, T):
    return st.m_hold & np.isfinite(T)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    log(f"  wrote {p.relative_to(REPO)}")


# ------------------------------------------------------------------------------------------------------- sheets
def sheet_inputs(run: str, day: str, label: str):
    plt = setup(); C = cmaps()
    r = PLAN[run]; st = store_for(r["store"]); k = day_index(st, day)
    names = [n for n in run_channels(run) if n not in SKIP_INPUT and n in st.stored]
    has_dir = "coarse_u10m" in names and "coarse_v10m" in names
    n = len(names) + int(has_dir)
    ncol = 4 if n <= 12 else 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.1 * nrow + 1.3))
    axes = np.atleast_2d(axes).ravel()
    spd_fields = {nm: input_abs(st, k, nm) for nm in names if CH.get(nm, ("", "", ""))[2] == "seq" and nm.startswith("coarse_")
                  and nm != "coarse_t2m_range"}
    vmax_spd = float(np.nanpercentile(np.concatenate([v[st.land] for v in spd_fields.values()]), 99.5)) if spd_fields else None
    a = 0
    if has_dir:
        ax = axes[a]; a += 1
        s = input_abs(st, k, "coarse_tmp"); u = input_abs(st, k, "coarse_u10m"); v = input_abs(st, k, "coarse_v10m")
        panel(fig, ax, st, s, "POWER 10 m wind (speed + direction)", C["speed"], "arrows: daily-mean u, v (grid-relative)",
              vmin=0, vmax=vmax_spd)
        step = max(1, st.shape[1] // 14)
        yy, xx = np.mgrid[step // 2:st.shape[0]:step, step // 2:st.shape[1]:step]
        flip = -1.0 if st.LAT[0, 0] < st.LAT[-1, 0] else 1.0           # row 0 south -> image is upside down
        ax.quiver(xx, yy, u[yy, xx], flip * v[yy, xx], color="white", edgecolor=INK, linewidth=.4, scale=None,
                  width=.006, zorder=7)
    for nm in names:
        ax = axes[a]; a += 1
        title, unit, kind = CH.get(nm, (nm, "", "seq"))
        f = input_abs(st, k, nm)
        mk = st.land
        if kind == "seq" and nm in spd_fields:
            panel(fig, ax, st, f, title, C["speed"], "", unit, vmin=0, vmax=vmax_spd)
        elif kind == "div":
            lim = float(np.nanpercentile(np.abs(f[mk]), 99)) or 1.0
            from matplotlib.colors import TwoSlopeNorm
            panel(fig, ax, st, f, title, C["div"], "", unit, norm=TwoSlopeNorm(0, -lim, lim))
        elif kind == "terrain":
            panel(fig, ax, st, f, title, C["terrain"], "", unit, vmin=np.nanmin(f[mk]), vmax=np.nanpercentile(f[mk], 99.5))
        elif kind == "frac":
            panel(fig, ax, st, f, title, C["frac"], "", unit or "fraction", vmin=0, vmax=1)
        elif kind == "cat":
            panel(fig, ax, st, f, title, C["cat"], "", unit, vmin=0, vmax=100)
        else:
            panel(fig, ax, st, f, title, C["seq"], "", unit, vmin=np.nanpercentile(f[mk], 1), vmax=np.nanpercentile(f[mk], 99))
    for ax in axes[a:]:
        ax.axis("off")
    top = header(fig, f"Model inputs, {r['store']} ({wind_plan.STORES[r['store']][3]} km), channel set {r['chan']}, {day} ({label})",
                 f"Every channel run {run} sees on this day except day-of-year sine/cosine (one value over the whole "
                 f"domain) and the land mask. Same-day POWER fields are shown in physical units (the training-year "
                 f"day-of-year climatology the store removes has been added back); all POWER speed panels share one "
                 f"colour scale. Orange box: held-out region. Dotted lines: true latitude and longitude (Albers grid).",
                 chars=int(44 * ncol))
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, f"inputs_{r['store']}_{r['chan']}_{day}.png")


def sheet_inference(run: str, day: str, label: str):
    plt = setup(); C = cmaps()
    from matplotlib.colors import TwoSlopeNorm
    r = PLAN[run]; st = store_for(r["store"]); k = day_index(st, day)
    TN = target_name(r["store"])
    T = truth_abs(st, k)
    P = input_abs(st, k, "coarse_tmp")
    M = predict(st, run, np.array([k]))[0] + st.clim[st.doy[k]]
    m = score_mask(st, T)
    stn = station_values(st, r["store"], day)
    land = st.land & np.isfinite(T)
    vmin = 0.0; vmax = float(np.nanpercentile(np.r_[T[land], P[land], M[land]], 99.5))
    e_p, e_m = P - T, M - T
    elim = float(np.nanpercentile(np.abs(np.r_[e_p[m], e_m[m]]), 99)) or 1.0
    rp, rm = rmse(P, T, m), rmse(M, T, m)

    def stn_rmse(F):
        d = [F[i, j] - v for kind, i, j, v in stn if kind == "isd" and st.m_hold[i, j]]
        return (float(np.sqrt(np.mean(np.square(d)))), len(d)) if d else (np.nan, 0)

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.8))
    panel(fig, ax[0, 0], st, P, "POWER, interpolated (input)", C["speed"],
          f"holdout RMSE {rp:.2f}, bias {bias(P, T, m):+.2f} m/s", stn=stn, vmin=vmin, vmax=vmax)
    panel(fig, ax[0, 1], st, T, f"{TN} (target)", C["speed"], f"holdout mean {np.nanmean(T[m]):.2f} m/s",
          stn=stn, vmin=vmin, vmax=vmax)
    panel(fig, ax[0, 2], st, M, "Model output (downscaled)", C["speed"],
          f"holdout RMSE {rm:.2f}, bias {bias(M, T, m):+.2f} m/s", stn=stn, vmin=vmin, vmax=vmax)
    cm = M - P; clim_ = float(np.nanpercentile(np.abs(cm[st.land]), 99)) or 1.0
    panel(fig, ax[0, 3], st, cm, "Correction made (model - POWER)", C["div"], "what the network added to its input",
          norm=TwoSlopeNorm(0, -clim_, clim_))
    panel(fig, ax[1, 0], st, e_p, f"POWER error (POWER - {TN})", C["div"], "same scale as the panel to the right",
          norm=TwoSlopeNorm(0, -elim, elim))
    panel(fig, ax[1, 1], st, e_m, f"Model error (model - {TN})", C["div"], "shrinkage against the left panel is skill",
          norm=TwoSlopeNorm(0, -elim, elim))

    sc = ax[1, 2]
    need, made = (T - P)[m], (M - P)[m]
    ok = np.isfinite(need) & np.isfinite(made); need, made = need[ok], made[ok]
    lim = float(np.nanpercentile(np.abs(np.r_[need, made]), 99.5)) or 1.0
    rc = float(np.corrcoef(need, made)[0, 1]) if need.size > 2 else np.nan
    slope = float(np.polyfit(need, made, 1)[0]) if need.size > 2 else np.nan
    sc.hexbin(need, made, gridsize=40, cmap="Blues", mincnt=1, extent=(-lim, lim, -lim, lim))
    sc.plot([-lim, lim], [-lim, lim], color=INK3, lw=1, ls="--", label="perfect")
    sc.plot([-lim, lim], [-lim * slope, lim * slope], color=ORANGE, lw=1.6, label=f"fit, slope {slope:.2f}")
    sc.axhline(0, color=GRAT, lw=.6); sc.axvline(0, color=GRAT, lw=.6)
    sc.set_xlim(-lim, lim); sc.set_ylim(-lim, lim); sc.set_aspect("equal")
    sc.set_xlabel(f"correction needed ({TN} - POWER), m/s", fontsize=8.5, color=INK2)
    sc.set_ylabel("correction made (model - POWER), m/s", fontsize=8.5, color=INK2)
    sc.set_title(f"Holdout cells, r = {rc:.2f}", fontsize=10.5, fontweight="bold", pad=9)
    sc.legend(fontsize=7.5, loc="upper left", framealpha=.9); sc.tick_params(labelsize=7.5, labelcolor=INK3)

    bx = ax[1, 3]
    sp, n_s = stn_rmse(P); sm, _ = stn_rmse(M)
    bars = [("POWER, grid", rp, "#9aa5b1"), ("model, grid", rm, "#2a78d6")]
    if n_s:
        bars += [(f"POWER, ISD ({n_s})", sp, "#c4cbd3"), (f"model, ISD ({n_s})", sm, "#86b6ef")]
    bx.barh([b[0] for b in bars][::-1], [b[1] for b in bars][::-1], color=[b[2] for b in bars][::-1], height=.5)
    for yi, (_, v, _) in enumerate(bars[::-1]):
        bx.text(v, yi, f"  {v:.2f}", va="center", fontsize=10, fontweight="bold", color=INK)
    bx.set_xlim(0, np.nanmax([b[1] for b in bars]) * 1.35)
    bx.set_xlabel("RMSE on this day over the held-out region, m/s", fontsize=8.5, color=INK2)
    bx.tick_params(labelsize=9, labelcolor=INK2)
    for s_ in ("top", "right"):
        bx.spines[s_].set_visible(False)
    bx.set_title("This day, holdout", fontsize=10.5, fontweight="bold", pad=9)

    ttl = f"{run}: {wind_plan.CHANNELS[r['chan']][0]}, {r['arch']}, {wind_plan.STORES[r['store']][3]} km, {day} ({label})"
    top = header(fig, ttl,
                 f"Daily mean 10 m wind speed. Metrics use the land cells inside the orange held-out region, which the "
                 f"model never trained on. Circles: ISD stations, squares: Oklahoma Mesonet, coloured with the observed "
                 f"daily mean on the same scale as the map (a dot that disappears into the map is a match). ISD RMSE "
                 f"counts only stations inside the holdout. Grey: no {TN} data or water.", chars=230)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, f"inference_{run}_{day}.png")


def sheet_testyear(run: str):
    plt = setup(); C = cmaps()
    from matplotlib.colors import TwoSlopeNorm
    r = PLAN[run]; st = store_for(r["store"]); te = st.i_test
    TN = target_name(r["store"])
    T = np.asarray(st._zarr["target"][te[0]:te[-1] + 1, 0], "float32") + st.clim[st.doy[te]]
    P = np.asarray(st._zarr["input"][te[0]:te[-1] + 1, st.stored.index("coarse_tmp")], "float32") \
        + st.clim_coarse[st.doy[te]]
    M = predict(st, run, te) + st.clim[st.doy[te]]
    ok = np.isfinite(T)
    with np.errstate(invalid="ignore"):
        n = ok.sum(0).astype("float32"); n[n == 0] = np.nan
        mean = lambda X: np.where(ok, X, 0).sum(0) / n                                   # noqa: E731
        Tm, Pm, Mm = mean(T), mean(P), mean(M)
        rP = np.sqrt(mean((P - T) ** 2)); rM = np.sqrt(mean((M - T) ** 2))
    del T, P, M
    land = st.land & np.isfinite(Tm); m = st.m_hold & np.isfinite(Tm)
    fig, ax = plt.subplots(2, 4, figsize=(20, 9.8))
    vmax = float(np.nanpercentile(np.r_[Tm[land], Pm[land], Mm[land]], 99.5))
    vmin = float(np.nanpercentile(np.r_[Tm[land], Pm[land], Mm[land]], 0.5))
    panel(fig, ax[0, 0], st, Tm, f"{TN}, 2023 mean", C["speed"], f"holdout {np.nanmean(Tm[m]):.2f} m/s", vmin=vmin, vmax=vmax)
    panel(fig, ax[0, 1], st, Pm, "POWER interpolated, 2023 mean", C["speed"], f"holdout {np.nanmean(Pm[m]):.2f} m/s",
          vmin=vmin, vmax=vmax)
    panel(fig, ax[0, 2], st, Mm, "Model, 2023 mean", C["speed"], f"holdout {np.nanmean(Mm[m]):.2f} m/s", vmin=vmin, vmax=vmax)
    bP, bM = Pm - Tm, Mm - Tm
    bl = float(np.nanpercentile(np.abs(np.r_[bP[land], bM[land]]), 99)) or 1.0
    panel(fig, ax[0, 3], st, bP, f"POWER bias (2023 mean - {TN})", C["div"],
          f"holdout {np.nanmean(bP[m]):+.2f} m/s", norm=TwoSlopeNorm(0, -bl, bl))
    panel(fig, ax[1, 0], st, bM, f"Model bias (2023 mean - {TN})", C["div"],
          f"holdout {np.nanmean(bM[m]):+.2f} m/s; same scale as POWER bias", norm=TwoSlopeNorm(0, -bl, bl))
    rv = float(np.nanpercentile(np.r_[rP[land], rM[land]], 99))
    panel(fig, ax[1, 1], st, rP, f"POWER RMSE vs {TN}, 2023", C["seq"],
          f"holdout {np.sqrt(np.nanmean(rP[m] ** 2)):.2f} m/s", vmin=0, vmax=rv)
    panel(fig, ax[1, 2], st, rM, f"Model RMSE vs {TN}, 2023", C["seq"],
          f"holdout {np.sqrt(np.nanmean(rM[m] ** 2)):.2f} m/s", vmin=0, vmax=rv)
    d = rM - rP; dl = float(np.nanpercentile(np.abs(d[land]), 99)) or 1.0
    panel(fig, ax[1, 3], st, d, "RMSE change (model - POWER)", C["div"],
          f"blue = model better; better at {100 * np.mean(d[m] < 0):.0f}% of holdout cells",
          norm=TwoSlopeNorm(0, -dl, dl))
    ttl = f"{run}: test year 2023, {wind_plan.CHANNELS[r['chan']][0]}, {r['arch']}, {wind_plan.STORES[r['store']][3]} km"
    top = header(fig, ttl,
                 f"Per-cell statistics over all 365 days of 2023 against {TN}. Outside the orange box the model saw these "
                 "cells in training (other years), so only the box is out-of-sample in space. Holdout RMSE is the "
                 "root of the mean squared error over all holdout cells and days.", chars=230)
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, f"testyear_{run}.png")


def sheet_rows(runs: list[str], day: str, label: str, fname: str, title: str, show_power: bool):
    """One row per run (different grids or holdouts) on the same day, shared colour scales across rows."""
    plt = setup(); C = cmaps()
    from matplotlib.colors import TwoSlopeNorm
    runs = [x for x in runs if (REPO / "checkpoints" / x / "last.pt").exists()]
    if not runs:
        log(f"  {fname}: no trained runs, skipped"); return
    rows = []
    for run in runs:
        r = PLAN[run]; st = store_for(r["store"]); k = day_index(st, day)
        T = truth_abs(st, k); P = input_abs(st, k, "coarse_tmp")
        M = predict(st, run, np.array([k]))[0] + st.clim[st.doy[k]]
        rows.append((run, r, st, T, P, M))
    land_vals = np.concatenate([np.r_[T[st.land & np.isfinite(T)], M[st.land & np.isfinite(T)]] for _, _, st, T, P, M in rows])
    vmax = float(np.nanpercentile(land_vals, 99.5))
    elim = float(np.nanpercentile(np.abs(np.concatenate([(M - T)[st.m_hold & np.isfinite(T)] for _, _, st, T, P, M in rows])), 99))
    ncol = 4 if show_power else 3
    fig, ax = plt.subplots(len(rows), ncol, figsize=(5 * ncol, 4.3 * len(rows) + 1.2), squeeze=False)
    for i, (run, r, st, T, P, M) in enumerate(rows):
        m = score_mask(st, T); c = 0
        tag = f"{wind_plan.STORES[r['store']][3]} km" if show_power else r["store"].replace("w10", "") or "austin"
        if show_power:
            panel(fig, ax[i, c], st, P, f"{tag}: POWER interpolated", C["speed"],
                  f"holdout RMSE {rmse(P, T, m):.2f} m/s", vmin=0, vmax=vmax); c += 1
        panel(fig, ax[i, c], st, T, f"{tag}: AORC", C["speed"], f"holdout mean {np.nanmean(T[m]):.2f} m/s",
              vmin=0, vmax=vmax); c += 1
        panel(fig, ax[i, c], st, M, f"{tag}: model", C["speed"], f"holdout RMSE {rmse(M, T, m):.2f} m/s",
              vmin=0, vmax=vmax); c += 1
        panel(fig, ax[i, c], st, M - T, f"{tag}: model - AORC", C["div"], f"holdout bias {bias(M, T, m):+.2f} m/s",
              norm=TwoSlopeNorm(0, -elim, elim))
    top = header(fig, f"{title}, {day} ({label})",
                 "Same day, shared colour scales across rows. Orange box: the held-out region of that row's run. "
                 "Runs: " + ", ".join(runs) + ".", chars=int(40 * ncol))
    fig.tight_layout(rect=(0, 0, 1, top))
    save(fig, fname)


# --------------------------------------------------------------------------------------------------------- main
def main(a):
    runs = [x for x in a.runs if (REPO / "checkpoints" / x / "last.pt").exists()]
    for x in set(a.runs) - set(runs):
        log(f"  {x}: no checkpoint, skipped")
    unknown = [x for x in runs if x not in PLAN]
    if unknown:
        raise SystemExit(f"not in scripts/wind_plan.py: {unknown}")
    if a.days:
        days = {d: d for d in a.days}
        labels = {d: "chosen" for d in a.days}
    else:
        picked = pick_days(store_for("w10"))
        days = {d: d for d in picked.values()}
        labels = {d: lab for lab, d in picked.items()}
    only = set(a.only or ["inputs", "inference", "testyear", "ladder", "loro"])
    for day in days:
        lab = labels[day]
        log(f"== {day} ({lab})")
        done_inputs = set()                                     # one inputs sheet per (store, channel set) per day
        for run in runs:
            r = PLAN[run]
            if "inputs" in only and (r["store"], r["chan"]) not in done_inputs:
                sheet_inputs(run, day, lab)
                done_inputs.add((r["store"], r["chan"]))
            if "inference" in only:
                sheet_inference(run, day, lab)
        if "ladder" in only:
            sheet_rows(["w25_L_s1337", "w10_L_s1337", "w5_L_s1337", "w1_L_s1337"], day, lab, f"ladder_{day}.png",
                       "Resolution ladder (channel set L, seed 1337)", show_power=True)
        if "loro" in only:
            sheet_rows(LORO_RUNS, day, lab, f"loro_{day}.png", "Leave-one-region-out at 10 km (channel set L, seed 1337)",
                       show_power=False)
    if "testyear" in only:
        for run in runs:
            sheet_testyear(run)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    ap.add_argument("--days", nargs="+", default=None, help="YYYY-MM-DD; default: windy, typical and calm 2023 days")
    ap.add_argument("--only", nargs="+", choices=["inputs", "inference", "testyear", "ladder", "loro"])
    main(ap.parse_args())
