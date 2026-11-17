"""Results figures for the wind program (WIND_RESULTS.md): the final pipeline stage by stage, on maps and in numbers.

Stages compared on the 10 km fusion stores, test year 2023 (all in m/s, climatology added back):
  POWER (interpolated input) -> no-station model (DeepSD L, 3 seeds) -> station fusion trained 2019-21 (LO, 3 seeds)
  -> station fusion trained 2019-22 -> product: diffusion corrector trained on 2022 with three-band calibration
  (obsvalcal3; 25 members, ensemble mean). The MSE corrector (obsvaldet) enters the numbers only.

Sheets (outputs/wind/maps/results_*.png):
  results_pipeline_<day>.png    row 1 the stages, one calibrated member and AORC; row 2 their errors and the 80% range
  results_testyear.png          2023 per-cell RMSE and mean bias of each stage
  results_texture_<day>.png     the holdout box at full resolution (AORC, no-station, fusion, ensemble mean, members)
                                and the amplitude spectrum of 2023 relative to AORC
  results_uncertainty.png       spread and spread/error maps, rank histograms raw vs calibrated, band amplitude
  results_findings.png          stage scores (both folds), input ablation, architectures, resolution ladder, drift
  results_product_2023.png      the product over 2023: mean, bias, RMSE, 80% range, its coverage, CRPS, gain over
                                POWER, and the daily holdout-mean series with the members' band (--only product)

Fold store 0 for maps (stations of fold 1 are withheld); the stage scores average both fold stores.
Reads saved samples and cached fields; only the 2019-22 fusion runs are predicted (CPU, a few minutes).
Usage (repo root, Vista login node): python scripts/wind_results_figs.py [--days 2023-02-10 2023-04-22 2023-11-16]
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
import wind_diffusion_maps as wdm                                                # noqa: E402
import wind_inference_maps as wim                                                # noqa: E402
import wind_obs_eval as woe                                                      # noqa: E402

OUT = REPO / "outputs" / "wind"
DIF = OUT / "diffusion"
SEEDS = (1337, 7, 42)
INK, INK2, INK3, GRAT = wim.INK, wim.INK2, wim.INK3, wim.GRAT
# reference categorical palette (light), fixed order; grey for "before" / reference series
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
GREY = "#9aa1a9"
STAGES = ["POWER", "No-station model", "Fusion 2019-21", "Fusion 2019-22", "Product (ens. mean)"]
SUB = {"POWER": "interpolated input", "No-station model": "DeepSD, 3 seeds", "Fusion 2019-21": "same-day ISD inputs",
       "Fusion 2019-22": "retrained with 2022", "Product (ens. mean)": "fusion + 2022 diffusion corrector"}


def log(*a):
    print(*a, flush=True)


# --------------------------------------------------------------------------------------------------------- data
def load_fold(f, meta, isd):
    F = woe.Fold(f, meta, isd)
    te, cl = F.te, F.clim
    base = wim.store_for("w10")
    assert (base.times[base.i_test] == F.st.times[te]).all()
    X = {"POWER": F.Pabs,
         "No-station model": np.mean([wim.predict(base, f"w10_L_s{s}", te) for s in SEEDS], 0) + cl,
         "Fusion 2019-21": np.load(DIF / f"obsval_w10o{f}_det.npy") + cl,
         "Fusion 2019-22": np.mean([wim.predict(F.st, f"w10o0_LO4y_s{s}", te) for s in SEEDS], 0) + cl}
    E = np.load(DIF / f"obsvalcal3_w10o{f}_test.npy", mmap_mode="r")
    X["Product (ens. mean)"] = np.asarray(E.mean(0)) + cl
    X["MSE corrector"] = np.asarray(np.load(DIF / f"obsvaldet_w10o{f}_test.npy", mmap_mode="r")[0]) + cl
    return F, X, E


def stn_mae(F, A, rows):
    e = []
    for _, i, j, o in rows:
        ok = np.isfinite(o)
        if ok.sum() >= 100:
            e.append(np.mean(np.abs(A[ok, i, j] - o[ok])))
    return float(np.mean(e))


def scores(F, X):
    m = F.mask
    out = {}
    for k, A in X.items():
        d = (A - F.Tabs)[:, m]
        out[k] = dict(rmse=float(np.sqrt(np.nanmean(d ** 2))), bias=float(np.nanmean(d)),
                      isd_mae=stn_mae(F, A, F.eval_isd), meso_mae=stn_mae(F, A, F.eval_meso))
    return out


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRAT)
    ax.tick_params(labelsize=8, colors=INK3, labelcolor=INK2)
    ax.grid(axis=grid_axis, color=GRAT, lw=.6, alpha=.7); ax.set_axisbelow(True)


def title(ax, t, sub=""):
    ax.set_title(t, fontsize=10.5, fontweight="bold", pad=17 if sub else 8, loc="left")
    if sub:
        ax.text(0, 1.012, sub, transform=ax.transAxes, ha="left", va="bottom", fontsize=7.5, color=INK3)


def mapsub(ax, t, sub):
    """Re-title a wim.panel map with a readable one-line subtitle."""
    ax.set_title(t, fontsize=11.5, fontweight="bold", pad=20)
    ax.text(.5, 1.015, sub, transform=ax.transAxes, ha="center", va="bottom", fontsize=9, color=INK2)


def suptitle(fig, t, sub, top=.93):
    fig.suptitle(t, x=.01, y=.995, ha="left", fontsize=15, fontweight="bold")
    fig.text(.01, .968, sub, fontsize=9, color=INK3, va="top")
    return top


# ------------------------------------------------------------------------------------------------------ sheets
def sheet_pipeline(F, X, E, day, label):
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.colors import TwoSlopeNorm
    st = F.st; m = F.mask
    k = int(np.where(st.times[F.te] == pd.Timestamp(day))[0][0])
    T = F.Tabs[k]
    mem = np.asarray(E[:, k]) + F.clim[k]
    q10, q90 = np.percentile(mem, [10, 90], axis=0)
    cols = {s: X[s][k] for s in STAGES}
    cols["Product: one member"] = mem[0]
    stn = [(kd, i, j, float(v)) for kd, i, j, v in wdm.withheld_stations(F, k) if np.isfinite(v)]
    land = st.land & np.isfinite(T)
    vmax = float(np.nanpercentile(np.r_[T[land], mem[0][land]], 99.5))
    errs = {s: A - T for s, A in cols.items()}
    elim = float(np.nanpercentile(np.abs(np.concatenate([e[m] for e in errs.values()])), 99)) or 1.0

    def isd_day(A):
        v = [abs(A[i, j] - o) for kd, i, j, o in stn if kd == "isd"]
        return np.mean(v) if v else np.nan

    fig, ax = plt.subplots(2, 7, figsize=(30, 9.3))
    for n, (s, A) in enumerate(cols.items()):
        wim.panel(fig, ax[0, n], st, A, s, C["speed"], "", stn=stn, vmin=0, vmax=vmax)
        mapsub(ax[0, n], s, f"RMSE {wim.rmse(A, T, m):.2f}  |  ISD MAE {isd_day(A):.2f}")
        wim.panel(fig, ax[1, n], st, errs[s], f"Error: {s}", C["div"], "", norm=TwoSlopeNorm(0, -elim, elim))
        mapsub(ax[1, n], f"Error: {s}", f"bias {wim.bias(A, T, m):+.2f} m/s")
    wim.panel(fig, ax[0, 6], st, T, "AORC (truth)", C["speed"], "", stn=stn, vmin=0, vmax=vmax)
    mapsub(ax[0, 6], "AORC (truth)", f"holdout mean {np.nanmean(T[m]):.2f} m/s")
    inside = np.mean(((T >= q10) & (T <= q90))[m])
    wim.panel(fig, ax[1, 6], st, q90 - q10, "Product: 80% range", C["seq"], "", vmin=0,
              vmax=float(np.nanpercentile((q90 - q10)[land], 99)))
    mapsub(ax[1, 6], "Product: 80% range (q90 - q10)", f"AORC inside at {inside:.0%} of holdout cells")
    top = suptitle(fig, f"From POWER to the product, {day} ({label}), 10 m wind speed",
                   "Stages: POWER (interpolated input), no-station model (DeepSD, 3 seeds), station fusion trained "
                   "2019-21, the same retrained with 2022, the product (fusion 2019-21 + diffusion corrector trained on "
                   "2022, calibrated; mean and one of 25 members), AORC.\nRow 2: errors against AORC on one scale. "
                   "RMSE over land cells in the orange held-out box; ISD MAE at the stations withheld from the inputs "
                   f"(circles, fold {1 - F.f}). Squares: Oklahoma Mesonet (never an input). Markers show the observed "
                   "daily mean.", top=.9)
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, f"results_pipeline_{day}.png")


def sheet_testyear(F, X):
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.colors import TwoSlopeNorm
    st = F.st; m = F.mask; T = F.Tabs
    land = st.land & np.isfinite(T).all(0)
    R = {s: np.sqrt(np.nanmean((X[s] - T) ** 2, 0)) for s in STAGES}
    B = {s: np.nanmean(X[s] - T, 0) for s in STAGES}
    rv = float(np.nanpercentile(np.concatenate([r[land] for r in R.values()]), 99))
    bl = float(np.nanpercentile(np.abs(np.concatenate([b[land] for b in B.values()])), 99))
    fig, ax = plt.subplots(2, 5, figsize=(23, 9.2))
    for n, s in enumerate(STAGES):
        wim.panel(fig, ax[0, n], st, R[s], f"{s}: RMSE", C["seq"], "", vmin=0, vmax=rv)
        mapsub(ax[0, n], f"{s}: RMSE", f"{SUB[s]}  |  holdout {np.sqrt(np.nanmean((X[s] - T)[:, m] ** 2)):.3f}")
        wim.panel(fig, ax[1, n], st, B[s], f"{s}: mean bias", C["div"], "", norm=TwoSlopeNorm(0, -bl, bl))
        mapsub(ax[1, n], f"{s}: mean bias", f"holdout {np.nanmean((X[s] - T)[:, m]):+.2f} m/s")
    top = suptitle(fig, "Test year 2023: per-cell error of each stage against AORC (fold store 0)",
                   "Row 1: root-mean-square error over the 365 days; row 2: mean bias (red too windy, blue too calm). "
                   "Only the orange box is out of sample in space: outside it the models trained on these cells "
                   "(other years)\nand the fusion models see a random 60% of the ISD stations each day. Holdout "
                   "numbers pool all box cells and days; WIND_RESULTS.md averages both fold stores.", top=.91)
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, "results_testyear.png")


def spectrum(A, box, cell_km=10.0):
    """Radially averaged power spectrum of (N, h, w) fields in the box: (wavelength km, power)."""
    i0, i1, j0, j1 = box
    Z = np.asarray(A[..., i0:i1, j0:j1], "float64")
    Z = Z.reshape(-1, *Z.shape[-2:])
    mu = np.nanmean(Z, axis=(1, 2), keepdims=True)
    Z = np.where(np.isfinite(Z), Z - mu, 0.0)
    h, w = Z.shape[1:]
    win = np.outer(np.hanning(h), np.hanning(w))
    P = (np.abs(np.fft.fft2(Z * win)) ** 2).mean(0)
    ky, kx = np.meshgrid(np.fft.fftfreq(h), np.fft.fftfreq(w), indexing="ij")
    kr = np.hypot(ky, kx)
    edges = np.arange(1, min(h, w) // 2 + 1) / min(h, w)
    idx = np.digitize(kr.ravel(), edges)
    pw = np.array([P.ravel()[idx == b].mean() for b in range(1, len(edges))])
    kc = 0.5 * (edges[1:] + edges[:-1])
    return cell_km / kc, pw


def sheet_texture(F, X, E, day, label, spec):
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.patches import Rectangle
    st = F.st
    i0, i1, j0, j1 = st.box
    pad = 6
    a0, a1 = max(i0 - pad, 0), min(i1 + pad, st.shape[0]); b0, b1 = max(j0 - pad, 0), min(j1 + pad, st.shape[1])
    k = int(np.where(st.times[F.te] == pd.Timestamp(day))[0][0])
    T = F.Tabs[k]
    mem = np.asarray(E[:3, k]) + F.clim[k]
    P = {"AORC (truth)": T, "No-station model": X["No-station model"][k], "Fusion 2019-21 (base)": X["Fusion 2019-21"][k],
         "Product: ensemble mean": X["Product (ens. mean)"][k], "Product: member 1": mem[0],
         "Product: member 2": mem[1], "Product: member 3": mem[2]}
    crop = lambda A: A[a0:a1, b0:b1]                                                  # noqa: E731
    v = np.concatenate([crop(A)[np.isfinite(crop(A))] for A in P.values()])
    vmin, vmax = np.percentile(v, [0.5, 99.5])
    fig, ax = plt.subplots(2, 4, figsize=(21, 10))
    ax = ax.ravel()
    m = F.mask
    for n, (s, A) in enumerate(P.items()):
        a = ax[n]
        im = a.imshow(np.ma.masked_invalid(crop(np.where(st.land, A, np.nan))), cmap=C["speed"], vmin=vmin, vmax=vmax,
                      interpolation="nearest")
        a.add_patch(Rectangle((j0 - b0 - .5, i0 - a0 - .5), j1 - j0, i1 - i0, fill=False, ec=wim.ORANGE, lw=1.6))
        a.set_xticks([]); a.set_yticks([])
        sub = f"holdout mean {np.nanmean(A[m]):.2f} m/s" if n == 0 else f"holdout RMSE {wim.rmse(A, T, m):.2f} m/s"
        title(a, s, sub)
        cb = fig.colorbar(im, ax=a, fraction=.042, pad=.02); cb.ax.tick_params(labelsize=7, labelcolor=INK3)
        cb.set_label("m/s", fontsize=7.5, color=INK3)
    a = ax[7]
    wl, p_t = spec["AORC"]
    ser = [("No-station model", GREY), ("Fusion 2019-21 (base)", C2), ("Product: ensemble mean", C3),
           ("Product: members, raw", C4), ("Product: members, calibrated", C1)]
    for s, col in ser:
        _, p = spec[s]
        a.plot(wl, np.sqrt(p / p_t), color=col, lw=2, label=s, marker="o", ms=3.5)
    a.axhline(1, color=INK3, ls="--", lw=1)
    a.text(wl.max(), 1.02, "AORC", color=INK3, fontsize=7.5, ha="right", va="bottom")
    a.set_xscale("log"); a.invert_xaxis()
    from matplotlib.ticker import FixedLocator, NullLocator
    a.xaxis.set_major_locator(FixedLocator([300, 200, 100, 50, 30, 20]))
    a.xaxis.set_minor_locator(NullLocator())
    a.set_xticklabels(["300", "200", "100", "50", "30", "20"])
    a.set_xlabel("wavelength (km)", fontsize=8.5, color=INK2); a.set_ylabel("amplitude / AORC", fontsize=8.5, color=INK2)
    a.set_ylim(0, 1.6)
    style(a, "y")
    title(a, "Amplitude by wavelength, 2023", "daily anomalies in the holdout box, all 365 days; 1 = AORC")
    a.legend(fontsize=7.5, frameon=False, loc="lower left")
    top = suptitle(fig, f"Texture in the held-out box, {day} ({label})",
                   "Full-resolution 10 km cells around the holdout (orange). The deterministic stages are smooth below "
                   "about 100 km; the calibrated members carry texture of AORC's strength,\nbut its placement is "
                   "random, so the members differ from each other and the ensemble mean stays smooth.")
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, f"results_texture_{day}.png")


def sheet_uncertainty(F, Eraw, Ecal):
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.colors import TwoSlopeNorm
    st = F.st; m = F.mask; Ta = F.Ta
    Sd = np.sqrt(np.mean(np.asarray(Ecal.var(0, ddof=1)), 0))
    Em = np.asarray(Ecal.mean(0))
    err = np.sqrt(np.nanmean((Em - Ta) ** 2, 0))
    land = st.land & np.isfinite(Ta).all(0)
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 6)
    a = fig.add_subplot(gs[0, 0:2])
    wim.panel(fig, a, st, Sd, "Product spread, 2023 (rms sd of 25 members)", C["seq"],
              f"holdout {np.sqrt(np.nanmean(Sd[m] ** 2)):.2f} m/s", vmin=0, vmax=float(np.nanpercentile(Sd[land], 99)))
    a = fig.add_subplot(gs[0, 2:4])
    rat_h = np.sqrt(np.nanmean(Sd[m] ** 2)) / np.sqrt(np.nanmean(err[m] ** 2))
    wim.panel(fig, a, st, Sd / err, "Spread / error of the ensemble mean", C["div"],
              f"1 = calibrated; holdout {rat_h:.2f}. Outside the box errors are smaller (trained cells)", unit="",
              norm=TwoSlopeNorm(1, 0.4, 1.6))
    # band amplitude of members, by calibration
    a = fig.add_subplot(gs[0, 4:6])
    tags = [("obsval", "raw", GREY), ("obsvalcal", "one factor", C2), ("obsvalcal2", "two bands", C4),
            ("obsvalcal3", "three bands (product)", C1)]
    bands = [("lt100", "< 100 km"), ("100_300", "100-300 km"), ("gt300", "> 300 km")]
    w = .2
    for n, (tg, lab, col) in enumerate(tags):
        d = json.loads((DIF / f"{tg}_score.json").read_text())
        vals = [np.mean([d[s]["member"]["bands"][b]["sd_ratio"] for s in ("w10o0", "w10o1")]) for b, _ in bands]
        xs = np.arange(3) + (n - 1.5) * w
        a.bar(xs, vals, width=w * .9, color=col, label=lab)
        for x, v in zip(xs, vals):
            if tg in ("obsval", "obsvalcal3") or v > 1.15:
                a.text(x, v + .02, f"{v:.2f}", ha="center", fontsize=7, color=INK2)
    a.axhline(1, color=INK, lw=1, ls="--")
    a.set_xticks(np.arange(3)); a.set_xticklabels([b[1] for b in bands])
    a.set_ylim(0, 1.5); a.set_ylabel("member sd / AORC sd", fontsize=8.5, color=INK2)
    style(a, "y")
    title(a, "Texture strength by scale band, 2023", "1 = AORC; mean of both fold stores")
    a.legend(fontsize=7.5, frameon=False, ncol=2, loc="upper center")
    # rank histograms, raw vs calibrated
    nm = Ecal.shape[0]
    ii, jj = np.where(m)
    sel = np.random.default_rng(0).choice(len(ii), min(400, len(ii)), replace=False)
    yg = Ta[:, ii[sel], jj[sel]].ravel(); okg = np.isfinite(yg)

    def rk_sets(E):
        Xg = np.asarray(E[:, :, ii[sel], jj[sel]]).reshape(nm, -1)
        out = {"Holdout grid vs AORC": wdm.ranks(Xg[:, okg], yg[okg])}
        for kind, rows in (("Withheld ISD stations", F.eval_isd), ("Oklahoma Mesonet", F.eval_meso)):
            r = []
            for _, i, j, o in rows:
                ok = np.isfinite(o)
                if ok.sum() >= 100:
                    r.append(wdm.ranks(np.asarray(E[:, ok, i, j]) + F.clim[ok, i, j][None], o[ok]))
            out[kind] = np.concatenate(r)
        return out

    R0, R1 = rk_sets(Eraw), rk_sets(Ecal)
    for n, lab in enumerate(R0):
        a = fig.add_subplot(gs[1, 2 * n:2 * n + 2])
        for R, col, nmx, off in ((R0, GREY, "raw", -.2), (R1, C1, "calibrated (product)", .2)):
            h = np.bincount(R[lab], minlength=nm + 1) / len(R[lab]) * (nm + 1)
            a.bar(np.arange(nm + 1) + off, h, width=.4, color=col,
                  label=f"{nmx}: {h[0] / (nm + 1) + h[-1] / (nm + 1):.0%} outside all members")
        a.axhline(1, color=INK, lw=1, ls="--")
        a.set_xlabel("rank of the observation among the 25 members", fontsize=8.5, color=INK2)
        a.set_ylabel("relative frequency (1 = flat)", fontsize=8.5, color=INK2)
        style(a, "y")
        title(a, f"Rank histogram: {lab}", f"flat = calibrated, U = too narrow; ideal outside-all {2 / (nm + 1):.0%}")
        a.legend(fontsize=7.5, frameon=False, loc="upper center")
    top = suptitle(fig, "Uncertainty of the product, test year 2023 (fold store 0)",
                   "The raw diffusion ensemble is too narrow (spread/skill 0.59). Its member deviations are widened by "
                   "scale band with factors fitted on 2022 only (x3.5 above 300 km, x1.18 at 100-300 km, x1.35 below "
                   "100 km).\nThe last panel of row 1 shows why three bands: one factor makes the fine texture a third "
                   "too strong. ISD remains too narrow because airports read windier than their 10 km cell.")
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, "results_uncertainty.png")


def sheet_findings(S):
    plt = wim.setup()
    summ = json.loads((OUT / "wind_summary.json").read_text())
    cmp_ = summ["comparisons"]
    drift = json.loads((OUT / "drift.json").read_text())["years"]
    fig, ax = plt.subplots(2, 3, figsize=(21, 11))
    # (a) grid RMSE by stage
    order = STAGES[:4] + ["MSE corrector", "Product (ens. mean)"]
    names = ["POWER", "No-station model", "Fusion 2019-21", "Fusion 2019-22", "Fusion 2019-21\n+ MSE corrector",
             "Fusion 2019-21\n+ diffusion (product)"]
    a = ax[0, 0]
    v = [S[s]["rmse"] for s in order]
    y = np.arange(len(order))[::-1]
    a.barh(y, v, color=[GREY] * 5 + [C1], height=.6)
    for yy, vv in zip(y, v):
        a.text(vv + .01, yy, f"{vv:.3f}", va="center", fontsize=8.5, color=INK)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8.5); a.set_xlim(0, max(v) * 1.18)
    a.set_xlabel("holdout grid RMSE vs AORC (m/s)", fontsize=8.5, color=INK2)
    style(a, "x"); title(a, "(a) Grid error by stage, 2023", "mean of both fold stores; lower is better")
    # (b) station MAE by stage
    a = ax[0, 1]
    h = .36
    for n, (key, lab, col) in enumerate((("isd_mae", "withheld ISD", C1), ("meso_mae", "Oklahoma Mesonet", C2))):
        vv = [S[s][key] for s in order]
        a.barh(y + (h / 2 if n == 0 else -h / 2), vv, height=h * .92, color=col, label=lab)
        for yy, x in zip(y, vv):
            a.text(x + .01, yy + (h / 2 if n == 0 else -h / 2), f"{x:.2f}", va="center", fontsize=7.5, color=INK2)
    a.set_yticks(y); a.set_yticklabels(names, fontsize=8.5)
    a.set_xlabel("station MAE, daily mean speed (m/s)", fontsize=8.5, color=INK2)
    style(a, "x"); title(a, "(b) Station error by stage, 2023", "stations never used as inputs; lower is better")
    a.legend(fontsize=8, frameon=False, loc="lower right")

    def ci_plot(a, rows, ttl, sub, xlabel):
        yy = np.arange(len(rows))[::-1]
        for (lab, c), y_ in zip(rows, yy):
            col = C1 if c["hi"] < 0 else (C2 if c["lo"] > 0 else GREY)
            a.plot([c["lo"], c["hi"]], [y_, y_], color=col, lw=2.2, solid_capstyle="round")
            a.plot(c["d"], y_, "o", color=col, ms=8, mec="white", mew=1.5)
        lo_, hi_ = min(min(c["lo"] for _, c in rows), 0), max(max(c["hi"] for _, c in rows), 0)
        off = .03 * (hi_ - lo_)
        for (lab, c), y_ in zip(rows, yy):
            a.text(c["hi"] + off, y_, f"{c['d']:+.3f}", va="center", fontsize=7.5, color=INK2)
        a.set_xlim(lo_ - 2 * off, hi_ + 6 * off)
        a.axvline(0, color=INK, lw=1)
        a.set_yticks(yy); a.set_yticklabels([r[0] for r in rows], fontsize=8.5)
        a.set_xlabel(xlabel, fontsize=8.5, color=INK2)
        style(a, "x"); title(a, ttl, sub)

    # (c) inputs
    lab = {"A1": "+ basic statics", "A2": "+ terrain", "A3": "+ land-cover fractions", "A4": "all 21 statics",
           "A5": "statics + direction", "A6": "statics + shear", "A7": "statics + gust, stability", "A8": "all 27"}
    ci_plot(ax[0, 2], [(f"{k} {v}", cmp_[f"ablation {k} vs A0"]) for k, v in lab.items()],
            "(c) Inputs: change in ISD RMSE vs POWER speed only (A0)",
            "95% CI from 5,000 station bootstraps; orange = significantly worse", "ISD RMSE difference (m/s)")
    # (d) architectures
    ci_plot(ax[1, 0], [(n, cmp_[f"arch {n} vs deepsd"]) for n in
                       ("linear", "swin", "swinir", "unet", "edsr", "restormer", "convnext")],
            "(d) Architecture: change in ISD RMSE vs DeepSD (all 27 inputs)",
            "blue = significantly better; differences are 1.65x the seed noise", "ISD RMSE difference (m/s)")
    # (e) resolution ladder
    a = ax[1, 1]
    steps = [("25 -> 10 km", "w25->w10"), ("10 -> 5 km", "w10->w5"), ("5 -> 1 km", "w5->w1")]
    for n, (lab_, key) in enumerate(steps):
        for off, nm_, col in ((.15, "ladder", C1), (-.15, "target itself", GREY)):
            c = cmp_[f"{nm_} {key}"]
            y_ = 2 - n + off
            a.plot([c["lo"], c["hi"]], [y_, y_], color=col, lw=2.2, solid_capstyle="round")
            a.plot(c["d"], y_, "o", color=col, ms=8, mec="white", mew=1.5,
                   label=("model (DeepSD, ladder inputs)" if nm_ == "ladder" else "AORC itself") if n == 0 else None)
            a.text(c["hi"] + .005, y_, f"{c['d']:+.3f}", va="center", fontsize=7.5, color=INK2)
    a.axvline(0, color=INK, lw=1)
    a.set_xlim(-.15, .075)
    a.set_yticks([2, 1, 0]); a.set_yticklabels([s[0] for s in steps], fontsize=8.5)
    a.set_xlabel("change in ISD RMSE when the grid is refined (m/s)", fontsize=8.5, color=INK2)
    style(a, "x"); a.legend(fontsize=8, frameon=False, loc="lower left")
    title(a, "(e) Resolution: finer target, no better model", "95% CI; negative = finer grid is better at stations")
    # (f) drift
    a = ax[1, 2]
    yrs = sorted(drift)
    xs = [int(t) for t in yrs]
    for key, reg, lab_, col in (("power_minus_aorc", "holdout", "POWER - AORC, holdout", C2),
                                ("power_minus_aorc", "train_region", "POWER - AORC, training region", C4),
                                ("A0_bias", "holdout", "model bias (A0), holdout", C1)):
        vv = [drift[t][reg][key] for t in yrs]
        a.plot(xs, vv, "o-", color=col, lw=2, ms=6, label=lab_)
        a.text(xs[-1] + .08, vv[-1], f"{vv[-1]:+.2f}", va="center", fontsize=7.5, color=INK2)
    a.axhline(0, color=INK, lw=1)
    a.axvspan(2018.6, 2021.4, color="#e9e7e1", lw=0, zorder=0)
    a.text(2020, 1.75, "training years", ha="center", fontsize=7.5, color=INK3)
    a.set_xticks(xs); a.set_xlim(2018.6, 2023.6); a.set_ylim(-0.8, 1.9)
    a.set_ylabel("m/s", fontsize=8.5, color=INK2)
    style(a, "y"); a.legend(fontsize=8, frameon=False, loc="lower left")
    title(a, "(f) Drift: the POWER - AORC gap shrinks after 2020",
          "the model learns 2019-21's offset and is too calm by 2022-23")
    top = suptitle(fig, "Wind downscaling: what moved the error, test year 2023",
                   "(a, b) the final pipeline stage by stage on the 10 km fusion stores. (c-e) the no-station "
                   "experiments, at the 44 ISD stations of the holdout area. (f) year-by-year means over land.")
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, "results_findings.png")


def sheet_product(F, X, E):
    """The product over 2023: mean field vs AORC, per-cell error, 80% range, its coverage and CRPS, and the daily
    holdout-mean series with the members' 10-90% band."""
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.colors import TwoSlopeNorm
    st = F.st; m = F.mask; T = F.Tabs; cl = F.clim
    land = st.land & np.isfinite(T).all(0)
    Em = X["Product (ens. mean)"]
    q10 = np.empty_like(Em); q90 = np.empty_like(Em); crps = np.empty_like(Em)
    nm = E.shape[0]
    w = (2 * np.arange(1, nm + 1) - nm - 1)[:, None, None].astype("float32")
    for k in range(E.shape[1]):                              # day by day keeps memory small
        Xk = np.sort(np.asarray(E[:, k]) + cl[k][None], 0)
        q10[k], q90[k] = np.percentile(Xk, [10, 90], axis=0)
        crps[k] = np.mean(np.abs(Xk - T[k][None]), 0) - (w * Xk).sum(0) / (nm * (nm - 1))
    inside = ((T >= q10) & (T <= q90)).astype("float32"); inside[~np.isfinite(T)] = np.nan
    fig = plt.figure(figsize=(24, 13.5))
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, .75])
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]
    Tm, Mm = np.nanmean(T, 0), np.nanmean(Em, 0)
    vmin, vmax = np.nanpercentile(np.r_[Tm[land], Mm[land]], [0.5, 99.5])
    wim.panel(fig, ax[0], st, Tm, "AORC, 2023 mean", C["speed"], "", vmin=vmin, vmax=vmax)
    mapsub(ax[0], "AORC, 2023 mean", f"holdout {np.nanmean(Tm[m]):.2f} m/s")
    wim.panel(fig, ax[1], st, Mm, "Product (ensemble mean), 2023 mean", C["speed"], "", vmin=vmin, vmax=vmax)
    mapsub(ax[1], "Product (ensemble mean), 2023 mean", f"holdout {np.nanmean(Mm[m]):.2f} m/s")
    b = Mm - Tm; bl = float(np.nanpercentile(np.abs(b[land]), 99))
    wim.panel(fig, ax[2], st, b, "Product bias, 2023", C["div"], "", norm=TwoSlopeNorm(0, -bl, bl))
    mapsub(ax[2], "Product bias, 2023", f"holdout {np.nanmean(b[m]):+.2f} m/s (POWER +0.84)")
    r = np.sqrt(np.nanmean((Em - T) ** 2, 0))
    wim.panel(fig, ax[3], st, r, "Product RMSE, 2023", C["seq"], "", vmin=0, vmax=float(np.nanpercentile(r[land], 99)))
    mapsub(ax[3], "Product RMSE, 2023", f"holdout {np.sqrt(np.nanmean((Em - T)[:, m] ** 2)):.3f} m/s (POWER 1.128)")
    wd = np.nanmean(q90 - q10, 0)
    wim.panel(fig, ax[4], st, wd, "80% range width, 2023 mean", C["seq"], "", vmin=0,
              vmax=float(np.nanpercentile(wd[land], 99)))
    mapsub(ax[4], "80% range width (q90 - q10)", f"holdout {np.nanmean(wd[m]):.2f} m/s")
    cv = np.nanmean(inside, 0)
    wim.panel(fig, ax[5], st, cv, "AORC inside the 80% range", C["div"], "", unit="fraction of days",
              norm=TwoSlopeNorm(.8, .5, 1.0))
    mapsub(ax[5], "AORC inside the 80% range", f"holdout {np.nanmean(inside[:, m]):.0%} of cell-days (target 80%)")
    cr = np.nanmean(crps, 0)
    wim.panel(fig, ax[6], st, cr, "CRPS, 2023", C["seq"], "", vmin=0, vmax=float(np.nanpercentile(cr[land], 99)))
    mapsub(ax[6], "CRPS, 2023 (lower is better)", f"holdout {np.nanmean(crps[:, m]):.3f} m/s (ens. mean MAE "
           f"{np.nanmean(np.abs(Em - T)[:, m]):.3f})")
    P = X["POWER"]
    wim.panel(fig, ax[7], st, np.sqrt(np.nanmean((P - T) ** 2, 0)) - r, "RMSE gain over POWER", C["div"], "",
              norm=TwoSlopeNorm(0, -1.2, 1.2))
    mapsub(ax[7], "RMSE gain over POWER (POWER - product)", "red = product better")
    # daily holdout-mean series
    a = fig.add_subplot(gs[2, :])
    days = pd.DatetimeIndex(st.times[F.te])
    hm = lambda A: np.nanmean(A[:, m], 1)                                              # noqa: E731
    mem = np.stack([np.nanmean((np.asarray(E[i]) + cl)[:, m], 1) for i in range(nm)])
    lo, hi = np.percentile(mem, [10, 90], axis=0)
    a.fill_between(days, lo, hi, color=C1, alpha=.22, lw=0, label="product, members 10-90%")
    a.plot(days, hm(P), color=GREY, lw=1, label="POWER")
    a.plot(days, hm(T), color=INK, lw=1.4, label="AORC")
    a.plot(days, hm(Em), color=C1, lw=1.4, label="product, ensemble mean")
    a.set_ylabel("holdout mean speed (m/s)", fontsize=8.5, color=INK2)
    a.set_xlim(days[0], days[-1]); style(a, "y")
    a.legend(fontsize=8, frameon=False, ncol=4, loc="upper right")
    inband = np.mean((hm(T) >= lo) & (hm(T) <= hi))
    title(a, "Daily mean over the held-out box, 2023",
          f"AORC inside the members' 10-90% band on {inband:.0%} of days; daily-mean RMSE: POWER "
          f"{np.sqrt(np.mean((hm(P) - hm(T)) ** 2)):.2f}, product {np.sqrt(np.mean((hm(Em) - hm(T)) ** 2)):.2f} m/s")
    top = suptitle(fig, "The product: calibrated diffusion ensemble, test year 2023 (fold store 0)",
                   "Station fusion network (2019-21) + residual diffusion corrector trained on its 2022 errors, 25 "
                   "members widened by scale band (factors fitted on 2022). Rows 1-2: per-cell statistics over the 365 "
                   "days;\nonly the orange box is out of sample in space. Row 3: the daily mean over the box, where the "
                   "band is the spread of the members' own box means.", top=.93)
    fig.tight_layout(rect=(0, 0, 1, top))
    wim.save(fig, "results_product_2023.png")


def main(a):
    if a.only == "product":
        meta = json.loads((OUT / "obs_stations.json").read_text())["stations"]
        d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
        isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
        F, X, E = load_fold(0, meta, isd)
        return sheet_product(F, X, E)
    meta = json.loads((OUT / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    S = {}
    for f in (1, 0):                                    # fold 0 last: its fields stay loaded for the maps
        F, X, E = load_fold(f, meta, isd)
        S[f] = scores(F, X)
        log(f"fold {f}:", json.dumps(S[f]))
    Sm = {k: {q: (S[0][k][q] + S[1][k][q]) / 2 for q in S[0][k]} for k in S[0]}
    (OUT / "results_stage_scores.json").write_text(json.dumps(dict(per_fold=S, mean=Sm), indent=1))
    sheet_findings(Sm)
    sheet_testyear(F, X)
    days = dict(zip(a.days, ("windy", "typical", "calm"))) if a.days else {}
    for day, lab in days.items():
        sheet_pipeline(F, X, E, day, lab)
    # texture spectra over all 2023 days (members: 5 of 25)
    box = F.st.box
    Eraw = np.load(DIF / f"obsval_w10o{F.f}_test.npy", mmap_mode="r")
    an = lambda A: A - F.clim                                                         # noqa: E731  daily anomaly
    spec = {"AORC": spectrum(F.Ta, box), "No-station model": spectrum(an(X["No-station model"]), box),
            "Fusion 2019-21 (base)": spectrum(an(X["Fusion 2019-21"]), box),
            "Product: ensemble mean": spectrum(an(X["Product (ens. mean)"]), box),
            "Product: members, raw": spectrum(np.asarray(Eraw[:5]), box),
            "Product: members, calibrated": spectrum(np.asarray(E[:5]), box)}
    for day, lab in list(days.items())[:2]:
        sheet_texture(F, X, E, day, lab, spec)
    sheet_uncertainty(F, Eraw, E)
    sheet_product(F, X, E)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="*", default=["2023-02-10", "2023-04-22", "2023-11-16"])
    ap.add_argument("--only", choices=["product"], help="draw only this sheet")
    main(ap.parse_args())
