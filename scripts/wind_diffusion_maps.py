"""Maps for the residual-diffusion ensemble (WIND_WORK_LOG 7d): what the members look like and whether they are calibrated.

Reads the saved 2023 samples (outputs/wind/diffusion/<tag>_w10o<f>_test.npy, 25 members, anomaly) and the
deterministic base (<tag>_w10o<f>_det.npy); no GPU or checkpoint needed. Fold store f (default 0): the withheld ISD
stations are fold 1-f. Sheets:

  diffusion_<tag>_<day>.png   row 1  AORC | base network | MSE corrector (control tag) | ensemble mean
                              row 2  three members | ensemble spread (sd)
                              row 3  errors vs AORC of base, corrector, ensemble mean, one member (shared scale)
  diffusion_<tag>_testyear.png  per-cell 2023 RMSE of base and ensemble mean, mean spread, spread/error ratio map,
                              rank histograms (holdout grid, withheld ISD, Mesonet), binned spread vs error

Usage (repo root, login node is fine):
    python scripts/wind_diffusion_maps.py [--tag obsval] [--control obsvaldet] [--fold 0] [--days 2023-02-10 ...]
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

DIF = REPO / "outputs" / "wind" / "diffusion"
INK3 = wim.INK3


def load(tag, f):
    E = np.load(DIF / f"{tag}_w10o{f}_test.npy", mmap_mode="r")
    det = np.load(DIF / f"{tag}_w10o{f}_det.npy")
    return E, det


def withheld_stations(F, day=None):
    """[(kind, row, col, obs)] at the withheld ISD fold and Mesonet; obs on `day` (index into test days) or series."""
    out = []
    for kind, rows in (("isd", F.eval_isd), ("meso", F.eval_meso)):
        for _, i, j, o in rows:
            out.append((kind, i, j, o if day is None else o[day]))
    return out


def sheet_day(F, tag, control, E, det, ctl, day, label):
    plt = wim.setup(); C = wim.cmaps()
    from matplotlib.colors import TwoSlopeNorm
    st = F.st
    k = int(np.where(st.times[st.i_test] == pd.Timestamp(day))[0][0])
    cl = F.clim[k]
    T = F.Tabs[k]
    B = det[k] + cl
    Mem = np.asarray(E[:, k]) + cl
    Em = Mem.mean(0); Sd = Mem.std(0, ddof=1)
    Ct = (ctl[k] + cl) if ctl is not None else None
    m = F.mask
    land = st.land & np.isfinite(T)
    stn = [(kd, i, j, float(v)) for kd, i, j, v in withheld_stations(F, k) if np.isfinite(v)]
    vmax = float(np.nanpercentile(np.r_[T[land], Mem[:3][:, land].ravel()], 99.5))
    errs = {"base network": B - T, "ensemble mean": Em - T, "member 1": Mem[0] - T}
    if Ct is not None:
        errs = {"base network": B - T, "MSE corrector": Ct - T, "ensemble mean": Em - T, "member 1": Mem[0] - T}
    elim = float(np.nanpercentile(np.abs(np.concatenate([e[m] for e in errs.values()])), 99)) or 1.0
    rm = {lab: wim.rmse(e + T, T, m) for lab, e in errs.items()}

    fig, ax = plt.subplots(3, 4, figsize=(20, 14.2))
    kw = dict(vmin=0, vmax=vmax)
    wim.panel(fig, ax[0, 0], st, T, "AORC (target)", C["speed"], f"holdout mean {np.nanmean(T[m]):.2f} m/s", stn=stn, **kw)
    wim.panel(fig, ax[0, 1], st, B, "Base network (station fusion)", C["speed"],
              f"holdout RMSE {rm['base network']:.2f}", stn=stn, **kw)
    if Ct is not None:
        wim.panel(fig, ax[0, 2], st, Ct, "MSE corrector on 2022 (control)", C["speed"],
                  f"holdout RMSE {rm['MSE corrector']:.2f}", stn=stn, **kw)
    else:
        ax[0, 2].axis("off")
    wim.panel(fig, ax[0, 3], st, Em, "Ensemble mean (25 members)", C["speed"],
              f"holdout RMSE {rm['ensemble mean']:.2f}", stn=stn, **kw)
    for n in range(3):
        wim.panel(fig, ax[1, n], st, Mem[n], f"Diffusion member {n + 1}", C["speed"],
                  f"holdout RMSE {wim.rmse(Mem[n], T, m):.2f}", stn=stn, **kw)
    wim.panel(fig, ax[1, 3], st, Sd, "Ensemble spread (sd of members)", C["seq"],
              f"holdout mean {np.nanmean(Sd[m]):.2f} m/s; |error of mean| {np.nanmean(np.abs(Em - T)[m]):.2f}",
              vmin=0, vmax=float(np.nanpercentile(Sd[land], 99)))
    norm = TwoSlopeNorm(0, -elim, elim)
    for n, (lab, e) in enumerate(errs.items()):
        wim.panel(fig, ax[2, n], st, e, f"Error: {lab} - AORC", C["div"],
                  f"holdout RMSE {rm[lab]:.2f}, bias {wim.bias(e + T, T, m):+.2f}", norm=norm)
    for a in ax[2, len(errs):]:
        a.axis("off")
    fig.suptitle(f"Residual diffusion ({tag}), fold store w10o{F.f}, {day} ({label})", x=.01, y=.995, ha="left",
                 fontsize=15, fontweight="bold")
    fig.text(.01, .968, "Daily mean 10 m wind speed. Members are the base network plus a sampled residual learned on "
             "2022; their fine texture is realistic in amplitude but its placement is random, so the mean is smooth.\n"
             f"Circles: ISD stations withheld from the inputs (fold {1 - F.f}); squares: Mesonet (never an input). "
             "Orange box: spatial holdout, where all scores are computed.", fontsize=9, color=INK3, va="top")
    fig.tight_layout(rect=(0, 0, 1, .94))
    wim.save(fig, f"diffusion_{tag}_{day}.png")


def ranks(X, y):
    """Rank of y among the members X (m, N) -> 0..m, ties broken at random."""
    rng = np.random.default_rng(0)
    lo = (X < y[None]).sum(0); eq = (X == y[None]).sum(0)
    return lo + rng.integers(0, eq + 1)


def sheet_testyear(F, tag, E, det):
    plt = wim.setup(); C = wim.cmaps()
    st = F.st; m = F.mask; Ta = F.Ta
    Em = np.asarray(E.mean(0))
    Sd = np.asarray(E.std(0, ddof=1))
    r_det = np.sqrt(np.nanmean((det - Ta) ** 2, 0))
    r_em = np.sqrt(np.nanmean((Em - Ta) ** 2, 0))
    sp = np.sqrt(np.nanmean(Sd ** 2, 0))
    ratio = sp / r_em
    land = st.land & np.isfinite(Ta).all(0)
    nm = E.shape[0]

    fig, ax = plt.subplots(2, 4, figsize=(20, 9.6))
    vmax = float(np.nanpercentile(np.r_[r_det[land], r_em[land]], 99))
    wim.panel(fig, ax[0, 0], st, r_det, "Base network RMSE, 2023", C["seq"], f"holdout {np.sqrt(np.nanmean((det - Ta)[:, m] ** 2)):.3f}",
              vmin=0, vmax=vmax)
    wim.panel(fig, ax[0, 1], st, r_em, "Ensemble-mean RMSE, 2023", C["seq"], f"holdout {np.sqrt(np.nanmean((Em - Ta)[:, m] ** 2)):.3f}",
              vmin=0, vmax=vmax)
    wim.panel(fig, ax[0, 2], st, sp, "Ensemble spread, 2023 (rms sd)", C["seq"], f"holdout {np.sqrt(np.nanmean(sp[m] ** 2)):.3f}",
              vmin=0, vmax=vmax)
    from matplotlib.colors import TwoSlopeNorm
    wim.panel(fig, ax[0, 3], st, ratio, "Spread / error of the mean", C["div"], "1 = calibrated; < 1 = too narrow",
              unit="", norm=TwoSlopeNorm(1, 0, 2))

    # rank histograms: holdout grid (subsample), withheld ISD, Mesonet
    ii, jj = np.where(m)
    sel = np.random.default_rng(0).choice(len(ii), min(400, len(ii)), replace=False)
    Xg = np.asarray(E[:, :, ii[sel], jj[sel]]).reshape(nm, -1); yg = Ta[:, ii[sel], jj[sel]].ravel()
    ok = np.isfinite(yg)
    sets = {"holdout grid vs AORC": ranks(Xg[:, ok], yg[ok])}
    for kind, rows in (("withheld ISD", F.eval_isd), ("Mesonet", F.eval_meso)):
        rk = []
        for _, i, j, o in rows:
            okk = np.isfinite(o)
            if okk.sum() < 100:
                continue
            X = np.asarray(E[:, okk, i, j]) + F.clim[okk, i, j][None]
            rk.append(ranks(X, o[okk]))
        if rk:
            sets[kind] = np.concatenate(rk)
    for n, (lab, rk) in enumerate(sets.items()):
        a = ax[1, n]
        h = np.bincount(rk, minlength=nm + 1) / len(rk) * (nm + 1)
        a.bar(np.arange(nm + 1), h, color="#3b6ea8", width=.9)
        a.axhline(1, color=INK3, lw=1, ls="--")
        out = h[0] / (nm + 1) + h[-1] / (nm + 1)
        a.set_title(f"Rank histogram, {lab}", fontsize=10.5, fontweight="bold", pad=17)
        a.text(.5, 1.012, f"flat = calibrated; U = too narrow. Outside all members: {out:.0%} (ideal {2 / (nm + 1):.0%})",
               transform=a.transAxes, ha="center", va="bottom", fontsize=7, color=INK3)
        a.set_xlabel("rank of observation among members"); a.set_ylabel("relative frequency")
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
    # binned spread vs error (holdout grid sample)
    a = ax[1, 3]
    s = np.asarray(Xg[:, ok].std(0, ddof=1)); e = np.abs(Xg[:, ok].mean(0) - yg[ok])
    q = np.quantile(s, np.linspace(0, 1, 11))
    b = np.clip(np.digitize(s, q[1:-1]), 0, 9)
    xs = [s[b == k].mean() for k in range(10)]; ys = [np.sqrt(np.mean(e[b == k] ** 2)) for k in range(10)]
    lim = max(max(xs), max(ys)) * 1.1
    a.plot([0, lim], [0, lim], color=INK3, ls="--", lw=1, label="calibrated")
    a.plot(xs, ys, "o-", color="#c4572a", label="deciles of spread")
    a.set_xlim(0, lim); a.set_ylim(0, lim)
    a.set_xlabel("ensemble spread (m/s)"); a.set_ylabel("RMSE of ensemble mean (m/s)")
    a.set_title("Spread vs error, holdout grid", fontsize=10.5, fontweight="bold", pad=17)
    a.text(.5, 1.012, "points above the line: too narrow; a rising line means spread carries information",
           transform=a.transAxes, ha="center", va="bottom", fontsize=7, color=INK3)
    a.legend(fontsize=8, frameon=False)
    for sp_ in ("top", "right"):
        a.spines[sp_].set_visible(False)
    fig.suptitle(f"Residual diffusion ({tag}), test year 2023, fold store w10o{F.f}", x=.01, ha="left",
                 fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, .955))
    wim.save(fig, f"diffusion_{tag}_testyear.png")


def main(a):
    meta = json.loads((REPO / "outputs" / "wind" / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(REPO / "outputs" / "wind" / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    F = woe.Fold(a.fold, meta, isd)
    E, det = load(a.tag, a.fold)
    ctl = None
    if a.control and (DIF / f"{a.control}_w10o{a.fold}_test.npy").exists():
        ctl = np.asarray(np.load(DIF / f"{a.control}_w10o{a.fold}_test.npy", mmap_mode="r")[0])
    if a.days:
        days = {dd: "chosen" for dd in a.days}
    else:
        days = {dd: lab for lab, dd in wim.pick_days(wim.store_for("w10")).items()}
    for day, lab in days.items():
        sheet_day(F, a.tag, a.control, E, det, ctl, day, lab)
    sheet_testyear(F, a.tag, E, det)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="obsval")
    ap.add_argument("--control", default="obsvaldet")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--days", nargs="+", default=None)
    main(ap.parse_args())
