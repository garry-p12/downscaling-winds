"""Spread calibration for the residual-diffusion ensemble, fitted on 2022 only (WIND_WORK_LOG 7d).

The obsval ensemble (trained on all of 2022) is too narrow in 2023 (spread/skill ~0.59). To learn an inflation factor
without touching 2023, two cross-fit models are trained on alternating months of 2022 (tags <cv>a: odd months,
<cv>b: even months) and each samples the months it did not see (slurm/vista_wind_diffusion_cv.sh). On those
out-of-fold 2022 members, over the holdout land cells (out of sample in space and time), find the factor alpha that
minimises the fair CRPS when members are spread about their mean:  X' = mean + alpha * (X - mean).
Then write the inflated 2023 ensemble as tag <tag>cal and score it with wind_diffusion.py --score-only.

--mode scale (tag <tag>cal2): one alpha inflates the members' fine texture too (sd ratio < 100 km 0.84 -> 1.33), but
what is missing is large-scale uncertainty. So split each member's deviation from the mean with the same Gaussian as
the < 100 km band: X' = mean + a_hi * D_hi + a_lo * D_lo. a_hi is set so that the members' < 100 km sd matches AORC's
on the out-of-fold 2022 holdout; a_lo is then fitted by CRPS, also on 2022.

Usage: python scripts/wind_diffusion_calib.py [--cv obsvalcv] [--tag obsval] [--mode single|scale]
    -> outputs/wind/diffusion/<tag>_calib[_scale].json, <tag>cal[2]_w10o{0,1}_{test,det}.npy
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
import wind_inference_maps as wim                                                 # noqa: E402
from wind_diffusion import crps_ens                                               # noqa: E402

DIF = REPO / "outputs" / "wind" / "diffusion"


def oof(cv, store="w10o0"):
    """Out-of-fold 2022 members (m, N) and truth (N) at holdout land cells, both halves concatenated."""
    st = wim.store_for(store)
    Xs, ys = [], []
    for h in ("a", "b"):
        E = np.load(DIF / f"{cv}{h}_{store}_val.npy", mmap_mode="r")
        days = np.load(DIF / f"{cv}{h}_{store}_val_days.npy")
        T = np.asarray(st._zarr["target"][days[0]:days[-1] + 1, 0], "float32")[days - days[0]]
        m = st.m_hold & st.land & np.isfinite(T).all(0)
        ii, jj = np.where(m)
        Xs.append(np.asarray(E[:, :, ii, jj]).reshape(E.shape[0], -1))
        ys.append(T[:, ii, jj].ravel())
    return np.concatenate(Xs, 1), np.concatenate(ys)


def inflate(X, a):
    mu = X.mean(0, keepdims=True)
    return mu + a * (X - mu)


def split(E, sig):
    """Deviation of each member from the ensemble mean, split into (low, high) with a Gaussian of sig cells."""
    from scipy.ndimage import gaussian_filter
    E = np.asarray(E, "float32")
    mu = E.mean(0, keepdims=True)
    D = E - mu
    lo = gaussian_filter(D, sigma=(0, 0, sig, sig), mode="nearest")
    return mu, lo, D - lo


def main_scale(a):
    from scipy.ndimage import gaussian_filter
    st = wim.store_for("w10o0")
    sig = st.sigma_cells(100)
    parts, sd_m, sd_t = [], [], []
    for h in ("a", "b"):
        E = np.load(DIF / f"{a.cv}{h}_w10o0_val.npy")
        days = np.load(DIF / f"{a.cv}{h}_w10o0_val_days.npy")
        T = np.asarray(st._zarr["target"][days[0]:days[-1] + 1, 0], "float32")[days - days[0]]
        m = st.m_hold & st.land & np.isfinite(T).all(0)
        mu, lo, hi = split(E, sig)
        Tf = np.nan_to_num(T)
        t_hp = Tf - gaussian_filter(Tf, sigma=(0, sig, sig), mode="nearest")
        e_hp = E - gaussian_filter(E, sigma=(0, 0, sig, sig), mode="nearest")
        sd_m.append(e_hp[:, :, m].std()); sd_t.append(t_hp[:, m].std())
        ii, jj = np.where(m)
        sub = np.random.default_rng(0).choice(len(ii) * len(days), min(100_000, len(ii) * len(days)), replace=False)
        g = lambda A: A[..., ii, jj].reshape(*A.shape[:-2][:-1], -1)[..., sub]
        parts.append((g(mu)[0], g(lo), g(hi), T[:, ii, jj].ravel()[sub]))
        del E, mu, lo, hi, e_hp
    ratio = float(np.mean(sd_m) / np.mean(sd_t))
    a_hi = float(np.round(1 / ratio, 2))
    mu = np.concatenate([p[0] for p in parts]); lo = np.concatenate([p[1] for p in parts], 1)
    hi = np.concatenate([p[2] for p in parts], 1); y = np.concatenate([p[3] for p in parts])
    grid = np.round(np.arange(1.0, 6.01, 0.1), 2)
    crps = {float(g_): crps_ens(mu[None] + a_hi * hi + g_ * lo, y) for g_ in grid}
    a_lo = min(crps, key=crps.get)
    X = mu[None] + a_hi * hi + a_lo * lo
    q10, q90 = np.percentile(X, [10, 90], axis=0)
    res = dict(cv=a.cv, member_lt100_sd_ratio_2022=ratio, a_hi=a_hi, a_lo=a_lo, crps_best=crps[a_lo],
               crps_alo1=crps[1.0], cover80_best=float(np.mean((y >= q10) & (y <= q90))),
               spread_skill=float(np.sqrt(np.mean(X.var(0, ddof=1))) / np.sqrt(np.mean((X.mean(0) - y) ** 2))))
    print(json.dumps(res, indent=1), flush=True)
    (DIF / f"{a.tag}_calib_scale.json").write_text(json.dumps(dict(res, crps_curve=crps), indent=1))
    for f in (0, 1):
        mu, lo, hi = split(np.load(DIF / f"{a.tag}_w10o{f}_test.npy"), sig)
        np.save(DIF / f"{a.tag}cal2_w10o{f}_test.npy", (mu + a_hi * hi + a_lo * lo).astype("float32"))
        np.save(DIF / f"{a.tag}cal2_w10o{f}_det.npy", np.load(DIF / f"{a.tag}_w10o{f}_det.npy"))
        del mu, lo, hi
    print(f"wrote {a.tag}cal2 (a_hi {a_hi}, a_lo {a_lo}); scoring", flush=True)
    subprocess.run([sys.executable, "-u", str(REPO / "scripts" / "wind_diffusion.py"), "--score-only",
                    "--tag", f"{a.tag}cal2"], check=True)


def main(a):
    if a.mode == "scale":
        return main_scale(a)
    X, y = oof(a.cv)
    sub = np.random.default_rng(0).choice(X.shape[1], min(200_000, X.shape[1]), replace=False)
    X, y = X[:, sub], y[sub]
    mu = X.mean(0)
    spread = float(np.sqrt(np.mean(X.var(0, ddof=1))))
    skill = float(np.sqrt(np.mean((mu - y) ** 2)))
    grid = np.round(np.arange(0.8, 4.01, 0.05), 2)
    crps = {float(g): crps_ens(inflate(X, g), y) for g in grid}
    best = min(crps, key=crps.get)
    lo, hi = np.percentile(inflate(X, best), [10, 90], axis=0)
    res = dict(cv=a.cv, n=int(len(y)), spread=spread, skill=skill, spread_skill=spread / skill,
               alpha_ratio=skill / spread, alpha_crps=best, crps_raw=crps[1.0] if 1.0 in crps else None,
               crps_best=crps[best], cover80_best=float(np.mean((y >= lo) & (y <= hi))))
    print(json.dumps(res, indent=1))
    (DIF / f"{a.tag}_calib.json").write_text(json.dumps(dict(res, crps_curve=crps), indent=1))
    alpha = best
    for f in (0, 1):
        E = np.load(DIF / f"{a.tag}_w10o{f}_test.npy")
        np.save(DIF / f"{a.tag}cal_w10o{f}_test.npy", inflate(E, alpha).astype("float32"))
        np.save(DIF / f"{a.tag}cal_w10o{f}_det.npy", np.load(DIF / f"{a.tag}_w10o{f}_det.npy"))
        del E
    print(f"wrote {a.tag}cal (alpha {alpha}); scoring", flush=True)
    subprocess.run([sys.executable, "-u", str(REPO / "scripts" / "wind_diffusion.py"), "--score-only",
                    "--tag", f"{a.tag}cal"], check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="obsvalcv")
    ap.add_argument("--tag", default="obsval")
    ap.add_argument("--mode", default="single", choices=["single", "scale"])
    main(ap.parse_args())
