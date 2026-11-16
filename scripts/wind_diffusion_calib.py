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

--mode band3 (tag <tag>cal3): the two-band version leaves the 100-300 km band 1.22x too strong in 2023, since a_lo
also widens it. Three bands (> 300, 100-300, < 100 km): a_mid and a_hi set each band's member sd to AORC's on 2022,
a_lo by CRPS on 2022.

Usage: python scripts/wind_diffusion_calib.py [--cv obsvalcv] [--tag obsval] [--mode single|scale|band3]
    -> outputs/wind/diffusion/<tag>_calib[_scale|_band3].json, <tag>cal[2|3]_w10o{0,1}_{test,det}.npy
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


def split3(E, s100, s300):
    """Member deviations from the ensemble mean in three bands: (mu, > 300 km, 100-300 km, < 100 km)."""
    from scipy.ndimage import gaussian_filter
    E = np.asarray(E, "float32")
    mu = E.mean(0, keepdims=True)
    D = E - mu
    g100 = gaussian_filter(D, sigma=(0, 0, s100, s100), mode="nearest")
    g300 = gaussian_filter(D, sigma=(0, 0, s300, s300), mode="nearest")
    return mu, g300, g100 - g300, D - g100


def main_band3(a):
    """Three factors: a_hi (< 100 km) and a_mid (100-300 km) make each band's member sd match AORC's on the
    out-of-fold 2022 holdout (with the same band filter as the score, wind_obs_eval.bands); a_lo (> 300 km) by CRPS.
    The band sd is quadratic in the factors (bands are linear), so it is solved from precomputed components; a_lo
    and the two amplitude factors are alternated until they settle."""
    import wind_obs_eval as woe
    st = wim.store_for("w10o0")
    s100, s300 = st.sigma_cells(100), st.sigma_cells(300)
    labs = ("gt300", "100_300", "lt100")
    comp = {b: [] for b in labs}          # per band: list over halves of (mu, lo, mid, hi, truth) band values at mask
    crps_parts = []
    rng = np.random.default_rng(0)
    for h in ("a", "b"):
        E = np.load(DIF / f"{a.cv}{h}_w10o0_val.npy")
        days = np.load(DIF / f"{a.cv}{h}_w10o0_val_days.npy")
        T = np.asarray(st._zarr["target"][days[0]:days[-1] + 1, 0], "float32")[days - days[0]]
        land = st.land & np.isfinite(T).all(0)
        m = st.m_hold & land
        mu, lo, mid, hi = split3(E, s100, s300)
        del E
        k = rng.choice(lo.shape[0], 6, replace=False)                 # members used for the band sd fit
        f = lambda X: np.where(land, np.nan_to_num(X), 0.0)           # noqa: E731
        for lo_k, hi_k, lab in ((300, None, "gt300"), (100, 300, "100_300"), (None, 100, "lt100")):
            def bp(X):
                X = f(X).reshape(-1, *X.shape[-2:])
                hh = woe.highpass(X, st.sigma_cells(hi_k)) if hi_k else X
                hh = hh if lo_k is None else hh - woe.highpass(X, st.sigma_cells(lo_k))
                return hh[:, m].ravel()
            mu_b = np.tile(bp(mu[0]), len(k))                         # same order as the member-major bp(lo[k])
            comp[lab].append((mu_b, bp(lo[k]), bp(mid[k]), bp(hi[k]), bp(T)))
        ii, jj = np.where(m)
        sub = rng.choice(len(ii) * len(days), min(100_000, len(ii) * len(days)), replace=False)
        g = lambda A: A[..., ii, jj].reshape(A.shape[0], -1)[:, sub]  # noqa: E731
        crps_parts.append((g(mu)[0], g(lo), g(mid), g(hi), T[:, ii, jj].ravel()[sub]))
        del mu, lo, mid, hi
    C = {b: [np.concatenate([p[i] for p in comp[b]]) for i in range(5)] for b in labs}
    mu = np.concatenate([p[0] for p in crps_parts]); y = np.concatenate([p[4] for p in crps_parts])
    LO, MID, HI = (np.concatenate([p[i] for p in crps_parts], 1) for i in (1, 2, 3))

    def sd_ratio(b, f_lo, f_mid, f_hi):
        mu_b, l_b, md_b, h_b, t_b = C[b]
        return float((mu_b + f_lo * l_b + f_mid * md_b + f_hi * h_b).std() / t_b.std())

    def solve(b, which, f):
        """Factor for component `which` (1 mid, 2 hi) giving sd ratio 1 in band b, others fixed (bisection)."""
        lo_, hi_ = 0.0, 8.0
        for _ in range(40):
            x = (lo_ + hi_) / 2
            ff = list(f); ff[which] = x
            lo_, hi_ = (x, hi_) if sd_ratio(b, *ff) < 1 else (lo_, x)
        return round((lo_ + hi_) / 2, 2)

    f = [2.4, 1.0, 1.17]                                              # start from the two-band answer
    grid = np.round(np.arange(1.0, 6.01, 0.1), 2)
    for it in range(4):
        f[2] = solve("lt100", 2, f)
        f[1] = solve("100_300", 1, f)
        crps = {float(x): crps_ens(mu[None] + x * LO + f[1] * MID + f[2] * HI, y) for x in grid}
        f[0] = min(crps, key=crps.get)
        print(f"iter {it}: a_lo {f[0]}, a_mid {f[1]}, a_hi {f[2]}, CRPS {crps[f[0]]:.4f}", flush=True)
    X = mu[None] + f[0] * LO + f[1] * MID + f[2] * HI
    q10, q90 = np.percentile(X, [10, 90], axis=0)
    res = dict(cv=a.cv, a_lo=f[0], a_mid=f[1], a_hi=f[2], crps_best=crps[f[0]],
               crps_two_band=crps_ens(mu[None] + 2.4 * (LO + MID) + 1.17 * HI, y),
               cover80_best=float(np.mean((y >= q10) & (y <= q90))),
               spread_skill=float(np.sqrt(np.mean(X.var(0, ddof=1))) / np.sqrt(np.mean((X.mean(0) - y) ** 2))),
               band_sd_ratio_2022={b: sd_ratio(b, *f) for b in labs},
               band_sd_ratio_2022_raw={b: sd_ratio(b, 1, 1, 1) for b in labs})
    print(json.dumps(res, indent=1), flush=True)
    (DIF / f"{a.tag}_calib_band3.json").write_text(json.dumps(dict(res, crps_curve=crps), indent=1))
    del LO, MID, HI, X
    for fo in (0, 1):
        mu_, lo_, mid_, hi_ = split3(np.load(DIF / f"{a.tag}_w10o{fo}_test.npy"), s100, s300)
        np.save(DIF / f"{a.tag}cal3_w10o{fo}_test.npy", (mu_ + f[0] * lo_ + f[1] * mid_ + f[2] * hi_).astype("float32"))
        np.save(DIF / f"{a.tag}cal3_w10o{fo}_det.npy", np.load(DIF / f"{a.tag}_w10o{fo}_det.npy"))
        del mu_, lo_, mid_, hi_
    print(f"wrote {a.tag}cal3 {f}; scoring", flush=True)
    subprocess.run([sys.executable, "-u", str(REPO / "scripts" / "wind_diffusion.py"), "--score-only",
                    "--tag", f"{a.tag}cal3"], check=True)


def main(a):
    if a.mode == "scale":
        return main_scale(a)
    if a.mode == "band3":
        return main_band3(a)
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
    ap.add_argument("--mode", default="single", choices=["single", "scale", "band3"])
    main(ap.parse_args())
