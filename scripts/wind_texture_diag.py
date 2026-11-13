"""How much of AORC's day-to-day fine-scale wind is predictable, and from what?

The deterministic model's daily anomaly is too smooth (slope 0.26 correction-made vs needed on a windy day). Before
changing losses or architectures, measure the ceiling. All on the 10 km store, AORC daily-mean speed ANOMALY (the
store's day-of-year climatology removed), high-passed at --cutoff km (Gaussian, evaluation.fields.highpass).

  truth        fine-scale sd of AORC anomaly over the holdout, test year
  model        fine-scale sd of the model's anomaly, its correlation with truth, regression slope
  ridge_flow   SPATIALLY HONEST: ridge fit in the training region (train days) on flow x terrain features built
               from same-day POWER u, v, speed and the statics; applied to the holdout, test days
  ridge+model  the same features plus the model's own fine-scale field (what the features add to the model)
  cond_clim    TEMPORAL ceiling: per cell, mean fine-scale anomaly in bins of POWER direction (8 sectors) x speed
               (3 terciles) over 2019-2021, looked up on 2023 days. Uses the holdout cells' own AORC history, as the
               store's climatology already does; a ceiling, not a spatially held-out result.
  unpredictable  1 - best r^2

Also: share of the ABSOLUTE field's fine-scale variance that the climatology already supplies.

Usage: python scripts/wind_texture_diag.py [--run w10_L_s1337] [--cutoff 100]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
from evaluation.fields import Store, highpass                                    # noqa: E402
import wind_inference_maps as wim                                                 # noqa: E402
import wind_plan                                                                  # noqa: E402


def stats(p, t):
    p, t = p.ravel(), t.ravel()
    ok = np.isfinite(p) & np.isfinite(t); p, t = p[ok], t[ok]
    r = float(np.corrcoef(p, t)[0, 1])
    return dict(sd_ratio=float(p.std() / t.std()), r=r, r2=r * r, slope=float(np.polyfit(t, p, 1)[0]),
                rmse=float(np.sqrt(np.mean((p - t) ** 2))), sd_truth=float(t.std()))


def main(run, cutoff, out):
    cfg, zarr = wind_plan.STORES["w10"][:2]
    st = Store(zarr, cfg_name=cfg); z = st._zarr
    sig = st.sigma_cells(cutoff)
    n = st.ds.sizes["time"]
    tr, te = st.i_train, st.i_test
    T = np.asarray(z["target"][:, 0], "float32")                       # anomaly, NaN off-land
    valid = np.isfinite(T)
    land = st.land & valid.all(0)
    fill = lambda X: np.where(land, X, 0.0)                              # noqa: E731
    Thp = highpass(fill(np.nan_to_num(T)), sig)
    A = wim.predict(st, run, np.arange(n))                               # model anomaly, all days (cached)
    Ahp = highpass(fill(A), sig)

    def dyn(name):
        v = np.asarray(z["input"][:, st.stored.index(name)], "float32")
        cl = "clim_coarse_tmp" if name == "coarse_tmp" else f"clim_{name}"
        return v + st.ds[cl].values[st.doy]
    S, U, V = dyn("coarse_tmp"), dyn("coarse_u10m"), dyn("coarse_v10m")
    Sa = np.asarray(z["input"][:, st.stored.index("coarse_tmp")], "float32")     # POWER speed anomaly
    stat = {nm: np.asarray(z["input"][0, st.stored.index(nm)], "float32") for nm in
            ("dem", "tpi", "elev_std", "slope_mean", "northness", "eastness", "lc_forest", "lc_water",
             "lc_developed", "lc_herbaceous", "lc_shrub", "lc_cultivated", "coastal_dist")}
    dem = np.where(land, stat["dem"], np.nanmean(stat["dem"][land]))
    gy, gx = np.gradient(dem, st.dx_km * 1000.0)                         # m/m; row 0 north -> gy is -d/dnorth
    if st.LAT[0, 0] > st.LAT[-1, 0]:
        gy = -gy                                                         # dh/dnorth
    dirn = (np.degrees(np.arctan2(-U, -V)) + 360) % 360                  # meteorological direction (from)

    mtr, mho = st.m_train & land, st.m_hold & land
    rng = np.random.default_rng(0)
    va = np.where(st.split == "val")[0]
    trd = np.sort(rng.choice(tr, size=min(400, len(tr)), replace=False))
    D = np.r_[trd, va, te]                                              # only these days get flow features
    pos = {d: i for i, d in enumerate(D)}

    # flow x terrain features, each high-passed like the target (the low-pass part is the offset problem)
    zs = {k: (v - v[land].mean()) / (v[land].std() + 1e-9) for k, v in stat.items()}
    SD, UD, VD, SaD = S[D], U[D], V[D], Sa[D]
    recipes = {"upslope": lambda: UD * gx[None] + VD * gy[None],        # w = u.grad h
               "cross": lambda: UD * gy[None] - VD * gx[None],
               "Sa": lambda: SaD}
    for k in ("tpi", "elev_std", "slope_mean", "lc_forest", "lc_water", "lc_developed", "lc_herbaceous",
              "lc_shrub", "lc_cultivated", "coastal_dist", "dem"):
        recipes[f"S*{k}"] = (lambda k=k: SD * zs[k][None])
        recipes[f"Sa*{k}"] = (lambda k=k: SaD * zs[k][None])
    for k in ("northness", "eastness"):                                  # aspect facing the wind
        recipes[f"U*{k}"] = (lambda k=k: UD * zs[k][None]); recipes[f"V*{k}"] = (lambda k=k: VD * zs[k][None])
    F = {k: highpass(fill(f()), sig).astype("float32") for k, f in recipes.items()}
    AhpD = Ahp[D]
    names = list(F)
    log = lambda *a: print(*a, flush=True)                               # noqa: E731
    log(f"{len(names)} flow x terrain features, cutoff {cutoff} km (sigma {sig:.2f} cells)")

    def design(days, mask, model=False):
        ix = [pos[d] for d in days]
        X = np.stack([F[k][ix][:, mask].ravel() for k in names], 1)
        if model:
            X = np.c_[X, AhpD[ix][:, mask].ravel()]
        return X

    def ridge(X, y, lam=1e-2):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        w = np.linalg.solve(Xs.T @ Xs + lam * len(y) * np.eye(Xs.shape[1]), Xs.T @ (y - y.mean()))
        return lambda X2: ((X2 - mu) / sd) @ w + y.mean()

    res = {"run": run, "cutoff_km": cutoff}
    t_te = Thp[te][:, mho]
    res["truth_hp_sd_holdout_test"] = float(t_te.std())
    res["model"] = stats(Ahp[te][:, mho], t_te)
    y = Thp[trd][:, mtr].ravel()
    f1 = ridge(design(trd, mtr), y)
    res["ridge_flow"] = stats(f1(design(te, mho)), t_te.ravel())
    f2 = ridge(design(trd, mtr, True), y)
    res["ridge_flow+model"] = stats(f2(design(te, mho, True)), t_te.ravel())
    # same ridge but trained on the holdout cells' own train-year days (is the spatial transfer the limit?)
    f3 = ridge(design(trd, mho, True), Thp[trd][:, mho].ravel())
    res["ridge_flow+model_fit_on_holdout_train_years"] = stats(f3(design(te, mho, True)), t_te.ravel())

    # conditional climatology: per cell, 8 direction sectors x 3 speed terciles, train years
    sec = (((dirn + 22.5) % 360) // 45).astype(int)
    q = np.quantile(S[tr][:, land], [1 / 3, 2 / 3])
    spd = np.digitize(S, q)
    b = sec * 3 + spd                                                     # (days, H, W) bin 0..23
    CC = np.zeros((24, *st.shape), "float32"); NN = np.zeros((24, *st.shape), "float32")
    for k in tr:
        bk = b[k]
        for j in range(24):
            m = bk == j
            CC[j][m] += Thp[k][m]; NN[j][m] += 1
    CC = np.where(NN >= 5, CC / np.maximum(NN, 1), 0.0)
    ii, jj = np.indices(st.shape)
    cc_te = np.stack([CC[b[k], ii, jj] for k in te])
    res["cond_clim"] = stats(cc_te[:, mho], t_te)
    X4 = np.c_[cc_te[:, mho].ravel(), Ahp[te][:, mho].ravel()]
    # combine on 2022 (validation) days to avoid fitting on test
    cc_va = np.stack([CC[b[k], ii, jj] for k in va])
    w = np.linalg.lstsq(np.c_[cc_va[:, mho].ravel(), Ahp[va][:, mho].ravel(), np.ones(cc_va[:, mho].size)],
                        Thp[va][:, mho].ravel(), rcond=None)[0]
    res["cond_clim+model"] = stats(X4 @ w[:2] + w[2], t_te.ravel()); res["cond_clim+model"]["weights"] = w.tolist()

    # share of the ABSOLUTE field's fine-scale variance (holdout, test) that the climatology already supplies
    clim = st.clim[st.doy[te]]
    Tabs_hp = highpass(fill(np.nan_to_num(T[te]) + clim), sig)[:, mho]
    clim_hp = highpass(fill(clim), sig)[:, mho]
    res["absolute_field"] = dict(sd_abs_hp=float(Tabs_hp.std()), sd_clim_hp=float(clim_hp.std()),
                                 r2_clim_vs_abs=float(np.corrcoef(clim_hp.ravel(), Tabs_hp.ravel())[0, 1] ** 2),
                                 sd_anom_hp=float(t_te.std()))
    best = max(v["r2"] for k, v in res.items() if isinstance(v, dict) and "r2" in v)
    res["unpredictable_fraction_at_best"] = 1 - best
    Path(out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(f"{k:45s} {v}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="w10_L_s1337")
    ap.add_argument("--cutoff", type=float, default=100.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    main(a.run, a.cutoff, a.out or str(REPO / "outputs" / "wind" / f"texture_diag_{a.run}_{int(a.cutoff)}km.json"))
