"""Score the station-fusion wind runs (plan groups "obs" and "obs4") against the no-station models, test year 2023.

Each fusion run was trained on the fold-0 store and is scored on BOTH fold stores: in store f the holdout stations
of fold f are inputs, so stations are scored only at the other fold (never inputs). Oklahoma Mesonet is never an
input anywhere.

Per station (daily means, 2023):  rmse, bias, debiased rmse (mean error removed), r_dep = correlation of the daily
departure from POWER (prediction - POWER vs obs - POWER, each de-meaned): the day-to-day local skill POWER lacks.
Grid, holdout land cells vs AORC: rmse, bias, ssim, kl (wind_suite.grid_scores) and, per scale band
(< 100, 100-300, > 300 km), r2 and sd ratio of the daily anomaly.

References at the same stations and on the same grid: POWER interpolated, AORC itself, the no-station models
(w10_L seeds, w10_A8_s1337), and plain kernel interpolation of the station departures (POWER + obs_dep25 / obs_dep75
channel), the non-ML baseline the network has to beat.

Usage: DOWNSCALE_DEVICE=cuda python scripts/wind_obs_eval.py     -> outputs/wind/obs_eval.json, obs_eval.md
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
os.environ.setdefault("DOWNSCALE_MAX_WORKERS", "4")
from evaluation.fields import highpass                                           # noqa: E402
import wind_inference_maps as wim                                                 # noqa: E402
import wind_plan                                                                  # noqa: E402
import wind_suite                                                                 # noqa: E402

OUT = REPO / "outputs" / "wind"
BASELINES = ["w10_L_s1337", "w10_L_s7", "w10_L_s42", "w10_A8_s1337"]
BANDS = ((None, 100, "lt100"), (100, 300, "100_300"), (300, None, "gt300"))


def log(*a):
    print(*a, flush=True)


def stn_metrics(pred, obs, power):
    ok = np.isfinite(pred) & np.isfinite(obs) & np.isfinite(power)
    if ok.sum() < 100:
        return None
    p, o, w = pred[ok], obs[ok], power[ok]
    e = p - o
    dp, do = (p - w) - (p - w).mean(), (o - w) - (o - w).mean()
    r = float(np.corrcoef(dp, do)[0, 1]) if dp.std() > 1e-6 else 0.0
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), bias=float(e.mean()), rmse_db=float(e.std()), r_dep=r,
                n=int(ok.sum()))


def summarise(per):
    per = [x for x in per if x]
    if not per:
        return None
    out = {k: float(np.mean([x[k] for x in per])) for k in ("rmse", "bias", "rmse_db", "r_dep")}
    out["r_dep_median"] = float(np.median([x["r_dep"] for x in per])); out["n_stations"] = len(per)
    return out


def bands(P, T, land, mask, st):
    f = lambda X: np.where(land, np.nan_to_num(X), 0.0)                            # noqa: E731
    out = {}
    for lo, hi, lab in BANDS:
        def bp(X):
            X = f(X)
            h = highpass(X, st.sigma_cells(hi)) if hi else X
            return h if lo is None else h - highpass(X, st.sigma_cells(lo))
        p, t = bp(P)[:, mask].ravel(), bp(T)[:, mask].ravel()
        out[lab] = dict(r2=float(np.corrcoef(p, t)[0, 1] ** 2), sd_ratio=float(p.std() / t.std()))
    return out


class Fold:
    def __init__(self, f, stations, isd):
        self.f = f
        self.st = st = wim.store_for(f"w10o{f}")
        self.te = st.i_test
        z = st._zarr
        te = self.te
        self.Ta = np.asarray(z["target"][te[0]:te[-1] + 1, 0], "float32")                 # anomaly
        self.land = st.land & np.isfinite(self.Ta).all(0)
        self.mask = st.m_hold & self.land
        clim = st.clim[st.doy[te]]
        self.Tabs = self.Ta + clim
        ic = st.stored.index("coarse_tmp")
        self.Pan = np.asarray(z["input"][te[0]:te[-1] + 1, ic], "float32")
        self.Pabs = self.Pan + st.clim_coarse[st.doy[te]]
        self.clim = clim
        self.dep = {nm: np.asarray(z["input"][te[0]:te[-1] + 1, st.stored.index(nm)], "float32")
                    for nm in ("obs_dep25", "obs_dep75")}
        crop = self.Tabs[:, self.mask]
        self.L = float(np.percentile(crop, 99.9) - crop.min())
        times = st.times[te]
        ev = [s for s in stations if s["in_holdout"] and s["fold"] == 1 - f]
        self.eval_isd = [(s["key"], s["row"], s["col"],
                          isd.get(s["key"], pd.Series(dtype=float)).reindex(times).values) for s in ev]
        meso = wind_suite.mesonet_cells(st) or []
        self.eval_meso = [(m["key"], m["row"], m["col"], m["obs"].reindex(times).values) for m in meso]
        log(f"fold {f}: {len(self.eval_isd)} withheld ISD stations, {len(self.eval_meso)} Mesonet, "
            f"{int(self.mask.sum())} holdout cells")

    def score(self, Mabs, Man):
        st = self.st
        res = {"grid": wind_suite.grid_scores(Mabs, self.Tabs, self.mask, self.L),
               "bands": bands(Man, self.Ta, self.land, self.mask, st)}
        for net, rows in (("isd", self.eval_isd), ("meso", self.eval_meso)):
            per = [stn_metrics(Mabs[:, i, j], o, self.Pabs[:, i, j]) for _, i, j, o in rows]
            res[net] = summarise(per)
            res[f"{net}_per_station"] = {k: m for (k, _, _, _), m in zip(rows, per) if m}
        return res


def mean_folds(a, b):
    """Average two nested score dicts (fold 0 and fold 1)."""
    if isinstance(a, dict):
        return {k: mean_folds(a[k], b[k]) for k in a if k in b and not k.endswith("_per_station")}
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (a + b) / 2
    return a


def main():
    stations = json.loads((OUT / "obs_stations.json").read_text())["stations"]
    d = pd.read_csv(OUT / "isd_domain_daily.csv", parse_dates=["date"])
    isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
    folds = [Fold(f, stations, isd) for f in (0, 1)]
    res = {"per_fold": {0: {}, 1: {}}}
    # references
    for F in folds:
        r = res["per_fold"][F.f]
        r["POWER"] = F.score(F.Pabs, F.Pan)
        r["AORC"] = F.score(F.Tabs, F.Ta)
        for nm in ("obs_dep25", "obs_dep75"):
            r[f"kernel_{nm[4:]}"] = F.score(F.Pabs + F.dep[nm], F.Pan + F.dep[nm])
    # no-station models: predicted on the plain w10 store (same days, same grid)
    base_st = wim.store_for("w10")
    for run in BASELINES:
        if not (REPO / "checkpoints" / run / "last.pt").exists():
            continue
        A = wim.predict(base_st, run, folds[0].te)
        for F in folds:
            res["per_fold"][F.f][run] = F.score(A + F.clim, A)
        log(f"  {run} scored")
    seedsum = {}                                                  # seed means of the DeepSD LO fusion runs
    for r in wind_plan.plan():
        if r["group"] not in ("obs", "obs4") or not (REPO / "checkpoints" / r["name"] / "last.pt").exists():
            continue
        key = f"w10o0_LO{'4y' if r['group'] == 'obs4' else ''}_mean" if r["chan"] == "LO" and r["arch"] == "deepsd" \
            else None
        for F in folds:
            A = wim.predict(F.st, r["name"], F.te)
            res["per_fold"][F.f][r["name"]] = F.score(A + F.clim, A)
            if key:
                n, S = seedsum.get((key, F.f), (0, 0))
                seedsum[(key, F.f)] = (n + 1, S + A)
        log(f"  {r['name']} scored")
    for (key, f), (n, S) in seedsum.items():
        A = S / n
        res["per_fold"][f][f"{key}{n}"] = folds[f].score(A + folds[f].clim, A)
    names = list(res["per_fold"][0])
    res["mean"] = {n: mean_folds(res["per_fold"][0][n], res["per_fold"][1][n]) for n in names}
    (OUT / "obs_eval.json").write_text(json.dumps(res, indent=1, default=float))

    lines = ["| Variant | grid RMSE | grid bias | SSIM | r2 <100 km | sd <100 | r2 100-300 | ISD RMSE | ISD de-biased | "
             "ISD r_dep | Meso RMSE | Meso de-biased | Meso r_dep |", "|" + "---|" * 13]
    for n in names:
        m = res["mean"][n]; g = m["grid"]; b = m["bands"]; i = m["isd"] or {}; s = m["meso"] or {}
        fmt = lambda x: "" if x is None else f"{x:.3f}"                              # noqa: E731
        lines.append(f"| {n} | {g['rmse']:.3f} | {g['bias']:+.3f} | {g['ssim']:.3f} | {b['lt100']['r2']:.2f} | "
                     f"{b['lt100']['sd_ratio']:.2f} | {b['100_300']['r2']:.2f} | {fmt(i.get('rmse'))} | "
                     f"{fmt(i.get('rmse_db'))} | {fmt(i.get('r_dep'))} | {fmt(s.get('rmse'))} | "
                     f"{fmt(s.get('rmse_db'))} | {fmt(s.get('r_dep'))} |")
    txt = ("Station fusion, test year 2023, mean of the two folds (ISD scored only at the fold that was not an "
           "input; Mesonet never an input).\n\n" + "\n".join(lines) + "\n")
    (OUT / "obs_eval.md").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
