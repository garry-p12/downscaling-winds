"""How much of the obsval diffusion ensemble-mean gain is a fixed offset and how much varies by day?

Compares, on both fold stores (2023, holdout): det; det + one constant (the mean ensemble shift); det + the per-cell
time-mean shift; det + only the day-varying part; and the oracle det with its own bias removed. ISD (withheld fold)
and Mesonet MAE for det, det + constant and the ensemble mean.

Usage (repo root): python scripts/wind_diffusion_decomp.py [tag ...]   (default obsval; e.g. obsval obsvaldet)
"""
import json, sys, numpy as np, pandas as pd
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import wind_obs_eval as woe
OUT = "outputs/wind/diffusion"
meta = json.load(open("outputs/wind/obs_stations.json"))["stations"]
d = pd.read_csv("outputs/wind/isd_domain_daily.csv", parse_dates=["date"])
isd = {k: g.set_index("date")["ws"] for k, g in d.groupby("key")}
TAGS = sys.argv[1:] or ["obsval"]
for tag, f in [(t, f) for t in TAGS for f in (0, 1)]:
    F = woe.Fold(f, meta, isd); m = F.mask; Ta = F.Ta
    E = np.load(f"{OUT}/{tag}_w10o{f}_test.npy", mmap_mode="r"); det = np.load(f"{OUT}/{tag}_w10o{f}_det.npy")
    em = np.asarray(E.mean(0)); shift = em - det
    static = shift.mean(0, keepdims=True)                  # per-cell time-mean shift
    const = np.nanmean(shift[:, m])                        # one number
    r = lambda X: float(np.sqrt(np.nanmean((X - Ta)[:, m] ** 2)))
    b = lambda X: float(np.nanmean((X - Ta)[:, m]))
    print(f"{tag} fold {f}: det {r(det):.3f} (bias {b(det):+.3f}) | ens mean {r(em):.3f} ({b(em):+.3f}) | "
          f"det+const {const:+.3f} -> {r(det + const):.3f} | det+static cell shift {r(det + static):.3f} | "
          f"det+daily shift minus static {r(det + shift - static):.3f} | oracle det-bias {r(det - b(det)):.3f}")
    # station side: ISD and Meso MAE of det, det+const, ens mean
    for net, rows in (("isd", F.eval_isd), ("meso", F.eval_meso)):
        out = {}
        for lab, X in (("det", det), ("det+const", det + const), ("ensmean", em)):
            e = []
            for _, i, j, o in rows:
                ok = np.isfinite(o)
                if ok.sum() >= 100:
                    e.append(np.mean(np.abs(X[ok, i, j] + F.clim[ok, i, j] - o[ok])))
            out[lab] = np.mean(e)
        print("   ", net, " ".join(f"{k} MAE {v:.3f}" for k, v in out.items()))
