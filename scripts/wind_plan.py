"""Single source of truth for the wind program (WIND_PLAN.md): every training run, which store and config it uses,
which channels it sees, and which evaluation group it belongs to.

The slurm trainer (slurm/vista_wind_train.sh) and the evaluation suite (scripts/wind_suite.py) both read this
file, so a run cannot be trained under one definition and scored under another.

Usage:
    python scripts/wind_plan.py --groups ablation arch           # one line per run, for the trainer
    python scripts/wind_plan.py --summary                         # counts and estimated GPU hours
"""
from __future__ import annotations
import argparse

# ----------------------------------------------------------------------------------------------- stores and configs
STORES = {
    # tag: (config, zarr, stored channel count, km)
    "w10": ("configs/data_sc_wind_dyn.yaml", "data_store_sc/wind_dyn/dataset.zarr", 27, 10),
    "w10e": ("configs/data_sc_wind_era5.yaml", "data_store_sc/wind_era5/dataset.zarr", 27, 10),
    "w25": ("configs/data_sc_wind25.yaml", "data_store_sc/wind25/dataset.zarr", 27, 25),
    "w5": ("configs/data_sc_wind5.yaml", "data_store_sc/wind5/dataset.zarr", 27, 5),
    "w1": ("configs/data_sc_wind1.yaml", "/scratch/11755/gurup12/wind_km1/dataset.zarr", 15, 1),
}
# station fusion (scripts/wind_obs_store.py): the w10 store + 4 ISD channels; fold f = which holdout stations are inputs
for f in (0, 1):
    STORES[f"w10o{f}"] = (f"configs/data_sc_wind_obs_f{f}.yaml", f"data_store_sc/wind_obs_f{f}/dataset.zarr", 31, 10)
for reg in ("west", "coast", "north"):
    STORES[f"w10{reg}"] = (f"configs/data_sc_wind_dyn_{reg}.yaml", STORES["w10"][1], 27, 10)
    STORES[f"w5{reg}"] = (f"configs/data_sc_wind5_{reg}.yaml", STORES["w5"][1], 27, 5)

# ------------------------------------------------------------------------------------------------------ channel sets
POWER_ONLY = ["coarse_tmp", "land_mask", "doy_sin", "doy_cos"]
BASE = POWER_ONLY + ["dem", "landcover", "coastal_dist", "urban_frac"]           # the temperature "v0 base" set
TERRAIN = ["elev_std", "slope_mean", "tpi", "northness", "eastness"]
LANDCOVER = ["lc_water", "lc_developed", "lc_barren", "lc_forest", "lc_shrub", "lc_herbaceous", "lc_cultivated",
             "lc_wetland"]
STATIC = BASE + TERRAIN + LANDCOVER                                                # = the 21-channel wind store
DIRECTION = ["coarse_u10m", "coarse_v10m"]
SHEAR = ["coarse_ws50m", "coarse_wind"]
GUST_STAB = ["coarse_ws10m_max", "coarse_t2m_range"]
FULL = STATIC + DIRECTION + SHEAR + GUST_STAB
OBS = ["obs_dep25", "obs_dep75", "obs_w25", "obs_w75"]
LADDER = ["coarse_tmp", "coarse_u10m", "coarse_v10m", "coarse_ws50m", "land_mask", "doy_sin", "doy_cos", "dem",
          "elev_std", "tpi", "coastal_dist", "urban_frac", "lc_forest", "lc_water", "lc_developed"]

CHANNELS = {
    "A0": ("POWER speed only", POWER_ONLY),
    "A1": ("+ base statics", BASE),
    "A2": ("+ base + terrain", BASE + TERRAIN),
    "A3": ("+ base + land cover (roughness)", BASE + LANDCOVER),
    "A4": ("all statics", STATIC),
    "A5": ("all statics + direction (u, v)", STATIC + DIRECTION),
    "A6": ("all statics + shear (50 m, 2 m)", STATIC + SHEAR),
    "A7": ("all statics + gust and stability", STATIC + GUST_STAB),
    "A8": ("full: statics + all same-day POWER", FULL),
    "L": ("ladder set (15 channels, stored at every grid)", LADDER),
    "LO": ("ladder set + same-day ISD stations (19 channels)", LADDER + OBS),
    "A8O": ("full + same-day ISD stations (31 channels)", FULL + OBS),
}

SEEDS3 = (1337, 7, 42)
SEEDS5 = (1337, 7, 42, 2024, 31337)
# measured on Vista GH200 for the temperature program; transformers are an estimate
MINUTES = {("deepsd", 10): 12, ("deepsd", 25): 4, ("deepsd", 5): 15, ("deepsd", 1): 200, ("linear", 10): 6,
           ("unet", 10): 20, ("edsr", 10): 20, ("convnext", 10): 25, ("swin", 10): 35, ("swinir", 10): 40,
           ("restormer", 10): 45}


def run(group, store, arch, chan, seed, name=None):
    cfg, zarr, nch, km = STORES[store]
    nm = name or f"{store}_{chan if arch == 'deepsd' else arch + '_' + chan}_s{seed}"
    return dict(name=nm, group=group, store=store, cfg=cfg, zarr=zarr, nch=nch, km=km, arch=arch, chan=chan,
                seed=seed, channels=CHANNELS[chan][1], minutes=MINUTES.get((arch, km), 30))


def plan():
    R = []
    # E1 predictor ablation: DeepSD, AORC 10 km, 3 seeds per channel set
    for c in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"):
        for s in SEEDS3:
            R.append(run("ablation", "w10", "deepsd", c, s))
    # E2 architecture sweep on the full channel set (DeepSD A8 comes from E1)
    for a in ("linear", "unet", "edsr", "convnext", "swin", "swinir", "restormer"):
        for s in SEEDS3:
            R.append(run("arch", "w10", a, "A8", s))
    # E3 target swap: ERA5-Land-trained vs AORC-trained, 5 seeds each (AORC seeds 1-3 come from E1)
    for s in SEEDS5:
        R.append(run("target", "w10e", "deepsd", "A8", s))
    for s in SEEDS5[3:]:
        R.append(run("target", "w10", "deepsd", "A8", s))
    # E4 resolution ladder, one channel set at every grid: 5 seeds at 25/10/5 km, 3 at 1 km
    for st in ("w25", "w10", "w5"):
        for s in SEEDS5:
            R.append(run("ladder", st, "deepsd", "L", s))
    for s in SEEDS3:
        R.append(run("ladder1", "w1", "deepsd", "L", s))
    # E5 leave-one-region-out at 10 and 5 km, ladder channels, 2 seeds
    for reg in ("west", "coast", "north"):
        for g in ("w10", "w5"):
            for s in SEEDS3[:2]:
                R.append(run("loro", f"{g}{reg}", "deepsd", "L", s))
    # E7 station fusion (WIND_WORK_LOG 7c): trained on fold-0 store, scored on both folds by scripts/wind_obs_eval.py
    for s in SEEDS3:
        R.append(run("obs", "w10o0", "deepsd", "LO", s))
    R.append(run("obs", "w10o0", "deepsd", "A8O", 1337))
    R.append(run("obs", "w10o0", "unet", "LO", 1337))
    return R


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    R = [r for r in plan() if a.groups is None or r["group"] in a.groups]
    if a.summary:
        from collections import Counter
        c = Counter(r["group"] for r in R)
        for g in c:
            h = sum(r["minutes"] for r in R if r["group"] == g) / 60
            print(f"{g:10s} {c[g]:3d} runs  ~{h:5.1f} GPU-h")
        print(f"{'total':10s} {len(R):3d} runs  ~{sum(r['minutes'] for r in R) / 60:5.1f} GPU-h")
    else:
        for r in R:
            print(r["name"], r["cfg"], r["zarr"], r["nch"], r["arch"], r["seed"], ",".join(r["channels"]))
