# Wind downscaling: work log and handoff

Status as of 14 November 2026 (fusion results, stacking and drift in section 7d). This file records everything done and submitted for the wind downscaling work, so it can be
moved into its own repository and continued from there. `WIND_PLAN.md` holds the experiment design; this file holds the
methods as implemented, every job submitted, what went wrong, what has been learned so far, and what remains.

---

## 1. Goal

Downscale NASA POWER daily 10 m wind speed (0.5° x 0.625°) to 10 km, and on a ladder down to 1 km, over the South-Central
US domain, with the same grid, holdout, years, models and evaluation as the temperature and radiation work, so wind can be
reported in Paper 1 ("POWER temperature, radiation and wind").

Questions (from `WIND_PLAN.md`):

| | Question |
|---|---|
| Q1 | Does the architecture matter? |
| Q2 | Which inputs carry the information (statics, direction, shear, gustiness, stability)? |
| Q3 | Does the training target (AORC vs ERA5-Land) set the ceiling? |
| Q4 | How much of the error is a uniform daily offset, and does the same-day block-mean correction remove it? |
| Q5 | How far down does it pay to go (25, 10, 5, 1 km)? |
| Q6 | Do the results hold in other regions (west, coast, north)? |
| Q7 | Why was the first model about 0.42 m/s too low in 2023? |
| Q8 | Do the station results hold on a second network (Oklahoma Mesonet, not airports)? |
| Q9 | Can a seed ensemble give calibrated intervals? |

---

## 2. Where everything lives

| What | Where |
|---|---|
| Code (local) | `~/downscaling-temperature` (Mac), the temperature repo `garry-p12/downscaling-temperature` |
| Code (cluster) | `vista2:/work/11755/gurup12/vista/downscaling` (`$WORK/downscaling`) |
| Allocation | TACC Vista, project `ATM23014` (462 SUs at the start of this work, expires 2027-10-31) |
| Python on Vista | `$WORK/miniforge3/envs/dsc/bin/python` |
| Stores | `data_store_sc/wind_dyn`, `wind_era5`, `wind25`, `wind5` (on `$WORK`); 1 km store on `/scratch/11755/gurup12/wind_km1` |
| Checkpoints | `$WORK/downscaling/checkpoints/<run name>/last.pt` |
| Results | `$WORK/downscaling/outputs/wind/` (`suite_*.json`, `series/*.npz`, `wind_summary.md/.json`, `drift.json`, `mesonet_daily.csv`) |
| Logs | `$WORK/downscaling/logs/wind_*_<jobid>.out/.err`; job IDs in `logs/wind_jobs.txt` |
| Paper drafts | `~/downscaling-precipitation/docs/planned_papers/` (Paper 1 has the first-run wind section only) |

The cluster is reached with `ssh vista2` from the Mac. Commands are run there by hand; Claude cannot ssh from its own environment.

### Files added for the wind work (none of them overwrote an existing file)

```
WIND_PLAN.md                         experiment design
configs/data_sc_wind_dyn.yaml        10 km, AORC target, 27 channels (all ablations are subsets of this store)
configs/data_sc_wind_era5.yaml       10 km, ERA5-Land target
configs/data_sc_wind25.yaml          25 km ladder store
configs/data_sc_wind5.yaml           5 km ladder store
configs/data_sc_wind1.yaml           1 km store, 15 ladder channels, only 4 POWER parameters (memory)
configs/data_sc_wind_dyn_{west,coast,north}.yaml   10 km leave-one-region-out holdouts
configs/data_sc_wind5_{west,coast,north}.yaml      5 km leave-one-region-out holdouts
scripts/wind_plan.py                 single source of truth: every run, store, channels, seed, group
scripts/era5land_wind.py             ERA5-Land hourly u10/v10 download (CDS) and daily mean speed
scripts/mesonet_wind.py              Oklahoma Mesonet download (serial, rate-limited) and daily means
scripts/wind_suite.py                evaluation of every run (grid, budget, offset, spectra, stations)
scripts/wind_report.py               seed statistics, paired station bootstraps, uncertainty test -> wind_summary.md
scripts/wind_drift.py                year-by-year AORC vs POWER drift and model bias (Q7)
scripts/wind_texture_diag.py         how much of AORC's fine-scale daily anomaly is predictable (section 7c)
scripts/wind_inference_maps.py       input, inference, test-year, ladder and LORO map sheets -> outputs/wind/maps/
slurm/wind_00_setup.sh               login-node setup and checks (also patches data/build_dataset.py, see below)
slurm/vista_wind_data.sh             POWER download, 10/25/5 km stores
slurm/vista_wind_era5.sh             ERA5-Land download and store
slurm/vista_wind_km1_build.sh        1 km store
slurm/vista_wind_mesonet.sh          Mesonet download (4 s between files)
slurm/vista_wind_train.sh            trains any plan groups, sharded, resumable, DRYRUN mode
slurm/vista_wind_eval.sh             drift (if ablation), suite, report
slurm/wind_submit.sh                 submits the whole chain with dependencies
slurm/wind_resubmit_km1.sh           resubmits the 1 km build, its training and its eval
```

Older wind files from before this program (still in the repo): `configs/data_sc_wind.yaml`, `scripts/wind_eval.py`,
`scripts/make_aorc_wind.py`, checkpoints `wind_s7`, `wind_s42`, `wind_s1337`.

### One change to shared code

`slurm/wind_00_setup.sh` inserts one line into `data/build_dataset.py`, in `POWER_CHANNELS`, after the `T2M_MAX` entry:

```python
"WS10M": "coarse_tmp",
```

This makes the downscaled field `coarse_tmp` when POWER parameters are given as a list. The edit is idempotent. It exists on
Vista; check whether the Mac copy has it before committing anything to the new repo.

---

## 3. Data

| Source | Use | Notes |
|---|---|---|
| NASA POWER daily, 2019-2023 | input | WS10M (downscaled field), U10M, V10M, WS50M, WS2M, WS10M_MAX, T2M_RANGE |
| AORC v1.1 hourly u, v | target | daily mean of hourly speed (not speed of mean components), 5 year stores |
| ERA5-Land hourly u10, v10 | alternative target | CDS API (`~/.cdsapirc` on Vista), daily mean of hourly speed |
| 21 static fields | inputs | as in the temperature store: DEM, land cover class and 8 fractions, coast distance, urban fraction, terrain (relief, slope, TPI, northness, eastness) |
| ISD-Lite hourly | stations | wind speed columns 37-43 (tenths of m/s); daily mean needs >= 20 hourly reports; 44 stations in the 10 km holdout, 45 at 25 and 5 km |
| Oklahoma Mesonet MTS | second station network | WSPD (10 m) and WS2M, 5-minute; daily mean needs >= 80% of 288 samples; 2023 only |

Grid: South-Central domain, Albers, 93 x 100 cells at 10 km. Years: train 2019-2021, validate 2022, test 2023.
Holdout: Austin box plus 1° buffer (1,295 cells at 10 km, 210 at 25 km, 5,250 at 5 km).

LORO holdout boxes (all with 1° buffer):

| Region | Longitude | Latitude |
|---|---|---|
| west | -102.4 to -100.8 | 30.6 to 31.9 |
| coast | -96.6 to -95.0 | 28.6 to 29.9 |
| north | -99.4 to -97.8 | 33.0 to 34.3 |

---

## 4. Methods as implemented

### Model

Residual on interpolated POWER WS10M in anomaly space (15-day-smoothed day-of-year climatology removed from inputs and
target, fit on training days), the temperature program's loss and training settings, 20 epochs, fp32. Trained with
`python -m training.train --arch <arch> --seed <s> --zarr <store> --run-name <name> --use-channels <list>`, with
`DOWNSCALE_CONFIG_data=<config>` exported.

### Stores

| Tag | Config | Store | Channels | Grid |
|---|---|---|---|---|
| w10 | data_sc_wind_dyn | data_store_sc/wind_dyn | 27 | 10 km, AORC |
| w10e | data_sc_wind_era5 | data_store_sc/wind_era5 | 27 | 10 km, ERA5-Land |
| w25 | data_sc_wind25 | data_store_sc/wind25 | 27 | 25 km |
| w5 | data_sc_wind5 | data_store_sc/wind5 | 27 | 5 km |
| w1 | data_sc_wind1 | /scratch/.../wind_km1 | 15 | 1 km |
| w10{west,coast,north} | data_sc_wind_dyn_{reg} | same as w10 | 27 | 10 km, moved holdout |
| w5{west,coast,north} | data_sc_wind5_{reg} | same as w5 | 27 | 5 km, moved holdout |

### Channel sets (`scripts/wind_plan.py`)

| Set | Channels |
|---|---|
| A0 | POWER speed, land mask, day-of-year sin/cos (4) |
| A1 | A0 + DEM, land cover class, coast distance, urban fraction (8) |
| A2 | A1 + terrain: elev_std, slope, TPI, northness, eastness |
| A3 | A1 + 8 land-cover fractions |
| A4 | all 21 statics (= the first wind run) |
| A5 | A4 + POWER u, v (direction) |
| A6 | A4 + POWER 50 m and 2 m speed (shear) |
| A7 | A4 + daily max speed and diurnal temperature range (gust, stability) |
| A8 | all 27 |
| L | 15-channel ladder set: speed, u, v, 50 m speed, mask, season (2), DEM, elev_std, TPI, coast distance, urban, forest, water, developed |

### Runs (85, about 31.6 GPU-h estimated)

Naming: `{store}_{chan}_s{seed}` for DeepSD, `{store}_{arch}_{chan}_s{seed}` otherwise. Seeds: 1337, 7, 42 (and 2024, 31337 where five).

| Group | Contents | Runs |
|---|---|---|
| ablation (E1) | DeepSD, w10, A0-A8, 3 seeds | 27 |
| arch (E2) | linear, unet, edsr, convnext, swin, swinir, restormer on w10 A8, 3 seeds | 21 |
| target (E3) | DeepSD A8 on w10e, 5 seeds, plus w10 A8 seeds 2024 and 31337 | 7 |
| ladder (E5) | DeepSD L on w25, w10, w5, 5 seeds | 15 |
| ladder1 (E5) | DeepSD L on w1, 3 seeds | 3 |
| loro (E6) | DeepSD L on w10 and w5 for each region, 2 seeds | 12 |

### Evaluation (`scripts/wind_suite.py`, test year 2023, holdout of each store)

- Grid: RMSE, MAE, bias, r, SSIM, KL divergence of the speed distribution, 50/90/99th percentile ratios, mean-cube
  ratio (wind power density proxy), CSI and POD at 6 and 8 m/s.
- Error budget: uniform daily offset, static pattern, remainder (`evaluation.fields.decompose`).
- Offset correction: ridge on 6 x 6 block means of the training-region residual, fit on training days and applied to
  test days; plus the oracle (true offset) and the recovered fraction.
- Spectra: amplitude ratio and coherence by wavelength, effective and coherence resolution.
- Stations: ISD and Mesonet, raw and offset-corrected; per-station series for 2022-2023 saved for the uncertainty test.
- References per store: interpolated POWER and the target itself at the stations.
- Target group: every run also scored against the other target (cross-scoring).
- The JSON is written after every run, so an interrupted eval resumes.

### Summary (`scripts/wind_report.py`)

Seed mean and SD per variant; paired station bootstraps (5,000 draws, on seed-mean per-station RMSE, shared stations)
for: each ablation set vs A0, A8 vs A4, A4 vs A1, each variant vs POWER, AORC vs POWER, each architecture vs DeepSD,
ERA5-trained vs AORC-trained, each ladder step raw and offset-corrected, the target itself across grids, LORO 10 to 5 km;
the architecture-to-seed noise ratio; cross-scores; seed-ensemble intervals (calibrated on 2022, scored on 2023:
coverage, width, CRPS).

### Drift (`scripts/wind_drift.py`, Q7)

Year-by-year mean AORC and POWER over the holdout and the training region, their difference, and the yearly bias of the
A0, A4 and A8 models. Runs at the start of the ablation eval; writes `outputs/wind/drift.json`.

---

## 5. Job ledger

### Before this program

| Job | What | Result |
|---|---|---|
| 1135948 | first wind run: DeepSD, 21 statics, AORC 10 km, 3 seeds (`wind_s7`, `wind_s42`, `wind_s1337`) | completed; grid RMSE 1.128 -> 0.786 m/s; ISD 1.152 -> 0.971; station change -0.18 m/s, 95% CI -0.39 to +0.02; grid bias -0.42 (overcorrects POWER's +0.84) |

### Submission 1 (`bash slurm/wind_submit.sh`, 12 Nov 2026)

| Job | Name | Depends on | Result |
|---|---|---|---|
| 1137868 | wind-data (gg) | | COMPLETED, 13 min, 11.5 GB |
| 1137869 | wind-era5 (gg) | | COMPLETED, 4 h 01 min |
| 1137870 | wind-km1 (gg) | 1137868 | FAILED, out of memory at 238 GB (all 7 POWER parameters regridded to 1 km) |
| 1137871 | wind-meso (gg) | | COMPLETED but rate-limited after 1 station (1 s delay) |
| 1137872 | train ablation shard 0 | data | COMPLETED, 0 failed |
| 1137873 | train ablation shard 1 | data | COMPLETED, 2 h 54 min, 0 failed |
| 1137874-76 | train arch shards 0-2 | data | COMPLETED, about 2 h 40 min each, 0 failed |
| 1137877 | train target | data, era5 | COMPLETED, 0 failed |
| 1137878 | train ladder | data | COMPLETED, 0 failed |
| 1137879 | train loro | data | COMPLETED, 0 failed |
| 1137880-82 | train ladder1 | km1 | CANCELLED (dependency never satisfied) |
| 1137883 | eval ablation, arch, target (5 h) | all training, meso | PENDING at time of writing |
| 1137884 | eval ladder, loro (3 h) | all training, meso | COMPLETED 13 Nov, 6.5 min, 30 Mesonet stations |
| 1137885 | eval ladder1 | ladder1 training | CANCELLED |
| 1137861, 1137886 | idev GPU checks | | abandoned in the queue; the GPU stack was confirmed by the training jobs instead |

### Mesonet resubmissions

| Job | Result |
|---|---|
| 1139015 | COMPLETED 5 h 27 min, 13 stations, then rate-limited (delay now 4 s) |
| 1140598 | COMPLETED 8 h 35 min, 30 of 41 stations, 10,567 valid 10 m station-days, then rate-limited |

Decision: stop at 30 stations. `outputs/wind/mesonet_daily.csv` is the 30-station file; earlier partial files were renamed
`mesonet_daily_partial.csv`.

### 1 km resubmissions (`bash slurm/wind_resubmit_km1.sh`)

| Job | What | Result |
|---|---|---|
| 1139010 | km1 build | FAILED, out of memory at 236 GB: the fixed config had not reached Vista |
| 1139011-13, 1139014 | ladder1 training and eval | CANCELLED |
| 1139933 | km1 build (4 POWER parameters) | COMPLETED, 57 min, peak 167 GB of 237 |
| 1139934-36 | train ladder1, 3 shards | PENDING at time of writing |
| 1139937 | eval ladder1 (8 h) | PENDING, waits for 1139934-36 |

Trained so far: 82 of 85 runs (all except the three 1 km seeds).

Checked 13 Nov, about 12:40: 1137883 (eval ablation, arch, target) still PENDING (Priority); 1139934-36 PENDING
(Priority); 1139937 PENDING (Dependency). Nothing failed.

---

## 6. Problems met and fixes (keep these in mind)

| Problem | Fix |
|---|---|
| `sbatch --parsable` printed a banner line before the job ID | take `| tail -n 1` |
| `GROUPS` is a bash built-in, so the trainer ignored it | variable renamed `WGROUPS` |
| `DOWNSCALE_CONFIG_data` not seen by the trainer | must be `export`ed; `load_config` reads the environment |
| ERA5 and 1 km builds could start before the main data job finished | wait loop in the ERA5 job; `afterok` dependency for 1 km |
| 1 km build ran out of memory twice | `data_sc_wind1.yaml` now overrides `sources.power.parameters` to the 4 used; peak 167 GB on a gg node (237 GB). gh nodes are too small for it. |
| Fixed config "delivered" to the Mac but the old file stayed | always check file size or `grep` after copying, on the Mac and on Vista |
| Mesonet blocks after hours of downloads | looks like a daily cap; serial, 4 s delay, resumes on resubmission; stopped at 30 stations |
| Pasting a block that starts with `ssh vista2` runs only the ssh | paste Mac commands and Vista commands separately |
| Vista upgrade on 11 Nov (Rocky 9.8, new CUDA driver) | training ran normally afterwards |
| 19% of 1 km cells had no land-cover data (filled from nearest cell) | likely Mexico and Gulf, outside NLCD; same as the temperature 1 km store; to be noted in the paper |
| Cross-scoring skipped because of a relative-path existence check | replaced with try/except |
| Build warning "Mean of empty slice" in the climatology | harmless (cells with no truth) |

---

## 7. Results so far (ladder and LORO eval, 13 Nov; ablation, architecture and target still pending)

Summary rows (columns: seeds, grid RMSE, grid RMSE after offset correction, ISD RMSE, ISD after correction, Mesonet
RMSE, coherence resolution in km, SSIM, KL):

| Variant | Seeds | Grid | +offset | ISD | +offset | Meso | Coh. res. | SSIM | KL |
|---|---|---|---|---|---|---|---|---|---|
| w25 L | 5 | 0.763 | 0.558 | 0.971 | 0.803 | 0.668 | 145.9 | 0.635 | 0.074 |
| w5 L | 5 | 0.761 | 0.567 | 0.941 | 0.742 | 0.641 | 204.0 | 0.729 | 0.066 |
| w10 coast L | 2 | 0.749 | 0.625 | 1.066 | 0.916 | 0.660 | 110.4 | 0.777 | 0.027 |
| w10 north L | 2 | 0.646 | 0.584 | 0.921 | 0.838 | 0.658 | 247.0 | 0.717 | 0.016 |
| w10 west L | 2 | 0.741 | 0.675 | 0.996 | 0.927 | 0.655 | 186.3 | 0.714 | 0.027 |
| w5 coast L | 2 | 0.741 | 0.603 | 1.043 | 0.951 | 0.638 | 110.6 | 0.823 | 0.036 |
| w5 north L | 2 | 0.666 | 0.591 | 0.913 | 0.812 | 0.648 | 245.9 | 0.766 | 0.021 |
| w5 west L | 2 | 0.754 | 0.679 | 1.015 | 0.928 | 0.645 | 161.4 | 0.764 | 0.028 |

(The w10 L row was cut off in the output that was checked; it is in `wind_summary.md`.)

Paired ISD comparisons (b minus a, m/s; negative = b better):

| Comparison | d | 95% CI | b worse at |
|---|---|---|---|
| w10 L vs POWER | -0.191 | -0.395 to +0.014 | 18/44 |
| AORC vs POWER | -0.636 | -0.795 to -0.481 | 6/44 |
| ladder 25 -> 10 km | -0.005 | -0.054 to +0.042 | 23/44 |
| ladder 25 -> 10 km, offset-corrected | -0.038 | -0.093 to +0.011 | 18/44 |
| AORC itself 25 -> 10 km | -0.069 | -0.136 to -0.004 | 16/44 |
| ladder 10 -> 5 km | -0.018 | -0.045 to +0.009 | 20/44 |
| ladder 10 -> 5 km, offset-corrected | -0.021 | -0.043 to +0.001 | 18/44 |
| AORC itself 10 -> 5 km | -0.029 | -0.053 to -0.003 | 16/44 |
| LORO west 10 -> 5 km | +0.018 | -0.017 to +0.056 | 12/20 |
| LORO west, offset-corrected | +0.002 | -0.030 to +0.034 | 11/20 |
| LORO coast 10 -> 5 km | -0.013 | -0.042 to +0.016 | 14/31 |
| LORO coast, offset-corrected | -0.020 | -0.050 to +0.010 | 11/31 |
| LORO north 10 -> 5 km | -0.008 | -0.033 to +0.017 | 19/39 |
| LORO north, offset-corrected | -0.026 | -0.050 to -0.002 | 13/39 |

Reading so far (preliminary):

1. The model is not reliably better than interpolated POWER at stations (-0.19 m/s, CI crosses zero), while AORC itself
   is far better (-0.64 m/s). The target holds station skill the model does not learn from POWER.
2. Finer grids do not help the model (25 -> 10 and 10 -> 5 km both n.s.), although AORC itself improves at each step.
3. The same-day offset correction is large for wind: 5 km ISD 0.94 -> 0.74 m/s; grid MSE down about 44%. This goes
   against the plan's hypothesis (Q4) that wind error would be more local than temperature's; confirm with the budget.
4. LORO agrees: no reliable 10 -> 5 km gain in any region (north after offset correction is a weak exception).

---

## 7b. Inference maps (13 Nov)

`scripts/wind_inference_maps.py` draws, in m/s with each field's day-of-year climatology added back:

| Sheet | Content |
|---|---|
| `inputs_<store>_<chan>_<day>.png` | every channel the run sees that day (except day-of-year sin/cos and the land mask), POWER direction as arrows |
| `inference_<run>_<day>.png` | POWER, AORC, model, correction made; POWER error and model error on one scale; correction needed vs made; holdout RMSE bars (grid and ISD); ISD circles and Mesonet squares coloured with observed daily means |
| `testyear_<run>.png` | 2023 mean (AORC, POWER, model), bias, per-cell RMSE, RMSE change |
| `ladder_<day>.png` | 25, 10, 5 km (and 1 km once trained) on the same day |
| `loro_<day>.png` | west, coast and north holdouts at 10 km |

Days are picked automatically from the 10 km holdout mean AORC speed in 2023: windy 2023-02-10 (5.95 m/s), typical
2023-04-22 (3.18), calm 2023-11-16 (1.40). Predictions are read from `.field_cache` when the eval has made them;
otherwise only the needed days are run on CPU and nothing is written to the cache (its key has no device, so a CPU
entry would become the field a later GPU eval scores). It runs on a login node in about 2 minutes for the 6 ladder and
LORO runs.

First set made 13 Nov: w25/w10/w5 L s1337 and the three 10 km LORO runs (45 sheets, in `outputs/wind/maps/`, copied
to the Mac). Reading:

- The 2023 sheet for w10 L reproduces the eval: POWER RMSE 1.13, model 0.78; POWER bias +0.84, model -0.43. The model
  removes POWER's high bias almost everywhere and then goes too low, most in the holdout and the east.
- On the windy day (2023-02-10) the model beats POWER on the grid (1.21 vs 1.33) but is worse at the 43 ISD stations
  (1.52 vs 1.26): it lowers speeds by 1-3 m/s nearly everywhere, and the correction-made vs needed slope is only 0.26.
  The station problem is a level problem, not a pattern problem.
- The ladder sheet shows why finer grids do not help: at 5 km AORC gains terrain-scale detail (ridges, Hill Country
  valleys) that the model reproduces only in a smoothed form, and the error map at all three grids is dominated by
  the same broad negative offset in the holdout.
- Land-cover fractions show straight-line stripes in Mexico (south-west corner, e.g. developed and water fraction).
  Probably nearest-cell fill outside NLCD; check before the paper's input figure.

To redraw after eval 1137883 has cached the ablation, architecture and target fields:

```bash
cd $WORK/downscaling
python scripts/wind_inference_maps.py --runs w10_A0_s1337 w10_A4_s1337 w10_A8_s1337 w10_unet_A8_s1337 \
    w10_swinir_A8_s1337 w10e_A8_s1337 --only inference testyear
python scripts/wind_inference_maps.py --runs w1_L_s1337 --only ladder testyear     # after the 1 km training
scp "vista2:/work/11755/gurup12/vista/downscaling/outputs/wind/maps/*.png" outputs/wind/maps/   # on the Mac
```

## 7c. Why structure and texture do not improve (13 Nov diagnosis)

All on the 10 km holdout, test year 2023 unless stated, w10 L s1337, anomaly space (store climatology removed).

Predictability of AORC's daily anomaly below 100 km (`scripts/wind_texture_diag.py`, high-pass Gaussian at 100 km):

| Predictor | r2 | amplitude / AORC |
|---|---|---|
| model | 0.15 | 0.53 |
| ridge on 29 flow x terrain features (u.grad h, speed x TPI, roughness, aspect), fit in the training region | 0.10 | 0.20 |
| ridge + model | 0.16 | 0.44 |
| per-cell climatology by POWER direction (8) x speed (3), 2019-21, holdout cells' own history | 0.19 | 0.45 |
| that + model (weights fit on 2022) | 0.22 | 0.47 |

Scale bands, r2 between daily anomalies over the holdout:

| Band | sd AORC | sd ERA5-Land | sd model | AORC~ERA5-Land | AORC~model | AORC~POWER |
|---|---|---|---|---|---|---|
| < 100 km | 0.23 | 0.06 | 0.13 | 0.05 | 0.17 | 0.04 |
| 100-300 km | 0.27 | 0.13 | 0.12 | 0.18 | 0.18 | 0.14 |
| > 300 km | 1.15 | 1.07 | 0.96 | 0.89 | 0.88 | 0.87 |

At the 44 ISD stations (2022-23), daily departure from POWER with each series' mean removed:
r(station, AORC) median 0.88, r(station, model) 0.32; de-biased RMSE POWER 0.67, AORC 0.32, model 0.68 m/s.

Reading:

1. AORC's fine-scale daily wind is real at stations (r 0.88) but carries information POWER does not have; ERA5-Land
   does not have it either (r2 0.05 with AORC below 100 km). Most likely AORC's wind takes in surface observations
   (airport stations among them, so ISD is not fully independent of AORC; check the AORC wind provenance).
2. No loss, architecture or terrain feature can place that structure from POWER: the best predictor found explains
   22% of it, the model 15%. At stations the model adds no daily skill over POWER once bias is removed; its station
   gains are level (bias) plus the static texture carried by the climatology.
3. So "drastically better structure" needs either new information (observations, or a finer analysis such as
   HRRR/RTMA, as an input) or a generative model that gives the right texture statistics without placing it.
4. Mesonet obs are NaN in `_ref_*.npz` (obs only stored for ISD keys); redo the station check with
   `mesonet_daily.csv`.

## 7d. Station fusion (A) and residual diffusion (B), submitted 13 Nov

Decision (13 Nov): fine-scale structure needs observations, so add same-day ISD stations as inputs (A), with
residual diffusion on top for texture (B). The product becomes POWER + stations.

- `scripts/wind_obs_store.py`: 196 ISD stations active 2019-2023 on land in the domain, 190 with >= 300 train days
  (42 in the holdout, folds A 21 / B 21, random, seed 0). Departure = daily mean ISD speed - interpolated POWER WS10M,
  minus each station's own train-period mean and annual harmonic. Four channels: kernel mean of departures at 25 and
  75 km and the kernel weights. Outside the holdout a random 60% of stations per day (seeded by day); inside, fold f
  in store f. Stores `data_store_sc/wind_obs_f{0,1}` (31 channels; first 27 identical to wind_dyn, checked).
  Outputs: `outputs/wind/obs_stations.json`, `obs_departures.npy`, `isd_domain_daily.csv`.
- Sanity: the 75 km kernel departure alone tracks the WITHHELD stations' departures at median r 0.63 (both folds),
  against 0.32 for the no-station model.
- Plan group `obs` (5 runs, fold-0 store): DeepSD LO x 3 seeds, DeepSD A8O, U-Net LO.
- `scripts/wind_obs_eval.py`: scores both folds, ISD only at the non-input fold, Mesonet (never input); references
  POWER, AORC, kernel interpolation, no-station models -> `outputs/wind/obs_eval.{json,md}`.
- `scripts/wind_diffusion.py`: EDM residual diffusion (models/diffusion.py) on the mean of the three LO seeds,
  conditioned on LO channels + the deterministic field; 25 members of 2023 on both fold stores; CRPS, spread/skill,
  80% coverage, band sd ratios -> `outputs/wind/diffusion/obs_score.json`.

| Job | What | Partition |
|---|---|---|
| 1142301 | wind-obs: train group obs, obs eval, maps | gh-dev, 2 h (COMPLETED 1 h 16 min) |
| 1142302 | wind-diff: residual diffusion, afterok 1142301 | gh-dev, 2 h (COMPLETED 4.5 min) |
| 1143913 | wind-diff on 2022 residuals, 600 epochs, tag obsval | gh-dev, 45 min (pending, est. start 14 Nov 12:50) |

### References and the non-ML bar (13 Nov, login node, before the fusion runs)

2023, mean of both folds; ISD only at the withheld fold (21 stations each), Mesonet (30) never an input.
`outputs/wind/obs_eval.md` (references) and `obs_hybrid.md` (model + kernel residuals, `scripts/wind_obs_hybrid.py`).

| Variant | grid RMSE | grid bias | r2 <100 | r2 100-300 | ISD de-biased | ISD r_dep | Meso de-biased | Meso r_dep |
|---|---|---|---|---|---|---|---|---|
| POWER | 1.128 | +0.84 | 0.00 | 0.11 | 0.671 | 0 | 0.627 | 0 |
| kernel 75 km over POWER | 1.161 | +0.98 | 0.01 | 0.16 | 0.517 | 0.64 | 0.517 | 0.58 |
| w10 L (3 seeds) | 0.773 | -0.43 | 0.15 | 0.14 | 0.671 | 0.38 | 0.523 | 0.52 |
| w10 A8 s1337 | 0.838 | -0.48 | 0.16 | 0.12 | 0.722 | 0.34 | 0.522 | 0.53 |
| **w10 L + 75 km station residuals** | **0.623** | -0.32 | 0.15 | 0.20 | **0.520** | **0.65** | **0.491** | **0.65** |
| AORC | 0 | 0 | 1 | 1 | 0.311 | 0.88 | 0.352 | 0.82 |

The fusion network must beat model + residuals (grid 0.62, ISD de-biased 0.52, r_dep 0.65); if not, that simple
combination is the product and diffusion goes on top of it. Stations do not move < 100 km (spacing ~70 km).

### Fusion results (job 1142301, 13 Nov 20:40-21:56, 1 h 16 min, no errors)

`outputs/wind/obs_eval.md`. 2023, mean of both folds.

| Variant | grid RMSE | grid bias | r2 <100 | r2 100-300 | ISD de-biased | ISD r_dep | Meso de-biased | Meso r_dep |
|---|---|---|---|---|---|---|---|---|
| w10 L + 75 km residuals (bar) | **0.623** | -0.32 | 0.15 | 0.20 | **0.520** | **0.65** | 0.491 | 0.65 |
| w10o0 LO s1337 | 0.645 | -0.31 | 0.17 | 0.24 | 0.614 | 0.56 | 0.446 | 0.69 |
| w10o0 LO s7 | 0.652 | -0.31 | 0.16 | 0.24 | 0.619 | 0.55 | 0.449 | 0.69 |
| w10o0 LO s42 | 0.665 | -0.34 | 0.16 | 0.23 | 0.618 | 0.55 | 0.443 | 0.70 |
| w10o0 A8O s1337 | 0.677 | -0.35 | 0.17 | 0.24 | 0.630 | 0.53 | 0.438 | 0.70 |
| w10o0 U-Net LO s1337 | 0.662 | -0.27 | 0.14 | 0.11 | 0.622 | 0.55 | **0.433** | **0.72** |

- The network does not beat the hybrid on the grid or at the withheld ISD stations. It does beat it at Mesonet
  (de-biased 0.44 against 0.49, r 0.70 against 0.65). Likely cause: MSE training shrinks the station signal. The
  hybrid adds it at full weight, and that pays off at ISD airports, whose neighbours are also airports.
- Texture is unchanged: r2 < 100 km is 0.14-0.17 for every variant. At 100-300 km it rises from 0.14 to 0.24.
- Grid bias of -0.31 is about 22% of the MSE.
- Maps: `outputs/wind/maps/{inputs,inference,testyear}_w10o0_*`. On the test-year RMSE map, the error comes in round
  blobs 50-80 km across, the size of the station kernels. A station that does not represent its area spreads its
  error over a kernel disk.

### Stacking: residuals on top of the fusion network (`wind_obs_hybrid.py --runs w10o0_LO_s1337 --out obs_stack`)

The network's own residuals at its input stations, de-biased and kernel-spread as in the hybrid, then added back.
`outputs/wind/obs_stack.md`.

| Variant | grid RMSE | grid bias | r2 <100 | r2 100-300 | ISD de-biased | ISD r_dep | Meso de-biased | Meso r_dep |
|---|---|---|---|---|---|---|---|---|
| w10o0 LO s1337 | 0.645 | -0.31 | 0.17 | 0.24 | 0.614 | 0.56 | **0.446** | **0.69** |
| + 75 km residuals | **0.599** | -0.30 | 0.17 | 0.26 | 0.531 | 0.64 | 0.488 | 0.66 |
| + 25 km residuals | 0.629 | -0.30 | 0.15 | 0.24 | 0.561 | 0.62 | 0.539 | 0.63 |
| + blend | 0.609 | -0.30 | 0.14 | 0.27 | 0.542 | 0.63 | 0.515 | 0.64 |

- Stacking gives the best grid RMSE so far (0.599) and matches the hybrid at ISD. It gives up the network's lead at
  Mesonet.
- The scorers disagree. Full-weight ISD residuals help against AORC and ISD but hurt against Mesonet. AORC tracks
  ISD (r 0.88) more closely than Mesonet (0.82) and may share sources with the airport network. That makes Mesonet the
  most independent judge, although its stations sit in the training region and are only temporally independent.
- Any residual weight between 0 and 1 must be chosen on 2022, not on the test year.

### Drift: the 2023 bias is half POWER, half AORC (`scripts/wind_bias_years.py`, login node)

`outputs/wind/bias_years.md`, model w10_L_s1337. Grid columns: mean over land cells. ISD columns: all 190 stations,
each against its own cell.

| year | split | POWER-AORC hold | model-AORC hold | POWER-AORC out | model-AORC out | ISD-AORC | ISD-POWER | ISD-model |
|---|---|---|---|---|---|---|---|---|
| 2019 | train | +1.617 | +0.269 | +0.829 | +0.095 | +0.426 | -0.698 | +0.263 |
| 2020 | train | +1.336 | +0.002 | +0.704 | -0.019 | +0.241 | -0.699 | +0.256 |
| 2021 | train | +0.857 | -0.379 | +0.538 | -0.144 | +0.015 | -0.625 | +0.269 |
| 2022 | val | +0.775 | -0.577 | +0.467 | -0.260 | +0.040 | -0.543 | +0.433 |
| 2023 | test | +0.844 | -0.433 | +0.484 | -0.206 | +0.092 | -0.503 | +0.413 |

- AORC gets windier relative to the stations: ISD-AORC falls from 0.43 to about 0.05, mostly over 2019-2021.
- POWER gets calmer relative to the stations: ISD-POWER rises from -0.70 to -0.50.
- The holdout drifts about twice as much as the training region (POWER-AORC falls 1.6 to 0.8, against 0.8 to 0.5).
- The model is stable against the stations in the training years (ISD-model +0.26) and drifts by 0.15 in 2022-23,
  following POWER.
- Of the 2023 grid bias (-0.43 against a train-period mean of -0.04), about 0.15 is POWER drift, which the stations
  can correct. The rest is AORC drifting away from the stations. No model should chase that part. The de-biased
  station scores (rmse_db, r_dep) are blind to both, so they remain the honest measure.

### Residual diffusion, first run (job 1142302, 14 Nov 03:07-03:12, 4.5 min)

Base: mean of the three LO seeds. Trained on train-year residuals, 60 epochs (about 8,000 steps, loss still falling),
25 members of 2023 on both fold stores. `outputs/wind/diffusion/obs_score.json` and `obs_centered_score.json`
(`--score-only --center`: members re-centred so that the ensemble mean equals the deterministic field). Mean of folds.

| | grid RMSE | member <100 km sd ratio / r2 | grid CRPS (det MAE) | spread/skill | ISD CRPS (det MAE) | ISD cover80 | Meso CRPS (det MAE) | Meso cover80 |
|---|---|---|---|---|---|---|---|---|
| deterministic | 0.648 | 0.51 / 0.18 | (0.493) | | (0.709) | | (0.460) | |
| ensemble, raw | mean 0.701, member 0.788 | 0.89 / 0.05 | 0.416 | 0.50 | 0.613 | 0.31 | 0.335 | 0.56 |
| ensemble, centred | mean 0.648, member 0.740 | 0.90 / 0.06 | 0.372 | 0.55 | 0.574 | 0.34 | 0.333 | 0.57 |

- Texture amplitude comes back (sd ratio below 100 km rises from 0.51 to 0.90). As expected, its placement is random
  (r2 0.06), so the ensemble is useful as a distribution and not as a sharper point estimate.
- The raw ensemble mean is worse than the deterministic field. Centring fixes that at no cost.
- The ensemble is too narrow: spread/skill 0.55, and ISD 80% coverage about 0.33. Cause: the residuals were learned on
  the train years, where the base model is in-sample (residual sd 0.50 against a test RMSE of 0.65 plus the drift
  bias).
- Fix submitted as job 1143913 (gh-dev, 45 min): `EPOCHS=600 TAG=obsval EXTRA="--train-split val"`. It learns the 2022
  residuals, which are out-of-sample for the base model, then scores raw and centred
  (`outputs/wind/diffusion/obsval_score.json`, `obsval_centered_score.json`).

## 8. What remains

0. Redraw the maps for the ablation, architecture, target and 1 km runs (section 7b).
1. Wait for eval 1137883 (ablation, architecture, target, plus drift). Then `cat outputs/wind/wind_summary.md`.
2. Wait for the 1 km training (1139934-36) and its eval (1139937); add the 1 km rung to the ladder.
3. Copy results to the Mac:
   ```bash
   mkdir -p ~/downscaling-temperature/outputs/wind
   scp -r "vista2:/work/11755/gurup12/vista/downscaling/outputs/wind/*" ~/downscaling-temperature/outputs/wind/
   ```
4. Write the wind section of Paper 1 in the structure of the temperature section: architecture, inputs, target, error
   budget and offset, resolution ladder with LORO, stations on two networks, uncertainty, drift. AMS/JAMC style, plain
   language, no code or repository references in the text, no em dashes. Figures in the shared style
   (`papers_ams/common/figstyle.py`).
5. Paper notes to carry: Mesonet is 30 of 41 stations, and those stations lie in the training region for the Austin
   holdout (temporally independent only); 1 km land-cover gap of 19%; ISD panel is 44 stations at 10 km and 45 at 25/5 km
   (comparisons use shared stations).

### Possible extensions

- Rerun Mesonet scoring if more stations are downloaded (a small Mesonet-only pass that rescores existing runs).
- A diffusion or other generative model if the uncertainty test (E9) shows the seed ensemble is under-dispersive.
- Wind-specific predictors not yet tried: terrain exposure by direction, surface roughness length, boundary-layer height.
- Hub-height (50-100 m) targets for wind energy use.

---

## 9. Quick reference

```bash
# on Vista
cd $WORK/downscaling
squeue -u $USER
sacct -u $USER -S today --format=JobID,JobName%15,State,Elapsed,MaxRSS
cat logs/wind_jobs.txt
python scripts/wind_plan.py --summary
DRYRUN=1 WGROUPS=ladder bash slurm/vista_wind_train.sh          # what would train, trains nothing
sbatch -t 05:00:00 --export=ALL,WGROUPS=ablation,SHARD=0,NSHARDS=2 slurm/vista_wind_train.sh
sbatch -t 04:00:00 --export=ALL,WGROUPS="ablation arch target" slurm/vista_wind_eval.sh
sbatch slurm/vista_wind_mesonet.sh                               # resumes the download
```

Everything is resumable: training skips runs with `checkpoints/<name>/last.pt`, the suite skips runs already in its JSON
(delete or rename `outputs/wind/suite_<groups>.json` to rescore), and the Mesonet download skips files already on disk.
Check `scontrol show res` for maintenance windows before long submissions.
