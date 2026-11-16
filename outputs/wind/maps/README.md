# Wind map sheets

Drawn by `scripts/wind_inference_maps.py` (inputs, inference, test year, ladder, LORO) and
`scripts/wind_diffusion_maps.py` (diffusion). Speeds in m/s with each field's day-of-year climatology added back.
Run names follow `scripts/wind_plan.py`; all are seed 1337. Test year 2023.

Days: windy 2023-02-10 (holdout mean AORC 5.95 m/s), typical 2023-04-22 (3.18), calm 2023-11-16 (1.40).

| Sheet | What it shows |
|---|---|
| `inputs_<store>_<chan>_<day>.png` | every input channel the run sees that day (except season and land mask), POWER direction as arrows |
| `inference_<run>_<day>.png` | POWER, AORC, model, correction made; POWER and model error; correction needed vs made; holdout RMSE (grid and ISD); ISD circles and Mesonet squares with observed daily means |
| `testyear_<run>.png` | 2023 mean (AORC, POWER, model), bias, per-cell RMSE, RMSE change against POWER |
| `ladder_<day>.png` | 25, 10 and 5 km on the same day |
| `loro_<day>.png` | west, coast and north holdouts at 10 km |
| `diffusion_obsval_<day>.png`, `_testyear.png` | AORC, base network, MSE corrector, diffusion ensemble mean, three members, spread, errors; test-year RMSE, spread, rank histograms |
| `diffusion_obsvalcal3_*` | the same after the three-band spread calibration |

Runs: `w10_A0/A4/A8` (predictor sets), `w10_linear/swin/swinir/unet_A8` (architectures), `w10e_A8` (ERA5-Land target),
`w25/w10/w5_L` (ladder), `w10west/coast/north_L` (LORO), `w10o0_LO`, `w10o0_A8O`, `w10o0_unet_LO` (station fusion,
fold 0), `w10o0_LO4y` (station fusion trained on 2019-22).
