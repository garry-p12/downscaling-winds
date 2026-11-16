Two-stage control, test year 2023, mean of the two fold stores (holdout land cells; ISD at the withheld fold; Mesonet never an input).

| Variant | grid RMSE | grid bias | ISD MAE | Meso MAE |
|---|---|---|---|---|
| network 2019-21 (3 seeds) | 0.648 | -0.320 | 0.709 | 0.460 |
| network 2019-22 (3 seeds) | 0.544 | -0.187 | 0.621 | 0.458 |
| network 2019-22 minus its own 2023 bias (oracle) | 0.511 | +0.000 | 0.579 | 0.482 |
| network 2019-21 + MSE corrector on 2022 | 0.517 | +0.066 | 0.546 | 0.480 |
| network 2019-21 + diffusion on 2022 (ens. mean) | 0.502 | +0.086 | 0.522 | 0.466 |
