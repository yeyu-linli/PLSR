# PLSR attribution for the Ganjiang River Basin

Code accompanying: *Assessing hydrological responses to land-use changes using SWAT and
partial least squares regression: a case study in the Ganjiang River Basin, China*
(Ecological Indicators).

## What it does
Partitions the simulated 1980-2018 change of six hydrological components among five
land-use predictors with partial least squares regression (PLSR), and reports
model-fit statistics (Table 4), standardized coefficients with bootstrap intervals
and contribution rates (Tables 5 and 6).

Response groups: SURQ+WYLD, GWQ+PERC, ET, LATQ.
Screening: studentized residuals > 1.7 sigma removed (max three rounds).
Components: seven-fold cross-validated Q2, Wold rule (>5 percentage points per component).
Uncertainty: 2,000 sub-basin bootstrap resamples (seed 42). Importance: VIP.

## Input
example_input.csv - one row per sub-basin:
Subbasin, AGRL, FRST, PAST, WATR, URBN (land-use change rate, %),
SURQ, WYLD, GWQ, LATQ, PERC, ET (component change, mm/yr; 2018 minus 1980).

## Run
`
pip install numpy pandas scikit-learn
python plsr_main.py example_input.csv
`
Outputs model_summary.csv (Table 4) and coefficients.csv (Tables 5-6).

## Reference results (example_input.csv)
| model | responses | N | nc | R2 | Q2 |
|---|---|---|---|---|---|
| PLSR1 | SURQ and WYLD | 57 | 3 | 0.994 | 0.991 |
| PLSR2 | GWQ and PERC | 61 | 3 | 0.982 | 0.850 |
| PLSR3 | ET | 60 | 3 | 0.992 | 0.943 |
| PLSR4 | LATQ | 59 | 3 | 0.657 | 0.594 |

Python 3.12, scikit-learn 1.8.
