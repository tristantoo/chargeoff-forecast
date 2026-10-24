# Credit card charge-off forecasting with calibrated intervals

How well can the US credit card charge-off rate be forecast 1 to 4 quarters ahead, how wide do the intervals have to be to be honest, and do unemployment and delinquency help? Data: Federal Reserve charge-off and delinquency rates for credit cards at all commercial banks, plus the BLS unemployment rate, all from FRED.

## Run it

```
uv sync
uv run pytest                                  # 4 tests, including the leakage test
uv run python scripts/load_sqlserver.py        # SQL Server LocalDB load + reconciliation (exit 1 on mismatch)
uv run python -m chargeoff.run                 # downloads data if missing, about 4 minutes
```

Set `CHARGEOFF_SQL_CONN` to an ODBC connection string to use a server other than LocalDB.

## Data

| Series | What | Quarters used |
|---|---|---|
| CORCCACBS | charge-off rate on credit card loans, all commercial banks, SA, % | 1985Q1 to 2026Q2 (166) |
| DRCCLACBS | delinquency rate on credit card loans, SA, % | 1991Q1 to 2026Q2 |
| UNRATE | unemployment rate, monthly, averaged to quarters | through 2026Q3 |

FRED has no October 2025 unemployment value, so 2025Q4 is the mean of November and December. SQL Server holds the three series in `fred.series_quarterly` (primary key on series and date, CHECK value >= 0, CHECK quarter start date) with a pivot view `fred.v_model_frame`. The loader reconciles row counts, sums, and first and last dates per series plus the model frame row count against pandas; all 13 checks match.

**Revision caveat.** FRED serves the latest vintage. Charge-off and delinquency data are revised, so a forecaster in 2009 saw slightly different history than this backtest does. Using ALFRED vintages would fix this; it is not done here.

## Eval protocol

* Rolling origin, expanding window, refit at every quarter. 82 evaluation origins (2005Q1 to 2025Q2), horizons 1 to 4. No random splits.
* Each model only receives rows dated at or before its origin. `tests/test_evaluate.py` checks this two ways: a spy records the last date every fit sees, and every value after an origin is overwritten with 1e6 to confirm the forecasts made at that origin do not change.
* Models: random walk; random walk with drift (the series is seasonally adjusted, so seasonal naive would equal the random walk); ARIMA with d from a KPSS test and p, q in 0 to 2 chosen by AICc on the training window only; ETS (simple vs damped trend, by AICc); regression with ARIMA errors on unemployment and delinquency (ARIMAX). ARIMAX uses one direct model per horizon with regressors lagged h quarters, so every value is known at the origin and no regressor is forecast. It shares the ARIMA order chosen at that origin, so the ARIMA vs ARIMAX comparison isolates the regressors.
* Accuracy: MASE (scaled by the mean absolute quarterly change from 1985Q1 to 2004Q4, 0.258 points) and RMSE by horizon.
* Significance: Diebold-Mariano on squared error with the Harvey, Leybourne and Newbold small-sample correction, at each horizon.
* Intervals: empirical coverage and Winkler score of the models' own 80% and 95% intervals vs split conformal intervals built from each model's past absolute errors at the same horizon (only errors whose targets were already observed at the origin; 28 warm-up origins from 1998Q1 to 2004Q4 calibrate the first ones).
* Regimes: accuracy recomputed for targets in 2008 to 2010 and 2020 to 2021.

## Results

**Lead and lag (quarterly changes).** Delinquency leads charge-offs by 1 quarter (r = 0.59), with 2 quarters close behind (r = 0.48). The planned 2-quarter lead is not the peak. Same-quarter unemployment changes correlate weakly with charge-off changes (r = 0.19), and no lag up to 8 quarters does better.

**Point accuracy (MASE, lower is better; 82 origins).**

| Model | h=1 | h=2 | h=3 | h=4 |
|---|---|---|---|---|
| random walk | 1.403 | 2.310 | 3.249 | 4.144 |
| drift | 1.411 | 2.337 | 3.289 | 4.229 |
| ARIMA | 1.416 | 2.162 | 3.029 | 3.912 |
| ETS | 1.381 | 2.202 | 3.083 | 4.036 |
| **ARIMAX** | **1.273** | **1.742** | **2.664** | **2.986** |

ARIMAX has the lowest MASE at every horizon (28% below the random walk at h=4), but its RMSE at h=1 is worse than the random walk's (0.685 vs 0.612) because of one large miss in 2020Q3, when the pandemic jump in unemployment pushed its forecast to 8% while charge-offs fell.

**Is the improvement significant? No, not at 5%.** Diebold-Mariano p-values, ARIMAX vs random walk: 0.70, 0.10, 0.27, 0.19 for h = 1 to 4. ARIMAX vs ARIMA (the test of the regressors): 0.76, 0.14, 0.11, 0.07. No other model beats the random walk either (all p > 0.48). With 82 overlapping origins dominated by two crises, the lower MASE is suggestive, not established.

**What would falsify "unemployment and delinquency help":** a DM p-value above 0.05 at every horizon for ARIMAX vs ARIMA. That is what happened, so the claim is **not supported** by this test. The h=4 p of 0.07 is the closest call.

**Intervals.** The models' own intervals are too narrow at longer horizons: random walk 80% intervals covered 69.5% of outcomes at h=4, and 95% intervals 82.9%. Conformal intervals came closer to nominal: 79.3% to 90.2% coverage at 80%, and 92.7% to 98.8% at 95%, across all models and horizons. At 95% conformal also had the better Winkler score for most model and horizon pairs (ARIMA h=4: 11.40 vs 15.26); at 80% the two were within about 5% of each other, because conformal buys its coverage with wider intervals.

**Regimes.** In 2008 to 2010 ARIMAX was far better one quarter ahead (MASE 1.19 vs 2.86 for the random walk) but overshot the 2010 peak at h=4. In 2020 to 2021 it was the worst model at h=1 (2.42 vs 1.23) because charge-offs fell while unemployment spiked. A model fit to normal recessions misread a recession with government income support.

**Next four quarters (ARIMAX from 2026Q2).** 3.64%, 3.59%, 3.62%, 3.58% for 2026Q3 to 2027Q2; 80% interval at 2027Q2 is 2.48% to 4.68%. See `outputs/fan_chart.png`.

**Stress scenario (illustrative scenario, not a CCAR submission).** Unemployment rises from 4.27% by 0.75 points a quarter to 7.27% in 2027Q2. A regression with ARIMA(2,0,2) errors on same-quarter unemployment puts the 2027Q2 charge-off rate at 3.96% vs 3.73% with unemployment flat: only 0.23 points higher, and the 95% intervals overlap almost entirely. This understates what happened in 2008 to 2010, when charge-offs rose about 6 points. The same-quarter link is weak in this data and the larger effects arrive through delinquency with a lag, so this simple scenario model should not be used for stress sizing.

## Files

* `src/chargeoff/data.py` download, quarterly shaping, long format for SQL
* `src/chargeoff/models.py` the five models
* `src/chargeoff/evaluate.py` backtest, DM test, Winkler score, conformal intervals, lead-lag
* `src/chargeoff/run.py` pipeline, stress scenario, charts
* `sql/01_schema.sql` table, constraints, pivot view; `scripts/load_sqlserver.py` load and reconcile
* `outputs/` metrics.json, accuracy_by_horizon.csv, dm_vs_random_walk.csv, intervals_by_horizon.csv, regime_accuracy.csv, lead_lag.csv, stress_scenario.csv, backtest_forecasts.csv, fan_chart.png, backtest_chart.png
* `docs/memo.md` one-page summary

## Authorship

Built with AI coding assistance (Claude Code). I set the scope and questions and reviewed the methods, code and results.
