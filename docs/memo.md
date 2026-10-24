# Memo: credit card charge-offs, next four quarters

**To:** credit risk lead | **Data through:** 2026Q2 (FRED, latest vintage)

**Bottom line.** Expect the credit card charge-off rate to stay near its current 3.82%: the forecast is 3.64% for 2026Q3 and 3.58% for 2027Q2. The honest range is wide. The 80% interval for 2027Q2 runs from 2.48% to 4.68%.

**How much to trust it.** I backtested five models from 82 quarterly forecast origins (2005Q1 to 2025Q2), refitting each quarter on only the data available then. A model using lagged unemployment and delinquency had the lowest scaled error at every horizon, 28% below a no-change forecast four quarters out. That gain is **not statistically significant** (Diebold-Mariano p = 0.07 to 0.76), so treat it as a lean, not a finding. The same model badly misread 2020, when unemployment spiked and charge-offs fell.

**Intervals.** The standard model intervals are too narrow: a no-change forecast's 80% interval covered only 69.5% of outcomes four quarters out. Intervals calibrated on past forecast errors covered 79% to 90%. Use the calibrated ones for planning.

**Early warning.** Delinquency changes lead charge-off changes by about one quarter (correlation 0.59). Watch the delinquency rate (2.85% in 2026Q2) as the first signal.

**Stress view (illustrative scenario, not a CCAR submission).** If unemployment rose 3 points to 7.27% by 2027Q2, a same-quarter unemployment model adds only 0.23 points (3.96% vs 3.73%). That is too small to be credible next to 2008 to 2010, when charge-offs rose about 6 points; the model misses lagged effects, and I would not size reserves with it.

**Caveat.** Charge-off data are revised; this backtest uses today's vintage, not what was known at each origin.
