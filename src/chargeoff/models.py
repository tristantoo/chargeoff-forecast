"""Forecasting models. Each takes the history known at the origin and returns an h-row frame
(index 1..h) with columns mean, lo80, hi80, lo95, hi95."""
import itertools
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import kpss

Z80, Z95 = norm.ppf(0.9), norm.ppf(0.975)
COLS = ["mean", "lo80", "hi80", "lo95", "hi95"]


def _frame(mean, se) -> pd.DataFrame:
    mean, se = np.asarray(mean, float), np.asarray(se, float)
    return pd.DataFrame({"mean": mean, "lo80": mean - Z80 * se, "hi80": mean + Z80 * se,
                         "lo95": mean - Z95 * se, "hi95": mean + Z95 * se}, index=range(1, len(mean) + 1))


def random_walk(hist: pd.DataFrame, h: int) -> pd.DataFrame:
    y = hist["y"].dropna()
    sigma = y.diff().std()
    k = np.arange(1, h + 1)
    return _frame(np.repeat(y.iloc[-1], h), sigma * np.sqrt(k))


def drift(hist: pd.DataFrame, h: int) -> pd.DataFrame:
    # The series is seasonally adjusted, so seasonal naive equals the random walk; drift is the second benchmark.
    y = hist["y"].dropna()
    dy = y.diff().dropna()
    k = np.arange(1, h + 1)
    return _frame(y.iloc[-1] + k * dy.mean(), dy.std() * np.sqrt(k))


def choose_d(y: pd.Series) -> int:
    """KPSS level-stationarity test at 5%, as auto.arima does: difference once if rejected."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return int(kpss(y, regression="c", nlags="auto")[1] < 0.05)


def select_order(y: pd.Series) -> tuple:
    """(p, d, q) with p, q in 0..2 minimising AICc for the KPSS-chosen d (AICc is not comparable across d)."""
    d = choose_d(y)
    best, best_aicc = None, np.inf
    for p, q in itertools.product(range(3), range(3)):
        fit = _sarimax(y, None, (p, d, q))
        if fit is not None and fit.aicc < best_aicc:
            best, best_aicc = (p, d, q), fit.aicc
    return best


def _sarimax(y, x, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return SARIMAX(y.to_numpy(), exog=None if x is None else x.to_numpy(), order=order,
                           trend="c" if order[1] == 0 else "n").fit(disp=False)
        except (np.linalg.LinAlgError, ValueError):
            return None


def _pred(fit, h, exog=None) -> pd.DataFrame:
    p = fit.get_forecast(h, exog=exog)
    return _frame(p.predicted_mean, np.sqrt(p.var_pred_mean))


def arima(hist: pd.DataFrame, h: int, order: tuple | None = None) -> pd.DataFrame:
    y = hist["y"].dropna()
    return _pred(_sarimax(y, None, order or select_order(y)), h)


def ets(hist: pd.DataFrame, h: int) -> pd.DataFrame:
    """Additive-error ETS: simple vs damped additive trend, chosen by AICc."""
    y = pd.Series(hist["y"].dropna().to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [ETSModel(y, error="add", trend=None).fit(disp=False),
                ETSModel(y, error="add", trend="add", damped_trend=True).fit(disp=False)]
    fit = min(fits, key=lambda f: f.aicc)
    sf = fit.get_prediction(start=len(y), end=len(y) + h - 1).summary_frame(alpha=0.05)
    se = (sf["pi_upper"] - sf["mean"]).to_numpy() / Z95
    return _frame(sf["mean"].to_numpy(), se)


def arimax(hist: pd.DataFrame, h: int, order: tuple | None = None) -> pd.DataFrame:
    """Regression with ARIMA errors on unemployment and delinquency, one direct model per horizon.
    For horizon k the regressors are lagged k quarters, so every value used is known at the origin
    and no regressor has to be forecast."""
    order = order or select_order(hist["y"].dropna())
    x = hist[["ur", "dq"]]
    rows = []
    for k in range(1, h + 1):
        xk = x.shift(k)
        ok = xk.notna().all(axis=1) & hist["y"].notna()
        fit = _sarimax(hist["y"][ok], xk[ok], order)
        fut = x.iloc[len(x) - k:].to_numpy()  # x[t+1-k .. t] feeds steps 1..k
        rows.append(_pred(fit, k, exog=fut).iloc[[-1]])
    out = pd.concat(rows)
    out.index = range(1, h + 1)
    return out


def univariate_and_x(hist: pd.DataFrame, h: int) -> dict:
    """All models at one origin, sharing one ARIMA order so the ARIMAX comparison isolates the regressors."""
    order = select_order(hist["y"].dropna())
    return {"random_walk": random_walk(hist, h), "drift": drift(hist, h), "ets": ets(hist, h),
            "arima": arima(hist, h, order), "arimax": arimax(hist, h, order)}
