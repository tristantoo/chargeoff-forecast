import math

import numpy as np
import pandas as pd
import pytest

from chargeoff import evaluate, models


def synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.period_range("2000Q1", periods=n, freq="Q")
    ur = 5 + np.cumsum(rng.normal(0, 0.2, n))
    dq = 3 + 0.5 * (ur - 5) + rng.normal(0, 0.1, n)
    y = 4 + 0.8 * np.r_[dq[:2], dq[:-2]] - 2.4 + rng.normal(0, 0.1, n)
    return pd.DataFrame({"y": y, "dq": dq, "ur": ur}, index=idx)


def test_no_future_data_used():
    df = synthetic()
    origins = df.index[40:44]
    seen = []

    def spy(hist, h):
        seen.append(hist.index.max())
        return models.univariate_and_x(hist, h)

    bt = evaluate.backtest(df, spy, origins, h=4)
    assert all(s <= o for s, o in zip(seen, origins))
    # Poison everything after one origin: forecasts made at that origin must not change.
    t = origins[0]
    poisoned = df.copy()
    poisoned.loc[t + 1:, ["y", "dq", "ur"]] = 1e6
    bt2 = evaluate.backtest(poisoned, models.univariate_and_x, [t], h=4)
    a = bt[bt["origin"] == t].sort_values(["model", "h"])["mean"].to_numpy()
    b = bt2.sort_values(["model", "h"])["mean"].to_numpy()
    np.testing.assert_allclose(a, b)


def test_dm_hand_case():
    stat, p = evaluate.dm_test([1, 2, 3, 4], [1, 1, 1, 1], h=1)
    # d = [0, 3, 8, 15], mean 6.5, gamma0 = 32.25, DM = 6.5 / sqrt(32.25 / 4), HLN factor sqrt(3 / 4)
    assert stat == pytest.approx(6.5 / math.sqrt(32.25 / 4) * math.sqrt(0.75))
    assert 0 < p < 1


def test_winkler_and_conformal():
    w = evaluate.winkler([1, 1, 1], [3, 3, 3], [4, 2, 0], alpha=0.2)
    np.testing.assert_allclose(w, [12, 2, 12])
    r = np.arange(1, 10, dtype=float)               # n = 9
    assert evaluate.conformal_q(r, 0.2) == 8         # k = ceil(10 * 0.8) = 8
    assert evaluate.conformal_q(r, 0.05) == np.inf   # k = 10 > n


def test_conformal_uses_only_observed_targets():
    df = synthetic()
    bt = evaluate.add_conformal(evaluate.backtest(df, models.univariate_and_x, df.index[30:50], h=2), min_n=3)
    g = bt[(bt["model"] == "random_walk") & (bt["h"] == 2)].sort_values("origin")
    # first origins have fewer than 3 observed past errors, so no interval yet
    assert g["clo80"].isna().iloc[:4].all() and g["clo80"].notna().iloc[-1]
