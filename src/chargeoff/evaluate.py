"""Rolling-origin backtest, accuracy, Diebold-Mariano, interval scores, and split conformal intervals."""
import math

import numpy as np
import pandas as pd
from scipy.stats import t as tdist


def backtest(df: pd.DataFrame, fit_all, origins, h: int = 4) -> pd.DataFrame:
    """At each origin t, pass only rows <= t to fit_all(hist, h) -> {model: frame}; score t+1..t+h."""
    rows = []
    for t in origins:
        hist = df.loc[:t]
        fcs = fit_all(hist, h)
        for k in range(1, h + 1):
            target = t + k
            actual = df["y"].get(target, np.nan)
            for name, fc in fcs.items():
                rows.append({"origin": t, "h": k, "target": target, "model": name, "actual": actual,
                             **fc.loc[k].to_dict()})
    return pd.DataFrame(rows)


def dm_test(e1, e2, h: int) -> tuple[float, float]:
    """Diebold-Mariano on squared-error loss with the Harvey, Leybourne and Newbold correction.
    Negative statistic means model 1 has lower loss. Returns (statistic, two-sided p)."""
    d = np.asarray(e1, float) ** 2 - np.asarray(e2, float) ** 2
    n = len(d)
    dc = d - d.mean()
    gamma = [np.sum(dc[k:] * dc[:n - k]) / n for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:
        var = gamma[0] / n
    stat = d.mean() / math.sqrt(var) * math.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return stat, 2 * tdist.sf(abs(stat), n - 1)


def winkler(lo, hi, y, alpha: float) -> np.ndarray:
    lo, hi, y = map(lambda a: np.asarray(a, float), (lo, hi, y))
    return (hi - lo) + (2 / alpha) * (np.maximum(lo - y, 0) + np.maximum(y - hi, 0))


def conformal_q(abs_resid: np.ndarray, alpha: float) -> float:
    n = len(abs_resid)
    k = math.ceil((n + 1) * (1 - alpha))
    return np.inf if k > n else float(np.sort(abs_resid)[k - 1])


def add_conformal(bt: pd.DataFrame, min_n: int = 8) -> pd.DataFrame:
    """Split conformal intervals: at each origin, use |errors| of the same model and horizon whose
    targets were already observed at that origin (target <= origin). Expanding window."""
    bt = bt.sort_values(["model", "h", "origin"]).copy()
    for c in ["clo80", "chi80", "clo95", "chi95"]:
        bt[c] = np.nan
    for (_, _), g in bt.groupby(["model", "h"]):
        err = (g["actual"] - g["mean"]).abs()
        for i, row in g.iterrows():
            past = err[(g["target"] <= row["origin"]) & err.notna()].to_numpy()
            if len(past) < min_n:
                continue
            q80, q95 = conformal_q(past, 0.2), conformal_q(past, 0.05)
            bt.loc[i, ["clo80", "chi80", "clo95", "chi95"]] = [row["mean"] - q80, row["mean"] + q80,
                                                               row["mean"] - q95, row["mean"] + q95]
    return bt.sort_index()


def accuracy(bt: pd.DataFrame, scale: float) -> pd.DataFrame:
    e = bt.assign(err=bt["actual"] - bt["mean"])
    return e.groupby(["model", "h"]).agg(n=("err", "size"),
                                         MASE=("err", lambda x: x.abs().mean() / scale),
                                         RMSE=("err", lambda x: math.sqrt((x ** 2).mean()))).reset_index()


def intervals(bt: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (m, k), g in bt.groupby(["model", "h"]):
        y = g["actual"]
        r = {"model": m, "h": k, "n": len(g)}
        for lvl, a in [(80, 0.2), (95, 0.05)]:
            for kind, lo, hi in [("model", f"lo{lvl}", f"hi{lvl}"), ("conformal", f"clo{lvl}", f"chi{lvl}")]:
                r[f"{kind}_cov{lvl}"] = ((y >= g[lo]) & (y <= g[hi])).mean()
                r[f"{kind}_winkler{lvl}"] = winkler(g[lo], g[hi], y, a).mean()
                r[f"{kind}_width{lvl}"] = (g[hi] - g[lo]).mean()
        out.append(r)
    return pd.DataFrame(out)


def dm_table(bt: pd.DataFrame, base: str = "random_walk") -> pd.DataFrame:
    rows = []
    for k, g in bt.groupby("h"):
        w = g.pivot(index="origin", columns="model", values="mean")
        e = g.drop_duplicates("origin").set_index("origin")["actual"].to_frame().join(w)
        for m in w.columns.drop(base):
            stat, p = dm_test(e["actual"] - e[m], e["actual"] - e[base], k)
            rows.append({"model": m, "vs": base, "h": k, "dm_stat": stat, "p": p})
    return pd.DataFrame(rows)


def cross_corr(a: pd.Series, b: pd.Series, max_lag: int = 8) -> pd.DataFrame:
    """corr(diff a at t-k, diff b at t) for k = 0..max_lag: a leads b by k quarters."""
    da, db = a.diff(), b.diff()
    rows = []
    for k in range(max_lag + 1):
        pair = pd.concat([da.shift(k), db], axis=1).dropna()
        rows.append({"lead_k": k, "corr": pair.iloc[:, 0].corr(pair.iloc[:, 1]), "n": len(pair)})
    return pd.DataFrame(rows)
