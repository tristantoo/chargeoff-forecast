"""Download and shape the three FRED series into one quarterly frame."""
import urllib.request
from pathlib import Path

import pandas as pd

SERIES = {"CORCCACBS": "y", "DRCCLACBS": "dq", "UNRATE": "ur"}
URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"


def download(raw: Path) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    for sid in SERIES:
        urllib.request.urlretrieve(URL.format(sid), raw / f"{sid}.csv")


def read_series(raw: Path, sid: str) -> pd.Series:
    df = pd.read_csv(raw / f"{sid}.csv", parse_dates=["observation_date"], na_values=".")
    s = df.set_index("observation_date")[sid].astype(float).dropna()
    if sid == "UNRATE":  # monthly: average quarters with at least 2 months (FRED has no October 2025 value)
        q = s.groupby(s.index.to_period("Q"))
        s = q.mean()[q.count() >= 2]
    else:
        s.index = s.index.to_period("Q")
    if (s < 0).any():
        raise ValueError(f"{sid} has negative values")
    return s.rename(SERIES[sid])


def load(raw: Path) -> pd.DataFrame:
    """Quarterly frame indexed by Period, columns y (charge-off %), dq (delinquency %), ur (unemployment %).
    Starts where charge-offs start (1985Q1) and ends at the last charge-off quarter."""
    df = pd.concat([read_series(raw, s) for s in SERIES], axis=1)
    y = df["y"].dropna()
    return df.loc[y.index.min(): y.index.max()]


def long_format(raw: Path) -> pd.DataFrame:
    """(obs_date, series_id, value) rows for SQL Server, quarters dated on their first day."""
    rows = []
    for sid in SERIES:
        s = read_series(raw, sid)
        rows.append(pd.DataFrame({"obs_date": s.index.to_timestamp().date, "series_id": sid, "value": s.values}))
    return pd.concat(rows, ignore_index=True)
