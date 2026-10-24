"""Load the quarterly FRED series into SQL Server and reconcile against pandas.

uv run python scripts/load_sqlserver.py            # LocalDB, database "chargeoff"
Set CHARGEOFF_SQL_CONN to use another server (any ODBC connection string, without DATABASE=).
Exit code 1 if SQL and pandas disagree on any reconciled number.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

import pyodbc

from chargeoff import data

ROOT = Path(__file__).resolve().parents[1]


def connect(db: str | None = None) -> pyodbc.Connection:
    conn = os.environ.get("CHARGEOFF_SQL_CONN")
    if not conn:
        # The legacy "SQL Server" ODBC driver cannot resolve (localdb)\..., so use the instance pipe.
        subprocess.run(["sqllocaldb", "start", "MSSQLLocalDB"], capture_output=True)
        info = subprocess.run(["sqllocaldb", "info", "MSSQLLocalDB"], capture_output=True, text=True).stdout
        conn = f"DRIVER={{SQL Server}};SERVER={re.search(r'(np:\S+)', info).group(1)};Trusted_Connection=yes;"
    return pyodbc.connect(conn + (f"DATABASE={db};" if db else ""), autocommit=True)


def main(db: str = "chargeoff") -> int:
    cx = connect()
    cx.execute(f"IF DB_ID('{db}') IS NULL CREATE DATABASE {db}")
    cx = connect(db)
    for batch in re.split(r"^\s*GO\s*$", (ROOT / "sql" / "01_schema.sql").read_text(), flags=re.M | re.I):
        if batch.strip():
            cx.execute(batch)
    long = data.long_format(ROOT / "data" / "raw")
    cur = cx.cursor()
    cur.fast_executemany = True
    cur.executemany("INSERT fred.series_quarterly (obs_date, series_id, value) VALUES (?, ?, ?)",
                    [(d, s, round(float(v), 4)) for d, s, v in long.itertuples(index=False)])

    ok = True
    for sid, g in long.groupby("series_id"):
        n, total, lo, hi = cx.execute("SELECT COUNT(*), SUM(value), MIN(obs_date), MAX(obs_date) "
                                      "FROM fred.series_quarterly WHERE series_id = ?", sid).fetchone()
        checks = [("rows", n, len(g)), ("sum", round(float(total), 2), round(g["value"].round(4).sum(), 2)),
                  ("first", str(lo), str(g["obs_date"].min())), ("last", str(hi), str(g["obs_date"].max()))]
        for name, sql_v, pd_v in checks:
            match = sql_v == pd_v
            ok &= match
            print(f"{sid:10s} {name:6s} sql={sql_v} pandas={pd_v} {'OK' if match else 'MISMATCH'}")
    n_frame = cx.execute("SELECT COUNT(*) FROM fred.v_model_frame WHERE chargeoff_rate IS NOT NULL").fetchone()[0]
    n_pd = len(data.load(ROOT / "data" / "raw"))
    ok &= n_frame == n_pd
    print(f"model frame rows sql={n_frame} pandas={n_pd} {'OK' if n_frame == n_pd else 'MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
