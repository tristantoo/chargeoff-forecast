"""End-to-end: data -> rolling-origin backtest -> metrics, tables, charts, stress scenario.

uv run python -m chargeoff.run            # uses data/raw, downloads if missing
"""
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from chargeoff import data, evaluate, models  # noqa: E402

WARMUP_START = pd.Period("1998Q1")  # extra origins whose errors calibrate the conformal intervals
EVAL_START = pd.Period("2005Q1")
SCALE_END = pd.Period("2004Q4")     # MASE scale: mean |quarterly change| through the first eval origin
H = 4
REGIMES = {"2008_2010": (2008, 2010), "2020_2021": (2020, 2021)}


def stress(df: pd.DataFrame, rise: float = 3.0) -> pd.DataFrame:
    """Charge-off path conditional on unemployment: regression with ARIMA errors on same-quarter
    unemployment (as bank stress models condition on same-quarter macro paths). Baseline holds
    unemployment flat at its last value; stress raises it linearly by `rise` points over 4 quarters."""
    y, x = df["y"], df[["ur"]]
    order = models.select_order(y)
    fit = models._sarimax(y, x, order)
    last = x["ur"].iloc[-1]
    t = df.index[-1]
    out = []
    for name, path in [("baseline", np.repeat(last, H)), ("stress", last + rise * np.arange(1, H + 1) / H)]:
        fc = models._pred(fit, H, exog=path.reshape(-1, 1))
        out.append(fc.assign(scenario=name, quarter=[str(t + k) for k in range(1, H + 1)], unemployment=path))
    res = pd.concat(out)
    res.attrs["coef_ur"] = float(fit.params[1 if order[1] == 0 else 0])
    res.attrs["order"] = order
    return res


def main(argv=None) -> None:
    root = Path(argv[0]) if argv else Path(".")
    raw, outd = root / "data" / "raw", root / "outputs"
    outd.mkdir(exist_ok=True)
    if not all((raw / f"{s}.csv").exists() for s in data.SERIES):
        data.download(raw)
    df = data.load(raw)
    last = df.index[-1]

    lead_dq = evaluate.cross_corr(df["dq"], df["y"])
    lead_ur = evaluate.cross_corr(df["ur"], df["y"])
    pd.concat([lead_dq.assign(leader="delinquency"), lead_ur.assign(leader="unemployment")]).to_csv(
        outd / "lead_lag.csv", index=False)

    origins = pd.period_range(WARMUP_START, last - H, freq="Q")
    bt = evaluate.add_conformal(evaluate.backtest(df, models.univariate_and_x, origins, H))
    bt.to_csv(outd / "backtest_forecasts.csv", index=False)
    ev = bt[bt["origin"] >= EVAL_START]

    scale = df["y"].loc[:SCALE_END].diff().abs().mean()
    acc = evaluate.accuracy(ev, scale)
    ints = evaluate.intervals(ev)
    dm = pd.concat([evaluate.dm_table(ev),
                    evaluate.dm_table(ev[ev["model"].isin(["arima", "arimax"])], base="arima")])
    reg = pd.concat([evaluate.accuracy(ev[ev["target"].map(lambda p: a <= p.year <= b)], scale).assign(regime=r)
                     for r, (a, b) in REGIMES.items()])
    acc.to_csv(outd / "accuracy_by_horizon.csv", index=False)
    ints.to_csv(outd / "intervals_by_horizon.csv", index=False)
    dm.to_csv(outd / "dm_vs_random_walk.csv", index=False)
    reg.to_csv(outd / "regime_accuracy.csv", index=False)

    best = acc.groupby("model")["MASE"].mean().idxmin()
    final = models.univariate_and_x(df, H)
    st = stress(df)
    st.to_csv(outd / "stress_scenario.csv", index=False)

    ax_dm = dm[(dm["model"] == "arimax") & (dm["vs"] == "arima")].set_index("h")["p"]
    metrics = {
        "protocol": {
            "data": "FRED CORCCACBS, DRCCLACBS, UNRATE (quarterly mean), latest vintage",
            "quarters": [str(df.index[0]), str(last)], "n_quarters": len(df),
            "eval_origins": [str(EVAL_START), str(last - H)], "n_eval_origins": int(ev["origin"].nunique()),
            "warmup_origins_for_conformal": [str(WARMUP_START), str(EVAL_START - 1)],
            "horizons": list(range(1, H + 1)), "refit": "every origin, expanding window",
            "mase_scale": f"mean |diff| of charge-off rate {df.index[0]} to {SCALE_END} = {scale:.4f}",
        },
        "lead_lag_best": {
            "delinquency_leads_by": int(lead_dq.loc[lead_dq["corr"].idxmax(), "lead_k"]),
            "delinquency_corr": float(lead_dq["corr"].max()),
            "unemployment_leads_by": int(lead_ur.loc[lead_ur["corr"].idxmax(), "lead_k"]),
            "unemployment_corr": float(lead_ur["corr"].max()),
        },
        "best_model_by_mean_mase": best,
        "accuracy": acc.to_dict("records"),
        "dm_vs_random_walk": dm.to_dict("records"),
        "regressors_help_dm_p_arimax_vs_arima": ax_dm.to_dict(),
        "intervals": ints.to_dict("records"),
        "final_forecast": {m: fc.assign(quarter=[str(last + k) for k in fc.index]).to_dict("records")
                           for m, fc in final.items()},
        "stress": {"label": "illustrative scenario, not a CCAR submission", "arima_order": list(st.attrs["order"]),
                   "coef_unemployment": st.attrs["coef_ur"], "paths": st.to_dict("records")},
    }
    (outd / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    plot_backtest(df, ev, best, outd / "backtest_chart.png")
    plot_fan(df, final[best], bt[(bt["model"] == best)], last, best, outd / "fan_chart.png")
    print(acc.pivot(index="model", columns="h", values="MASE").round(3))
    print(dm.pivot(index=["model", "vs"], columns="h", values="p").round(3))
    print(ints[["model", "h", "model_cov80", "conformal_cov80", "model_cov95", "conformal_cov95"]].round(3))
    print("best:", best)


def plot_backtest(df, ev, best, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    y = df["y"].loc[EVAL_START:]
    for ax, k in zip(axes, [1, 4]):
        ax.plot(y.index.to_timestamp(), y.values, color="black", lw=1.5, label="actual")
        for m, c in [("random_walk", "#9a9a9a"), (best, "#1f6fb2")]:
            g = ev[(ev["model"] == m) & (ev["h"] == k)]
            ax.plot(g["target"].map(lambda p: p.to_timestamp()), g["mean"], color=c, lw=1, label=f"{m}, h={k}")
        ax.set_ylabel("charge-off rate, %")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Rolling-origin forecasts vs actual, credit card charge-off rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_fan(df, fc, bt_best, last, best, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    y = df["y"].loc[last - 40:]
    ax.plot(y.index.to_timestamp(), y.values, color="black", lw=1.5, label="actual")
    q = [(last + k).to_timestamp() for k in fc.index]
    ax.fill_between(q, fc["lo95"], fc["hi95"], color="#1f6fb2", alpha=0.15, label="95% model interval")
    ax.fill_between(q, fc["lo80"], fc["hi80"], color="#1f6fb2", alpha=0.3, label="80% model interval")
    ax.plot(q, fc["mean"], color="#1f6fb2", lw=1.5, label=f"{best} forecast")
    ax.set_title(f"Next {len(fc)} quarters from {last}")
    ax.set_ylabel("charge-off rate, %")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1:])
