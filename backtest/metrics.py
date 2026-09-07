"""Performance metrics shared by strategies and benchmarks."""

from __future__ import annotations

import numpy as np


def performance_metrics(
    returns,
    *,
    periods_per_year: int = 365,
    positions=None,
    turnovers=None,
) -> dict:
    arr = np.asarray(returns, dtype=float)
    if not len(arr):
        return {
            "steps": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "max_drawdown": 0.0,
            "average_abs_exposure": 0.0,
            "annualized_turnover": 0.0,
        }

    equity = np.cumprod(1.0 + arr)
    with_initial = np.concatenate(([1.0], equity))
    peak = np.maximum.accumulate(with_initial)
    years = len(arr) / float(periods_per_year)
    annualized_return = (
        float(equity[-1] ** (1.0 / years) - 1.0)
        if years > 0 and equity[-1] > 0
        else -1.0
    )
    annualized_vol = float(arr.std() * np.sqrt(periods_per_year))
    sharpe = (
        float(arr.mean() / (arr.std() + 1e-12) * np.sqrt(periods_per_year))
        if len(arr) > 1
        else 0.0
    )
    downside = np.minimum(arr, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        float(arr.mean() / (downside_deviation + 1e-12) * np.sqrt(periods_per_year))
        if len(arr) > 1
        else 0.0
    )
    pos = np.asarray(positions, dtype=float) if positions is not None else np.array([])
    turn = np.asarray(turnovers, dtype=float) if turnovers is not None else np.array([])
    return {
        "steps": int(len(arr)),
        "total_return": float(equity[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(((with_initial - peak) / peak).min()),
        "average_abs_exposure": float(np.mean(np.abs(pos))) if len(pos) else 0.0,
        "annualized_turnover": (
            float(np.mean(turn) * periods_per_year) if len(turn) else 0.0
        ),
    }


def metrics_by_context(returns, contexts, *, periods_per_year: int) -> dict:
    arr = np.asarray(returns, dtype=float)
    labels = np.asarray(contexts)
    if len(arr) != len(labels):
        raise ValueError("returns and contexts must have equal length")
    return {
        str(label): performance_metrics(
            arr[labels == label], periods_per_year=periods_per_year
        )
        for label in sorted(set(labels.tolist()))
    }
