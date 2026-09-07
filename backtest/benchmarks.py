"""Causal static expert benchmarks for the OOS return boundaries."""

from __future__ import annotations

import numpy as np


def static_expert_mix(
    close: np.ndarray,
    indices,
    signals: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    risk: float,
    vol_target: float,
    transaction_cost_bps: float,
    max_abs_position: float = 1.0,
) -> dict[str, list[float]]:
    returns: list[float] = []
    positions: list[float] = []
    turnovers: list[float] = []
    previous_position = 0.0
    cost_rate = transaction_cost_bps / 10_000.0
    for raw_t in indices:
        t = int(raw_t)
        raw_position = sum(
            weights[name] * signals[name][t - 1] for name in weights
        )
        recent = close[max(0, t - 21) : t]
        realised_vol = (
            float(np.std(np.diff(recent) / recent[:-1])) + 1e-9
            if len(recent) > 2
            else vol_target
        )
        risk_scale = float(np.clip(vol_target / realised_vol, 0.2, 3.0))
        position = float(np.tanh(raw_position)) * risk * risk_scale
        position = float(
            np.clip(position, -max_abs_position, max_abs_position)
        )
        turnover = abs(position - previous_position)
        asset_return = close[t] / close[t - 1] - 1.0
        returns.append(position * asset_return - turnover * cost_rate)
        positions.append(position)
        turnovers.append(turnover)
        previous_position = position
    return {
        "returns": returns,
        "positions": positions,
        "turnovers": turnovers,
    }
