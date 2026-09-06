"""Opponent model — inferring the "other players" from market microstructure.

In a self-evolving trading game the *opponent* is the aggregate of other
participants (crowd, market makers, macro flows). We cannot see their orders
directly, so we reconstruct a coarse opponent state purely from public, *past*
price/volume — strictly causal.

State fields (all derived from data up to ``t`` only):
    trend      — where the crowd is currently pushing the price (tanh-squashed)
    aggression — ratio of recent vol to longer-term vol (>1 = crowded/frantic)
    vol        — recent realised volatility
"""

from __future__ import annotations

import numpy as np


def opponent_state(close: np.ndarray, volume: np.ndarray, t: int, lookback: int = 20) -> dict:
    lo = max(0, t - lookback)
    window = close[lo : t + 1]
    if len(window) < 2:
        return {"trend": 0.0, "aggression": 0.0, "vol": 0.0}

    rets = np.diff(window) / window[:-1]
    trend = window[-1] / window[0] - 1.0
    vol = float(np.std(rets))

    long_lo = max(0, t - 60)
    long_window = close[long_lo : t + 1]
    if len(long_window) > 2:
        long_rets = np.diff(long_window) / long_window[:-1]
        baseline_vol = float(np.std(long_rets)) + 1e-9
    else:
        baseline_vol = vol + 1e-9

    aggression = float(np.clip(vol / baseline_vol, 0.0, 3.0))
    return {
        "trend": float(np.tanh(trend * 5.0)),
        "aggression": aggression,
        "vol": vol,
    }
