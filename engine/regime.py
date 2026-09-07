"""Market-regime detector.

A regime is a coarse description of the recent market state
(bull / bear / choppy / volatile). It is used to (a) tag which kind of
environment an expert was successful in, and (b) feed the opponent model.

All logic is *causal*: it only ever reads prices up to and including time ``t``.
"""

from __future__ import annotations

import numpy as np

from data.opponent_model import opponent_state


def detect_regime(close: np.ndarray, t: int, lookback: int = 60) -> str:
    """Return a coarse regime label for the window ending at ``t``.

    Parameters
    ----------
    close : np.ndarray
        Full close-price array (the function only reads ``close[max(0,t-lookback):t+1]``).
    t : int
        Current index. Signal/decision is made *using* information up to ``t``.
    """
    lo = max(0, t - lookback)
    window = close[lo : t + 1]
    if len(window) < 5:
        return "undefined"

    rets = np.diff(window) / window[:-1]
    trend = window[-1] / window[0] - 1.0
    vol = float(np.std(rets))

    if vol > 0.03:
        return "volatile"
    if trend > 0.10:
        return "bull"
    if trend < -0.10:
        return "bear"
    return "choppy"


def detect_context(
    close: np.ndarray, volume: np.ndarray, t: int
) -> str:
    """Map causal opponent features to a predeclared cross-asset context."""
    state = opponent_state(close, volume, t, lookback=20)
    if state["aggression"] >= 1.25:
        return "high_relative_volatility"
    if state["trend"] >= 0.25:
        return "uptrend"
    if state["trend"] <= -0.25:
        return "downtrend"
    return "neutral"


def precompute_contexts(
    close: np.ndarray,
    volume: np.ndarray,
    *,
    method: str = "fixed",
    calibration_window: int = 252,
) -> np.ndarray:
    """Compute causal contexts once, independently of strategy parameters."""
    if method == "fixed":
        return np.array(
            [detect_context(close, volume, t) for t in range(len(close))],
            dtype=object,
        )
    if method != "ranked":
        raise ValueError("context_method must be 'fixed' or 'ranked'")

    states = [opponent_state(close, volume, t, lookback=20) for t in range(len(close))]
    trends = np.array([state["trend"] for state in states], dtype=float)
    aggression = np.array([state["aggression"] for state in states], dtype=float)
    contexts = np.full(len(close), "neutral", dtype=object)
    minimum_history = min(60, calibration_window)
    for t in range(len(close)):
        lo = max(0, t - calibration_window)
        if t - lo < minimum_history:
            continue
        past_trend = trends[lo:t]
        past_aggression = aggression[lo:t]
        if aggression[t] >= np.quantile(past_aggression, 0.75):
            contexts[t] = "high_relative_volatility"
        elif trends[t] >= np.quantile(past_trend, 2.0 / 3.0):
            contexts[t] = "uptrend"
        elif trends[t] <= np.quantile(past_trend, 1.0 / 3.0):
            contexts[t] = "downtrend"
    return contexts
