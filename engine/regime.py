"""Market-regime detector.

A regime is a coarse description of the recent market state
(bull / bear / choppy / volatile). It is used to (a) tag which kind of
environment an expert was successful in, and (b) feed the opponent model.

All logic is *causal*: it only ever reads prices up to and including time ``t``.
"""

from __future__ import annotations

import numpy as np


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
