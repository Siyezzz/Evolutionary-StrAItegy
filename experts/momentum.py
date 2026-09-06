"""Momentum expert — ride the trend.

Signal = tanh of the recent price change over ``lookback`` days. Purely causal:
it only compares ``close[t]`` with ``close[t-lookback]``.
"""

from __future__ import annotations

import numpy as np

from .base import Expert


class MomentumExpert(Expert):
    name = "momentum"

    def __init__(self, lookback: int = 20, scale: float = 0.15):
        self.lookback = lookback
        self.scale = scale

    def signal(self, ctx: dict) -> float:
        t = ctx["t"]
        close = ctx["close"]
        if t < self.lookback:
            return 0.0
        mom = close[t] / close[t - self.lookback] - 1.0
        return float(np.tanh(mom / self.scale))
