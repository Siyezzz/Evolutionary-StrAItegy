"""Long-horizon time-series momentum challenger."""

from __future__ import annotations

import numpy as np

from .base import Expert


class LongHorizonMomentumExpert(Expert):
    name = "long_horizon_momentum"

    def __init__(self, lookback: int = 252, scale: float = 0.20):
        self.lookback = lookback
        self.scale = scale

    def signal(self, ctx: dict) -> float:
        t = ctx["t"]
        close = ctx["close"]
        if t < self.lookback:
            return 0.0
        trailing_return = close[t] / close[t - self.lookback] - 1.0
        return float(np.tanh(trailing_return / self.scale))
