"""Long/cash trend filter based on the trailing 200-day mean."""

from __future__ import annotations

import numpy as np

from .base import Expert


class LongOnlyTrendExpert(Expert):
    name = "long_only_trend"

    def __init__(self, lookback: int = 200):
        self.lookback = lookback

    def signal(self, ctx: dict) -> float:
        t = ctx["t"]
        close = ctx["close"]
        if t < self.lookback:
            return 0.0
        trailing_mean = float(np.mean(close[t - self.lookback + 1 : t + 1]))
        return 1.0 if close[t] > trailing_mean else 0.0
