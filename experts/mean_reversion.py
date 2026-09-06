"""Mean-reversion expert — fade excursions from the recent average.

Signal = -tanh of the normalised deviation of ``close[t]`` from its trailing
mean. Causal: only the window ending at ``t`` is used.
"""

from __future__ import annotations

import numpy as np

from .base import Expert


class MeanReversionExpert(Expert):
    name = "mean_reversion"

    def __init__(self, lookback: int = 15, scale: float = 0.10):
        self.lookback = lookback
        self.scale = scale

    def signal(self, ctx: dict) -> float:
        t = ctx["t"]
        close = ctx["close"]
        if t < self.lookback:
            return 0.0
        window = close[t - self.lookback + 1 : t + 1]
        mean = float(np.mean(window))
        dev = (close[t] - mean) / (mean + 1e-9)
        return float(-np.tanh(dev / self.scale))
