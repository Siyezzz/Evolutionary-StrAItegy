"""News-sentiment expert — act on conviction spikes.

In live trading this expert would query an NLP news/social feed and return a
sentiment score in ``[-1, 1]``. For the offline historical backtest we use a
*deterministic, causal* stand-in: the latest return amplified by a volume-surge
ratio (a volume spike on a move is the closest offline proxy for a news shock).
This keeps the module honest — it never reads the future — while preserving the
role of a sentiment-driven skill in the gene pool.

Swap ``_sentiment_proxy`` for a real ``fetch_sentiment(date)`` call in
``data/fetch_news.py`` to go live.
"""

from __future__ import annotations

import numpy as np

from .base import Expert


class NewsSentimentExpert(Expert):
    name = "news_sentiment"

    def __init__(self, lookback: int = 5, scale: float = 0.08):
        self.lookback = lookback
        self.scale = scale

    def signal(self, ctx: dict) -> float:
        t = ctx["t"]
        close = ctx["close"]
        volume = ctx.get("volume")
        if t < 2:
            return 0.0
        ret = close[t] / close[t - 1] - 1.0

        vol_ratio = 1.0
        if volume is not None and t >= 1:
            recent = volume[max(0, t - self.lookback) : t + 1]
            prev = volume[max(0, t - 2 * self.lookback) : t - self.lookback + 1]
            if len(recent) and len(prev):
                recent_mean = float(np.mean(recent)) + 1e-9
                prev_mean = float(np.mean(prev)) + 1e-9
                vol_ratio = float(np.clip(prev_mean / recent_mean, 0.3, 3.0))

        return float(np.tanh(ret / self.scale * vol_ratio))
