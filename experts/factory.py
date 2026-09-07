"""Build the configured expert gene pool."""

from __future__ import annotations

from .long_horizon_momentum import LongHorizonMomentumExpert
from .long_only_trend import LongOnlyTrendExpert
from .mean_reversion import MeanReversionExpert
from .momentum import MomentumExpert
from .news_sentiment import NewsSentimentExpert


EXPERTS = {
    "momentum": MomentumExpert,
    "mean_reversion": MeanReversionExpert,
    "news_sentiment": NewsSentimentExpert,
    "long_horizon_momentum": LongHorizonMomentumExpert,
    "long_only_trend": LongOnlyTrendExpert,
}


def build_experts(spec) -> list:
    names = (
        [part.strip() for part in spec.split(",") if part.strip()]
        if isinstance(spec, str)
        else list(spec)
    )
    unknown = [name for name in names if name not in EXPERTS]
    if unknown:
        raise ValueError(f"unknown experts: {', '.join(unknown)}")
    if not names:
        raise ValueError("at least one expert is required")
    return [EXPERTS[name]() for name in names]
