"""News / sentiment source (live placeholder).

In production this module would call a news/NLP API and return a sentiment score
in ``[-1, 1]`` for a given date. The offline backtest does not need it because
``experts/news_sentiment.py`` uses a deterministic volume-surge proxy instead.

Keep the function signature stable so the expert can call it without changes
when going live.
"""

from __future__ import annotations


def fetch_sentiment(date: str) -> float:
    """Return a sentiment score in [-1, 1] for ``date``.

    TODO: replace with a real call, e.g.:
        from agentic_search / WebSearch -> summarise headlines -> sentiment model
    For now returns 0.0 (neutral) so the pipeline runs offline.
    """
    _ = date
    return 0.0
