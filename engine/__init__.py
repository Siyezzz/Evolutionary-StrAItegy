"""Evolution engine package.

The engine turns a pool of expert strategies into a self-improving policy via
three cooperating mechanisms:

* ``bandit``     – allocates weight across experts (Thompson / UCB style)
* ``regret``     – no-regret learner that hedges against the opponent
* ``evolve``     – Evolution Strategy that searches the meta-parameters

All modules are deliberately dependency-light (numpy only) so the project runs
anywhere. The only external data dependency is a CSV of daily OHLCV prices.
"""

from __future__ import annotations


def load_config(path: str) -> dict:
    """Load config.yaml.

    Uses PyYAML when available, otherwise falls back to a tiny flat parser so
    the project runs even without extra dependencies installed.
    """
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        cfg: dict = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or ":" not in line:
                    continue
                key, value = line.split(":", 1)
                # strip an inline comment if present
                value = value.split("#", 1)[0].strip()
                cfg[key.strip()] = value
        return cfg
