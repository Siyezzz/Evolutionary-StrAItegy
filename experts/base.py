"""Base class for expert strategies (skills)."""

from __future__ import annotations


class Expert:
    """A strategy skill.

    ``signal(ctx)`` must return a position intent in ``[-1, 1]`` using ONLY the
    information available in ``ctx`` up to time ``ctx["t"]``. Reading any index
    ``> ctx["t"]`` would be look-ahead and is forbidden.
    """

    name = "base"

    def signal(self, ctx: dict) -> float:
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<Expert {self.name}>"
