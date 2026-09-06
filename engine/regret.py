"""No-regret learner — regret matching against the opponent.

Game-theoretic core. Instead of trusting any single expert, we keep a running
*cumulative regret* for every expert (how much we lost by not having followed
that expert), and play a distribution proportional to positive regret.

This is the same family of algorithm (regret matching / counterfactual regret
minimisation) used by self-play poker agents such as Libratus/Pluribus: it
converges to a strategy that cannot be exploited by an opponent that reveals its
tendencies. Here the "opponent" is the market crowd, whose behaviour we infer
causally from price/volume.

``update()`` consumes only rewards that have already happened, so there is no
look-ahead.
"""

from __future__ import annotations


class RegretMatcher:
    def __init__(self, names):
        self.names = list(names)
        self.regret = {n: 0.0 for n in names}
        self.strategy = {n: 1.0 / len(names) for n in names}

    def update(self, rewards: dict) -> dict:
        """Update cumulative regret with this step's per-expert rewards."""
        avg = sum(rewards[n] for n in self.names) / len(self.names)
        for n in self.names:
            self.regret[n] = max(0.0, self.regret[n] + (rewards[n] - avg))
        total = sum(self.regret.values())
        if total <= 1e-9:
            w = {n: 1.0 / len(self.names) for n in self.names}
        else:
            w = {n: self.regret[n] / total for n in self.names}
        self.strategy = w
        return w

    def weights(self) -> dict:
        return self.strategy
