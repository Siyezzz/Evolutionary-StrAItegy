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

    def update(
        self,
        rewards: dict,
        played_weights: dict | None = None,
        context: str | None = None,
    ) -> dict:
        """Update external regret against the distribution actually played.

        Regret is the reward of an action minus the learner's realised expected
        reward.  Using the uniform expert average here would only be correct
        while the strategy itself is uniform.
        """
        played = self.strategy if played_weights is None else played_weights
        realised = sum(played[n] * rewards[n] for n in self.names)
        for n in self.names:
            self.regret[n] = max(
                0.0, self.regret[n] + (rewards[n] - realised)
            )
        total = sum(self.regret.values())
        if total <= 1e-9:
            w = {n: 1.0 / len(self.names) for n in self.names}
        else:
            w = {n: self.regret[n] / total for n in self.names}
        self.strategy = w
        return w

    def weights(self, context: str | None = None) -> dict:
        return self.strategy


class ContextualRegretMatcher:
    """Regret matching per causal market context with global shrinkage."""

    def __init__(self, names, shrinkage: float = 20.0):
        self.names = list(names)
        self.shrinkage = max(float(shrinkage), 0.0)
        self.global_matcher = RegretMatcher(names)
        self.local: dict[str, RegretMatcher] = {}
        self.counts: dict[str, int] = {}

    def _local(self, context: str) -> RegretMatcher:
        if context not in self.local:
            self.local[context] = RegretMatcher(self.names)
            self.counts[context] = 0
        return self.local[context]

    def weights(self, context: str | None = None) -> dict:
        global_weights = self.global_matcher.weights()
        if context is None or context not in self.local:
            return global_weights
        local_weights = self.local[context].weights()
        count = self.counts[context]
        local_share = count / (count + self.shrinkage) if count else 0.0
        return {
            name: local_share * local_weights[name]
            + (1.0 - local_share) * global_weights[name]
            for name in self.names
        }

    def update(
        self,
        rewards: dict,
        played_weights: dict | None = None,
        context: str | None = None,
    ) -> dict:
        self.global_matcher.update(rewards)
        if context is None:
            return self.weights()
        matcher = self._local(context)
        # Each local learner measures regret against its own played distribution;
        # the shrinkage blend is an allocation layer, not its counterfactual base.
        matcher.update(rewards)
        self.counts[context] += 1
        return self.weights(context)


def make_regret_matcher(names, config: dict):
    mode = str(config.get("regret_mode", "global")).lower()
    if mode == "global":
        return RegretMatcher(names)
    if mode == "contextual":
        return ContextualRegretMatcher(
            names, shrinkage=float(config.get("context_shrinkage", 20.0))
        )
    raise ValueError("regret_mode must be 'global' or 'contextual'")
