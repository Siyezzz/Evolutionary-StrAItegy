"""Expert allocator (legacy class name: ``ExpertBandit``).

The simulator observes the counterfactual reward of every expert, so this is a
full-information expert-advice setting rather than a true bandit problem. The
current implementation maintains an approximate Gaussian score over the
mean reward (risk-adjusted return) of every expert and allocates weight
proportionally to the posterior mean (deterministic ``weights()``) while still
supporting stochastic ``sample()`` for live exploration.

Crucially, ``update()`` is called *after* a return is realised, so the learner
never sees the future — it only ever learns from information that already
existed at decision time.
"""

from __future__ import annotations

import numpy as np


class ExpertBandit:
    def __init__(self, names, prior_precision: float = 1.0):
        self.names = list(names)
        self.mu = {n: 0.0 for n in names}
        self.prec = {n: float(prior_precision) for n in names}
        self.counts = {n: 0 for n in names}

    # --- deterministic allocation (used for training / evaluation) ----------
    def weights(self) -> dict:
        vals = np.array([self.mu[n] for n in self.names], dtype=float)
        if vals.max() - vals.min() < 1e-9:
            return {n: 1.0 / len(self.names) for n in self.names}
        vals = vals - vals.max()
        w = np.exp(vals)
        w = w / w.sum()
        return {n: float(w[i]) for i, n in enumerate(self.names)}

    # --- stochastic sampling (used for live exploration) -------------------
    def sample(self) -> dict:
        samples = {
            n: np.random.normal(self.mu[n], 1.0 / np.sqrt(self.prec[n]))
            for n in self.names
        }
        vals = np.array([samples[n] for n in self.names], dtype=float)
        vals = vals - vals.max()
        w = np.exp(vals)
        w = w / w.sum()
        return {n: float(w[i]) for i, n in enumerate(self.names)}

    # --- online learning ---------------------------------------------------
    def update(self, name: str, reward: float) -> None:
        self.counts[name] += 1
        c = self.counts[name]
        # running mean
        self.mu[name] = self.mu[name] + (reward - self.mu[name]) / c
        # precision grows with observations (sharper posterior)
        self.prec[name] = 1.0 + c


class ExponentialWeightsAllocator:
    """Full-information exponential weights with an anytime learning rate.

    Rewards are scaled by a declared daily risk unit and clipped to [-1, 1],
    matching the bounded-reward setting used by standard expert-advice bounds.
    """

    def __init__(self, names, reward_scale: float = 0.02):
        self.names = list(names)
        self.reward_scale = max(float(reward_scale), 1e-9)
        self.log_weight = {name: 0.0 for name in self.names}
        self.rounds = 0

    def weights(self) -> dict:
        values = np.array([self.log_weight[name] for name in self.names])
        values -= values.max()
        weights = np.exp(values)
        weights /= weights.sum()
        return {
            name: float(weights[index])
            for index, name in enumerate(self.names)
        }

    def update_all(self, rewards: dict) -> None:
        self.rounds += 1
        eta = float(np.sqrt(8.0 * np.log(len(self.names)) / self.rounds))
        for name in self.names:
            bounded = float(
                np.clip(rewards[name] / self.reward_scale, -1.0, 1.0)
            )
            self.log_weight[name] += eta * bounded


def make_allocator(names, config: dict):
    kind = str(config.get("allocator", "legacy")).lower()
    if kind == "legacy":
        return ExpertBandit(names)
    if kind == "exponential":
        return ExponentialWeightsAllocator(
            names, reward_scale=float(config.get("allocator_reward_scale", 0.02))
        )
    raise ValueError("allocator must be 'legacy' or 'exponential'")
