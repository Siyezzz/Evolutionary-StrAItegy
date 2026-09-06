"""Expert allocator — a multi-armed bandit over the expert pool.

Each expert is treated as an "arm". We maintain a Gaussian posterior over the
mean reward (risk-adjusted return) of every expert and allocate weight
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
