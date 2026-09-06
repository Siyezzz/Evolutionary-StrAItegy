"""Evolution engine: online simulation + Evolution-Strategy meta-search.

Two pieces live here:

1. ``precompute_signals`` — expert signals depend ONLY on past data and the
   expert's own logic, never on the meta-parameters. So we compute each expert's
   full causal signal series ONCE. This is safe (no look-ahead) and makes the
   expensive Evolution Strategy search cheap.

2. ``simulate_period`` — the *causal* online simulator. For each day it blends the
   precomputed expert signals through the bandit + regret learners, sizes the
   position with volatility targeting, and updates the learners *after* the return
   is realised. No future index is ever read.

3. ``es_search`` — an Evolution Strategy that searches the meta-parameters
   (expert-blend, risk, volatility target) on an *in-sample* window. Each
   candidate is scored with ``simulate_period``; the best are kept (elitism) and
   mutated to form the next generation. This is the "self-evolution" step: the
   policy mutates and selects the fittest offspring, never touching the
   out-of-sample window.
"""

from __future__ import annotations

import numpy as np

from .bandit import ExpertBandit
from .regret import RegretMatcher
from .regime import detect_regime
from data.opponent_model import opponent_state


def precompute_signals(close: np.ndarray, volume: np.ndarray, experts) -> dict:
    """Compute every expert's causal signal series once.

    ``signals[name][t]`` is the position intent at time ``t`` using only data up
    to ``t``. Because it never depends on the meta-parameters, it is valid for all
    evolution candidates and for both train and test windows.
    """
    n = len(close)
    sig: dict[str, np.ndarray] = {e.name: np.empty(n, dtype=float) for e in experts}
    for t in range(n):
        ctx = {"t": t, "close": close, "volume": volume}
        for e in experts:
            sig[e.name][t] = e.signal(ctx)
    return sig


def simulate_period(
    close: np.ndarray,
    volume: np.ndarray,
    start: int,
    end: int,
    experts,
    params: dict,
    bandit: ExpertBandit,
    regret: RegretMatcher,
    signals: dict,
    record: bool = False,
    store: dict | None = None,
):
    """Online-simulate the period ``[start, end)`` and return the pnl list.

    The position at step ``t`` uses the signal known at ``t-1`` and earns the
    return from ``t-1`` to ``t``. Learners are updated *after* the return, so the
    simulation is strictly causal (no look-ahead).
    """
    names = [e.name for e in experts]
    pnls = []
    for t in range(start + 1, end):
        s_t = t - 1
        bw = bandit.weights()
        rw = regret.weights()
        pos = 0.0
        rewards = {}
        for e in experts:
            w = params["blend"] * bw[e.name] + (1.0 - params["blend"]) * rw[e.name]
            sig = signals[e.name][s_t]
            pos += w * sig
            rewards[e.name] = sig * (close[t] / close[t - 1] - 1.0)

        r = close[t] / close[t - 1] - 1.0
        pnl = pos * r

        # volatility targeting: scale down when realised vol is high
        vt = params["vol_target"]
        recent = close[max(0, t - 21) : t]
        if len(recent) > 2:
            rv = float(np.std(np.diff(recent) / recent[:-1])) + 1e-9
        else:
            rv = vt
        risk_scale = float(np.clip(vt / rv, 0.2, 3.0))
        pos = float(np.tanh(pos)) * params["risk"] * risk_scale
        pnl = pos * r

        pnls.append(pnl)
        for e in experts:
            bandit.update(e.name, rewards[e.name])
        regret.update(rewards)

        if record and store is not None:
            store["t"].append(t)
            store["pos"].append(pos)
            store["pnl"].append(pnl)
    return pnls


def _fitness(pnls: list[float]) -> float:
    arr = np.array(pnls, dtype=float)
    if len(arr) < 10:
        return -1e9
    sharpe = arr.mean() / (arr.std() + 1e-9) * np.sqrt(252)
    eq = np.cumprod(1.0 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    # reward return quality, penalise deep drawdowns
    return sharpe - 0.5 * abs(dd.min())


def es_search(
    close: np.ndarray,
    volume: np.ndarray,
    tr_start: int,
    tr_end: int,
    experts,
    base_params: dict,
    config: dict,
    rng: np.random.Generator,
    signals: dict,
):
    """Evolution Strategy over meta-parameters, scored on the in-sample window."""
    names = [e.name for e in experts]
    pop_size = int(config["es_population"])
    generations = int(config["es_generations"])
    sigma = float(config["es_sigma"])

    pop = []
    for _ in range(pop_size):
        p = dict(base_params)
        p["blend"] = float(np.clip(rng.normal(0.5, 0.25), 0.0, 1.0))
        p["risk"] = float(np.clip(rng.normal(1.0, 0.4), 0.3, 3.0))
        p["vol_target"] = float(np.clip(rng.normal(0.02, 0.01), 0.005, 0.06))
        pop.append(p)

    best = None
    best_fit = -1e9
    for _ in range(generations):
        scored = []
        for p in pop:
            bandit = ExpertBandit(names)
            regret = RegretMatcher(names)
            pnls = simulate_period(
                close, volume, tr_start, tr_end, experts, p, bandit, regret, signals
            )
            f = _fitness(pnls)
            scored.append((f, p))
            if f > best_fit:
                best_fit = f
                best = dict(p)

        scored.sort(key=lambda x: x[0], reverse=True)
        keep = max(1, pop_size // 4)
        elites = [dict(p) for _, p in scored[:keep]]

        new_pop = list(elites)
        while len(new_pop) < pop_size:
            parent = elites[rng.integers(0, len(elites))]
            child = dict(parent)
            child["blend"] = float(np.clip(parent["blend"] + rng.normal(0, sigma * 0.3), 0, 1))
            child["risk"] = float(np.clip(parent["risk"] + rng.normal(0, sigma), 0.3, 3.0))
            child["vol_target"] = float(
                np.clip(parent["vol_target"] + rng.normal(0, sigma * 0.05), 0.005, 0.06)
            )
            new_pop.append(child)
        pop = new_pop

    return best if best is not None else dict(base_params)
