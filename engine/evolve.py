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

from .bandit import make_allocator
from .regret import make_regret_matcher
from .regime import detect_context


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
    bandit,
    regret,
    signals: dict,
    record: bool = False,
    store: dict | None = None,
    initial_position: float = 0.0,
    contexts: np.ndarray | None = None,
):
    """Online-simulate the period ``[start, end)`` and return the pnl list.

    The position at step ``t`` uses the signal known at ``t-1`` and earns the
    return from ``t-1`` to ``t``. Learners are updated *after* the return, so the
    simulation is strictly causal (no look-ahead).
    """
    names = [e.name for e in experts]
    pnls = []
    previous_position = float(initial_position)
    first_return = max(1, start)
    final_return = min(end, len(close))
    for t in range(first_return, final_return):
        s_t = t - 1
        context = (
            str(contexts[s_t])
            if contexts is not None
            else detect_context(close, volume, s_t)
        )
        bw = bandit.weights()
        rw = regret.weights(context)
        pos = 0.0
        rewards = {}
        for e in experts:
            w = params["blend"] * bw[e.name] + (1.0 - params["blend"]) * rw[e.name]
            sig = signals[e.name][s_t]
            pos += w * sig
            rewards[e.name] = sig * (close[t] / close[t - 1] - 1.0)

        r = close[t] / close[t - 1] - 1.0
        # volatility targeting: scale down when realised vol is high
        vt = params["vol_target"]
        recent = close[max(0, t - 21) : t]
        if len(recent) > 2:
            rv = float(np.std(np.diff(recent) / recent[:-1])) + 1e-9
        else:
            rv = vt
        risk_scale = float(np.clip(vt / rv, 0.2, 3.0))
        pos = float(np.tanh(pos)) * params["risk"] * risk_scale
        max_abs_position = float(params.get("max_abs_position", 1.0))
        pos = float(np.clip(pos, -max_abs_position, max_abs_position))
        cost_rate = float(params.get("transaction_cost_bps", 0.0)) / 10_000.0
        turnover = abs(pos - previous_position)
        pnl = pos * r - turnover * cost_rate

        pnls.append(pnl)
        if hasattr(bandit, "update_all"):
            bandit.update_all(rewards)
        else:
            for e in experts:
                bandit.update(e.name, rewards[e.name])
        regret.update(rewards, played_weights=rw, context=context)
        previous_position = pos

        if record and store is not None:
            store.setdefault("t", []).append(t)
            store.setdefault("pos", []).append(pos)
            store.setdefault("pnl", []).append(pnl)
            store.setdefault("turnover", []).append(turnover)
            store.setdefault("context", []).append(context)
    if store is not None:
        store["final_position"] = previous_position
    return pnls


def _fitness(pnls: list[float], periods_per_year: int = 365) -> float:
    arr = np.array(pnls, dtype=float)
    if len(arr) < 10:
        return -1e9
    sharpe = arr.mean() / (arr.std() + 1e-9) * np.sqrt(periods_per_year)
    eq = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
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
    contexts: np.ndarray | None = None,
):
    """Evolution Strategy over meta-parameters, scored on the in-sample window."""
    names = [e.name for e in experts]
    pop_size = int(config["es_population"])
    generations = int(config["es_generations"])
    sigma = float(config["es_sigma"])

    pop = []
    # The incumbent is always evaluated. Other candidates mutate around it,
    # which lets walk-forward epochs inherit rather than restart evolution.
    pop.append(dict(base_params))
    for _ in range(pop_size - 1):
        p = dict(base_params)
        p["blend"] = float(
            np.clip(rng.normal(base_params["blend"], 0.25), 0.0, 1.0)
        )
        p["risk"] = float(
            np.clip(rng.normal(base_params["risk"], 0.4), 0.3, 3.0)
        )
        p["vol_target"] = float(
            np.clip(rng.normal(base_params["vol_target"], 0.01), 0.005, 0.06)
        )
        pop.append(p)

    best = None
    best_fit = -1e9
    for _ in range(generations):
        scored = []
        for p in pop:
            bandit = make_allocator(names, config)
            regret = make_regret_matcher(names, config)
            pnls = simulate_period(
                close, volume, tr_start, tr_end, experts, p, bandit, regret,
                signals, contexts=contexts,
            )
            f = _fitness(pnls, int(config.get("periods_per_year", 365)))
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
