"""Walk-forward evaluation — the honest, no-look-ahead backtest.

The full history is walked in consecutive, non-overlapping epochs. For each
epoch:

1. **In-sample (train)**: everything strictly before the test window. The
   Evolution Strategy searches the best meta-parameters here. This is the only
   window the optimiser is allowed to see.

2. **Out-of-sample (test)**: the next ``test_horizon`` days, which the model has
   NEVER seen during training. The policy — with learners warmed up on the train
   window — keeps adapting *online* but can only use information that has already
   arrived. The realised returns here are the genuine "predict the past" result.

Expert signals are precomputed once (they are causal and param-independent), so
the Evolution Strategy search is cheap and the whole pipeline stays strictly
out-of-sample.
"""

from __future__ import annotations

import sys

import numpy as np

from engine.bandit import make_allocator
from engine.regret import make_regret_matcher
from engine.evolve import es_search, simulate_period, precompute_signals
from engine.regime import precompute_contexts


def walk_forward(
    data: dict,
    experts,
    config: dict,
    rng: np.random.Generator,
    *,
    return_trace: bool = False,
):
    close = data["close"]
    volume = data["volume"]
    n = len(close)

    train_window = int(config["train_window"])
    horizon = int(config["test_horizon"])
    min_train = int(config.get("min_train", 252))
    window_mode = str(config.get("window_mode", "expanding")).lower()
    if window_mode not in {"rolling", "expanding"}:
        raise ValueError("window_mode must be 'rolling' or 'expanding'")

    base = {
        "blend": float(config.get("blend_init", 0.5)),
        "risk": float(config["risk"]),
        "vol_target": float(config["vol_target"]),
        "transaction_cost_bps": float(config.get("transaction_cost_bps", 0.0)),
        "max_abs_position": float(config.get("max_abs_position", 1.0)),
    }

    # causal, param-independent signals — computed exactly once
    signals = precompute_signals(close, volume, experts)
    contexts = precompute_contexts(
        close,
        volume,
        method=str(config.get("context_method", "fixed")),
        calibration_window=int(config.get("context_calibration_window", 252)),
    )

    oos_pnls: list[float] = []
    epochs: list[dict] = []
    trace = {"t": [], "pos": [], "pnl": [], "turnover": [], "context": []}

    k = 0
    while True:
        tr_end = train_window + k * horizon
        tr_start = max(0, tr_end - train_window) if window_mode == "rolling" else 0
        te_start = tr_end
        te_end = min(tr_end + horizon, n)

        if te_end <= te_start or tr_end < min_train:
            break

        # 1) evolve on in-sample only
        best = es_search(
            close, volume, tr_start, tr_end, experts, base, config, rng, signals,
            contexts,
        )

        # 2) warm up learners on train, then test out-of-sample (online, causal)
        bandit = make_allocator([e.name for e in experts], config)
        regret = make_regret_matcher([e.name for e in experts], config)
        warm_store: dict = {
            "t": [], "pos": [], "pnl": [], "turnover": [], "context": []
        }
        simulate_period(
            close, volume, tr_start, tr_end, experts, best, bandit, regret,
            signals, store=warm_store, contexts=contexts,
        )

        store = {"t": [], "pos": [], "pnl": [], "turnover": [], "context": []}
        test_pnls = simulate_period(
            close, volume, te_start, te_end, experts, best, bandit, regret,
            signals, record=True, store=store,
            initial_position=float(warm_store["final_position"]),
            contexts=contexts,
        )

        arr = np.array(test_pnls, dtype=float)
        eq = np.cumprod(1.0 + arr)
        epoch_return = float(eq[-1] - 1.0) if len(eq) else 0.0
        epoch_sharpe = (
            float(
                arr.mean() / (arr.std() + 1e-9)
                * np.sqrt(int(config.get("periods_per_year", 365)))
            )
            if len(arr) > 1
            else 0.0
        )

        oos_pnls.extend(test_pnls)
        for key in trace:
            trace[key].extend(store[key])
        epochs.append(
            {
                "epoch": k,
                "train_start": int(tr_start),
                "train_end": int(tr_end),
                "test_start": int(te_start),
                "test_end": int(te_end),
                "params": best,
                "epoch_return": epoch_return,
                "epoch_sharpe": epoch_sharpe,
            }
        )
        # Inherit the in-sample winner as the next epoch's incumbent. Test
        # outcomes are never used for selection, preserving OOS validity.
        if bool(config.get("inherit_champion", True)):
            base = dict(best)
        k += 1
        if k % 10 == 0:
            print(f"  ... epoch {k} done (OOS steps so far: {len(oos_pnls)})", file=sys.stderr)

    if return_trace:
        return oos_pnls, epochs, trace
    return oos_pnls, epochs
