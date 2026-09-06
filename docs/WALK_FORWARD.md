# Walk-forward, no-look-ahead validation

The single most important property of this project: **the model is never allowed
to know the history it is about to predict.** This is enforced structurally, not
by convention.

## The rule

At decision time `t`, a strategy may only read data with index `<= t`. The
position it takes then earns the return from `t` to `t+1`. Learners
(`bandit`, `regret`) are updated *after* that return is realised. Nothing indexed
`> t` is ever touched.

## Walk-forward layout

History is split into consecutive, non-overlapping epochs. Each epoch:

```
|<----------- in-sample (train) ---------->|<-- out-of-sample (test) -->|
^                                          ^                          ^
train_start                                train_end = test_start     test_end
```

1. **In-sample**: everything strictly before `test_start`. The Evolution Strategy
   searches meta-parameters here. This is the *only* window the optimiser sees.
2. **Out-of-sample**: the next `test_horizon` days, which training never used. The
   policy — warmed up on the train window — keeps adapting *online*, but can only
   use information that has already arrived. The realised return here is the
   genuine "predict the past" result.

Epochs advance by `test_horizon` and repeat until history runs out. Aggregating
all out-of-sample steps yields an unbiased estimate of live performance.

## Where leakage is (deliberately) impossible

| Place | Only sees |
|-------|-----------|
| expert `signal(ctx)` | `close[:t+1]`, `volume[:t+1]` |
| `opponent_state` | prices/volume up to `t` |
| `detect_regime` | window ending at `t` |
| `bandit` / `regret` update | reward from return `t-1 -> t`, learned at step `t` |
| ES `es_search` | train window only |

The test window is passed to `es_search` *never*; it is touched solely by the
online `simulate_period` that consumes one new bar at a time.

## Running it

```bash
python evolve_run.py
```

It prints an out-of-sample report (total return, Sharpe, max drawdown) and writes
`artifacts/oos_equity.csv` plus `memory/performance.json`. Re-run with different
`seed` / `train_window` in `config.yaml` to stress-test robustness.
