# Evolutionary-StrAItegy

**Can self-evolution be combined with game theory?**

This repository is a small, runnable proof-of-concept that says *yes*. It builds a
trading policy that **improves itself** across iterations by treating the market
as a *repeated game* against an opponent (the crowd / market makers / macro
flows), and it does so under a hard constraint: **the model is never allowed to
know the history it is about to predict.**

> Status: research prototype / MVP. Everything runs in simulation. No real capital
> is touched. The "risk gate" is intentionally conservative.

---

## The idea in one paragraph

A pool of expert strategies (momentum, mean-reversion, news-sentiment) each emit
a position intent from *past* data only. Two learning mechanisms blend them: a
**multi-armed bandit** that allocates weight by reward, and a **regret-matching**
learner that hedges against the opponent the way self-play poker agents (Libratus,
Pluribus) do. An **Evolution Strategy** mutates the meta-parameters (how much to
trust the bandit vs the regret learner, how much risk to take, the volatility
target), keeps the fittest on in-sample data, and repeats. The whole thing is
driven by a scheduler that re-runs the loop every few hours, and every successful
configuration can be *distilled* into a reusable expert "skill" — so the system
compounds its own knowledge over time.

---

## Architecture

```
感知层 (market + news) → 对手建模 (opponent state) → 进化引擎 → 专家 Skill 库
                                                  ↓
                                 回测·评估 → 记忆 (regime) → 风控闸门 (sim only)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design, and
[`docs/WALK_FORWARD.md`](docs/WALK_FORWARD.md) for how look-ahead is structurally
prevented.

| Module | Responsibility |
|--------|----------------|
| `engine/bandit.py` | multi-armed bandit over experts (weight allocation) |
| `engine/regret.py` | regret-matching / no-regret learner (game-theoretic core) |
| `engine/regime.py` | causal market-regime detection |
| `engine/evolve.py` | online simulator + Evolution-Strategy meta-search |
| `experts/` | independent strategy skills (the gene pool) |
| `data/` | market loading, news placeholder, opponent model |
| `backtest/simulate.py` | walk-forward, out-of-sample evaluation |
| `evolve_run.py` | single run entry point (what a scheduler calls) |

---

## Quick start

```bash
# 1. install dependencies (numpy/pandas are required; yfinance is optional)
pip install -r requirements.txt

# 2. (optional) fetch real history, e.g. BTC-USD daily 2018-2024
python -c "from data.fetch_market import fetch_yfinance; fetch_yfinance()"

# 3. run the self-evolving walk-forward backtest
python evolve_run.py
```

If no CSV is present, the run automatically generates a **synthetic** market as a
stand-in so the project is always runnable. Replace
`data/market_btcusd_daily.csv` with real data for genuine results.

### Output

```
================================================================
Evolutionary-StrAItegy — walk-forward out-of-sample report
================================================================
data source        : csv  (2557 daily bars)
out-of-sample steps: 1953
total OOS return   : xx.xx%
OOS Sharpe         : x.xx
max drawdown       : -x.xx%
epochs             : 93
```

Files written:
- `artifacts/oos_equity.csv` — the out-of-sample equity curve
- `memory/performance.json` — machine-readable run metrics
- `memory/regime_memory.md` — append-only log of what worked, in which regime

---

## Why "predict history" is honest here

The backtest uses **walk-forward** validation: each epoch trains only on data
strictly before the test window, then predicts the next `test_horizon` days it has
never seen. Because every feature, opponent estimate and learner update is causal
(only past data), there is no look-ahead leakage — the "future" the model
predicts is genuinely unknown to it at decision time. Details in
[`docs/WALK_FORWARD.md`](docs/WALK_FORWARD.md).

---

## Roadmap

- [ ] Real news/NLP sentiment feed (`data/fetch_news.py` is a placeholder)
- [ ] Richer opponent model (order-flow / limit-order-book proxies)
- [ ] Skill distillation: persist robust evolved configs as versioned expert skills
- [ ] Scheduler integration (run `evolve_run.py` every N hours)
- [ ] Paper-trading sandbox, then small-capital with a human gate

---

## Disclaimer

This is an educational/research prototype. It is **not** investment advice and
does not execute live trades. Use at your own risk.
