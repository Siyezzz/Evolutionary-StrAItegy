# Architecture

> Can self-evolution be combined with game theory? Yes — by treating the market
> as a *repeated game* against an opponent (the crowd / market makers / macro
> flows) and letting a policy improve itself across iterations without ever
> peeking at the future it is asked to predict.

## The self-evolving loop

```
        ┌─────────────────── scheduler (every N hours) ───────────────────┐
        │                                                                  │
  感知层 (market + news) ─▶ 对手建模 (opponent state) ─▶ 进化引擎 ─▶ 专家 Skill 库
        │                                   ▲                │
        │                                   │                ▼
        └────────── 记忆 (regime memory) ◀── 回测·评估 (fitness) ◀── 风控闸门 (sim only)
```

1. **Perception layer** — pull market prices, volume and (live) news/sentiment.
2. **Opponent model** — infer the aggregate "other players'" state causally from
   past price/volume (trend, aggression, volatility).
3. **Evolution engine** — mutate the policy's meta-parameters, select the fittest
   on in-sample data, keep elites, repeat. This is the *self-evolution* step.
4. **Expert Skill library** — each expert is an independent, versioned strategy
   module (momentum, mean-reversion, news-sentiment, ...). The engine does not
   rewrite their logic; it decides *how much weight* each deserves right now.
5. **Backtest / evaluation** — score candidates with a risk-adjusted fitness
   (Sharpe minus drawdown penalty) strictly on in-sample data.
6. **Regime memory** — persist which parameters worked in which regime, so the
   next cycle starts from accumulated knowledge.
7. **Risk gate** — by default everything runs in simulation; real capital always
   requires a human confirmation.

## Game-theoretic core

The policy blends three cooperating mechanisms:

| Mechanism | Role | Inspiration |
|-----------|------|-------------|
| `bandit`  | allocate weight across experts by posterior reward | full-information expert advice |
| `regret`  | hedge across expert actions via cumulative regret | external regret matching |
| `evolve`  | search meta-parameters (blend, risk, vol target) | Evolution Strategies + elitism |

The `regret` learner is the explicitly game-theoretic piece. It treats expert
selection as a full-information repeated decision problem and minimizes external
regret against the observed return sequence. The market is not assumed to be a
rational adversary, and this MVP does not implement CFR, self-play, or establish
a Nash equilibrium. Regime-conditioned games are the next research step.

## Expert as a Skill

Each expert is intentionally structured like a reusable *skill*:

- a `signal(ctx)` that returns a position intent in `[-1, 1]` using only data up
  to `ctx["t"]` (never the future),
- a name, tunable hyperparameters, and an associated regime it tends to win in.

This makes "distilling an expert" a first-class operation: when an evolved
policy proves robust out-of-sample, the winning configuration can be persisted as
a versioned skill and reloaded as part of the gene pool next time — the system
compounds its own knowledge.

## Why this is safe to automate

- **No look-ahead.** All features, opponent state and rewards use only data up to
  the current step. See `docs/WALK_FORWARD.md`.
- **Simulation-first.** The loop runs entirely in simulation; the risk gate stops
  anything from touching real capital without a human.
- **Reproducible.** `config.yaml` + a fixed `seed` make every run repeatable.
