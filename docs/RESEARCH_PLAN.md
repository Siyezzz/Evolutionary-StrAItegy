# Research plan: a small, falsifiable evolutionary finance system

## Scope

Start with one narrow problem: **daily BTC-USD exposure allocation** using only
information available by the previous close. The action is a bounded position in
`[-1, 1]`; the outcome is next-day net return after turnover costs. This is a
research simulator, not a live-trading system.

The economic hypotheses are deliberately small:

1. medium-horizon trend persistence can earn a time-series momentum premium;
2. short-horizon overreaction can mean-revert;
3. volatility/liquidity state changes which behavior is rewarded;
4. turnover is costly and must be priced into both training and evaluation.

## The game, stated honestly

At each date the learner chooses a mixture over causal experts. Nature then
reveals the next return, making every expert's counterfactual reward observable.
This is a **full-information repeated game** (online learning), not a bandit and
not poker-style CFR. External-regret matching is useful because it asks whether
the adaptive mixture did worse than committing to one expert in hindsight.

`opponent_state` remains research scaffolding. Fixed and causal rolling-ranked
contexts are connected to an optional contextual regret learner, but both tested
variants failed their complete promotion gates. The global regret learner stays
the default. A future contextual successor must improve net OOS performance or
worst-regime regret across multiple seeds without exceeding a predeclared
turnover and drawdown budget.

## Three separate learning loops

Do not collapse these into one vague claim of "self-evolution":

- **Within a test stream:** online weights update only after outcomes arrive.
- **Across walk-forward epochs:** ES may mutate the previous in-sample champion,
  but this mechanism stays behind an ablation flag. OOS results are reported but
  never used to choose the next parent.
- **Across research runs (SEA):** store hypotheses, failures and reproducible
  evidence. Promote a lesson only after independent outcomes; SEA is research
  memory, not a source of trading signals and not permission to overfit history.

## Evaluation contract

Before adding another expert, implement and freeze this contract:

1. rolling or expanding training window, then a disjoint 21-day test block;
2. every return from the first OOS boundary through the final bar counted once;
3. transaction cost charged on absolute position turnover;
4. benchmarks: cash, buy-and-hold, equal-weight experts, best fixed expert and
   the prior non-contextual learner;
5. report CAGR/total return, annualized volatility, Sharpe, Sortino, maximum
   drawdown, turnover, exposure and worst-regime result;
6. paired comparisons over multiple seeds and cost assumptions (5/10/25 bps);
7. a final untouched lockbox period used once for a promotion decision;
8. no winner is persisted as a production champion from OOS alone.

## Milestones

### M0 — trustworthy harness (implemented)

- Fix OOS boundary coverage, maximum-drawdown initialization and regret payoff.
- Put turnover cost inside fitness and OOS PnL.
- Test cross-epoch inheritance against a restart baseline; keep it disabled when
  paired OOS evidence is negative.
- Add executable regression tests.

### M1 — measurement before sophistication (implemented, provenance incomplete)

- Add dated positions/trades, benchmark returns and the full metric set.
- Replace the expanding all-history fit with configurable rolling windows.
- Add deterministic experiment manifests and machine-readable run IDs.
- Test data quality: ordering, duplicates, missing/non-positive prices and volume.

### M2 — contextual game (challengers tested; none promoted)

- Freeze regime thresholds on training data only.
- Maintain regret state per regime with shrinkage to a global learner.
- Compare against the M1 non-contextual baseline using paired seeds.
- Remove or rename `NewsSentimentExpert` until actual point-in-time news exists;
  its current feature is a price/volume proxy, not historical news sentiment.

### M3 — robust evolution

- Separate parameter search, model selection and lockbox evaluation.
- Evolve meaningful structural genes (lookbacks, cost-aware turnover penalty,
  regime shrinkage), not redundant leverage parameters.
- Promote only stable candidates that pass pre-declared drawdown and turnover
  gates; retain a recoverable predecessor.

### M4 — new data, then paper trading

- Add point-in-time macro/financial data only with publication timestamps.
- Prefer interpretable additions such as funding, basis or realized-volatility
  structure before opaque NLP features.
- Run a shadow/paper portfolio with a human gate. Live execution is out of scope
  until historical, lockbox and shadow results all agree.

## Current evidence (2026-09-06)

On the bundled 2018-01-01 through 2024-12-30 BTC-USD data, a quick fixed-seed
smoke experiment (`population=4`, `generations=2`, 10 bps turnover cost) covered
2,052 OOS daily returns across 98 epochs and produced approximately 8.21% total
return, 0.30 Sharpe and -10.18% maximum drawdown. This is a plumbing result, not
evidence of alpha: it has not yet been compared with the required benchmarks or
multiple seeds. A simple buy-and-hold calculation over the same 2,052 return
boundaries produced approximately 1,030.11% total return, 0.83 Sharpe and -76.63%
maximum drawdown. The prototype reduced exposure and drawdown, but did not beat
the passive asset on return or Sharpe in this smoke comparison.

The first predeclared inheritance ablation used 3 seeds and 5/10/25 bps costs
(9 paired cases, run `7f01b227bed56dfb`). Inheritance beat restart on OOS Sharpe
in only 1/9 pairs; mean Sharpe delta was -0.2401 and mean total-return delta was
-0.0312. It also reduced average turnover, so the result is a trade-off rather
than proof that persistence is always harmful. The default is nevertheless
`inherit_champion: false` until a guarded successor wins a new paired test.

All six fixed seed/mechanism trajectories lost net return monotonically as the
cost assumption rose from 5 to 10 to 25 bps. This validates cost sensitivity in
the harness, not any particular cost estimate. The original `12 x 6` default ES
budget also took roughly one minute per ten epochs on the test host, contradicting
the old runtime claim. Routine defaults are now `4 x 2`; larger searches must be
registered as additional model-selection trials.

The paper-motivated exponential-weights allocator was then compared with the
legacy allocator at 10 bps using seeds 7/42/101 (run `7a24605f293168e7`). It lost
OOS Sharpe in all 3/3 pairs, with mean Sharpe delta -0.1200 and mean total-return
delta -0.1416. It also raised exposure and drawdown in every run. The
implementation remains available as a challenger, while `allocator: legacy`
stays the default. This is evidence against this configured successor, not
against exponential weights in general.

The first contextual-regret transfer test (run `b4de7caf581ca7ba`) compared
global and contextual regret on BTC, SPY and GLD with a frozen seed and budget.
Contextual regret won Sharpe in only 1/3 markets; mean Sharpe delta was -0.0999
and mean return delta -0.0616. It increased turnover in all three. SPY and GLD
were dominated by the neutral context (2,455/3,270 and 2,428/3,270 OOS steps),
showing that the fixed context mapping did not transfer cleanly. `regret_mode:
global` remains the default.

A causal rolling-ranked context then replaced fixed thresholds without changing
the frozen seed, cost or search budget (run `df90a81df2f03c75`). Contextual
regret won Sharpe in 2/3 markets and raised mean Sharpe by 0.1403, while mean
total-return delta remained -0.0163. It worsened drawdown on BTC and SPY and
raised turnover everywhere, so it does not meet the lockbox gate. The repeated
global controls matched the fixed-context report exactly.

All absolute results above the hard-position-limit fix are retained as debugging
history but are superseded for performance interpretation. Volatility targeting
had allowed final positions to exceed the advertised `[-1, 1]` action space;
`max_abs_position` now clamps the executable position after every sizing
transform, with a regression test.

Post-fix expert diagnostics (`6f0cf7cbf364d0d4`) showed that the original gene
pool was structurally weak on SPY and GLD. A screened pool containing medium- and
long-horizon trend experts beat the old pool on Sharpe in 3/3 development markets
(run `d838b711cadac1f9`, mean delta +1.3112), but at a 1.0 position cap it worsened
drawdown in all three. Reducing the cap to 0.5 and running every 5/10/25 bps case
(run `b969658aab15cdcb`) produced monotonic cost degradation in 3/3 markets. At
10 bps the conservative candidate beat the old pool on Sharpe and maximum
drawdown in 3/3 markets, satisfying the predeclared lockbox gates.

The TLT lockbox was therefore opened exactly once on 2026-09-07. Under the frozen
10 bps, seed-42 comparison, the candidate improved Sharpe from -0.4721 to -0.3178
(delta +0.1543), so the primary directional prediction passed. It nevertheless
reduced total return from -16.70% to -23.28% and worsened maximum drawdown from
-17.47% to -31.89%; the risk prediction failed. The candidate was not promoted,
the old pool remains the executable default, and TLT is permanently consumed as
an evaluation set. The next iteration must use only BTC/SPY/GLD development data
or genuinely new data—not TLT—to design explicit downside-risk control.

That next development-only test was preregistered before execution as
`experiments/risk_matched_preregistration.json` (run `e9a66fed69d5c7ad`). A
0.2 position cap brought candidate/predecessor annualized-volatility ratios to
1.37 on BTC, 1.46 on SPY and 1.41 on GLD. At 10 bps it improved Sharpe and maximum
drawdown in 3/3 markets; mean Sharpe delta was +1.3099 and mean total-return delta
was +0.5157. Returns declined monotonically across 5/10/25 bps in every market.
This supports risk-budget parity as the next design direction, but it is still a
single-seed development result and has no fresh lockbox confirmation. It is not
the default and must next survive paired seeds 7/42/101 before any new lockbox is
designated.
