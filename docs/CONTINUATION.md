# Continuation message

## Objective

Keep this repository a small, paper-grounded finance/economics laboratory that
combines full-information repeated-game learning, causal expert strategies and
Evolution Strategy search. Every successor must be reproducible, cost-aware and
compared with a recoverable predecessor. Preserve failures; do not optimize on
evaluation data.

## Authoritative state at handoff (2026-09-07)

- The walk-forward boundary, regret payoff, transaction costs, initial-capital
  drawdown, causal contexts, annualization and final executable-position bound
  have regression coverage.
- Champion inheritance, exponential weights and two contextual-regret variants
  all failed paired development tests. They remain challengers, not defaults.
- The screened trend pool at cap 0.5 passed BTC/SPY/GLD development gates, then
  failed the one-time TLT risk prediction. On TLT it improved Sharpe by +0.1543
  but worsened total return by -0.0658 and maximum drawdown by -0.1441.
- TLT was opened exactly once. `experiments/LOCKBOX_ACCESS_LOG.jsonl` and
  `artifacts/lockbox_tlt_result.json` are permanent. Never rerun, split, tune on,
  or describe TLT as a fresh holdout.
- A follow-up development-only candidate uses the screened pool with
  `max_abs_position=0.2`. Run `e9a66fed69d5c7ad` passed its frozen seed-42 gates:
  Sharpe wins 3/3, drawdown no worse 3/3, risk-ratio band 3/3, and monotonic
  degradation at 5/10/25 bps. It is promising but not promoted.
- `config.yaml` intentionally still points to the recoverable old expert pool.

## Next experiment—do this first

1. Freeze a paired multi-seed preregistration for seeds 7, 42 and 101 on only
   BTC, SPY and GLD. Compare the 0.2-cap screened pool against the old pool at
   5/10/25 bps; include Sharpe, return, maximum drawdown, realized volatility,
   turnover and exposure. Do not omit failed cells.
2. Require Sharpe wins in at least 6/9 seed-market pairs, positive mean return
   delta, drawdown no worse in at least 6/9, and realized-volatility ratio inside
   a frozen band. Report uncertainty across seeds; one seed is not robustness.
3. If it fails, retain the report and diagnose the risk budget using development
   data only. If it passes, make `max_abs_position`/realized-volatility budget a
   named structural candidate rather than silently changing the default.
4. Only after those gates pass, designate a genuinely new asset snapshot as a
   new lockbox before strategy evaluation. Record provider, adjustment policy,
   retrieval time and SHA-256. Prefer a second data provider or reconcile random
   rows against one; SPY/GLD/TLT currently share yfinance vendor risk.
5. Never reuse TLT as the new lockbox. Never tune a threshold after viewing a
   fresh holdout result.

## Evidence map

- Research decisions and exact results: `docs/RESEARCH_PLAN.md`
- Academic sources and applicability limits: `docs/LITERATURE.md`
- Dataset provenance and hashes: `data/DATA_CARD.md`
- One-time protocol: `experiments/LOCKBOX_PROTOCOL.md`
- Frozen TLT authorization: `experiments/lockbox_authorization.json`
- Risk-matched preregistration/report:
  `experiments/risk_matched_preregistration.json` and
  `artifacts/risk_matched_cost_grid_e9a66fed69d5c7ad.json`

## SEA handoff

SEA locally activated two lessons during this work: exact half-open OOS coverage
(`fd6742e75b344d6e906ee05466dac7bf`) and preserve the predecessor plus require
paired OOS evidence (`c8d8fe58c0c44571886feb7f4536146c`). A candidate lesson
about refusing Sharpe-only promotion is
`9e0e41b8d5e54d2b8cece1f19ba5f786`.

SEA's notice changed to version `2026-09-07.1` at handoff. The prior sharing
choice no longer counts as acknowledgement. The registry is currently not
configured and sharing is inactive. Do not record more SEA feedback or publish
anything until the owner explicitly acknowledges the new notice and chooses a
mode. This restriction does not block repository research.
