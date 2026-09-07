# One-time lockbox protocol

## Current state

`data/lockbox_tlt_daily.csv` was the TLT transfer lockbox. It was opened exactly
once on 2026-09-07 under `experiments/lockbox_authorization.json`. The append-only
record is `experiments/LOCKBOX_ACCESS_LOG.jsonl`, and the frozen result is
`artifacts/lockbox_tlt_result.json`. The evaluator refuses any rerun when either
record exists.

## Preconditions to open

A candidate configuration must be fully frozen before access and must:

1. beat its declared predecessor on net OOS Sharpe in at least two of BTC, SPY
   and GLD;
2. have positive mean Sharpe delta across those three development markets;
3. not worsen maximum drawdown in more than one development market;
4. report 5/10/25 bps cost sensitivity with no selectively omitted run;
5. pass all repository tests and store a manifest containing code revision,
   data hashes, parameters, seed and every prior model-selection trial;
6. state a directional lockbox prediction before the TLT data is loaded.

Champion inheritance, exponential weighting and both contextual-regret variants
failed these gates. The conservative screened expert pool at a 0.5 position cap
passed all six development gates and was the sole authorized candidate.

## One-time evaluation

The one-time evaluator verified the TLT SHA-256, code hash and every earlier
selection-report hash before loading with `allow_lockbox=True`. It then ran the
frozen predecessor and candidate with seed 42 and 10 bps costs. The candidate
improved Sharpe but worsened both total return and maximum drawdown, so its
primary prediction passed and its risk prediction failed. It was not promoted.
The TLT history is now spent evaluation data and must never be reused as a new
lockbox or tuning set.
