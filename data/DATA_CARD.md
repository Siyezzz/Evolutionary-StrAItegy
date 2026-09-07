# Bundled BTC-USD data card

- File: `market_btcusd_daily.csv`
- Rows: 2,556 daily observations
- Date range: 2018-01-01 through 2024-12-30
- Columns: date, adjusted close, volume
- Minimum/maximum close: 3,236.76171875 / 106,140.6015625
- Zero-volume rows: 0
- SHA-256: `4b6b530b61113f8b8ba9ee9492f98ce01149c7fde67a6eed5bbaa92652173951`

The values and header format are consistent with the repository's yfinance
loader, but the original commit does not contain a download receipt, upstream
version, retrieval timestamp or raw-source checksum. Therefore the provenance
cannot yet be independently audited. The file is acceptable for plumbing and
exploratory tests, not for a final research claim.

Before lockbox evaluation, regenerate a versioned snapshot from a declared
provider, record timezone/corporate-action conventions and retrieval timestamp,
and reconcile a random sample against a second source.

## Independent transfer snapshots

Downloaded on 2026-09-07 through `yfinance==0.2.66`, with `auto_adjust=True`,
daily interval, requested start `2010-01-01` and exclusive end `2025-01-01`:

| File | Asset | Rows | First / last date | SHA-256 |
|---|---|---:|---|---|
| `market_spy_daily.csv` | SPY US equity ETF | 3,774 | 2010-01-04 / 2024-12-31 | `a453dab49a341fafaae84a8f91016191482c65b8e87d7d531fe62f11f1df385e` |
| `market_gld_daily.csv` | GLD gold ETF | 3,774 | 2010-01-04 / 2024-12-31 | `f39df9f258f885e0f3b46a754c8e41603698e1176378606b4222055e0074a2fa` |

Both pass the repository's ordering, uniqueness, positive-price, finite-value
and non-negative-volume checks. They are independent asset histories but share
one data vendor, so provider-level errors remain correlated. A second-vendor
reconciliation is still required before lockbox promotion.

## Consumed lockbox

- Asset: TLT long-duration US Treasury ETF
- File: `lockbox_tlt_daily.csv`
- Rows: 3,774
- Date range: 2010-01-04 through 2024-12-31
- Retrieval: 2026-09-07, `yfinance==0.2.66`, adjusted daily data
- SHA-256: `9ae352abef1e7e41af70e4db494f1a5a1e95cb60961809154e0df5b70f6d8c16`
- Status: **opened exactly once on 2026-09-07; no longer eligible as a lockbox**

The normal loader still rejects files beginning with `lockbox_`. The only
authorized evaluation is recorded in `experiments/LOCKBOX_ACCESS_LOG.jsonl` and
`artifacts/lockbox_tlt_result.json`; the evaluator now refuses all reruns. TLT
must not be used for tuning or presented as fresh confirmation in later work.
