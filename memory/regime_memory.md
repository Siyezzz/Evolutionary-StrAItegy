# Regime Memory

This file is the long-term, append-only memory of the self-evolving system.
Each run appends a block describing which meta-parameters worked and in which
market regime, so future evolution cycles start from accumulated knowledge
("which skill works under which regime").

Format per run:
- a `## Run <timestamp>` heading
- one summary line (symbol, source, OOS steps, return, sharpe, maxDD)
- one line per epoch with the in-sample-optimal meta-parameters

## Run 2026-09-05T12:47:35.037189
- symbol=BTC-USD source=csv oos_steps=1953 return=12.59% sharpe=0.47 maxDD=-6.52%
  - epoch 0: blend=0.42 risk=0.34 vol_target=0.018 ret=+0.29%
  - epoch 1: blend=0.49 risk=0.34 vol_target=0.016 ret=+1.63%
  - epoch 2: blend=0.00 risk=0.30 vol_target=0.016 ret=+0.11%
  - epoch 3: blend=0.19 risk=0.30 vol_target=0.016 ret=-0.62%
  - epoch 4: blend=0.37 risk=0.40 vol_target=0.015 ret=-0.34%
  - epoch 5: blend=0.21 risk=0.46 vol_target=0.023 ret=-0.09%
  - epoch 6: blend=0.22 risk=0.30 vol_target=0.016 ret=+0.97%
  - epoch 7: blend=0.17 risk=0.30 vol_target=0.018 ret=+0.02%
  - epoch 8: blend=0.76 risk=0.30 vol_target=0.016 ret=+0.28%
  - epoch 9: blend=0.64 risk=0.30 vol_target=0.016 ret=-0.00%

<!-- New runtime observations append below; generated reports remain in artifacts/. -->
