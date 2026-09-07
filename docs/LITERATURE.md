# Literature map and implementation boundaries

This file records what the research literature supports, and equally important,
what it does **not** establish for this repository.

## Online learning and game theory

- Hart and Mas-Colell, [A Simple Adaptive Procedure Leading to Correlated
  Equilibrium](https://www.ma.huji.ac.il/~hart/abs/adapt.html), Econometrica 68
  (2000), introduce regret matching and prove convergence of empirical play to
  correlated equilibrium under their repeated-game procedure. Our current
  single learner facing a market return sequence does not satisfy enough of that
  setup to claim correlated- or Nash-equilibrium convergence.
- Freund and Schapire, [A Decision-Theoretic Generalization of On-Line Learning
  and an Application to Boosting](https://www.sciencedirect.com/science/article/pii/S002200009791504X),
  JCSS 55 (1997), study bounded-loss online resource allocation and
  multiplicative expert weighting. Because this simulator observes every
  expert's counterfactual reward after each day, "full-information expert
  advice" is a better description than "multi-armed bandit."
- Hart and Mas-Colell, [A General Class of Adaptive
  Strategies](https://www.sciencedirect.com/science/article/pii/S0022053100927467/pdf),
  Journal of Economic Theory 98 (2001), characterize Hannan-consistent adaptive
  strategies. A proper project claim should report external regret directly;
  portfolio return alone does not verify the no-regret property.

## Financial hypothesis

- Moskowitz, Ooi and Pedersen, [Time Series
  Momentum](https://fairmodel.econ.yale.edu/ec439/mosk.pdf), Journal of Financial
  Economics 104 (2012), report return persistence over roughly 1–12 months in 58
  liquid futures instruments and longer-horizon partial reversal. This supports
  testing a medium-horizon momentum expert, but neither their asset universe nor
  evidence directly validates a 20-day BTC signal.
- The present mean-reversion expert has no project-specific empirical support
  yet. It must remain a challenger and be tested net of costs, not described as
  an established premium.
- Hurst, Ooi and Pedersen, [A Century of Evidence on Trend-Following
  Investing](https://research.cbs.dk/en/publications/a-century-of-evidence-on-trend-following-investing/),
  report long-run trend-following evidence across equities, bonds, commodities
  and currencies. This motivates a longer-horizon cross-asset challenger, but
  does not select this repository's exact lookback or sizing rule.
- Faber, [A Quantitative Approach to Tactical Asset
  Allocation](https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID2403936_code649342.pdf?abstractid=962461&mirid=1),
  studies a simple moving-average long/cash allocation across asset classes.
  The new 200-day long-only challenger is a daily approximation that must be
  evaluated separately; it is not presented as a replication of Faber's result.

## Backtest validity

- White, [A Reality Check for Data
  Snooping](https://onlinelibrary.wiley.com/doi/pdf/10.1111%2F1468-0262.00152),
  Econometrica 68 (2000), formalizes the risk that reusing one history for model
  search makes an apparently best rule look predictive by chance.
- Harvey, Liu and Zhu, [...and the Cross-Section of Expected
  Returns](https://academic.oup.com/rfs/article-pdf/29/1/5/24450794/hhv059.pdf),
  Review of Financial Studies 29 (2016), show why conventional significance
  thresholds are inadequate after testing many financial factors. Every ES
  candidate counts toward the research trial budget even when only the winner is
  printed.
- Bailey, Borwein, López de Prado and Zhu, [The Probability of Backtest
  Overfitting](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf), propose
  combinatorially symmetric cross-validation for estimating selection
  overfitting. This is a later milestone; the immediate safeguards are frozen
  experiment manifests, paired comparisons and an untouched lockbox.

## Trading costs

- Frazzini, Israel and Moskowitz, [Trading Costs of Asset Pricing
  Anomalies](https://pages.stern.nyu.edu/~afrazzin/pdf/Trading%20Cost%20of%20Asset%20Pricing%20Anomalies%20-%20Frazzini%2C%20Israel%20and%20Moskowitz.pdf),
  use live institutional equity trading data to show that implementability and
  capacity depend materially on costs and the strategy style. Their estimates
  cannot be transplanted to BTC; this project therefore reports a sensitivity
  grid (5/10/25 bps) rather than pretending one cost is known.

## Promotion rule

Papers motivate hypotheses and evaluation methods; they do not validate this
implementation. A feature is promoted only when code tests, paired OOS evidence,
cost sensitivity and a final untouched period agree. Negative findings remain in
the evidence log and in SEA.
## Risk-managed momentum and volatility scaling

- Moreira and Muir, *Volatility-Managed Portfolios* (2017): portfolios that
  reduce exposure when realized volatility is high can improve Sharpe and
  investor utility across several factors. This supports testing a causal risk
  budget, but it does not establish an optimal target for this single-asset
  ensemble: https://doi.org/10.1111/jofi.12513
- Barroso and Santa-Clara, *Momentum Has Its Moments* (2015): momentum risk is
  time-varying and volatility scaling materially reduced historical momentum
  crashes in their equity strategy. Their 12% target and long-short construction
  are not parameters we may copy without a new validation:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2041429
- Daniel and Moskowitz, *Momentum Crashes* (2016): momentum losses cluster after
  market declines, in high-volatility panic states and during rebounds; a dynamic
  mean/variance-scaled implementation improved their evidence. This motivates
  explicit downside diagnostics, not an ex-post TLT regime rule:
  https://www.nber.org/papers/w20439

The one-time TLT failure also reveals a design issue not resolved by those
papers: the candidate and predecessor carried materially different realized
volatility. The next development-only ablation therefore compares a 0.2 hard cap
against the predecessor and predeclares an acceptable volatility-ratio band.
This is risk matching, not evidence that lower leverage creates alpha.
