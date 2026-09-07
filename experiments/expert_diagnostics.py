"""Cost-aware fixed-expert diagnostics on development markets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from backtest.benchmarks import static_expert_mix
from backtest.metrics import performance_metrics
from data.fetch_market import load_csv
from engine.evolve import precompute_signals
from experts.mean_reversion import MeanReversionExpert
from experts.momentum import MomentumExpert
from experts.news_sentiment import NewsSentimentExpert
from experts.long_horizon_momentum import LongHorizonMomentumExpert
from experts.long_only_trend import LongOnlyTrendExpert


ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    datasets = {
        "BTC-USD": ("data/market_btcusd_daily.csv", 365),
        "SPY": ("data/market_spy_daily.csv", 252),
        "GLD": ("data/market_gld_daily.csv", 252),
    }
    experts = [
        MomentumExpert(),
        MeanReversionExpert(),
        NewsSentimentExpert(),
        LongHorizonMomentumExpert(),
        LongOnlyTrendExpert(),
    ]
    manifest = {
        "first_oos_index": 504,
        "transaction_cost_bps": 10.0,
        "risk": 1.0,
        "vol_target": 0.02,
        "max_abs_position": 1.0,
        "experts": [expert.name for expert in experts],
        "datasets": {
            name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for name, (path, _) in datasets.items()
        },
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    rows = []
    for symbol, (path, periods) in datasets.items():
        data = load_csv(str(ROOT / path))
        signals = precompute_signals(data["close"], data["volume"], experts)
        indices = range(504, len(data["close"]))
        mixes = {
            expert.name: {
                candidate.name: float(candidate.name == expert.name)
                for candidate in experts
            }
            for expert in experts
        }
        mixes["equal"] = {
            expert.name: 1.0 / len(experts) for expert in experts
        }
        for name, weights in mixes.items():
            result = static_expert_mix(
                data["close"],
                indices,
                signals,
                weights,
                risk=1.0,
                vol_target=0.02,
                transaction_cost_bps=10.0,
                max_abs_position=1.0,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "expert": name,
                    **performance_metrics(
                        result["returns"],
                        periods_per_year=periods,
                        positions=result["positions"],
                        turnovers=result["turnovers"],
                    ),
                }
            )
    report = {"run_id": run_id, "manifest": manifest, "rows": rows}
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / f"expert_diagnostics_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "rows": rows}, indent=2))
    return report


if __name__ == "__main__":
    run()
