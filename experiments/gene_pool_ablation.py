"""Paired old-vs-evidence-screened expert pool transfer experiment."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from backtest.metrics import performance_metrics
from backtest.simulate import walk_forward
from data.fetch_market import load_csv
from engine import load_config
from evolve_run import _coerce
from experts.factory import build_experts


ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    config = _coerce(load_config(str(ROOT / "config.yaml")))
    config.update(
        {
            "es_population": 3,
            "es_generations": 1,
            "transaction_cost_bps": 10.0,
            "inherit_champion": False,
            "allocator": "legacy",
            "regret_mode": "global",
        }
    )
    pools = {
        "old": ["momentum", "mean_reversion", "news_sentiment"],
        "screened": [
            "momentum", "long_horizon_momentum", "long_only_trend"
        ],
    }
    datasets = {
        "BTC-USD": ("data/market_btcusd_daily.csv", 365),
        "SPY": ("data/market_spy_daily.csv", 252),
        "GLD": ("data/market_gld_daily.csv", 252),
    }
    seed = 42
    manifest = {
        "pools": pools,
        "seed": seed,
        "cost_bps": 10.0,
        "max_abs_position": 1.0,
        "es_population": 3,
        "es_generations": 1,
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
        for pool_name, expert_names in pools.items():
            cfg = dict(config)
            cfg["periods_per_year"] = periods
            experts = build_experts(expert_names)
            pnls, epochs, trace = walk_forward(
                data,
                experts,
                cfg,
                np.random.default_rng(seed),
                return_trace=True,
            )
            rows.append(
                {
                    "symbol": symbol,
                    "pool": pool_name,
                    "epochs": len(epochs),
                    **performance_metrics(
                        pnls,
                        periods_per_year=periods,
                        positions=trace["pos"],
                        turnovers=trace["turnover"],
                    ),
                }
            )
    paired = []
    for symbol in datasets:
        pair = {row["pool"]: row for row in rows if row["symbol"] == symbol}
        paired.append(
            {
                "symbol": symbol,
                "sharpe_delta_screened_minus_old": (
                    pair["screened"]["sharpe"] - pair["old"]["sharpe"]
                ),
                "return_delta_screened_minus_old": (
                    pair["screened"]["total_return"]
                    - pair["old"]["total_return"]
                ),
                "drawdown_delta_screened_minus_old": (
                    pair["screened"]["max_drawdown"]
                    - pair["old"]["max_drawdown"]
                ),
            }
        )
    summary = {
        "screened_sharpe_wins": sum(
            item["sharpe_delta_screened_minus_old"] > 0 for item in paired
        ),
        "markets": len(paired),
        "mean_sharpe_delta": float(
            np.mean([item["sharpe_delta_screened_minus_old"] for item in paired])
        ),
        "mean_return_delta": float(
            np.mean([item["return_delta_screened_minus_old"] for item in paired])
        ),
    }
    report = {
        "run_id": run_id,
        "manifest": manifest,
        "rows": rows,
        "paired": paired,
        "summary": summary,
    }
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / f"gene_pool_ablation_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    return report


if __name__ == "__main__":
    run()
