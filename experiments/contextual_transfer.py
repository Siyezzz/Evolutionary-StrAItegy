"""Frozen one-seed transfer test of contextual versus global regret."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

import numpy as np

from backtest.metrics import metrics_by_context, performance_metrics
from backtest.simulate import walk_forward
from data.fetch_market import load_csv
from engine import load_config
from evolve_run import _coerce
from experts.mean_reversion import MeanReversionExpert
from experts.momentum import MomentumExpert
from experts.news_sentiment import NewsSentimentExpert


ROOT = Path(__file__).resolve().parents[1]


def run(context_method: str = "fixed") -> dict:
    base = _coerce(load_config(str(ROOT / "config.yaml")))
    base.update(
        {
            "es_population": 3,
            "es_generations": 1,
            "transaction_cost_bps": 10.0,
            "inherit_champion": False,
            "allocator": "legacy",
            "context_method": context_method,
        }
    )
    seed = 42
    datasets = {
        "BTC-USD": ("data/market_btcusd_daily.csv", 365),
        "SPY": ("data/market_spy_daily.csv", 252),
        "GLD": ("data/market_gld_daily.csv", 252),
    }
    modes = ["global", "contextual"]
    manifest = {
        "seed": seed,
        "modes": modes,
        "cost_bps": 10.0,
        "es_population": 3,
        "es_generations": 1,
        "context_shrinkage": float(base["context_shrinkage"]),
        "context_method": context_method,
        "datasets": {
            name: {
                "path": path,
                "periods_per_year": periods,
                "sha256": hashlib.sha256((ROOT / path).read_bytes()).hexdigest(),
            }
            for name, (path, periods) in datasets.items()
        },
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    experts = [MomentumExpert(), MeanReversionExpert(), NewsSentimentExpert()]
    rows = []
    for symbol, (path, periods) in datasets.items():
        data = load_csv(str(ROOT / path))
        for mode in modes:
            cfg = dict(base)
            cfg.update(
                {
                    "symbol": symbol,
                    "data_csv": path,
                    "periods_per_year": periods,
                    "regret_mode": mode,
                }
            )
            pnls, epochs, trace = walk_forward(
                data,
                experts,
                cfg,
                np.random.default_rng(seed),
                return_trace=True,
            )
            metrics = performance_metrics(
                pnls,
                periods_per_year=periods,
                positions=trace["pos"],
                turnovers=trace["turnover"],
            )
            by_context = metrics_by_context(
                pnls, trace["context"], periods_per_year=periods
            )
            rows.append(
                {
                    "symbol": symbol,
                    "mode": mode,
                    "epochs": len(epochs),
                    "metrics": metrics,
                    "context_metrics": by_context,
                }
            )
    paired = []
    for symbol in datasets:
        pair = {row["mode"]: row for row in rows if row["symbol"] == symbol}
        paired.append(
            {
                "symbol": symbol,
                "sharpe_delta_contextual_minus_global": (
                    pair["contextual"]["metrics"]["sharpe"]
                    - pair["global"]["metrics"]["sharpe"]
                ),
                "return_delta_contextual_minus_global": (
                    pair["contextual"]["metrics"]["total_return"]
                    - pair["global"]["metrics"]["total_return"]
                ),
                "drawdown_delta_contextual_minus_global": (
                    pair["contextual"]["metrics"]["max_drawdown"]
                    - pair["global"]["metrics"]["max_drawdown"]
                ),
            }
        )
    summary = {
        "contextual_sharpe_wins": sum(
            item["sharpe_delta_contextual_minus_global"] > 0
            for item in paired
        ),
        "markets": len(paired),
        "mean_sharpe_delta": float(
            np.mean([item["sharpe_delta_contextual_minus_global"] for item in paired])
        ),
        "mean_return_delta": float(
            np.mean([item["return_delta_contextual_minus_global"] for item in paired])
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
    (output / f"contextual_transfer_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--context-method", choices=["fixed", "ranked"], default="fixed"
    )
    run(parser.parse_args().context_method)
