"""Paired legacy-vs-exponential allocator experiment."""

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
from experts.mean_reversion import MeanReversionExpert
from experts.momentum import MomentumExpert
from experts.news_sentiment import NewsSentimentExpert


ROOT = Path(__file__).resolve().parents[1]


def run() -> dict:
    config = _coerce(load_config(str(ROOT / "config.yaml")))
    config.update(
        {
            "es_population": 3,
            "es_generations": 1,
            "transaction_cost_bps": 10.0,
            "inherit_champion": False,
        }
    )
    seeds = [7, 42, 101]
    variants = ["legacy", "exponential"]
    data_path = ROOT / str(config["data_csv"])
    data = load_csv(str(data_path))
    experts = [MomentumExpert(), MeanReversionExpert(), NewsSentimentExpert()]
    manifest = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "seeds": seeds,
        "variants": variants,
        "transaction_cost_bps": 10.0,
        "es_population": 3,
        "es_generations": 1,
        "inherit_champion": False,
        "window_mode": config["window_mode"],
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    rows = []
    for seed in seeds:
        for allocator in variants:
            cfg = dict(config)
            cfg["seed"] = seed
            cfg["allocator"] = allocator
            pnls, epochs, trace = walk_forward(
                data,
                experts,
                cfg,
                np.random.default_rng(seed),
                return_trace=True,
            )
            rows.append(
                {
                    "seed": seed,
                    "allocator": allocator,
                    "epochs": len(epochs),
                    **performance_metrics(
                        pnls,
                        periods_per_year=int(cfg["periods_per_year"]),
                        positions=trace["pos"],
                        turnovers=trace["turnover"],
                    ),
                }
            )
    paired = []
    for seed in seeds:
        pair = {row["allocator"]: row for row in rows if row["seed"] == seed}
        paired.append(
            {
                "seed": seed,
                "sharpe_delta_exponential_minus_legacy": (
                    pair["exponential"]["sharpe"] - pair["legacy"]["sharpe"]
                ),
                "return_delta_exponential_minus_legacy": (
                    pair["exponential"]["total_return"]
                    - pair["legacy"]["total_return"]
                ),
            }
        )
    summary = {
        "exponential_sharpe_wins": sum(
            item["sharpe_delta_exponential_minus_legacy"] > 0 for item in paired
        ),
        "paired_cases": len(paired),
        "mean_sharpe_delta": float(
            np.mean([item["sharpe_delta_exponential_minus_legacy"] for item in paired])
        ),
        "mean_return_delta": float(
            np.mean([item["return_delta_exponential_minus_legacy"] for item in paired])
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
    path = output / f"allocator_ablation_{run_id}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    return report


if __name__ == "__main__":
    run()
