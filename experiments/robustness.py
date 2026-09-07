"""Paired multi-seed, multi-cost inheritance ablation.

This is intentionally a modest research smoke test, not a promotion test. Run:

    python -m experiments.robustness
"""

from __future__ import annotations

import csv
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
    base = _coerce(load_config(str(ROOT / "config.yaml")))
    # Small enough for routine validation; the frozen lockbox protocol must use
    # a separately declared, larger search budget.
    base["es_population"] = 3
    base["es_generations"] = 1
    seeds = [7, 42, 101]
    costs = [5.0, 10.0, 25.0]
    data_path = ROOT / str(base["data_csv"])
    data = load_csv(str(data_path))
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    experts = [MomentumExpert(), MeanReversionExpert(), NewsSentimentExpert()]

    manifest = {
        "data_sha256": data_hash,
        "seeds": seeds,
        "transaction_cost_bps": costs,
        "variants": ["inherit", "restart"],
        "es_population": base["es_population"],
        "es_generations": base["es_generations"],
        "window_mode": base["window_mode"],
        "train_window": base["train_window"],
        "test_horizon": base["test_horizon"],
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    rows = []
    for cost in costs:
        for seed in seeds:
            for inherit in (True, False):
                cfg = dict(base)
                cfg["seed"] = seed
                cfg["transaction_cost_bps"] = cost
                cfg["inherit_champion"] = inherit
                pnls, epochs, trace = walk_forward(
                    data,
                    experts,
                    cfg,
                    np.random.default_rng(seed),
                    return_trace=True,
                )
                metrics = performance_metrics(
                    pnls,
                    periods_per_year=int(cfg["periods_per_year"]),
                    positions=trace["pos"],
                    turnovers=trace["turnover"],
                )
                rows.append(
                    {
                        "run_id": run_id,
                        "seed": seed,
                        "transaction_cost_bps": cost,
                        "variant": "inherit" if inherit else "restart",
                        "epochs": len(epochs),
                        **metrics,
                    }
                )

    paired = []
    for cost in costs:
        for seed in seeds:
            pair = {
                row["variant"]: row
                for row in rows
                if row["seed"] == seed
                and row["transaction_cost_bps"] == cost
            }
            paired.append(
                {
                    "seed": seed,
                    "transaction_cost_bps": cost,
                    "sharpe_delta_inherit_minus_restart": (
                        pair["inherit"]["sharpe"] - pair["restart"]["sharpe"]
                    ),
                    "return_delta_inherit_minus_restart": (
                        pair["inherit"]["total_return"]
                        - pair["restart"]["total_return"]
                    ),
                }
            )
    summary = {
        "inherit_sharpe_wins": sum(
            item["sharpe_delta_inherit_minus_restart"] > 0 for item in paired
        ),
        "paired_cases": len(paired),
        "mean_sharpe_delta": float(
            np.mean([item["sharpe_delta_inherit_minus_restart"] for item in paired])
        ),
        "mean_return_delta": float(
            np.mean([item["return_delta_inherit_minus_restart"] for item in paired])
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
    json_path = output / f"robustness_{run_id}.json"
    csv_path = output / f"robustness_{run_id}.csv"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"run_id": run_id, **summary}, indent=2))
    return report


if __name__ == "__main__":
    run()
