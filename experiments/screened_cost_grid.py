"""Frozen conservative screened-pool cost sensitivity experiment."""

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


def _hash_files(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def run() -> dict:
    base = _coerce(load_config(str(ROOT / "config.yaml")))
    base.update(
        {
            "es_population": 3,
            "es_generations": 1,
            "inherit_champion": False,
            "allocator": "legacy",
            "regret_mode": "global",
            "max_abs_position": 0.5,
        }
    )
    expert_names = ["momentum", "long_horizon_momentum", "long_only_trend"]
    datasets = {
        "BTC-USD": ("data/market_btcusd_daily.csv", 365),
        "SPY": ("data/market_spy_daily.csv", 252),
        "GLD": ("data/market_gld_daily.csv", 252),
    }
    costs = [5.0, 10.0, 25.0]
    seed = 42
    code_paths = [
        "engine/evolve.py",
        "engine/bandit.py",
        "engine/regret.py",
        "backtest/simulate.py",
        "backtest/metrics.py",
        "experts/momentum.py",
        "experts/long_horizon_momentum.py",
        "experts/long_only_trend.py",
    ]
    manifest = {
        "prediction": (
            "At 10 bps, Sharpe improves versus the old pool in >=2/3 markets; "
            "drawdown is no worse in >=2/3; return is non-increasing with cost."
        ),
        "experts": expert_names,
        "seed": seed,
        "costs_bps": costs,
        "max_abs_position": 0.5,
        "es_population": 3,
        "es_generations": 1,
        "code_sha256": _hash_files(code_paths),
        "datasets": {
            name: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for name, (path, _) in datasets.items()
        },
    }
    run_id = hashlib.sha256(
        json.dumps(manifest, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    experts = build_experts(expert_names)
    rows = []
    for symbol, (path, periods) in datasets.items():
        data = load_csv(str(ROOT / path))
        for cost in costs:
            cfg = dict(base)
            cfg["transaction_cost_bps"] = cost
            cfg["periods_per_year"] = periods
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
                    "cost_bps": cost,
                    "epochs": len(epochs),
                    **performance_metrics(
                        pnls,
                        periods_per_year=periods,
                        positions=trace["pos"],
                        turnovers=trace["turnover"],
                    ),
                }
            )
    monotonic = {}
    for symbol in datasets:
        ordered = sorted(
            (row for row in rows if row["symbol"] == symbol),
            key=lambda row: row["cost_bps"],
        )
        monotonic[symbol] = all(
            ordered[index]["total_return"] >= ordered[index + 1]["total_return"]
            for index in range(len(ordered) - 1)
        )
    report = {
        "run_id": run_id,
        "manifest": manifest,
        "rows": rows,
        "return_monotonic_by_market": monotonic,
    }
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / f"screened_cost_grid_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "return_monotonic": monotonic}, indent=2))
    return report


if __name__ == "__main__":
    run()
