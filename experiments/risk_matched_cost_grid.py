"""Development-only risk-matched screened-pool experiment."""

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
PREREGISTRATION = ROOT / "experiments" / "risk_matched_preregistration.json"


def _hash_files(paths: list[str]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.encode("utf-8"))
        digest.update((ROOT / path).read_bytes())
    return digest.hexdigest()


def run() -> dict:
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    if preregistration["status"] != "frozen-before-run":
        raise RuntimeError("risk-matched experiment is not preregistered")
    if any("TLT" in market for market in preregistration["markets"]):
        raise RuntimeError("consumed TLT data is prohibited in development")

    base = _coerce(load_config(str(ROOT / "config.yaml")))
    base.update(
        {
            "es_population": 3,
            "es_generations": 1,
            "inherit_champion": False,
            "allocator": "legacy",
            "regret_mode": "global",
            "max_abs_position": 0.2,
        }
    )
    expert_names = preregistration["candidate"]["experts"]
    datasets = {
        "BTC-USD": ("data/market_btcusd_daily.csv", 365),
        "SPY": ("data/market_spy_daily.csv", 252),
        "GLD": ("data/market_gld_daily.csv", 252),
    }
    costs = [float(value) for value in preregistration["costs_bps"]]
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
        "preregistration": str(PREREGISTRATION.relative_to(ROOT)).replace("\\", "/"),
        "preregistration_sha256": hashlib.sha256(
            PREREGISTRATION.read_bytes()
        ).hexdigest(),
        "experts": expert_names,
        "seed": seed,
        "costs_bps": costs,
        "max_abs_position": 0.2,
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
            config = dict(base)
            config["transaction_cost_bps"] = cost
            config["periods_per_year"] = periods
            pnls, epochs, trace = walk_forward(
                data,
                experts,
                config,
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

    old_report = json.loads(
        (ROOT / preregistration["predecessor"]["report"]).read_text(
            encoding="utf-8"
        )
    )
    old = {
        row["symbol"]: row
        for row in old_report["rows"]
        if row["pool"] == preregistration["predecessor"]["pool"]
    }
    candidate = {
        row["symbol"]: row for row in rows if row["cost_bps"] == 10.0
    }
    paired = []
    for symbol in datasets:
        paired.append(
            {
                "symbol": symbol,
                "sharpe_delta": candidate[symbol]["sharpe"] - old[symbol]["sharpe"],
                "return_delta": (
                    candidate[symbol]["total_return"] - old[symbol]["total_return"]
                ),
                "drawdown_delta": (
                    candidate[symbol]["max_drawdown"] - old[symbol]["max_drawdown"]
                ),
                "annualized_volatility_ratio": (
                    candidate[symbol]["annualized_volatility"]
                    / old[symbol]["annualized_volatility"]
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
    summary = {
        "sharpe_wins": sum(row["sharpe_delta"] > 0.0 for row in paired),
        "drawdown_no_worse": sum(row["drawdown_delta"] >= 0.0 for row in paired),
        "volatility_ratio_in_0.75_1.50": sum(
            0.75 <= row["annualized_volatility_ratio"] <= 1.5 for row in paired
        ),
        "mean_sharpe_delta": float(np.mean([row["sharpe_delta"] for row in paired])),
        "mean_return_delta": float(np.mean([row["return_delta"] for row in paired])),
        "return_monotonic_by_market": monotonic,
    }
    report = {
        "run_id": run_id,
        "manifest": manifest,
        "rows": rows,
        "paired_at_10_bps": paired,
        "summary": summary,
    }
    output = ROOT / "artifacts"
    output.mkdir(exist_ok=True)
    (output / f"risk_matched_cost_grid_{run_id}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps({"run_id": run_id, "summary": summary}, indent=2))
    return report


if __name__ == "__main__":
    run()
