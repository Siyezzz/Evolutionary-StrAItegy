"""Single evolution run — the entry point a scheduler would call every N hours.

What it does:
    1. load configuration (config.yaml)
    2. load market data (real CSV, or synthetic fallback if missing)
    3. build the expert gene pool
    4. run the walk-forward self-evolution
    5. write an out-of-sample report + persist memory (performance.json, regime log)
    6. print a concise summary

Run with:  python evolve_run.py
"""

from __future__ import annotations

import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from engine import load_config  # noqa: E402
from data.fetch_market import load_csv, fetch_yfinance  # noqa: E402
from data.generate_synthetic import generate  # noqa: E402
from experts.factory import build_experts  # noqa: E402
from backtest.simulate import walk_forward  # noqa: E402
from backtest.metrics import metrics_by_context, performance_metrics  # noqa: E402
from backtest.benchmarks import static_expert_mix  # noqa: E402
from engine.evolve import precompute_signals  # noqa: E402

_NUMERIC = {
    "train_window": int,
    "test_horizon": int,
    "min_train": int,
    "es_population": int,
    "es_generations": int,
    "es_sigma": float,
    "risk": float,
    "vol_target": float,
    "blend_init": float,
    "transaction_cost_bps": float,
    "periods_per_year": int,
    "max_abs_position": float,
    "seed": int,
}


def _coerce(cfg: dict) -> dict:
    for k, fn in _NUMERIC.items():
        if k in cfg:
            cfg[k] = fn(cfg[k])
    if "inherit_champion" in cfg and isinstance(cfg["inherit_champion"], str):
        cfg["inherit_champion"] = cfg["inherit_champion"].lower() in {
            "1", "true", "yes", "on"
        }
    return cfg


def _metrics(returns: np.ndarray) -> dict:
    """Backward-compatible wrapper used by older callers and tests."""
    return performance_metrics(returns)


def main() -> None:
    cfg = _coerce(load_config(os.path.join(HERE, "config.yaml")))
    rng = np.random.default_rng(int(cfg.get("seed", 42)))

    csv_path = os.path.join(HERE, cfg["data_csv"])
    if not os.path.exists(csv_path):
        print(f"[data] {csv_path} not found -> generating synthetic market (stand-in).")
        generate(csv_path)
        source = "synthetic"
    else:
        source = "csv"

    data = load_csv(csv_path)
    experts = build_experts(cfg["experts"])

    oos, epochs, trace = walk_forward(
        data, experts, cfg, rng, return_trace=True
    )

    arr = np.array(oos, dtype=float)
    eq = np.cumprod(1.0 + arr)
    periods = int(cfg.get("periods_per_year", 365))
    strategy_metrics = performance_metrics(
        arr,
        periods_per_year=periods,
        positions=trace["pos"],
        turnovers=trace["turnover"],
    )
    context_metrics = metrics_by_context(
        arr, trace["context"], periods_per_year=periods
    )
    total_return = strategy_metrics["total_return"]
    sharpe = strategy_metrics["sharpe"]
    max_dd = strategy_metrics["max_drawdown"]

    first_oos = int(cfg["train_window"])
    close = data["close"]
    benchmark_returns = close[first_oos:] / close[first_oos - 1 : -1] - 1.0
    benchmark_metrics = performance_metrics(
        benchmark_returns[: len(arr)], periods_per_year=periods
    )
    signals = precompute_signals(data["close"], data["volume"], experts)
    static_metrics = {}
    static_weights = {
        "equal_weight_experts": {
            expert.name: 1.0 / len(experts) for expert in experts
        }
    }
    static_weights.update(
        {
            f"fixed_{expert.name}": {
                candidate.name: float(candidate.name == expert.name)
                for candidate in experts
            }
            for expert in experts
        }
    )
    for name, weights in static_weights.items():
        result = static_expert_mix(
            data["close"],
            trace["t"],
            signals,
            weights,
            risk=float(cfg["risk"]),
            vol_target=float(cfg["vol_target"]),
            transaction_cost_bps=float(cfg.get("transaction_cost_bps", 0.0)),
            max_abs_position=float(cfg.get("max_abs_position", 1.0)),
        )
        static_metrics[name] = performance_metrics(
            result["returns"],
            periods_per_year=periods,
            positions=result["positions"],
            turnovers=result["turnovers"],
        )
    best_fixed_name = max(
        (name for name in static_metrics if name.startswith("fixed_")),
        key=lambda name: static_metrics[name]["sharpe"],
    )

    print("=" * 64)
    print("Evolutionary-StrAItegy — walk-forward out-of-sample report")
    print("=" * 64)
    print(f"data source        : {source}  ({len(data['close'])} daily bars)")
    print(f"out-of-sample steps: {len(arr)}")
    print(f"total OOS return   : {total_return * 100:.2f}%")
    print(f"OOS Sharpe         : {sharpe:.2f}")
    print(f"max drawdown       : {max_dd * 100:.2f}%")
    print(f"annualized vol     : {strategy_metrics['annualized_volatility'] * 100:.2f}%")
    print(f"Sortino            : {strategy_metrics['sortino']:.2f}")
    print(f"average exposure   : {strategy_metrics['average_abs_exposure']:.2f}")
    print(f"annual turnover    : {strategy_metrics['annualized_turnover']:.2f}x")
    print(
        f"buy & hold return  : {benchmark_metrics['total_return'] * 100:.2f}% "
        f"(Sharpe {benchmark_metrics['sharpe']:.2f}, "
        f"maxDD {benchmark_metrics['max_drawdown'] * 100:.2f}%)"
    )
    equal = static_metrics["equal_weight_experts"]
    print(
        f"equal experts      : {equal['total_return'] * 100:.2f}% "
        f"(Sharpe {equal['sharpe']:.2f})"
    )
    best_fixed = static_metrics[best_fixed_name]
    print(
        f"best fixed hindsight: {best_fixed_name.removeprefix('fixed_')} "
        f"{best_fixed['total_return'] * 100:.2f}% "
        f"(Sharpe {best_fixed['sharpe']:.2f})"
    )
    print(f"epochs             : {len(epochs)}")
    print(f"transaction costs  : {cfg.get('transaction_cost_bps', 0.0):.1f} bps/turnover")
    print("-" * 64)
    print("first 5 epochs (best meta-params found in-sample):")
    for e in epochs[:5]:
        p = e["params"]
        print(
            f"  epoch {e['epoch']:>2}  blend={p['blend']:.2f} risk={p['risk']:.2f} "
            f"vol_target={p['vol_target']:.3f}  -> OOS ret {e['epoch_return']*100:+.2f}%"
        )

    # --- persist artifacts & memory ----------------------------------------
    os.makedirs(os.path.join(HERE, "artifacts"), exist_ok=True)
    with open(os.path.join(HERE, "artifacts", "oos_equity.csv"), "w") as fh:
        fh.write("step,equity\n")
        for i, v in enumerate(eq):
            fh.write(f"{i},{v:.6f}\n")

    perf = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "symbol": cfg.get("symbol"),
        "data_source": source,
        "oos_steps": len(arr),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "metrics": strategy_metrics,
        "context_metrics": context_metrics,
        "buy_and_hold": benchmark_metrics,
        "static_benchmarks": static_metrics,
        "best_fixed_expert_hindsight": best_fixed_name,
        "epochs": epochs,
    }
    with open(os.path.join(HERE, "memory", "performance.json"), "w") as fh:
        json.dump(perf, fh, indent=2)

    with open(os.path.join(HERE, "memory", "regime_memory.md"), "a") as fh:
        fh.write(f"\n## Run {datetime.datetime.now(datetime.timezone.utc).isoformat()}\n")
        fh.write(
            f"- symbol={cfg.get('symbol')} source={source} oos_steps={len(arr)} "
            f"return={total_return*100:.2f}% sharpe={sharpe:.2f} maxDD={max_dd*100:.2f}%\n"
        )
        for e in epochs[:10]:
            p = e["params"]
            fh.write(
                f"  - epoch {e['epoch']}: blend={p['blend']:.2f} risk={p['risk']:.2f} "
                f"vol_target={p['vol_target']:.3f} ret={e['epoch_return']*100:+.2f}%\n"
            )

    print("\nArtifacts written to artifacts/ and memory/.")


if __name__ == "__main__":
    main()
