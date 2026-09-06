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
from experts.momentum import MomentumExpert  # noqa: E402
from experts.mean_reversion import MeanReversionExpert  # noqa: E402
from experts.news_sentiment import NewsSentimentExpert  # noqa: E402
from backtest.simulate import walk_forward  # noqa: E402

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
    "seed": int,
}


def _coerce(cfg: dict) -> dict:
    for k, fn in _NUMERIC.items():
        if k in cfg:
            cfg[k] = fn(cfg[k])
    return cfg


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
    experts = [MomentumExpert(), MeanReversionExpert(), NewsSentimentExpert()]

    oos, epochs = walk_forward(data, experts, cfg, rng)

    arr = np.array(oos, dtype=float)
    eq = np.cumprod(1.0 + arr)
    total_return = float(eq[-1] - 1.0) if len(eq) else 0.0
    sharpe = float(arr.mean() / (arr.std() + 1e-9) * np.sqrt(252)) if len(arr) > 1 else 0.0
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0

    print("=" * 64)
    print("Evolutionary-StrAItegy — walk-forward out-of-sample report")
    print("=" * 64)
    print(f"data source        : {source}  ({len(data['close'])} daily bars)")
    print(f"out-of-sample steps: {len(arr)}")
    print(f"total OOS return   : {total_return * 100:.2f}%")
    print(f"OOS Sharpe         : {sharpe:.2f}")
    print(f"max drawdown       : {max_dd * 100:.2f}%")
    print(f"epochs             : {len(epochs)}")
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
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "symbol": cfg.get("symbol"),
        "data_source": source,
        "oos_steps": len(arr),
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
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
