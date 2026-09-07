"""Execute the frozen TLT lockbox comparison exactly once.

The append-only access log is created before the protected CSV is loaded. Any
existing log or result permanently blocks a second execution.
"""

from __future__ import annotations

from datetime import datetime, timezone
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
AUTHORIZATION = ROOT / "experiments" / "lockbox_authorization.json"
ACCESS_LOG = ROOT / "experiments" / "LOCKBOX_ACCESS_LOG.jsonl"
RESULT = ROOT / "artifacts" / "lockbox_tlt_result.json"
CODE_PATHS = [
    "engine/evolve.py",
    "engine/bandit.py",
    "engine/regret.py",
    "backtest/simulate.py",
    "backtest/metrics.py",
    "experts/momentum.py",
    "experts/long_horizon_momentum.py",
    "experts/long_only_trend.py",
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_code() -> str:
    digest = hashlib.sha256()
    for relative in sorted(CODE_PATHS):
        digest.update(relative.encode("utf-8"))
        digest.update((ROOT / relative).read_bytes())
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _verify_development_gate(authorization: dict) -> None:
    gate = authorization["development_gate"]
    old_report = json.loads((ROOT / gate["predecessor_report"]).read_text())
    candidate_report = json.loads((ROOT / gate["candidate_report"]).read_text())
    cost = float(gate["comparison_cost_bps"])
    old = {
        row["symbol"]: row
        for row in old_report["rows"]
        if row["pool"] == "old"
    }
    candidate = {
        row["symbol"]: row
        for row in candidate_report["rows"]
        if float(row["cost_bps"]) == cost
    }
    if old.keys() != candidate.keys():
        raise RuntimeError("development reports cover different markets")
    sharpe_deltas = [candidate[s]["sharpe"] - old[s]["sharpe"] for s in old]
    drawdown_no_worse = sum(
        candidate[s]["max_drawdown"] >= old[s]["max_drawdown"] for s in old
    )
    if sum(delta > 0.0 for delta in sharpe_deltas) < gate["sharpe_wins_required"]:
        raise RuntimeError("development Sharpe gate failed")
    if float(np.mean(sharpe_deltas)) <= 0.0:
        raise RuntimeError("development mean Sharpe gate failed")
    if drawdown_no_worse < gate["drawdown_no_worse_required"]:
        raise RuntimeError("development drawdown gate failed")
    if not all(candidate_report["return_monotonic_by_market"].values()):
        raise RuntimeError("development cost-sensitivity gate failed")


def _verify_frozen_inputs(authorization: dict) -> None:
    if authorization["status"] != "authorized":
        raise RuntimeError("lockbox manifest is not authorized")
    if _hash_code() != authorization["frozen_code_sha256"]:
        raise RuntimeError("frozen implementation hash mismatch")
    lockbox = authorization["lockbox"]
    if _sha256(ROOT / lockbox["path"]) != lockbox["sha256"]:
        raise RuntimeError("lockbox data hash mismatch")
    for relative, expected in authorization["prior_model_selection_trials"].items():
        if _sha256(ROOT / relative) != expected:
            raise RuntimeError(f"prior trial hash mismatch: {relative}")
    _verify_development_gate(authorization)


def _evaluate(data: dict, common: dict, spec: dict) -> dict:
    config = _coerce(load_config(str(ROOT / "config.yaml")))
    config.update(common)
    config["max_abs_position"] = spec["max_abs_position"]
    experts = build_experts(spec["experts"])
    pnls, epochs, trace = walk_forward(
        data,
        experts,
        config,
        np.random.default_rng(common["seed"]),
        return_trace=True,
    )
    return {
        "experts": spec["experts"],
        "max_abs_position": spec["max_abs_position"],
        "epochs": len(epochs),
        "steps": len(pnls),
        **performance_metrics(
            pnls,
            periods_per_year=common["periods_per_year"],
            positions=trace["pos"],
            turnovers=trace["turnover"],
        ),
    }


def run() -> dict:
    if ACCESS_LOG.exists() or RESULT.exists():
        raise RuntimeError("TLT lockbox has already been accessed; rerun refused")
    authorization = json.loads(AUTHORIZATION.read_text(encoding="utf-8"))
    _verify_frozen_inputs(authorization)
    auth_hash = _sha256(AUTHORIZATION)

    ACCESS_LOG.parent.mkdir(exist_ok=True)
    with ACCESS_LOG.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "opened",
                    "timestamp_utc": _utc_now(),
                    "authorization_sha256": auth_hash,
                    "lockbox_sha256": authorization["lockbox"]["sha256"],
                },
                sort_keys=True,
            )
            + "\n"
        )

    data = load_csv(
        str(ROOT / authorization["lockbox"]["path"]), allow_lockbox=True
    )
    frozen = authorization["frozen_comparison"]
    predecessor = _evaluate(data, frozen["common"], frozen["predecessor"])
    candidate = _evaluate(data, frozen["common"], frozen["candidate"])
    result = {
        "status": "completed",
        "authorization_sha256": auth_hash,
        "completed_at_utc": _utc_now(),
        "symbol": "TLT",
        "predecessor": predecessor,
        "candidate": candidate,
        "paired": {
            "sharpe_delta_candidate_minus_predecessor": (
                candidate["sharpe"] - predecessor["sharpe"]
            ),
            "return_delta_candidate_minus_predecessor": (
                candidate["total_return"] - predecessor["total_return"]
            ),
            "drawdown_delta_candidate_minus_predecessor": (
                candidate["max_drawdown"] - predecessor["max_drawdown"]
            ),
        },
        "prediction_outcome": {
            "primary_sharpe": candidate["sharpe"] > predecessor["sharpe"],
            "risk_drawdown": (
                candidate["max_drawdown"] >= predecessor["max_drawdown"]
            ),
        },
    }
    RESULT.parent.mkdir(exist_ok=True)
    with RESULT.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    with ACCESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "event": "completed",
                    "timestamp_utc": result["completed_at_utc"],
                    "result": str(RESULT.relative_to(ROOT)).replace("\\", "/"),
                    "result_sha256": _sha256(RESULT),
                },
                sort_keys=True,
            )
            + "\n"
        )
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
