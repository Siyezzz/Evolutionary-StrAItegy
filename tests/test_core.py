from __future__ import annotations

import unittest

import numpy as np

from backtest.simulate import walk_forward
from engine.evolve import precompute_signals, simulate_period
from engine.bandit import ExpertBandit, ExponentialWeightsAllocator
from engine.regret import ContextualRegretMatcher, RegretMatcher
from experts.base import Expert
from evolve_run import _metrics
from data.fetch_market import load_csv, validate_market_data
from engine.regime import precompute_contexts
from experts.long_horizon_momentum import LongHorizonMomentumExpert
from experts.long_only_trend import LongOnlyTrendExpert
from experiments.lockbox_tlt_once import run as run_lockbox


class LongExpert(Expert):
    name = "long"

    def signal(self, ctx: dict) -> float:
        return 1.0


class ShortExpert(Expert):
    name = "short"

    def signal(self, ctx: dict) -> float:
        return -1.0


class CoreTests(unittest.TestCase):
    def test_completed_lockbox_refuses_rerun(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "already been accessed"):
            run_lockbox()

    def test_final_position_respects_hard_bound(self) -> None:
        close = np.array([100.0, 100.0, 100.0, 100.0])
        volume = np.ones_like(close)
        experts = [LongExpert()]
        signals = precompute_signals(close, volume, experts)
        store: dict = {}
        simulate_period(
            close,
            volume,
            1,
            4,
            experts,
            {
                "blend": 0.5,
                "risk": 3.0,
                "vol_target": 0.02,
                "transaction_cost_bps": 0.0,
                "max_abs_position": 1.0,
            },
            ExpertBandit(["long"]),
            RegretMatcher(["long"]),
            signals,
            record=True,
            store=store,
        )
        self.assertLessEqual(max(np.abs(store["pos"])), 1.0)

    def test_long_horizon_experts_are_causal(self) -> None:
        close = np.linspace(100.0, 200.0, 300)
        volume = np.ones_like(close)
        extended = np.concatenate((close, np.array([1.0e9])))
        for expert in (LongHorizonMomentumExpert(), LongOnlyTrendExpert()):
            before = expert.signal({"t": 299, "close": close, "volume": volume})
            after = expert.signal(
                {"t": 299, "close": extended, "volume": np.ones_like(extended)}
            )
            self.assertEqual(before, after)

    def test_ranked_contexts_do_not_change_when_future_is_appended(self) -> None:
        close = np.linspace(100.0, 150.0, 300)
        volume = np.ones_like(close)
        before = precompute_contexts(close, volume, method="ranked")
        extended_close = np.concatenate((close, np.array([1000.0])))
        extended_volume = np.concatenate((volume, np.array([999.0])))
        after = precompute_contexts(
            extended_close, extended_volume, method="ranked"
        )
        np.testing.assert_array_equal(before, after[:-1])

    def test_lockbox_filename_requires_explicit_evaluator(self) -> None:
        with self.assertRaisesRegex(PermissionError, "sealed"):
            load_csv("data/lockbox_tlt_daily.csv")

    def test_contextual_regret_separates_opposite_states(self) -> None:
        matcher = ContextualRegretMatcher(["trend", "revert"], shrinkage=1.0)
        for _ in range(20):
            matcher.update({"trend": 1.0, "revert": 0.0}, context="up")
            matcher.update({"trend": 0.0, "revert": 1.0}, context="down")
        self.assertGreater(
            matcher.weights("up")["trend"], matcher.weights("up")["revert"]
        )
        self.assertGreater(
            matcher.weights("down")["revert"], matcher.weights("down")["trend"]
        )

    def test_exponential_allocator_learns_better_expert(self) -> None:
        allocator = ExponentialWeightsAllocator(["winner", "loser"])
        for _ in range(20):
            allocator.update_all({"winner": 0.01, "loser": -0.01})
        self.assertGreater(
            allocator.weights()["winner"], allocator.weights()["loser"]
        )

    def test_market_data_rejects_duplicate_dates(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            validate_market_data(
                {
                    "dates": np.array(["2024-01-01", "2024-01-01"]),
                    "close": np.array([100.0, 101.0]),
                    "volume": np.array([1.0, 1.0]),
                }
            )

    def test_drawdown_includes_initial_capital(self) -> None:
        metrics = _metrics(np.array([-0.25, 0.10]))
        self.assertAlmostEqual(metrics["max_drawdown"], -0.25)

    def test_regret_uses_played_strategy_payoff(self) -> None:
        matcher = RegretMatcher(["a", "b"])
        matcher.strategy = {"a": 0.9, "b": 0.1}
        matcher.update({"a": 1.0, "b": 0.0})
        self.assertAlmostEqual(matcher.regret["a"], 0.1)
        self.assertEqual(matcher.regret["b"], 0.0)

    def test_period_includes_test_boundary_return(self) -> None:
        close = np.array([100.0, 110.0, 121.0, 133.1])
        volume = np.ones_like(close)
        experts = [LongExpert()]
        signals = precompute_signals(close, volume, experts)
        params = {
            "blend": 0.5,
            "risk": 1.0,
            "vol_target": 0.02,
            "transaction_cost_bps": 0.0,
        }
        pnls = simulate_period(
            close,
            volume,
            2,
            4,
            experts,
            params,
            ExpertBandit(["long"]),
            RegretMatcher(["long"]),
            signals,
        )
        self.assertEqual(len(pnls), 2)

    def test_turnover_cost_is_charged(self) -> None:
        close = np.array([100.0, 100.0])
        volume = np.ones_like(close)
        experts = [LongExpert()]
        signals = precompute_signals(close, volume, experts)
        params = {
            "blend": 0.5,
            "risk": 1.0,
            "vol_target": 0.02,
            "transaction_cost_bps": 10.0,
        }
        pnls = simulate_period(
            close,
            volume,
            1,
            2,
            experts,
            params,
            ExpertBandit(["long"]),
            RegretMatcher(["long"]),
            signals,
        )
        self.assertLess(pnls[0], 0.0)

    def test_walk_forward_covers_every_oos_return(self) -> None:
        close = np.linspace(100.0, 200.0, 22)
        data = {
            "close": close,
            "volume": np.ones_like(close),
            "dates": np.arange(len(close)),
        }
        config = {
            "train_window": 10,
            "test_horizon": 5,
            "min_train": 5,
            "es_population": 2,
            "es_generations": 1,
            "es_sigma": 0.1,
            "blend_init": 0.5,
            "risk": 1.0,
            "vol_target": 0.02,
            "transaction_cost_bps": 0.0,
        }
        pnls, epochs = walk_forward(
            data, [LongExpert()], config, np.random.default_rng(7)
        )
        self.assertEqual(len(pnls), len(close) - config["train_window"])
        self.assertEqual(len(epochs), 3)


if __name__ == "__main__":
    unittest.main()
