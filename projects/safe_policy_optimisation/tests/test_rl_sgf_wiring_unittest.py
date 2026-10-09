"""RL-SGF factory and CLI wiring."""

from __future__ import annotations

import argparse
import unittest

import gymnasium as gym

from projects.safe_policy_optimisation.stages import train_ppo_lagrangian, train_rl_sgf
from projects.safe_policy_optimisation.utils.safe_rl import build_safe_rl_baseline
from safe_rl_baselines import CPO, RLSGF


class RLSGFWiringTests(unittest.TestCase):
    def test_factory_forwards_rl_sgf_hyperparameters(self) -> None:
        model = build_safe_rl_baseline(
            "rl_sgf", gym.make("CartPole-v1"), cost_limit=1.0, seed=0, net_arch=(16,),
            rl_sgf_step_size=0.2, rl_sgf_alpha=2.0, rl_sgf_episodes_per_iter=7, rl_sgf_baseline="critic",
            cost_gamma=0.95,
        )
        self.assertIsInstance(model, RLSGF)
        self.assertEqual((model.step_size, model.alpha, model.episodes_per_iter, model.baseline),
                         (0.2, 2.0, 7, "critic"))
        self.assertEqual(model.cost_gamma, 0.95)

    def test_other_baselines_ignore_rl_sgf_keys(self) -> None:
        model = build_safe_rl_baseline(
            "cpo", gym.make("CartPole-v1"), cost_limit=1.0, seed=0, net_arch=(16,), rl_sgf_step_size=0.2,
        )
        self.assertIsInstance(model, CPO)

    def test_stage_parser_and_hyperparameter_forwarding(self) -> None:
        args = train_rl_sgf.build_parser().parse_args(
            ["--env-id", "CartPole-v1", "--rl-sgf-step-size", "0.3", "--rl-sgf-episodes-per-iter", "11"]
        )
        hp = train_ppo_lagrangian._baseline_hyperparameters_from_args(args)
        self.assertEqual(hp["rl_sgf_step_size"], 0.3)
        self.assertEqual(hp["rl_sgf_episodes_per_iter"], 11)
        self.assertEqual(hp["rl_sgf_baseline"], "none")
        self.assertEqual(args.algorithms, ["rl_sgf"])

    def test_ppo_lagrangian_parser_still_builds_hyperparameters(self) -> None:
        args = train_ppo_lagrangian.build_parser().parse_args(["--env-id", "CartPole-v1"])
        hp = train_ppo_lagrangian._baseline_hyperparameters_from_args(args)
        self.assertIsNone(hp["rl_sgf_step_size"])
        self.assertIsInstance(args, argparse.Namespace)


if __name__ == "__main__":
    unittest.main()
