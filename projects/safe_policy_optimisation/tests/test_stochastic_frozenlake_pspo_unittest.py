"""Exact safety, reachability and initial-actor preflight tests."""

from __future__ import annotations

import inspect
import unittest

import gymnasium as gym
import numpy as np

from projects.safe_policy_optimisation.scripts.run_stochastic_frozenlake_pspo import (
    INITIALISATION_SCHEME,
    build_safe_actor,
    default_output,
    episode_cap,
    goal_indicator,
    screen_name,
    training_arguments,
)
from projects.safe_policy_optimisation.utils.frozen_lake_experiment import (
    ENV_ID,
    SparseFrozenLake,
    proper_goal_policies,
    structured_layout,
    synthesise_shield,
)


class StochasticFrozenLakeTests(unittest.TestCase):
    def test_large_layout_geometry(self):
        grid = np.asarray([list(row) for row in structured_layout()])
        self.assertEqual(grid.shape, (128, 128))
        self.assertEqual(np.count_nonzero(grid == "H"), 2304)
        self.assertEqual(grid[0, 0], "S")
        self.assertEqual(grid[-1, -1], "G")
        self.assertTrue(np.all(grid[1, :] == "F"))
        self.assertTrue(np.all(grid[:, 1] == "F"))
        self.assertTrue(np.all(grid[64, :] == "F"))

    def test_doubling_size_doubles_the_unsafe_islands(self):
        """The scalability sweep's premise: holes scale with the layout."""
        previous = None
        for size in (128, 256, 512):
            grid = np.asarray([list(row) for row in structured_layout(size)])
            self.assertEqual(grid.shape, (size, size))
            holes = np.count_nonzero(grid == "H")
            # Four square islands, so each island edge is sqrt(holes / 4).
            edge = int(round((holes / 4) ** 0.5))
            self.assertEqual(4 * edge * edge, holes)
            self.assertEqual(edge, 3 * size // 16)
            if previous is not None:
                self.assertEqual(edge, 2 * previous)
            previous = edge
            # Geometry is fractional, so the hole fraction is size-invariant.
            self.assertAlmostEqual(holes / size**2, 0.140625, places=6)
            self.assertEqual(grid[0, 0], "S")
            self.assertEqual(grid[-1, -1], "G")

    def test_size_derived_run_identifiers_and_horizon(self):
        self.assertEqual(screen_name(256), "pspo-frozenlake256-safety-only")
        self.assertEqual(
            screen_name(256, "state_id_lookup"),
            "pspo-frozenlake256-safety-only-ids",
        )
        self.assertEqual(
            default_output(256).name,
            "pspo_stochastic_frozenlake256_safety_only",
        )
        self.assertEqual(
            default_output(256, "state_id_lookup").name,
            "pspo_stochastic_frozenlake256_safety_only_state_id_lookup",
        )
        self.assertNotEqual(default_output(128), default_output(256))
        # The horizon must outgrow the 2*(size-1) start-to-goal distance.
        for size in (128, 256, 512):
            self.assertEqual(episode_cap(size), 5000 * size // 128)
            self.assertGreater(episode_cap(size), 4 * 2 * (size - 1))

    def test_shield_is_closed_under_every_positive_outcome(self):
        env = SparseFrozenLake(size=16)
        mask, winning, successors, probabilities = synthesise_shield(env)
        self.assertTrue(winning[0])
        self.assertFalse(winning[env.desc.reshape(-1) == b"H"].any())
        for state, action in zip(*np.nonzero(mask)):
            self.assertTrue(winning[successors[state, action][probabilities[state, action] > 0]].all())
        self.assertTrue(np.allclose(probabilities.sum(axis=2), 1))
        self.assertTrue(np.allclose(probabilities[0, 1], [0.1, 0.8, 0.1]))
        self.assertLess(successors.nbytes + probabilities.nbytes, 100_000)

    def test_distinct_witnesses_have_strict_progress_and_safety(self):
        env = SparseFrozenLake(size=16)
        mask, winning, successors, probabilities = synthesise_shield(env)
        goal = env._n_states - 1
        distances, policies, proof = proper_goal_policies(mask, winning, successors, probabilities, goal)
        self.assertEqual(len(policies), 4)
        self.assertTrue(proof["almost_sure_goal_reachability"])
        self.assertGreater(min(proof["pairwise_policy_differing_winning_states"]), 0)
        states = np.flatnonzero(winning)
        states = states[states != goal]
        for policy in policies:
            self.assertTrue(mask[states, policy[states]].all())
            selected = successors[states, policy[states]]
            positive = probabilities[states, policy[states]] > 0
            self.assertTrue(np.any(positive & (distances[selected] < distances[states, None]), axis=1).all())

    def test_analytic_actor_uses_only_equal_safe_action_logits(self):
        env = SparseFrozenLake(size=16)
        mask, winning, _, _ = synthesise_shield(env)
        payload, audit = build_safe_actor(mask)
        self.assertEqual(payload["architecture"]["n_hidden"], 2)
        self.assertEqual(payload["architecture"]["input_dim"], 256)
        self.assertEqual(audit["all_winning_state_greedy_safety"], 1)
        self.assertEqual(audit["initialisation"], INITIALISATION_SCHEME)
        self.assertFalse(audit["reward_information_used"])
        self.assertFalse(audit["goal_information_used"])
        self.assertFalse(audit["witness_policy_used"])
        state = payload["state_dict"]
        import torch
        logits = torch.nn.functional.linear(torch.tanh(torch.nn.functional.linear(
            torch.tanh(state["0.weight"].T + state["0.bias"]),
            state["2.weight"], state["2.bias"],
        )), state["4.weight"], state["4.bias"])
        actual = logits.detach().numpy()
        np.testing.assert_allclose(actual[mask], 2.0, atol=1e-5)
        np.testing.assert_allclose(actual[~mask], -2.0, atol=1e-5)
        greedy = actual.argmax(axis=1)
        self.assertTrue(mask[np.flatnonzero(winning), greedy[winning]].all())

    def test_frozenlake_initialisers_cannot_accept_reward_or_goal_inputs(self):
        tabular = inspect.signature(build_safe_actor).parameters
        self.assertEqual(set(tabular), {"mask", "state_representation"})
        for parameters in (tabular,):
            for forbidden in ("reward", "goal", "preferred", "witness", "distance"):
                self.assertNotIn(forbidden, parameters)

    def test_goal_reward_and_hole_cost(self):
        env = SparseFrozenLake(desc=["SF", "HG"], is_slippery=False)
        env.reset(seed=0)
        _, reward, done, _, info = env.step(1)
        self.assertTrue(done)
        self.assertEqual(info["cost"], 1)
        self.assertAlmostEqual(reward, -0.001)
        env.reset(seed=0)
        _, reward, done, _, _ = env.step(2)
        self.assertFalse(done)
        self.assertAlmostEqual(reward, -0.001)
        _, reward, done, _, info = env.step(1)
        self.assertTrue(done)
        self.assertTrue(info["success"])
        self.assertAlmostEqual(reward, 0.999)

    def test_module_qualified_gym_registration(self):
        env = gym.make(ENV_ID, size=16, max_episode_steps=10)
        self.assertEqual(env.observation_space.n, 256)
        self.assertEqual(env.action_space.n, 4)
        env.close()

    def test_preparation_keeps_exhaustive_certificate(self):
        # No certificate-samples flag: the stage's None default certifies all W.
        import inspect
        self.assertNotIn('"--certificate-samples"', inspect.getsource(training_arguments))

    def test_goal_success_is_not_positive_return(self):
        self.assertEqual(goal_indicator(-1.0, 2000, 0.001), 1)
        self.assertEqual(goal_indicator(-5.0, 5000, 0.001), 0)
        self.assertEqual(goal_indicator(0.6, 400, 0.001), 1)


if __name__ == "__main__":
    unittest.main()
