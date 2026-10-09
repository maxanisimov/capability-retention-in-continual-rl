"""Historical feature-observation initialiser interface."""
import inspect
import unittest
from projects.safe_policy_optimisation.archive.scripts.run_frozenlake_features_pspo import build_safe_actor_features

class FeatureInitialiserTests(unittest.TestCase):
    def test_initialiser_cannot_accept_reward_or_goal_inputs(self):
        parameters = inspect.signature(build_safe_actor_features).parameters
        self.assertEqual(set(parameters), {"env", "mask", "epochs", "seed"})
        for forbidden in ("reward", "goal", "preferred", "witness", "distance"):
            self.assertNotIn(forbidden, parameters)
