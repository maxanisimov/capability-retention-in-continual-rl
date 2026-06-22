try:
    from .sb3_clamped_ppo import (
        ClampRule,
        ClampedPPO,
        extract_feature_actor_parameters_and_network,
    )
except ModuleNotFoundError:
    ClampRule = None
    ClampedPPO = None
    extract_feature_actor_parameters_and_network = None

try:
    from .sb3_discrete_sac import (
        DiscreteSAC,
        DiscreteSACPolicy,
    )
except ModuleNotFoundError:
    DiscreteSAC = None
    DiscreteSACPolicy = None

__all__ = [
    "ClampRule",
    "ClampedPPO",
    "DiscreteSAC",
    "DiscreteSACPolicy",
    "extract_feature_actor_parameters_and_network",
]
