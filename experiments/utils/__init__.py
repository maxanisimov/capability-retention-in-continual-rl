try:
    from .sb3_policy_introspection import (
        extract_feature_actor_parameters_and_network,
    )
except ModuleNotFoundError:
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
    "DiscreteSAC",
    "DiscreteSACPolicy",
    "extract_feature_actor_parameters_and_network",
]
