# # ckn_controller/optimal_configs.py
#
# OPTIMAL_CONFIGS = {
#     # ==========================================================
#     # Error Budget E = 0.120 (Target Accuracy = 0.880)
#     # ==========================================================
#     (35, 0.120): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.55,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "resnet101"],
#             "theta": 0.60,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.75,
#         },
#     },
#
#     (50, 0.120): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.55,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "resnet101"],
#             "theta": 0.60,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.55,
#         },
#     },
#
#     (100, 0.120): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.55,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "resnet101"],
#             "theta": 0.60,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.55,
#         },
#     },
#
#     # ==========================================================
#     # Error Budget E = 0.140 (Target Accuracy = 0.860)
#     # ==========================================================
#     (35, 0.140): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.40,
#         },
#         "po2": {
#             "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
#             "theta": 0.45,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.45,
#         },
#     },
#
#     (50, 0.140): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.40,
#         },
#         "po2": {
#             "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
#             "theta": 0.45,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.40,
#         },
#     },
#
#     (100, 0.140): {
#         "sequential": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.40,
#         },
#         "po2": {
#             "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
#             "theta": 0.45,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "convnext_tiny"],
#             "theta": 0.40,
#         },
#     },
#
#     # ==========================================================
#     # Error Budget E = 0.160 (Target Accuracy = 0.840)
#     # ==========================================================
#     (35, 0.160): {
#         "sequential": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "convnext_tiny"],
#             "theta": 0.25,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#     },
#
#     (50, 0.160): {
#         "sequential": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "convnext_tiny"],
#             "theta": 0.25,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#     },
#
#     (100, 0.160): {
#         "sequential": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#         "po2": {
#             "models": ["resnet34", "resnet18", "convnext_tiny"],
#             "theta": 0.25,
#             "parallel_first_n": 2,
#         },
#         "spec_b": {
#             "models": ["resnet34", "resnet50"],
#             "theta": 0.25,
#         },
#     },
# }
#
#
# def get_optimal_config(cost: int, error: float, policy: str):
#     """
#     Returns the precomputed optimal configuration for:
#         - cost budget (C)
#         - error budget (E)
#         - policy ('sequential', 'po2', 'spec_b')
#
#     Example:
#         cfg = get_optimal_config(50, 0.120, "po2")
#     """
#     key = (int(cost), round(float(error), 3))
#     policy = str(policy).lower()
#
#     if key not in OPTIMAL_CONFIGS:
#         available = sorted(OPTIMAL_CONFIGS.keys())
#         raise ValueError(
#             f"No optimal configuration found for C={cost}, E={error}. "
#             f"Available keys: {available}"
#         )
#
#     if policy not in OPTIMAL_CONFIGS[key]:
#         available_policies = list(OPTIMAL_CONFIGS[key].keys())
#         raise ValueError(
#             f"No policy='{policy}' for C={cost}, E={error}. "
#             f"Available policies: {available_policies}"
#         )
#
#     return OPTIMAL_CONFIGS[key][policy]



# ckn_controller/optimal_configs.py

OPTIMAL_CONFIGS = {
    (25, 0.140): {
        "sequential": {
            "models": ["resnet34", "convnext_tiny"],
            "theta": 0.40,
        },
        "po2": {
            "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
            "theta": 0.45,
            "parallel_first_n": 2,
        },
        "spec_b": {
            "models": ["resnet34", "resnet50"],
            "theta": 0.45,
        },
    },

    # =====================================================
    # C = 50
    # =====================================================

    (50, 0.090): {
        "sequential": {
            "models": ["resnet34", "vit_b_16"],
            "theta": 0.90,
        },
        "po2": {
            "models": ["resnet18", "resnet34", "vit_b_16"],
            "theta": 0.95,
            "parallel_first_n": 2,
        },
        # spec_b infeasible for C=50, E=0.090
    },

    (50, 0.100): {
        "sequential": {
            "models": ["resnet34", "resnet101"],
            "theta": 0.75,
        },
        "po2": {
            "models": ["resnet34", "mobilenet_v3_small", "resnet101"],
            "theta": 0.80,
            "parallel_first_n": 2,
        },
        # spec_b infeasible for C=50, E=0.100
    },

    (50, 0.120): {
        "sequential": {
            "models": ["resnet34", "convnext_tiny"],
            "theta": 0.55,
        },
        "po2": {
            "models": ["resnet34", "resnet18", "resnet101"],
            "theta": 0.60,
            "parallel_first_n": 2,
        },
        "spec_b": {
            "models": ["resnet34", "convnext_tiny"],
            "theta": 0.55,
        },
    },
(50, 0.140): {
    "sequential": {
        "models": ["resnet34", "convnext_tiny"],
        "theta": 0.40,
    },
    "po2": {
        "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
        "theta": 0.45,
        "parallel_first_n": 2,
    },
    "spec_b": {
        "models": ["resnet34", "convnext_tiny"],
        "theta": 0.40,
    },
},

    # =====================================================
    # C = 100
    # =====================================================

    (100, 0.090): {
        "sequential": {
            "models": ["resnet34", "vit_b_16"],
            "theta": 0.90,
        },
        "po2": {
            "models": ["resnet18", "resnet34", "vit_b_16"],
            "theta": 0.95,
            "parallel_first_n": 2,
        },
        "spec_b": {
            "models": ["resnet18", "resnet101"],
            "theta": 1.00,
        },
    },

    (100, 0.100): {
        "sequential": {
            "models": ["resnet34", "resnet101"],
            "theta": 0.75,
        },
        "po2": {
            "models": ["resnet34", "mobilenet_v3_small", "resnet101"],
            "theta": 0.80,
            "parallel_first_n": 2,
        },
        "spec_b": {
            "models": ["resnet34", "resnet101"],
            "theta": 0.75,
        },
    },

    (100, 0.120): {
        "sequential": {
            "models": ["resnet34", "convnext_tiny"],
            "theta": 0.55,
        },
        "po2": {
            "models": ["resnet34", "resnet18", "resnet101"],
            "theta": 0.60,
            "parallel_first_n": 2,
        },
        "spec_b": {
            "models": ["resnet34", "convnext_tiny"],
            "theta": 0.55,
        },
    },
(100, 0.140): {
    "sequential": {
        "models": ["resnet34", "convnext_tiny"],
        "theta": 0.40,
    },
    "po2": {
        "models": ["resnet34", "mobilenet_v3_small", "convnext_tiny"],
        "theta": 0.45,
        "parallel_first_n": 2,
    },
    "spec_b": {
        "models": ["resnet34", "convnext_tiny"],
        "theta": 0.40,
    },
},
}


def get_optimal_config(cost: int, error: float, policy: str):
    key = (int(cost), round(float(error), 3))
    policy = str(policy).lower()

    if key not in OPTIMAL_CONFIGS:
        available = sorted(OPTIMAL_CONFIGS.keys())
        raise ValueError(
            f"No optimal config found for C={cost}, E={error}. "
            f"Available C,E values: {available}"
        )

    if policy not in OPTIMAL_CONFIGS[key]:
        available_policies = list(OPTIMAL_CONFIGS[key].keys())
        raise ValueError(
            f"No policy={policy} for C={cost}, E={error}. "
            f"Available policies: {available_policies}"
        )

    return OPTIMAL_CONFIGS[key][policy]