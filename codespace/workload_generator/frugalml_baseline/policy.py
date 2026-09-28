from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config import (
    FRUGALML_E,
    POLICY_PATHS,
)


def normalize_error(
    error_constraint: float,
) -> float:
    return round(
        float(error_constraint),
        3,
    )


def normalize_label(
    label: Any,
) -> str:
    """
    Convert labels into consistent dictionary keys.
    """

    if label is None:
        return ""

    return " ".join(
        str(label)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def get_policy_path(
    error_constraint: float,
) -> Path:
    normalized_e = normalize_error(
        error_constraint
    )

    if normalized_e not in POLICY_PATHS:
        supported = ", ".join(
            f"{value:.3f}"
            for value in sorted(
                POLICY_PATHS
            )
        )

        raise ValueError(
            f"Unsupported FrugalML "
            f"error constraint E={normalized_e:.3f}. "
            f"Supported values: {supported}"
        )

    return POLICY_PATHS[
        normalized_e
    ]


def load_policy(
    error_constraint: float = FRUGALML_E,
) -> dict[str, Any]:
    """
    Load one E-specific FrugalML policy.
    """

    normalized_e = normalize_error(
        error_constraint
    )

    path = get_policy_path(
        normalized_e
    )

    if not path.exists():
        raise FileNotFoundError(
            f"FrugalML policy file "
            f"not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        policy = json.load(file)

    required_fields = {
        "error_constraint",
        "base_model",
        "default_addon_model",
        "default_threshold",
        "class_policies",
    }

    missing_fields = (
        required_fields
        - set(policy)
    )

    if missing_fields:
        raise ValueError(
            f"Policy {path} is missing "
            f"required fields: "
            f"{sorted(missing_fields)}"
        )

    file_e = normalize_error(
        policy["error_constraint"]
    )

    if file_e != normalized_e:
        raise ValueError(
            f"Policy error mismatch: "
            f"requested E={normalized_e:.3f}, "
            f"file contains E={file_e:.3f}"
        )

    normalized_class_policies: dict[
        str,
        dict[str, Any],
    ] = {}

    for label, class_policy in (
        policy.get(
            "class_policies",
            {},
        ).items()
    ):
        normalized_class_policies[
            normalize_label(label)
        ] = {
            "threshold": float(
                class_policy[
                    "threshold"
                ]
            ),
            "addon_model": str(
                class_policy[
                    "addon_model"
                ]
            ),
        }

    policy["error_constraint"] = (
        normalized_e
    )

    policy["target_accuracy"] = float(
        policy.get(
            "target_accuracy",
            1.0 - normalized_e,
        )
    )

    policy["base_model"] = str(
        policy["base_model"]
    )

    policy["default_addon_model"] = str(
        policy[
            "default_addon_model"
        ]
    )

    policy["default_threshold"] = float(
        policy[
            "default_threshold"
        ]
    )

    policy["class_policies"] = (
        normalized_class_policies
    )

    return policy


def get_class_decision(
    predicted_label: Any,
    policy: dict[str, Any],
) -> tuple[
    float,
    str,
    bool,
]:
    """
    Return:
        threshold
        add-on model
        whether a class-specific policy was found
    """

    normalized_prediction = (
        normalize_label(
            predicted_label
        )
    )

    class_policy = (
        policy[
            "class_policies"
        ].get(
            normalized_prediction
        )
    )

    if class_policy is None:
        return (
            float(
                policy[
                    "default_threshold"
                ]
            ),
            str(
                policy[
                    "default_addon_model"
                ]
            ),
            False,
        )

    return (
        float(
            class_policy[
                "threshold"
            ]
        ),
        str(
            class_policy[
                "addon_model"
            ]
        ),
        True,
    )


def get_policy_models(
    policy: dict[str, Any],
) -> list[str]:
    """
    Return models that may be used by this policy.

    This does not mean that all models execute for every request.
    Normally only the base model runs. The add-on model runs only
    when the request is deferred.
    """

    models = {
        str(
            policy["base_model"]
        ),
        str(
            policy[
                "default_addon_model"
            ]
        ),
    }

    for class_policy in (
        policy[
            "class_policies"
        ].values()
    ):
        models.add(
            str(
                class_policy[
                    "addon_model"
                ]
            )
        )

    return sorted(models)