from __future__ import annotations

from pathlib import Path

from ckn_controller.ckn_config import (
    M_TOTAL,
    MODEL_PROFILES,
)


# ============================================================
# Easy experiment configuration
# ============================================================

# Change this value before each experiment.
#
# Supported values:
#   0.090
#   0.100
#   0.120
#   0.140
FRUGALML_E = 0.100


# Dataset folder containing ImageNet WNID folders.
BASE_FOLDER = (
    "/Users/agamage/Desktop/D2I/"
    "Codes Original/Mode-S/archive/train.X1"
)


# Fixed request deadline used for the workload.
DEFAULT_DEADLINE_MS = 2000


# Image formats that can be selected.
IMG_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
)


# ============================================================
# Output folders
# ============================================================

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CONFIG_DIR = Path("configs")
CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOG_DIR = Path("logs")
LOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# FrugalML policy files
# ============================================================

POLICY_PATHS = {
    0.090: (
        CONFIG_DIR
        / "frugalml_E0.090.json"
    ),
    0.100: (
        CONFIG_DIR
        / "frugalml_E0.100.json"
    ),
    0.120: (
        CONFIG_DIR
        / "frugalml_E0.120.json"
    ),
    0.140: (
        CONFIG_DIR
        / "frugalml_E0.140.json"
    ),
}


# ============================================================
# Workload profiles
# ============================================================

# IAR is in milliseconds.
#
# Arrival rate:
#   arrival_rate = 1000 / IAR
#
# Examples:
#   IAR=2000 ms -> 0.5 requests/second
#   IAR=1000 ms -> 1 request/second
#   IAR=500 ms  -> 2 requests/second
#   IAR=200 ms  -> 5 requests/second
#   IAR=100 ms  -> 10 requests/second

IAR_PROFILES_MS = {
    "short": [
        2000,
        1000,
        667,
        500,
        200,
        100,
    ],

    "medium": [
        2000,
        1000,
        750,
        500,
        250,
        200,
        100,
    ],

    "long": [
        2000,
        1500,
        1000,
        750,
        500,
        250,
        200,
        100,
        50,
    ],
}


# Use the same candidate models defined by DYSCI.
CANDIDATE_MODELS = list(
    M_TOTAL
)


def get_model_cost(
    model_name: str,
) -> float:
    """
    Return the cost of invoking one model.

    The function first checks whether MODEL_PROFILES contains
    a cost field. If no cost field exists, it tries model-size
    fields. If neither is available, it returns 1.0.

    For the final paper evaluation, make sure this matches
    the same cost definition used by DYSCI.
    """

    profile = MODEL_PROFILES.get(
        model_name,
        {},
    )

    # Try explicit model cost fields.
    for key in (
        "cost",
        "size_cost",
        "model_cost",
    ):
        if key in profile:
            try:
                return float(
                    profile[key]
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    # Fall back to model-size fields.
    for key in (
        "size",
        "model_size",
        "bytes",
    ):
        if key in profile:
            try:
                return float(
                    profile[key]
                )
            except (
                TypeError,
                ValueError,
            ):
                pass

    # Final fallback.
    return 1.0