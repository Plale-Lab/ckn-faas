from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from label_utils import (
    wnid_matches_text_label,
)

from config import (
    BASE_FOLDER,
    DEFAULT_DEADLINE_MS,
    FRUGALML_E,
    IAR_PROFILES_MS,
    IMG_EXTS,
    LOG_DIR,
    RESULT_DIR,
    get_model_cost,
)

from iluvatar_client import (
    choose_final_result,
    execute_frugalml_request,
)

from policy import (
    get_policy_models,
    load_policy,
)


# ============================================================
# Easy run configuration
# ============================================================

# Choose:
#   "short"
#   "medium"
#   "long"
RUN_PROFILE = "short"


# Duration in seconds for each IAR value.
#
# Example:
#   30 seconds per IAR is useful for testing.
#   120 seconds per IAR is better for final experiments.
RUN_DURATION_SEC = 120.0


# Reproducible random image selection and Poisson arrivals.
RANDOM_SEED = 42


# Optional custom output file.
#
# Use None for automatic naming:
#   results/frugalml_E0.120.csv
CUSTOM_OUTPUT_PATH = None


# ============================================================
# Logging
# ============================================================

LOGGER = logging.getLogger(
    "frugalml"
)

LOGGER.setLevel(
    logging.INFO
)

if not LOGGER.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    log_file = (
        LOG_DIR
        / (
            f"frugalml_"
            f"E{FRUGALML_E:.3f}.log"
        )
    )

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )

    file_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        file_handler
    )

    console_handler = (
        logging.StreamHandler()
    )

    console_handler.setFormatter(
        formatter
    )

    LOGGER.addHandler(
        console_handler
    )


# ============================================================
# Load FrugalML policy
# ============================================================

POLICY = load_policy(
    FRUGALML_E
)


def generate_id(
    index: int,
) -> str:
    """
    Generate a unique request ID.
    """

    return (
        f"FRUGAL-{index:07d}"
    )


def pick_random_image(
    base_folder: str,
) -> tuple[str, str]:
    """
    Select a random image.

    Returns:
        image_path
        selected_folder

    selected_folder is the WNID ground-truth folder.
    """

    base_path = Path(
        base_folder
    )

    if not base_path.exists():
        raise FileNotFoundError(
            "Dataset folder does not exist: "
            f"{base_path}"
        )

    class_folders = [
        folder
        for folder
        in base_path.iterdir()
        if folder.is_dir()
    ]

    if not class_folders:
        raise FileNotFoundError(
            "No class folders were found in: "
            f"{base_path}"
        )

    selected_folder = (
        random.choice(
            class_folders
        )
    )

    images = [
        image
        for image
        in selected_folder.iterdir()
        if (
            image.is_file()
            and image.suffix.lower()
            in IMG_EXTS
        )
    ]

    if not images:
        raise FileNotFoundError(
            "No supported images were found in: "
            f"{selected_folder}"
        )

    selected_image = (
        random.choice(
            images
        )
    )

    return (
        str(selected_image),
        selected_folder.name,
    )


def image_to_b64(
    image_path: str,
) -> str:
    """
    Convert an image into a base64 string for Iluvatar.
    """

    image_bytes = Path(
        image_path
    ).read_bytes()

    return (
        base64.b64encode(
            image_bytes
        ).decode(
            "utf-8"
        )
    )


def make_per_model_json(
    results: list[
        dict[str, Any]
    ],
) -> str:
    """
    Convert per-model execution results into one JSON string
    for CSV logging.
    """

    per_model = {}

    for result in results:
        model_name = result.get(
            "model"
        )

        if not model_name:
            continue

        per_model[
            model_name
        ] = {
            "label": result.get(
                "label"
            ),

            "probability": float(
                result.get(
                    "probability",
                    0.0,
                )
            ),

            "latency_s": float(
                result.get(
                    "latency",
                    -1.0,
                )
            ),

            "success": bool(
                result.get(
                    "success",
                    False,
                )
            ),

            "container_state": (
                result.get(
                    "container_state",
                    "UNKNOWN",
                )
            ),

            "server_address": (
                result.get(
                    "server_address"
                )
            ),
        }

    return json.dumps(
        per_model,
        sort_keys=True,
    )


async def run_one_request(
    *,
    request_id: str,
    iar_ms: int,
    global_start: float,
) -> dict[str, Any]:
    """
    Run one FrugalML request.

    FrugalML:
        1. Runs the base model.
        2. Checks predicted class and confidence.
        3. Uses the learned class threshold.
        4. Accepts the base result or invokes the add-on model.
    """

    image_path, selected_folder = (
        pick_random_image(
            BASE_FOLDER
        )
    )

    image_b64 = image_to_b64(
        image_path
    )

    request_start_sec = (
        time.perf_counter()
        - global_start
    )

    start_time = (
        time.perf_counter()
    )

    try:
        results, decision = (
            await execute_frugalml_request(
                request_id=request_id,
                image_b64=image_b64,
                policy=POLICY,
            )
        )

        final = choose_final_result(
            results,
            decision,
        )

        e2e_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        correct = False

        if final["success"]:
            correct = bool(
                wnid_matches_text_label(
                    selected_folder,
                    final["label"],
                )
            )

        executed_models = [
            result["model"]
            for result in results
            if result.get(
                "model"
            )
        ]

        actual_cost = sum(
            get_model_cost(
                model
            )
            for model
            in executed_models
        )

        deadline_met = (
            e2e_time_ms
            <= DEFAULT_DEADLINE_MS
        )

        LOGGER.info(
            "[REQ_END] "
            "id=%s "
            "E=%.3f "
            "IAR=%sms "
            "executed=%s "
            "deferred=%s "
            "latency_ms=%.2f "
            "correct=%s",
            request_id,
            FRUGALML_E,
            iar_ms,
            executed_models,
            decision.get(
                "deferred",
                False,
            ),
            e2e_time_ms,
            correct,
        )

        return {
            "ID": request_id,
            "System": "FrugalML",

            "E": FRUGALML_E,
            "TargetAccuracy": (
                1.0 - FRUGALML_E
            ),

            "IAR": iar_ms,
            "ArrivalRate": (
                1000.0
                / float(iar_ms)
            ),

            "RequestStartSec": (
                request_start_sec
            ),

            "ImagePath": (
                image_path
            ),

            "selected_folder": (
                selected_folder
            ),

            "label": (
                final["label"]
            ),

            "final_model": (
                final["model"]
            ),

            "final_confidence": (
                final["confidence"]
            ),

            # Actual correctness.
            "accuracy": (
                1.0
                if correct
                else 0.0
            ),

            "correct": correct,

            "success": (
                final["success"]
            ),

            "e2e_time_ms": (
                e2e_time_ms
            ),

            # Included for compatibility with your graph code.
            "RespTime": (
                e2e_time_ms
            ),

            "deadline_ms": (
                DEFAULT_DEADLINE_MS
            ),

            "deadline_met": (
                deadline_met
            ),

            "actual_cost": (
                actual_cost
            ),

            "number_of_model_calls": (
                len(
                    executed_models
                )
            ),

            "selected_models": (
                json.dumps(
                    get_policy_models(
                        POLICY
                    )
                )
            ),

            "executed_models": (
                json.dumps(
                    executed_models
                )
            ),

            "per_model": (
                make_per_model_json(
                    results
                )
            ),

            "combiner_policy": (
                final[
                    "combiner_policy"
                ]
            ),

            "frugalml_base_model": (
                decision.get(
                    "base_model"
                )
            ),

            "frugalml_base_label": (
                decision.get(
                    "base_label"
                )
            ),

            "frugalml_base_confidence": (
                decision.get(
                    "base_confidence",
                    0.0,
                )
            ),

            "frugalml_threshold": (
                decision.get(
                    "threshold",
                    0.0,
                )
            ),

            "frugalml_addon_model": (
                decision.get(
                    "addon_model"
                )
            ),

            "frugalml_deferred": (
                decision.get(
                    "deferred",
                    False,
                )
            ),

            "frugalml_decision": (
                decision.get(
                    "decision"
                )
            ),

            "frugalml_used_class_policy": (
                decision.get(
                    "used_class_policy",
                    False,
                )
            ),
        }

    except Exception as error:
        e2e_time_ms = (
            time.perf_counter()
            - start_time
        ) * 1000.0

        LOGGER.exception(
            "[REQ_FAIL] "
            "id=%s "
            "E=%.3f "
            "IAR=%sms "
            "error=%s",
            request_id,
            FRUGALML_E,
            iar_ms,
            error,
        )

        return {
            "ID": request_id,
            "System": "FrugalML",

            "E": FRUGALML_E,
            "TargetAccuracy": (
                1.0 - FRUGALML_E
            ),

            "IAR": iar_ms,
            "ArrivalRate": (
                1000.0
                / float(iar_ms)
            ),

            "RequestStartSec": (
                request_start_sec
            ),

            "ImagePath": (
                image_path
            ),

            "selected_folder": (
                selected_folder
            ),

            "accuracy": 0.0,
            "correct": False,
            "success": False,

            "e2e_time_ms": (
                e2e_time_ms
            ),

            "RespTime": (
                e2e_time_ms
            ),

            "deadline_ms": (
                DEFAULT_DEADLINE_MS
            ),

            "deadline_met": False,

            "error": str(
                error
            ),
        }


async def generate_workload(
    *,
    profile_name: str,
    duration_sec: float,
    output_path: Path | None = None,
) -> Path:
    """
    Generate a Poisson workload for every IAR value in the
    selected profile.
    """

    if (
        profile_name
        not in IAR_PROFILES_MS
    ):
        raise ValueError(
            f"Unknown profile="
            f"{profile_name!r}. "
            f"Available profiles="
            f"{sorted(IAR_PROFILES_MS)}"
        )

    if output_path is None:
        output_path = (
            RESULT_DIR
            / (
                f"frugalml_"
                f"E{FRUGALML_E:.3f}.csv"
            )
        )

    rows: list[
        dict[str, Any]
    ] = []

    request_index = 0

    global_start = (
        time.perf_counter()
    )

    for iar_ms in (
        IAR_PROFILES_MS[
            profile_name
        ]
    ):
        phase_start = (
            time.perf_counter()
        )

        phase_end = (
            phase_start
            + duration_sec
        )

        iar_sec = (
            float(iar_ms)
            / 1000.0
        )

        tasks = []

        LOGGER.info(
            "[PHASE_START] "
            "E=%.3f "
            "IAR=%sms "
            "arrival_rate=%.3f req/s "
            "duration=%.1fs",
            FRUGALML_E,
            iar_ms,
            1000.0 / float(iar_ms),
            duration_sec,
        )

        while (
            time.perf_counter()
            < phase_end
        ):
            request_id = (
                generate_id(
                    request_index
                )
            )

            request_index += 1

            task = asyncio.create_task(
                run_one_request(
                    request_id=(
                        request_id
                    ),
                    iar_ms=iar_ms,
                    global_start=(
                        global_start
                    ),
                )
            )

            tasks.append(
                task
            )

            sleep_sec = (
                np.random.exponential(
                    scale=iar_sec
                )
            )

            await asyncio.sleep(
                float(
                    sleep_sec
                )
            )

        phase_rows = (
            await asyncio.gather(
                *tasks
            )
        )

        rows.extend(
            phase_rows
        )

        LOGGER.info(
            "[PHASE_END] "
            "E=%.3f "
            "IAR=%sms "
            "requests=%d",
            FRUGALML_E,
            iar_ms,
            len(phase_rows),
        )

    if not rows:
        raise RuntimeError(
            "No FrugalML workload "
            "results were generated"
        )

    fieldnames = sorted(
        {
            key
            for row in rows
            for key in row.keys()
        }
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    LOGGER.info(
        "Saved %d requests to %s",
        len(rows),
        output_path,
    )

    return output_path


if __name__ == "__main__":
    # Set reproducible random seeds.
    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    if CUSTOM_OUTPUT_PATH is None:
        output_path = None
    else:
        output_path = Path(
            CUSTOM_OUTPUT_PATH
        )

    print(
        "\n"
        "========================================\n"
        "Starting FrugalML workload\n"
        "========================================"
    )

    print(
        f"Error constraint: E={FRUGALML_E:.3f}"
    )

    print(
        f"Target accuracy: {1.0 - FRUGALML_E:.3f}"
    )

    print(
        f"Profile: {RUN_PROFILE}"
    )

    print(
        f"Duration per IAR: "
        f"{RUN_DURATION_SEC} seconds"
    )

    print(
        f"Dataset: {BASE_FOLDER}"
    )

    print(
        f"Policy base model: "
        f"{POLICY['base_model']}"
    )

    print(
        f"Default add-on model: "
        f"{POLICY['default_addon_model']}"
    )

    print(
        f"Default threshold: "
        f"{POLICY['default_threshold']}"
    )

    print(
        "Possible policy models: "
        f"{get_policy_models(POLICY)}"
    )

    print(
        "========================================\n"
    )

    output = asyncio.run(
        generate_workload(
            profile_name=(
                RUN_PROFILE
            ),
            duration_sec=(
                RUN_DURATION_SEC
            ),
            output_path=(
                output_path
            ),
        )
    )

    print(
        "\n"
        "========================================\n"
        "FrugalML workload completed\n"
        "========================================"
    )

    print(
        f"Results saved to: {output}"
    )