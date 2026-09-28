from __future__ import annotations

import base64
import os
import random
import time
from pathlib import Path
from typing import Any

import aiohttp

from config import BASE_FOLDER, REMOTE_CKN_URL
from logger import log_result


IMG_EXTS = (
    ".jpg",
    ".jpeg",
    ".png",
)


def pick_random_image(
    base_folder: str,
) -> tuple[str, str]:
    """
    Returns:
        image_path
        selected_folder

    selected_folder is the ground-truth ImageNet WNID.
    """

    if not os.path.isdir(base_folder):
        raise FileNotFoundError(
            f"Dataset directory not found: {base_folder}"
        )

    folders = [
        folder_name
        for folder_name in os.listdir(base_folder)
        if os.path.isdir(
            os.path.join(base_folder, folder_name)
        )
    ]

    if not folders:
        raise FileNotFoundError(
            f"No class folders found in {base_folder}"
        )

    random.shuffle(folders)

    for selected_folder in folders:
        folder_path = os.path.join(
            base_folder,
            selected_folder,
        )

        images = [
            filename
            for filename in os.listdir(folder_path)
            if filename.lower().endswith(IMG_EXTS)
        ]

        if not images:
            continue

        selected_image = random.choice(images)

        image_path = os.path.join(
            folder_path,
            selected_image,
        )

        return image_path, selected_folder

    raise FileNotFoundError(
        f"No supported images found under {base_folder}"
    )


def img_to_b64(image_path: str) -> str:
    image_bytes = Path(image_path).read_bytes()

    return base64.b64encode(
        image_bytes
    ).decode("utf-8")


async def call_remote_ckn(
    req_id: str,
    deadline: int,
) -> dict[str, Any]:
    """
    Send one request to the local FrugalML controller.
    """

    image_path, selected_folder = pick_random_image(
        BASE_FOLDER
    )

    image_b64 = img_to_b64(image_path)

    payload = {
        "transaction_id": req_id,
        "deadline": deadline,
        "image_b64": image_b64,
        "selected_folder": selected_folder,
        "image_path": image_path,
    }

    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(
        timeout=timeout
    ) as session:
        async with session.post(
            REMOTE_CKN_URL,
            json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()


async def send_request(
    req_id: str,
    deadline: int,
    iar_ms: float,
    mode: str,
    current_time_sec: float,
) -> None:
    """
    This uses the same logger call as the DYSCI workload.
    """

    start_wall = time.time()
    start_perf = time.perf_counter()

    try:
        result = await call_remote_ckn(
            req_id=req_id,
            deadline=deadline,
        )

        duration_sec = (
            time.perf_counter() - start_perf
        )

        log_result(
            mode,
            req_id,
            deadline,
            iar_ms,
            duration_sec,
            current_time_sec,
            result,
        )

        print(
            f"[REQUEST COMPLETE] "
            f"id={req_id} "
            f"executed={result.get('executed_models', [])} "
            f"deferred={result.get('frugalml_deferred', False)} "
            f"label={result.get('label', '')} "
            f"confidence={result.get('accuracy', 0.0):.4f} "
            f"e2e_ms={result.get('e2e_time_ms', -1):.2f}"
        )

    except Exception as exc:
        duration_sec = time.time() - start_wall

        # DYSCI-compatible error dictionary
        error_result = {
            "selected_models": [],
            "executed_models": [],
            "label": -1,
            "accuracy": 0.0,
            "combiner_policy": "FAILED",
            "e2e_time_ms": duration_sec * 1000.0,
            "success": False,
            "selected_folder": "",
            "ensemble_size": 0,
            "threshold": -1,
            "threshold_stage": [],
            "selected_theta": None,
            "parallel_first_n": -1,
            "parallel_batch_size": -1,
            "stop_reason": "frugalml_request_error",
            "qps_est": -1,
            "main_policy": "frugalml",
            "alpha": -1,
            "alpha_mode": "static",
            "baseline_mode": "frugalml",
            "configured_cascade_mode": "sequential",
            "effective_cascade_mode": "sequential",
            "cascade_mode": "sequential",
            "use_two_instances": True,
            "per_model": {},
            "wait_times": {},
            "deadline_fast_path_enabled": False,
            "deadline_fast_path_triggered": False,
            "selected_est_latency_s": -1,
            "raw_selected_est_latency_s": -1,
            "recent_latency_overhead_s": -1,
            "risk_details": {},
            "risk_selected_mode": "",
            "constraint_solver_details": {},
            "static_config_details": {},
            "static_cost": None,
            "static_error": None,
            "static_policy": None,
            "frugalml_deferred": False,
            "frugalml_base_model": "",
            "frugalml_addon_model": "",
            "frugalml_base_label": "",
            "frugalml_base_confidence": -1,
            "frugalml_class_threshold": -1,
            "error": str(exc),
        }

        log_result(
            mode,
            req_id,
            deadline,
            iar_ms,
            duration_sec,
            current_time_sec,
            error_result,
        )

        print(
            f"[REQUEST ERROR] id={req_id} error={exc}"
        )