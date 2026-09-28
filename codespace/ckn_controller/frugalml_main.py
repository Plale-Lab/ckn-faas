from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from ckn_controller.ckn_config import (
    MODEL_PROFILES,
    QPS_WINDOW_SEC,
    USE_TWO_ILUVATAR_INSTANCES,
)

# Reuse the working DYSCI/Iluvatar infrastructure.
from ckn_controller.ckn_main import (
    LOG_PATH,
    build_stub_map_for_selected_models,
    close_channels,
    get_qps,
    get_recent_latency_overhead,
    get_stub_for_model,
    get_wait_results_for_models,
    logger,
    record_request,
    send_request,
)


# ============================================================
# FrugalML configuration
# ============================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

FRUGALML_POLICY_PATH = os.environ.get(
    "FRUGALML_POLICY_PATH",
    os.path.join(CURRENT_DIR, "frugalml_policy.json"),
)


def load_frugalml_policy(policy_path: str) -> Dict[str, Any]:
    """
    Load and validate the FrugalML policy.
    """

    if not os.path.isfile(policy_path):
        raise FileNotFoundError(
            f"FrugalML policy file was not found: {policy_path}"
        )

    with open(policy_path, "r", encoding="utf-8") as file:
        policy = json.load(file)

    required_fields = {
        "base_model",
        "default_addon_model",
        "default_threshold",
        "class_policies",
    }

    missing_fields = required_fields.difference(policy.keys())

    if missing_fields:
        raise ValueError(
            "FrugalML policy is missing required fields: "
            f"{sorted(missing_fields)}"
        )

    base_model = str(policy["base_model"])
    default_addon = str(policy["default_addon_model"])

    if base_model not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown FrugalML base model: {base_model}"
        )

    if default_addon not in MODEL_PROFILES:
        raise ValueError(
            f"Unknown default add-on model: {default_addon}"
        )

    default_threshold = float(policy["default_threshold"])

    if not 0.0 <= default_threshold <= 1.0:
        raise ValueError(
            "default_threshold must be between 0 and 1."
        )

    class_policies = policy.get("class_policies", {})

    if not isinstance(class_policies, dict):
        raise ValueError(
            "class_policies must be a JSON object."
        )

    for class_name, class_rule in class_policies.items():
        if not isinstance(class_rule, dict):
            raise ValueError(
                f"Policy for class '{class_name}' must be an object."
            )

        threshold = float(
            class_rule.get(
                "threshold",
                default_threshold,
            )
        )

        addon_model = str(
            class_rule.get(
                "addon_model",
                default_addon,
            )
        )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Threshold for class '{class_name}' must "
                "be between 0 and 1."
            )

        if addon_model not in MODEL_PROFILES:
            raise ValueError(
                f"Unknown add-on model '{addon_model}' "
                f"for class '{class_name}'."
            )

    return policy


FRUGALML_POLICY = load_frugalml_policy(
    FRUGALML_POLICY_PATH
)


# ============================================================
# Policy lookup helpers
# ============================================================

def normalize_class_name(value: Any) -> str:
    """
    Normalize model-predicted text labels for policy lookup.

    Examples:
        "Leatherback Turtle" -> "leatherback_turtle"
        "leatherback_turtle" -> "leatherback_turtle"
    """

    if value is None:
        return ""

    normalized = str(value).strip().lower()
    normalized = normalized.replace("-", "_")
    normalized = normalized.replace(" ", "_")

    while "__" in normalized:
        normalized = normalized.replace("__", "_")

    return normalized


def get_class_policy(
    predicted_label: Any,
) -> Tuple[float, str, bool, str]:
    """
    Return:
        threshold
        add-on model
        whether a class-specific policy was found
        normalized class-policy key
    """

    default_threshold = float(
        FRUGALML_POLICY["default_threshold"]
    )

    default_addon_model = str(
        FRUGALML_POLICY["default_addon_model"]
    )

    normalized_prediction = normalize_class_name(
        predicted_label
    )

    class_policies = FRUGALML_POLICY.get(
        "class_policies",
        {},
    )

    # Direct normalized lookup.
    for raw_class_name, rule in class_policies.items():
        normalized_policy_name = normalize_class_name(
            raw_class_name
        )

        if normalized_policy_name != normalized_prediction:
            continue

        threshold = float(
            rule.get(
                "threshold",
                default_threshold,
            )
        )

        addon_model = str(
            rule.get(
                "addon_model",
                default_addon_model,
            )
        )

        return (
            threshold,
            addon_model,
            True,
            normalized_policy_name,
        )

    return (
        default_threshold,
        default_addon_model,
        False,
        normalized_prediction,
    )


def get_all_possible_models() -> list[str]:
    """
    Get all models that may be used by the policy.

    This allows us to create the required gRPC channels before
    running the base model.
    """

    models = {
        str(FRUGALML_POLICY["base_model"]),
        str(FRUGALML_POLICY["default_addon_model"]),
    }

    for rule in FRUGALML_POLICY.get(
        "class_policies",
        {},
    ).values():
        addon_model = str(
            rule.get(
                "addon_model",
                FRUGALML_POLICY["default_addon_model"],
            )
        )

        models.add(addon_model)

    return sorted(models)


def make_per_model_entry(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "label": result.get("label"),
        "probability": float(
            result.get("probability", 0.0)
        ),
        "latency_s": float(
            result.get("latency", -1.0)
        ),
        "success": bool(
            result.get("success", False)
        ),
        "state": result.get(
            "container_state",
            "UNKNOWN",
        ),
        "server_address": result.get(
            "server_address",
        ),
    }


def safe_wait_value(
    wait_results: Dict[str, float],
    model_name: str,
) -> float:
    value = wait_results.get(model_name, -1.0)

    try:
        value = float(value)
    except (TypeError, ValueError):
        return -1.0

    if value == float("inf"):
        return -1.0

    return value


# ============================================================
# FrugalML execution
# ============================================================

async def main_frugalml_invoke(
    transaction_id: str,
    deadline: int,
    image_b64: str,
    selected_folder: Optional[str] = None,
) -> dict:
    """
    FrugalML runtime:

    1. Run the base model.
    2. Select a threshold using the base model's predicted class.
    3. Accept the base model when confidence >= threshold.
    4. Otherwise run the selected add-on model.
    5. Return the add-on model's prediction when it succeeds.

    This function returns the same major result fields as
    main_ensemble_invoke(), allowing the existing workload logger
    to write the same DYSCI CSV format.
    """

    start_time = time.perf_counter()

    if not image_b64:
        raise ValueError("image_b64 is required")

    if selected_folder is None:
        selected_folder = "unknown"

    try:
        deadline_ms = int(deadline)
    except (TypeError, ValueError):
        deadline_ms = 2000

    # --------------------------------------------------------
    # Request/QPS information
    # --------------------------------------------------------

    now = time.perf_counter()
    record_request(now, QPS_WINDOW_SEC)
    qps_est = get_qps(QPS_WINDOW_SEC)

    base_model = str(
        FRUGALML_POLICY["base_model"]
    )

    default_addon_model = str(
        FRUGALML_POLICY["default_addon_model"]
    )

    default_threshold = float(
        FRUGALML_POLICY["default_threshold"]
    )

    all_policy_models = get_all_possible_models()

    logger.info(
        f"[FRUGALML_REQ_START] "
        f"tx={transaction_id} "
        f"deadline_ms={deadline_ms} "
        f"qps={qps_est:.4f} "
        f"selected_folder={selected_folder} "
        f"base_model={base_model} "
        f"default_addon={default_addon_model} "
        f"default_threshold={default_threshold:.4f} "
        f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES}"
    )

    # Collect waits only for models FrugalML may execute.
    try:
        all_wait_results = get_wait_results_for_models(
            all_policy_models
        )
    except Exception as exc:
        logger.exception(
            f"[FRUGALML_WAIT_ERROR] "
            f"tx={transaction_id} error={exc}"
        )
        all_wait_results = {}

    # Create channels using the same mechanism as DYSCI.
    channel_map, stub_map = (
        build_stub_map_for_selected_models(
            all_policy_models
        )
    )

    base_result: Optional[Dict[str, Any]] = None
    addon_result: Optional[Dict[str, Any]] = None

    results: list[Dict[str, Any]] = []
    per_model: Dict[str, Dict[str, Any]] = {}

    threshold = default_threshold
    addon_model = default_addon_model
    used_class_policy = False
    policy_class_key = ""

    deferred = False
    stop_reason = "frugalml_not_started"

    try:
        # ====================================================
        # Stage 1: Base model
        # ====================================================

        logger.info(
            f"[FRUGALML_BASE_START] "
            f"tx={transaction_id} model={base_model}"
        )

        try:
            base_stub = get_stub_for_model(
                base_model,
                stub_map,
            )

            base_result = await send_request(
                base_stub,
                base_model,
                image_b64,
            )

        except Exception as exc:
            logger.exception(
                f"[FRUGALML_BASE_FAIL] "
                f"tx={transaction_id} "
                f"model={base_model} "
                f"error={exc}"
            )

            base_result = {
                "model": base_model,
                "label": None,
                "probability": 0.0,
                "latency": -1.0,
                "success": False,
                "container_state": "ERROR",
                "server_address": None,
                "error": str(exc),
            }

        per_model[base_model] = make_per_model_entry(
            base_result
        )

        base_success = bool(
            base_result.get("success", False)
        )

        base_label = base_result.get("label")
        base_confidence = float(
            base_result.get("probability", 0.0)
        )

        if base_success:
            results.append(base_result)

        # Select threshold and add-on based on predicted class.
        (
            threshold,
            addon_model,
            used_class_policy,
            policy_class_key,
        ) = get_class_policy(base_label)

        logger.info(
            f"[FRUGALML_BASE_RESULT] "
            f"tx={transaction_id} "
            f"model={base_model} "
            f"success={base_success} "
            f"label={base_label} "
            f"confidence={base_confidence:.6f} "
            f"threshold={threshold:.6f} "
            f"class_policy={used_class_policy} "
            f"policy_key={policy_class_key} "
            f"addon_model={addon_model}"
        )

        # If the base failed, defer automatically.
        deferred = (
            (not base_success)
            or (base_confidence < threshold)
        )

        # ====================================================
        # Stage 2: Accept base or invoke add-on
        # ====================================================

        if not deferred:
            stop_reason = (
                f"frugalml_base_confidence_pass="
                f"{base_model}"
            )

            logger.info(
                f"[FRUGALML_ACCEPT_BASE] "
                f"tx={transaction_id} "
                f"model={base_model} "
                f"confidence={base_confidence:.6f} "
                f"threshold={threshold:.6f}"
            )

        else:
            logger.info(
                f"[FRUGALML_DEFER] "
                f"tx={transaction_id} "
                f"base_model={base_model} "
                f"base_success={base_success} "
                f"base_confidence={base_confidence:.6f} "
                f"threshold={threshold:.6f} "
                f"addon_model={addon_model}"
            )

            try:
                addon_stub = get_stub_for_model(
                    addon_model,
                    stub_map,
                )

                addon_result = await send_request(
                    addon_stub,
                    addon_model,
                    image_b64,
                )

            except Exception as exc:
                logger.exception(
                    f"[FRUGALML_ADDON_FAIL] "
                    f"tx={transaction_id} "
                    f"model={addon_model} "
                    f"error={exc}"
                )

                addon_result = {
                    "model": addon_model,
                    "label": None,
                    "probability": 0.0,
                    "latency": -1.0,
                    "success": False,
                    "container_state": "ERROR",
                    "server_address": None,
                    "error": str(exc),
                }

            per_model[addon_model] = make_per_model_entry(
                addon_result
            )

            addon_success = bool(
                addon_result.get("success", False)
            )

            if addon_success:
                results.append(addon_result)

                stop_reason = (
                    f"frugalml_deferred_to_addon="
                    f"{addon_model}"
                )

                logger.info(
                    f"[FRUGALML_ADDON_RESULT] "
                    f"tx={transaction_id} "
                    f"model={addon_model} "
                    f"label={addon_result.get('label')} "
                    f"confidence="
                    f"{float(addon_result.get('probability', 0.0)):.6f}"
                )

            elif base_success:
                stop_reason = (
                    "frugalml_addon_failed_fallback_base"
                )

                logger.warning(
                    f"[FRUGALML_ADDON_FALLBACK] "
                    f"tx={transaction_id} "
                    f"addon={addon_model} failed; "
                    f"returning base={base_model}"
                )

            else:
                stop_reason = (
                    "frugalml_all_models_failed"
                )

    finally:
        await close_channels(channel_map)

    # --------------------------------------------------------
    # Final result selection
    # --------------------------------------------------------

    base_success = bool(
        base_result
        and base_result.get("success", False)
    )

    addon_success = bool(
        addon_result
        and addon_result.get("success", False)
    )

    if deferred and addon_success:
        final_result = addon_result

    elif base_success:
        final_result = base_result

    else:
        final_result = None

    end_time = time.perf_counter()
    e2e_ms = (end_time - start_time) * 1000.0

    # The selected models represent the configured FrugalML path.
    selected_models = [
        base_model,
        addon_model,
    ]

    # Remove a duplicate when the base and add-on are accidentally
    # configured as the same model.
    selected_models = list(
        dict.fromkeys(selected_models)
    )

    executed_models = [
        result["model"]
        for result in results
        if result.get("success", False)
    ]

    selected_wait_times = {
        model_name: safe_wait_value(
            all_wait_results,
            model_name,
        )
        for model_name in selected_models
    }

    # Approximate selected latency from model profile and wait.
    base_profile_latency = float(
        MODEL_PROFILES.get(
            base_model,
            {},
        ).get(
            "latency",
            -1.0,
        )
    )

    addon_profile_latency = float(
        MODEL_PROFILES.get(
            addon_model,
            {},
        ).get(
            "latency",
            -1.0,
        )
    )

    base_wait = selected_wait_times.get(
        base_model,
        -1.0,
    )

    addon_wait = selected_wait_times.get(
        addon_model,
        -1.0,
    )

    selected_est_latency_s = -1.0

    if base_profile_latency >= 0.0:
        selected_est_latency_s = max(
            0.0,
            base_profile_latency,
        )

        if base_wait >= 0.0:
            selected_est_latency_s += base_wait

        if deferred and addon_profile_latency >= 0.0:
            selected_est_latency_s += addon_profile_latency

            if addon_wait >= 0.0:
                selected_est_latency_s += addon_wait

    # --------------------------------------------------------
    # DYSCI-compatible response
    # --------------------------------------------------------

    common_result = {
        "selected_models": selected_models,
        "per_model": per_model,
        "wait_times": selected_wait_times,
        "selected_folder": selected_folder,

        # Configured FrugalML path has base + add-on.
        "ensemble_size": len(selected_models),

        "threshold": threshold,
        "threshold_stage": [],
        "selected_theta": threshold,

        # Sequential execution: one stage at a time.
        "parallel_first_n": 1,
        "parallel_batch_size": 1,

        "stop_reason": stop_reason,

        # These values make the CSV clearly identify FrugalML.
        "main_policy": "frugalml",
        "alpha": -1.0,
        "alpha_mode": "static",
        "qps_est": qps_est,
        "baseline_mode": "frugalml",

        "configured_cascade_mode": "sequential",
        "effective_cascade_mode": "sequential",
        "cascade_mode": "sequential",

        "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,

        # FrugalML does not use DYSCI's deadline fast path.
        "deadline_fast_path_enabled": False,
        "deadline_fast_path_triggered": False,

        "selected_est_latency_s": selected_est_latency_s,
        "raw_selected_est_latency_s": (
            selected_est_latency_s
        ),
        "recent_latency_overhead_s": (
            get_recent_latency_overhead()
        ),

        "log_file": LOG_PATH,

        # DYSCI-only details are intentionally empty.
        "risk_details": {},
        "risk_selected_mode": "",
        "constraint_solver_details": {},
        "static_config_details": {},
        "static_cost": None,
        "static_error": None,
        "static_policy": None,

        # Additional fields. Existing loggers using result.get()
        # will safely ignore these when no matching CSV columns exist.
        "frugalml_deferred": deferred,
        "frugalml_used_class_policy": (
            used_class_policy
        ),
        "frugalml_policy_class": (
            policy_class_key
        ),
        "frugalml_base_model": base_model,
        "frugalml_addon_model": addon_model,
        "frugalml_base_label": (
            base_result.get("label")
            if base_result
            else None
        ),
        "frugalml_base_confidence": float(
            base_result.get("probability", 0.0)
            if base_result
            else 0.0
        ),
        "frugalml_class_threshold": threshold,
    }

    if final_result is None:
        logger.error(
            f"[FRUGALML_REQ_END] "
            f"tx={transaction_id} "
            f"success=False "
            f"stop_reason={stop_reason} "
            f"executed_models={executed_models} "
            f"e2e_ms={e2e_ms:.2f}"
        )

        return {
            **common_result,
            "executed_models": executed_models,
            "label": -1,
            "accuracy": 0.0,
            "combiner_policy": "FAILED",
            "e2e_time_ms": e2e_ms,
            "success": False,
        }

    final_label = final_result.get("label")
    final_confidence = float(
        final_result.get("probability", 0.0)
    )

    logger.info(
        f"[FRUGALML_REQ_END] "
        f"tx={transaction_id} "
        f"success=True "
        f"stop_reason={stop_reason} "
        f"selected_models={selected_models} "
        f"executed_models={executed_models} "
        f"final_model={final_result.get('model')} "
        f"final_label={final_label} "
        f"final_confidence={final_confidence:.6f} "
        f"deferred={deferred} "
        f"e2e_ms={e2e_ms:.2f}"
    )

    return {
        **common_result,
        "executed_models": executed_models,
        "label": final_label,

        # Your existing DYSCI logger uses "accuracy" for the
        # returned confidence/probability.
        "accuracy": final_confidence,

        "combiner_policy": "single_model",
        "e2e_time_ms": e2e_ms,
        "success": True,
    }