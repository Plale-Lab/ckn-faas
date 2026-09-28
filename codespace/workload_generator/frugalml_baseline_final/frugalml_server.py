# from __future__ import annotations
#
# import json
# import time
# from pathlib import Path
# from typing import Any
#
# import aiohttp
# from aiohttp import web
#
# from config import (
#     FRUGALML_HOST,
#     FRUGALML_PORT,
#     FRUGALML_POLICY_FILE,
#     MODEL_INVOKE_PATH,
#     get_server_address_for_model,
# )
#
#
# def load_policy(
#     policy_path: Path,
# ) -> dict[str, Any]:
#     if not policy_path.exists():
#         raise FileNotFoundError(
#             f"Policy file not found: {policy_path}"
#         )
#
#     with policy_path.open(
#         "r",
#         encoding="utf-8",
#     ) as file:
#         policy = json.load(file)
#
#     required_fields = {
#         "base_model",
#         "default_addon_model",
#         "default_threshold",
#         "class_policies",
#     }
#
#     missing = required_fields.difference(policy)
#
#     if missing:
#         raise ValueError(
#             f"Policy missing fields: {sorted(missing)}"
#         )
#
#     return policy
#
#
# POLICY = load_policy(FRUGALML_POLICY_FILE)
#
#
# def normalize_label(label: Any) -> str:
#     if label is None:
#         return ""
#
#     if isinstance(label, list):
#         return ", ".join(str(value) for value in label)
#
#     return str(label).strip()
#
#
# def extract_response_body(
#     response_data: dict[str, Any],
# ) -> dict[str, Any]:
#     body: Any = response_data.get(
#         "body",
#         response_data,
#     )
#
#     if isinstance(body, str):
#         try:
#             body = json.loads(body)
#         except json.JSONDecodeError:
#             body = response_data
#
#     if not isinstance(body, dict):
#         return response_data
#
#     return body
#
#
# def parse_model_response(
#     model_name: str,
#     response_data: dict[str, Any],
#     measured_latency_sec: float,
#     server_address: str,
# ) -> dict[str, Any]:
#     body = extract_response_body(response_data)
#
#     label = (
#         body.get("label")
#         or body.get("prediction")
#         or body.get("predicted_label")
#         or body.get("class_name")
#         or body.get("class")
#         or ""
#     )
#
#     probability = body.get("probability")
#
#     if probability is None:
#         probability = body.get("confidence")
#
#     if probability is None:
#         probability = body.get("score")
#
#     if probability is None:
#         probability = body.get("prob")
#
#     try:
#         probability = float(probability)
#     except (TypeError, ValueError):
#         probability = 0.0
#
#     returned_latency = body.get("latency")
#
#     try:
#         returned_latency = float(returned_latency)
#     except (TypeError, ValueError):
#         returned_latency = measured_latency_sec
#
#     success = bool(
#         body.get(
#             "success",
#             response_data.get("success", True),
#         )
#     )
#
#     return {
#         "model": model_name,
#         "label": normalize_label(label),
#         "probability": probability,
#         "latency": returned_latency,
#         "success": success,
#         "container_state": body.get(
#             "container_state",
#             response_data.get(
#                 "container_state",
#                 "UNKNOWN",
#             ),
#         ),
#         "server_address": server_address,
#         "correct": body.get("correct"),
#         "accuracy": body.get("accuracy"),
#         "error": body.get(
#             "error",
#             response_data.get("error", ""),
#         ),
#     }
#
#
# async def invoke_model(
#     transaction_id: str,
#     model_name: str,
#     image_b64: str,
#     deadline_ms: int,
#     selected_folder: str,
# ) -> dict[str, Any]:
#     """
#     Calls one model.
#
#     This uses an HTTP /invoke wrapper. When your current DYSCI server
#     invokes Iluvatar through gRPC, replace only this function with the
#     same send_request(stub, model_name, image_b64) function used by
#     your DYSCI code.
#     """
#
#     server_address = get_server_address_for_model(
#         model_name
#     )
#
#     endpoint = (
#         f"{server_address.rstrip('/')}"
#         f"{MODEL_INVOKE_PATH}"
#     )
#
#     payload = {
#         "transaction_id": transaction_id,
#         "model_name": model_name,
#         "image_b64": image_b64,
#         "deadline": deadline_ms,
#         "deadline_ms": deadline_ms,
#         "selected_folder": selected_folder,
#     }
#
#     timeout = aiohttp.ClientTimeout(total=600)
#     invoke_start = time.perf_counter()
#
#     try:
#         async with aiohttp.ClientSession(
#             timeout=timeout
#         ) as session:
#             async with session.post(
#                 endpoint,
#                 json=payload,
#             ) as response:
#                 response.raise_for_status()
#                 response_data = await response.json()
#
#         measured_latency_sec = (
#             time.perf_counter() - invoke_start
#         )
#
#         return parse_model_response(
#             model_name=model_name,
#             response_data=response_data,
#             measured_latency_sec=measured_latency_sec,
#             server_address=server_address,
#         )
#
#     except Exception as exc:
#         measured_latency_sec = (
#             time.perf_counter() - invoke_start
#         )
#
#         return {
#             "model": model_name,
#             "label": "",
#             "probability": 0.0,
#             "latency": measured_latency_sec,
#             "success": False,
#             "container_state": "ERROR",
#             "server_address": server_address,
#             "correct": None,
#             "accuracy": None,
#             "error": str(exc),
#         }
#
#
# def normalize_policy_key(label: str) -> str:
#     return str(label).strip().lower()
#
#
# def get_class_policy(
#     predicted_label: str,
# ) -> tuple[float, str, bool]:
#     class_policies = POLICY.get(
#         "class_policies",
#         {},
#     )
#
#     normalized_prediction = normalize_policy_key(
#         predicted_label
#     )
#
#     for policy_label, class_rule in (
#         class_policies.items()
#     ):
#         if (
#             normalize_policy_key(policy_label)
#             == normalized_prediction
#         ):
#             return (
#                 float(class_rule["threshold"]),
#                 str(
#                     class_rule.get(
#                         "addon_model",
#                         POLICY["default_addon_model"],
#                     )
#                 ),
#                 True,
#             )
#
#     return (
#         float(POLICY["default_threshold"]),
#         str(POLICY["default_addon_model"]),
#         False,
#     )
#
#
# def get_correctness(
#     final_result: dict[str, Any],
#     selected_folder: str,
# ) -> tuple[bool, float]:
#     """
#     Uses correctness returned by your model wrapper when available.
#     """
#
#     returned_correct = final_result.get("correct")
#
#     if returned_correct is not None:
#         correct = bool(returned_correct)
#         return correct, float(correct)
#
#     returned_accuracy = final_result.get("accuracy")
#
#     if returned_accuracy is not None:
#         try:
#             accuracy = float(returned_accuracy)
#             return accuracy > 0.0, accuracy
#         except (TypeError, ValueError):
#             pass
#
#     # Works only when the model returns a WNID directly.
#     correct = (
#         normalize_label(final_result.get("label")).lower()
#         == selected_folder.lower()
#     )
#
#     return correct, float(correct)
#
#
# def empty_dysci_fields() -> dict[str, Any]:
#     """
#     Default values for DYSCI-specific columns not used by FrugalML.
#     """
#
#     return {
#         "wait_times": {},
#         "threshold_stage": [],
#         "parallel_first_n": 1,
#         "parallel_batch_size": 1,
#         "qps_est": -1,
#         "alpha": -1,
#         "alpha_mode": "static",
#         "configured_cascade_mode": "sequential",
#         "effective_cascade_mode": "sequential",
#         "cascade_mode": "sequential",
#         "use_two_instances": True,
#         "deadline_fast_path_enabled": False,
#         "deadline_fast_path_triggered": False,
#         "selected_est_latency_s": -1,
#         "raw_selected_est_latency_s": -1,
#         "recent_latency_overhead_s": -1,
#         "risk_details": {},
#         "risk_selected_mode": "",
#         "constraint_solver_details": {},
#         "static_config_details": {},
#         "static_cost": None,
#         "static_error": None,
#         "static_policy": None,
#     }
#
#
# async def handle_invoke(
#     request: web.Request,
# ) -> web.Response:
#     request_start = time.perf_counter()
#
#     try:
#         payload = await request.json()
#
#         transaction_id = str(
#             payload.get(
#                 "transaction_id",
#                 f"REQ-{time.time_ns()}",
#             )
#         )
#
#         deadline_ms = int(
#             payload.get(
#                 "deadline",
#                 payload.get("deadline_ms", 2000),
#             )
#         )
#
#         image_b64 = str(
#             payload.get("image_b64", "")
#         )
#
#         selected_folder = str(
#             payload.get("selected_folder", "")
#         )
#
#         if not image_b64:
#             raise ValueError(
#                 "Missing image_b64 in request."
#             )
#
#         base_model = str(POLICY["base_model"])
#
#         base_result = await invoke_model(
#             transaction_id=transaction_id,
#             model_name=base_model,
#             image_b64=image_b64,
#             deadline_ms=deadline_ms,
#             selected_folder=selected_folder,
#         )
#
#         threshold, addon_model, used_class_policy = (
#             get_class_policy(
#                 base_result.get("label", "")
#             )
#         )
#
#         base_confidence = float(
#             base_result.get("probability", 0.0)
#         )
#
#         deferred = (
#             not base_result.get("success", False)
#             or base_confidence < threshold
#         )
#
#         results = [base_result]
#         addon_result: dict[str, Any] | None = None
#
#         if deferred:
#             addon_result = await invoke_model(
#                 transaction_id=transaction_id,
#                 model_name=addon_model,
#                 image_b64=image_b64,
#                 deadline_ms=deadline_ms,
#                 selected_folder=selected_folder,
#             )
#
#             results.append(addon_result)
#
#         if (
#             deferred
#             and addon_result is not None
#             and addon_result.get("success", False)
#         ):
#             final_result = addon_result
#             stop_reason = (
#                 f"frugalml_deferred_to_addon={addon_model}"
#             )
#
#         elif base_result.get("success", False):
#             final_result = base_result
#
#             if deferred:
#                 stop_reason = (
#                     "frugalml_addon_failed_fallback_base"
#                 )
#             else:
#                 stop_reason = (
#                     f"frugalml_base_confidence_pass="
#                     f"{base_model}"
#                 )
#
#         else:
#             final_result = {
#                 "model": "",
#                 "label": -1,
#                 "probability": 0.0,
#                 "latency": -1,
#                 "success": False,
#                 "container_state": "ERROR",
#                 "server_address": "",
#                 "error": "Both FrugalML models failed.",
#             }
#
#             stop_reason = "frugalml_all_models_failed"
#
#         successful_results = [
#             result
#             for result in results
#             if result.get("success", False)
#         ]
#
#         executed_models = [
#             result["model"]
#             for result in successful_results
#         ]
#
#         selected_models = [
#             base_model,
#             addon_model,
#         ]
#
#         per_model = {}
#
#         for result in results:
#             model_name = result.get("model")
#
#             if not model_name:
#                 continue
#
#             per_model[model_name] = {
#                 "label": result.get("label", ""),
#                 "probability": float(
#                     result.get("probability", 0.0)
#                 ),
#                 "latency_s": float(
#                     result.get("latency", -1)
#                 ),
#                 "success": bool(
#                     result.get("success", False)
#                 ),
#                 "state": result.get(
#                     "container_state",
#                     "UNKNOWN",
#                 ),
#                 "server_address": result.get(
#                     "server_address",
#                     "",
#                 ),
#             }
#
#         correct, correctness_value = get_correctness(
#             final_result=final_result,
#             selected_folder=selected_folder,
#         )
#
#         e2e_ms = (
#             time.perf_counter() - request_start
#         ) * 1000.0
#
#         success = bool(
#             final_result.get("success", False)
#         )
#
#         result = {
#             # Fields used at the beginning of the DYSCI CSV
#             "selected_models": selected_models,
#             "executed_models": executed_models,
#             "label": final_result.get("label", -1),
#
#             # DYSCI uses this field as output confidence in many rows.
#             "accuracy": float(
#                 final_result.get("probability", 0.0)
#             ),
#
#             "combiner_policy": "single_model",
#             "e2e_time_ms": e2e_ms,
#             "success": success,
#             "selected_folder": selected_folder,
#             "ensemble_size": len(selected_models),
#             "threshold": threshold,
#             "selected_theta": threshold,
#             "stop_reason": stop_reason,
#
#             # Policy fields
#             "main_policy": "frugalml",
#             "baseline_mode": "frugalml",
#             "per_model": per_model,
#
#             # Additional FrugalML fields
#             "frugalml_deferred": deferred,
#             "frugalml_used_class_policy": (
#                 used_class_policy
#             ),
#             "frugalml_base_model": base_model,
#             "frugalml_addon_model": addon_model,
#             "frugalml_base_label": base_result.get(
#                 "label",
#                 "",
#             ),
#             "frugalml_base_confidence": (
#                 base_confidence
#             ),
#             "frugalml_class_threshold": threshold,
#             "frugalml_final_correct": correct,
#             "frugalml_correctness_value": (
#                 correctness_value
#             ),
#             "error": (
#                 ""
#                 if success
#                 else final_result.get("error", "")
#             ),
#
#             **empty_dysci_fields(),
#         }
#
#         print(
#             f"[FRUGALML] "
#             f"id={transaction_id} "
#             f"base={base_model} "
#             f"label={base_result.get('label', '')} "
#             f"confidence={base_confidence:.4f} "
#             f"threshold={threshold:.4f} "
#             f"class_policy={used_class_policy} "
#             f"deferred={deferred} "
#             f"executed={executed_models} "
#             f"final={result['label']} "
#             f"e2e_ms={e2e_ms:.2f}"
#         )
#
#         return web.json_response(result)
#
#     except Exception as exc:
#         e2e_ms = (
#             time.perf_counter() - request_start
#         ) * 1000.0
#
#         error_result = {
#             "selected_models": [],
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": "",
#             "ensemble_size": 0,
#             "threshold": -1,
#             "selected_theta": None,
#             "stop_reason": "frugalml_server_error",
#             "main_policy": "frugalml",
#             "baseline_mode": "frugalml",
#             "per_model": {},
#             "frugalml_deferred": False,
#             "frugalml_used_class_policy": False,
#             "frugalml_base_model": "",
#             "frugalml_addon_model": "",
#             "frugalml_base_label": "",
#             "frugalml_base_confidence": -1,
#             "frugalml_class_threshold": -1,
#             "frugalml_final_correct": False,
#             "error": str(exc),
#             **empty_dysci_fields(),
#         }
#
#         return web.json_response(
#             error_result,
#             status=500,
#         )
#
#
# async def handle_health(
#     request: web.Request,
# ) -> web.Response:
#     return web.json_response(
#         {
#             "status": "healthy",
#             "policy": "frugalml",
#             "base_model": POLICY["base_model"],
#             "default_addon_model": (
#                 POLICY["default_addon_model"]
#             ),
#             "default_threshold": (
#                 POLICY["default_threshold"]
#             ),
#             "class_policy_count": len(
#                 POLICY.get("class_policies", {})
#             ),
#         }
#     )
#
#
# def create_app() -> web.Application:
#     app = web.Application()
#
#     app.router.add_post(
#         "/invoke",
#         handle_invoke,
#     )
#
#     app.router.add_get(
#         "/health",
#         handle_health,
#     )
#
#     return app
#
#
# if __name__ == "__main__":
#     print("========================================")
#     print("Starting FrugalML controller")
#     print("========================================")
#     print(
#         f"Policy file: "
#         f"{FRUGALML_POLICY_FILE.resolve()}"
#     )
#     print(f"Base model: {POLICY['base_model']}")
#     print(
#         f"Default add-on: "
#         f"{POLICY['default_addon_model']}"
#     )
#     print(
#         f"Default threshold: "
#         f"{POLICY['default_threshold']}"
#     )
#     print(
#         f"Class policies: "
#         f"{len(POLICY.get('class_policies', {}))}"
#     )
#     print(
#         f"Listening on: "
#         f"http://{FRUGALML_HOST}:{FRUGALML_PORT}"
#     )
#     print("========================================")
#
#     web.run_app(
#         create_app(),
#         host=FRUGALML_HOST,
#         port=FRUGALML_PORT,
#     )


from __future__ import annotations

from typing import Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from codespace.ckn_controller.frugalml_main import (
    FRUGALML_POLICY,
    FRUGALML_POLICY_PATH,
    main_frugalml_invoke,
)


app = FastAPI(
    title="FrugalML Baseline Server",
    description=(
        "FrugalML confidence-based base/add-on routing "
        "using the existing Iluvatar gRPC infrastructure."
    ),
    version="1.0.0",
)


class InvokeReq(BaseModel):
    transaction_id: str
    deadline: int
    image_b64: str
    selected_folder: Optional[str] = None


@app.get("/health")
async def health() -> dict:
    return {
        "status": "healthy",
        "server": "frugalml",
        "policy_path": FRUGALML_POLICY_PATH,
        "base_model": FRUGALML_POLICY["base_model"],
        "default_addon_model": (
            FRUGALML_POLICY["default_addon_model"]
        ),
        "default_threshold": (
            FRUGALML_POLICY["default_threshold"]
        ),
        "class_policy_count": len(
            FRUGALML_POLICY.get(
                "class_policies",
                {},
            )
        ),
    }


@app.post("/invoke")
async def invoke(req: InvokeReq) -> dict:
    try:
        return await main_frugalml_invoke(
            transaction_id=req.transaction_id,
            deadline=req.deadline,
            image_b64=req.image_b64,
            selected_folder=req.selected_folder,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "FrugalML inference failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc


if __name__ == "__main__":
    print("==========================================")
    print("Starting FrugalML baseline server")
    print("==========================================")
    print(f"Policy file: {FRUGALML_POLICY_PATH}")
    print(
        "Base model: "
        f"{FRUGALML_POLICY['base_model']}"
    )
    print(
        "Default add-on model: "
        f"{FRUGALML_POLICY['default_addon_model']}"
    )
    print(
        "Default threshold: "
        f"{FRUGALML_POLICY['default_threshold']}"
    )
    print(
        "Class-specific policies: "
        f"{len(FRUGALML_POLICY.get('class_policies', {}))}"
    )
    print("URL: http://0.0.0.0:9000")
    print("==========================================")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
    )