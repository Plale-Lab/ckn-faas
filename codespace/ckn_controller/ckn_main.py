# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# # Use ONE set of protobuf imports (avoid duplicates)
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     MAX_MODEL_SIZE,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     MODEL_SIZES,
#
#     # --- Cascade parameters ---
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#
#     # --- Alpha control ---
#     ALPHA_MODE,        # "static" or "adaptive"
#     ALPHA_STATIC,      # used if static mode
#     ALPHA_MIN,         # used if adaptive mode
#     ALPHA_MAX,
#
#     # --- Load normalization ---
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#
#     # --- Cascade execution mode ---
#     CASCADE_MODE,      # "sequential" or "parallel_first"
#     PARALLEL_FIRST_N,  # e.g. 2
#     PARALLEL_BATCH_SIZE,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
#
#
# # -----------------------------
# # Logging setup (file + console)
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("mode_s")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking (sliding window)
# # -----------------------------
# REQUEST_TS = deque()
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Load-adaptive alpha (QPS only)
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     load_score = qps_norm
#     alpha_now = alpha_max - load_score * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     """
#     Check whether there is enough remaining deadline budget to invoke next_model.
#
#     Returns:
#         stop_now, elapsed_ms, remaining_ms
#     """
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Ilúvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     """
#     Stage-specific threshold:
#       - if THRESHOLD_STAGE list exists and has entry for stage_index, use it
#       - otherwise fallback to THRESHOLD
#     """
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#     return float(THRESHOLD)
#
#
# # -----------------------------
# # Model selection: up to ENSEMBLE_SIZE
# # -----------------------------
# def build_model_set(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Greedy selection up to size = ensemble_size (if feasible).
#
#     cost = latency_term + accuracy_term
#       latency_term = clamp(Te/D, 0.1, 0.9), only if Te/D < 1
#       accuracy_term = alpha_now * (MaxMS[size] / sum_bytes_MS)
#
#     Te(MS) = max(wait+compute) across models in MS
#     """
#     D_sec = deadline_ms / 1000.0
#     if D_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(MAX_MODEL_SIZE, K // C, len(M_TOTAL))
#     target_k = min(ensemble_size, max_parallel_models)
#     if target_k <= 0:
#         return [], {}
#
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     MaxMS: Dict[int, int] = {}
#     rolling_sum = 0
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_parallel_models:
#             MaxMS[size] = rolling_sum
#
#     def estimate_Te(MS: List[str]) -> float:
#         times = []
#         for m in MS:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(MS: List[str]):
#         T_e = estimate_Te(MS)
#         raw = T_e / D_sec
#         if raw >= 1.0:
#             return None, T_e, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, T_e, raw
#
#     def accuracy_term_for(MS: List[str]):
#         size = len(MS)
#         sum_bytes = sum(MODEL_SIZES[m] for m in MS)
#         if sum_bytes <= 0 or size not in MaxMS:
#             return None, sum_bytes
#         at = alpha_now * (MaxMS[size] / sum_bytes)
#         return at, sum_bytes
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, target_k, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         M_D = sorted_by_acc[:k]
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     if POLICY == "randomized":
#         import random
#         k = target_k
#         M_D = random.sample(M_TOTAL, k)
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     MS: List[str] = []
#     remaining = list(M_TOTAL)
#
#     logger.info(
#         f"[SELECT_BEGIN] deadline_ms={deadline_ms} D_sec={D_sec:.3f} "
#         f"target_k={target_k} max_parallel_models={max_parallel_models} alpha={alpha_now:.3f}"
#     )
#
#     for step in range(1, target_k + 1):
#         best_candidate = None
#         best_cost = float("inf")
#         best_dbg = None
#
#         for m in remaining:
#             trial = MS + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, T_e, raw = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             at, sum_bytes = accuracy_term_for(trial)
#             if at is None:
#                 continue
#
#             cost = lt + at
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#                 best_dbg = (trial, T_e, raw, lt, at, sum_bytes, cost)
#
#         if best_candidate is None:
#             logger.info(f"[SELECT_STOP] step={step} reason=no_feasible_candidate current_MS={MS}")
#             break
#
#         trial, T_e, raw, lt, at, sum_bytes, cost = best_dbg
#         logger.info(
#             f"[SELECT_CHOOSE] step={step}/{target_k} choose={best_candidate} "
#             f"trial={trial} Te={T_e:.4f}s raw={raw:.3f} lt={lt:.3f} at={at:.3f} "
#             f"sum_bytes={sum_bytes} cost={cost:.3f}"
#         )
#
#         MS.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not MS:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         MS = [fastest]
#         logger.warning(f"[SELECT_FALLBACK] no_feasible_set fallback_fastest={fastest}")
#
#     logger.info(f"[SELECT_DONE] selected_MS={MS} size={len(MS)} target_k={target_k}")
#     return MS, {m: wait_results.get(m, float("inf")) for m in MS}
#
#
# # -----------------------------
# # Sequential cascade execution
# # -----------------------------
# async def execute_sequential_cascade(
#     stub,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     """
#     Sequential behavior:
#     - run first model
#     - if first model passes -> return it directly
#     - if first fails, continue with next models
#     - before sending next model, check remaining deadline budget
#     - if deadline is not enough, stop and return current result(s)
#     - once >1 model executed, aggregation will happen later in caller
#     """
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={i+1}/{len(model_list)} model={m} start=True")
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#         stage_thr = _get_stage_threshold(i)
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if i == 0 and passed:
#             stop_reason = f"early_exit_first_pass={m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i > 0 and passed:
#             stop_reason = f"early_exit_with_aggregation_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} next_model={next_model} "
#                 f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Parallel-first cascade execution
# # -----------------------------
# async def execute_parallel_first_cascade(
#     stub,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     """
#     Parallel-first behavior:
#     - run first N models in parallel (typically 2)
#     - if first model passes -> return it directly
#     - if first fails but second passes -> aggregate both
#     - else before sending next model, check deadline budget
#     - if deadline is not enough, stop and return current result(s)
#     """
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         m1, m2 = first_batch[0], first_batch[1]
#         logger.info(f"[PARALLEL_FIRST] tx={transaction_id} first_batch={[m1, m2]} rest={rest_models}")
#
#         t1 = asyncio.create_task(send_request(stub, m1, image_b64))
#         t2 = asyncio.create_task(send_request(stub, m2, image_b64))
#
#         r1 = None
#         r2 = None
#         pass1 = False
#         pass2 = False
#
#         try:
#             r1 = await t1
#             if r1.get("success", False):
#                 conf1 = float(r1.get("probability", 0.0))
#                 thr1 = _get_stage_threshold(0)
#                 pass1 = conf1 >= thr1
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m1} label={r1.get('label')} conf={conf1:.4f} "
#                     f"rpc_latency_s={float(r1.get('latency', -1)):.4f} container_state={r1.get('container_state')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={m1} conf={conf1:.4f} "
#                     f"threshold={thr1:.2f} pass={pass1}"
#                 )
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m1} success=False")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m1} error={e}")
#
#         if r1 is not None and r1.get("success", False) and pass1:
#             if not t2.done():
#                 t2.cancel()
#                 try:
#                     await t2
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception:
#                     pass
#
#             results = [r1]
#             stop_reason = f"early_exit_first_pass={m1}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         try:
#             r2 = await t2
#             if r2.get("success", False):
#                 conf2 = float(r2.get("probability", 0.0))
#                 thr2 = _get_stage_threshold(1)
#                 pass2 = conf2 >= thr2
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m2} label={r2.get('label')} conf={conf2:.4f} "
#                     f"rpc_latency_s={float(r2.get('latency', -1)):.4f} container_state={r2.get('container_state')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={m2} conf={conf2:.4f} "
#                     f"threshold={thr2:.2f} pass={pass2}"
#                 )
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m2} success=False")
#         except asyncio.CancelledError:
#             pass
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m2} error={e}")
#
#         if r1 is not None and r1.get("success", False):
#             results.append(r1)
#         if r2 is not None and r2.get("success", False):
#             results.append(r2)
#
#         if (
#             r1 is not None and r1.get("success", False) and (not pass1) and
#             r2 is not None and r2.get("success", False) and pass2
#         ):
#             stop_reason = f"early_exit_second_pass_aggregate=({m1},{m2})"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#         logger.info(f"[RUN] tx={transaction_id} step=1/{len(model_list)} model={m} start=True")
#         try:
#             r = await send_request(stub, m, image_b64)
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m} label={r.get('label')} conf={conf:.4f} "
#                     f"rpc_latency_s={float(r.get('latency', -1)):.4f} container_state={r.get('container_state')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                     f"threshold={thr:.2f} pass={passed}"
#                 )
#                 results.append(r)
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                     return results, stop_reason
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#     for j, m in enumerate(rest_models):
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=m,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_model={m} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"next_est_ms={float(MODEL_PROFILES[m]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         stage_index = start_idx + j
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={stage_index+1}/{len(model_list)} model={m} start=True")
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         thr = _get_stage_threshold(stage_index)
#         passed = conf >= thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if passed:
#             stop_reason = f"early_exit_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Main invoke
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ) -> dict:
#     start_time = time.perf_counter()
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     # QPS estimate
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     # gRPC connection
#     async_channel = grpc.aio.insecure_channel(SERVER_ADDRESS)
#     stub = pb2_grpc.IluvatarWorkerStub(async_channel)
#
#     # Wait estimates
#     wait_results = wait_time_iluvatar.main()
#     all_wait_vals = [float(v) for v in wait_results.values() if v is not None]
#     if all_wait_vals:
#         logger.info(
#             f"[WAIT_SUMMARY] tx={transaction_id} mean_wait={sum(all_wait_vals)/len(all_wait_vals):.4f}s "
#             f"max_wait={max(all_wait_vals):.4f}s min_wait={min(all_wait_vals):.4f}s"
#         )
#     else:
#         logger.info(f"[WAIT_SUMMARY] tx={transaction_id} no_wait_data=True")
#
#     # Alpha
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     # Select ensemble
#     M_D, total_estimates = build_model_set(
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha_now,
#     )
#
#     if not M_D:
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         M_D = [fastest_model]
#         logger.warning(f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}")
#
#     # KEEP SELECTED ORDER EXACTLY AS RETURNED BY build_model_set()
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#     logger.info(f"[SELECT_WAITS] tx={transaction_id} waits_selected={ {m: float(total_estimates.get(m, -1)) for m in M_D} }")
#
#     # Execute according to config
#     if str(CASCADE_MODE).lower() == "sequential":
#         results, stop_reason = await execute_sequential_cascade(
#             stub=stub,
#             transaction_id=transaction_id,
#             model_list=M_D,
#             image_b64=image_b64,
#             start_time=start_time,
#             deadline_ms=deadline,
#         )
#     else:
#         results, stop_reason = await execute_parallel_first_cascade(
#             stub=stub,
#             transaction_id=transaction_id,
#             model_list=M_D,
#             image_b64=image_b64,
#             parallel_first_n=PARALLEL_FIRST_N,
#             start_time=start_time,
#             deadline_ms=deadline,
#         )
#
#     # If no successful results
#     if not results:
#         end_time = time.perf_counter()
#         e2e_ms = (end_time - start_time) * 1000
#         logger.error(f"[REQ_END] tx={transaction_id} success=False reason=no_successful_models e2e_ms={e2e_ms:.2f}")
#         return {
#             "selected_models": M_D,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": selected_folder,
#             "wait_times": total_estimates,
#             "per_model": {},
#             "ensemble_size": ENSEMBLE_SIZE,
#             "threshold": THRESHOLD,
#             "threshold_stage": THRESHOLD_STAGE,
#             "main_policy": POLICY,
#             "alpha": alpha_now,
#             "alpha_mode": ALPHA_MODE,
#             "qps_est": qps_est,
#             "stop_reason": stop_reason,
#             "log_file": LOG_PATH,
#             "cascade_mode": CASCADE_MODE,
#             "parallel_first_n": PARALLEL_FIRST_N,
#         }
#
#     # Single model -> no aggregation
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#         logger.info(
#             f"[SINGLE] tx={transaction_id} model={single.get('model')} "
#             f"label={final_result['label']} conf={final_result['accuracy']:.4f}"
#         )
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#         logger.info(
#             f"[COMBINE] tx={transaction_id} executed_models={[r['model'] for r in results]} "
#             f"combiner={final_result.get('combiner_policy')} final_label={final_result.get('label')} "
#             f"final_acc={final_result.get('accuracy')}"
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     # Feedback updates
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (1 - ETA) * OMEGA[res["model"]] + ETA * float(res.get("probability", 0.0))
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     per_model = {}
#     for res in results:
#         per_model[res["model"]] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#         }
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} alpha={alpha_now:.3f} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#         "selected_folder": selected_folder,
#         "wait_times": total_estimates,
#         "per_model": per_model,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "stop_reason": stop_reason,
#         "log_file": LOG_PATH,
#         "cascade_mode": CASCADE_MODE,
#         "parallel_first_n": PARALLEL_FIRST_N,
#     }



###########
#fully works code start
#####

# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# # Use ONE set of protobuf imports (avoid duplicates)
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     MAX_MODEL_SIZE,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     MODEL_SIZES,
#
#     # --- Cascade parameters ---
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#
#     # --- Alpha control ---
#     ALPHA_MODE,        # "static" or "adaptive"
#     ALPHA_STATIC,      # used if static mode
#     ALPHA_MIN,         # used if adaptive mode
#     ALPHA_MAX,
#
#     # --- Load normalization ---
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#
#     # --- Cascade execution mode ---
#     CASCADE_MODE,        # "sequential" or "parallel_first"
#     PARALLEL_FIRST_N,    # e.g. 2
#     PARALLEL_BATCH_SIZE, # e.g. 1 for sequential after first batch
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
#
#
# # -----------------------------
# # Logging setup (file + console)
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking (sliding window)
# # -----------------------------
# REQUEST_TS = deque()
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Load-adaptive alpha (QPS only)
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     load_score = qps_norm
#     alpha_now = alpha_max - load_score * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     """
#     Check whether there is enough remaining deadline budget to invoke next_model.
#
#     Returns:
#         stop_now, elapsed_ms, remaining_ms
#     """
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Ilúvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     """
#     Stage-specific threshold:
#       - if THRESHOLD_STAGE list exists and has entry for stage_index, use it
#       - otherwise fallback to THRESHOLD
#     """
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#     return float(THRESHOLD)
#
#
# # -----------------------------
# # Model selection: up to ENSEMBLE_SIZE
# # -----------------------------
# def build_model_set(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Greedy selection up to size = ensemble_size (if feasible).
#
#     cost = latency_term + accuracy_term
#       latency_term = clamp(Te/D, 0.1, 0.9), only if Te/D < 1
#       accuracy_term = alpha_now * (MaxMS[size] / sum_bytes_MS)
#
#     Te(MS) = max(wait+compute) across models in MS
#     """
#     D_sec = deadline_ms / 1000.0
#     if D_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ENSEMBLE_SIZE, K // C, len(M_TOTAL))
#     target_k = min(ensemble_size, max_parallel_models)
#     if target_k <= 0:
#         return [], {}
#
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     MaxMS: Dict[int, int] = {}
#     rolling_sum = 0
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_parallel_models:
#             MaxMS[size] = rolling_sum
#
#     def estimate_Te(MS: List[str]) -> float:
#         times = []
#         for m in MS:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(MS: List[str]):
#         T_e = estimate_Te(MS)
#         raw = T_e / D_sec
#         if raw >= 1.0:
#             return None, T_e, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, T_e, raw
#
#     def accuracy_term_for(MS: List[str]):
#         size = len(MS)
#         sum_bytes = sum(MODEL_SIZES[m] for m in MS)
#         if sum_bytes <= 0 or size not in MaxMS:
#             return None, sum_bytes
#         at = alpha_now * (MaxMS[size] / sum_bytes)
#         return at, sum_bytes
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, target_k, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         M_D = sorted_by_acc[:k]
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     if POLICY == "randomized":
#         import random
#         k = target_k
#         M_D = random.sample(M_TOTAL, k)
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     MS: List[str] = []
#     remaining = list(M_TOTAL)
#
#     logger.info(
#         f"[SELECT_BEGIN] deadline_ms={deadline_ms} D_sec={D_sec:.3f} "
#         f"target_k={target_k} max_parallel_models={max_parallel_models} alpha={alpha_now:.3f}"
#     )
#
#     for step in range(1, target_k + 1):
#         best_candidate = None
#         best_cost = float("inf")
#         best_dbg = None
#
#         for m in remaining:
#             trial = MS + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, T_e, raw = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             at, sum_bytes = accuracy_term_for(trial)
#             if at is None:
#                 continue
#
#             cost = lt + at
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#                 best_dbg = (trial, T_e, raw, lt, at, sum_bytes, cost)
#
#         if best_candidate is None:
#             logger.info(f"[SELECT_STOP] step={step} reason=no_feasible_candidate current_MS={MS}")
#             break
#
#         trial, T_e, raw, lt, at, sum_bytes, cost = best_dbg
#         logger.info(
#             f"[SELECT_CHOOSE] step={step}/{target_k} choose={best_candidate} "
#             f"trial={trial} Te={T_e:.4f}s raw={raw:.3f} lt={lt:.3f} at={at:.3f} "
#             f"sum_bytes={sum_bytes} cost={cost:.3f}"
#         )
#
#         MS.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not MS:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         MS = [fastest]
#         logger.warning(f"[SELECT_FALLBACK] no_feasible_set fallback_fastest={fastest}")
#
#     logger.info(f"[SELECT_DONE] selected_MS={MS} size={len(MS)} target_k={target_k}")
#     return MS, {m: wait_results.get(m, float("inf")) for m in MS}
#
#
# # -----------------------------
# # Sequential cascade execution
# # -----------------------------
# async def execute_sequential_cascade(
#     stub,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     """
#     Sequential behavior:
#     - run first model
#     - if first model passes -> return it directly
#     - if first fails, continue with next models
#     - before sending next model, check remaining deadline budget
#     - if deadline is not enough, stop and return current result(s)
#     - once >1 model executed, aggregation will happen later in caller
#     """
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={i+1}/{len(model_list)} model={m} start=True")
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#         stage_thr = _get_stage_threshold(i)
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if i == 0 and passed:
#             stop_reason = f"early_exit_first_pass={m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i > 0 and passed:
#             stop_reason = f"early_exit_with_aggregation_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} next_model={next_model} "
#                 f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
# async def execute_parallel_first_finish_cascade(
#     stub,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
# ) -> Tuple[List[dict], str]:
#     """
#     Parallel-first-finish behavior:
#     1) Launch first N models in parallel (normally N=2)
#     2) Whichever model finishes first becomes the first decision model
#     3) If first finished model passes threshold -> return it directly
#     4) Otherwise wait for the remaining models in the first batch
#     5) Aggregate first-batch outputs and check aggregated confidence
#     6) If aggregate passes threshold -> stop
#     7) Else continue with remaining models in batches of parallel_batch_size:
#          - before sending next batch, check deadline budget
#          - send next batch in parallel
#          - aggregate all executed models
#          - check aggregated confidence
#     8) Return all executed results to caller
#        (caller can do final combine_outputs with update_weights=True)
#     """
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     # -------------------------------------------------
#     # Case A: first parallel batch
#     # -------------------------------------------------
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(send_request(stub, m, image_b64)): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 first_thr = _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={first_model} "
#                     f"conf={first_conf:.4f} threshold={first_thr:.2f} pass={first_pass}"
#                 )
#             else:
#                 logger.warning(
#                     f"[RUN_BAD] tx={transaction_id} first_finished_model={first_model} success=False"
#                 )
#         except Exception as e:
#             logger.exception(
#                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
#             )
#
#         # If first finished model passes -> cancel remaining first-batch tasks
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception:
#                     pass
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         # Wait for remaining first-batch tasks
#         remaining_results: List[dict] = []
#         for p in pending:
#             model_name = task_map[p]
#             try:
#                 res = await p
#                 if res.get("success", False):
#                     conf = float(res.get("probability", 0.0))
#                     thr = _get_stage_threshold(1)
#                     passed = conf >= thr
#
#                     logger.info(
#                         f"[RUN_OK] tx={transaction_id} model={model_name} "
#                         f"label={res.get('label')} conf={conf:.4f} "
#                         f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                         f"container_state={res.get('container_state')}"
#                     )
#                     logger.info(
#                         f"[THRESH] tx={transaction_id} model={model_name} "
#                         f"conf={conf:.4f} threshold={thr:.2f} pass={passed}"
#                     )
#                     remaining_results.append(res)
#                 else:
#                     logger.warning(f"[RUN_BAD] tx={transaction_id} model={model_name} success=False")
#             except Exception as e:
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#         # Keep successful outputs from first batch
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#         results.extend(remaining_results)
#
#         # Aggregate first batch if we have at least 2 successful results
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,  # temporary check only
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_thr = _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = (
#                     f"early_exit_aggregate_pass_after_first_batch="
#                     f"{[r['model'] for r in results]}"
#                 )
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     # -------------------------------------------------
#     # Case B: only one model in first stage
#     # -------------------------------------------------
#     elif first_n == 1:
#         m = first_batch[0]
#         logger.info(f"[RUN] tx={transaction_id} step=1/{len(model_list)} model={m} start=True")
#         try:
#             r = await send_request(stub, m, image_b64)
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m} "
#                     f"label={r.get('label')} conf={conf:.4f} "
#                     f"rpc_latency_s={float(r.get('latency', -1)):.4f} "
#                     f"container_state={r.get('container_state')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={m} "
#                     f"conf={conf:.4f} threshold={thr:.2f} pass={passed}"
#                 )
#
#                 results.append(r)
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                     return results, stop_reason
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     # -------------------------------------------------
#     # Case C: continue with remaining models in batches
#     # -------------------------------------------------
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         # Deadline check before launching batch
#         first_next_model = batch[0]
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         logger.info(f"[PARALLEL_BATCH] tx={transaction_id} batch={batch}")
#
#         task_map = {
#             asyncio.create_task(send_request(stub, m, image_b64)): m
#             for m in batch
#         }
#
#         batch_raw = await asyncio.gather(*task_map.keys(), return_exceptions=True)
#
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#             if not res.get("success", False):
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
#                 f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')}"
#             )
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         # Only check aggregation if we got at least one new success
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,  # temporary check only
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Main invoke
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ) -> dict:
#     start_time = time.perf_counter()
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     # QPS estimate
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     # gRPC connection
#     async_channel = grpc.aio.insecure_channel(SERVER_ADDRESS)
#     stub = pb2_grpc.IluvatarWorkerStub(async_channel)
#
#     # Wait estimates
#     wait_results = wait_time_iluvatar.main()
#     all_wait_vals = [float(v) for v in wait_results.values() if v is not None]
#     if all_wait_vals:
#         logger.info(
#             f"[WAIT_SUMMARY] tx={transaction_id} mean_wait={sum(all_wait_vals)/len(all_wait_vals):.4f}s "
#             f"max_wait={max(all_wait_vals):.4f}s min_wait={min(all_wait_vals):.4f}s"
#         )
#     else:
#         logger.info(f"[WAIT_SUMMARY] tx={transaction_id} no_wait_data=True")
#
#     # Alpha
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} parallel_first_n={PARALLEL_FIRST_N} "
#         f"parallel_batch_size={PARALLEL_BATCH_SIZE}"
#     )
#
#     # Select ensemble
#     M_D, total_estimates = build_model_set(
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha_now,
#     )
#
#     if not M_D:
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         M_D = [fastest_model]
#         logger.warning(f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}")
#
#     # KEEP SELECTED ORDER EXACTLY AS RETURNED BY build_model_set()
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#     logger.info(f"[SELECT_WAITS] tx={transaction_id} waits_selected={ {m: float(total_estimates.get(m, -1)) for m in M_D} }")
#
#     # Execute according to config
#     if str(CASCADE_MODE).lower() == "sequential":
#         results, stop_reason = await execute_sequential_cascade(
#             stub=stub,
#             transaction_id=transaction_id,
#             model_list=M_D,
#             image_b64=image_b64,
#             start_time=start_time,
#             deadline_ms=deadline,
#         )
#     else:
#         # results, stop_reason = await execute_parallel_first_cascade(
#         #     stub=stub,
#         #     transaction_id=transaction_id,
#         #     model_list=M_D,
#         #     image_b64=image_b64,
#         #     parallel_first_n=PARALLEL_FIRST_N,
#         #     parallel_batch_size=PARALLEL_BATCH_SIZE,
#         #     start_time=start_time,
#         #     deadline_ms=deadline,
#         # )
#
#         results, stop_reason = await execute_parallel_first_finish_cascade(
#             stub=stub,
#             transaction_id=transaction_id,
#             model_list=M_D,
#             image_b64=image_b64,
#             parallel_first_n=PARALLEL_FIRST_N,
#             parallel_batch_size=PARALLEL_BATCH_SIZE,
#             start_time=start_time,
#             deadline_ms=deadline,
#             selected_folder=selected_folder,
#         )
#
#     # If no successful results
#     if not results:
#         end_time = time.perf_counter()
#         e2e_ms = (end_time - start_time) * 1000
#         logger.error(f"[REQ_END] tx={transaction_id} success=False reason=no_successful_models e2e_ms={e2e_ms:.2f}")
#         return {
#             "selected_models": M_D,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": selected_folder,
#             "wait_times": total_estimates,
#             "per_model": {},
#             "ensemble_size": ENSEMBLE_SIZE,
#             "threshold": THRESHOLD,
#             "threshold_stage": THRESHOLD_STAGE,
#             "main_policy": POLICY,
#             "alpha": alpha_now,
#             "alpha_mode": ALPHA_MODE,
#             "qps_est": qps_est,
#             "stop_reason": stop_reason,
#             "log_file": LOG_PATH,
#             "cascade_mode": CASCADE_MODE,
#             "parallel_first_n": PARALLEL_FIRST_N,
#             "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         }
#
#     # Single model -> no aggregation
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#         logger.info(
#             f"[SINGLE] tx={transaction_id} model={single.get('model')} "
#             f"label={final_result['label']} conf={final_result['accuracy']:.4f}"
#         )
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#         logger.info(
#             f"[COMBINE] tx={transaction_id} executed_models={[r['model'] for r in results]} "
#             f"combiner={final_result.get('combiner_policy')} final_label={final_result.get('label')} "
#             f"final_acc={final_result.get('accuracy')}"
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     # Feedback updates
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (1 - ETA) * OMEGA[res["model"]] + ETA * float(res.get("probability", 0.0))
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     per_model = {}
#     for res in results:
#         per_model[res["model"]] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#         }
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} alpha={alpha_now:.3f} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#         "selected_folder": selected_folder,
#         "wait_times": total_estimates,
#         "per_model": per_model,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "stop_reason": stop_reason,
#         "log_file": LOG_PATH,
#         "cascade_mode": CASCADE_MODE,
#         "parallel_first_n": PARALLEL_FIRST_N,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#     }


###########
#fully works code end
#####


###########
#dual instance works code start
#####

#
# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     MAX_MODEL_SIZE,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     MODEL_SIZES,
#
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#
#     ALPHA_MODE,
#     ALPHA_STATIC,
#     ALPHA_MIN,
#     ALPHA_MAX,
#
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#
#     CASCADE_MODE,
#     PARALLEL_FIRST_N,
#     PARALLEL_BATCH_SIZE,
#
#     # Manual multi-instance routing
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
#
#
# # -----------------------------
# # Logging setup
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking
# # -----------------------------
# REQUEST_TS = deque()
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Adaptive alpha
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     load_score = qps_norm
#     alpha_now = alpha_max - load_score * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Manual routing helpers
# # -----------------------------
# def get_server_address_for_model(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return LARGE_MODEL_SERVER_ADDRESS
#
#     raise ValueError(f"Model {model_name} is not assigned to any Iluvatar instance.")
#
#
# def build_stub_map_for_selected_models(model_list: List[str]):
#     addresses = {get_server_address_for_model(m) for m in model_list}
#     channel_map = {}
#     stub_map = {}
#
#     for addr in addresses:
#         channel = grpc.aio.insecure_channel(addr)
#         stub = pb2_grpc.IluvatarWorkerStub(channel)
#         channel_map[addr] = channel
#         stub_map[addr] = stub
#
#     return channel_map, stub_map
#
#
# async def close_channels(channel_map):
#     for channel in channel_map.values():
#         try:
#             await channel.close()
#         except Exception:
#             pass
#
#
# def get_stub_for_model(model_name: str, stub_map):
#     addr = get_server_address_for_model(model_name)
#     return stub_map[addr]
#
#
# def get_wait_results_for_models(model_list: List[str]) -> Dict[str, float]:
#     """
#     Query wait times from the correct instance(s) using manual routing.
#     Requires wait_time_iluvatar.main(server_address=..., model_list=...)
#     """
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return wait_time_iluvatar.main(
#             server_address=SERVER_ADDRESS,
#             model_list=model_list,
#         )
#
#     small_models = [m for m in model_list if m in SMALL_INSTANCE_MODELS]
#     large_models = [m for m in model_list if m in LARGE_INSTANCE_MODELS]
#
#     merged = {}
#
#     if small_models:
#         small_waits = wait_time_iluvatar.main(
#             server_address=SMALL_MODEL_SERVER_ADDRESS,
#             model_list=small_models,
#         )
#         merged.update(small_waits)
#
#     if large_models:
#         large_waits = wait_time_iluvatar.main(
#             server_address=LARGE_MODEL_SERVER_ADDRESS,
#             model_list=large_models,
#         )
#         merged.update(large_waits)
#
#     return merged
#
#
# # -----------------------------
# # Iluvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#         "server_address": get_server_address_for_model(model_name),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#     return float(THRESHOLD)
#
#
# # -----------------------------
# # Model selection
# # -----------------------------
# def build_model_set(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Greedy selection up to size = ensemble_size (if feasible).
#
#     cost = (1 - alpha_now) * latency_term + alpha_now * size_cost
#       latency_term = clamp(Te/D, 0.1, 0.9), only if Te/D < 1
#       size_cost    = MaxMS[size] / sum_bytes_MS
#
#     Te(MS) = max(wait+compute) across models in MS
#     """
#     D_sec = deadline_ms / 1000.0
#     if D_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ENSEMBLE_SIZE, K // C, len(M_TOTAL))
#     target_k = min(ensemble_size, max_parallel_models)
#     if target_k <= 0:
#         return [], {}
#
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     MaxMS: Dict[int, int] = {}
#     rolling_sum = 0
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_parallel_models:
#             MaxMS[size] = rolling_sum
#
#     def estimate_Te(MS: List[str]) -> float:
#         times = []
#         for m in MS:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(MS: List[str]):
#         T_e = estimate_Te(MS)
#         raw = T_e / D_sec
#         if raw >= 1.0:
#             return None, T_e, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, T_e, raw
#
#     def size_cost_for(MS: List[str]):
#         size = len(MS)
#         sum_bytes = sum(MODEL_SIZES[m] for m in MS)
#         if sum_bytes <= 0 or size not in MaxMS:
#             return None, sum_bytes
#         sc = MaxMS[size] / sum_bytes
#         return sc, sum_bytes
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, target_k, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         M_D = sorted_by_acc[:k]
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     if POLICY == "randomized":
#         import random
#         k = target_k
#         M_D = random.sample(M_TOTAL, k)
#         return M_D, {m: wait_results.get(m, float("inf")) for m in M_D}
#
#     MS: List[str] = []
#     remaining = list(M_TOTAL)
#
#     logger.info(
#         f"[SELECT_BEGIN] deadline_ms={deadline_ms} D_sec={D_sec:.3f} "
#         f"target_k={target_k} max_parallel_models={max_parallel_models} alpha={alpha_now:.3f}"
#     )
#
#     for step in range(1, target_k + 1):
#         best_candidate = None
#         best_cost = float("inf")
#         best_dbg = None
#
#         for m in remaining:
#             trial = MS + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, T_e, raw = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             sc, sum_bytes = size_cost_for(trial)
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#                 best_dbg = (trial, T_e, raw, lt, sc, sum_bytes, cost)
#
#         if best_candidate is None:
#             logger.info(f"[SELECT_STOP] step={step} reason=no_feasible_candidate current_MS={MS}")
#             break
#
#         trial, T_e, raw, lt, sc, sum_bytes, cost = best_dbg
#         logger.info(
#             f"[SELECT_CHOOSE] step={step}/{target_k} choose={best_candidate} "
#             f"trial={trial} Te={T_e:.4f}s raw={raw:.3f} lt={lt:.3f} "
#             f"size_cost={sc:.3f} sum_bytes={sum_bytes} cost={cost:.3f}"
#         )
#
#         MS.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not MS:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         MS = [fastest]
#         logger.warning(f"[SELECT_FALLBACK] no_feasible_set fallback_fastest={fastest}")
#
#     logger.info(f"[SELECT_DONE] selected_MS={MS} size={len(MS)} target_k={target_k}")
#     return MS, {m: wait_results.get(m, float("inf")) for m in MS}
#
#
# # -----------------------------
# # Sequential cascade execution
# # -----------------------------
# async def execute_sequential_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={i+1}/{len(model_list)} model={m} start=True")
#
#         stub = get_stub_for_model(m, stub_map)
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#         stage_thr = _get_stage_threshold(i)
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if i == 0 and passed:
#             stop_reason = f"early_exit_first_pass={m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i > 0 and passed:
#             stop_reason = f"early_exit_with_aggregation_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} next_model={next_model} "
#                 f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Full parallel execution
# # -----------------------------
# async def execute_full_parallel_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
# ) -> Tuple[List[dict], str]:
#     logger.info(f"[FULL_PARALLEL_START] tx={transaction_id} models={model_list}")
#
#     tasks = []
#     task_models = []
#
#     for m in model_list:
#         stub = get_stub_for_model(m, stub_map)
#         tasks.append(asyncio.create_task(send_request(stub, m, image_b64)))
#         task_models.append(m)
#
#     raw_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     results: List[dict] = []
#     for idx, out in enumerate(raw_results):
#         m = task_models[idx]
#
#         if isinstance(out, Exception):
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#             continue
#
#         res = out
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#             continue
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} "
#             f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         results.append(res)
#
#     return results, "full_parallel_executed"
#
#
# # -----------------------------
# # Parallel-first-finish cascade
# # -----------------------------
# async def execute_parallel_first_finish_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64)): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 first_thr = _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')} "
#                     f"server={first_result.get('server_address')}"
#                 )
#             else:
#                 logger.warning(
#                     f"[RUN_BAD] tx={transaction_id} first_finished_model={first_model} success=False"
#                 )
#         except Exception as e:
#             logger.exception(
#                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
#             )
#
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#                 try:
#                     await p
#                 except Exception:
#                     pass
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         remaining_results: List[dict] = []
#         for p in pending:
#             model_name = task_map[p]
#             try:
#                 res = await p
#                 if res.get("success", False):
#                     remaining_results.append(res)
#                 else:
#                     logger.warning(f"[RUN_BAD] tx={transaction_id} model={model_name} success=False")
#             except Exception as e:
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#         results.extend(remaining_results)
#
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_thr = _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_first_batch={[r['model'] for r in results]}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#         try:
#             stub = get_stub_for_model(m, stub_map)
#             r = await send_request(stub, m, image_b64)
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#                 results.append(r)
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                     return results, stop_reason
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         first_next_model = batch[0]
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         logger.info(f"[PARALLEL_BATCH] tx={transaction_id} batch={batch}")
#
#         tasks = [
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64))
#             for m in batch
#         ]
#
#         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
#
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#             if not res.get("success", False):
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
#                 f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
#             )
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Main invoke
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ) -> dict:
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} parallel_first_n={PARALLEL_FIRST_N} "
#         f"parallel_batch_size={PARALLEL_BATCH_SIZE} "
#         f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES}"
#     )
#
#     # Wait estimates using manual routing
#     wait_results = get_wait_results_for_models(M_TOTAL)
#
#     all_wait_vals = [float(v) for v in wait_results.values() if v is not None and v != float("inf")]
#     if all_wait_vals:
#         logger.info(
#             f"[WAIT_SUMMARY] tx={transaction_id} mean_wait={sum(all_wait_vals)/len(all_wait_vals):.4f}s "
#             f"max_wait={max(all_wait_vals):.4f}s min_wait={min(all_wait_vals):.4f}s"
#         )
#     else:
#         logger.info(f"[WAIT_SUMMARY] tx={transaction_id} no_wait_data=True")
#
#     M_D, total_estimates = build_model_set(
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha_now,
#     )
#
#     if not M_D:
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         M_D = [fastest_model]
#         total_estimates = {fastest_model: wait_results.get(fastest_model, float("inf"))}
#         logger.warning(f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}")
#
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#     logger.info(
#         f"[SELECT_WAITS] tx={transaction_id} waits_selected="
#         f"{ {m: float(total_estimates.get(m, -1)) for m in M_D} }"
#     )
#     logger.info(
#         f"[SELECT_ROUTE] tx={transaction_id} routes="
#         f"{ {m: get_server_address_for_model(m) for m in M_D} }"
#     )
#
#     channel_map, stub_map = build_stub_map_for_selected_models(M_D)
#
#     try:
#         if str(CASCADE_MODE).lower() == "sequential":
#             results, stop_reason = await execute_sequential_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#             )
#         elif str(CASCADE_MODE).lower() == "full_parallel":
#             results, stop_reason = await execute_full_parallel_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#             )
#         else:
#             results, stop_reason = await execute_parallel_first_finish_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 parallel_first_n=PARALLEL_FIRST_N,
#                 parallel_batch_size=PARALLEL_BATCH_SIZE,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_folder=selected_folder,
#             )
#     finally:
#         await close_channels(channel_map)
#
#     if not results:
#         end_time = time.perf_counter()
#         e2e_ms = (end_time - start_time) * 1000
#         logger.error(f"[REQ_END] tx={transaction_id} success=False reason=no_successful_models e2e_ms={e2e_ms:.2f}")
#         return {
#             "selected_models": M_D,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": selected_folder,
#             "wait_times": total_estimates,
#             "per_model": {},
#             "ensemble_size": ENSEMBLE_SIZE,
#             "threshold": THRESHOLD,
#             "threshold_stage": THRESHOLD_STAGE,
#             "main_policy": POLICY,
#             "alpha": alpha_now,
#             "alpha_mode": ALPHA_MODE,
#             "qps_est": qps_est,
#             "stop_reason": stop_reason,
#             "log_file": LOG_PATH,
#             "cascade_mode": CASCADE_MODE,
#             "parallel_first_n": PARALLEL_FIRST_N,
#             "parallel_batch_size": PARALLEL_BATCH_SIZE,
#             "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#         }
#
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#         logger.info(
#             f"[SINGLE] tx={transaction_id} model={single.get('model')} "
#             f"label={final_result['label']} conf={final_result['accuracy']:.4f}"
#         )
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#         logger.info(
#             f"[COMBINE] tx={transaction_id} executed_models={[r['model'] for r in results]} "
#             f"combiner={final_result.get('combiner_policy')} final_label={final_result.get('label')} "
#             f"final_acc={final_result.get('accuracy')}"
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (1 - ETA) * OMEGA[res["model"]] + ETA * float(res.get("probability", 0.0))
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     per_model = {}
#     for res in results:
#         per_model[res["model"]] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#             "server_address": res.get("server_address"),
#         }
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} alpha={alpha_now:.3f} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#         "selected_folder": selected_folder,
#         "wait_times": total_estimates,
#         "per_model": per_model,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "stop_reason": stop_reason,
#         "log_file": LOG_PATH,
#         "cascade_mode": CASCADE_MODE,
#         "parallel_first_n": PARALLEL_FIRST_N,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#     }



###########
#dual instance works code end
#####

###########
#new model selection f  code start
#####


# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     MAX_MODEL_SIZE,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     MODEL_SIZES,
#
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#
#     ALPHA_MODE,
#     ALPHA_STATIC,
#     ALPHA_MIN,
#     ALPHA_MAX,
#
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#
#     CASCADE_MODE,
#     PARALLEL_FIRST_N,
#     PARALLEL_BATCH_SIZE,
#     NETWORK_LATENCY,
#     USE_DEADLINE_FAST_PATH,
#
#     # Manual multi-instance routing
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
# from ckn_controller.ensemble_selector import (
#     build_model_set_by_mode,
#     estimate_selected_set_latency_for_mode,
# )
#
#
# # -----------------------------
# # Logging setup
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking
# # -----------------------------
# REQUEST_TS = deque()
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Adaptive alpha
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     load_score = qps_norm
#     alpha_now = alpha_max - load_score * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Manual routing helpers
# # -----------------------------
# def get_server_address_for_model(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return LARGE_MODEL_SERVER_ADDRESS
#
#     raise ValueError(f"Model {model_name} is not assigned to any Iluvatar instance.")
#
#
# def build_stub_map_for_selected_models(model_list: List[str]):
#     addresses = {get_server_address_for_model(m) for m in model_list}
#     channel_map = {}
#     stub_map = {}
#
#     for addr in addresses:
#         channel = grpc.aio.insecure_channel(addr)
#         stub = pb2_grpc.IluvatarWorkerStub(channel)
#         channel_map[addr] = channel
#         stub_map[addr] = stub
#
#     return channel_map, stub_map
#
#
# async def close_channels(channel_map):
#     for channel in channel_map.values():
#         try:
#             await channel.close()
#         except Exception:
#             pass
#
#
# def get_stub_for_model(model_name: str, stub_map):
#     addr = get_server_address_for_model(model_name)
#     return stub_map[addr]
#
#
# def get_wait_results_for_models(model_list: List[str]) -> Dict[str, float]:
#     """
#     Query wait times from the correct instance(s) using manual routing.
#     Requires wait_time_iluvatar.main(server_address=..., model_list=...)
#     """
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return wait_time_iluvatar.main(
#             server_address=SERVER_ADDRESS,
#             model_list=model_list,
#         )
#
#     small_models = [m for m in model_list if m in SMALL_INSTANCE_MODELS]
#     large_models = [m for m in model_list if m in LARGE_INSTANCE_MODELS]
#
#     merged = {}
#
#     if small_models:
#         small_waits = wait_time_iluvatar.main(
#             server_address=SMALL_MODEL_SERVER_ADDRESS,
#             model_list=small_models,
#         )
#         merged.update(small_waits)
#
#     if large_models:
#         large_waits = wait_time_iluvatar.main(
#             server_address=LARGE_MODEL_SERVER_ADDRESS,
#             model_list=large_models,
#         )
#         merged.update(large_waits)
#
#     return merged
#
#
# # -----------------------------
# # Iluvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#         "server_address": get_server_address_for_model(model_name),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#     return float(THRESHOLD)
#
#
# # -----------------------------
# # Sequential cascade execution
# # -----------------------------
# async def execute_sequential_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={i+1}/{len(model_list)} model={m} start=True")
#
#         stub = get_stub_for_model(m, stub_map)
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#         stage_thr = _get_stage_threshold(i)
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if i == 0 and passed:
#             stop_reason = f"early_exit_first_pass={m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i > 0 and passed:
#             stop_reason = f"early_exit_with_aggregation_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} next_model={next_model} "
#                 f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # MODE-S execution
# # -----------------------------
# async def execute_mode_s_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
# ) -> Tuple[List[dict], str]:
#     logger.info(f"[MODE_S_START] tx={transaction_id} models={model_list}")
#
#     tasks = []
#     task_models = []
#
#     for m in model_list:
#         stub = get_stub_for_model(m, stub_map)
#         tasks.append(asyncio.create_task(send_request(stub, m, image_b64)))
#         task_models.append(m)
#
#     raw_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     results: List[dict] = []
#     for idx, out in enumerate(raw_results):
#         m = task_models[idx]
#
#         if isinstance(out, Exception):
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#             continue
#
#         res = out
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#             continue
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} "
#             f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         results.append(res)
#
#     return results, "mode_s_executed"
#
#
# # -----------------------------
# # Parallel-first-finish cascade
# # -----------------------------
# # async def execute_parallel_first_finish_cascade(
# #     stub_map,
# #     transaction_id: str,
# #     model_list: List[str],
# #     image_b64: str,
# #     parallel_first_n: int,
# #     parallel_batch_size: int,
# #     start_time: float,
# #     deadline_ms: int,
# #     selected_folder: str,
# # ) -> Tuple[List[dict], str]:
# #     results: List[dict] = []
# #     stop_reason = "ran_all"
# #
# #     first_n = max(0, min(int(parallel_first_n), len(model_list)))
# #     batch_size = max(1, int(parallel_batch_size))
# #
# #     first_batch = model_list[:first_n]
# #     rest_models = model_list[first_n:]
# #
# #     if first_n >= 2:
# #         logger.info(
# #             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
# #             f"first_batch={first_batch} rest={rest_models}"
# #         )
# #
# #         task_map = {
# #             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64)): m
# #             for m in first_batch
# #         }
# #
# #         done, pending = await asyncio.wait(
# #             set(task_map.keys()),
# #             return_when=asyncio.FIRST_COMPLETED,
# #         )
# #
# #         first_task = next(iter(done))
# #         first_model = task_map[first_task]
# #
# #         first_result = None
# #         first_pass = False
# #
# #         try:
# #             first_result = first_task.result()
# #             if first_result.get("success", False):
# #                 first_conf = float(first_result.get("probability", 0.0))
# #                 first_thr = _get_stage_threshold(0)
# #                 first_pass = first_conf >= first_thr
# #
# #                 logger.info(
# #                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
# #                     f"label={first_result.get('label')} conf={first_conf:.4f} "
# #                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
# #                     f"container_state={first_result.get('container_state')} "
# #                     f"server={first_result.get('server_address')}"
# #                 )
# #             else:
# #                 logger.warning(
# #                     f"[RUN_BAD] tx={transaction_id} first_finished_model={first_model} success=False"
# #                 )
# #         except Exception as e:
# #             logger.exception(
# #                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
# #             )
# #
# #         if first_result is not None and first_result.get("success", False) and first_pass:
# #             for p in pending:
# #                 p.cancel()
# #                 try:
# #                     await p
# #                 except Exception:
# #                     pass
# #
# #             results = [first_result]
# #             stop_reason = f"early_exit_first_finished_pass={first_model}"
# #             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
# #             return results, stop_reason
# #
# #         remaining_results: List[dict] = []
# #         for p in pending:
# #             model_name = task_map[p]
# #             try:
# #                 res = await p
# #                 if res.get("success", False):
# #                     remaining_results.append(res)
# #                 else:
# #                     logger.warning(f"[RUN_BAD] tx={transaction_id} model={model_name} success=False")
# #             except Exception as e:
# #                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
# #
# #         if first_result is not None and first_result.get("success", False):
# #             results.append(first_result)
# #         results.extend(remaining_results)
# #
# #         if len(results) >= 2:
# #             temp_final = combine_outputs(
# #                 results,
# #                 policy="weighted_majority",
# #                 historical_acc=OMEGA,
# #                 model_weights=MODEL_WEIGHTS,
# #                 ground_truth=selected_folder,
# #                 gamma=GAMMA,
# #                 update_weights=False,
# #                 label_matcher=wnid_matches_text_label,
# #                 rho=RHO,
# #             )
# #
# #             agg_conf = float(temp_final.get("accuracy", 0.0))
# #             agg_thr = _get_stage_threshold(1)
# #             agg_pass = agg_conf >= agg_thr
# #
# #             logger.info(
# #                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
# #                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
# #                 f"threshold={agg_thr:.2f} pass={agg_pass}"
# #             )
# #
# #             if agg_pass:
# #                 stop_reason = f"early_exit_aggregate_pass_after_first_batch={[r['model'] for r in results]}"
# #                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
# #                 return results, stop_reason
# #
# #     elif first_n == 1:
# #         m = first_batch[0]
# #         try:
# #             stub = get_stub_for_model(m, stub_map)
# #             r = await send_request(stub, m, image_b64)
# #             if r.get("success", False):
# #                 conf = float(r.get("probability", 0.0))
# #                 thr = _get_stage_threshold(0)
# #                 passed = conf >= thr
# #                 results.append(r)
# #
# #                 if passed:
# #                     stop_reason = f"early_exit_first_pass={m}"
# #                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
# #                     return results, stop_reason
# #             else:
# #                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
# #         except Exception as e:
# #             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
# #
# #     start_idx = first_n
# #
# #     for batch_start in range(0, len(rest_models), batch_size):
# #         batch = rest_models[batch_start: batch_start + batch_size]
# #
# #         first_next_model = batch[0]
# #         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
# #             start_time=start_time,
# #             deadline_ms=deadline_ms,
# #             next_model=first_next_model,
# #         )
# #
# #         logger.info(
# #             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
# #             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
# #             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
# #             f"stop={stop_now}"
# #         )
# #
# #         if stop_now:
# #             stop_reason = f"deadline_stop_before_{first_next_model}"
# #             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
# #             break
# #
# #         logger.info(f"[PARALLEL_BATCH] tx={transaction_id} batch={batch}")
# #
# #         tasks = [
# #             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64))
# #             for m in batch
# #         ]
# #
# #         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
# #
# #         successful_in_batch = False
# #
# #         for local_idx, out in enumerate(batch_raw):
# #             m = batch[local_idx]
# #             stage_index = start_idx + batch_start + local_idx
# #
# #             if isinstance(out, Exception):
# #                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
# #                 continue
# #
# #             res = out
# #             if not res.get("success", False):
# #                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
# #                 continue
# #
# #             conf = float(res.get("probability", 0.0))
# #             thr = _get_stage_threshold(stage_index)
# #             passed = conf >= thr
# #
# #             logger.info(
# #                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
# #                 f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
# #                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
# #             )
# #             logger.info(
# #                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
# #                 f"threshold={thr:.2f} pass={passed}"
# #             )
# #
# #             results.append(res)
# #             successful_in_batch = True
# #
# #         if successful_in_batch and len(results) >= 2:
# #             last_stage_idx = start_idx + batch_start + len(batch) - 1
# #             agg_thr = _get_stage_threshold(last_stage_idx)
# #
# #             temp_final = combine_outputs(
# #                 results,
# #                 policy="weighted_majority",
# #                 historical_acc=OMEGA,
# #                 model_weights=MODEL_WEIGHTS,
# #                 ground_truth=selected_folder,
# #                 gamma=GAMMA,
# #                 update_weights=False,
# #                 label_matcher=wnid_matches_text_label,
# #                 rho=RHO,
# #             )
# #
# #             agg_conf = float(temp_final.get("accuracy", 0.0))
# #             agg_pass = agg_conf >= agg_thr
# #
# #             logger.info(
# #                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
# #                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
# #                 f"threshold={agg_thr:.2f} pass={agg_pass}"
# #             )
# #
# #             if agg_pass:
# #                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
# #                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
# #                 break
# #
# #     return results, stop_reason
#
# # -----------------------------
# # Parallel-first-finish cascade
# # -----------------------------
# async def execute_parallel_first_finish_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64)): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 first_thr = _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')} "
#                     f"server={first_result.get('server_address')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} first_finished_model={first_model} "
#                     f"conf={first_conf:.4f} threshold={first_thr:.2f} pass={first_pass}"
#                 )
#             else:
#                 logger.warning(
#                     f"[RUN_BAD] tx={transaction_id} first_finished_model={first_model} success=False"
#                 )
#
#         except asyncio.CancelledError:
#             logger.warning(
#                 f"[RUN_CANCELLED] tx={transaction_id} first_finished_model={first_model}"
#             )
#             first_result = None
#
#         except Exception as e:
#             logger.exception(
#                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
#             )
#             first_result = None
#
#         # Early exit if first finished model already passes threshold
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception as e:
#                     logger.exception(
#                         f"[CANCEL_WAIT_FAIL] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')} error={e}"
#                     )
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         # Otherwise, wait for remaining first-batch tasks and aggregate
#         remaining_results: List[dict] = []
#
#         for p in pending:
#             model_name = task_map[p]
#             try:
#                 res = await p
#                 if res.get("success", False):
#                     remaining_results.append(res)
#
#                     logger.info(
#                         f"[RUN_OK] tx={transaction_id} model={model_name} "
#                         f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#                         f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                         f"container_state={res.get('container_state')} "
#                         f"server={res.get('server_address')}"
#                     )
#                 else:
#                     logger.warning(
#                         f"[RUN_BAD] tx={transaction_id} model={model_name} success=False"
#                     )
#             except asyncio.CancelledError:
#                 logger.warning(
#                     f"[RUN_CANCELLED] tx={transaction_id} model={model_name}"
#                 )
#             except Exception as e:
#                 logger.exception(
#                     f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}"
#                 )
#
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#
#         results.extend(remaining_results)
#
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_thr = _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = (
#                     f"early_exit_aggregate_pass_after_first_batch="
#                     f"{[r['model'] for r in results]}"
#                 )
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#         try:
#             stub = get_stub_for_model(m, stub_map)
#             r = await send_request(stub, m, image_b64)
#
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#                 results.append(r)
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m} "
#                     f"label={r.get('label')} conf={conf:.4f} "
#                     f"rpc_latency_s={float(r.get('latency', -1)):.4f} "
#                     f"container_state={r.get('container_state')} "
#                     f"server={r.get('server_address')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} model={m} "
#                     f"conf={conf:.4f} threshold={thr:.2f} pass={passed}"
#                 )
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                     return results, stop_reason
#             else:
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#
#         except asyncio.CancelledError:
#             logger.warning(f"[RUN_CANCELLED] tx={transaction_id} model={m}")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         first_next_model = batch[0]
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         logger.info(f"[PARALLEL_BATCH] tx={transaction_id} batch={batch}")
#
#         tasks = [
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64))
#             for m in batch
#         ]
#
#         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
#
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, asyncio.CancelledError):
#                 logger.warning(f"[RUN_CANCELLED] tx={transaction_id} model={m}")
#                 continue
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#             if not res.get("success", False):
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
#                 f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
#             )
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Main invoke
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ) -> dict:
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} parallel_first_n={PARALLEL_FIRST_N} "
#         f"parallel_batch_size={PARALLEL_BATCH_SIZE} "
#         f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES}"
#     )
#
#     wait_results = get_wait_results_for_models(M_TOTAL)
#
#     all_wait_vals = [float(v) for v in wait_results.values() if v is not None and v != float("inf")]
#     if all_wait_vals:
#         logger.info(
#             f"[WAIT_SUMMARY] tx={transaction_id} mean_wait={sum(all_wait_vals)/len(all_wait_vals):.4f}s "
#             f"max_wait={max(all_wait_vals):.4f}s min_wait={min(all_wait_vals):.4f}s"
#         )
#     else:
#         logger.info(f"[WAIT_SUMMARY] tx={transaction_id} no_wait_data=True")
#
#     M_D, total_estimates = build_model_set_by_mode(
#         cascade_mode=CASCADE_MODE,
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha_now,
#         parallel_first_n=PARALLEL_FIRST_N,
#         network_latency=NETWORK_LATENCY,
#         logger=logger,
#     )
#
#     if not M_D:
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         M_D = [fastest_model]
#         total_estimates = {fastest_model: wait_results.get(fastest_model, float("inf"))}
#         logger.warning(f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}")
#
#     selected_est_latency = estimate_selected_set_latency_for_mode(
#         cascade_mode=CASCADE_MODE,
#         selected_models=M_D,
#         wait_results=wait_results,
#         network_latency=NETWORK_LATENCY,
#         parallel_first_n=PARALLEL_FIRST_N,
#     )
#
#     deadline_sec = float(deadline) / 1000.0 if deadline is not None else 0.0
#     deadline_fast_path_triggered = False
#
#     if (
#         USE_DEADLINE_FAST_PATH
#         and deadline_sec > 0.0
#         and selected_est_latency >= deadline_sec
#     ):
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         logger.warning(
#             f"[DEADLINE_FAST_PATH] tx={transaction_id} "
#             f"mode={CASCADE_MODE} selected_models={M_D} "
#             f"selected_est_latency={selected_est_latency:.4f}s "
#             f"deadline={deadline_sec:.4f}s "
#             f"fallback_fastest={fastest_model}"
#         )
#         M_D = [fastest_model]
#         total_estimates = {fastest_model: wait_results.get(fastest_model, float('inf'))}
#         selected_est_latency = estimate_selected_set_latency_for_mode(
#             cascade_mode=CASCADE_MODE,
#             selected_models=M_D,
#             wait_results=wait_results,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#         )
#         deadline_fast_path_triggered = True
#
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#     logger.info(
#         f"[SELECT_WAITS] tx={transaction_id} waits_selected="
#         f"{ {m: float(total_estimates.get(m, -1)) for m in M_D} }"
#     )
#     logger.info(
#         f"[SELECT_ROUTE] tx={transaction_id} routes="
#         f"{ {m: get_server_address_for_model(m) for m in M_D} }"
#     )
#
#     channel_map, stub_map = build_stub_map_for_selected_models(M_D)
#
#     try:
#         mode = str(CASCADE_MODE).lower()
#
#         if mode == "sequential":
#             results, stop_reason = await execute_sequential_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#             )
#         elif mode in ("mode_s", "full_parallel"):
#             results, stop_reason = await execute_mode_s_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#             )
#         else:
#             results, stop_reason = await execute_parallel_first_finish_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 parallel_first_n=PARALLEL_FIRST_N,
#                 parallel_batch_size=PARALLEL_BATCH_SIZE,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_folder=selected_folder,
#             )
#     finally:
#         await close_channels(channel_map)
#
#     if not results:
#         end_time = time.perf_counter()
#         e2e_ms = (end_time - start_time) * 1000
#         logger.error(f"[REQ_END] tx={transaction_id} success=False reason=no_successful_models e2e_ms={e2e_ms:.2f}")
#         return {
#             "selected_models": M_D,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": selected_folder,
#             "wait_times": total_estimates,
#             "per_model": {},
#             "ensemble_size": ENSEMBLE_SIZE,
#             "threshold": THRESHOLD,
#             "threshold_stage": THRESHOLD_STAGE,
#             "main_policy": POLICY,
#             "alpha": alpha_now,
#             "alpha_mode": ALPHA_MODE,
#             "qps_est": qps_est,
#             "stop_reason": stop_reason,
#             "log_file": LOG_PATH,
#             "cascade_mode": CASCADE_MODE,
#             "parallel_first_n": PARALLEL_FIRST_N,
#             "parallel_batch_size": PARALLEL_BATCH_SIZE,
#             "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#             "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#             "deadline_fast_path_triggered": deadline_fast_path_triggered,
#             "selected_est_latency_s": selected_est_latency,
#         }
#
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#         logger.info(
#             f"[SINGLE] tx={transaction_id} model={single.get('model')} "
#             f"label={final_result['label']} conf={final_result['accuracy']:.4f}"
#         )
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#         logger.info(
#             f"[COMBINE] tx={transaction_id} executed_models={[r['model'] for r in results]} "
#             f"combiner={final_result.get('combiner_policy')} final_label={final_result.get('label')} "
#             f"final_acc={final_result.get('accuracy')}"
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (1 - ETA) * OMEGA[res["model"]] + ETA * float(res.get("probability", 0.0))
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     per_model = {}
#     for res in results:
#         per_model[res["model"]] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#             "server_address": res.get("server_address"),
#         }
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} alpha={alpha_now:.3f} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#         "selected_folder": selected_folder,
#         "wait_times": total_estimates,
#         "per_model": per_model,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "stop_reason": stop_reason,
#         "log_file": LOG_PATH,
#         "cascade_mode": CASCADE_MODE,
#         "parallel_first_n": PARALLEL_FIRST_N,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#         "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#         "deadline_fast_path_triggered": deadline_fast_path_triggered,
#         "selected_est_latency_s": selected_est_latency,
#     }
#
#
# ###########
# #new model selection f  code end
# #####



# # ###########
# # # final working code  05-04-2026
# # #####
#
#
# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#     ALPHA_MODE,
#     ALPHA_STATIC,
#     ALPHA_MIN,
#     ALPHA_MAX,
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#     CASCADE_MODE,
#     PARALLEL_FIRST_N,
#     PARALLEL_BATCH_SIZE,
#     NETWORK_LATENCY,
#     USE_DEADLINE_FAST_PATH,
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
# from ckn_controller.ensemble_selector import (
#     build_model_set_by_mode,
#     estimate_selected_set_latency_for_mode,
# )
#
#
# # -----------------------------
# # Logging setup
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking
# # -----------------------------
# REQUEST_TS = deque()
#
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Recent latency overhead tracking
# # -----------------------------
# LATENCY_OVERHEAD_HISTORY = deque(maxlen=50)
# LATENCY_OVERHEAD_WEIGHT = 0.7
#
#
# def get_recent_latency_overhead() -> float:
#     if not LATENCY_OVERHEAD_HISTORY:
#         return 0.0
#
#     vals = [v for v in LATENCY_OVERHEAD_HISTORY if v > 0]
#     if not vals:
#         return 0.0
#
#     return sum(vals) / len(vals)
#
#
# def apply_latency_correction(estimated_latency: float) -> float:
#     if estimated_latency == float("inf"):
#         return estimated_latency
#
#     recent_overhead = get_recent_latency_overhead()
#     return estimated_latency + (LATENCY_OVERHEAD_WEIGHT * recent_overhead)
#
#
# def update_latency_overhead(actual_latency_s: float, estimated_latency_s: float) -> None:
#     if estimated_latency_s == float("inf"):
#         return
#
#     overhead = actual_latency_s - estimated_latency_s
#     LATENCY_OVERHEAD_HISTORY.append(overhead)
#
#
# # -----------------------------
# # Adaptive alpha
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     load_score = qps_norm
#     alpha_now = alpha_max - load_score * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Manual routing helpers
# # -----------------------------
# def get_server_address_for_model(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return LARGE_MODEL_SERVER_ADDRESS
#
#     raise ValueError(f"Model {model_name} is not assigned to any Iluvatar instance.")
#
#
# def build_stub_map_for_selected_models(model_list: List[str]):
#     addresses = {get_server_address_for_model(m) for m in model_list}
#     channel_map = {}
#     stub_map = {}
#
#     for addr in addresses:
#         channel = grpc.aio.insecure_channel(addr)
#         stub = pb2_grpc.IluvatarWorkerStub(channel)
#         channel_map[addr] = channel
#         stub_map[addr] = stub
#
#     return channel_map, stub_map
#
#
# async def close_channels(channel_map):
#     for channel in channel_map.values():
#         try:
#             await channel.close()
#         except Exception:
#             pass
#
#
# def get_stub_for_model(model_name: str, stub_map):
#     addr = get_server_address_for_model(model_name)
#     return stub_map[addr]
#
#
# def get_wait_results_for_models(model_list: List[str]) -> Dict[str, float]:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return wait_time_iluvatar.main(
#             server_address=SERVER_ADDRESS,
#             model_list=model_list,
#         )
#
#     small_models = [m for m in model_list if m in SMALL_INSTANCE_MODELS]
#     large_models = [m for m in model_list if m in LARGE_INSTANCE_MODELS]
#
#     merged = {}
#
#     if small_models:
#         small_waits = wait_time_iluvatar.main(
#             server_address=SMALL_MODEL_SERVER_ADDRESS,
#             model_list=small_models,
#         )
#         merged.update(small_waits)
#
#     if large_models:
#         large_waits = wait_time_iluvatar.main(
#             server_address=LARGE_MODEL_SERVER_ADDRESS,
#             model_list=large_models,
#         )
#         merged.update(large_waits)
#
#     return merged
#
#
# # -----------------------------
# # Iluvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#         "server_address": get_server_address_for_model(model_name),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#     return float(THRESHOLD)
#
#
# # -----------------------------
# # Sequential cascade execution
# # -----------------------------
# async def execute_sequential_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#         logger.info(f"[RUN] tx={transaction_id} step={i+1}/{len(model_list)} model={m} start=True")
#
#         stub = get_stub_for_model(m, stub_map)
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}")
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#         stage_thr = _get_stage_threshold(i)
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.2f} pass={passed}"
#         )
#
#         results.append(res)
#
#         if i == 0 and passed:
#             stop_reason = f"early_exit_first_pass={m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i > 0 and passed:
#             stop_reason = f"early_exit_with_aggregation_at_{m}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} next_model={next_model} "
#                 f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # MODE-S execution
# # -----------------------------
# async def execute_mode_s_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
# ) -> Tuple[List[dict], str]:
#     logger.info(f"[MODE_S_START] tx={transaction_id} models={model_list}")
#
#     tasks = []
#     task_models = []
#
#     for m in model_list:
#         stub = get_stub_for_model(m, stub_map)
#         tasks.append(asyncio.create_task(send_request(stub, m, image_b64)))
#         task_models.append(m)
#
#     raw_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     results: List[dict] = []
#
#     for idx, out in enumerate(raw_results):
#         m = task_models[idx]
#
#         if isinstance(out, Exception):
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#             continue
#
#         res = out
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#             continue
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} "
#             f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#         results.append(res)
#
#     return results, "mode_s_executed"
#
#
# # -----------------------------
# # Parallel-first-finish cascade
# # -----------------------------
# async def execute_parallel_first_finish_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64)): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 first_thr = _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')} "
#                     f"server={first_result.get('server_address')}"
#                 )
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} first_finished_model={first_model} "
#                     f"conf={first_conf:.4f} threshold={first_thr:.2f} pass={first_pass}"
#                 )
#
#         except asyncio.CancelledError:
#             logger.warning(f"[RUN_CANCELLED] tx={transaction_id} first_finished_model={first_model}")
#             first_result = None
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}")
#             first_result = None
#
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception as e:
#                     logger.exception(
#                         f"[CANCEL_WAIT_FAIL] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')} error={e}"
#                     )
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         remaining_results: List[dict] = []
#
#         for p in pending:
#             model_name = task_map[p]
#             try:
#                 res = await p
#                 if res.get("success", False):
#                     remaining_results.append(res)
#
#                     logger.info(
#                         f"[RUN_OK] tx={transaction_id} model={model_name} "
#                         f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#                         f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                         f"container_state={res.get('container_state')} "
#                         f"server={res.get('server_address')}"
#                     )
#             except asyncio.CancelledError:
#                 logger.warning(f"[RUN_CANCELLED] tx={transaction_id} model={model_name}")
#             except Exception as e:
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#
#         results.extend(remaining_results)
#
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_thr = _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = (
#                     f"early_exit_aggregate_pass_after_first_batch="
#                     f"{[r['model'] for r in results]}"
#                 )
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#         try:
#             stub = get_stub_for_model(m, stub_map)
#             r = await send_request(stub, m, image_b64)
#
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#                 results.append(r)
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} model={m} "
#                     f"label={r.get('label')} conf={conf:.4f} "
#                     f"rpc_latency_s={float(r.get('latency', -1)):.4f} "
#                     f"container_state={r.get('container_state')} "
#                     f"server={r.get('server_address')}"
#                 )
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                     return results, stop_reason
#
#         except asyncio.CancelledError:
#             logger.warning(f"[RUN_CANCELLED] tx={transaction_id} model={m}")
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         first_next_model = batch[0]
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         logger.info(f"[PARALLEL_BATCH] tx={transaction_id} batch={batch}")
#
#         tasks = [
#             asyncio.create_task(send_request(get_stub_for_model(m, stub_map), m, image_b64))
#             for m in batch
#         ]
#
#         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, asyncio.CancelledError):
#                 logger.warning(f"[RUN_CANCELLED] tx={transaction_id} model={m}")
#                 continue
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#             if not res.get("success", False):
#                 logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} conf={conf:.4f} "
#                 f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
#             )
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Main invoke
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ) -> dict:
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} parallel_first_n={PARALLEL_FIRST_N} "
#         f"parallel_batch_size={PARALLEL_BATCH_SIZE} "
#         f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES}"
#     )
#
#     wait_results = get_wait_results_for_models(M_TOTAL)
#
#     all_wait_vals = [
#         float(v) for v in wait_results.values()
#         if v is not None and v != float("inf")
#     ]
#
#     if all_wait_vals:
#         logger.info(
#             f"[WAIT_SUMMARY] tx={transaction_id} "
#             f"mean_wait={sum(all_wait_vals)/len(all_wait_vals):.4f}s "
#             f"max_wait={max(all_wait_vals):.4f}s "
#             f"min_wait={min(all_wait_vals):.4f}s"
#         )
#     else:
#         logger.info(f"[WAIT_SUMMARY] tx={transaction_id} no_wait_data=True")
#
#     M_D, total_estimates = build_model_set_by_mode(
#         cascade_mode=CASCADE_MODE,
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha_now,
#         parallel_first_n=PARALLEL_FIRST_N,
#         network_latency=NETWORK_LATENCY,
#         logger=logger,
#     )
#
#     if not M_D:
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         M_D = [fastest_model]
#         total_estimates = {fastest_model: wait_results.get(fastest_model, float("inf"))}
#         logger.warning(f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}")
#
#     raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#         cascade_mode=CASCADE_MODE,
#         selected_models=M_D,
#         wait_results=wait_results,
#         network_latency=NETWORK_LATENCY,
#         parallel_first_n=PARALLEL_FIRST_N,
#     )
#
#     selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#
#     logger.info(
#         f"[LATENCY_EST] tx={transaction_id} "
#         f"raw_selected_est_latency={raw_selected_est_latency:.6f}s "
#         f"corrected_selected_est_latency={selected_est_latency:.6f}s "
#         f"recent_overhead={get_recent_latency_overhead():.6f}s"
#     )
#
#     deadline_sec = float(deadline) / 1000.0 if deadline is not None else 0.0
#     deadline_fast_path_triggered = False
#
#     if (
#         USE_DEADLINE_FAST_PATH
#         and deadline_sec > 0.0
#         and selected_est_latency >= deadline_sec
#     ):
#         fastest_model = min(MODEL_PROFILES, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         logger.warning(
#             f"[DEADLINE_FAST_PATH] tx={transaction_id} "
#             f"mode={CASCADE_MODE} selected_models={M_D} "
#             f"selected_est_latency={selected_est_latency:.4f}s "
#             f"deadline={deadline_sec:.4f}s "
#             f"fallback_fastest={fastest_model}"
#         )
#
#         M_D = [fastest_model]
#         total_estimates = {fastest_model: wait_results.get(fastest_model, float("inf"))}
#
#         raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#             cascade_mode=CASCADE_MODE,
#             selected_models=M_D,
#             wait_results=wait_results,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#         )
#         selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#         deadline_fast_path_triggered = True
#
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#     logger.info(
#         f"[SELECT_WAITS] tx={transaction_id} waits_selected="
#         f"{ {m: float(total_estimates.get(m, -1)) for m in M_D} }"
#     )
#     logger.info(
#         f"[SELECT_ROUTE] tx={transaction_id} routes="
#         f"{ {m: get_server_address_for_model(m) for m in M_D} }"
#     )
#
#     channel_map, stub_map = build_stub_map_for_selected_models(M_D)
#
#     try:
#         mode = str(CASCADE_MODE).lower()
#
#         if mode == "sequential":
#             results, stop_reason = await execute_sequential_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#             )
#         elif mode in ("mode_s", "full_parallel"):
#             results, stop_reason = await execute_mode_s_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#             )
#         else:
#             results, stop_reason = await execute_parallel_first_finish_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 parallel_first_n=PARALLEL_FIRST_N,
#                 parallel_batch_size=PARALLEL_BATCH_SIZE,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_folder=selected_folder,
#             )
#     finally:
#         await close_channels(channel_map)
#
#     if not results:
#         end_time = time.perf_counter()
#         e2e_ms = (end_time - start_time) * 1000
#         actual_latency_s = e2e_ms / 1000.0
#
#         update_latency_overhead(
#             actual_latency_s=actual_latency_s,
#             estimated_latency_s=raw_selected_est_latency,
#         )
#
#         logger.error(
#             f"[REQ_END] tx={transaction_id} success=False reason=no_successful_models "
#             f"e2e_ms={e2e_ms:.2f}"
#         )
#
#         return {
#             "selected_models": M_D,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#             "selected_folder": selected_folder,
#             "wait_times": total_estimates,
#             "per_model": {},
#             "ensemble_size": ENSEMBLE_SIZE,
#             "threshold": THRESHOLD,
#             "threshold_stage": THRESHOLD_STAGE,
#             "main_policy": POLICY,
#             "alpha": alpha_now,
#             "alpha_mode": ALPHA_MODE,
#             "qps_est": qps_est,
#             "stop_reason": stop_reason,
#             "log_file": LOG_PATH,
#             "cascade_mode": CASCADE_MODE,
#             "parallel_first_n": PARALLEL_FIRST_N,
#             "parallel_batch_size": PARALLEL_BATCH_SIZE,
#             "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#             "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#             "deadline_fast_path_triggered": deadline_fast_path_triggered,
#             "selected_est_latency_s": selected_est_latency,
#             "raw_selected_est_latency_s": raw_selected_est_latency,
#             "recent_latency_overhead_s": get_recent_latency_overhead(),
#         }
#
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (
#                 (1 - ETA) * OMEGA[res["model"]]
#                 + ETA * float(res.get("probability", 0.0))
#             )
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     per_model = {}
#     for res in results:
#         per_model[res["model"]] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#             "server_address": res.get("server_address"),
#         }
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000
#     actual_latency_s = e2e_ms / 1000.0
#
#     update_latency_overhead(
#         actual_latency_s=actual_latency_s,
#         estimated_latency_s=raw_selected_est_latency,
#     )
#
#     logger.info(
#         f"[LATENCY_OVERHEAD_UPDATE] tx={transaction_id} "
#         f"actual_latency={actual_latency_s:.6f}s "
#         f"raw_est_latency={raw_selected_est_latency:.6f}s "
#         f"recent_overhead={get_recent_latency_overhead():.6f}s"
#     )
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} alpha={alpha_now:.3f} "
#         f"cascade_mode={CASCADE_MODE}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#         "selected_folder": selected_folder,
#         "wait_times": total_estimates,
#         "per_model": per_model,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "stop_reason": stop_reason,
#         "log_file": LOG_PATH,
#         "cascade_mode": CASCADE_MODE,
#         "parallel_first_n": PARALLEL_FIRST_N,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#         "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#         "deadline_fast_path_triggered": deadline_fast_path_triggered,
#         "selected_est_latency_s": selected_est_latency,
#         "raw_selected_est_latency_s": raw_selected_est_latency,
#         "recent_latency_overhead_s": get_recent_latency_overhead(),
#     }
#
#
#
# # ###########
# # # final working code  05-04-2026 end
# # #####



# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#     ALPHA_MODE,
#     ALPHA_STATIC,
#     ALPHA_MIN,
#     ALPHA_MAX,
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#     CASCADE_MODE,
#     PARALLEL_FIRST_N,
#     PARALLEL_BATCH_SIZE,
#     NETWORK_LATENCY,
#     USE_DEADLINE_FAST_PATH,
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
#     BASELINE_MODE,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
# from ckn_controller.ensemble_selector import (
#     build_model_set_by_mode,
#     estimate_selected_set_latency_for_mode,
# )
#
# # -----------------------------
# # Logging
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
#
#     fh = RotatingFileHandler(LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5)
#     fh.setFormatter(fmt)
#     logger.addHandler(fh)
#
#     ch = logging.StreamHandler()
#     ch.setFormatter(fmt)
#     logger.addHandler(ch)
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# REQUEST_TS = deque()
# LATENCY_OVERHEAD_HISTORY = deque(maxlen=50)
#
# # -----------------------------
# # QPS
# # -----------------------------
# def record_request(now: float):
#     REQUEST_TS.append(now)
#     cutoff = now - QPS_WINDOW_SEC
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
# def get_qps():
#     return len(REQUEST_TS) / max(QPS_WINDOW_SEC, 1e-9)
#
# # -----------------------------
# # Latency correction
# # -----------------------------
# def get_recent_latency_overhead():
#     if not LATENCY_OVERHEAD_HISTORY:
#         return 0.0
#     return sum(LATENCY_OVERHEAD_HISTORY) / len(LATENCY_OVERHEAD_HISTORY)
#
# def apply_latency_correction(est):
#     return est + 0.7 * get_recent_latency_overhead()
#
# def update_latency_overhead(actual, est):
#     LATENCY_OVERHEAD_HISTORY.append(actual - est)
#
# # -----------------------------
# # Adaptive alpha
# # -----------------------------
# def compute_alpha(qps):
#     if ALPHA_MODE == "static":
#         return ALPHA_STATIC
#
#     norm = max(0, min(1, (qps - QPS_LOW) / (QPS_HIGH - QPS_LOW)))
#     return ALPHA_MAX - norm * (ALPHA_MAX - ALPHA_MIN)
#
# # -----------------------------
# # Routing
# # -----------------------------
# def get_server(model):
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#     if model in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#     return LARGE_MODEL_SERVER_ADDRESS
#
# def build_stubs(models):
#     channels = {}
#     stubs = {}
#     for m in models:
#         addr = get_server(m)
#         if addr not in channels:
#             ch = grpc.aio.insecure_channel(addr)
#             channels[addr] = ch
#             stubs[addr] = pb2_grpc.IluvatarWorkerStub(ch)
#     return channels, stubs
#
# def get_stub(model, stub_map):
#     return stub_map[get_server(model)]
#
# # -----------------------------
# # Inference
# # -----------------------------
# async def send_request(stub, model, image):
#     req = pb2.InvokeRequest(
#         function_name=model,
#         function_version="1",
#         json_args=json.dumps({"model_name": model, "image_data": image}),
#         transaction_id=str(uuid.uuid4()),
#     )
#     resp = await stub.invoke(req)
#     res = json.loads(resp.json_result)
#
#     return {
#         "model": model,
#         "label": res["body"]["Prediction Class"],
#         "probability": float(res["body"]["Probability"]),
#         "latency": resp.duration_us / 1e6,
#         "success": resp.success,
#     }
#
# # -----------------------------
# # MAIN INVOKE
# # -----------------------------
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None
# ):
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now)
#     qps = get_qps()
#     alpha = compute_alpha(qps)
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} qps={qps:.2f} "
#         f"alpha={alpha:.3f} baseline={BASELINE_MODE}"
#     )
#
#     # -----------------------------
#     # Wait times
#     # -----------------------------
#     wait_results = wait_time_iluvatar.main(
#         server_address=SERVER_ADDRESS,
#         model_list=M_TOTAL,
#     )
#
#     # -----------------------------
#     # MODEL SELECTION (UPDATED)
#     # -----------------------------
#     M_D, total_estimates, effective_mode = build_model_set_by_mode(
#         cascade_mode=CASCADE_MODE,
#         wait_results=wait_results,
#         deadline_ms=deadline,
#         ensemble_size=ENSEMBLE_SIZE,
#         alpha_now=alpha,
#         parallel_first_n=PARALLEL_FIRST_N,
#         network_latency=NETWORK_LATENCY,
#         logger=logger,
#     )
#
#     if not M_D:
#         fastest = min(MODEL_PROFILES, key=lambda m: MODEL_PROFILES[m]["latency"])
#         M_D = [fastest]
#         effective_mode = "sequential"
#
#     logger.info(
#         f"[SELECT] models={M_D} mode={effective_mode}"
#     )
#
#     # -----------------------------
#     # LATENCY ESTIMATION
#     # -----------------------------
#     raw_est = estimate_selected_set_latency_for_mode(
#         cascade_mode=effective_mode,
#         selected_models=M_D,
#         wait_results=wait_results,
#         network_latency=NETWORK_LATENCY,
#         parallel_first_n=PARALLEL_FIRST_N,
#     )
#
#     est = apply_latency_correction(raw_est)
#
#     logger.info(f"[EST] raw={raw_est:.4f} corrected={est:.4f}")
#
#     # -----------------------------
#     # EXECUTION
#     # -----------------------------
#     channels, stubs = build_stubs(M_D)
#
#     results = []
#     stop_reason = "ran_all"
#
#     try:
#         mode = effective_mode.lower()
#
#         # ---------- Sequential ----------
#         if mode == "sequential":
#             for m in M_D:
#                 stub = get_stub(m, stubs)
#                 res = await send_request(stub, m, image_b64)
#                 if res["success"]:
#                     results.append(res)
#
#                     if res["probability"] >= THRESHOLD:
#                         stop_reason = f"early_exit_{m}"
#                         break
#
#         # ---------- Full Parallel ----------
#         elif mode in ("mode_s", "full_parallel"):
#             tasks = [
#                 asyncio.create_task(send_request(get_stub(m, stubs), m, image_b64))
#                 for m in M_D
#             ]
#             out = await asyncio.gather(*tasks)
#
#             for r in out:
#                 if r["success"]:
#                     results.append(r)
#
#         # ---------- Power2 ----------
#         else:
#             first_batch = M_D[:PARALLEL_FIRST_N]
#             rest = M_D[PARALLEL_FIRST_N:]
#
#             tasks = [
#                 asyncio.create_task(send_request(get_stub(m, stubs), m, image_b64))
#                 for m in first_batch
#             ]
#
#             done, pending = await asyncio.wait(
#                 tasks, return_when=asyncio.FIRST_COMPLETED
#             )
#
#             first = list(done)[0].result()
#
#             if first["success"] and first["probability"] >= THRESHOLD:
#                 results = [first]
#                 stop_reason = "early_exit_first"
#             else:
#                 for t in pending:
#                     results.append(await t)
#
#                 for m in rest:
#                     stub = get_stub(m, stubs)
#                     r = await send_request(stub, m, image_b64)
#                     if r["success"]:
#                         results.append(r)
#
#     finally:
#         for ch in channels.values():
#             await ch.close()
#
#     # -----------------------------
#     # AGGREGATION
#     # -----------------------------
#     if not results:
#         return {
#             "success": False,
#             "selected_models": M_D,
#             "baseline_mode": BASELINE_MODE,
#             "effective_mode": effective_mode,
#         }
#
#     if len(results) == 1:
#         final = {
#             "label": results[0]["label"],
#             "accuracy": results[0]["probability"],
#             "combiner_policy": "single"
#         }
#     else:
#         final = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#
#     # -----------------------------
#     # END
#     # -----------------------------
#     end_time = time.perf_counter()
#     e2e = (end_time - start_time) * 1000
#
#     update_latency_overhead(e2e / 1000.0, raw_est)
#
#     logger.info(
#         f"[END] tx={transaction_id} time={e2e:.2f} "
#         f"mode={effective_mode} stop={stop_reason}"
#     )
#
#     return {
#         "selected_models": M_D,
#         "executed_models": [r["model"] for r in results],
#         "label": final["label"],
#         "accuracy": final["accuracy"],
#         "e2e_time_ms": e2e,
#         "success": True,
#         "baseline_mode": BASELINE_MODE,
#         "configured_mode": CASCADE_MODE,
#         "effective_mode": effective_mode,
#         "stop_reason": stop_reason,
#     }



# import grpc
# import json
# import uuid
# import time
# import asyncio
# import logging
# import os
# from collections import deque
# from typing import Dict, List, Tuple, Optional
# from logging.handlers import RotatingFileHandler
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     POLICY,
#     ETA,
#     MODEL_PROFILES,
#     OMEGA,
#     DEFAULT_WEIGHTS,
#     GAMMA,
#     RHO,
#     WEIGHTS_STATE_PATH,
#     ENSEMBLE_SIZE,
#     THRESHOLD,
#     THRESHOLD_STAGE,
#     ALPHA_MODE,
#     ALPHA_STATIC,
#     ALPHA_MIN,
#     ALPHA_MAX,
#     QPS_WINDOW_SEC,
#     QPS_LOW,
#     QPS_HIGH,
#     CASCADE_MODE,
#     PARALLEL_FIRST_N,
#     PARALLEL_BATCH_SIZE,
#     NETWORK_LATENCY,
#     USE_DEADLINE_FAST_PATH,
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
#     BASELINE_MODE,
#     ENABLE_RISK_STRATEGY_SELECTION,
#     RISK_ALPHA_LATENCY,
#     RISK_BETA_COST,
#     RISK_GAMMA_ERROR,
#     HARDWARE_COST_RATE,
#     # USE_SEQ_CONSTRAINT_SOLVER,
#     # SEQ_CASCADE_PROFILE_CSV,
#     # SEQ_MAX_COST,
#     # SEQ_TARGET_ACCURACY,
#     USE_CONSTRAINT_SOLVER,
#     CONSTRAINT_SOLVER_TIME_MODE,
#     CONSTRAINT_MAX_COST,
#     CONSTRAINT_TARGET_ACCURACY,
#     CONSTRAINT_POLICY_ORDER,
#     CONSTRAINT_SEQ_PROFILE_CSV,
#     CONSTRAINT_POWER2_PROFILE_CSV,
#     CONSTRAINT_FULL_PARALLEL_PROFILE_CSV,
#     BASE_REQUEST_OVERHEAD,
#     SEQUENTIAL_STAGE_OVERHEAD,
#     PO2_FIRST_BATCH_OVERHEAD,
#     PO2_TAIL_OVERHEAD,
#     FULL_PARALLEL_AGG_OVERHEAD,
#     ET_SCALE_SEQ,
#     ET_OFFSET_SEQ,
#     ET_SCALE_P2,
#     ET_OFFSET_P2,
#     USE_STATIC_OPTIMAL_CONFIG,
#     STATIC_COST,
#     STATIC_ERROR,
#     STATIC_POLICY,
# )
#
# import wait_time_iluvatar
# from ckn_controller.output_combiner import combine_outputs
# from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
# from ckn_controller.label_utils import wnid_matches_text_label
# from ckn_controller.ensemble_selector import (
#     build_model_set_by_mode,
#     estimate_selected_set_latency_for_mode,
# )
# from ckn_controller.risk_selector import select_min_risk_strategy
# from ckn_controller.sequential_cascade_constraint_solver import (
#     SequentialCascadeConstraintSolver,
# )
# from ckn_controller.constraint_solver import ConstraintSolver
# from ckn_controller.optimal_configs import get_optimal_config
#
# # -----------------------------
# # Logging setup
# # -----------------------------
# LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
# os.makedirs(LOG_DIR, exist_ok=True)
# LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")
#
# logger = logging.getLogger("cascade")
# logger.setLevel(logging.INFO)
#
# if not logger.handlers:
#     fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
#
#     file_handler = RotatingFileHandler(
#         LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
#     )
#     file_handler.setFormatter(fmt)
#     logger.addHandler(file_handler)
#
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(fmt)
#     logger.addHandler(console_handler)
#
#
# MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
# active_models = set()
#
#
# # -----------------------------
# # QPS tracking
# # -----------------------------
# REQUEST_TS = deque()
#
#
# def record_request(now: float, window_sec: float) -> None:
#     REQUEST_TS.append(now)
#     cutoff = now - window_sec
#
#     while REQUEST_TS and REQUEST_TS[0] < cutoff:
#         REQUEST_TS.popleft()
#
#
# def get_qps(window_sec: float) -> float:
#     return len(REQUEST_TS) / max(window_sec, 1e-9)
#
#
# # -----------------------------
# # Recent latency overhead tracking
# # -----------------------------
# LATENCY_OVERHEAD_HISTORY = deque(maxlen=50)
# LATENCY_OVERHEAD_WEIGHT = 0.7
#
#
# def get_recent_latency_overhead() -> float:
#     if not LATENCY_OVERHEAD_HISTORY:
#         return 0.0
#
#     vals = [v for v in LATENCY_OVERHEAD_HISTORY if v > 0]
#     if not vals:
#         return 0.0
#
#     return sum(vals) / len(vals)
#
#
# def apply_latency_correction(estimated_latency: float) -> float:
#     if estimated_latency == float("inf"):
#         return estimated_latency
#
#     recent_overhead = get_recent_latency_overhead()
#     return estimated_latency + (LATENCY_OVERHEAD_WEIGHT * recent_overhead)
#
#
# def update_latency_overhead(actual_latency_s: float, estimated_latency_s: float) -> None:
#     if estimated_latency_s == float("inf"):
#         return
#
#     overhead = actual_latency_s - estimated_latency_s
#     LATENCY_OVERHEAD_HISTORY.append(overhead)
#
#
# # -----------------------------
# # Adaptive alpha
# # -----------------------------
# def _clamp01(x: float) -> float:
#     return max(0.0, min(1.0, x))
#
#
# def compute_adaptive_alpha(
#     qps: float,
#     alpha_min: float,
#     alpha_max: float,
#     qps_low: float,
#     qps_high: float,
# ) -> float:
#     if qps_high <= qps_low:
#         qps_norm = 0.0
#     else:
#         qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))
#
#     alpha_now = alpha_max - qps_norm * (alpha_max - alpha_min)
#     alpha_now = max(alpha_min, min(alpha_max, alpha_now))
#
#     logger.info(
#         f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
#     )
#
#     return alpha_now
#
#
# # -----------------------------
# # Deadline helper
# # -----------------------------
# def should_stop_before_next_model(
#     start_time: float,
#     deadline_ms: int,
#     next_model: str,
#     safety_margin_ms: float = 0.0,
# ) -> Tuple[bool, float, float]:
#     elapsed_ms = (time.perf_counter() - start_time) * 1000.0
#     remaining_ms = float(deadline_ms) - elapsed_ms
#     next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0
#
#     stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)
#
#     return stop_now, elapsed_ms, remaining_ms
#
#
# # -----------------------------
# # Manual routing helpers
# # -----------------------------
# def get_server_address_for_model(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return LARGE_MODEL_SERVER_ADDRESS
#
#     raise ValueError(f"Model {model_name} is not assigned to any Iluvatar instance.")
#
#
# def build_stub_map_for_selected_models(model_list: List[str]):
#     addresses = {get_server_address_for_model(m) for m in model_list}
#
#     channel_map = {}
#     stub_map = {}
#
#     for addr in addresses:
#         channel = grpc.aio.insecure_channel(addr)
#         stub = pb2_grpc.IluvatarWorkerStub(channel)
#
#         channel_map[addr] = channel
#         stub_map[addr] = stub
#
#     return channel_map, stub_map
#
#
# async def close_channels(channel_map):
#     for channel in channel_map.values():
#         try:
#             await channel.close()
#         except Exception:
#             pass
#
#
# def get_stub_for_model(model_name: str, stub_map):
#     addr = get_server_address_for_model(model_name)
#     return stub_map[addr]
#
#
# def get_wait_results_for_models(model_list: List[str]) -> Dict[str, float]:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return wait_time_iluvatar.main(
#             server_address=SERVER_ADDRESS,
#             model_list=model_list,
#         )
#
#     small_models = [m for m in model_list if m in SMALL_INSTANCE_MODELS]
#     large_models = [m for m in model_list if m in LARGE_INSTANCE_MODELS]
#
#     merged = {}
#
#     if small_models:
#         small_waits = wait_time_iluvatar.main(
#             server_address=SMALL_MODEL_SERVER_ADDRESS,
#             model_list=small_models,
#         )
#         merged.update(small_waits)
#
#     if large_models:
#         large_waits = wait_time_iluvatar.main(
#             server_address=LARGE_MODEL_SERVER_ADDRESS,
#             model_list=large_models,
#         )
#         merged.update(large_waits)
#
#     return merged
#
#
# # -----------------------------
# # Iluvatar invocation
# # -----------------------------
# async def send_request(stub, model_name: str, image_b64: str) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps(
#             {
#                 "model_name": model_name,
#                 "image_data": image_b64,
#             }
#         ),
#         transaction_id=str(uuid.uuid4()),
#     )
#
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#         "server_address": get_server_address_for_model(model_name),
#     }
#
#
# def _get_stage_threshold(stage_index: int) -> float:
#     try:
#         if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
#             return float(THRESHOLD_STAGE[stage_index])
#     except Exception:
#         pass
#
#     return float(THRESHOLD)
#
#
#
# async def execute_sequential_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     start_time: float,
#     deadline_ms: int,
#     selected_theta: Optional[float] = None,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     for i, m in enumerate(model_list):
#         t0 = time.perf_counter()
#
#         logger.info(
#             f"[RUN] tx={transaction_id} step={i + 1}/{len(model_list)} model={m} start=True"
#         )
#
#         stub = get_stub_for_model(m, stub_map)
#
#         try:
#             res = await send_request(stub, m, image_b64)
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#             continue
#
#         wall_s = time.perf_counter() - t0
#
#         if not res.get("success", False):
#             logger.warning(
#                 f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}"
#             )
#             continue
#
#         conf = float(res.get("probability", 0.0))
#         label = res.get("label")
#
#         # Use solver-selected theta if available.
#         # Otherwise fall back to existing stage threshold logic.
#         if selected_theta is not None:
#             stage_thr = float(selected_theta)
#         else:
#             stage_thr = _get_stage_threshold(i)
#
#         passed = conf >= stage_thr
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#
#         logger.info(
#             f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#             f"threshold={stage_thr:.4f} pass={passed} "
#             f"source={'seq_constraint_solver' if selected_theta is not None else 'config'}"
#         )
#
#         results.append(res)
#
#         if passed:
#             if i == 0:
#                 stop_reason = f"early_exit_first_pass={m}"
#             else:
#                 stop_reason = f"early_exit_with_aggregation_at_{m}"
#
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         if i + 1 < len(model_list):
#             next_model = model_list[i + 1]
#
#             stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#                 start_time=start_time,
#                 deadline_ms=deadline_ms,
#                 next_model=next_model,
#             )
#
#             logger.info(
#                 f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} "
#                 f"next_model={next_model} elapsed_ms={elapsed_ms:.2f} "
#                 f"remaining_ms={remaining_ms:.2f} "
#                 f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
#                 f"stop={stop_now}"
#             )
#
#             if stop_now:
#                 stop_reason = f"deadline_stop_before_{next_model}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
#
# # -----------------------------
# # Full parallel / MODE-S execution
# # -----------------------------
# async def execute_mode_s_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
# ) -> Tuple[List[dict], str]:
#     logger.info(f"[MODE_S_START] tx={transaction_id} models={model_list}")
#
#     tasks = []
#     task_models = []
#
#     for m in model_list:
#         stub = get_stub_for_model(m, stub_map)
#         tasks.append(asyncio.create_task(send_request(stub, m, image_b64)))
#         task_models.append(m)
#
#     raw_results = await asyncio.gather(*tasks, return_exceptions=True)
#
#     results: List[dict] = []
#
#     for idx, out in enumerate(raw_results):
#         m = task_models[idx]
#
#         if isinstance(out, Exception):
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#             continue
#
#         res = out
#
#         if not res.get("success", False):
#             logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
#             continue
#
#         logger.info(
#             f"[RUN_OK] tx={transaction_id} model={m} "
#             f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#             f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#             f"container_state={res.get('container_state')} server={res.get('server_address')}"
#         )
#
#         results.append(res)
#
#     return results, "mode_s_executed"
#
#
# # -----------------------------
# # Parallel-first-finish cascade
# # -----------------------------
# async def execute_parallel_first_finish_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
#     selected_theta: Optional[float] = None,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(
#                 send_request(get_stub_for_model(m, stub_map), m, image_b64)
#             ): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 # first_thr = _get_stage_threshold(0)
#                 first_thr = float(selected_theta) if selected_theta is not None else _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')} "
#                     f"server={first_result.get('server_address')}"
#                 )
#
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} first_finished_model={first_model} "
#                     f"conf={first_conf:.4f} threshold={first_thr:.2f} pass={first_pass}"
#                 )
#
#         except Exception as e:
#             logger.exception(
#                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
#             )
#             first_result = None
#
#         # if first_result is not None and first_result.get("success", False) and first_pass:
#         #     for p in pending:
#         #         p.cancel()
#         #
#         #     for p in pending:
#         #         try:
#         #             await p
#         #         except Exception:
#         #             pass
#         #
#         #     results = [first_result]
#         #     stop_reason = f"early_exit_first_finished_pass={first_model}"
#         #
#         #     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#         #
#         #     return results, stop_reason
#
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     logger.info(
#                         f"[RUN_CANCELLED] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')}"
#                     )
#                 except Exception as e:
#                     logger.exception(
#                         f"[CANCEL_WAIT_FAIL] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')} error={e}"
#                     )
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         remaining_results: List[dict] = []
#
#         for p in pending:
#             model_name = task_map[p]
#
#             try:
#                 res = await p
#
#                 if res.get("success", False):
#                     remaining_results.append(res)
#
#                     logger.info(
#                         f"[RUN_OK] tx={transaction_id} model={model_name} "
#                         f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#                         f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                         f"container_state={res.get('container_state')} "
#                         f"server={res.get('server_address')}"
#                     )
#
#             except Exception as e:
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#
#         results.extend(remaining_results)
#
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             # agg_thr = _get_stage_threshold(1)
#             agg_thr = float(selected_theta) if selected_theta is not None else _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_first_batch={[r['model'] for r in results]}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#
#         try:
#             stub = get_stub_for_model(m, stub_map)
#             r = await send_request(stub, m, image_b64)
#
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#
#                 results.append(r)
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     return results, stop_reason
#
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         first_next_model = batch[0]
#
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         tasks = [
#             asyncio.create_task(
#                 send_request(get_stub_for_model(m, stub_map), m, image_b64)
#             )
#             for m in batch
#         ]
#
#         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#
#             if not res.get("success", False):
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} "
#                 f"conf={conf:.4f} rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
#             )
#
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
# async def execute_speculative_cloud_cascade(
#     stub_map,
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     selected_theta: Optional[float] = None,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#
#     if len(model_list) < 2:
#         raise ValueError("spec_b requires two models: [edge_model, cloud_model]")
#
#     theta = float(selected_theta) if selected_theta is not None else float(THRESHOLD)
#
#     edge_model = model_list[0]
#     cloud_model = model_list[1]
#
#     logger.info(
#         f"[SPEC_B_START] tx={transaction_id} edge={edge_model} "
#         f"cloud={cloud_model} theta={theta:.4f}"
#     )
#
#     task_map = {
#         asyncio.create_task(
#             send_request(get_stub_for_model(edge_model, stub_map), edge_model, image_b64)
#         ): edge_model,
#         asyncio.create_task(
#             send_request(get_stub_for_model(cloud_model, stub_map), cloud_model, image_b64)
#         ): cloud_model,
#     }
#
#     done, pending = await asyncio.wait(
#         set(task_map.keys()),
#         return_when=asyncio.FIRST_COMPLETED,
#     )
#
#     first_task = next(iter(done))
#     first_model = task_map[first_task]
#
#     try:
#         first_result = first_task.result()
#     except Exception as e:
#         logger.exception(f"[SPEC_B_FAIL] tx={transaction_id} first_model={first_model} error={e}")
#         first_result = None
#
#     if first_result is not None and first_result.get("success", False):
#         first_conf = float(first_result.get("probability", 0.0))
#         first_pass = first_conf >= theta
#
#         logger.info(
#             f"[SPEC_B_FIRST] tx={transaction_id} model={first_model} "
#             f"conf={first_conf:.4f} theta={theta:.4f} pass={first_pass}"
#         )
#
#         if first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception:
#                     pass
#
#             return [first_result], f"spec_b_early_exit_first_pass={first_model}"
#
#         results.append(first_result)
#
#     for p in pending:
#         model_name = task_map[p]
#
#         try:
#             res = await p
#             if res.get("success", False):
#                 results.append(res)
#
#                 logger.info(
#                     f"[SPEC_B_SECOND] tx={transaction_id} model={model_name} "
#                     f"conf={float(res.get('probability', 0.0)):.4f}"
#                 )
#
#         except Exception as e:
#             logger.exception(f"[SPEC_B_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#     if len(results) == 0:
#         return [], "spec_b_no_successful_models"
#
#     if len(results) == 1:
#         return results, f"spec_b_only_one_success={results[0].get('model')}"
#
#     return results, "spec_b_aggregate_after_both"
#
#
#
# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None,
# ) -> dict:
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} baseline_mode={BASELINE_MODE} "
#         f"parallel_first_n={PARALLEL_FIRST_N} parallel_batch_size={PARALLEL_BATCH_SIZE} "
#         f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES}"
#     )
#
#     wait_results = get_wait_results_for_models(M_TOTAL)
#
#     selected_theta = None
#     static_config_details = {}
#     constraint_solver_details = {}
#     risk_details = {}
#
#     parallel_first_n_runtime = PARALLEL_FIRST_N
#
#     # ------------------------------------------------------------
#     # 1. Static optimal config from  table
#     # ------------------------------------------------------------
#     if USE_STATIC_OPTIMAL_CONFIG:
#         cfg = get_optimal_config(
#             cost=STATIC_COST,
#             error=STATIC_ERROR,
#             policy=STATIC_POLICY,
#         )
#
#         M_D = list(cfg["models"])
#         selected_theta = float(cfg["theta"])
#         parallel_first_n_runtime = int(cfg.get("parallel_first_n", PARALLEL_FIRST_N))
#
#         if STATIC_POLICY == "sequential":
#             effective_cascade_mode = "sequential"
#         elif STATIC_POLICY == "po2":
#             effective_cascade_mode = "parallel_first_finish"
#         elif STATIC_POLICY == "spec_b":
#             effective_cascade_mode = "spec_b"
#         else:
#             raise ValueError(f"Unknown STATIC_POLICY={STATIC_POLICY}")
#
#         total_estimates = {
#             m: wait_results.get(m, float("inf"))
#             for m in M_D
#         }
#
#         static_config_details = {
#             "enabled": True,
#             "cost": STATIC_COST,
#             "error": STATIC_ERROR,
#             "target_accuracy": 1.0 - float(STATIC_ERROR),
#             "policy": STATIC_POLICY,
#             "models": M_D,
#             "theta": selected_theta,
#             "parallel_first_n": parallel_first_n_runtime,
#         }
#
#         logger.info(
#             f"[STATIC_OPTIMAL_CONFIG] tx={transaction_id} "
#             f"C={STATIC_COST} E={STATIC_ERROR} policy={STATIC_POLICY} "
#             f"models={M_D} theta={selected_theta} "
#             f"parallel_first_n={parallel_first_n_runtime} "
#             f"effective_mode={effective_cascade_mode}"
#         )
#
#     # ------------------------------------------------------------
#     # 2. Normal MODE-S selection
#     # ------------------------------------------------------------
#     else:
#         M_D, total_estimates, effective_cascade_mode = build_model_set_by_mode(
#             cascade_mode=CASCADE_MODE,
#             wait_results=wait_results,
#             deadline_ms=deadline,
#             ensemble_size=ENSEMBLE_SIZE,
#             alpha_now=alpha_now,
#             parallel_first_n=PARALLEL_FIRST_N,
#             network_latency=NETWORK_LATENCY,
#             logger=logger,
#         )
#
#     # ------------------------------------------------------------
#     # 3. Empty selection fallback
#     # ------------------------------------------------------------
#     if not M_D:
#         fastest_model = min(
#             MODEL_PROFILES,
#             key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#         )
#
#         M_D = [fastest_model]
#         total_estimates = {
#             fastest_model: wait_results.get(fastest_model, float("inf"))
#         }
#         effective_cascade_mode = "sequential"
#         selected_theta = None
#
#         logger.warning(
#             f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}"
#         )
#
#     # ------------------------------------------------------------
#     # 4. Constraint solver only when static config is disabled
#     # ------------------------------------------------------------
#     if (not USE_STATIC_OPTIMAL_CONFIG) and USE_CONSTRAINT_SOLVER:
#         constraint_solver = ConstraintSolver(
#             seq_csv=CONSTRAINT_SEQ_PROFILE_CSV,
#             power2_csv=CONSTRAINT_POWER2_PROFILE_CSV,
#             full_parallel_csv=CONSTRAINT_FULL_PARALLEL_PROFILE_CSV,
#         )
#
#         constraint_result = constraint_solver.solve(
#             max_cost=CONSTRAINT_MAX_COST,
#             target_accuracy=CONSTRAINT_TARGET_ACCURACY,
#             policy_order=CONSTRAINT_POLICY_ORDER,
#             time_mode=CONSTRAINT_SOLVER_TIME_MODE,
#             wait_results=wait_results,
#             model_profiles=MODEL_PROFILES,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#             base_request_overhead=BASE_REQUEST_OVERHEAD,
#             sequential_stage_overhead=SEQUENTIAL_STAGE_OVERHEAD,
#             po2_first_batch_overhead=PO2_FIRST_BATCH_OVERHEAD,
#             po2_tail_overhead=PO2_TAIL_OVERHEAD,
#             full_parallel_agg_overhead=FULL_PARALLEL_AGG_OVERHEAD,
#             et_scale_seq=ET_SCALE_SEQ,
#             et_offset_seq=ET_OFFSET_SEQ,
#             et_scale_p2=ET_SCALE_P2,
#             et_offset_p2=ET_OFFSET_P2,
#         )
#
#         constraint_solver_details = {
#             "success": constraint_result.success,
#             "policy": constraint_result.policy,
#             "reason": constraint_result.reason,
#             "feasible_count": constraint_result.feasible_count,
#             "total_count": constraint_result.total_count,
#             "solver_latency_ms": constraint_result.solver_latency_ms,
#             "selected_config": constraint_result.selected_config,
#             "time_mode": CONSTRAINT_SOLVER_TIME_MODE,
#             "max_cost": CONSTRAINT_MAX_COST,
#             "target_accuracy": CONSTRAINT_TARGET_ACCURACY,
#         }
#
#         if constraint_result.success:
#             selected = constraint_result.selected_config
#             selected_policy = selected["policy"]
#
#             M_D = list(selected["models"])
#
#             total_estimates = {
#                 m: wait_results.get(m, float("inf"))
#                 for m in M_D
#             }
#
#             if selected_policy == "sequential":
#                 effective_cascade_mode = "sequential"
#                 selected_theta = float(selected["theta"])
#
#             elif selected_policy in ("power2", "po2"):
#                 effective_cascade_mode = "parallel_first_finish"
#                 selected_theta = float(selected.get("theta", THRESHOLD))
#                 parallel_first_n_runtime = int(
#                     selected.get("parallel_first_n", PARALLEL_FIRST_N)
#                 )
#
#             elif selected_policy == "full_parallel":
#                 effective_cascade_mode = "full_parallel"
#                 selected_theta = None
#
#             else:
#                 effective_cascade_mode = "sequential"
#                 selected_theta = None
#
#             logger.info(
#                 f"[CONSTRAINT_SOLVER] tx={transaction_id} "
#                 f"policy={selected_policy} models={M_D} "
#                 f"theta={selected_theta} "
#                 f"parallel_first_n={parallel_first_n_runtime}"
#             )
#
#         else:
#             fastest_model = min(
#                 MODEL_PROFILES,
#                 key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#             )
#
#             M_D = [fastest_model]
#             total_estimates = {
#                 fastest_model: wait_results.get(fastest_model, float("inf"))
#             }
#             effective_cascade_mode = "sequential"
#             selected_theta = None
#
#             logger.warning(
#                 f"[CONSTRAINT_SOLVER_FALLBACK] tx={transaction_id} "
#                 f"reason={constraint_result.reason} fallback_fastest={fastest_model}"
#             )
#
#     # ------------------------------------------------------------
#     # 5. Risk selection only when static and constraint solver are disabled
#     # ------------------------------------------------------------
#     elif (
#         (not USE_STATIC_OPTIMAL_CONFIG)
#         and (not USE_CONSTRAINT_SOLVER)
#         and ENABLE_RISK_STRATEGY_SELECTION
#         and len(M_D) > 1
#     ):
#         risk_selected_mode, risk_details = select_min_risk_strategy(
#             selected_models=M_D,
#             wait_results=wait_results,
#             deadline_ms=deadline,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#             model_profiles=MODEL_PROFILES,
#             omega=OMEGA,
#             small_instance_models=SMALL_INSTANCE_MODELS,
#             large_instance_models=LARGE_INSTANCE_MODELS,
#             hardware_cost_rate=HARDWARE_COST_RATE,
#             estimate_latency_fn=estimate_selected_set_latency_for_mode,
#             alpha_latency=RISK_ALPHA_LATENCY,
#             beta_cost=RISK_BETA_COST,
#             gamma_error=RISK_GAMMA_ERROR,
#         )
#
#         effective_cascade_mode = risk_selected_mode
#
#         logger.info(
#             f"[RISK_SELECT] tx={transaction_id} "
#             f"selected_mode={risk_selected_mode} details={risk_details}"
#         )
#
#     # ------------------------------------------------------------
#     # 6. Estimate selected latency
#     # ------------------------------------------------------------
#     raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#         cascade_mode=effective_cascade_mode,
#         selected_models=M_D,
#         wait_results=wait_results,
#         network_latency=NETWORK_LATENCY,
#         parallel_first_n=parallel_first_n_runtime,
#     )
#
#     selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#
#     deadline_sec = float(deadline) / 1000.0 if deadline is not None else 0.0
#     deadline_fast_path_triggered = False
#
#     if (
#         USE_DEADLINE_FAST_PATH
#         and deadline_sec > 0.0
#         and selected_est_latency >= deadline_sec
#     ):
#         fastest_model = min(
#             MODEL_PROFILES,
#             key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#         )
#
#         logger.warning(
#             f"[DEADLINE_FAST_PATH] tx={transaction_id} "
#             f"mode={effective_cascade_mode} selected_models={M_D} "
#             f"selected_est_latency={selected_est_latency:.4f}s "
#             f"deadline={deadline_sec:.4f}s fallback_fastest={fastest_model}"
#         )
#
#         M_D = [fastest_model]
#         total_estimates = {
#             fastest_model: wait_results.get(fastest_model, float("inf"))
#         }
#         effective_cascade_mode = "sequential"
#         selected_theta = None
#
#         raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#             cascade_mode=effective_cascade_mode,
#             selected_models=M_D,
#             wait_results=wait_results,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=parallel_first_n_runtime,
#         )
#
#         selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#         deadline_fast_path_triggered = True
#
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#
#     logger.info(
#         f"[EFFECTIVE_MODE] tx={transaction_id} baseline_mode={BASELINE_MODE} "
#         f"configured_cascade_mode={CASCADE_MODE} "
#         f"effective_cascade_mode={effective_cascade_mode}"
#     )
#
#     logger.info(
#         f"[SELECT_WAITS] tx={transaction_id} waits_selected="
#         f"{ {m: float(total_estimates.get(m, -1)) for m in M_D} }"
#     )
#
#     logger.info(
#         f"[SELECT_ROUTE] tx={transaction_id} routes="
#         f"{ {m: get_server_address_for_model(m) for m in M_D} }"
#     )
#
#     channel_map, stub_map = build_stub_map_for_selected_models(M_D)
#
#     try:
#         mode = str(effective_cascade_mode).lower()
#
#         if mode == "sequential":
#             results, stop_reason = await execute_sequential_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_theta=selected_theta,
#             )
#
#         elif mode in ("mode_s", "full_parallel"):
#             results, stop_reason = await execute_mode_s_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#             )
#
#         elif mode == "spec_b":
#             results, stop_reason = await execute_speculative_cloud_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 selected_theta=selected_theta,
#             )
#
#         else:
#             results, stop_reason = await execute_parallel_first_finish_cascade(
#                 stub_map=stub_map,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 parallel_first_n=parallel_first_n_runtime,
#                 parallel_batch_size=PARALLEL_BATCH_SIZE,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_folder=selected_folder,
#                 selected_theta=selected_theta,
#             )
#
#     finally:
#         await close_channels(channel_map)
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000.0
#     actual_latency_s = e2e_ms / 1000.0
#
#     update_latency_overhead(
#         actual_latency_s=actual_latency_s,
#         estimated_latency_s=raw_selected_est_latency,
#     )
#
#     per_model = {}
#
#     for res in results:
#         model_name = res.get("model")
#         if not model_name:
#             continue
#
#         per_model[model_name] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#             "server_address": res.get("server_address"),
#         }
#
#     base_return = {
#         "selected_models": M_D,
#         "per_model": per_model,
#         "wait_times": total_estimates,
#         "selected_folder": selected_folder,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "selected_theta": selected_theta,
#         "parallel_first_n": parallel_first_n_runtime,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         "stop_reason": stop_reason,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "baseline_mode": BASELINE_MODE,
#         "configured_cascade_mode": CASCADE_MODE,
#         "effective_cascade_mode": effective_cascade_mode,
#         "cascade_mode": CASCADE_MODE,
#         "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#         "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#         "deadline_fast_path_triggered": deadline_fast_path_triggered,
#         "selected_est_latency_s": selected_est_latency,
#         "raw_selected_est_latency_s": raw_selected_est_latency,
#         "recent_latency_overhead_s": get_recent_latency_overhead(),
#         "log_file": LOG_PATH,
#         "risk_details": risk_details,
#         "risk_selected_mode": effective_cascade_mode,
#         "constraint_solver_details": constraint_solver_details,
#         "static_config_details": static_config_details,
#         "static_cost": STATIC_COST if USE_STATIC_OPTIMAL_CONFIG else None,
#         "static_error": STATIC_ERROR if USE_STATIC_OPTIMAL_CONFIG else None,
#         "static_policy": STATIC_POLICY if USE_STATIC_OPTIMAL_CONFIG else None,
#     }
#
#     if not results:
#         logger.error(
#             f"[REQ_END] tx={transaction_id} success=False "
#             f"reason=no_successful_models e2e_ms={e2e_ms:.2f}"
#         )
#
#         return {
#             **base_return,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#         }
#
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (
#                 (1 - ETA) * OMEGA[res["model"]]
#                 + ETA * float(res.get("probability", 0.0))
#             )
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} "
#         f"success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} "
#         f"alpha={alpha_now:.3f} "
#         f"configured_cascade_mode={CASCADE_MODE} "
#         f"effective_cascade_mode={effective_cascade_mode} "
#         f"baseline_mode={BASELINE_MODE}"
#     )
#
#     return {
#         **base_return,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#     }



import grpc
import json
import uuid
import time
import asyncio
import logging
import os
from collections import deque
from typing import Dict, List, Tuple, Optional
from logging.handlers import RotatingFileHandler

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
from ckn_controller import ckn_config as config

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    M_TOTAL,
    POLICY,
    ETA,
    MODEL_PROFILES,
    OMEGA,
    DEFAULT_WEIGHTS,
    GAMMA,
    RHO,
    WEIGHTS_STATE_PATH,
    ENSEMBLE_SIZE,
    THRESHOLD,
    THRESHOLD_STAGE,
    ALPHA_MODE,
    ALPHA_STATIC,
    ALPHA_MIN,
    ALPHA_MAX,
    QPS_WINDOW_SEC,
    QPS_LOW,
    QPS_HIGH,
    CASCADE_MODE,
    PARALLEL_FIRST_N,
    PARALLEL_BATCH_SIZE,
    NETWORK_LATENCY,
    USE_DEADLINE_FAST_PATH,
    USE_TWO_ILUVATAR_INSTANCES,
    SMALL_MODEL_SERVER_ADDRESS,
    LARGE_MODEL_SERVER_ADDRESS,
    SMALL_INSTANCE_MODELS,
    LARGE_INSTANCE_MODELS,
    BASELINE_MODE,
    ENABLE_RISK_STRATEGY_SELECTION,
    RISK_ALPHA_LATENCY,
    RISK_BETA_COST,
    RISK_GAMMA_ERROR,
    HARDWARE_COST_RATE,
    # USE_SEQ_CONSTRAINT_SOLVER,
    # SEQ_CASCADE_PROFILE_CSV,
    # SEQ_MAX_COST,
    # SEQ_TARGET_ACCURACY,
    USE_CONSTRAINT_SOLVER,
    CONSTRAINT_SOLVER_TIME_MODE,
    CONSTRAINT_MAX_COST,
    CONSTRAINT_TARGET_ACCURACY,
    CONSTRAINT_POLICY_ORDER,
    CONSTRAINT_SEQ_PROFILE_CSV,
    CONSTRAINT_POWER2_PROFILE_CSV,
    CONSTRAINT_FULL_PARALLEL_PROFILE_CSV,
    BASE_REQUEST_OVERHEAD,
    SEQUENTIAL_STAGE_OVERHEAD,
    PO2_FIRST_BATCH_OVERHEAD,
    PO2_TAIL_OVERHEAD,
    FULL_PARALLEL_AGG_OVERHEAD,
    ET_SCALE_SEQ,
    ET_OFFSET_SEQ,
    ET_SCALE_P2,
    ET_OFFSET_P2,
    USE_STATIC_OPTIMAL_CONFIG,
    STATIC_COST,
    STATIC_ERROR,
    STATIC_POLICY,
)

import wait_time_iluvatar_new_est
from ckn_controller.dysci_latency_estimator import (
    DYSCILatencyEstimator,
    ESTIMATOR_SOURCE,
    QUERY_ILUVATAR_FOR_DEBUG,
    choose_estimate,
)
from ckn_controller.output_combiner import combine_outputs
from ckn_controller.weights_io import load_model_weights, save_model_weights_atomic
from ckn_controller.label_utils import wnid_matches_text_label
from ckn_controller.ensemble_selector import (
    build_model_set_by_mode,
    estimate_selected_set_latency_for_mode,
)
from ckn_controller.risk_selector import select_min_risk_strategy
from ckn_controller.sequential_cascade_constraint_solver import (
    SequentialCascadeConstraintSolver,
)
from ckn_controller.constraint_solver import ConstraintSolver
from ckn_controller.optimal_configs import get_optimal_config


# ------------------------------------------------------------
# Multi-node configuration compatibility
# ------------------------------------------------------------
# These values are read with getattr so this ckn_main.py can still import
# before ckn_config.py is updated. Multi-node mode becomes active after
# USE_MULTI_ILUVATAR_INSTANCES and ILUVATAR_NODES are added to ckn_config.py.
USE_MULTI_ILUVATAR_INSTANCES = bool(
    getattr(config, "USE_MULTI_ILUVATAR_INSTANCES", False)
)
ILUVATAR_NODES = getattr(config, "ILUVATAR_NODES", {})


# -----------------------------
# Logging setup
# -----------------------------
LOG_DIR = os.environ.get("MODE_S_LOG_DIR", "./logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "mode_s_cascade.log")

logger = logging.getLogger("cascade")
logger.setLevel(logging.INFO)

if not logger.handlers:
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(
        LOG_PATH, maxBytes=10 * 1024 * 1024, backupCount=5
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)


MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
active_models = set()


DYSCI_ESTIMATOR = DYSCILatencyEstimator(
    model_profiles=MODEL_PROFILES,
)

logger.info(
    f"[LATENCY_ESTIMATOR_INIT] "
    f"primary_source={ESTIMATOR_SOURCE} "
    f"query_iluvatar_debug={QUERY_ILUVATAR_FOR_DEBUG}"
)


# -----------------------------
# QPS tracking
# -----------------------------
REQUEST_TS = deque()


def record_request(now: float, window_sec: float) -> None:
    REQUEST_TS.append(now)
    cutoff = now - window_sec

    while REQUEST_TS and REQUEST_TS[0] < cutoff:
        REQUEST_TS.popleft()


def get_qps(window_sec: float) -> float:
    return len(REQUEST_TS) / max(window_sec, 1e-9)


# -----------------------------
# Recent latency overhead tracking
# -----------------------------
LATENCY_OVERHEAD_HISTORY = deque(maxlen=50)
LATENCY_OVERHEAD_WEIGHT = 0.7


def get_recent_latency_overhead() -> float:
    if not LATENCY_OVERHEAD_HISTORY:
        return 0.0

    vals = [v for v in LATENCY_OVERHEAD_HISTORY if v > 0]
    if not vals:
        return 0.0

    return sum(vals) / len(vals)


def apply_latency_correction(estimated_latency: float) -> float:
    if estimated_latency == float("inf"):
        return estimated_latency

    recent_overhead = get_recent_latency_overhead()
    return estimated_latency + (LATENCY_OVERHEAD_WEIGHT * recent_overhead)


def update_latency_overhead(actual_latency_s: float, estimated_latency_s: float) -> None:
    if estimated_latency_s == float("inf"):
        return

    overhead = actual_latency_s - estimated_latency_s
    LATENCY_OVERHEAD_HISTORY.append(overhead)


# -----------------------------
# Adaptive alpha
# -----------------------------
def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def compute_adaptive_alpha(
    qps: float,
    alpha_min: float,
    alpha_max: float,
    qps_low: float,
    qps_high: float,
) -> float:
    if qps_high <= qps_low:
        qps_norm = 0.0
    else:
        qps_norm = _clamp01((qps - qps_low) / (qps_high - qps_low))

    alpha_now = alpha_max - qps_norm * (alpha_max - alpha_min)
    alpha_now = max(alpha_min, min(alpha_max, alpha_now))

    logger.info(
        f"[ALPHA_DEBUG] qps={qps:.4f} qps_norm={qps_norm:.4f} alpha={alpha_now:.6f}"
    )

    return alpha_now


# -----------------------------
# Deadline helper
# -----------------------------
def should_stop_before_next_model(
    start_time: float,
    deadline_ms: int,
    next_model: str,
    safety_margin_ms: float = 0.0,
) -> Tuple[bool, float, float]:
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    remaining_ms = float(deadline_ms) - elapsed_ms
    next_latency_ms = float(MODEL_PROFILES[next_model]["latency"]) * 1000.0

    stop_now = remaining_ms <= (next_latency_ms + safety_margin_ms)

    return stop_now, elapsed_ms, remaining_ms


# -----------------------------
# Multi-node routing helpers
# -----------------------------
def _single_or_dual_instance_route(model_name: str) -> dict:
    """
    Compatibility route used until multi-node mode is enabled.
    """
    if not USE_TWO_ILUVATAR_INSTANCES:
        return {
            "model": model_name,
            "node_id": "single_instance",
            "node_type": "single",
            "server_address": SERVER_ADDRESS,
            "estimated_wait_s": float("inf"),
        }

    if model_name in SMALL_INSTANCE_MODELS:
        return {
            "model": model_name,
            "node_id": "edge_1",
            "node_type": "edge",
            "server_address": SMALL_MODEL_SERVER_ADDRESS,
            "estimated_wait_s": float("inf"),
        }

    if model_name in LARGE_INSTANCE_MODELS:
        return {
            "model": model_name,
            "node_id": "cloud_1",
            "node_type": "cloud",
            "server_address": LARGE_MODEL_SERVER_ADDRESS,
            "estimated_wait_s": float("inf"),
        }

    raise ValueError(
        f"Model {model_name} is not assigned to any Iluvatar instance."
    )

def get_device_for_node(
    node_id: str,
) -> str:
    """
    Return the execution device configured for a node.

    Default to CPU for backward compatibility.
    """

    node = ILUVATAR_NODES.get(
        node_id,
        {},
    )

    return str(
        node.get(
            "device",
            "cpu",
        )
    ).lower()


def validate_multi_node_config() -> None:
    if not isinstance(ILUVATAR_NODES, dict) or not ILUVATAR_NODES:
        raise ValueError(
            "USE_MULTI_ILUVATAR_INSTANCES=True, but ILUVATAR_NODES is empty "
            "or is not a dictionary."
        )

    for node_id, node in ILUVATAR_NODES.items():
        if not isinstance(node, dict):
            raise ValueError(
                f"ILUVATAR_NODES['{node_id}'] must be a dictionary."
            )

        if not node.get("address"):
            raise ValueError(
                f"ILUVATAR_NODES['{node_id}'] is missing 'address'."
            )

        if not isinstance(node.get("models", []), list):
            raise ValueError(
                f"ILUVATAR_NODES['{node_id}']['models'] must be a list."
            )

        node_type = str(node.get("type", "")).lower()
        if node_type not in ("edge", "cloud"):
            raise ValueError(
                f"ILUVATAR_NODES['{node_id}']['type'] must be "
                f"'edge' or 'cloud', not '{node_type}'."
            )


def get_candidate_nodes_for_model(model_name: str) -> List[str]:
    """
    Return enabled nodes on which model_name is deployed.
    """
    if not USE_MULTI_ILUVATAR_INSTANCES:
        return [_single_or_dual_instance_route(model_name)["node_id"]]

    candidates: List[str] = []

    for node_id, node in ILUVATAR_NODES.items():
        if not bool(node.get("enabled", True)):
            continue

        if model_name in node.get("models", []):
            candidates.append(node_id)

    return candidates


def get_route_for_model(model_name: str, routing_plan: Dict[str, dict]) -> dict:
    route = routing_plan.get(model_name)

    if route is None:
        raise ValueError(
            f"No route was created for selected model '{model_name}'."
        )

    return route


def get_server_address_for_model(
    model_name: str,
    routing_plan: Optional[Dict[str, dict]] = None,
) -> str:
    """
    Return the request-specific address when a routing plan is available.
    Fall back to the previous single/dual-instance mapping otherwise.
    """
    if routing_plan is not None and model_name in routing_plan:
        return str(routing_plan[model_name]["server_address"])

    return str(_single_or_dual_instance_route(model_name)["server_address"])


def build_stub_map_for_selected_models(
    model_list: List[str],
    routing_plan: Dict[str, dict],
):
    addresses = {
        get_server_address_for_model(m, routing_plan)
        for m in model_list
    }

    channel_map = {}
    stub_map = {}

    for address in addresses:
        channel = grpc.aio.insecure_channel(address)
        stub = pb2_grpc.IluvatarWorkerStub(channel)

        channel_map[address] = channel
        stub_map[address] = stub

    return channel_map, stub_map


async def close_channels(channel_map):
    for channel in channel_map.values():
        try:
            await channel.close()
        except Exception:
            pass


def get_stub_for_model(
    model_name: str,
    stub_map,
    routing_plan: Dict[str, dict],
):
    address = get_server_address_for_model(model_name, routing_plan)
    return stub_map[address]


def _query_waits_from_node(
    node_id: str,
    server_address: str,
    model_list: List[str],
) -> Dict[str, float]:
    """
    Query one Iluvatar node through the existing wait_time_iluvatar.main().
    """
    if not model_list:
        return {}

    try:
        waits = wait_time_iluvatar_new_est.main(
            server_address=server_address,
            model_list=model_list,
        )
    except Exception as error:
        logger.exception(
            f"[WAIT_NODE_FAIL] node={node_id} address={server_address} "
            f"models={model_list} error={error}"
        )
        return {model: float("inf") for model in model_list}

    return {
        model: float(waits.get(model, float("inf")))
        for model in model_list
    }


# def get_wait_results_for_models(
#     model_list: List[str],
# ) -> Tuple[Dict[str, float], Dict[str, dict], Dict[str, Dict[str, float]]]:
#     """
#     Query all eligible nodes and select the node with the minimum current
#     Iluvatar estimated invocation time for each model.
#
#     Returns:
#         best_model_waits:
#             Existing model-level format used by the selector.
#
#         model_routes:
#             Per-request best node/address for every model.
#
#         node_wait_results:
#             Full node-specific wait-time information for logging/evaluation.
#     """
#     requested_models = list(dict.fromkeys(model_list))
#
#     # --------------------------------------------------------
#     # Existing single/dual-instance compatibility mode
#     # --------------------------------------------------------
#     if not USE_MULTI_ILUVATAR_INSTANCES:
#         node_wait_results: Dict[str, Dict[str, float]] = {}
#         model_routes: Dict[str, dict] = {}
#         best_model_waits: Dict[str, float] = {}
#
#         if not USE_TWO_ILUVATAR_INSTANCES:
#             waits = _query_waits_from_node(
#                 node_id="single_instance",
#                 server_address=SERVER_ADDRESS,
#                 model_list=requested_models,
#             )
#             node_wait_results["single_instance"] = waits
#
#             for model in requested_models:
#                 wait = float(waits.get(model, float("inf")))
#                 route = _single_or_dual_instance_route(model)
#                 route["estimated_wait_s"] = wait
#                 model_routes[model] = route
#                 best_model_waits[model] = wait
#
#             return best_model_waits, model_routes, node_wait_results
#
#         small_models = [
#             model for model in requested_models
#             if model in SMALL_INSTANCE_MODELS
#         ]
#         large_models = [
#             model for model in requested_models
#             if model in LARGE_INSTANCE_MODELS
#         ]
#
#         if small_models:
#             waits = _query_waits_from_node(
#                 node_id="edge_1",
#                 server_address=SMALL_MODEL_SERVER_ADDRESS,
#                 model_list=small_models,
#             )
#             node_wait_results["edge_1"] = waits
#
#         if large_models:
#             waits = _query_waits_from_node(
#                 node_id="cloud_1",
#                 server_address=LARGE_MODEL_SERVER_ADDRESS,
#                 model_list=large_models,
#             )
#             node_wait_results["cloud_1"] = waits
#
#         for model in requested_models:
#             route = _single_or_dual_instance_route(model)
#             node_id = route["node_id"]
#             wait = float(
#                 node_wait_results.get(node_id, {}).get(
#                     model,
#                     float("inf"),
#                 )
#             )
#             route["estimated_wait_s"] = wait
#             model_routes[model] = route
#             best_model_waits[model] = wait
#
#         return best_model_waits, model_routes, node_wait_results
#
#     # --------------------------------------------------------
#     # General N-edge / M-cloud mode
#     # --------------------------------------------------------
#     validate_multi_node_config()
#     requested_set = set(requested_models)
#     node_wait_results: Dict[str, Dict[str, float]] = {}
#
#     for node_id, node in ILUVATAR_NODES.items():
#         if not bool(node.get("enabled", True)):
#             logger.info(f"[WAIT_NODE_SKIP] node={node_id} enabled=False")
#             continue
#
#         address = str(node["address"])
#         deployed_models = list(node.get("models", []))
#         models_to_query = [
#             model for model in deployed_models
#             if model in requested_set
#         ]
#
#         if not models_to_query:
#             continue
#
#         node_wait_results[node_id] = _query_waits_from_node(
#             node_id=node_id,
#             server_address=address,
#             model_list=models_to_query,
#         )
#
#     best_model_waits: Dict[str, float] = {}
#     model_routes: Dict[str, dict] = {}
#
#     for model in requested_models:
#         candidate_node_ids = get_candidate_nodes_for_model(model)
#
#         if not candidate_node_ids:
#             logger.error(
#                 f"[ROUTE_MISSING] model={model} "
#                 f"reason=no_enabled_node_supports_model"
#             )
#             best_model_waits[model] = float("inf")
#             continue
#
#         candidates = []
#
#         for node_id in candidate_node_ids:
#             node = ILUVATAR_NODES[node_id]
#             wait_s = float(
#                 node_wait_results.get(node_id, {}).get(
#                     model,
#                     float("inf"),
#                 )
#             )
#
#             candidates.append(
#                 {
#                     "model": model,
#                     "node_id": node_id,
#                     "node_type": str(node.get("type", "unknown")),
#                     "server_address": str(node["address"]),
#                     "estimated_wait_s": wait_s,
#                 }
#             )
#
#         # min() also gives a deterministic candidate when all estimates are inf.
#         selected_route = min(
#             candidates,
#             key=lambda route: (
#                 float(route["estimated_wait_s"]),
#                 str(route["node_id"]),
#             ),
#         )
#
#         best_model_waits[model] = float(
#             selected_route["estimated_wait_s"]
#         )
#         model_routes[model] = selected_route
#
#         logger.info(
#             f"[NODE_SELECT] model={model} "
#             f"candidates="
#             f"{[(r['node_id'], r['estimated_wait_s']) for r in candidates]} "
#             f"selected_node={selected_route['node_id']} "
#             f"selected_address={selected_route['server_address']} "
#             f"selected_wait_s={selected_route['estimated_wait_s']}"
#         )
#
#     return best_model_waits, model_routes, node_wait_results

def get_wait_results_for_models(
    model_list: List[str],
) -> Tuple[
    Dict[str, float],
    Dict[str, dict],
    Dict[str, Dict[str, float]],
]:
    """
    Get latency estimates and routes for the requested models.

    In multi-node mode:

        1. DYSCI computes its application-side estimate.

        2. Iluvatar estimate is optionally collected.

        3. choose_estimate() decides which one drives routing.

    ESTIMATOR_SOURCE is configured inside
    dysci_latency_estimator.py.

    Return values remain compatible with the existing code:

        best_model_waits
        model_routes
        node_wait_results
    """

    requested_models = list(
        dict.fromkeys(
            model_list
        )
    )

    # ========================================================
    # Legacy single/two-instance mode
    # ========================================================

    if not USE_MULTI_ILUVATAR_INSTANCES:

        wait_results = (
            wait_time_iluvatar_new_est
            .get_wait_results(
                requested_models
            )
        )

        model_routes = {}

        node_wait_results = {}

        for model in requested_models:

            route = (
                _single_or_dual_instance_route(
                    model
                )
            )

            wait_s = float(
                wait_results.get(
                    model,
                    float("inf"),
                )
            )

            route[
                "estimated_wait_s"
            ] = wait_s

            route[
                "estimate_source"
            ] = "iluvatar"

            model_routes[
                model
            ] = route

            node_id = route[
                "node_id"
            ]

            node_wait_results.setdefault(
                node_id,
                {},
            )[model] = wait_s

        return (
            wait_results,
            model_routes,
            node_wait_results,
        )


    # ========================================================
    # Multi-node mode
    # ========================================================

    validate_multi_node_config()

    requested_set = set(
        requested_models
    )

    # --------------------------------------------------------
    # Iluvatar estimates
    #
    # Keep this available even when DYSCI is primary.
    # --------------------------------------------------------

    iluvatar_node_results: Dict[
        str,
        Dict[str, float]
    ] = {}

    should_query_iluvatar = (
        QUERY_ILUVATAR_FOR_DEBUG
        or str(
            ESTIMATOR_SOURCE
        ).lower() == "iluvatar"
    )

    if should_query_iluvatar:

        for (
            node_id,
            node
        ) in ILUVATAR_NODES.items():

            if not bool(
                node.get(
                    "enabled",
                    True,
                )
            ):
                continue

            address = str(
                node["address"]
            )

            deployed_models = list(
                node.get(
                    "models",
                    [],
                )
            )

            models_to_query = [
                model
                for model in deployed_models
                if model in requested_set
            ]

            if not models_to_query:
                continue

            try:

                iluvatar_node_results[
                    node_id
                ] = _query_waits_from_node(
                    node_id=node_id,
                    server_address=address,
                    model_list=models_to_query,
                )

            except Exception as exc:

                logger.exception(
                    f"[ILUVATAR_ESTIMATE_FAIL] "
                    f"node={node_id} "
                    f"error={exc}"
                )

                iluvatar_node_results[
                    node_id
                ] = {}


    # --------------------------------------------------------
    # Build candidate routes
    # --------------------------------------------------------

    best_model_waits: Dict[
        str,
        float
    ] = {}

    model_routes: Dict[
        str,
        dict
    ] = {}

    # Preserve existing return structure.
    #
    # IMPORTANT:
    # Values stored here are the SELECTED estimator values,
    # not necessarily Iluvatar values.
    node_wait_results: Dict[
        str,
        Dict[str, float]
    ] = {}


    for model in requested_models:

        candidate_node_ids = (
            get_candidate_nodes_for_model(
                model
            )
        )

        if not candidate_node_ids:

            logger.error(
                f"[ROUTE_MISSING] "
                f"model={model} "
                f"reason="
                f"no_enabled_node_supports_model"
            )

            best_model_waits[
                model
            ] = float("inf")

            continue


        candidates = []


        for node_id in candidate_node_ids:

            node = ILUVATAR_NODES[
                node_id
            ]

            node_type = str(
                node.get(
                    "type",
                    "unknown",
                )
            )

            device = (
                get_device_for_node(
                    node_id
                )
            )

            # ================================================
            # DYSCI estimate
            # ================================================

            dysci_info = (
                DYSCI_ESTIMATOR.estimate(
                    node_id=node_id,
                    node_type=node_type,
                    device=device,
                    model=model,
                )
            )

            dysci_estimate_s = float(
                dysci_info[
                    "estimated_e2e_s"
                ]
            )


            # ================================================
            # Iluvatar estimate
            # ================================================

            iluvatar_estimate_s = float(
                iluvatar_node_results
                .get(
                    node_id,
                    {}
                )
                .get(
                    model,
                    float("inf"),
                )
            )


            # ================================================
            # Choose primary estimate
            # ================================================

            (
                selected_estimate_s,
                estimate_source,
            ) = choose_estimate(
                dysci_estimate_s=(
                    dysci_estimate_s
                ),
                iluvatar_estimate_s=(
                    iluvatar_estimate_s
                ),
            )


            candidate = {
                "model":
                    model,

                "node_id":
                    node_id,

                "node_type":
                    node_type,

                "device":
                    device,

                "server_address":
                    str(
                        node["address"]
                    ),

                # --------------------------------------------
                # Existing code expects this field.
                #
                # We keep the name for compatibility.
                #
                # When source=dysci this is DYSCI's predicted
                # invocation completion/E2E latency.
                # --------------------------------------------

                "estimated_wait_s":
                    float(
                        selected_estimate_s
                    ),

                "estimate_source":
                    estimate_source,

                # --------------------------------------------
                # DYSCI details
                # --------------------------------------------

                "dysci_estimated_e2e_s":
                    dysci_estimate_s,

                "dysci_estimated_wait_s":
                    float(
                        dysci_info[
                            "estimated_wait_s"
                        ]
                    ),

                "dysci_estimated_service_s":
                    float(
                        dysci_info[
                            "estimated_service_s"
                        ]
                    ),

                "dysci_outstanding":
                    int(
                        dysci_info[
                            "outstanding"
                        ]
                    ),

                "dysci_outstanding_work_s":
                    float(
                        dysci_info[
                            "outstanding_work_s"
                        ]
                    ),

                "dysci_estimator_source":
                    str(
                        dysci_info[
                            "source"
                        ]
                    ),

                "dysci_history_samples":
                    int(
                        dysci_info[
                            "history_samples"
                        ]
                    ),

                # --------------------------------------------
                # Iluvatar value retained for comparison
                # --------------------------------------------

                "iluvatar_estimate_s":
                    iluvatar_estimate_s,
            }


            candidates.append(
                candidate
            )


            node_wait_results.setdefault(
                node_id,
                {},
            )[model] = float(
                selected_estimate_s
            )


            logger.info(
                f"[LATENCY_ESTIMATE] "
                f"model={model} "
                f"node={node_id} "
                f"device={device} "
                f"selected_source="
                f"{estimate_source} "
                f"selected_s="
                f"{selected_estimate_s:.6f} "
                f"dysci_s="
                f"{dysci_estimate_s:.6f} "
                f"iluvatar_s="
                f"{iluvatar_estimate_s:.6f} "
                f"outstanding="
                f"{dysci_info['outstanding']} "
                f"work_s="
                f"{dysci_info['outstanding_work_s']:.6f} "
                f"dysci_source="
                f"{dysci_info['source']} "
                f"samples="
                f"{dysci_info['history_samples']}"
            )


        # ====================================================
        # Pick best node using whichever estimator is active.
        # ====================================================

        selected_route = min(
            candidates,
            key=lambda route: (
                float(
                    route[
                        "estimated_wait_s"
                    ]
                ),
                str(
                    route[
                        "node_id"
                    ]
                ),
            ),
        )


        best_model_waits[
            model
        ] = float(
            selected_route[
                "estimated_wait_s"
            ]
        )


        model_routes[
            model
        ] = selected_route


        logger.info(
            f"[NODE_SELECT] "
            f"model={model} "
            f"candidates="
            f"{[(r['node_id'], r['estimated_wait_s'], r['estimate_source']) for r in candidates]} "
            f"selected_node="
            f"{selected_route['node_id']} "
            f"selected_address="
            f"{selected_route['server_address']} "
            f"selected_estimate_s="
            f"{selected_route['estimated_wait_s']:.6f} "
            f"estimate_source="
            f"{selected_route['estimate_source']}"
        )


    return (
        best_model_waits,
        model_routes,
        node_wait_results,
    )


def build_routing_plan(
    selected_models: List[str],
    model_routes: Dict[str, dict],
) -> Dict[str, dict]:
    """
    Extract routes only for the models selected for this request.
    """
    routing_plan: Dict[str, dict] = {}

    for model in selected_models:
        route = model_routes.get(model)

        if route is None:
            raise ValueError(
                f"Selected model '{model}' has no available Iluvatar route."
            )

        routing_plan[model] = dict(route)

    return routing_plan


# -----------------------------
# Iluvatar invocation
# -----------------------------
# async def send_request(
#     stub,
#     model_name: str,
#     image_b64: str,
#     route: dict,
# ) -> dict:
#     request = pb2.InvokeRequest(
#         function_name=model_name,
#         function_version="1",
#         json_args=json.dumps(
#             {
#                 "model_name": model_name,
#                 "image_data": image_b64,
#             }
#         ),
#         transaction_id=str(uuid.uuid4()),
#     )
#
#     response = await stub.invoke(request)
#     result_json = json.loads(response.json_result)
#
#     return {
#         "model": model_name,
#         "label": result_json["body"]["Prediction Class"],
#         "probability": float(result_json["body"]["Probability"]),
#         "latency": response.duration_us / 1e6,
#         "success": response.success,
#         "container_state": pb2.ContainerState.Name(response.container_state),
#         "node_id": route.get("node_id"),
#         "node_type": route.get("node_type"),
#         "server_address": route.get("server_address"),
#         "estimated_wait_s": float(
#             route.get("estimated_wait_s", float("inf"))
#         ),
#     }


async def send_request(
    stub,
    model_name: str,
    image_b64: str,
    route: dict,
) -> dict:

    request_id = str(
        uuid.uuid4()
    )

    request = pb2.InvokeRequest(
        function_name=model_name,
        function_version="1",
        json_args=json.dumps(
            {
                "model_name":
                    model_name,

                "image_data":
                    image_b64,
            }
        ),
        transaction_id=request_id,
    )


    # ========================================================
    # Route information
    # ========================================================

    node_id = str(
        route.get(
            "node_id",
            "unknown",
        )
    )

    node_type = str(
        route.get(
            "node_type",
            "unknown",
        )
    )

    device = str(
        route.get(
            "device",
            "cpu",
        )
    ).lower()


    # ========================================================
    # Tell DYSCI the request is being dispatched.
    #
    # Do this BEFORE stub.invoke().
    #
    # Therefore even a request currently travelling toward
    # Iluvatar contributes to DYSCI-visible outstanding load.
    # ========================================================

    load_info = (
        DYSCI_ESTIMATOR.request_started(
            request_id=request_id,
            node_id=node_id,
            node_type=node_type,
            device=device,
            model=model_name,
        )
    )


    logger.info(
        f"[DYSCI_DISPATCH] "
        f"request_id={request_id} "
        f"model={model_name} "
        f"node={node_id} "
        f"device={device} "
        f"outstanding_before="
        f"{load_info['outstanding_before']} "
        f"outstanding_after="
        f"{load_info['outstanding_after']} "
        f"work_before_s="
        f"{load_info['work_before_s']:.6f} "
        f"work_after_s="
        f"{load_info['work_after_s']:.6f}"
    )


    wall_start = (
        time.perf_counter()
    )

    success = False


    try:

        # ====================================================
        # Actual Iluvatar invocation
        # ====================================================

        response = await stub.invoke(
            request
        )


        wall_s = (
            time.perf_counter()
            - wall_start
        )


        success = bool(
            response.success
        )


        result_json = json.loads(
            response.json_result
        )


        result = {
            "model":
                model_name,

            "label":
                result_json[
                    "body"
                ][
                    "Prediction Class"
                ],

            "probability":
                float(
                    result_json[
                        "body"
                    ][
                        "Probability"
                    ]
                ),

            # -----------------------------------------------
            # Iluvatar-reported invocation duration
            # -----------------------------------------------

            "latency":
                response.duration_us
                / 1e6,

            # -----------------------------------------------
            # DYSCI-observed wall time:
            #
            # immediately before invoke
            #       ->
            # response received
            # -----------------------------------------------

            "dysci_wall_s":
                wall_s,

            "success":
                success,

            "container_state":
                pb2.ContainerState.Name(
                    response.container_state
                ),

            "node_id":
                node_id,

            "node_type":
                node_type,

            "device":
                device,

            "server_address":
                route.get(
                    "server_address"
                ),

            # Existing field retained.
            "estimated_wait_s":
                float(
                    route.get(
                        "estimated_wait_s",
                        float("inf"),
                    )
                ),

            "estimate_source":
                route.get(
                    "estimate_source"
                ),

            "dysci_estimated_e2e_s":
                float(
                    route.get(
                        "dysci_estimated_e2e_s",
                        float("inf"),
                    )
                ),

            "dysci_estimated_wait_s":
                float(
                    route.get(
                        "dysci_estimated_wait_s",
                        0.0,
                    )
                ),

            "dysci_estimated_service_s":
                float(
                    route.get(
                        "dysci_estimated_service_s",
                        MODEL_PROFILES[
                            model_name
                        ][
                            "latency"
                        ],
                    )
                ),

            "dysci_outstanding_at_estimate":
                route.get(
                    "dysci_outstanding"
                ),

            "dysci_outstanding_work_at_estimate_s":
                route.get(
                    "dysci_outstanding_work_s"
                ),

            "dysci_estimator_source":
                route.get(
                    "dysci_estimator_source"
                ),

            "iluvatar_estimate_s":
                float(
                    route.get(
                        "iluvatar_estimate_s",
                        float("inf"),
                    )
                ),
        }


        return result


    except asyncio.CancelledError:

        logger.info(
            f"[DYSCI_CANCELLED] "
            f"request_id={request_id} "
            f"model={model_name} "
            f"node={node_id}"
        )

        raise


    except Exception as exc:

        logger.exception(
            f"[DYSCI_INVOKE_FAIL] "
            f"request_id={request_id} "
            f"model={model_name} "
            f"node={node_id} "
            f"error={exc}"
        )

        raise


    finally:

        # ====================================================
        # Always remove the request from outstanding state.
        #
        # Only successful requests are learned.
        # ====================================================

        completion = (
            DYSCI_ESTIMATOR.request_finished(
                request_id=request_id,
                learn=success,
            )
        )


        if completion is not None:

            history_ewma_s = (
                completion.get(
                    "history_ewma_s"
                )
            )

            if history_ewma_s is None:
                history_text = "None"
            else:
                history_text = (
                    f"{float(history_ewma_s):.6f}"
                )


            logger.info(
                f"[DYSCI_COMPLETE] "
                f"request_id={request_id} "
                f"model={model_name} "
                f"node={node_id} "
                f"device={device} "
                f"success={success} "
                f"actual_wall_s="
                f"{completion['actual_s']:.6f} "
                f"load_at_dispatch="
                f"{completion['load_at_dispatch']} "
                f"work_at_dispatch_s="
                f"{completion['work_at_dispatch_s']:.6f} "
                f"remaining_outstanding="
                f"{completion['remaining_outstanding']} "
                f"remaining_work_s="
                f"{completion['remaining_work_s']:.6f} "
                f"history_ewma_s="
                f"{history_text} "
                f"history_samples="
                f"{completion['history_samples']}"
            )

def _get_stage_threshold(stage_index: int) -> float:
    try:
        if isinstance(THRESHOLD_STAGE, list) and len(THRESHOLD_STAGE) > stage_index:
            return float(THRESHOLD_STAGE[stage_index])
    except Exception:
        pass

    return float(THRESHOLD)



async def execute_sequential_cascade(
    stub_map,
    routing_plan: Dict[str, dict],
    transaction_id: str,
    model_list: List[str],
    image_b64: str,
    start_time: float,
    deadline_ms: int,
    selected_theta: Optional[float] = None,
) -> Tuple[List[dict], str]:
    results: List[dict] = []
    stop_reason = "ran_all"

    for i, m in enumerate(model_list):
        t0 = time.perf_counter()

        logger.info(
            f"[RUN] tx={transaction_id} step={i + 1}/{len(model_list)} model={m} start=True"
        )

        # stub = get_stub_for_model(m, stub_map, routing_plan)
        #
        # try:
        #     res = await send_request(stub, m, image_b64, get_route_for_model(m, routing_plan))
        # except Exception as e:
        #     logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
        #     continue

        route = get_route_for_model(
            model_name=m,
            routing_plan=routing_plan,
        )

        stub = get_stub_for_model(
            model_name=m,
            stub_map=stub_map,
            routing_plan=routing_plan,
        )

        try:
            res = await send_request(
                stub=stub,
                model_name=m,
                image_b64=image_b64,
                route=route,
            )

        except Exception as error:
            logger.exception(
                f"[RUN_FAIL] tx={transaction_id} "
                f"model={m} "
                f"node={route.get('node_id')} "
                f"server={route.get('server_address')} "
                f"error_type={type(error).__name__} "
                f"error={error}"
            )
            continue

        wall_s = time.perf_counter() - t0

        # if not res.get("success", False):
        #     logger.warning(
        #         f"[RUN_BAD] tx={transaction_id} model={m} success=False wall_s={wall_s:.4f}"
        #     )
        #     continue

        if not res.get("success", False):
            logger.warning(
                f"[RUN_BAD] tx={transaction_id} "
                f"model={m} "
                f"node={route.get('node_id')} "
                f"server={route.get('server_address')} "
                f"success={res.get('success')} "
                f"container_state={res.get('container_state')} "
                f"wall_s={wall_s:.4f} "
                f"full_result={res}"
            )
            continue

        conf = float(res.get("probability", 0.0))
        label = res.get("label")

        # Use solver-selected theta if available.
        # Otherwise fall back to existing stage threshold logic.
        if selected_theta is not None:
            stage_thr = float(selected_theta)
        else:
            stage_thr = _get_stage_threshold(i)

        passed = conf >= stage_thr

        logger.info(
            f"[RUN_OK] tx={transaction_id} model={m} label={label} conf={conf:.4f} "
            f"rpc_latency_s={float(res.get('latency', -1)):.4f} wall_s={wall_s:.4f} "
            f"container_state={res.get('container_state')} server={res.get('server_address')}"
        )

        logger.info(
            f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
            f"threshold={stage_thr:.4f} pass={passed} "
            f"source={'seq_constraint_solver' if selected_theta is not None else 'config'}"
        )

        results.append(res)

        if passed:
            if i == 0:
                stop_reason = f"early_exit_first_pass={m}"
            else:
                stop_reason = f"early_exit_with_aggregation_at_{m}"

            logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
            break

        if i + 1 < len(model_list):
            next_model = model_list[i + 1]

            stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
                start_time=start_time,
                deadline_ms=deadline_ms,
                next_model=next_model,
            )

            logger.info(
                f"[DEADLINE_CHECK] tx={transaction_id} after_model={m} "
                f"next_model={next_model} elapsed_ms={elapsed_ms:.2f} "
                f"remaining_ms={remaining_ms:.2f} "
                f"next_est_ms={float(MODEL_PROFILES[next_model]['latency']) * 1000.0:.2f} "
                f"stop={stop_now}"
            )

            if stop_now:
                stop_reason = f"deadline_stop_before_{next_model}"
                logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
                break

    return results, stop_reason


# -----------------------------
# Full parallel / MODE-S execution
# -----------------------------
async def execute_mode_s_cascade(
    stub_map,
    routing_plan: Dict[str, dict],
    transaction_id: str,
    model_list: List[str],
    image_b64: str,
) -> Tuple[List[dict], str]:
    logger.info(f"[MODE_S_START] tx={transaction_id} models={model_list}")

    tasks = []
    task_models = []

    for m in model_list:
        stub = get_stub_for_model(m, stub_map, routing_plan)
        tasks.append(asyncio.create_task(send_request(stub, m, image_b64, get_route_for_model(m, routing_plan))))
        task_models.append(m)

    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: List[dict] = []

    for idx, out in enumerate(raw_results):
        m = task_models[idx]

        if isinstance(out, Exception):
            logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
            continue

        res = out

        if not res.get("success", False):
            logger.warning(f"[RUN_BAD] tx={transaction_id} model={m} success=False")
            continue

        logger.info(
            f"[RUN_OK] tx={transaction_id} model={m} "
            f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
            f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
            f"container_state={res.get('container_state')} server={res.get('server_address')}"
        )

        results.append(res)

    return results, "mode_s_executed"


# -----------------------------
# Parallel-first-finish cascade
# -----------------------------
# async def execute_parallel_first_finish_cascade(
#     stub_map,
#     routing_plan: Dict[str, dict],
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     parallel_first_n: int,
#     parallel_batch_size: int,
#     start_time: float,
#     deadline_ms: int,
#     selected_folder: str,
#     selected_theta: Optional[float] = None,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#     stop_reason = "ran_all"
#
#     first_n = max(0, min(int(parallel_first_n), len(model_list)))
#     batch_size = max(1, int(parallel_batch_size))
#
#     first_batch = model_list[:first_n]
#     rest_models = model_list[first_n:]
#
#     if first_n >= 2:
#         logger.info(
#             f"[PARALLEL_FIRST_FINISH] tx={transaction_id} "
#             f"first_batch={first_batch} rest={rest_models}"
#         )
#
#         task_map = {
#             asyncio.create_task(
#                 send_request(
#                     get_stub_for_model(m, stub_map, routing_plan),
#                     m,
#                     image_b64,
#                     get_route_for_model(m, routing_plan),
#                 )
#             ): m
#             for m in first_batch
#         }
#
#         done, pending = await asyncio.wait(
#             set(task_map.keys()),
#             return_when=asyncio.FIRST_COMPLETED,
#         )
#
#         first_task = next(iter(done))
#         first_model = task_map[first_task]
#
#         first_result = None
#         first_pass = False
#
#         try:
#             first_result = first_task.result()
#
#             if first_result.get("success", False):
#                 first_conf = float(first_result.get("probability", 0.0))
#                 # first_thr = _get_stage_threshold(0)
#                 first_thr = float(selected_theta) if selected_theta is not None else _get_stage_threshold(0)
#                 first_pass = first_conf >= first_thr
#
#                 logger.info(
#                     f"[RUN_OK] tx={transaction_id} first_finished_model={first_model} "
#                     f"label={first_result.get('label')} conf={first_conf:.4f} "
#                     f"rpc_latency_s={float(first_result.get('latency', -1)):.4f} "
#                     f"container_state={first_result.get('container_state')} "
#                     f"server={first_result.get('server_address')}"
#                 )
#
#                 logger.info(
#                     f"[THRESH] tx={transaction_id} first_finished_model={first_model} "
#                     f"conf={first_conf:.4f} threshold={first_thr:.2f} pass={first_pass}"
#                 )
#
#         except Exception as e:
#             logger.exception(
#                 f"[RUN_FAIL] tx={transaction_id} first_finished_model={first_model} error={e}"
#             )
#             first_result = None
#
#         # if first_result is not None and first_result.get("success", False) and first_pass:
#         #     for p in pending:
#         #         p.cancel()
#         #
#         #     for p in pending:
#         #         try:
#         #             await p
#         #         except Exception:
#         #             pass
#         #
#         #     results = [first_result]
#         #     stop_reason = f"early_exit_first_finished_pass={first_model}"
#         #
#         #     logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#         #
#         #     return results, stop_reason
#
#         if first_result is not None and first_result.get("success", False) and first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     logger.info(
#                         f"[RUN_CANCELLED] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')}"
#                     )
#                 except Exception as e:
#                     logger.exception(
#                         f"[CANCEL_WAIT_FAIL] tx={transaction_id} "
#                         f"model={task_map.get(p, 'unknown')} error={e}"
#                     )
#
#             results = [first_result]
#             stop_reason = f"early_exit_first_finished_pass={first_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             return results, stop_reason
#
#         remaining_results: List[dict] = []
#
#         for p in pending:
#             model_name = task_map[p]
#
#             try:
#                 res = await p
#
#                 if res.get("success", False):
#                     remaining_results.append(res)
#
#                     logger.info(
#                         f"[RUN_OK] tx={transaction_id} model={model_name} "
#                         f"label={res.get('label')} conf={float(res.get('probability', 0.0)):.4f} "
#                         f"rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                         f"container_state={res.get('container_state')} "
#                         f"server={res.get('server_address')}"
#                     )
#
#             except Exception as e:
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#         if first_result is not None and first_result.get("success", False):
#             results.append(first_result)
#
#         results.extend(remaining_results)
#
#         if len(results) >= 2:
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             # agg_thr = _get_stage_threshold(1)
#             agg_thr = float(selected_theta) if selected_theta is not None else _get_stage_threshold(1)
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_first_batch={[r['model'] for r in results]}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 return results, stop_reason
#
#     elif first_n == 1:
#         m = first_batch[0]
#
#         try:
#             stub = get_stub_for_model(m, stub_map, routing_plan)
#             r = await send_request(stub, m, image_b64, get_route_for_model(m, routing_plan))
#
#             if r.get("success", False):
#                 conf = float(r.get("probability", 0.0))
#                 thr = _get_stage_threshold(0)
#                 passed = conf >= thr
#
#                 results.append(r)
#
#                 if passed:
#                     stop_reason = f"early_exit_first_pass={m}"
#                     return results, stop_reason
#
#         except Exception as e:
#             logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={e}")
#
#     start_idx = first_n
#
#     for batch_start in range(0, len(rest_models), batch_size):
#         batch = rest_models[batch_start: batch_start + batch_size]
#
#         first_next_model = batch[0]
#
#         stop_now, elapsed_ms, remaining_ms = should_stop_before_next_model(
#             start_time=start_time,
#             deadline_ms=deadline_ms,
#             next_model=first_next_model,
#         )
#
#         logger.info(
#             f"[DEADLINE_CHECK] tx={transaction_id} next_batch={batch} "
#             f"elapsed_ms={elapsed_ms:.2f} remaining_ms={remaining_ms:.2f} "
#             f"first_next_est_ms={float(MODEL_PROFILES[first_next_model]['latency']) * 1000.0:.2f} "
#             f"stop={stop_now}"
#         )
#
#         if stop_now:
#             stop_reason = f"deadline_stop_before_{first_next_model}"
#             logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#             break
#
#         tasks = [
#             asyncio.create_task(
#                 send_request(
#                     get_stub_for_model(m, stub_map, routing_plan),
#                     m,
#                     image_b64,
#                     get_route_for_model(m, routing_plan),
#                 )
#             )
#             for m in batch
#         ]
#
#         batch_raw = await asyncio.gather(*tasks, return_exceptions=True)
#         successful_in_batch = False
#
#         for local_idx, out in enumerate(batch_raw):
#             m = batch[local_idx]
#             stage_index = start_idx + batch_start + local_idx
#
#             if isinstance(out, Exception):
#                 logger.exception(f"[RUN_FAIL] tx={transaction_id} model={m} error={out}")
#                 continue
#
#             res = out
#
#             if not res.get("success", False):
#                 continue
#
#             conf = float(res.get("probability", 0.0))
#             thr = _get_stage_threshold(stage_index)
#             passed = conf >= thr
#
#             logger.info(
#                 f"[RUN_OK] tx={transaction_id} model={m} label={res.get('label')} "
#                 f"conf={conf:.4f} rpc_latency_s={float(res.get('latency', -1)):.4f} "
#                 f"container_state={res.get('container_state')} server={res.get('server_address')}"
#             )
#
#             logger.info(
#                 f"[THRESH] tx={transaction_id} model={m} conf={conf:.4f} "
#                 f"threshold={thr:.2f} pass={passed}"
#             )
#
#             results.append(res)
#             successful_in_batch = True
#
#         if successful_in_batch and len(results) >= 2:
#             last_stage_idx = start_idx + batch_start + len(batch) - 1
#             agg_thr = _get_stage_threshold(last_stage_idx)
#
#             temp_final = combine_outputs(
#                 results,
#                 policy="weighted_majority",
#                 historical_acc=OMEGA,
#                 model_weights=MODEL_WEIGHTS,
#                 ground_truth=selected_folder,
#                 gamma=GAMMA,
#                 update_weights=False,
#                 label_matcher=wnid_matches_text_label,
#                 rho=RHO,
#             )
#
#             agg_conf = float(temp_final.get("accuracy", 0.0))
#             agg_pass = agg_conf >= agg_thr
#
#             logger.info(
#                 f"[AGG_CHECK] tx={transaction_id} models={[r['model'] for r in results]} "
#                 f"agg_label={temp_final.get('label')} agg_conf={agg_conf:.4f} "
#                 f"threshold={agg_thr:.2f} pass={agg_pass}"
#             )
#
#             if agg_pass:
#                 stop_reason = f"early_exit_aggregate_pass_after_batch={batch}"
#                 logger.info(f"[EARLY_EXIT] tx={transaction_id} stop_reason={stop_reason}")
#                 break
#
#     return results, stop_reason
#
# async def execute_speculative_cloud_cascade(
#     stub_map,
#     routing_plan: Dict[str, dict],
#     transaction_id: str,
#     model_list: List[str],
#     image_b64: str,
#     selected_theta: Optional[float] = None,
# ) -> Tuple[List[dict], str]:
#     results: List[dict] = []
#
#     if len(model_list) < 2:
#         raise ValueError("spec_b requires two models: [edge_model, cloud_model]")
#
#     theta = float(selected_theta) if selected_theta is not None else float(THRESHOLD)
#
#     edge_model = model_list[0]
#     cloud_model = model_list[1]
#
#     logger.info(
#         f"[SPEC_B_START] tx={transaction_id} edge={edge_model} "
#         f"cloud={cloud_model} theta={theta:.4f}"
#     )
#
#     task_map = {
#         asyncio.create_task(
#             send_request(
#                 get_stub_for_model(edge_model, stub_map, routing_plan),
#                 edge_model,
#                 image_b64,
#                 get_route_for_model(edge_model, routing_plan),
#             )
#         ): edge_model,
#         asyncio.create_task(
#             send_request(
#                 get_stub_for_model(cloud_model, stub_map, routing_plan),
#                 cloud_model,
#                 image_b64,
#                 get_route_for_model(cloud_model, routing_plan),
#             )
#         ): cloud_model,
#     }
#
#     done, pending = await asyncio.wait(
#         set(task_map.keys()),
#         return_when=asyncio.FIRST_COMPLETED,
#     )
#
#     first_task = next(iter(done))
#     first_model = task_map[first_task]
#
#     try:
#         first_result = first_task.result()
#     except Exception as e:
#         logger.exception(f"[SPEC_B_FAIL] tx={transaction_id} first_model={first_model} error={e}")
#         first_result = None
#
#     if first_result is not None and first_result.get("success", False):
#         first_conf = float(first_result.get("probability", 0.0))
#         first_pass = first_conf >= theta
#
#         logger.info(
#             f"[SPEC_B_FIRST] tx={transaction_id} model={first_model} "
#             f"conf={first_conf:.4f} theta={theta:.4f} pass={first_pass}"
#         )
#
#         if first_pass:
#             for p in pending:
#                 p.cancel()
#
#             for p in pending:
#                 try:
#                     await p
#                 except asyncio.CancelledError:
#                     pass
#                 except Exception:
#                     pass
#
#             return [first_result], f"spec_b_early_exit_first_pass={first_model}"
#
#         results.append(first_result)
#
#     for p in pending:
#         model_name = task_map[p]
#
#         try:
#             res = await p
#             if res.get("success", False):
#                 results.append(res)
#
#                 logger.info(
#                     f"[SPEC_B_SECOND] tx={transaction_id} model={model_name} "
#                     f"conf={float(res.get('probability', 0.0)):.4f}"
#                 )
#
#         except Exception as e:
#             logger.exception(f"[SPEC_B_FAIL] tx={transaction_id} model={model_name} error={e}")
#
#     if len(results) == 0:
#         return [], "spec_b_no_successful_models"
#
#     if len(results) == 1:
#         return results, f"spec_b_only_one_success={results[0].get('model')}"
#
#     return results, "spec_b_aggregate_after_both"



async def execute_parallel_first_finish_cascade(
    stub_map,
    routing_plan: Dict[str, dict],
    transaction_id: str,
    model_list: List[str],
    image_b64: str,
    parallel_first_n: int,
    parallel_batch_size: int,
    start_time: float,
    deadline_ms: int,
    selected_folder: str,
    selected_theta: Optional[float] = None,
) -> Tuple[List[dict], str]:

    del parallel_batch_size
    del start_time
    del deadline_ms
    del selected_folder

    if len(model_list) < 3:
        raise ValueError(
            "Edge speculation requires at least three models: "
            "[edge_model_1, edge_model_2, cloud_model]"
        )

    if int(parallel_first_n) < 2:
        raise ValueError(
            "Edge speculation requires parallel_first_n >= 2."
        )

    threshold = (
        float(selected_theta)
        if selected_theta is not None
        else float(THRESHOLD)
    )

    edge_models = model_list[:2]
    cloud_model = model_list[2]

    logger.info(
        f"[EDGE_SPEC_START] tx={transaction_id} "
        f"edge_models={edge_models} "
        f"cloud_model={cloud_model} "
        f"threshold={threshold:.4f}"
    )

    task_map = {}

    for model_name in edge_models:
        route = get_route_for_model(
            model_name=model_name,
            routing_plan=routing_plan,
        )

        stub = get_stub_for_model(
            model_name=model_name,
            stub_map=stub_map,
            routing_plan=routing_plan,
        )

        task = asyncio.create_task(
            send_request(
                stub=stub,
                model_name=model_name,
                image_b64=image_b64,
                route=route,
            )
        )

        task_map[task] = model_name

    done, pending = await asyncio.wait(
        set(task_map.keys()),
        return_when=asyncio.FIRST_COMPLETED,
    )

    first_task = next(iter(done))
    first_model = task_map[first_task]

    first_result = None

    try:
        first_result = await first_task

    except Exception as error:
        logger.exception(
            f"[EDGE_SPEC_FAIL] tx={transaction_id} "
            f"model={first_model} error={error}"
        )

    if (
        first_result is not None
        and first_result.get("success", False)
    ):
        first_confidence = float(
            first_result.get("probability", 0.0)
        )

        logger.info(
            f"[EDGE_SPEC_FIRST] tx={transaction_id} "
            f"model={first_model} "
            f"node={first_result.get('node_id')} "
            f"confidence={first_confidence:.4f} "
            f"threshold={threshold:.4f}"
        )

        if first_confidence >= threshold:
            remaining_tasks = (
                set(done) | set(pending)
            ) - {first_task}

            for task in remaining_tasks:
                task.cancel()

            for task in remaining_tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(
                        f"[EDGE_SPEC_CANCEL] tx={transaction_id} "
                        f"model={task_map[task]}"
                    )
                except Exception:
                    pass

            return (
                [first_result],
                f"edge_spec_first_pass={first_model}",
            )

    # The first response did not pass.
    # Wait for the other edge model.
    remaining_tasks = (
        set(done) | set(pending)
    ) - {first_task}

    second_result = None
    second_model = None

    for task in remaining_tasks:
        second_model = task_map[task]

        try:
            second_result = await task

        except Exception as error:
            logger.exception(
                f"[EDGE_SPEC_FAIL] tx={transaction_id} "
                f"model={second_model} error={error}"
            )

        break

    if (
        second_result is not None
        and second_result.get("success", False)
    ):
        second_confidence = float(
            second_result.get("probability", 0.0)
        )

        logger.info(
            f"[EDGE_SPEC_SECOND] tx={transaction_id} "
            f"model={second_model} "
            f"node={second_result.get('node_id')} "
            f"confidence={second_confidence:.4f} "
            f"threshold={threshold:.4f}"
        )

        if second_confidence >= threshold:
            return (
                [second_result],
                f"edge_spec_second_pass={second_model}",
            )

    # Both edge models failed the confidence threshold.
    logger.info(
        f"[EDGE_SPEC_CLOUD] tx={transaction_id} "
        f"reason=both_edge_models_below_threshold "
        f"cloud_model={cloud_model}"
    )

    cloud_route = get_route_for_model(
        model_name=cloud_model,
        routing_plan=routing_plan,
    )

    cloud_stub = get_stub_for_model(
        model_name=cloud_model,
        stub_map=stub_map,
        routing_plan=routing_plan,
    )

    try:
        cloud_result = await send_request(
            stub=cloud_stub,
            model_name=cloud_model,
            image_b64=image_b64,
            route=cloud_route,
        )

    except Exception as error:
        logger.exception(
            f"[EDGE_SPEC_CLOUD_FAIL] tx={transaction_id} "
            f"model={cloud_model} error={error}"
        )

        fallback_results = []

        if (
            first_result is not None
            and first_result.get("success", False)
        ):
            fallback_results.append(first_result)

        if (
            second_result is not None
            and second_result.get("success", False)
        ):
            fallback_results.append(second_result)

        return (
            fallback_results,
            "edge_spec_cloud_failed",
        )

    if not cloud_result.get("success", False):
        fallback_results = []

        if (
            first_result is not None
            and first_result.get("success", False)
        ):
            fallback_results.append(first_result)

        if (
            second_result is not None
            and second_result.get("success", False)
        ):
            fallback_results.append(second_result)

        return (
            fallback_results,
            "edge_spec_cloud_unsuccessful",
        )

    return (
        [cloud_result],
        f"edge_spec_cloud_return={cloud_model}",
    )

# async def main_ensemble_invoke(
#     transaction_id: str,
#     deadline: int,
#     image_b64: str,
#     selected_folder: Optional[str] = None,
# ) -> dict:
#     start_time = time.perf_counter()
#
#     if not image_b64:
#         raise ValueError("image_b64 is required")
#
#     if selected_folder is None:
#         selected_folder = "unknown"
#
#     now = time.perf_counter()
#     record_request(now, QPS_WINDOW_SEC)
#     qps_est = get_qps(QPS_WINDOW_SEC)
#
#     if str(ALPHA_MODE).lower() == "static":
#         alpha_now = float(ALPHA_STATIC)
#     else:
#         alpha_now = compute_adaptive_alpha(
#             qps=qps_est,
#             alpha_min=ALPHA_MIN,
#             alpha_max=ALPHA_MAX,
#             qps_low=QPS_LOW,
#             qps_high=QPS_HIGH,
#         )
#
#     logger.info(
#         f"[REQ_START] tx={transaction_id} deadline_ms={deadline} qps={qps_est:.2f} "
#         f"alpha={alpha_now:.3f} alpha_mode={ALPHA_MODE} policy={POLICY} "
#         f"ensemble_size={ENSEMBLE_SIZE} selected_folder={selected_folder} "
#         f"cascade_mode={CASCADE_MODE} baseline_mode={BASELINE_MODE} "
#         f"parallel_first_n={PARALLEL_FIRST_N} parallel_batch_size={PARALLEL_BATCH_SIZE} "
#         f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES} "
#         f"use_multi_instances={USE_MULTI_ILUVATAR_INSTANCES}"
#     )
#
#     wait_results, all_model_routes, node_wait_results = (
#         get_wait_results_for_models(M_TOTAL)
#     )
#
#     selected_theta = None
#     static_config_details = {}
#     constraint_solver_details = {}
#     risk_details = {}
#
#     parallel_first_n_runtime = PARALLEL_FIRST_N
#
#     # ------------------------------------------------------------
#     # 1. Static optimal config from  table
#     # ------------------------------------------------------------
#     if USE_STATIC_OPTIMAL_CONFIG:
#         cfg = get_optimal_config(
#             cost=STATIC_COST,
#             error=STATIC_ERROR,
#             policy=STATIC_POLICY,
#         )
#
#         M_D = list(cfg["models"])
#         selected_theta = float(cfg["theta"])
#         parallel_first_n_runtime = int(cfg.get("parallel_first_n", PARALLEL_FIRST_N))
#
#         if STATIC_POLICY == "sequential":
#             effective_cascade_mode = "sequential"
#         elif STATIC_POLICY == "po2":
#             effective_cascade_mode = "parallel_first_finish"
#         elif STATIC_POLICY == "spec_b":
#             effective_cascade_mode = "spec_b"
#         else:
#             raise ValueError(f"Unknown STATIC_POLICY={STATIC_POLICY}")
#
#         total_estimates = {
#             m: wait_results.get(m, float("inf"))
#             for m in M_D
#         }
#
#         static_config_details = {
#             "enabled": True,
#             "cost": STATIC_COST,
#             "error": STATIC_ERROR,
#             "target_accuracy": 1.0 - float(STATIC_ERROR),
#             "policy": STATIC_POLICY,
#             "models": M_D,
#             "theta": selected_theta,
#             "parallel_first_n": parallel_first_n_runtime,
#         }
#
#         logger.info(
#             f"[STATIC_OPTIMAL_CONFIG] tx={transaction_id} "
#             f"C={STATIC_COST} E={STATIC_ERROR} policy={STATIC_POLICY} "
#             f"models={M_D} theta={selected_theta} "
#             f"parallel_first_n={parallel_first_n_runtime} "
#             f"effective_mode={effective_cascade_mode}"
#         )
#
#     # ------------------------------------------------------------
#     # 2. Normal MODE-S selection
#     # ------------------------------------------------------------
#     else:
#         M_D, total_estimates, effective_cascade_mode = build_model_set_by_mode(
#             cascade_mode=CASCADE_MODE,
#             wait_results=wait_results,
#             deadline_ms=deadline,
#             ensemble_size=ENSEMBLE_SIZE,
#             alpha_now=alpha_now,
#             parallel_first_n=PARALLEL_FIRST_N,
#             network_latency=NETWORK_LATENCY,
#             logger=logger,
#         )
#
#     # ------------------------------------------------------------
#     # 3. Empty selection fallback
#     # ------------------------------------------------------------
#     if not M_D:
#         fastest_model = min(
#             MODEL_PROFILES,
#             key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#         )
#
#         M_D = [fastest_model]
#         total_estimates = {
#             fastest_model: wait_results.get(fastest_model, float("inf"))
#         }
#         effective_cascade_mode = "sequential"
#         selected_theta = None
#
#         logger.warning(
#             f"[SELECT_EMPTY] tx={transaction_id} fallback_fastest={fastest_model}"
#         )
#
#     # ------------------------------------------------------------
#     # 4. Constraint solver only when static config is disabled
#     # ------------------------------------------------------------
#     if (not USE_STATIC_OPTIMAL_CONFIG) and USE_CONSTRAINT_SOLVER:
#         constraint_solver = ConstraintSolver(
#             seq_csv=CONSTRAINT_SEQ_PROFILE_CSV,
#             power2_csv=CONSTRAINT_POWER2_PROFILE_CSV,
#             full_parallel_csv=CONSTRAINT_FULL_PARALLEL_PROFILE_CSV,
#         )
#
#         constraint_result = constraint_solver.solve(
#             max_cost=CONSTRAINT_MAX_COST,
#             target_accuracy=CONSTRAINT_TARGET_ACCURACY,
#             policy_order=CONSTRAINT_POLICY_ORDER,
#             time_mode=CONSTRAINT_SOLVER_TIME_MODE,
#             wait_results=wait_results,
#             model_profiles=MODEL_PROFILES,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#             base_request_overhead=BASE_REQUEST_OVERHEAD,
#             sequential_stage_overhead=SEQUENTIAL_STAGE_OVERHEAD,
#             po2_first_batch_overhead=PO2_FIRST_BATCH_OVERHEAD,
#             po2_tail_overhead=PO2_TAIL_OVERHEAD,
#             full_parallel_agg_overhead=FULL_PARALLEL_AGG_OVERHEAD,
#             et_scale_seq=ET_SCALE_SEQ,
#             et_offset_seq=ET_OFFSET_SEQ,
#             et_scale_p2=ET_SCALE_P2,
#             et_offset_p2=ET_OFFSET_P2,
#         )
#
#         constraint_solver_details = {
#             "success": constraint_result.success,
#             "policy": constraint_result.policy,
#             "reason": constraint_result.reason,
#             "feasible_count": constraint_result.feasible_count,
#             "total_count": constraint_result.total_count,
#             "solver_latency_ms": constraint_result.solver_latency_ms,
#             "selected_config": constraint_result.selected_config,
#             "time_mode": CONSTRAINT_SOLVER_TIME_MODE,
#             "max_cost": CONSTRAINT_MAX_COST,
#             "target_accuracy": CONSTRAINT_TARGET_ACCURACY,
#         }
#
#         if constraint_result.success:
#             selected = constraint_result.selected_config
#             selected_policy = selected["policy"]
#
#             M_D = list(selected["models"])
#
#             total_estimates = {
#                 m: wait_results.get(m, float("inf"))
#                 for m in M_D
#             }
#
#             if selected_policy == "sequential":
#                 effective_cascade_mode = "sequential"
#                 selected_theta = float(selected["theta"])
#
#             elif selected_policy in ("power2", "po2"):
#                 effective_cascade_mode = "parallel_first_finish"
#                 selected_theta = float(selected.get("theta", THRESHOLD))
#                 parallel_first_n_runtime = int(
#                     selected.get("parallel_first_n", PARALLEL_FIRST_N)
#                 )
#
#             elif selected_policy == "full_parallel":
#                 effective_cascade_mode = "full_parallel"
#                 selected_theta = None
#
#             else:
#                 effective_cascade_mode = "sequential"
#                 selected_theta = None
#
#             logger.info(
#                 f"[CONSTRAINT_SOLVER] tx={transaction_id} "
#                 f"policy={selected_policy} models={M_D} "
#                 f"theta={selected_theta} "
#                 f"parallel_first_n={parallel_first_n_runtime}"
#             )
#
#         else:
#             fastest_model = min(
#                 MODEL_PROFILES,
#                 key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#             )
#
#             M_D = [fastest_model]
#             total_estimates = {
#                 fastest_model: wait_results.get(fastest_model, float("inf"))
#             }
#             effective_cascade_mode = "sequential"
#             selected_theta = None
#
#             logger.warning(
#                 f"[CONSTRAINT_SOLVER_FALLBACK] tx={transaction_id} "
#                 f"reason={constraint_result.reason} fallback_fastest={fastest_model}"
#             )
#
#     # ------------------------------------------------------------
#     # 5. Risk selection only when static and constraint solver are disabled
#     # ------------------------------------------------------------
#     elif (
#         (not USE_STATIC_OPTIMAL_CONFIG)
#         and (not USE_CONSTRAINT_SOLVER)
#         and ENABLE_RISK_STRATEGY_SELECTION
#         and len(M_D) > 1
#     ):
#         risk_selected_mode, risk_details = select_min_risk_strategy(
#             selected_models=M_D,
#             wait_results=wait_results,
#             deadline_ms=deadline,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=PARALLEL_FIRST_N,
#             model_profiles=MODEL_PROFILES,
#             omega=OMEGA,
#             small_instance_models=SMALL_INSTANCE_MODELS,
#             large_instance_models=LARGE_INSTANCE_MODELS,
#             hardware_cost_rate=HARDWARE_COST_RATE,
#             estimate_latency_fn=estimate_selected_set_latency_for_mode,
#             alpha_latency=RISK_ALPHA_LATENCY,
#             beta_cost=RISK_BETA_COST,
#             gamma_error=RISK_GAMMA_ERROR,
#         )
#
#         effective_cascade_mode = risk_selected_mode
#
#         logger.info(
#             f"[RISK_SELECT] tx={transaction_id} "
#             f"selected_mode={risk_selected_mode} details={risk_details}"
#         )
#
#     # ------------------------------------------------------------
#     # 6. Estimate selected latency
#     # ------------------------------------------------------------
#     raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#         cascade_mode=effective_cascade_mode,
#         selected_models=M_D,
#         wait_results=wait_results,
#         network_latency=NETWORK_LATENCY,
#         parallel_first_n=parallel_first_n_runtime,
#     )
#
#     selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#
#     deadline_sec = float(deadline) / 1000.0 if deadline is not None else 0.0
#     deadline_fast_path_triggered = False
#
#     if (
#         USE_DEADLINE_FAST_PATH
#         and deadline_sec > 0.0
#         and selected_est_latency >= deadline_sec
#     ):
#         fastest_model = min(
#             MODEL_PROFILES,
#             key=lambda m: float(MODEL_PROFILES[m]["latency"]),
#         )
#
#         logger.warning(
#             f"[DEADLINE_FAST_PATH] tx={transaction_id} "
#             f"mode={effective_cascade_mode} selected_models={M_D} "
#             f"selected_est_latency={selected_est_latency:.4f}s "
#             f"deadline={deadline_sec:.4f}s fallback_fastest={fastest_model}"
#         )
#
#         M_D = [fastest_model]
#         total_estimates = {
#             fastest_model: wait_results.get(fastest_model, float("inf"))
#         }
#         effective_cascade_mode = "sequential"
#         selected_theta = None
#
#         raw_selected_est_latency = estimate_selected_set_latency_for_mode(
#             cascade_mode=effective_cascade_mode,
#             selected_models=M_D,
#             wait_results=wait_results,
#             network_latency=NETWORK_LATENCY,
#             parallel_first_n=parallel_first_n_runtime,
#         )
#
#         selected_est_latency = apply_latency_correction(raw_selected_est_latency)
#         deadline_fast_path_triggered = True
#
#     logger.info(f"[SELECT] tx={transaction_id} selected_models={M_D}")
#
#     logger.info(
#         f"[EFFECTIVE_MODE] tx={transaction_id} baseline_mode={BASELINE_MODE} "
#         f"configured_cascade_mode={CASCADE_MODE} "
#         f"effective_cascade_mode={effective_cascade_mode}"
#     )
#
#     logger.info(
#         f"[SELECT_WAITS] tx={transaction_id} waits_selected="
#         f"{ {m: float(total_estimates.get(m, -1)) for m in M_D} }"
#     )
#
#     routing_plan = build_routing_plan(
#         selected_models=M_D,
#         model_routes=all_model_routes,
#     )
#
#     logger.info(
#         f"[SELECT_ROUTE] tx={transaction_id} routes={routing_plan}"
#     )
#
#     channel_map, stub_map = build_stub_map_for_selected_models(
#         model_list=M_D,
#         routing_plan=routing_plan,
#     )
#
#     try:
#         mode = str(effective_cascade_mode).lower()
#
#         if mode == "sequential":
#             results, stop_reason = await execute_sequential_cascade(
#                 stub_map=stub_map,
#                 routing_plan=routing_plan,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_theta=selected_theta,
#             )
#
#         elif mode in ("mode_s", "full_parallel"):
#             results, stop_reason = await execute_mode_s_cascade(
#                 stub_map=stub_map,
#                 routing_plan=routing_plan,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#             )
#
#         elif mode == "spec_b":
#             results, stop_reason = await execute_speculative_cloud_cascade(
#                 stub_map=stub_map,
#                 routing_plan=routing_plan,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 selected_theta=selected_theta,
#             )
#
#         else:
#             results, stop_reason = await execute_parallel_first_finish_cascade(
#                 stub_map=stub_map,
#                 routing_plan=routing_plan,
#                 transaction_id=transaction_id,
#                 model_list=M_D,
#                 image_b64=image_b64,
#                 parallel_first_n=parallel_first_n_runtime,
#                 parallel_batch_size=PARALLEL_BATCH_SIZE,
#                 start_time=start_time,
#                 deadline_ms=deadline,
#                 selected_folder=selected_folder,
#                 selected_theta=selected_theta,
#             )
#
#     finally:
#         await close_channels(channel_map)
#
#     end_time = time.perf_counter()
#     e2e_ms = (end_time - start_time) * 1000.0
#     actual_latency_s = e2e_ms / 1000.0
#
#     update_latency_overhead(
#         actual_latency_s=actual_latency_s,
#         estimated_latency_s=raw_selected_est_latency,
#     )
#
#     per_model = {}
#
#     for res in results:
#         model_name = res.get("model")
#         if not model_name:
#             continue
#
#         per_model[model_name] = {
#             "label": res.get("label"),
#             "probability": float(res.get("probability", 0.0)),
#             "latency_s": float(res.get("latency", -1)),
#             "success": bool(res.get("success", False)),
#             "state": res.get("container_state", "UNKNOWN"),
#             "node_id": res.get("node_id"),
#             "node_type": res.get("node_type"),
#             "server_address": res.get("server_address"),
#             "estimated_wait_s": float(
#                 res.get("estimated_wait_s", float("inf"))
#             ),
#         }
#
#     base_return = {
#         "selected_models": M_D,
#         "per_model": per_model,
#         "wait_times": total_estimates,
#         "selected_folder": selected_folder,
#         "ensemble_size": ENSEMBLE_SIZE,
#         "threshold": THRESHOLD,
#         "threshold_stage": THRESHOLD_STAGE,
#         "selected_theta": selected_theta,
#         "parallel_first_n": parallel_first_n_runtime,
#         "parallel_batch_size": PARALLEL_BATCH_SIZE,
#         "stop_reason": stop_reason,
#         "main_policy": POLICY,
#         "alpha": alpha_now,
#         "alpha_mode": ALPHA_MODE,
#         "qps_est": qps_est,
#         "baseline_mode": BASELINE_MODE,
#         "configured_cascade_mode": CASCADE_MODE,
#         "effective_cascade_mode": effective_cascade_mode,
#         "cascade_mode": CASCADE_MODE,
#         "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
#         "use_multi_instances": USE_MULTI_ILUVATAR_INSTANCES,
#         "routing_plan": routing_plan,
#         "node_wait_results": node_wait_results,
#         "deadline_fast_path_enabled": USE_DEADLINE_FAST_PATH,
#         "deadline_fast_path_triggered": deadline_fast_path_triggered,
#         "selected_est_latency_s": selected_est_latency,
#         "raw_selected_est_latency_s": raw_selected_est_latency,
#         "recent_latency_overhead_s": get_recent_latency_overhead(),
#         "log_file": LOG_PATH,
#         "risk_details": risk_details,
#         "risk_selected_mode": effective_cascade_mode,
#         "constraint_solver_details": constraint_solver_details,
#         "static_config_details": static_config_details,
#         "static_cost": STATIC_COST if USE_STATIC_OPTIMAL_CONFIG else None,
#         "static_error": STATIC_ERROR if USE_STATIC_OPTIMAL_CONFIG else None,
#         "static_policy": STATIC_POLICY if USE_STATIC_OPTIMAL_CONFIG else None,
#     }
#
#     if not results:
#         logger.error(
#             f"[REQ_END] tx={transaction_id} success=False "
#             f"reason=no_successful_models e2e_ms={e2e_ms:.2f}"
#         )
#
#         return {
#             **base_return,
#             "executed_models": [],
#             "label": -1,
#             "accuracy": 0.0,
#             "combiner_policy": "FAILED",
#             "e2e_time_ms": e2e_ms,
#             "success": False,
#         }
#
#     if len(results) == 1:
#         single = results[0]
#         final_result = {
#             "label": single.get("label"),
#             "accuracy": float(single.get("probability", 0.0)),
#             "combiner_policy": "single_model",
#             "success": True,
#         }
#
#     else:
#         final_result = combine_outputs(
#             results,
#             policy="weighted_majority",
#             historical_acc=OMEGA,
#             model_weights=MODEL_WEIGHTS,
#             ground_truth=selected_folder,
#             gamma=GAMMA,
#             update_weights=True,
#             label_matcher=wnid_matches_text_label,
#             rho=RHO,
#         )
#
#     save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)
#
#     for res in results:
#         try:
#             OMEGA[res["model"]] = (
#                 (1 - ETA) * OMEGA[res["model"]]
#                 + ETA * float(res.get("probability", 0.0))
#             )
#             active_models.add(res["model"])
#         except Exception:
#             pass
#
#     logger.info(
#         f"[REQ_END] tx={transaction_id} "
#         f"success={final_result.get('success', True)} "
#         f"stop_reason={stop_reason} e2e_ms={e2e_ms:.2f} "
#         f"alpha={alpha_now:.3f} "
#         f"configured_cascade_mode={CASCADE_MODE} "
#         f"effective_cascade_mode={effective_cascade_mode} "
#         f"baseline_mode={BASELINE_MODE}"
#     )
#
#     return {
#         **base_return,
#         "executed_models": [r["model"] for r in results],
#         "label": final_result["label"],
#         "accuracy": final_result["accuracy"],
#         "combiner_policy": final_result["combiner_policy"],
#         "e2e_time_ms": e2e_ms,
#         "success": final_result.get("success", True),
#     }

async def execute_speculative_cloud_cascade(
    stub_map,
    routing_plan: Dict[str, dict],
    transaction_id: str,
    model_list: List[str],
    image_b64: str,
    selected_theta: Optional[float] = None,
) -> Tuple[List[dict], str]:

    # --------------------------------------------------------
    # Validate models
    # --------------------------------------------------------

    if not model_list:
        return [], "no_models"

    # --------------------------------------------------------
    # Single-model fallback
    # --------------------------------------------------------

    if len(model_list) == 1:

        model_name = model_list[0]

        stub = get_stub_for_model(
            model_name,
            stub_map,
            routing_plan,
        )

        route = get_route_for_model(
            model_name,
            routing_plan,
        )

        try:

            result = await send_request(
                stub,
                model_name,
                image_b64,
                route,
            )

        except Exception as e:

            logger.exception(
                f"[SPEC_B_SINGLE_FAIL] "
                f"tx={transaction_id} "
                f"model={model_name} "
                f"error={e}"
            )

            return [], "single_model_failed"

        if not result.get("success", False):
            return [], "single_model_failed"

        return [
            result
        ], f"single_model={model_name}"

    # --------------------------------------------------------
    # Cloud speculation assumes two models:
    #
    # model_list[0] = first model
    # model_list[1] = second model
    #
    # Usually:
    # edge model + cloud model
    # --------------------------------------------------------

    model_a = model_list[0]
    model_b = model_list[1]

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    if selected_theta is not None:
        theta = float(selected_theta)
    else:
        theta = float(THRESHOLD)

    # --------------------------------------------------------
    # Get routes
    # --------------------------------------------------------

    route_a = get_route_for_model(
        model_a,
        routing_plan,
    )

    route_b = get_route_for_model(
        model_b,
        routing_plan,
    )

    # --------------------------------------------------------
    # Get stubs
    # --------------------------------------------------------

    stub_a = get_stub_for_model(
        model_a,
        stub_map,
        routing_plan,
    )

    stub_b = get_stub_for_model(
        model_b,
        stub_map,
        routing_plan,
    )

    logger.info(
        f"[SPEC_B_START] "
        f"tx={transaction_id} "
        f"model_a={model_a} "
        f"node_a={route_a.get('node_id')} "
        f"model_b={model_b} "
        f"node_b={route_b.get('node_id')} "
        f"theta={theta:.4f}"
    )

    # --------------------------------------------------------
    # Start BOTH models in parallel
    # --------------------------------------------------------

    task_a = asyncio.create_task(
        send_request(
            stub_a,
            model_a,
            image_b64,
            route_a,
        )
    )

    task_b = asyncio.create_task(
        send_request(
            stub_b,
            model_b,
            image_b64,
            route_b,
        )
    )

    # Map task -> model for logging
    task_to_model = {
        task_a: model_a,
        task_b: model_b,
    }

    # --------------------------------------------------------
    # Wait for FIRST completed model
    # --------------------------------------------------------

    done, pending = await asyncio.wait(
        {
            task_a,
            task_b,
        },
        return_when=asyncio.FIRST_COMPLETED,
    )

    # Normally one task is finished.
    # If both complete almost simultaneously, done may contain two.
    first_task = next(iter(done))

    first_model = task_to_model[
        first_task
    ]

    try:

        first_result = await first_task

    except Exception as e:

        logger.exception(
            f"[SPEC_B_FIRST_FAIL] "
            f"tx={transaction_id} "
            f"model={first_model} "
            f"error={e}"
        )

        first_result = None

    # --------------------------------------------------------
    # First result succeeded
    # --------------------------------------------------------

    if (
        first_result is not None
        and first_result.get(
            "success",
            False,
        )
    ):

        first_conf = float(
            first_result.get(
                "probability",
                0.0,
            )
        )

        logger.info(
            f"[SPEC_B_FIRST_RESULT] "
            f"tx={transaction_id} "
            f"model={first_model} "
            f"node={first_result.get('node_id')} "
            f"confidence={first_conf:.4f} "
            f"theta={theta:.4f}"
        )

        # ----------------------------------------------------
        # FIRST result passes threshold
        # ----------------------------------------------------

        if first_conf >= theta:

            logger.info(
                f"[SPEC_B_FIRST_ACCEPT] "
                f"tx={transaction_id} "
                f"model={first_model} "
                f"confidence={first_conf:.4f}"
            )

            # Cancel remaining local task
            for task in pending:

                if not task.done():
                    task.cancel()

            # Clean up cancelled tasks
            for task in pending:

                try:
                    await task

                except asyncio.CancelledError:
                    pass

                except Exception:
                    pass

            return [
                first_result
            ], f"early_exit_first_finish={first_model}"

    # --------------------------------------------------------
    # First result failed confidence or execution.
    #
    # Now wait for the other model.
    # --------------------------------------------------------

    if pending:

        second_task = next(
            iter(pending)
        )

    else:

        # Both may have finished at nearly the same time.
        remaining_done = [
            task
            for task in done
            if task is not first_task
        ]

        if not remaining_done:

            # No second result available
            if (
                first_result is not None
                and first_result.get(
                    "success",
                    False,
                )
            ):

                return [
                    first_result
                ], f"only_first_available={first_model}"

            return [], "spec_b_no_second_result"

        second_task = remaining_done[0]

    second_model = task_to_model[
        second_task
    ]

    logger.info(
        f"[SPEC_B_WAIT_SECOND] "
        f"tx={transaction_id} "
        f"first_model={first_model} "
        f"second_model={second_model}"
    )

    try:

        second_result = await second_task

    except asyncio.CancelledError:

        second_result = None

    except Exception as e:

        logger.exception(
            f"[SPEC_B_SECOND_FAIL] "
            f"tx={transaction_id} "
            f"model={second_model} "
            f"error={e}"
        )

        second_result = None

    # --------------------------------------------------------
    # Second model succeeded
    # --------------------------------------------------------

    if (
        second_result is not None
        and second_result.get(
            "success",
            False,
        )
    ):

        second_conf = float(
            second_result.get(
                "probability",
                0.0,
            )
        )

        logger.info(
            f"[SPEC_B_SECOND_RESULT] "
            f"tx={transaction_id} "
            f"model={second_model} "
            f"node={second_result.get('node_id')} "
            f"confidence={second_conf:.4f} "
            f"theta={theta:.4f}"
        )

        # ----------------------------------------------------
        # For a two-model speculative cascade,
        # second result is the final fallback.
        # ----------------------------------------------------

        return [
            second_result
        ], f"second_finish_used={second_model}"

    # --------------------------------------------------------
    # Second failed.
    # Fall back to first successful result if available.
    # --------------------------------------------------------

    if (
        first_result is not None
        and first_result.get(
            "success",
            False,
        )
    ):

        logger.warning(
            f"[SPEC_B_SECOND_FAILED_USE_FIRST] "
            f"tx={transaction_id} "
            f"first_model={first_model}"
        )

        return [
            first_result
        ], f"second_failed_use_first={first_model}"

    # --------------------------------------------------------
    # Both failed
    # --------------------------------------------------------

    logger.error(
        f"[SPEC_B_ALL_FAILED] "
        f"tx={transaction_id} "
        f"model_a={model_a} "
        f"model_b={model_b}"
    )

    return [], "spec_b_all_failed"

async def main_ensemble_invoke(
    transaction_id: str,
    deadline: int,
    image_b64: str,
    selected_folder: Optional[str] = None,
) -> dict:
    request_start_time = time.perf_counter()

    if not image_b64:
        raise ValueError("image_b64 is required")

    if selected_folder is None:
        selected_folder = "unknown"

    # =========================================================
    # QPS and adaptive alpha
    # =========================================================

    now = time.perf_counter()
    record_request(now, QPS_WINDOW_SEC)
    qps_est = get_qps(QPS_WINDOW_SEC)

    if str(ALPHA_MODE).lower() == "static":
        alpha_now = float(ALPHA_STATIC)
    else:
        alpha_now = compute_adaptive_alpha(
            qps=qps_est,
            alpha_min=ALPHA_MIN,
            alpha_max=ALPHA_MAX,
            qps_low=QPS_LOW,
            qps_high=QPS_HIGH,
        )

    logger.info(
        f"[REQ_START] tx={transaction_id} "
        f"deadline_ms={deadline} "
        f"qps={qps_est:.2f} "
        f"alpha={alpha_now:.3f} "
        f"alpha_mode={ALPHA_MODE} "
        f"policy={POLICY} "
        f"ensemble_size={ENSEMBLE_SIZE} "
        f"selected_folder={selected_folder} "
        f"cascade_mode={CASCADE_MODE} "
        f"baseline_mode={BASELINE_MODE} "
        f"parallel_first_n={PARALLEL_FIRST_N} "
        f"parallel_batch_size={PARALLEL_BATCH_SIZE} "
        f"use_two_instances={USE_TWO_ILUVATAR_INSTANCES} "
        f"use_multi_instances={USE_MULTI_ILUVATAR_INSTANCES}"
    )

    selected_theta = None
    static_config_details = {}
    constraint_solver_details = {}
    risk_details = {}

    parallel_first_n_runtime = PARALLEL_FIRST_N
    effective_cascade_mode = str(CASCADE_MODE).lower()

    M_D: List[str] = []
    total_estimates: Dict[str, float] = {}

    wait_results: Dict[str, float] = {}
    all_model_routes: Dict[str, dict] = {}
    node_wait_results: Dict[str, Dict[str, float]] = {}

    wait_collection_ms = 0.0

    # =========================================================
    # Multi-node deployment information
    # =========================================================

    active_edge_nodes = [
        node_id
        for node_id, node_config in ILUVATAR_NODES.items()
        if node_config.get("enabled", True)
        and str(node_config.get("type", "")).lower() == "edge"
    ]

    active_cloud_nodes = [
        node_id
        for node_id, node_config in ILUVATAR_NODES.items()
        if node_config.get("enabled", True)
        and str(node_config.get("type", "")).lower() == "cloud"
    ]

    deployment_config = (
        f"{len(active_edge_nodes)}E+"
        f"{len(active_cloud_nodes)}C"
    )

    logger.info(
        f"[DEPLOYMENT] tx={transaction_id} "
        f"config={deployment_config} "
        f"edge_nodes={active_edge_nodes} "
        f"cloud_nodes={active_cloud_nodes}"
    )

    # =========================================================
    # 1. Static optimal configuration
    #
    # In static mode, models are already known before wait-time
    # collection. Therefore query only the selected models.
    # =========================================================

    if USE_STATIC_OPTIMAL_CONFIG:
        cfg = get_optimal_config(
            cost=STATIC_COST,
            error=STATIC_ERROR,
            policy=STATIC_POLICY,
        )

        M_D = list(cfg["models"])
        selected_theta = float(cfg["theta"])

        parallel_first_n_runtime = int(
            cfg.get(
                "parallel_first_n",
                PARALLEL_FIRST_N,
            )
        )

        static_policy = str(STATIC_POLICY).lower()

        if static_policy == "sequential":
            effective_cascade_mode = "sequential"

        elif static_policy in ("po2", "power2"):
            effective_cascade_mode = "parallel_first_finish"

        elif static_policy == "spec_b":
            effective_cascade_mode = "spec_b"

        elif static_policy == "full_parallel":
            effective_cascade_mode = "full_parallel"

        else:
            raise ValueError(
                f"Unknown STATIC_POLICY={STATIC_POLICY}"
            )

        logger.info(
            f"[STATIC_OPTIMAL_CONFIG] tx={transaction_id} "
            f"C={STATIC_COST} "
            f"E={STATIC_ERROR} "
            f"policy={STATIC_POLICY} "
            f"models={M_D} "
            f"theta={selected_theta} "
            f"parallel_first_n={parallel_first_n_runtime} "
            f"effective_mode={effective_cascade_mode}"
        )

        wait_collection_start = time.perf_counter()

        (
            wait_results,
            all_model_routes,
            node_wait_results,
        ) = get_wait_results_for_models(M_D)

        wait_collection_ms = (
            time.perf_counter() - wait_collection_start
        ) * 1000.0

        total_estimates = {
            model_name: wait_results.get(
                model_name,
                float("inf"),
            )
            for model_name in M_D
        }

        static_config_details = {
            "enabled": True,
            "cost": STATIC_COST,
            "error": STATIC_ERROR,
            "target_accuracy": 1.0 - float(STATIC_ERROR),
            "policy": STATIC_POLICY,
            "models": list(M_D),
            "theta": selected_theta,
            "parallel_first_n": parallel_first_n_runtime,
        }

        logger.info(
            f"[WAIT_COLLECTION] tx={transaction_id} "
            f"mode=static "
            f"queried_models={M_D} "
            f"time_ms={wait_collection_ms:.2f}"
        )

    # =========================================================
    # 2. Dynamic selection
    #
    # Dynamic selection needs wait estimates for all models.
    # =========================================================

    else:
        wait_collection_start = time.perf_counter()

        (
            wait_results,
            all_model_routes,
            node_wait_results,
        ) = get_wait_results_for_models(M_TOTAL)

        wait_collection_ms = (
            time.perf_counter() - wait_collection_start
        ) * 1000.0

        (
            M_D,
            total_estimates,
            effective_cascade_mode,
        ) = build_model_set_by_mode(
            cascade_mode=CASCADE_MODE,
            wait_results=wait_results,
            deadline_ms=deadline,
            ensemble_size=ENSEMBLE_SIZE,
            alpha_now=alpha_now,
            parallel_first_n=PARALLEL_FIRST_N,
            network_latency=NETWORK_LATENCY,
            logger=logger,
        )

        logger.info(
            f"[WAIT_COLLECTION] tx={transaction_id} "
            f"mode=dynamic "
            f"queried_models={list(M_TOTAL)} "
            f"time_ms={wait_collection_ms:.2f}"
        )

    # =========================================================
    # 3. Empty selection fallback
    # =========================================================

    if not M_D:
        fastest_model = min(
            MODEL_PROFILES,
            key=lambda model_name: float(
                MODEL_PROFILES[model_name]["latency"]
            ),
        )

        M_D = [fastest_model]
        effective_cascade_mode = "sequential"
        selected_theta = None

        # In static mode, this model may not have been queried.
        if fastest_model not in all_model_routes:
            (
                fallback_waits,
                fallback_routes,
                fallback_node_waits,
            ) = get_wait_results_for_models(
                [fastest_model]
            )

            wait_results.update(fallback_waits)
            all_model_routes.update(fallback_routes)

            for node_id, model_waits in fallback_node_waits.items():
                node_wait_results.setdefault(
                    node_id,
                    {},
                ).update(model_waits)

        total_estimates = {
            fastest_model: wait_results.get(
                fastest_model,
                float("inf"),
            )
        }

        logger.warning(
            f"[SELECT_EMPTY] tx={transaction_id} "
            f"fallback_fastest={fastest_model}"
        )

    # =========================================================
    # 4. Constraint solver
    #
    # This section runs only when static mode is disabled.
    # Dynamic mode already queried all models.
    # =========================================================

    if (
        not USE_STATIC_OPTIMAL_CONFIG
        and USE_CONSTRAINT_SOLVER
    ):
        constraint_solver = ConstraintSolver(
            seq_csv=CONSTRAINT_SEQ_PROFILE_CSV,
            power2_csv=CONSTRAINT_POWER2_PROFILE_CSV,
            full_parallel_csv=(
                CONSTRAINT_FULL_PARALLEL_PROFILE_CSV
            ),
        )

        constraint_result = constraint_solver.solve(
            max_cost=CONSTRAINT_MAX_COST,
            target_accuracy=CONSTRAINT_TARGET_ACCURACY,
            policy_order=CONSTRAINT_POLICY_ORDER,
            time_mode=CONSTRAINT_SOLVER_TIME_MODE,
            wait_results=wait_results,
            model_profiles=MODEL_PROFILES,
            network_latency=NETWORK_LATENCY,
            parallel_first_n=PARALLEL_FIRST_N,
            base_request_overhead=BASE_REQUEST_OVERHEAD,
            sequential_stage_overhead=(
                SEQUENTIAL_STAGE_OVERHEAD
            ),
            po2_first_batch_overhead=(
                PO2_FIRST_BATCH_OVERHEAD
            ),
            po2_tail_overhead=PO2_TAIL_OVERHEAD,
            full_parallel_agg_overhead=(
                FULL_PARALLEL_AGG_OVERHEAD
            ),
            et_scale_seq=ET_SCALE_SEQ,
            et_offset_seq=ET_OFFSET_SEQ,
            et_scale_p2=ET_SCALE_P2,
            et_offset_p2=ET_OFFSET_P2,
        )

        constraint_solver_details = {
            "success": constraint_result.success,
            "policy": constraint_result.policy,
            "reason": constraint_result.reason,
            "feasible_count": (
                constraint_result.feasible_count
            ),
            "total_count": constraint_result.total_count,
            "solver_latency_ms": (
                constraint_result.solver_latency_ms
            ),
            "selected_config": (
                constraint_result.selected_config
            ),
            "time_mode": CONSTRAINT_SOLVER_TIME_MODE,
            "max_cost": CONSTRAINT_MAX_COST,
            "target_accuracy": (
                CONSTRAINT_TARGET_ACCURACY
            ),
        }

        if constraint_result.success:
            selected = constraint_result.selected_config
            selected_policy = str(
                selected["policy"]
            ).lower()

            M_D = list(selected["models"])

            total_estimates = {
                model_name: wait_results.get(
                    model_name,
                    float("inf"),
                )
                for model_name in M_D
            }

            if selected_policy == "sequential":
                effective_cascade_mode = "sequential"
                selected_theta = float(
                    selected["theta"]
                )

            elif selected_policy in (
                "power2",
                "po2",
            ):
                effective_cascade_mode = (
                    "parallel_first_finish"
                )

                selected_theta = float(
                    selected.get(
                        "theta",
                        THRESHOLD,
                    )
                )

                parallel_first_n_runtime = int(
                    selected.get(
                        "parallel_first_n",
                        PARALLEL_FIRST_N,
                    )
                )

            elif selected_policy == "full_parallel":
                effective_cascade_mode = "full_parallel"
                selected_theta = None

            elif selected_policy == "spec_b":
                effective_cascade_mode = "spec_b"

                selected_theta = float(
                    selected.get(
                        "theta",
                        THRESHOLD,
                    )
                )

            else:
                effective_cascade_mode = "sequential"
                selected_theta = None

            logger.info(
                f"[CONSTRAINT_SOLVER] tx={transaction_id} "
                f"policy={selected_policy} "
                f"models={M_D} "
                f"theta={selected_theta} "
                f"parallel_first_n={parallel_first_n_runtime}"
            )

        else:
            fastest_model = min(
                MODEL_PROFILES,
                key=lambda model_name: float(
                    MODEL_PROFILES[model_name]["latency"]
                ),
            )

            M_D = [fastest_model]

            total_estimates = {
                fastest_model: wait_results.get(
                    fastest_model,
                    float("inf"),
                )
            }

            effective_cascade_mode = "sequential"
            selected_theta = None

            logger.warning(
                f"[CONSTRAINT_SOLVER_FALLBACK] "
                f"tx={transaction_id} "
                f"reason={constraint_result.reason} "
                f"fallback_fastest={fastest_model}"
            )

    # =========================================================
    # 5. Risk strategy selection
    # =========================================================

    elif (
        not USE_STATIC_OPTIMAL_CONFIG
        and not USE_CONSTRAINT_SOLVER
        and ENABLE_RISK_STRATEGY_SELECTION
        and len(M_D) > 1
    ):
        (
            risk_selected_mode,
            risk_details,
        ) = select_min_risk_strategy(
            selected_models=M_D,
            wait_results=wait_results,
            deadline_ms=deadline,
            network_latency=NETWORK_LATENCY,
            parallel_first_n=PARALLEL_FIRST_N,
            model_profiles=MODEL_PROFILES,
            omega=OMEGA,
            small_instance_models=SMALL_INSTANCE_MODELS,
            large_instance_models=LARGE_INSTANCE_MODELS,
            hardware_cost_rate=HARDWARE_COST_RATE,
            estimate_latency_fn=(
                estimate_selected_set_latency_for_mode
            ),
            alpha_latency=RISK_ALPHA_LATENCY,
            beta_cost=RISK_BETA_COST,
            gamma_error=RISK_GAMMA_ERROR,
        )

        effective_cascade_mode = risk_selected_mode

        logger.info(
            f"[RISK_SELECT] tx={transaction_id} "
            f"selected_mode={risk_selected_mode} "
            f"details={risk_details}"
        )

    # =========================================================
    # 6. Estimate selected latency
    # =========================================================

    raw_selected_est_latency = (
        estimate_selected_set_latency_for_mode(
            cascade_mode=effective_cascade_mode,
            selected_models=M_D,
            wait_results=wait_results,
            network_latency=NETWORK_LATENCY,
            parallel_first_n=parallel_first_n_runtime,
        )
    )

    # selected_est_latency = apply_latency_correction(
    #     raw_selected_est_latency
    # )

    if str(ESTIMATOR_SOURCE).lower() == "dysci":
        # DYSCI estimate is learned from actual invocation wall time,
        # so do not apply the old global latency correction again.
        selected_est_latency = raw_selected_est_latency
    else:
        # Preserve the existing correction when Iluvatar is used.
        selected_est_latency = apply_latency_correction(
            raw_selected_est_latency
        )

    deadline_sec = (
        float(deadline) / 1000.0
        if deadline is not None
        else 0.0
    )

    deadline_fast_path_triggered = False

    # =========================================================
    # 7. Deadline fast-path
    # =========================================================

    if (
        USE_DEADLINE_FAST_PATH
        and deadline_sec > 0.0
        and selected_est_latency >= deadline_sec
    ):
        fastest_model = min(
            MODEL_PROFILES,
            key=lambda model_name: float(
                MODEL_PROFILES[model_name]["latency"]
            ),
        )

        logger.warning(
            f"[DEADLINE_FAST_PATH] tx={transaction_id} "
            f"mode={effective_cascade_mode} "
            f"selected_models={M_D} "
            f"selected_est_latency={selected_est_latency:.4f}s "
            f"deadline={deadline_sec:.4f}s "
            f"fallback_fastest={fastest_model}"
        )

        M_D = [fastest_model]
        effective_cascade_mode = "sequential"
        selected_theta = None

        # Static mode may not have a route for fallback model.
        if fastest_model not in all_model_routes:
            (
                fallback_waits,
                fallback_routes,
                fallback_node_waits,
            ) = get_wait_results_for_models(
                [fastest_model]
            )

            wait_results.update(fallback_waits)
            all_model_routes.update(fallback_routes)

            for node_id, model_waits in fallback_node_waits.items():
                node_wait_results.setdefault(
                    node_id,
                    {},
                ).update(model_waits)

        total_estimates = {
            fastest_model: wait_results.get(
                fastest_model,
                float("inf"),
            )
        }

        raw_selected_est_latency = (
            estimate_selected_set_latency_for_mode(
                cascade_mode=effective_cascade_mode,
                selected_models=M_D,
                wait_results=wait_results,
                network_latency=NETWORK_LATENCY,
                parallel_first_n=parallel_first_n_runtime,
            )
        )

        # selected_est_latency = apply_latency_correction(
        #     raw_selected_est_latency
        # )

        if str(ESTIMATOR_SOURCE).lower() == "dysci":
            # DYSCI estimate is learned from actual invocation wall time,
            # so do not apply the old global latency correction again.
            selected_est_latency = raw_selected_est_latency
        else:
            # Preserve the existing correction when Iluvatar is used.
            selected_est_latency = apply_latency_correction(
                raw_selected_est_latency
            )

        deadline_fast_path_triggered = True

    # =========================================================
    # 8. Build final routing plan
    # =========================================================

    # routing_plan = build_routing_plan(
    #     selected_models=M_D,
    #     model_routes=all_model_routes,
    # )
    #
    # logger.info(
    #     f"[SELECT] tx={transaction_id} "
    #     f"selected_models={M_D}"
    # )

    routing_plan = build_routing_plan(
        selected_models=M_D,
        model_routes=all_model_routes,
    )

    # Which physical node was selected for each model
    selected_nodes = {
        model_name: route.get("node_id")
        for model_name, route in routing_plan.items()
    }

    logger.info(
        f"[SELECTED_NODES] tx={transaction_id} "
        f"selected_nodes={selected_nodes}"
    )

    logger.info(
        f"[SELECT] tx={transaction_id} "
        f"selected_models={M_D}"
    )

    logger.info(
        f"[EFFECTIVE_MODE] tx={transaction_id} "
        f"baseline_mode={BASELINE_MODE} "
        f"configured_cascade_mode={CASCADE_MODE} "
        f"effective_cascade_mode={effective_cascade_mode}"
    )

    selected_wait_log = {
        model_name: float(
            total_estimates.get(
                model_name,
                -1,
            )
        )
        for model_name in M_D
    }

    logger.info(
        f"[SELECT_WAITS] tx={transaction_id} "
        f"waits_selected={selected_wait_log}"
    )

    logger.info(
        f"[SELECT_ROUTE] tx={transaction_id} "
        f"routes={routing_plan}"
    )

    # =========================================================
    # 9. Create gRPC channels
    # =========================================================

    channel_map, stub_map = (
        build_stub_map_for_selected_models(
            model_list=M_D,
            routing_plan=routing_plan,
        )
    )

    # Separate start time for model execution.
    # Wait-time collection is not included in cascade deadline checks.
    execution_start_time = time.perf_counter()

    try:
        mode = str(
            effective_cascade_mode
        ).lower()

        if mode == "sequential":
            results, stop_reason = (
                await execute_sequential_cascade(
                    stub_map=stub_map,
                    routing_plan=routing_plan,
                    transaction_id=transaction_id,
                    model_list=M_D,
                    image_b64=image_b64,
                    start_time=execution_start_time,
                    deadline_ms=deadline,
                    selected_theta=selected_theta,
                )
            )

        elif mode in (
            "mode_s",
            "full_parallel",
        ):
            results, stop_reason = (
                await execute_mode_s_cascade(
                    stub_map=stub_map,
                    routing_plan=routing_plan,
                    transaction_id=transaction_id,
                    model_list=M_D,
                    image_b64=image_b64,
                )
            )

        elif mode == "spec_b":
            results, stop_reason = (
                await execute_speculative_cloud_cascade(
                    stub_map=stub_map,
                    routing_plan=routing_plan,
                    transaction_id=transaction_id,
                    model_list=M_D,
                    image_b64=image_b64,
                    selected_theta=selected_theta,
                )
            )

        else:
            results, stop_reason = (
                await execute_parallel_first_finish_cascade(
                    stub_map=stub_map,
                    routing_plan=routing_plan,
                    transaction_id=transaction_id,
                    model_list=M_D,
                    image_b64=image_b64,
                    parallel_first_n=parallel_first_n_runtime,
                    parallel_batch_size=PARALLEL_BATCH_SIZE,
                    start_time=execution_start_time,
                    deadline_ms=deadline,
                    selected_folder=selected_folder,
                    selected_theta=selected_theta,
                )
            )

    finally:
        await close_channels(channel_map)

    # =========================================================
    # 10. Timing
    # =========================================================

    end_time = time.perf_counter()

    e2e_ms = (
        end_time - request_start_time
    ) * 1000.0

    execution_time_ms = (
        end_time - execution_start_time
    ) * 1000.0

    actual_execution_latency_s = (
        execution_time_ms / 1000.0
    )

    # # Only execution time is used for overhead correction.
    # update_latency_overhead(
    #     actual_latency_s=actual_execution_latency_s,
    #     estimated_latency_s=raw_selected_est_latency,
    # )

    if str(ESTIMATOR_SOURCE).lower() == "iluvatar":
        update_latency_overhead(
            actual_latency_s=actual_execution_latency_s,
            estimated_latency_s=raw_selected_est_latency,
        )

    # =========================================================
    # 11. Per-model result
    # =========================================================

    per_model = {}

    for res in results:
        model_name = res.get("model")

        if not model_name:
            continue

        # per_model[model_name] = {
        #     "label": res.get("label"),
        #     "probability": float(
        #         res.get(
        #             "probability",
        #             0.0,
        #         )
        #     ),
        #     "latency_s": float(
        #         res.get(
        #             "latency",
        #             -1,
        #         )
        #     ),
        #     "success": bool(
        #         res.get(
        #             "success",
        #             False,
        #         )
        #     ),
        #     "state": res.get(
        #         "container_state",
        #         "UNKNOWN",
        #     ),
        #     "node_id": res.get("node_id"),
        #     "node_type": res.get("node_type"),
        #     "server_address": res.get(
        #         "server_address"
        #     ),
        #     "estimated_wait_s": float(
        #         res.get(
        #             "estimated_wait_s",
        #             float("inf"),
        #         )
        #     ),
        # }
        per_model[model_name] = {
            "label": res.get("label"),

            "probability": float(
                res.get("probability", 0.0)
            ),

            "latency_s": float(
                res.get("latency", -1.0)
            ),

            "success": bool(
                res.get("success", False)
            ),

            "state": res.get(
                "container_state",
                "UNKNOWN",
            ),

            "node_id": res.get("node_id"),
            "node_type": res.get("node_type"),
            "device": res.get("device"),

            "server_address": res.get(
                "server_address"
            ),

            "estimated_wait_s": float(
                res.get(
                    "estimated_wait_s",
                    float("inf"),
                )
            ),

            # ====================================================
            # Estimator comparison
            # ====================================================

            "estimate_source": res.get(
                "estimate_source",
                "",
            ),

            "dysci_estimated_e2e_s": float(
                res.get(
                    "dysci_estimated_e2e_s",
                    -1.0,
                )
            ),

            "dysci_estimated_wait_s": float(
                res.get(
                    "dysci_estimated_wait_s",
                    -1.0,
                )
            ),

            "dysci_estimated_service_s": float(
                res.get(
                    "dysci_estimated_service_s",
                    -1.0,
                )
            ),

            "dysci_wall_s": float(
                res.get(
                    "dysci_wall_s",
                    -1.0,
                )
            ),

            "iluvatar_estimate_s": float(
                res.get(
                    "iluvatar_estimate_s",
                    -1.0,
                )
            ),

            "dysci_outstanding_at_estimate": res.get(
                "dysci_outstanding_at_estimate"
            ),

            "dysci_outstanding_work_at_estimate_s": res.get(
                "dysci_outstanding_work_at_estimate_s"
            ),

            "dysci_estimator_source": res.get(
                "dysci_estimator_source",
                "",
            ),
        }

    # =========================================================
    # 12. Shared return fields
    # =========================================================

    base_return = {
        "transaction_id": transaction_id,
        "deployment_config": deployment_config,
        "num_edge_nodes": len(active_edge_nodes),
        "num_cloud_nodes": len(active_cloud_nodes),
        "active_edge_nodes": active_edge_nodes,
        "active_cloud_nodes": active_cloud_nodes,
        "selected_nodes": selected_nodes,

        "selected_models": M_D,
        "per_model": per_model,
        "wait_times": total_estimates,
        "selected_folder": selected_folder,
        "ensemble_size": ENSEMBLE_SIZE,
        "threshold": THRESHOLD,
        "threshold_stage": THRESHOLD_STAGE,
        "selected_theta": selected_theta,
        "parallel_first_n": parallel_first_n_runtime,
        "parallel_batch_size": PARALLEL_BATCH_SIZE,
        "stop_reason": stop_reason,
        "main_policy": POLICY,
        "alpha": alpha_now,
        "alpha_mode": ALPHA_MODE,
        "qps_est": qps_est,
        "baseline_mode": BASELINE_MODE,
        "configured_cascade_mode": CASCADE_MODE,
        "effective_cascade_mode": effective_cascade_mode,
        "cascade_mode": CASCADE_MODE,
        "use_two_instances": USE_TWO_ILUVATAR_INSTANCES,
        "use_multi_instances": USE_MULTI_ILUVATAR_INSTANCES,
        "routing_plan": routing_plan,
        "node_wait_results": node_wait_results,
        "wait_collection_ms": wait_collection_ms,
        "execution_time_ms": execution_time_ms,
        "deadline_fast_path_enabled": (
            USE_DEADLINE_FAST_PATH
        ),
        "deadline_fast_path_triggered": (
            deadline_fast_path_triggered
        ),
        "selected_est_latency_s": (
            selected_est_latency
        ),
        "raw_selected_est_latency_s": (
            raw_selected_est_latency
        ),
        "recent_latency_overhead_s": (
            get_recent_latency_overhead()
        ),
        "log_file": LOG_PATH,
        "risk_details": risk_details,
        "risk_selected_mode": effective_cascade_mode,
        "constraint_solver_details": (
            constraint_solver_details
        ),
        "static_config_details": static_config_details,
        "static_cost": (
            STATIC_COST
            if USE_STATIC_OPTIMAL_CONFIG
            else None
        ),
        "static_error": (
            STATIC_ERROR
            if USE_STATIC_OPTIMAL_CONFIG
            else None
        ),
        "static_policy": (
            STATIC_POLICY
            if USE_STATIC_OPTIMAL_CONFIG
            else None
        ),
    }

    # =========================================================
    # 13. No successful result
    # =========================================================

    if not results:
        logger.error(
            f"[REQ_END] tx={transaction_id} "
            f"success=False "
            f"reason=no_successful_models "
            f"e2e_ms={e2e_ms:.2f} "
            f"execution_ms={execution_time_ms:.2f}"
        )

        return {
            **base_return,
            "executed_models": [],
            "label": -1,
            "accuracy": 0.0,
            "combiner_policy": "FAILED",
            "e2e_time_ms": e2e_ms,
            "success": False,
        }

    # =========================================================
    # 14. Final output
    # =========================================================

    if len(results) == 1:
        single = results[0]

        final_result = {
            "label": single.get("label"),
            "accuracy": float(
                single.get(
                    "probability",
                    0.0,
                )
            ),
            "combiner_policy": "single_model",
            "success": True,
        }

    else:
        final_result = combine_outputs(
            results,
            policy="weighted_majority",
            historical_acc=OMEGA,
            model_weights=MODEL_WEIGHTS,
            ground_truth=selected_folder,
            gamma=GAMMA,
            update_weights=True,
            label_matcher=wnid_matches_text_label,
            rho=RHO,
        )

    save_model_weights_atomic(
        WEIGHTS_STATE_PATH,
        MODEL_WEIGHTS,
    )

    for res in results:
        try:
            model_name = res["model"]

            OMEGA[model_name] = (
                (1 - ETA) * OMEGA[model_name]
                + ETA
                * float(
                    res.get(
                        "probability",
                        0.0,
                    )
                )
            )

            active_models.add(model_name)

        except Exception:
            pass

    logger.info(
        f"[REQ_END] tx={transaction_id} "
        f"success={final_result.get('success', True)} "
        f"stop_reason={stop_reason} "
        f"e2e_ms={e2e_ms:.2f} "
        f"wait_collection_ms={wait_collection_ms:.2f} "
        f"execution_ms={execution_time_ms:.2f} "
        f"alpha={alpha_now:.3f} "
        f"configured_cascade_mode={CASCADE_MODE} "
        f"effective_cascade_mode={effective_cascade_mode} "
        f"baseline_mode={BASELINE_MODE}"
    )

    return {
        **base_return,
        "executed_models": [
            result["model"]
            for result in results
        ],
        "label": final_result["label"],
        "accuracy": final_result["accuracy"],
        "combiner_policy": (
            final_result["combiner_policy"]
        ),
        "e2e_time_ms": e2e_ms,
        "success": final_result.get(
            "success",
            True,
        ),
    }