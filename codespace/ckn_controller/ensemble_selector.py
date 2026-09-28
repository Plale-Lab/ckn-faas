# import itertools
# from typing import Dict, List, Tuple
#
# from ckn_controller.ckn_config import (
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     MODEL_PROFILES,
#     MODEL_SIZES,
#     SELECTION_STRATEGY,
# )
#
#
# # -----------------------------
# # Basic helpers
# # -----------------------------
# def get_exit_probability(model_name: str) -> float:
#     """
#     First simple approximation:
#     use model accuracy as probability of passing threshold.
#     """
#     try:
#         p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
#     except Exception:
#         p = 0.5
#     return max(0.0, min(1.0, p))
#
#
# def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
#     wait_t = float(wait_results.get(model_name, float("inf")))
#     compute_t = float(MODEL_PROFILES[model_name]["latency"])
#     return wait_t + compute_t
#
#
# def build_maxms(max_models: int) -> Dict[int, int]:
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     maxms: Dict[int, int] = {}
#     rolling_sum = 0
#
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_models:
#             maxms[size] = rolling_sum
#
#     return maxms
#
#
# def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
#     size = len(ms)
#     sum_bytes = sum(MODEL_SIZES[m] for m in ms)
#     if sum_bytes <= 0 or size not in maxms:
#         return None, sum_bytes
#     sc = maxms[size] / sum_bytes
#     return sc, sum_bytes
#
#
# def generate_combinations(models: List[str], max_size: int):
#     all_sets = []
#     for k in range(1, max_size + 1):
#         all_sets.extend(list(itertools.combinations(models, k)))
#     return all_sets
#
#
# # -----------------------------
# # Sequential / E[T] helpers
# # -----------------------------
# def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
#     """
#     Prefer higher exit probability and lower latency.
#     """
#     def rank(m: str):
#         p_exit = get_exit_probability(m)
#         t = max(get_stage_time(m, wait_results), 1e-9)
#         return p_exit / t
#
#     return sorted(models, key=rank, reverse=True)
#
#
# def estimate_expected_latency_sequential(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     """
#     E[T] = T(m1)
#          + (1-P1) * (T(m2)+network_latency)
#          + (1-P1)(1-P2) * (T(m3)+network_latency)
#          + ...
#     """
#     if not ordered_models:
#         return 0.0
#
#     expected_time = 0.0
#     prob_reach = 1.0
#
#     for i, m in enumerate(ordered_models):
#         stage_time = get_stage_time(m, wait_results)
#         if stage_time == float("inf"):
#             return float("inf")
#
#         if i > 0:
#             stage_time += float(network_latency)
#
#         expected_time += prob_reach * stage_time
#         prob_reach *= (1.0 - get_exit_probability(m))
#
#     return expected_time
#
#
# # -----------------------------
# # Parallel-first-finish / E[T] helpers
# # -----------------------------
# def estimate_batch_exit_probability(batch_models: List[str]) -> float:
#     """
#     Probability that at least one model in the first batch exits early.
#     """
#     if not batch_models:
#         return 0.0
#
#     fail_prob = 1.0
#     for m in batch_models:
#         fail_prob *= (1.0 - get_exit_probability(m))
#     return 1.0 - fail_prob
#
#
# def estimate_first_batch_latency(
#     batch_models: List[str],
#     wait_results: Dict[str, float],
# ) -> float:
#     """
#     First-finish approximation for initial parallel batch.
#     """
#     if not batch_models:
#         return float("inf")
#
#     vals = [get_stage_time(m, wait_results) for m in batch_models]
#     return min(vals) if vals else float("inf")
#
#
# def estimate_expected_latency_parallel_first_finish(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     """
#     Approximation:
#       E[T] = first_batch_latency
#            + (1 - P_exit_first_batch) * (network_latency + tail_expected_latency)
#     """
#     if not ordered_models:
#         return float("inf")
#
#     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
#     first_batch = ordered_models[:first_n]
#     tail_models = ordered_models[first_n:]
#
#     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
#     if first_batch_latency == float("inf"):
#         return float("inf")
#
#     p_exit_first_batch = estimate_batch_exit_probability(first_batch)
#
#     if not tail_models:
#         return first_batch_latency
#
#     tail_models = reorder_for_sequential(tail_models, wait_results)
#     tail_latency = estimate_expected_latency_sequential(
#         ordered_models=tail_models,
#         wait_results=wait_results,
#         network_latency=network_latency,
#     )
#
#     if tail_latency == float("inf"):
#         return float("inf")
#
#     return first_batch_latency + (1.0 - p_exit_first_batch) * (network_latency + tail_latency)
#
#
# # -----------------------------
# # Original MODE-S selection
# # -----------------------------
# def build_model_set_mode_s(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Original MODE-S greedy selection.
#
#     Te(MS) = max(wait + compute)
#     cost = (1 - alpha_now) * latency_term + alpha_now * size_cost
#     """
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_parallel_models <= 0:
#         return [], {}
#
#     maxms = build_maxms(max_parallel_models)
#
#     def estimate_te(ms: List[str]) -> float:
#         times = []
#         for m in ms:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(ms: List[str]):
#         te = estimate_te(ms)
#         raw = te / d_sec
#         if raw >= 1.0:
#             return None, te, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, te, raw
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, max_parallel_models, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         selected = sorted_by_acc[:k]
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     if POLICY == "randomized":
#         import random
#         k = max_parallel_models
#         selected = random.sample(M_TOTAL, k)
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     if logger:
#         logger.info(
#             f"[SELECT_BEGIN_MODE_S] deadline_ms={deadline_ms} d_sec={d_sec:.3f} "
#             f"target_k={max_parallel_models} alpha={alpha_now:.3f}"
#         )
#
#     for step in range(1, max_parallel_models + 1):
#         best_candidate = None
#         best_cost = float("inf")
#         best_dbg = None
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, te, raw = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             sc, sum_bytes = size_cost_for_models(trial, maxms)
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#                 best_dbg = (trial, te, raw, lt, sc, sum_bytes, cost)
#
#         if best_candidate is None:
#             if logger:
#                 logger.info(
#                     f"[SELECT_STOP_MODE_S] step={step} reason=no_feasible_candidate current_ms={ms}"
#                 )
#             break
#
#         trial, te, raw, lt, sc, sum_bytes, cost = best_dbg
#
#         if logger:
#             logger.info(
#                 f"[SELECT_CHOOSE_MODE_S] step={step}/{max_parallel_models} choose={best_candidate} "
#                 f"trial={trial} Te={te:.4f}s raw={raw:.3f} lt={lt:.3f} "
#                 f"size_cost={sc:.3f} sum_bytes={sum_bytes} cost={cost:.3f}"
#             )
#
#         ms.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not ms:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         ms = [fastest]
#         if logger:
#             logger.warning(f"[SELECT_FALLBACK_MODE_S] no_feasible_set fallback_fastest={fastest}")
#
#     if logger:
#         logger.info(f"[SELECT_DONE_MODE_S] selected_ms={ms} size={len(ms)}")
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# # -----------------------------
# # Sequential selection - greedy / exhaustive
# # -----------------------------
# def build_model_set_sequential_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     if logger:
#         logger.info(
#             f"[SELECT_BEGIN_SEQ_GREEDY] deadline_ms={deadline_ms} d_sec={d_sec:.3f} "
#             f"target_k={max_models} network_latency={network_latency:.4f}"
#         )
#
#     for step in range(1, max_models + 1):
#         best_candidate = None
#         best_et = float("inf")
#         best_trial = None
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             trial_ordered = reorder_for_sequential(trial, wait_results)
#             et = estimate_expected_latency_sequential(
#                 ordered_models=trial_ordered,
#                 wait_results=wait_results,
#                 network_latency=network_latency,
#             )
#
#             if et == float("inf"):
#                 continue
#
#             if et / d_sec >= 1.0:
#                 continue
#
#             if et < best_et:
#                 best_et = et
#                 best_candidate = m
#                 best_trial = trial_ordered
#
#         if best_candidate is None:
#             if logger:
#                 logger.info(
#                     f"[SELECT_STOP_SEQ_GREEDY] step={step} reason=no_feasible_candidate current_ms={ms}"
#                 )
#             break
#
#         if logger:
#             logger.info(
#                 f"[SELECT_CHOOSE_SEQ_GREEDY] step={step}/{max_models} choose={best_candidate} "
#                 f"trial={best_trial} E[T]={best_et:.4f}s"
#             )
#
#         ms = best_trial
#         remaining = [x for x in M_TOTAL if x not in ms]
#
#     if not ms:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         ms = [fastest]
#         if logger:
#             logger.warning(f"[SELECT_FALLBACK_SEQ_GREEDY] fallback_fastest={fastest}")
#
#     if logger:
#         logger.info(f"[SELECT_DONE_SEQ_GREEDY] selected_ms={ms} size={len(ms)}")
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# def build_model_set_sequential_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     best_set = None
#     best_et = float("inf")
#     all_sets = generate_combinations(M_TOTAL, max_models)
#
#     if logger:
#         logger.info(
#             f"[SELECT_BEGIN_SEQ_EXH] deadline_ms={deadline_ms} d_sec={d_sec:.3f} "
#             f"max_models={max_models} total_candidates={len(all_sets)}"
#         )
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         if et == float("inf"):
#             continue
#
#         if et / d_sec >= 1.0:
#             continue
#
#         if et < best_et:
#             best_et = et
#             best_set = ordered
#
#     if best_set is None:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         best_set = [fastest]
#         if logger:
#             logger.warning(f"[SELECT_FALLBACK_SEQ_EXH] fallback_fastest={fastest}")
#
#     if logger:
#         logger.info(f"[SELECT_DONE_SEQ_EXH] best_set={best_set} E[T]={best_et:.4f}s")
#
#     return best_set, {m: wait_results.get(m, float("inf")) for m in best_set}
#
#
# # -----------------------------
# # Parallel-first-finish selection - greedy / exhaustive
# # -----------------------------
# def build_model_set_parallel_first_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     if logger:
#         logger.info(
#             f"[SELECT_BEGIN_PFF_GREEDY] deadline_ms={deadline_ms} d_sec={d_sec:.3f} "
#             f"target_k={max_models} parallel_first_n={parallel_first_n} "
#             f"network_latency={network_latency:.4f}"
#         )
#
#     for step in range(1, max_models + 1):
#         best_candidate = None
#         best_et = float("inf")
#         best_trial = None
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             et = estimate_expected_latency_parallel_first_finish(
#                 ordered_models=trial,
#                 wait_results=wait_results,
#                 parallel_first_n=parallel_first_n,
#                 network_latency=network_latency,
#             )
#
#             if et == float("inf"):
#                 continue
#
#             if et / d_sec >= 1.0:
#                 continue
#
#             if et < best_et:
#                 best_et = et
#                 best_candidate = m
#                 best_trial = trial
#
#         if best_candidate is None:
#             if logger:
#                 logger.info(
#                     f"[SELECT_STOP_PFF_GREEDY] step={step} reason=no_feasible_candidate current_ms={ms}"
#                 )
#             break
#
#         if logger:
#             logger.info(
#                 f"[SELECT_CHOOSE_PFF_GREEDY] step={step}/{max_models} choose={best_candidate} "
#                 f"trial={best_trial} E[T]={best_et:.4f}s"
#             )
#
#         ms = best_trial
#         remaining = [x for x in M_TOTAL if x not in ms]
#
#     if not ms:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         ms = [fastest]
#         if logger:
#             logger.warning(f"[SELECT_FALLBACK_PFF_GREEDY] fallback_fastest={fastest}")
#
#     if logger:
#         logger.info(f"[SELECT_DONE_PFF_GREEDY] selected_ms={ms} size={len(ms)}")
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# def build_model_set_parallel_first_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     best_set = None
#     best_et = float("inf")
#     all_sets = generate_combinations(M_TOTAL, max_models)
#
#     if logger:
#         logger.info(
#             f"[SELECT_BEGIN_PFF_EXH] deadline_ms={deadline_ms} d_sec={d_sec:.3f} "
#             f"max_models={max_models} total_candidates={len(all_sets)} "
#             f"parallel_first_n={parallel_first_n}"
#         )
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered_models=subset,
#             wait_results=wait_results,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#         )
#
#         if et == float("inf"):
#             continue
#
#         if et / d_sec >= 1.0:
#             continue
#
#         if et < best_et:
#             best_et = et
#             best_set = subset
#
#     if best_set is None:
#         fastest = min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#         best_set = [fastest]
#         if logger:
#             logger.warning(f"[SELECT_FALLBACK_PFF_EXH] fallback_fastest={fastest}")
#
#     if logger:
#         logger.info(f"[SELECT_DONE_PFF_EXH] best_set={best_set} E[T]={best_et:.4f}s")
#
#     return best_set, {m: wait_results.get(m, float("inf")) for m in best_set}
#
#
# # -----------------------------
# # Public latency estimator for selected set
# # -----------------------------
# def estimate_selected_set_latency_for_mode(
#     cascade_mode: str,
#     selected_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
#     parallel_first_n: int,
# ) -> float:
#     mode = str(cascade_mode).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         vals = []
#         for m in selected_models:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             vals.append(wait_t + compute_t)
#         return max(vals) if vals else float("inf")
#
#     if mode == "sequential":
#         return estimate_expected_latency_sequential(
#             ordered_models=selected_models,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#     return estimate_expected_latency_parallel_first_finish(
#         ordered_models=selected_models,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#
#
# # -----------------------------
# # Dispatcher
# # -----------------------------
# def build_model_set_by_mode(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     mode = str(cascade_mode).lower()
#     strategy = str(SELECTION_STRATEGY).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         return build_model_set_mode_s(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             alpha_now=alpha_now,
#             logger=logger,
#         )
#
#     if mode == "sequential":
#         if strategy == "exhaustive":
#             return build_model_set_sequential_exhaustive(
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=ensemble_size,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#         return build_model_set_sequential_greedy(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     if strategy == "exhaustive":
#         return build_model_set_parallel_first_exhaustive(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     return build_model_set_parallel_first_greedy(
#         wait_results=wait_results,
#         deadline_ms=deadline_ms,
#         ensemble_size=ensemble_size,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#         logger=logger,
#     )


# ######
# # final working code 12-04-2026 start
# ########
#
# import itertools
# from typing import Dict, List, Tuple
#
# from ckn_controller.ckn_config import (
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     MODEL_PROFILES,
#     MODEL_SIZES,
#     SELECTION_STRATEGY,
#     USE_DEADLINE_FAST_PATH,
# )
#
#
# # -----------------------------
# # Basic helpers
# # -----------------------------
# def get_exit_probability(model_name: str) -> float:
#     """
#     Approximate early-exit probability using model accuracy.
#     """
#     try:
#         p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
#     except Exception:
#         p = 0.5
#     return max(0.0, min(1.0, p))
#
#
# def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
#     wait_t = float(wait_results.get(model_name, float("inf")))
#     compute_t = float(MODEL_PROFILES[model_name]["latency"])
#     return wait_t + compute_t
#
#
# def build_maxms(max_models: int) -> Dict[int, int]:
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     maxms: Dict[int, int] = {}
#     rolling_sum = 0
#
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_models:
#             maxms[size] = rolling_sum
#
#     return maxms
#
#
# def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
#     size = len(ms)
#     sum_bytes = sum(MODEL_SIZES[m] for m in ms)
#     if sum_bytes <= 0 or size not in maxms:
#         return None, sum_bytes
#     sc = maxms[size] / sum_bytes
#     return sc, sum_bytes
#
#
# def generate_combinations(models: List[str], max_size: int):
#     all_sets = []
#     for k in range(1, max_size + 1):
#         all_sets.extend(list(itertools.combinations(models, k)))
#     return all_sets
#
#
# def _fastest_model() -> str:
#     return min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#
#
# # -----------------------------
# # Sequential / E[T] helpers
# # -----------------------------
# def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
#     """
#     Rank by exit-probability-per-unit-time.
#     """
#     def rank(m: str):
#         p_exit = get_exit_probability(m)
#         t = max(get_stage_time(m, wait_results), 1e-9)
#         return p_exit / t
#
#     return sorted(models, key=rank, reverse=True)
#
#
# def estimate_expected_latency_sequential(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     """
#     E[T] = T(m1)
#          + (1-P1) * (T(m2)+network_latency)
#          + (1-P1)(1-P2) * (T(m3)+network_latency)
#          + ...
#     """
#     if not ordered_models:
#         return 0.0
#
#     expected_time = 0.0
#     prob_reach = 1.0
#
#     for i, m in enumerate(ordered_models):
#         stage_time = get_stage_time(m, wait_results)
#         if stage_time == float("inf"):
#             return float("inf")
#
#         if i > 0:
#             stage_time += float(network_latency)
#
#         expected_time += prob_reach * stage_time
#         prob_reach *= (1.0 - get_exit_probability(m))
#
#     return expected_time
#
#
# # -----------------------------
# # Parallel-first-finish / E[T] helpers
# # -----------------------------
# def estimate_batch_exit_probability(batch_models: List[str]) -> float:
#     """
#     Probability that at least one model in the first batch exits early.
#     """
#     if not batch_models:
#         return 0.0
#
#     fail_prob = 1.0
#     for m in batch_models:
#         fail_prob *= (1.0 - get_exit_probability(m))
#     return 1.0 - fail_prob
#
#
# def estimate_first_batch_latency(
#     batch_models: List[str],
#     wait_results: Dict[str, float],
# ) -> float:
#     """
#     First-finish approximation for initial parallel batch.
#     """
#     if not batch_models:
#         return float("inf")
#
#     vals = [get_stage_time(m, wait_results) for m in batch_models]
#     return min(vals) if vals else float("inf")
#
#
# def estimate_expected_latency_parallel_first_finish(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     """
#     Approximation:
#       E[T] = first_batch_latency
#            + (1 - P_exit_first_batch) * (network_latency + tail_expected_latency)
#     """
#     if not ordered_models:
#         return float("inf")
#
#     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
#     first_batch = ordered_models[:first_n]
#     tail_models = ordered_models[first_n:]
#
#     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
#     if first_batch_latency == float("inf"):
#         return float("inf")
#
#     p_exit_first_batch = estimate_batch_exit_probability(first_batch)
#
#     if not tail_models:
#         return first_batch_latency
#
#     tail_models = reorder_for_sequential(tail_models, wait_results)
#     tail_latency = estimate_expected_latency_sequential(
#         ordered_models=tail_models,
#         wait_results=wait_results,
#         network_latency=network_latency,
#     )
#
#     if tail_latency == float("inf"):
#         return float("inf")
#
#     return first_batch_latency + (1.0 - p_exit_first_batch) * (network_latency + tail_latency)
#
#
# # -----------------------------
# # Original MODE-S selection
# # -----------------------------
# def build_model_set_mode_s(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Original MODE-S greedy selection.
#     """
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_parallel_models <= 0:
#         return [], {}
#
#     maxms = build_maxms(max_parallel_models)
#
#     def estimate_te(ms: List[str]) -> float:
#         times = []
#         for m in ms:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(ms: List[str]):
#         te = estimate_te(ms)
#         raw = te / d_sec
#         if raw >= 1.0:
#             return None, te, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, te, raw
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, max_parallel_models, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         selected = sorted_by_acc[:k]
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     if POLICY == "randomized":
#         import random
#         k = max_parallel_models
#         selected = random.sample(M_TOTAL, k)
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     for _ in range(max_parallel_models):
#         best_candidate = None
#         best_cost = float("inf")
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, _, _ = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             sc, _ = size_cost_for_models(trial, maxms)
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#
#         if best_candidate is None:
#             break
#
#         ms.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not ms:
#         fastest = _fastest_model()
#         ms = [fastest]
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# # -----------------------------
# # Sequential selection - GREEDY
# # -----------------------------
# def build_model_set_sequential_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Greedy construction:
#     - best size-1
#     - extend to best size-2 from current best
#     - extend to best size-3 from current best
#     ...
#
#     If USE_DEADLINE_FAST_PATH=True:
#       choose the largest feasible built set; if none, fallback fastest.
#
#     If USE_DEADLINE_FAST_PATH=False:
#       return the maximum-size built set.
#     """
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         ordered = reorder_for_sequential(trial, wait_results)
#         et = estimate_expected_latency_sequential(ordered, wait_results, network_latency)
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#             et = estimate_expected_latency_sequential(ordered, wait_results, network_latency)
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Sequential selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_sequential_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#         candidates.append((ordered, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Parallel-first-finish selection - GREEDY
# # -----------------------------
# def build_model_set_parallel_first_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         et = estimate_expected_latency_parallel_first_finish(
#             trial, wait_results, parallel_first_n, network_latency
#         )
#         if et < best_single_et:
#             best_single_et = et
#             best_single = trial
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             et = estimate_expected_latency_parallel_first_finish(
#                 trial, wait_results, parallel_first_n, network_latency
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = trial
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Parallel-first-finish selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_parallel_first_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         et = estimate_expected_latency_parallel_first_finish(
#             subset, wait_results, parallel_first_n, network_latency
#         )
#         candidates.append((subset, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Public latency estimator
# # -----------------------------
# def estimate_selected_set_latency_for_mode(
#     cascade_mode: str,
#     selected_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
#     parallel_first_n: int,
# ) -> float:
#     mode = str(cascade_mode).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         vals = []
#         for m in selected_models:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             vals.append(wait_t + compute_t)
#         return max(vals) if vals else float("inf")
#
#     if mode == "sequential":
#         return estimate_expected_latency_sequential(
#             ordered_models=selected_models,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#     return estimate_expected_latency_parallel_first_finish(
#         ordered_models=selected_models,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#
#
# # -----------------------------
# # Dispatcher
# # -----------------------------
# def build_model_set_by_mode(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     mode = str(cascade_mode).lower()
#     strategy = str(SELECTION_STRATEGY).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         return build_model_set_mode_s(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             alpha_now=alpha_now,
#             logger=logger,
#         )
#
#     if mode == "sequential":
#         if strategy == "exhaustive":
#             return build_model_set_sequential_exhaustive(
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=ensemble_size,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#         return build_model_set_sequential_greedy(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     if strategy == "exhaustive":
#         return build_model_set_parallel_first_exhaustive(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     return build_model_set_parallel_first_greedy(
#         wait_results=wait_results,
#         deadline_ms=deadline_ms,
#         ensemble_size=ensemble_size,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#         logger=logger,
#     )
#
#
# ######
# # final working code 12-04-2026 end
# ########


# ######
# # final working code 23-04-2026 start
# ########
#
#
#
# import itertools
# from typing import Dict, List, Tuple
#
# from ckn_controller.ckn_config import (
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     MODEL_PROFILES,
#     MODEL_SIZES,
#     SELECTION_STRATEGY,
#     USE_DEADLINE_FAST_PATH,
# )
#
#
# # -----------------------------
# # Basic helpers
# # -----------------------------
# def get_exit_probability(model_name: str) -> float:
#     """
#     Approximate early-exit probability using model accuracy.
#     """
#     try:
#         p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
#     except Exception:
#         p = 0.5
#     return max(0.0, min(1.0, p))
#
#
# def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
#     wait_t = float(wait_results.get(model_name, float("inf")))
#     compute_t = float(MODEL_PROFILES[model_name]["latency"])
#     return wait_t + compute_t
#
#
# def build_maxms(max_models: int) -> Dict[int, int]:
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     maxms: Dict[int, int] = {}
#     rolling_sum = 0
#
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_models:
#             maxms[size] = rolling_sum
#
#     return maxms
#
#
# def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
#     size = len(ms)
#     sum_bytes = sum(MODEL_SIZES[m] for m in ms)
#     if sum_bytes <= 0 or size not in maxms:
#         return None, sum_bytes
#     sc = maxms[size] / sum_bytes
#     return sc, sum_bytes
#
#
# def generate_combinations(models: List[str], max_size: int):
#     all_sets = []
#     for k in range(1, max_size + 1):
#         all_sets.extend(list(itertools.combinations(models, k)))
#     return all_sets
#
#
# def _fastest_model() -> str:
#     return min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#
#
# # -----------------------------
# # Sequential / E[T] helpers
# # -----------------------------
# def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
#     """
#     Rank by exit-probability-per-unit-time.
#     """
#     def rank(m: str):
#         p_exit = get_exit_probability(m)
#         t = max(get_stage_time(m, wait_results), 1e-9)
#         return p_exit / t
#
#     return sorted(models, key=rank, reverse=True)
#
#
# def estimate_expected_latency_sequential(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     """
#     E[T] = T(m1)
#          + (1-P1) * (T(m2)+network_latency)
#          + (1-P1)(1-P2) * (T(m3)+network_latency)
#          + ...
#     """
#     if not ordered_models:
#         return 0.0
#
#     expected_time = 0.0
#     prob_reach = 1.0
#
#     for i, m in enumerate(ordered_models):
#         stage_time = get_stage_time(m, wait_results)
#         if stage_time == float("inf"):
#             return float("inf")
#
#         if i > 0:
#             stage_time += float(network_latency)
#
#         expected_time += prob_reach * stage_time
#         prob_reach *= (1.0 - get_exit_probability(m))
#
#     return expected_time
#
#
# # -----------------------------
# # Parallel-first-finish / E[T] helpers
# # -----------------------------
# # def estimate_batch_exit_probability(batch_models: List[str]) -> float:
# #     """
# #     Approximate probability that the request stops after the first batch.
# #
# #     Uses:
# #         P(stop after first batch) ≈ 1 - Π(1 - p_m)
# #
# #     where p_m is the model-wise exit probability.
# #     """
# #     if not batch_models:
# #         return 0.0
# #
# #     fail_prob = 1.0
# #     for m in batch_models:
# #         fail_prob *= (1.0 - get_exit_probability(m))
# #
# #     return 1.0 - fail_prob
# #
# #
# # def estimate_first_batch_latency(
# #     batch_models: List[str],
# #     wait_results: Dict[str, float],
# # ) -> float:
# #     """
# #     Approximate first-batch latency.
# #
# #     Runtime idea:
# #     - launch first batch in parallel
# #     - if one model finishes first and passes threshold, stop early
# #     - otherwise wait for the rest of the first batch and aggregate
# #
# #     Approximation:
# #       E[T_first_batch] =
# #           p_stop_after_first_batch * t_fast
# #         + (1 - p_stop_after_first_batch) * t_slow
# #
# #     where:
# #       t_fast = fastest first-batch model time
# #       t_slow = slowest first-batch model time
# #       p_stop_after_first_batch ≈ 1 - Π(1 - p_m)
# #     """
# #     if not batch_models:
# #         return float("inf")
# #
# #     vals = [get_stage_time(m, wait_results) for m in batch_models]
# #     if not vals or any(v == float("inf") for v in vals):
# #         return float("inf")
# #
# #     t_fast = min(vals)
# #     t_slow = max(vals)
# #
# #     p_stop_after_first_batch = estimate_batch_exit_probability(batch_models)
# #
# #     return (
# #         p_stop_after_first_batch * t_fast
# #         + (1.0 - p_stop_after_first_batch) * t_slow
# #     )
# #
# #
# # def estimate_expected_latency_parallel_first_finish(
# #     ordered_models: List[str],
# #     wait_results: Dict[str, float],
# #     parallel_first_n: int,
# #     network_latency: float,
# # ) -> float:
# #     """
# #     Approximation:
# #       E[T] = E[first_batch_time]
# #            + P(go_to_tail) * (network_latency + tail_expected_latency)
# #
# #     where:
# #       P(stop after first batch) ≈ 1 - Π(1 - p_m)
# #       P(go_to_tail) = 1 - P(stop after first batch)
# #     """
# #     if not ordered_models:
# #         return float("inf")
# #
# #     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
# #
# #     # Reorder so better fast/likely-exit models are earlier
# #     ordered_models = reorder_for_sequential(ordered_models, wait_results)
# #
# #     first_batch = ordered_models[:first_n]
# #     tail_models = ordered_models[first_n:]
# #
# #     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
# #     if first_batch_latency == float("inf"):
# #         return float("inf")
# #
# #     p_stop_after_first_batch = estimate_batch_exit_probability(first_batch)
# #     p_go_to_tail = 1.0 - p_stop_after_first_batch
# #
# #     if not tail_models:
# #         return first_batch_latency
# #
# #     tail_models = reorder_for_sequential(tail_models, wait_results)
# #     tail_latency = estimate_expected_latency_sequential(
# #         ordered_models=tail_models,
# #         wait_results=wait_results,
# #         network_latency=network_latency,
# #     )
# #
# #     if tail_latency == float("inf"):
# #         return float("inf")
# #
# #     return first_batch_latency + p_go_to_tail * (float(network_latency) + tail_latency)
#
# # -----------------------------
# # Parallel-first-finish / E[T] helpers
# # -----------------------------
# def estimate_batch_exit_probability(batch_models: List[str]) -> float:
#     """
#     Approximate probability that the request stops after the first batch.
#
#     Uses:
#         P(stop after first batch) ≈ 1 - Π(1 - p_m)
#
#     where p_m is the model-wise exit probability.
#     """
#     if not batch_models:
#         return 0.0
#
#     fail_prob = 1.0
#     for m in batch_models:
#         fail_prob *= (1.0 - get_exit_probability(m))
#
#     return 1.0 - fail_prob
#
#
# def estimate_first_batch_latency(
#     batch_models: List[str],
#     wait_results: Dict[str, float],
# ) -> float:
#     """
#     Approximate first-batch latency based on actual runtime logic:
#
#     - launch first batch in parallel
#     - if first finished model passes threshold, stop at t_fast
#     - otherwise wait for rest of first batch and aggregate, reaching t_slow
#
#     Approximation:
#         E[T_first_batch] =
#             p_first_pass * t_fast
#           + (1 - p_first_pass) * t_slow
#
#     where:
#       t_fast = fastest first-batch model time
#       t_slow = slowest first-batch model time
#       p_first_pass = exit probability of the fastest model
#     """
#     if not batch_models:
#         return float("inf")
#
#     batch_with_times = []
#     for m in batch_models:
#         t = get_stage_time(m, wait_results)
#         if t == float("inf"):
#             return float("inf")
#         batch_with_times.append((m, t))
#
#     if not batch_with_times:
#         return float("inf")
#
#     # sort by completion time
#     batch_with_times.sort(key=lambda x: x[1])
#
#     first_model, t_fast = batch_with_times[0]
#     t_slow = max(t for _, t in batch_with_times)
#
#     p_first_pass = get_exit_probability(first_model)
#
#     return (p_first_pass * t_fast) + ((1.0 - p_first_pass) * t_slow)
#
#
# def estimate_expected_latency_parallel_first_finish(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     """
#     Approximation:
#       E[T] = E[first_batch_time]
#            + P(go_to_tail) * (network_latency + tail_expected_latency)
#
#     where:
#       P(stop after first batch) ≈ 1 - Π(1 - p_m)
#       P(go_to_tail) = 1 - P(stop after first batch)
#     """
#     if not ordered_models:
#         return float("inf")
#
#     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
#
#     # Reorder so better fast/likely-exit models are earlier
#     ordered_models = reorder_for_sequential(ordered_models, wait_results)
#
#     first_batch = ordered_models[:first_n]
#     tail_models = ordered_models[first_n:]
#
#     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
#     if first_batch_latency == float("inf"):
#         return float("inf")
#
#     p_stop_after_first_batch = estimate_batch_exit_probability(first_batch)
#     p_go_to_tail = 1.0 - p_stop_after_first_batch
#
#     if not tail_models:
#         return first_batch_latency
#
#     tail_models = reorder_for_sequential(tail_models, wait_results)
#     tail_latency = estimate_expected_latency_sequential(
#         ordered_models=tail_models,
#         wait_results=wait_results,
#         network_latency=network_latency,
#     )
#
#     if tail_latency == float("inf"):
#         return float("inf")
#
#     return first_batch_latency + p_go_to_tail * (float(network_latency) + tail_latency)
#
#
# # -----------------------------
# # Original MODE-S selection
# # -----------------------------
# def build_model_set_mode_s(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     """
#     Original MODE-S greedy selection.
#     """
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_parallel_models <= 0:
#         return [], {}
#
#     maxms = build_maxms(max_parallel_models)
#
#     def estimate_te(ms: List[str]) -> float:
#         times = []
#         for m in ms:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             times.append(wait_t + compute_t)
#         return max(times) if times else float("inf")
#
#     def latency_term_for(ms: List[str]):
#         te = estimate_te(ms)
#         raw = te / d_sec
#         if raw >= 1.0:
#             return None, te, raw
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, te, raw
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, max_parallel_models, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             if m in MODEL_PROFILES and "accuracy" in MODEL_PROFILES[m]:
#                 return float(MODEL_PROFILES[m]["accuracy"])
#             return float("-inf")
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         selected = sorted_by_acc[:k]
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     if POLICY == "randomized":
#         import random
#         k = max_parallel_models
#         selected = random.sample(M_TOTAL, k)
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     for _ in range(max_parallel_models):
#         best_candidate = None
#         best_cost = float("inf")
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, _, _ = latency_term_for(trial)
#             if lt is None:
#                 continue
#
#             sc, _ = size_cost_for_models(trial, maxms)
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#
#         if best_candidate is None:
#             break
#
#         ms.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not ms:
#         fastest = _fastest_model()
#         ms = [fastest]
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# # -----------------------------
# # Sequential selection - GREEDY
# # -----------------------------
# def build_model_set_sequential_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         ordered = reorder_for_sequential(trial, wait_results)
#         et = estimate_expected_latency_sequential(ordered, wait_results, network_latency)
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#             et = estimate_expected_latency_sequential(ordered, wait_results, network_latency)
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Sequential selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_sequential_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#         candidates.append((ordered, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Parallel-first-finish selection - GREEDY
# # -----------------------------
# def build_model_set_parallel_first_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         ordered = reorder_for_sequential(trial, wait_results)
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered, wait_results, parallel_first_n, network_latency
#         )
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#
#             et = estimate_expected_latency_parallel_first_finish(
#                 ordered, wait_results, parallel_first_n, network_latency
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Parallel-first-finish selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_parallel_first_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered, wait_results, parallel_first_n, network_latency
#         )
#         candidates.append((ordered, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Public latency estimator
# # -----------------------------
# def estimate_selected_set_latency_for_mode(
#     cascade_mode: str,
#     selected_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
#     parallel_first_n: int,
# ) -> float:
#     mode = str(cascade_mode).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         vals = []
#         for m in selected_models:
#             wait_t = float(wait_results.get(m, float("inf")))
#             compute_t = float(MODEL_PROFILES[m]["latency"])
#             vals.append(wait_t + compute_t)
#         return max(vals) if vals else float("inf")
#
#     if mode == "sequential":
#         return estimate_expected_latency_sequential(
#             ordered_models=selected_models,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#     return estimate_expected_latency_parallel_first_finish(
#         ordered_models=selected_models,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#
#
# # -----------------------------
# # Dispatcher
# # -----------------------------
# def build_model_set_by_mode(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     mode = str(cascade_mode).lower()
#     strategy = str(SELECTION_STRATEGY).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         return build_model_set_mode_s(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             alpha_now=alpha_now,
#             logger=logger,
#         )
#
#     if mode == "sequential":
#         if strategy == "exhaustive":
#             return build_model_set_sequential_exhaustive(
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=ensemble_size,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#         return build_model_set_sequential_greedy(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     if strategy == "exhaustive":
#         return build_model_set_parallel_first_exhaustive(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     return build_model_set_parallel_first_greedy(
#         wait_results=wait_results,
#         deadline_ms=deadline_ms,
#         ensemble_size=ensemble_size,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#         logger=logger,
#     )
#
# # ######
# # #  working code 23-04-2026 end
# # ########




# # # ######
# # # # final working code 05-04-2026 start
# # # ########
#
# import itertools
# from typing import Dict, List, Tuple
#
# from ckn_controller.ckn_config import (
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     MODEL_PROFILES,
#     MODEL_SIZES,
#     SELECTION_STRATEGY,
#     USE_DEADLINE_FAST_PATH,
#     BASE_REQUEST_OVERHEAD,
#     SEQUENTIAL_STAGE_OVERHEAD,
#     PO2_FIRST_BATCH_OVERHEAD,
#     PO2_TAIL_OVERHEAD,
#     FULL_PARALLEL_AGG_OVERHEAD,
#     ET_SCALE_SEQ,
#     ET_OFFSET_SEQ,
#     ET_SCALE_P2,
#     ET_OFFSET_P2,
# )
#
#
# # -----------------------------
# # Basic helpers
# # -----------------------------
# def get_exit_probability(model_name: str) -> float:
#     try:
#         p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
#     except Exception:
#         p = 0.5
#     return max(0.0, min(1.0, p))
#
#
# def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
#     wait_t = float(wait_results.get(model_name, float("inf")))
#     compute_t = float(MODEL_PROFILES[model_name]["latency"])
#     return wait_t
#
#
# def calibrate_sequential_et(et_raw: float) -> float:
#     if et_raw == float("inf"):
#         return et_raw
#     return float(ET_OFFSET_SEQ) + float(ET_SCALE_SEQ) * et_raw
#
#
# def calibrate_power2_et(et_raw: float) -> float:
#     if et_raw == float("inf"):
#         return et_raw
#     return float(ET_OFFSET_P2) + float(ET_SCALE_P2) * et_raw
#
#
# def build_maxms(max_models: int) -> Dict[int, int]:
#     sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
#     maxms: Dict[int, int] = {}
#     rolling_sum = 0
#
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#         if size <= max_models:
#             maxms[size] = rolling_sum
#
#     return maxms
#
#
# def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
#     size = len(ms)
#     sum_bytes = sum(MODEL_SIZES[m] for m in ms)
#
#     if sum_bytes <= 0 or size not in maxms:
#         return None, sum_bytes
#
#     sc = maxms[size] / sum_bytes
#     return sc, sum_bytes
#
#
# def generate_combinations(models: List[str], max_size: int):
#     all_sets = []
#     for k in range(1, max_size + 1):
#         all_sets.extend(list(itertools.combinations(models, k)))
#     return all_sets
#
#
# def _fastest_model() -> str:
#     return min(M_TOTAL, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#
#
# # -----------------------------
# # Sequential / E[T] helpers
# # -----------------------------
# def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
#     def rank(m: str):
#         p_exit = get_exit_probability(m)
#         t = max(get_stage_time(m, wait_results), 1e-9)
#         return p_exit / t
#
#     return sorted(models, key=rank, reverse=True)
#
#
# def estimate_expected_latency_sequential_raw(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     """
#     Sequential raw E[T] with overheads.
#
#     E[T]_raw =
#         BASE_REQUEST_OVERHEAD
#       + T1
#       + (1-P1)(T2 + b + SEQUENTIAL_STAGE_OVERHEAD)
#       + ...
#     """
#     if not ordered_models:
#         return 0.0
#
#     expected_time = float(BASE_REQUEST_OVERHEAD)
#     prob_reach = 1.0
#
#     for i, m in enumerate(ordered_models):
#         stage_time = get_stage_time(m, wait_results)
#
#         if stage_time == float("inf"):
#             return float("inf")
#
#         if i > 0:
#             stage_time += float(network_latency) + float(SEQUENTIAL_STAGE_OVERHEAD)
#
#         expected_time += prob_reach * stage_time
#         prob_reach *= (1.0 - get_exit_probability(m))
#
#     return expected_time
#
#
# def estimate_expected_latency_sequential(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     et_raw = estimate_expected_latency_sequential_raw(
#         ordered_models=ordered_models,
#         wait_results=wait_results,
#         network_latency=network_latency,
#     )
#     return calibrate_sequential_et(et_raw)
#
#
# # -----------------------------
# # Power-of-two / E[T] helpers
# # -----------------------------
# def estimate_batch_exit_probability(batch_models: List[str]) -> float:
#     if not batch_models:
#         return 0.0
#
#     fail_prob = 1.0
#     for m in batch_models:
#         fail_prob *= (1.0 - get_exit_probability(m))
#
#     return 1.0 - fail_prob
#
#
# def estimate_first_batch_latency(
#     batch_models: List[str],
#     wait_results: Dict[str, float],
# ) -> float:
#     """
#     FirstBatchLatency =
#         P1 * min(T1,T2) + (1-P1) * max(T1,T2)
#
#     P1 is the exit probability of the fastest model in first batch.
#     """
#     if not batch_models:
#         return float("inf")
#
#     batch_with_times = []
#
#     for m in batch_models:
#         t = get_stage_time(m, wait_results)
#
#         if t == float("inf"):
#             return float("inf")
#
#         batch_with_times.append((m, t))
#
#     batch_with_times.sort(key=lambda x: x[1])
#
#     first_model, t_fast = batch_with_times[0]
#     t_slow = max(t for _, t in batch_with_times)
#
#     p_first_pass = get_exit_probability(first_model)
#
#     return (p_first_pass * t_fast) + ((1.0 - p_first_pass) * t_slow)
#
#
# def estimate_expected_latency_parallel_first_finish_raw(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     """
#     Power-of-two raw E[T] with overheads.
#
#     E[T]_raw =
#         BASE_REQUEST_OVERHEAD
#       + PO2_FIRST_BATCH_OVERHEAD
#       + FirstBatchLatency
#       + P(go_tail)(b + PO2_TAIL_OVERHEAD + TailLatency_raw)
#     """
#     if not ordered_models:
#         return float("inf")
#
#     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
#
#     ordered_models = reorder_for_sequential(ordered_models, wait_results)
#
#     first_batch = ordered_models[:first_n]
#     tail_models = ordered_models[first_n:]
#
#     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
#
#     if first_batch_latency == float("inf"):
#         return float("inf")
#
#     p_stop_after_first_batch = estimate_batch_exit_probability(first_batch)
#     p_go_to_tail = 1.0 - p_stop_after_first_batch
#
#     raw_et = (
#         float(BASE_REQUEST_OVERHEAD)
#         + float(PO2_FIRST_BATCH_OVERHEAD)
#         + first_batch_latency
#     )
#
#     if tail_models:
#         tail_models = reorder_for_sequential(tail_models, wait_results)
#
#         tail_latency_raw = estimate_expected_latency_sequential_raw(
#             ordered_models=tail_models,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         if tail_latency_raw == float("inf"):
#             return float("inf")
#
#         raw_et += p_go_to_tail * (
#             float(network_latency)
#             + float(PO2_TAIL_OVERHEAD)
#             + tail_latency_raw
#         )
#
#     return raw_et
#
#
# def estimate_expected_latency_parallel_first_finish(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     et_raw = estimate_expected_latency_parallel_first_finish_raw(
#         ordered_models=ordered_models,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#     return calibrate_power2_et(et_raw)
#
#
# # -----------------------------
# # Original MODE-S / Full Parallel selection
# # -----------------------------
# def build_model_set_mode_s(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ensemble_size, K // C, len(M_TOTAL))
#
#     if max_parallel_models <= 0:
#         return [], {}
#
#     maxms = build_maxms(max_parallel_models)
#
#     def estimate_te(ms: List[str]) -> float:
#         times = []
#
#         for m in ms:
#             stage_time = get_stage_time(m, wait_results)
#             times.append(stage_time)
#
#         if not times:
#             return float("inf")
#
#         return (
#             float(BASE_REQUEST_OVERHEAD)
#             + max(times)
#             + float(FULL_PARALLEL_AGG_OVERHEAD)
#         )
#
#     def latency_term_for(ms: List[str]):
#         te = estimate_te(ms)
#         raw = te / d_sec
#
#         if raw >= 1.0:
#             return None, te, raw
#
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, te, raw
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, max_parallel_models, len(M_TOTAL))
#
#         def get_acc(m: str) -> float:
#             return float(MODEL_PROFILES.get(m, {}).get("accuracy", float("-inf")))
#
#         sorted_by_acc = sorted(M_TOTAL, key=get_acc, reverse=True)
#         selected = sorted_by_acc[:k]
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     if POLICY == "randomized":
#         import random
#
#         k = max_parallel_models
#         selected = random.sample(M_TOTAL, k)
#         return selected, {m: wait_results.get(m, float("inf")) for m in selected}
#
#     ms: List[str] = []
#     remaining = list(M_TOTAL)
#
#     for _ in range(max_parallel_models):
#         best_candidate = None
#         best_cost = float("inf")
#
#         for m in remaining:
#             trial = ms + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, _, _ = latency_term_for(trial)
#
#             if lt is None:
#                 continue
#
#             sc, _ = size_cost_for_models(trial, maxms)
#
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#
#         if best_candidate is None:
#             break
#
#         ms.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not ms:
#         fastest = _fastest_model()
#         ms = [fastest]
#
#     return ms, {m: wait_results.get(m, float("inf")) for m in ms}
#
#
# # -----------------------------
# # Sequential selection - GREEDY
# # -----------------------------
# def build_model_set_sequential_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         ordered = reorder_for_sequential(trial, wait_results)
#
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#
#             et = estimate_expected_latency_sequential(
#                 ordered_models=ordered,
#                 wait_results=wait_results,
#                 network_latency=network_latency,
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Sequential selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_sequential_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         candidates.append((ordered, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Power-of-two selection - GREEDY
# # -----------------------------
# def build_model_set_parallel_first_greedy(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in M_TOTAL:
#         trial = [m]
#         ordered = reorder_for_sequential(trial, wait_results)
#
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#         )
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in M_TOTAL if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#
#             et = estimate_expected_latency_parallel_first_finish(
#                 ordered_models=ordered,
#                 wait_results=wait_results,
#                 parallel_first_n=parallel_first_n,
#                 network_latency=network_latency,
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in M_TOTAL if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#
# # -----------------------------
# # Power-of-two selection - EXHAUSTIVE
# # -----------------------------
# def build_model_set_parallel_first_exhaustive(
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(M_TOTAL))
#
#     if max_models <= 0:
#         return [], {}
#
#     all_sets = generate_combinations(M_TOTAL, max_models)
#     candidates: List[Tuple[List[str], float]] = []
#
#     for subset in all_sets:
#         subset = list(subset)
#
#         if len(subset) * C > K:
#             continue
#
#         ordered = reorder_for_sequential(subset, wait_results)
#
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#         )
#
#         candidates.append((ordered, et))
#
#     if not candidates:
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible = [(ms, et) for ms, et in candidates if et < d_sec]
#
#         if feasible:
#             feasible.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible[0]
#             return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#         fastest = _fastest_model()
#         return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#     exact_size_candidates = [(ms, et) for ms, et in candidates if len(ms) == max_models]
#
#     if exact_size_candidates:
#         chosen_ms, _ = min(exact_size_candidates, key=lambda x: x[1])
#         return chosen_ms, {m: wait_results.get(m, float("inf")) for m in chosen_ms}
#
#     fastest = _fastest_model()
#     return [fastest], {fastest: wait_results.get(fastest, float("inf"))}
#
#
# # -----------------------------
# # Public latency estimator
# # -----------------------------
# def estimate_selected_set_latency_for_mode(
#     cascade_mode: str,
#     selected_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
#     parallel_first_n: int,
# ) -> float:
#     mode = str(cascade_mode).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         vals = [get_stage_time(m, wait_results) for m in selected_models]
#
#         if not vals:
#             return float("inf")
#
#         return (
#             float(BASE_REQUEST_OVERHEAD)
#             + max(vals)
#             + float(FULL_PARALLEL_AGG_OVERHEAD)
#         )
#
#     if mode == "sequential":
#         ordered = reorder_for_sequential(selected_models, wait_results)
#         return estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#     ordered = reorder_for_sequential(selected_models, wait_results)
#     return estimate_expected_latency_parallel_first_finish(
#         ordered_models=ordered,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#
#
# # -----------------------------
# # Dispatcher
# # -----------------------------
# def build_model_set_by_mode(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     mode = str(cascade_mode).lower()
#     strategy = str(SELECTION_STRATEGY).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         return build_model_set_mode_s(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             alpha_now=alpha_now,
#             logger=logger,
#         )
#
#     if mode == "sequential":
#         if strategy == "exhaustive":
#             return build_model_set_sequential_exhaustive(
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=ensemble_size,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#
#         return build_model_set_sequential_greedy(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     if strategy == "exhaustive":
#         return build_model_set_parallel_first_exhaustive(
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     return build_model_set_parallel_first_greedy(
#         wait_results=wait_results,
#         deadline_ms=deadline_ms,
#         ensemble_size=ensemble_size,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#         logger=logger,
#     )
#
#
#
# # # ######
# # # # final working code 05-04-2026 end
# # # ########



# import itertools
# import random
# from typing import Dict, List, Tuple
#
# from ckn_controller.ckn_config import (
#     M_TOTAL,
#     POLICY,
#     K,
#     C,
#     MODEL_PROFILES,
#     MODEL_SIZES,
#     SELECTION_STRATEGY,
#     USE_DEADLINE_FAST_PATH,
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
#     SMALL_ENSEMBLE_SIZE,
#     LARGE_ENSEMBLE_SIZE,
#     BASELINE_MODE,
#     BASE_REQUEST_OVERHEAD,
#     SEQUENTIAL_STAGE_OVERHEAD,
#     PO2_FIRST_BATCH_OVERHEAD,
#     PO2_TAIL_OVERHEAD,
#     FULL_PARALLEL_AGG_OVERHEAD,
#     ET_SCALE_SEQ,
#     ET_OFFSET_SEQ,
#     ET_SCALE_P2,
#     ET_OFFSET_P2,
# )
#
#
# # -----------------------------
# # Basic helpers
# # -----------------------------
# def get_exit_probability(model_name: str) -> float:
#     try:
#         p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
#     except Exception:
#         p = 0.5
#     return max(0.0, min(1.0, p))
#
#
# def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
#     wait_t = float(wait_results.get(model_name, float("inf")))
#     return wait_t
#
#
# def calibrate_sequential_et(et_raw: float) -> float:
#     if et_raw == float("inf"):
#         return et_raw
#     return float(ET_OFFSET_SEQ) + float(ET_SCALE_SEQ) * et_raw
#
#
# def calibrate_power2_et(et_raw: float) -> float:
#     if et_raw == float("inf"):
#         return et_raw
#     return float(ET_OFFSET_P2) + float(ET_SCALE_P2) * et_raw
#
#
# def _sorted_by_size(models: List[str], reverse: bool = False) -> List[str]:
#     return sorted(models, key=lambda m: MODEL_SIZES[m], reverse=reverse)
#
#
# def _estimates_for(selected: List[str], wait_results: Dict[str, float]) -> Dict[str, float]:
#     return {m: wait_results.get(m, float("inf")) for m in selected}
#
#
# def _fastest_model(model_pool: List[str] = None) -> str:
#     pool = model_pool if model_pool else M_TOTAL
#     return min(pool, key=lambda m: float(MODEL_PROFILES[m]["latency"]))
#
#
# def build_maxms_for_pool(model_pool: List[str], max_models: int) -> Dict[int, int]:
#     sorted_by_size = sorted(model_pool, key=lambda m: MODEL_SIZES[m], reverse=True)
#
#     maxms: Dict[int, int] = {}
#     rolling_sum = 0
#
#     for i, m in enumerate(sorted_by_size):
#         rolling_sum += MODEL_SIZES[m]
#         size = i + 1
#
#         if size <= max_models:
#             maxms[size] = rolling_sum
#
#     return maxms
#
#
# def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
#     size = len(ms)
#     sum_bytes = sum(MODEL_SIZES[m] for m in ms)
#
#     if sum_bytes <= 0 or size not in maxms:
#         return None, sum_bytes
#
#     sc = maxms[size] / sum_bytes
#     return sc, sum_bytes
#
#
# def generate_combinations(models: List[str], max_size: int):
#     all_sets = []
#
#     for k in range(1, max_size + 1):
#         all_sets.extend(list(itertools.combinations(models, k)))
#
#     return all_sets
#
#
# # -----------------------------
# # Sequential / E[T] helpers
# # -----------------------------
# def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
#     def rank(m: str):
#         p_exit = get_exit_probability(m)
#         t = max(get_stage_time(m, wait_results), 1e-9)
#         return p_exit / t
#
#     return sorted(models, key=rank, reverse=True)
#
#
# def estimate_expected_latency_sequential_raw(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     if not ordered_models:
#         return 0.0
#
#     expected_time = float(BASE_REQUEST_OVERHEAD)
#     prob_reach = 1.0
#
#     for i, m in enumerate(ordered_models):
#         stage_time = get_stage_time(m, wait_results)
#
#         if stage_time == float("inf"):
#             return float("inf")
#
#         if i > 0:
#             stage_time += float(network_latency) + float(SEQUENTIAL_STAGE_OVERHEAD)
#
#         expected_time += prob_reach * stage_time
#         prob_reach *= (1.0 - get_exit_probability(m))
#
#     return expected_time
#
#
# def estimate_expected_latency_sequential(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
# ) -> float:
#     et_raw = estimate_expected_latency_sequential_raw(
#         ordered_models=ordered_models,
#         wait_results=wait_results,
#         network_latency=network_latency,
#     )
#     return calibrate_sequential_et(et_raw)
#
#
# # -----------------------------
# # Power-of-two / E[T] helpers
# # -----------------------------
# def estimate_batch_exit_probability(batch_models: List[str]) -> float:
#     if not batch_models:
#         return 0.0
#
#     fail_prob = 1.0
#
#     for m in batch_models:
#         fail_prob *= (1.0 - get_exit_probability(m))
#
#     return 1.0 - fail_prob
#
#
# def estimate_first_batch_latency(
#     batch_models: List[str],
#     wait_results: Dict[str, float],
# ) -> float:
#     if not batch_models:
#         return float("inf")
#
#     batch_with_times = []
#
#     for m in batch_models:
#         t = get_stage_time(m, wait_results)
#
#         if t == float("inf"):
#             return float("inf")
#
#         batch_with_times.append((m, t))
#
#     batch_with_times.sort(key=lambda x: x[1])
#
#     first_model, t_fast = batch_with_times[0]
#     t_slow = max(t for _, t in batch_with_times)
#
#     p_first_pass = get_exit_probability(first_model)
#
#     return (p_first_pass * t_fast) + ((1.0 - p_first_pass) * t_slow)
#
#
# def estimate_expected_latency_parallel_first_finish_raw(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     if not ordered_models:
#         return float("inf")
#
#     first_n = max(1, min(int(parallel_first_n), len(ordered_models)))
#
#     first_batch = ordered_models[:first_n]
#     tail_models = ordered_models[first_n:]
#
#     first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)
#
#     if first_batch_latency == float("inf"):
#         return float("inf")
#
#     p_stop_after_first_batch = estimate_batch_exit_probability(first_batch)
#     p_go_to_tail = 1.0 - p_stop_after_first_batch
#
#     raw_et = (
#         float(BASE_REQUEST_OVERHEAD)
#         + float(PO2_FIRST_BATCH_OVERHEAD)
#         + first_batch_latency
#     )
#
#     if tail_models:
#         tail_latency_raw = estimate_expected_latency_sequential_raw(
#             ordered_models=tail_models,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         if tail_latency_raw == float("inf"):
#             return float("inf")
#
#         raw_et += p_go_to_tail * (
#             float(network_latency)
#             + float(PO2_TAIL_OVERHEAD)
#             + tail_latency_raw
#         )
#
#     return raw_et
#
#
# def estimate_expected_latency_parallel_first_finish(
#     ordered_models: List[str],
#     wait_results: Dict[str, float],
#     parallel_first_n: int,
#     network_latency: float,
# ) -> float:
#     et_raw = estimate_expected_latency_parallel_first_finish_raw(
#         ordered_models=ordered_models,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#     return calibrate_power2_et(et_raw)
#
#
# # -----------------------------
# # Generic full-parallel selector for a model pool
# # -----------------------------
# def build_model_set_mode_s_from_pool(
#     model_pool: List[str],
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_parallel_models = min(ensemble_size, K // C, len(model_pool))
#
#     if max_parallel_models <= 0:
#         return [], {}
#
#     maxms = build_maxms_for_pool(model_pool, max_parallel_models)
#
#     def estimate_te(ms: List[str]) -> float:
#         times = []
#
#         for m in ms:
#             stage_time = get_stage_time(m, wait_results)
#             times.append(stage_time)
#
#         if not times:
#             return float("inf")
#
#         return (
#             float(BASE_REQUEST_OVERHEAD)
#             + max(times)
#             + float(FULL_PARALLEL_AGG_OVERHEAD)
#         )
#
#     def latency_term_for(ms: List[str]):
#         te = estimate_te(ms)
#         raw = te / d_sec
#
#         if raw >= 1.0:
#             return None, te, raw
#
#         lt = min(max(raw, 0.1), 0.9)
#         return lt, te, raw
#
#     if POLICY.startswith("best_acc_"):
#         k = int(POLICY.split("_")[-1])
#         k = min(k, max_parallel_models, len(model_pool))
#
#         sorted_by_acc = sorted(
#             model_pool,
#             key=lambda m: float(MODEL_PROFILES.get(m, {}).get("accuracy", float("-inf"))),
#             reverse=True,
#         )
#
#         selected = sorted_by_acc[:k]
#         return selected, _estimates_for(selected, wait_results)
#
#     if POLICY == "randomized":
#         k = max_parallel_models
#         selected = random.sample(model_pool, k)
#         return selected, _estimates_for(selected, wait_results)
#
#     selected: List[str] = []
#     remaining = list(model_pool)
#
#     for _ in range(max_parallel_models):
#         best_candidate = None
#         best_cost = float("inf")
#
#         for m in remaining:
#             trial = selected + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             lt, _, _ = latency_term_for(trial)
#
#             if lt is None:
#                 continue
#
#             sc, _ = size_cost_for_models(trial, maxms)
#
#             if sc is None:
#                 continue
#
#             cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)
#
#             if cost < best_cost:
#                 best_cost = cost
#                 best_candidate = m
#
#         if best_candidate is None:
#             break
#
#         selected.append(best_candidate)
#         remaining.remove(best_candidate)
#
#     if not selected:
#         fastest = _fastest_model(model_pool)
#         selected = [fastest]
#
#     return selected, _estimates_for(selected, wait_results)
#
#
# # -----------------------------
# # Generic sequential selector for a model pool
# # -----------------------------
# def build_model_set_sequential_from_pool(
#     model_pool: List[str],
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(model_pool))
#
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in model_pool:
#         ordered = reorder_for_sequential([m], wait_results)
#
#         et = estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model(model_pool)
#         return [fastest], _estimates_for([fastest], wait_results)
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in model_pool if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#
#             et = estimate_expected_latency_sequential(
#                 ordered_models=ordered,
#                 wait_results=wait_results,
#                 network_latency=network_latency,
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in model_pool if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, _estimates_for(chosen_ms, wait_results)
#
#         fastest = _fastest_model(model_pool)
#         return [fastest], _estimates_for([fastest], wait_results)
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, _estimates_for(chosen_ms, wait_results)
#
#
# # -----------------------------
# # Generic power-of-two selector for a model pool
# # -----------------------------
# def build_model_set_parallel_first_from_pool(
#     model_pool: List[str],
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float]]:
#     d_sec = deadline_ms / 1000.0
#
#     if d_sec <= 0:
#         return [], {}
#
#     max_models = min(ensemble_size, K // C, len(model_pool))
#
#     if max_models <= 0:
#         return [], {}
#
#     built_sets: List[Tuple[List[str], float]] = []
#
#     best_single = None
#     best_single_et = float("inf")
#
#     for m in model_pool:
#         ordered = reorder_for_sequential([m], wait_results)
#
#         et = estimate_expected_latency_parallel_first_finish(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#         )
#
#         if et < best_single_et:
#             best_single_et = et
#             best_single = ordered
#
#     if best_single is None:
#         fastest = _fastest_model(model_pool)
#         return [fastest], _estimates_for([fastest], wait_results)
#
#     current_best = list(best_single)
#     built_sets.append((list(current_best), best_single_et))
#
#     remaining = [m for m in model_pool if m not in current_best]
#
#     while len(current_best) < max_models and remaining:
#         best_trial = None
#         best_trial_et = float("inf")
#
#         for m in remaining:
#             trial = current_best + [m]
#
#             if len(trial) * C > K:
#                 continue
#
#             ordered = reorder_for_sequential(trial, wait_results)
#
#             et = estimate_expected_latency_parallel_first_finish(
#                 ordered_models=ordered,
#                 wait_results=wait_results,
#                 parallel_first_n=parallel_first_n,
#                 network_latency=network_latency,
#             )
#
#             if et < best_trial_et:
#                 best_trial_et = et
#                 best_trial = ordered
#
#         if best_trial is None:
#             break
#
#         current_best = list(best_trial)
#         built_sets.append((list(current_best), best_trial_et))
#         remaining = [m for m in model_pool if m not in current_best]
#
#     if USE_DEADLINE_FAST_PATH:
#         feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]
#
#         if feasible_sets:
#             feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
#             chosen_ms, _ = feasible_sets[0]
#             return chosen_ms, _estimates_for(chosen_ms, wait_results)
#
#         fastest = _fastest_model(model_pool)
#         return [fastest], _estimates_for([fastest], wait_results)
#
#     chosen_ms, _ = built_sets[-1]
#     return chosen_ms, _estimates_for(chosen_ms, wait_results)
#
#
# # # -----------------------------
# # # Instance-aware baselines
# # # -----------------------------
# # def build_instance_aware_baseline(
# #     wait_results: Dict[str, float],
# #     logger=None,
# # ) -> Tuple[List[str], Dict[str, float], str]:
# #     baseline = str(BASELINE_MODE).lower()
# #
# #     if baseline == "none":
# #         return [], {}, ""
# #
# #     small_models = _sorted_by_size(SMALL_INSTANCE_MODELS)
# #     large_models = _sorted_by_size(LARGE_INSTANCE_MODELS)
# #
# #     if not small_models or not large_models:
# #         raise ValueError(
# #             f"Invalid instance model split. "
# #             f"SMALL_INSTANCE_MODELS={small_models}, LARGE_INSTANCE_MODELS={large_models}"
# #         )
# #
# #     if baseline == "smallest_to_largest":
# #         selected = [
# #             small_models[0],
# #             large_models[-1],
# #         ]
# #         return selected, _estimates_for(selected, wait_results), "sequential"
# #
# #     if baseline == "random_to_random":
# #         selected = [
# #             random.choice(small_models),
# #             random.choice(large_models),
# #         ]
# #         return selected, _estimates_for(selected, wait_results), "sequential"
# #
# #     if baseline == "smallest_to_ensemble_3":
# #         if len(large_models) < 3:
# #             raise ValueError(
# #                 "BASELINE_MODE='smallest_to_ensemble_3' requires at least "
# #                 "3 models in LARGE_INSTANCE_MODELS."
# #             )
# #
# #         selected = [
# #             small_models[0],
# #             *large_models[-3:],
# #         ]
# #         return selected, _estimates_for(selected, wait_results), "full_parallel"
# #
# #     if baseline == "power2_smallest_to_largest":
# #         if len(small_models) < 2:
# #             raise ValueError(
# #                 "BASELINE_MODE='power2_smallest_to_largest' requires at least "
# #                 "2 models in SMALL_INSTANCE_MODELS."
# #             )
# #
# #         selected = [
# #             small_models[0],
# #             small_models[1],
# #             large_models[-1],
# #         ]
# #         return selected, _estimates_for(selected, wait_results), "parallel_first_finish"
# #
# #     raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")
#
# # -----------------------------
# # Instance-aware baselines
# # -----------------------------
# def build_instance_aware_baseline(
#     wait_results: Dict[str, float],
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float], str]:
#     baseline = str(BASELINE_MODE).lower()
#
#     if baseline == "none":
#         return [], {}, ""
#
#     small_models = _sorted_by_size(SMALL_INSTANCE_MODELS)
#     large_models = _sorted_by_size(LARGE_INSTANCE_MODELS)
#
#     if not small_models or not large_models:
#         raise ValueError(
#             f"Invalid instance model split. "
#             f"SMALL_INSTANCE_MODELS={small_models}, LARGE_INSTANCE_MODELS={large_models}"
#         )
#
#     if baseline == "smallest_to_largest":
#         selected = [
#             small_models[0],
#             large_models[-1],
#         ]
#         return selected, _estimates_for(selected, wait_results), "sequential"
#
#     if baseline == "random_to_random":
#         selected = [
#             random.choice(small_models),
#             random.choice(large_models),
#         ]
#         return selected, _estimates_for(selected, wait_results), "sequential"
#
#     if baseline == "smallest_to_ensemble_3":
#         if len(large_models) < 3:
#             raise ValueError(
#                 "BASELINE_MODE='smallest_to_ensemble_3' requires at least "
#                 "3 models in LARGE_INSTANCE_MODELS."
#             )
#
#         selected = [
#             small_models[0],
#             *large_models[-3:],
#         ]
#         return selected, _estimates_for(selected, wait_results), "full_parallel"
#     if baseline == "full_ensemble_edge":
#         selected = list(small_models[:3])
#
#         if not selected:
#             raise ValueError(
#                 "BASELINE_MODE='full_ensemble_edge' requires at least "
#                 "one model in SMALL_INSTANCE_MODELS."
#             )
#
#         return selected, _estimates_for(selected, wait_results), "full_parallel"
#
#     if baseline == "full_ensemble_cloud":
#         selected = list(large_models[-3:])
#
#         if not selected:
#             raise ValueError(
#                 "BASELINE_MODE='full_ensemble_cloud' requires at least "
#                 "one model in LARGE_INSTANCE_MODELS."
#             )
#
#         return selected, _estimates_for(selected, wait_results), "full_parallel"
#
#     if baseline == "power2_smallest_to_largest":
#         if len(small_models) < 2:
#             raise ValueError(
#                 "BASELINE_MODE='power2_smallest_to_largest' requires at least "
#                 "2 models in SMALL_INSTANCE_MODELS."
#             )
#
#         selected = [
#             small_models[0],
#             small_models[1],
#             large_models[-1],
#         ]
#         return selected, _estimates_for(selected, wait_results), "parallel_first_finish"
#
#     # -----------------------------
#     # Speculative cloud baselines
#     # -----------------------------
#     if baseline == "spec_b_smallest_to_largest":
#         selected = [
#             small_models[0],   # edge/smallest model
#             large_models[-1],  # cloud/largest model
#         ]
#         return selected, _estimates_for(selected, wait_results), "spec_b"
#
#     if baseline == "spec_b_random_to_random":
#         selected = [
#             random.choice(small_models),  # edge random model
#             random.choice(large_models),  # cloud random model
#         ]
#         return selected, _estimates_for(selected, wait_results), "spec_b"
#
#     raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")
#
#
# # -----------------------------
# # Instance-aware MODE-S selector
# # -----------------------------
# def build_instance_aware_model_set(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float], str]:
#     mode = str(cascade_mode).lower()
#
#     small_count = min(int(SMALL_ENSEMBLE_SIZE), len(SMALL_INSTANCE_MODELS))
#     large_count = min(int(LARGE_ENSEMBLE_SIZE), len(LARGE_INSTANCE_MODELS))
#
#     selected: List[str] = []
#
#     if small_count > 0:
#         if mode in ("mode_s", "full_parallel"):
#             small_selected, _ = build_model_set_mode_s_from_pool(
#                 model_pool=SMALL_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=small_count,
#                 alpha_now=alpha_now,
#                 logger=logger,
#             )
#         elif mode == "sequential":
#             small_selected, _ = build_model_set_sequential_from_pool(
#                 model_pool=SMALL_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=small_count,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#         else:
#             small_selected, _ = build_model_set_parallel_first_from_pool(
#                 model_pool=SMALL_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=small_count,
#                 parallel_first_n=parallel_first_n,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#
#         selected.extend(small_selected[:small_count])
#
#     if large_count > 0:
#         if mode in ("mode_s", "full_parallel"):
#             large_selected, _ = build_model_set_mode_s_from_pool(
#                 model_pool=LARGE_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=large_count,
#                 alpha_now=alpha_now,
#                 logger=logger,
#             )
#         elif mode == "sequential":
#             large_selected, _ = build_model_set_sequential_from_pool(
#                 model_pool=LARGE_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=large_count,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#         else:
#             large_selected, _ = build_model_set_parallel_first_from_pool(
#                 model_pool=LARGE_INSTANCE_MODELS,
#                 wait_results=wait_results,
#                 deadline_ms=deadline_ms,
#                 ensemble_size=large_count,
#                 parallel_first_n=parallel_first_n,
#                 network_latency=network_latency,
#                 logger=logger,
#             )
#
#         selected.extend(large_selected[:large_count])
#
#     selected = list(dict.fromkeys(selected))
#
#     return selected, _estimates_for(selected, wait_results), mode
#
#
# # -----------------------------
# # Public latency estimator
# # -----------------------------
# def estimate_selected_set_latency_for_mode(
#     cascade_mode: str,
#     selected_models: List[str],
#     wait_results: Dict[str, float],
#     network_latency: float,
#     parallel_first_n: int,
# ) -> float:
#     mode = str(cascade_mode).lower()
#
#     if mode in ("mode_s", "full_parallel"):
#         vals = [get_stage_time(m, wait_results) for m in selected_models]
#
#         if not vals:
#             return float("inf")
#
#         return (
#             float(BASE_REQUEST_OVERHEAD)
#             + max(vals)
#             + float(FULL_PARALLEL_AGG_OVERHEAD)
#         )
#
#     if mode == "sequential":
#         ordered = reorder_for_sequential(selected_models, wait_results)
#         return estimate_expected_latency_sequential(
#             ordered_models=ordered,
#             wait_results=wait_results,
#             network_latency=network_latency,
#         )
#
#     ordered = selected_models
#     return estimate_expected_latency_parallel_first_finish(
#         ordered_models=ordered,
#         wait_results=wait_results,
#         parallel_first_n=parallel_first_n,
#         network_latency=network_latency,
#     )
#
#
# # -----------------------------
# # Dispatcher
# # -----------------------------
# def build_model_set_by_mode(
#     cascade_mode: str,
#     wait_results: Dict[str, float],
#     deadline_ms: int,
#     ensemble_size: int,
#     alpha_now: float,
#     parallel_first_n: int,
#     network_latency: float,
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float], str]:
#
#     baseline_models, baseline_estimates, baseline_exec_mode = build_instance_aware_baseline(
#         wait_results=wait_results,
#         logger=logger,
#     )
#
#     if baseline_models:
#         if logger:
#             logger.info(
#                 f"[BASELINE_SELECT] baseline_mode={BASELINE_MODE} "
#                 f"selected_models={baseline_models} "
#                 f"effective_cascade_mode={baseline_exec_mode}"
#             )
#
#         return baseline_models, baseline_estimates, baseline_exec_mode
#
#     if USE_TWO_ILUVATAR_INSTANCES:
#         selected, estimates, effective_mode = build_instance_aware_model_set(
#             cascade_mode=cascade_mode,
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             alpha_now=alpha_now,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#         if selected:
#             if logger:
#                 logger.info(
#                     f"[INSTANCE_AWARE_SELECT] "
#                     f"small_ensemble_size={SMALL_ENSEMBLE_SIZE} "
#                     f"large_ensemble_size={LARGE_ENSEMBLE_SIZE} "
#                     f"selected_models={selected} "
#                     f"effective_cascade_mode={effective_mode}"
#                 )
#
#             return selected, estimates, effective_mode
#
#     mode = str(cascade_mode).lower()
#
#     selected, estimates = build_model_set_mode_s_from_pool(
#         model_pool=M_TOTAL,
#         wait_results=wait_results,
#         deadline_ms=deadline_ms,
#         ensemble_size=ensemble_size,
#         alpha_now=alpha_now,
#         logger=logger,
#     )
#
#     if mode == "sequential":
#         selected, estimates = build_model_set_sequential_from_pool(
#             model_pool=M_TOTAL,
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     elif mode not in ("mode_s", "full_parallel"):
#         selected, estimates = build_model_set_parallel_first_from_pool(
#             model_pool=M_TOTAL,
#             wait_results=wait_results,
#             deadline_ms=deadline_ms,
#             ensemble_size=ensemble_size,
#             parallel_first_n=parallel_first_n,
#             network_latency=network_latency,
#             logger=logger,
#         )
#
#     return selected, estimates, mode




import itertools
import random
from typing import Dict, List, Tuple, Optional, Set

from ckn_controller import ckn_config as config
from ckn_controller.ckn_config import (
    M_TOTAL,
    POLICY,
    K,
    C,
    MODEL_PROFILES,
    MODEL_SIZES,
    SELECTION_STRATEGY,
    USE_DEADLINE_FAST_PATH,
    USE_TWO_ILUVATAR_INSTANCES,
    SMALL_INSTANCE_MODELS,
    LARGE_INSTANCE_MODELS,
    SMALL_ENSEMBLE_SIZE,
    LARGE_ENSEMBLE_SIZE,
    BASELINE_MODE,
    BASE_REQUEST_OVERHEAD,
    SEQUENTIAL_STAGE_OVERHEAD,
    PO2_FIRST_BATCH_OVERHEAD,
    PO2_TAIL_OVERHEAD,
    FULL_PARALLEL_AGG_OVERHEAD,
    ET_SCALE_SEQ,
    ET_OFFSET_SEQ,
    ET_SCALE_P2,
    ET_OFFSET_P2,
)


# ============================================================
# Multi-edge / multi-cloud compatibility
# ============================================================

USE_MULTI_ILUVATAR_INSTANCES = bool(
    getattr(config, "USE_MULTI_ILUVATAR_INSTANCES", False)
)

ILUVATAR_NODES = getattr(config, "ILUVATAR_NODES", {})


def _derive_model_pool_from_nodes(node_type: str) -> List[str]:
    """
    Build a unique model pool from all enabled nodes of one type.

    Example:
        node_type="edge"  -> all models deployed on enabled edge nodes
        node_type="cloud" -> all models deployed on enabled cloud nodes
    """
    if not USE_MULTI_ILUVATAR_INSTANCES:
        return []

    if not isinstance(ILUVATAR_NODES, dict):
        raise ValueError("ILUVATAR_NODES must be a dictionary.")

    models: List[str] = []
    seen: Set[str] = set()

    for node_id, node in ILUVATAR_NODES.items():
        if not isinstance(node, dict):
            raise ValueError(
                f"Configuration for node '{node_id}' must be a dictionary."
            )

        if not bool(node.get("enabled", True)):
            continue

        configured_type = str(node.get("type", "")).lower()

        if configured_type != str(node_type).lower():
            continue

        node_models = node.get("models", [])

        if not isinstance(node_models, list):
            raise ValueError(
                f"Node '{node_id}' field 'models' must be a list."
            )

        for model_name in node_models:
            if model_name not in seen:
                models.append(model_name)
                seen.add(model_name)

    return models


def get_edge_model_pool() -> List[str]:
    """Return the active edge model pool."""
    if USE_MULTI_ILUVATAR_INSTANCES:
        return _derive_model_pool_from_nodes("edge")

    return list(SMALL_INSTANCE_MODELS)


def get_cloud_model_pool() -> List[str]:
    """Return the active cloud model pool."""
    if USE_MULTI_ILUVATAR_INSTANCES:
        return _derive_model_pool_from_nodes("cloud")

    return list(LARGE_INSTANCE_MODELS)


def get_available_models(
    model_pool: List[str],
    wait_results: Dict[str, float],
) -> List[str]:
    """
    Keep models that have a finite node-selected wait estimate.

    ckn_main.py already converts node-level wait estimates into:
        wait_results[model] = best available estimate across eligible nodes.
    """
    available: List[str] = []

    for model_name in model_pool:
        wait_value = float(
            wait_results.get(model_name, float("inf"))
        )

        if wait_value != float("inf"):
            available.append(model_name)

    return available


# -----------------------------
# Basic helpers
# -----------------------------
def get_exit_probability(model_name: str) -> float:
    try:
        p = float(MODEL_PROFILES[model_name].get("accuracy", 0.5))
    except Exception:
        p = 0.5
    return max(0.0, min(1.0, p))


def get_stage_time(model_name: str, wait_results: Dict[str, float]) -> float:
    wait_t = float(wait_results.get(model_name, float("inf")))
    return wait_t


def calibrate_sequential_et(et_raw: float) -> float:
    if et_raw == float("inf"):
        return et_raw
    return float(ET_OFFSET_SEQ) + float(ET_SCALE_SEQ) * et_raw


def calibrate_power2_et(et_raw: float) -> float:
    if et_raw == float("inf"):
        return et_raw
    return float(ET_OFFSET_P2) + float(ET_SCALE_P2) * et_raw


def _sorted_by_size(models: List[str], reverse: bool = False) -> List[str]:
    return sorted(models, key=lambda m: MODEL_SIZES[m], reverse=reverse)


def _estimates_for(selected: List[str], wait_results: Dict[str, float]) -> Dict[str, float]:
    return {m: wait_results.get(m, float("inf")) for m in selected}


def _fastest_model(model_pool: List[str] = None) -> str:
    pool = model_pool if model_pool else M_TOTAL
    return min(pool, key=lambda m: float(MODEL_PROFILES[m]["latency"]))


def build_maxms_for_pool(model_pool: List[str], max_models: int) -> Dict[int, int]:
    sorted_by_size = sorted(model_pool, key=lambda m: MODEL_SIZES[m], reverse=True)

    maxms: Dict[int, int] = {}
    rolling_sum = 0

    for i, m in enumerate(sorted_by_size):
        rolling_sum += MODEL_SIZES[m]
        size = i + 1

        if size <= max_models:
            maxms[size] = rolling_sum

    return maxms


def size_cost_for_models(ms: List[str], maxms: Dict[int, int]):
    size = len(ms)
    sum_bytes = sum(MODEL_SIZES[m] for m in ms)

    if sum_bytes <= 0 or size not in maxms:
        return None, sum_bytes

    sc = maxms[size] / sum_bytes
    return sc, sum_bytes


def generate_combinations(models: List[str], max_size: int):
    all_sets = []

    for k in range(1, max_size + 1):
        all_sets.extend(list(itertools.combinations(models, k)))

    return all_sets


# -----------------------------
# Sequential / E[T] helpers
# -----------------------------
def reorder_for_sequential(models: List[str], wait_results: Dict[str, float]) -> List[str]:
    def rank(m: str):
        p_exit = get_exit_probability(m)
        t = max(get_stage_time(m, wait_results), 1e-9)
        return p_exit / t

    return sorted(models, key=rank, reverse=True)


def estimate_expected_latency_sequential_raw(
    ordered_models: List[str],
    wait_results: Dict[str, float],
    network_latency: float,
) -> float:
    if not ordered_models:
        return 0.0

    expected_time = float(BASE_REQUEST_OVERHEAD)
    prob_reach = 1.0

    for i, m in enumerate(ordered_models):
        stage_time = get_stage_time(m, wait_results)

        if stage_time == float("inf"):
            return float("inf")

        if i > 0:
            stage_time += float(network_latency) + float(SEQUENTIAL_STAGE_OVERHEAD)

        expected_time += prob_reach * stage_time
        prob_reach *= (1.0 - get_exit_probability(m))

    return expected_time


def estimate_expected_latency_sequential(
    ordered_models: List[str],
    wait_results: Dict[str, float],
    network_latency: float,
) -> float:
    et_raw = estimate_expected_latency_sequential_raw(
        ordered_models=ordered_models,
        wait_results=wait_results,
        network_latency=network_latency,
    )
    return calibrate_sequential_et(et_raw)


# -----------------------------
# Power-of-two / E[T] helpers
# -----------------------------
def estimate_batch_exit_probability(batch_models: List[str]) -> float:
    if not batch_models:
        return 0.0

    fail_prob = 1.0

    for m in batch_models:
        fail_prob *= (1.0 - get_exit_probability(m))

    return 1.0 - fail_prob


def estimate_first_batch_latency(
    batch_models: List[str],
    wait_results: Dict[str, float],
) -> float:
    if not batch_models:
        return float("inf")

    batch_with_times = []

    for m in batch_models:
        t = get_stage_time(m, wait_results)

        if t == float("inf"):
            return float("inf")

        batch_with_times.append((m, t))

    batch_with_times.sort(key=lambda x: x[1])

    first_model, t_fast = batch_with_times[0]
    t_slow = max(t for _, t in batch_with_times)

    p_first_pass = get_exit_probability(first_model)

    return (p_first_pass * t_fast) + ((1.0 - p_first_pass) * t_slow)


def estimate_expected_latency_parallel_first_finish_raw(
    ordered_models: List[str],
    wait_results: Dict[str, float],
    parallel_first_n: int,
    network_latency: float,
) -> float:
    if not ordered_models:
        return float("inf")

    first_n = max(1, min(int(parallel_first_n), len(ordered_models)))

    first_batch = ordered_models[:first_n]
    tail_models = ordered_models[first_n:]

    first_batch_latency = estimate_first_batch_latency(first_batch, wait_results)

    if first_batch_latency == float("inf"):
        return float("inf")

    p_stop_after_first_batch = estimate_batch_exit_probability(first_batch)
    p_go_to_tail = 1.0 - p_stop_after_first_batch

    raw_et = (
        float(BASE_REQUEST_OVERHEAD)
        + float(PO2_FIRST_BATCH_OVERHEAD)
        + first_batch_latency
    )

    if tail_models:
        tail_latency_raw = estimate_expected_latency_sequential_raw(
            ordered_models=tail_models,
            wait_results=wait_results,
            network_latency=network_latency,
        )

        if tail_latency_raw == float("inf"):
            return float("inf")

        raw_et += p_go_to_tail * (
            float(network_latency)
            + float(PO2_TAIL_OVERHEAD)
            + tail_latency_raw
        )

    return raw_et


def estimate_expected_latency_parallel_first_finish(
    ordered_models: List[str],
    wait_results: Dict[str, float],
    parallel_first_n: int,
    network_latency: float,
) -> float:
    et_raw = estimate_expected_latency_parallel_first_finish_raw(
        ordered_models=ordered_models,
        wait_results=wait_results,
        parallel_first_n=parallel_first_n,
        network_latency=network_latency,
    )
    return calibrate_power2_et(et_raw)


# -----------------------------
# Generic full-parallel selector for a model pool
# -----------------------------
def build_model_set_mode_s_from_pool(
    model_pool: List[str],
    wait_results: Dict[str, float],
    deadline_ms: int,
    ensemble_size: int,
    alpha_now: float,
    logger=None,
) -> Tuple[List[str], Dict[str, float]]:
    d_sec = deadline_ms / 1000.0

    if d_sec <= 0:
        return [], {}

    max_parallel_models = min(ensemble_size, K // C, len(model_pool))

    if max_parallel_models <= 0:
        return [], {}

    maxms = build_maxms_for_pool(model_pool, max_parallel_models)

    def estimate_te(ms: List[str]) -> float:
        times = []

        for m in ms:
            stage_time = get_stage_time(m, wait_results)
            times.append(stage_time)

        if not times:
            return float("inf")

        return (
            float(BASE_REQUEST_OVERHEAD)
            + max(times)
            + float(FULL_PARALLEL_AGG_OVERHEAD)
        )

    def latency_term_for(ms: List[str]):
        te = estimate_te(ms)
        raw = te / d_sec

        if raw >= 1.0:
            return None, te, raw

        lt = min(max(raw, 0.1), 0.9)
        return lt, te, raw

    if POLICY.startswith("best_acc_"):
        k = int(POLICY.split("_")[-1])
        k = min(k, max_parallel_models, len(model_pool))

        sorted_by_acc = sorted(
            model_pool,
            key=lambda m: float(MODEL_PROFILES.get(m, {}).get("accuracy", float("-inf"))),
            reverse=True,
        )

        selected = sorted_by_acc[:k]
        return selected, _estimates_for(selected, wait_results)

    if POLICY == "randomized":
        k = max_parallel_models
        selected = random.sample(model_pool, k)
        return selected, _estimates_for(selected, wait_results)

    selected: List[str] = []
    remaining = list(model_pool)

    for _ in range(max_parallel_models):
        best_candidate = None
        best_cost = float("inf")

        for m in remaining:
            trial = selected + [m]

            if len(trial) * C > K:
                continue

            lt, _, _ = latency_term_for(trial)

            if lt is None:
                continue

            sc, _ = size_cost_for_models(trial, maxms)

            if sc is None:
                continue

            cost = ((1.0 - alpha_now) * lt) + (alpha_now * sc)

            if cost < best_cost:
                best_cost = cost
                best_candidate = m

        if best_candidate is None:
            break

        selected.append(best_candidate)
        remaining.remove(best_candidate)

    if not selected:
        fastest = _fastest_model(model_pool)
        selected = [fastest]

    return selected, _estimates_for(selected, wait_results)


# -----------------------------
# Generic sequential selector for a model pool
# -----------------------------
def build_model_set_sequential_from_pool(
    model_pool: List[str],
    wait_results: Dict[str, float],
    deadline_ms: int,
    ensemble_size: int,
    network_latency: float,
    logger=None,
) -> Tuple[List[str], Dict[str, float]]:
    d_sec = deadline_ms / 1000.0

    if d_sec <= 0:
        return [], {}

    max_models = min(ensemble_size, K // C, len(model_pool))

    if max_models <= 0:
        return [], {}

    built_sets: List[Tuple[List[str], float]] = []

    best_single = None
    best_single_et = float("inf")

    for m in model_pool:
        ordered = reorder_for_sequential([m], wait_results)

        et = estimate_expected_latency_sequential(
            ordered_models=ordered,
            wait_results=wait_results,
            network_latency=network_latency,
        )

        if et < best_single_et:
            best_single_et = et
            best_single = ordered

    if best_single is None:
        fastest = _fastest_model(model_pool)
        return [fastest], _estimates_for([fastest], wait_results)

    current_best = list(best_single)
    built_sets.append((list(current_best), best_single_et))

    remaining = [m for m in model_pool if m not in current_best]

    while len(current_best) < max_models and remaining:
        best_trial = None
        best_trial_et = float("inf")

        for m in remaining:
            trial = current_best + [m]

            if len(trial) * C > K:
                continue

            ordered = reorder_for_sequential(trial, wait_results)

            et = estimate_expected_latency_sequential(
                ordered_models=ordered,
                wait_results=wait_results,
                network_latency=network_latency,
            )

            if et < best_trial_et:
                best_trial_et = et
                best_trial = ordered

        if best_trial is None:
            break

        current_best = list(best_trial)
        built_sets.append((list(current_best), best_trial_et))
        remaining = [m for m in model_pool if m not in current_best]

    if USE_DEADLINE_FAST_PATH:
        feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]

        if feasible_sets:
            feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
            chosen_ms, _ = feasible_sets[0]
            return chosen_ms, _estimates_for(chosen_ms, wait_results)

        fastest = _fastest_model(model_pool)
        return [fastest], _estimates_for([fastest], wait_results)

    chosen_ms, _ = built_sets[-1]
    return chosen_ms, _estimates_for(chosen_ms, wait_results)


# -----------------------------
# Generic power-of-two selector for a model pool
# -----------------------------
def build_model_set_parallel_first_from_pool(
    model_pool: List[str],
    wait_results: Dict[str, float],
    deadline_ms: int,
    ensemble_size: int,
    parallel_first_n: int,
    network_latency: float,
    logger=None,
) -> Tuple[List[str], Dict[str, float]]:
    d_sec = deadline_ms / 1000.0

    if d_sec <= 0:
        return [], {}

    max_models = min(ensemble_size, K // C, len(model_pool))

    if max_models <= 0:
        return [], {}

    built_sets: List[Tuple[List[str], float]] = []

    best_single = None
    best_single_et = float("inf")

    for m in model_pool:
        ordered = reorder_for_sequential([m], wait_results)

        et = estimate_expected_latency_parallel_first_finish(
            ordered_models=ordered,
            wait_results=wait_results,
            parallel_first_n=parallel_first_n,
            network_latency=network_latency,
        )

        if et < best_single_et:
            best_single_et = et
            best_single = ordered

    if best_single is None:
        fastest = _fastest_model(model_pool)
        return [fastest], _estimates_for([fastest], wait_results)

    current_best = list(best_single)
    built_sets.append((list(current_best), best_single_et))

    remaining = [m for m in model_pool if m not in current_best]

    while len(current_best) < max_models and remaining:
        best_trial = None
        best_trial_et = float("inf")

        for m in remaining:
            trial = current_best + [m]

            if len(trial) * C > K:
                continue

            ordered = reorder_for_sequential(trial, wait_results)

            et = estimate_expected_latency_parallel_first_finish(
                ordered_models=ordered,
                wait_results=wait_results,
                parallel_first_n=parallel_first_n,
                network_latency=network_latency,
            )

            if et < best_trial_et:
                best_trial_et = et
                best_trial = ordered

        if best_trial is None:
            break

        current_best = list(best_trial)
        built_sets.append((list(current_best), best_trial_et))
        remaining = [m for m in model_pool if m not in current_best]

    if USE_DEADLINE_FAST_PATH:
        feasible_sets = [(ms, et) for ms, et in built_sets if et < d_sec]

        if feasible_sets:
            feasible_sets.sort(key=lambda x: (len(x[0]), -x[1]), reverse=True)
            chosen_ms, _ = feasible_sets[0]
            return chosen_ms, _estimates_for(chosen_ms, wait_results)

        fastest = _fastest_model(model_pool)
        return [fastest], _estimates_for([fastest], wait_results)

    chosen_ms, _ = built_sets[-1]
    return chosen_ms, _estimates_for(chosen_ms, wait_results)


# # -----------------------------
# # Instance-aware baselines
# # -----------------------------
# def build_instance_aware_baseline(
#     wait_results: Dict[str, float],
#     logger=None,
# ) -> Tuple[List[str], Dict[str, float], str]:
#     baseline = str(BASELINE_MODE).lower()
#
#     if baseline == "none":
#         return [], {}, ""
#
#     edge_models = _sorted_by_size(get_edge_model_pool())
#     cloud_models = _sorted_by_size(get_cloud_model_pool())
#
#     if not edge_models or not cloud_models:
#         raise ValueError(
#             f"Invalid instance model split. "
#             f"SMALL_INSTANCE_MODELS={edge_models}, LARGE_INSTANCE_MODELS={cloud_models}"
#         )
#
#     if baseline == "smallest_to_largest":
#         selected = [
#             edge_models[0],
#             cloud_models[-1],
#         ]
#         return selected, _estimates_for(selected, wait_results), "sequential"
#
#     if baseline == "random_to_random":
#         selected = [
#             random.choice(edge_models),
#             random.choice(cloud_models),
#         ]
#         return selected, _estimates_for(selected, wait_results), "sequential"
#
#     if baseline == "smallest_to_ensemble_3":
#         if len(cloud_models) < 3:
#             raise ValueError(
#                 "BASELINE_MODE='smallest_to_ensemble_3' requires at least "
#                 "3 models in LARGE_INSTANCE_MODELS."
#             )
#
#         selected = [
#             edge_models[0],
#             *cloud_models[-3:],
#         ]
#         return selected, _estimates_for(selected, wait_results), "full_parallel"
#
#     if baseline == "power2_smallest_to_largest":
#         if len(edge_models) < 2:
#             raise ValueError(
#                 "BASELINE_MODE='power2_smallest_to_largest' requires at least "
#                 "2 models in SMALL_INSTANCE_MODELS."
#             )
#
#         selected = [
#             edge_models[0],
#             edge_models[1],
#             cloud_models[-1],
#         ]
#         return selected, _estimates_for(selected, wait_results), "parallel_first_finish"
#
#     raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")

# -----------------------------
# Instance-aware baselines
# -----------------------------
def build_instance_aware_baseline(
    wait_results: Dict[str, float],
    logger=None,
) -> Tuple[List[str], Dict[str, float], str]:
    baseline = str(BASELINE_MODE).lower()

    if baseline == "none":
        return [], {}, ""

    edge_models = _sorted_by_size(
        get_available_models(
            get_edge_model_pool(),
            wait_results,
        )
    )
    cloud_models = _sorted_by_size(
        get_available_models(
            get_cloud_model_pool(),
            wait_results,
        )
    )

    if not edge_models or not cloud_models:
        raise ValueError(
            f"Invalid or unavailable edge/cloud model pools. "
            f"edge_models={edge_models}, cloud_models={cloud_models}"
        )

    if baseline == "smallest_to_largest":
        selected = [
            edge_models[0],
            cloud_models[-1],
        ]
        return selected, _estimates_for(selected, wait_results), "sequential"

    if baseline == "random_to_random":
        selected = [
            random.choice(edge_models),
            random.choice(cloud_models),
        ]
        return selected, _estimates_for(selected, wait_results), "sequential"

    if baseline == "smallest_to_ensemble_3":
        if len(cloud_models) < 3:
            raise ValueError(
                "BASELINE_MODE='smallest_to_ensemble_3' requires at least "
                "3 models in LARGE_INSTANCE_MODELS."
            )

        selected = [
            edge_models[0],
            *cloud_models[-3:],
        ]
        return selected, _estimates_for(selected, wait_results), "full_parallel"
    if baseline == "full_ensemble_edge":
        selected = list(edge_models[:3])

        if not selected:
            raise ValueError(
                "BASELINE_MODE='full_ensemble_edge' requires at least "
                "one model in SMALL_INSTANCE_MODELS."
            )

        return selected, _estimates_for(selected, wait_results), "full_parallel"

    if baseline == "full_ensemble_cloud":
        selected = list(cloud_models[-3:])

        if not selected:
            raise ValueError(
                "BASELINE_MODE='full_ensemble_cloud' requires at least "
                "one model in LARGE_INSTANCE_MODELS."
            )

        return selected, _estimates_for(selected, wait_results), "full_parallel"

    if baseline == "power2_smallest_to_largest":
        if len(edge_models) < 2:
            raise ValueError(
                "BASELINE_MODE='power2_smallest_to_largest' requires at least "
                "2 models in SMALL_INSTANCE_MODELS."
            )

        selected = [
            edge_models[0],
            edge_models[1],
            cloud_models[-1],
        ]
        return selected, _estimates_for(selected, wait_results), "parallel_first_finish"

    # -----------------------------
    # Speculative cloud baselines
    # -----------------------------
    if baseline == "spec_b_smallest_to_largest":
        selected = [
            edge_models[0],   # edge/smallest model
            cloud_models[-1],  # cloud/largest model
        ]
        return selected, _estimates_for(selected, wait_results), "spec_b"

    if baseline == "spec_b_random_to_random":
        selected = [
            random.choice(edge_models),  # edge random model
            random.choice(cloud_models),  # cloud random model
        ]
        return selected, _estimates_for(selected, wait_results), "spec_b"

    raise ValueError(f"Unknown BASELINE_MODE: {BASELINE_MODE}")


# -----------------------------
# Instance-aware MODE-S selector
# -----------------------------
def build_instance_aware_model_set(
    cascade_mode: str,
    wait_results: Dict[str, float],
    deadline_ms: int,
    alpha_now: float,
    parallel_first_n: int,
    network_latency: float,
    logger=None,
) -> Tuple[List[str], Dict[str, float], str]:
    mode = str(cascade_mode).lower()

    edge_model_pool = get_available_models(
        get_edge_model_pool(),
        wait_results,
    )
    cloud_model_pool = get_available_models(
        get_cloud_model_pool(),
        wait_results,
    )

    edge_count = min(
        int(SMALL_ENSEMBLE_SIZE),
        len(edge_model_pool),
    )
    cloud_count = min(
        int(LARGE_ENSEMBLE_SIZE),
        len(cloud_model_pool),
    )

    selected: List[str] = []

    if edge_count > 0:
        if mode in ("mode_s", "full_parallel"):
            edge_selected, _ = build_model_set_mode_s_from_pool(
                model_pool=edge_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=edge_count,
                alpha_now=alpha_now,
                logger=logger,
            )
        elif mode == "sequential":
            edge_selected, _ = build_model_set_sequential_from_pool(
                model_pool=edge_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=edge_count,
                network_latency=network_latency,
                logger=logger,
            )
        else:
            edge_selected, _ = build_model_set_parallel_first_from_pool(
                model_pool=edge_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=edge_count,
                parallel_first_n=parallel_first_n,
                network_latency=network_latency,
                logger=logger,
            )

        selected.extend(edge_selected[:edge_count])

    if cloud_count > 0:
        if mode in ("mode_s", "full_parallel"):
            cloud_selected, _ = build_model_set_mode_s_from_pool(
                model_pool=cloud_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=cloud_count,
                alpha_now=alpha_now,
                logger=logger,
            )
        elif mode == "sequential":
            cloud_selected, _ = build_model_set_sequential_from_pool(
                model_pool=cloud_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=cloud_count,
                network_latency=network_latency,
                logger=logger,
            )
        else:
            cloud_selected, _ = build_model_set_parallel_first_from_pool(
                model_pool=cloud_model_pool,
                wait_results=wait_results,
                deadline_ms=deadline_ms,
                ensemble_size=cloud_count,
                parallel_first_n=parallel_first_n,
                network_latency=network_latency,
                logger=logger,
            )

        selected.extend(cloud_selected[:cloud_count])

    selected = list(dict.fromkeys(selected))

    return selected, _estimates_for(selected, wait_results), mode


# -----------------------------
# Public latency estimator
# -----------------------------
def estimate_selected_set_latency_for_mode(
    cascade_mode: str,
    selected_models: List[str],
    wait_results: Dict[str, float],
    network_latency: float,
    parallel_first_n: int,
) -> float:
    mode = str(cascade_mode).lower()

    if mode in ("mode_s", "full_parallel"):
        vals = [get_stage_time(m, wait_results) for m in selected_models]

        if not vals:
            return float("inf")

        return (
            float(BASE_REQUEST_OVERHEAD)
            + max(vals)
            + float(FULL_PARALLEL_AGG_OVERHEAD)
        )

    if mode == "sequential":
        ordered = reorder_for_sequential(selected_models, wait_results)
        return estimate_expected_latency_sequential(
            ordered_models=ordered,
            wait_results=wait_results,
            network_latency=network_latency,
        )

    ordered = selected_models
    return estimate_expected_latency_parallel_first_finish(
        ordered_models=ordered,
        wait_results=wait_results,
        parallel_first_n=parallel_first_n,
        network_latency=network_latency,
    )


# -----------------------------
# Dispatcher
# -----------------------------
def build_model_set_by_mode(
    cascade_mode: str,
    wait_results: Dict[str, float],
    deadline_ms: int,
    ensemble_size: int,
    alpha_now: float,
    parallel_first_n: int,
    network_latency: float,
    logger=None,
) -> Tuple[List[str], Dict[str, float], str]:

    baseline_models, baseline_estimates, baseline_exec_mode = build_instance_aware_baseline(
        wait_results=wait_results,
        logger=logger,
    )

    if baseline_models:
        if logger:
            logger.info(
                f"[BASELINE_SELECT] baseline_mode={BASELINE_MODE} "
                f"selected_models={baseline_models} "
                f"effective_cascade_mode={baseline_exec_mode}"
            )

        return baseline_models, baseline_estimates, baseline_exec_mode

    if USE_MULTI_ILUVATAR_INSTANCES or USE_TWO_ILUVATAR_INSTANCES:
        selected, estimates, effective_mode = build_instance_aware_model_set(
            cascade_mode=cascade_mode,
            wait_results=wait_results,
            deadline_ms=deadline_ms,
            alpha_now=alpha_now,
            parallel_first_n=parallel_first_n,
            network_latency=network_latency,
            logger=logger,
        )

        if selected:
            if logger:
                logger.info(
                    f"[INSTANCE_AWARE_SELECT] "
                    f"edge_ensemble_size={SMALL_ENSEMBLE_SIZE} "
                    f"cloud_ensemble_size={LARGE_ENSEMBLE_SIZE} "
                    f"selected_models={selected} "
                    f"effective_cascade_mode={effective_mode}"
                )

            return selected, estimates, effective_mode

    mode = str(cascade_mode).lower()

    selected, estimates = build_model_set_mode_s_from_pool(
        model_pool=M_TOTAL,
        wait_results=wait_results,
        deadline_ms=deadline_ms,
        ensemble_size=ensemble_size,
        alpha_now=alpha_now,
        logger=logger,
    )

    if mode == "sequential":
        selected, estimates = build_model_set_sequential_from_pool(
            model_pool=M_TOTAL,
            wait_results=wait_results,
            deadline_ms=deadline_ms,
            ensemble_size=ensemble_size,
            network_latency=network_latency,
            logger=logger,
        )

    elif mode not in ("mode_s", "full_parallel"):
        selected, estimates = build_model_set_parallel_first_from_pool(
            model_pool=M_TOTAL,
            wait_results=wait_results,
            deadline_ms=deadline_ms,
            ensemble_size=ensemble_size,
            parallel_first_n=parallel_first_n,
            network_latency=network_latency,
            logger=logger,
        )

    return selected, estimates, mode