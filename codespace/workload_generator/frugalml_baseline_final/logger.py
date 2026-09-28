# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# LOG_HEADER = ["ID", "Deadline", "IAR", "RespTime", "Model", "Accuracy", "Latency", "State", "Success"]
#
#
# def log_result(req_id, deadline, iar, response_time, response):
#     os.makedirs("data", exist_ok=True)
#     log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/iar_results.csv"
#     write_header = not os.path.exists(log_path)
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(LOG_HEADER)
#         writer.writerow([
#             req_id, deadline, iar, response_time,
#             response.get("model"), response.get("accuracy"),
#             response.get("latency"), response.get("container_state"),
#             response.get("success")
#         ])
#
#     # Rich real-time dashboard logging
#     table = Table(show_header=True, header_style="bold magenta")
#     for col in LOG_HEADER:
#         table.add_column(col)
#     table.add_row(
#         str(req_id), str(deadline), str(iar), f"{response_time:.3f}",
#         str(response.get("model")),
#         f"{response.get('accuracy', 0.0):.4f}",
#         f"{response.get('latency', -1):.3f}",
#         str(response.get("container_state")),
#         str(response.get("success"))
#     )
#     console.print(table)



##2 start

# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# # Add per-model wait columns to the header
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime","RunTime", "selected_models","label", "Accuracy", "combiner_policy", "e2e_time_ms", "Success", "selected_folder",
#     "random_image", "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
#     "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
# ]
#
# def log_result(mode, req_id, deadline, iar, response_time,current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#     if mode == "vary_deadline":
#         log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/deadline_results_M31_I5_D1.csv"
#     else:
#         log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/iar_results_3.csv"
#
#     write_header = not os.path.exists(log_path)
#
#     # Extract wait times from response (dict of model: wait_time)
#     waits = response.get("wait_times", {})
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(LOG_HEADER)
#         writer.writerow([
#             req_id, deadline, iar, response_time,current_time_sec,
#             response.get("selected_models", []), response.get("label",-1),
#             response.get("accuracy"), response.get("combiner_policy"),
#             response.get("e2e_time_ms"), response.get("success"),
#             response.get("selected_folder"),
#             waits.get("mobilenet_v3_small", -1),
#             waits.get("resnet18", -1),
#             waits.get("resnet34", -1),
#             waits.get("resnet50", -1),
#             waits.get("resnet101", -1),
#             waits.get("vit_b_16", -1)
#         ])
#
#     # Rich real-time terminal output
#     table = Table(show_header=True, header_style="bold magenta")
#
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     table.add_row(
#         str(req_id), str(deadline), str(iar), f"{response_time:.3f}", f"{current_time_sec:.3f}",
#         str(response.get("selected_models", [])),
#         str(response.get("label",-1)),
#         f"{response.get('accuracy', 0.0):.4f}",
#         f"{response.get('latency', -1):.3f}",
#         str(response.get("combiner_policy",-1)),
#         str(response.get("e2e_time_ms",-1)),
#         str(response.get("success")),
#         str(response.get("selected_folder")),
#         f"{waits.get('mobilenet_v3_small', -1):.2f}",
#         f"{waits.get('resnet18', -1):.2f}",
#         f"{waits.get('resnet34', -1):.2f}",
#         f"{waits.get('resnet50', -1):.2f}",
#         f"{waits.get('resnet101', -1):.2f}",
#         f"{waits.get('vit_b_16', -1):.2f}"
#     )
#
#     console.print(table)


    ### 2 end

# ########
# # final 05-04-2026 start
# ######
#
# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "vit_b_16",
# ]
#
# PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
# PER_MODEL_PROB_COLS  = [f"{m}_prob"  for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS  = [f"{m}_wait"  for m in MODEL_LIST]
#
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime", "RunTime",
#     "selected_models",
#     "executed_models",
#     "label",
#     "Accuracy",
#     "combiner_policy",
#     "e2e_time_ms",
#     "Success",
#     "selected_folder",
#     "ensemble_size",
#     "threshold",
#     "threshold_stage",
#     "parallel_first_n",
#     "stop_reason",
#     "qps_est",
#     "main_policy",
#     "alpha",
#     "alpha_mode",
#     *PER_MODEL_LABEL_COLS,
#     *PER_MODEL_PROB_COLS,
#     *PER_MODEL_WAIT_COLS,
# ]
#
# def _get_per_model_fields(response):
#     per_model = response.get("per_model", {}) or {}
#     labels = []
#     probs = []
#     for m in MODEL_LIST:
#         info = per_model.get(m, {}) or {}
#         labels.append(str(info.get("label", "")))
#         p = info.get("probability", None)
#         try:
#             probs.append(float(p) if p is not None else -1.0)
#         except Exception:
#             probs.append(-1.0)
#     return labels, probs
#
# def _get_per_model_waits(response):
#     waits = response.get("wait_times", {}) or {}
#     vals = []
#     for m in MODEL_LIST:
#         w = waits.get(m, None)
#         try:
#             vals.append(float(w) if w is not None else -1.0)
#         except Exception:
#             vals.append(-1.0)
#     return vals
#
# def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#     log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_power2_M3.csv"
#     write_header = not os.path.exists(log_path)
#
#     selected_models = response.get("selected_models", [])
#     executed_models = response.get("executed_models", [])
#     label = response.get("label", -1)
#     accuracy = response.get("accuracy", None)
#     combiner = response.get("combiner_policy", "")
#     e2e_ms = response.get("e2e_time_ms", -1)
#     success = response.get("success", False)
#     selected_folder = response.get("selected_folder", "")
#     ensemble_size = response.get("ensemble_size", -1)
#     threshold = response.get("threshold", "")
#     threshold_stage = response.get("threshold_stage", "")
#     parallel_first_n = response.get("parallel_first_n", "")
#     stop_reason = response.get("stop_reason", "")
#     qps_est = response.get("qps_est", "")
#     main_policy = response.get("main_policy", "")
#     alpha = response.get("alpha", "")
#     alpha_mode = response.get("alpha_mode", "")
#     selected_est_latency_s = response.get("selected_est_latency_s", -1)
#
#     model_labels, model_probs = _get_per_model_fields(response)
#     model_waits = _get_per_model_waits(response)
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(LOG_HEADER)
#
#         row = [
#             req_id, deadline, iar, response_time, current_time_sec,
#             selected_models,
#             executed_models,
#             label,
#             accuracy,
#             combiner,
#             e2e_ms,
#             success,
#             selected_folder,
#             ensemble_size,
#             threshold,
#             threshold_stage,
#             parallel_first_n,
#             stop_reason,
#             qps_est,
#             main_policy,
#             alpha,
#             alpha_mode,
#             *model_labels,
#             *model_probs,
#             *model_waits,
#             selected_est_latency_s,
#         ]
#         writer.writerow(row)
#
#     table = Table(show_header=True, header_style="bold magenta")
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     sm_str = str(selected_models)
#     em_str = str(executed_models)
#     acc_str = f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) and accuracy >= 0 else str(accuracy)
#     e2e_str = f"{e2e_ms:.2f}" if isinstance(e2e_ms, (int, float)) else str(e2e_ms)
#     rt_str = f"{response_time:.3f}"
#     ct_str = f"{current_time_sec:.3f}"
#     qps_str = f"{qps_est:.2f}" if isinstance(qps_est, (int, float)) else str(qps_est)
#     alpha_str = f"{alpha:.4f}" if isinstance(alpha, (int, float)) else str(alpha)
#
#     model_probs_str = [f"{p:.4f}" if isinstance(p, (int, float)) and p >= 0 else str(p) for p in model_probs]
#     model_waits_str = [f"{w:.2f}" if isinstance(w, (int, float)) and w >= 0 else str(w) for w in model_waits]
#
#     table.add_row(
#         str(req_id),
#         str(deadline),
#         str(iar),
#         rt_str,
#         ct_str,
#         sm_str,
#         em_str,
#         str(label),
#         acc_str,
#         str(combiner),
#         e2e_str,
#         str(success),
#         str(selected_folder),
#         str(ensemble_size),
#         str(threshold),
#         str(threshold_stage),
#         str(parallel_first_n),
#         str(stop_reason),
#         qps_str,
#         str(main_policy),
#         alpha_str,
#         str(alpha_mode),
#         *[str(x) for x in model_labels],
#         *model_probs_str,
#         *model_waits_str,
#     )
#
#     console.print(table)
#
#     # console.print(table)
#
# ########
# # final 05-04-2026 end
# ######


# ########
# # final 05-12-2026 start
# ######

# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
# PER_MODEL_PROB_COLS  = [f"{m}_prob" for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS  = [f"{m}_wait" for m in MODEL_LIST]
#
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime", "RunTime",
#     "selected_models",
#     "executed_models",
#     "label",
#     "Accuracy",
#     "combiner_policy",
#     "e2e_time_ms",
#     "Success",
#     "selected_folder",
#     "ensemble_size",
#     "threshold",
#     "threshold_stage",
#     "parallel_first_n",
#     "stop_reason",
#     "qps_est",
#     "main_policy",
#     "alpha",
#     "alpha_mode",
#
#     # new important fields
#     "baseline_mode",
#     "configured_cascade_mode",
#     "effective_cascade_mode",
#
#     *PER_MODEL_LABEL_COLS,
#     *PER_MODEL_PROB_COLS,
#     *PER_MODEL_WAIT_COLS,
#
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "recent_latency_overhead_s",
# ]
#
#
# def _get_per_model_fields(response):
#     per_model = response.get("per_model", {}) or {}
#     labels = []
#     probs = []
#
#     for m in MODEL_LIST:
#         info = per_model.get(m, {}) or {}
#         labels.append(str(info.get("label", "")))
#
#         p = info.get("probability", None)
#         try:
#             probs.append(float(p) if p is not None else -1.0)
#         except Exception:
#             probs.append(-1.0)
#
#     return labels, probs
#
#
# def _get_per_model_waits(response):
#     waits = response.get("wait_times", {}) or {}
#     vals = []
#
#     for m in MODEL_LIST:
#         w = waits.get(m, None)
#         try:
#             vals.append(float(w) if w is not None else -1.0)
#         except Exception:
#             vals.append(-1.0)
#
#     return vals
#
#
# def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#
#     log_path = (
# "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/threshold_accuracy_latency.csv"
#     )
#
#     write_header = not os.path.exists(log_path)
#
#     selected_models = response.get("selected_models", [])
#     executed_models = response.get("executed_models", [])
#     label = response.get("label", -1)
#     accuracy = response.get("accuracy", None)
#     combiner = response.get("combiner_policy", "")
#     e2e_ms = response.get("e2e_time_ms", -1)
#     success = response.get("success", False)
#     selected_folder = response.get("selected_folder", "")
#     ensemble_size = response.get("ensemble_size", -1)
#     threshold = response.get("threshold", "")
#     threshold_stage = response.get("threshold_stage", "")
#     parallel_first_n = response.get("parallel_first_n", "")
#     stop_reason = response.get("stop_reason", "")
#     qps_est = response.get("qps_est", "")
#     main_policy = response.get("main_policy", "")
#     alpha = response.get("alpha", "")
#     alpha_mode = response.get("alpha_mode", "")
#
#     baseline_mode = response.get("baseline_mode", "none")
#     configured_cascade_mode = response.get(
#         "configured_cascade_mode",
#         response.get("cascade_mode", ""),
#     )
#     effective_cascade_mode = response.get(
#         "effective_cascade_mode",
#         response.get("effective_mode", ""),
#     )
#
#     selected_est_latency_s = response.get("selected_est_latency_s", -1)
#     raw_selected_est_latency_s = response.get("raw_selected_est_latency_s", -1)
#     recent_latency_overhead_s = response.get("recent_latency_overhead_s", -1)
#
#     model_labels, model_probs = _get_per_model_fields(response)
#     model_waits = _get_per_model_waits(response)
#
#     row = [
#         req_id, deadline, iar, response_time, current_time_sec,
#         selected_models,
#         executed_models,
#         label,
#         accuracy,
#         combiner,
#         e2e_ms,
#         success,
#         selected_folder,
#         ensemble_size,
#         threshold,
#         threshold_stage,
#         parallel_first_n,
#         stop_reason,
#         qps_est,
#         main_policy,
#         alpha,
#         alpha_mode,
#
#         baseline_mode,
#         configured_cascade_mode,
#         effective_cascade_mode,
#
#         *model_labels,
#         *model_probs,
#         *model_waits,
#
#         selected_est_latency_s,
#         raw_selected_est_latency_s,
#         recent_latency_overhead_s,
#     ]
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#
#         if write_header:
#             writer.writerow(LOG_HEADER)
#
#         writer.writerow(row)
#
#     table = Table(show_header=True, header_style="bold magenta")
#
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     row_for_table = [str(x) for x in row]
#
#     table.add_row(*row_for_table)
#     console.print(table)

# ########
# # final 05-12-2026 end
# ######


# ########
# # final 05-14-2026 start
# ######

# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
# PER_MODEL_PROB_COLS = [f"{m}_prob" for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS = [f"{m}_wait" for m in MODEL_LIST]
#
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime", "RunTime",
#     "selected_models",
#     "executed_models",
#     "label",
#     "Accuracy",
#     "combiner_policy",
#     "e2e_time_ms",
#     "Success",
#     "selected_folder",
#     "ensemble_size",
#     "threshold",
#     "threshold_stage",
#     "parallel_first_n",
#     "stop_reason",
#     "qps_est",
#     "main_policy",
#     "alpha",
#     "alpha_mode",
#
#     "baseline_mode",
#     "configured_cascade_mode",
#     "effective_cascade_mode",
#
#     # Risk-based strategy selection fields
#     "risk_selected_mode",
#     "risk_full",
#     "risk_sequential",
#     "risk_parallel_first_finish",
#
#     "risk_full_latency_s",
#     "risk_sequential_latency_s",
#     "risk_parallel_first_finish_latency_s",
#
#     "risk_full_cost",
#     "risk_sequential_cost",
#     "risk_parallel_first_finish_cost",
#
#     "risk_full_accuracy",
#     "risk_sequential_accuracy",
#     "risk_parallel_first_finish_accuracy",
#
#     *PER_MODEL_LABEL_COLS,
#     *PER_MODEL_PROB_COLS,
#     *PER_MODEL_WAIT_COLS,
#
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "recent_latency_overhead_s",
# ]
#
#
# def _get_per_model_fields(response):
#     per_model = response.get("per_model", {}) or {}
#     labels = []
#     probs = []
#
#     for m in MODEL_LIST:
#         info = per_model.get(m, {}) or {}
#         labels.append(str(info.get("label", "")))
#
#         p = info.get("probability", None)
#         try:
#             probs.append(float(p) if p is not None else -1.0)
#         except Exception:
#             probs.append(-1.0)
#
#     return labels, probs
#
#
# def _get_per_model_waits(response):
#     waits = response.get("wait_times", {}) or {}
#     vals = []
#
#     for m in MODEL_LIST:
#         w = waits.get(m, None)
#         try:
#             vals.append(float(w) if w is not None else -1.0)
#         except Exception:
#             vals.append(-1.0)
#
#     return vals
#
#
# def _get_risk_fields(response):
#     risk_selected_mode = response.get("risk_selected_mode", "")
#
#     risk_details = response.get("risk_details", {}) or {}
#
#     def _get(mode, field, default=-1.0):
#         try:
#             return float(risk_details.get(mode, {}).get(field, default))
#         except Exception:
#             return default
#
#     return [
#         risk_selected_mode,
#
#         _get("full_parallel", "risk"),
#         _get("sequential", "risk"),
#         _get("parallel_first_finish", "risk"),
#
#         _get("full_parallel", "latency_s"),
#         _get("sequential", "latency_s"),
#         _get("parallel_first_finish", "latency_s"),
#
#         _get("full_parallel", "cost"),
#         _get("sequential", "cost"),
#         _get("parallel_first_finish", "cost"),
#
#         _get("full_parallel", "accuracy"),
#         _get("sequential", "accuracy"),
#         _get("parallel_first_finish", "accuracy"),
#     ]
#
#
# def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#
#     log_path = (
# "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/tem.csv"
#     )
#
#     write_header = not os.path.exists(log_path)
#
#     selected_models = response.get("selected_models", [])
#     executed_models = response.get("executed_models", [])
#     label = response.get("label", -1)
#     accuracy = response.get("accuracy", None)
#     combiner = response.get("combiner_policy", "")
#     e2e_ms = response.get("e2e_time_ms", -1)
#     success = response.get("success", False)
#     selected_folder = response.get("selected_folder", "")
#     ensemble_size = response.get("ensemble_size", -1)
#     threshold = response.get("threshold", "")
#     threshold_stage = response.get("threshold_stage", "")
#     parallel_first_n = response.get("parallel_first_n", "")
#     stop_reason = response.get("stop_reason", "")
#     qps_est = response.get("qps_est", "")
#     main_policy = response.get("main_policy", "")
#     alpha = response.get("alpha", "")
#     alpha_mode = response.get("alpha_mode", "")
#
#     baseline_mode = response.get("baseline_mode", "none")
#     configured_cascade_mode = response.get(
#         "configured_cascade_mode",
#         response.get("cascade_mode", ""),
#     )
#     effective_cascade_mode = response.get(
#         "effective_cascade_mode",
#         response.get("effective_mode", ""),
#     )
#
#     risk_fields = _get_risk_fields(response)
#
#     selected_est_latency_s = response.get("selected_est_latency_s", -1)
#     raw_selected_est_latency_s = response.get("raw_selected_est_latency_s", -1)
#     recent_latency_overhead_s = response.get("recent_latency_overhead_s", -1)
#
#     model_labels, model_probs = _get_per_model_fields(response)
#     model_waits = _get_per_model_waits(response)
#
#     row = [
#         req_id, deadline, iar, response_time, current_time_sec,
#         selected_models,
#         executed_models,
#         label,
#         accuracy,
#         combiner,
#         e2e_ms,
#         success,
#         selected_folder,
#         ensemble_size,
#         threshold,
#         threshold_stage,
#         parallel_first_n,
#         stop_reason,
#         qps_est,
#         main_policy,
#         alpha,
#         alpha_mode,
#
#         baseline_mode,
#         configured_cascade_mode,
#         effective_cascade_mode,
#
#         *risk_fields,
#
#         *model_labels,
#         *model_probs,
#         *model_waits,
#
#         selected_est_latency_s,
#         raw_selected_est_latency_s,
#         recent_latency_overhead_s,
#     ]
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#
#         if write_header:
#             writer.writerow(LOG_HEADER)
#
#         writer.writerow(row)
#
#     table = Table(show_header=True, header_style="bold magenta")
#
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     table.add_row(*[str(x) for x in row])
#     console.print(table)


# ########
# # final 05-14-2026 end
# ######


# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
# PER_MODEL_PROB_COLS = [f"{m}_prob" for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS = [f"{m}_wait" for m in MODEL_LIST]
#
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime", "RunTime",
#     "selected_models", "executed_models", "label", "Accuracy",
#     "combiner_policy", "e2e_time_ms", "Success", "selected_folder",
#     "ensemble_size", "threshold", "threshold_stage", "selected_theta",
#     "parallel_first_n", "stop_reason", "qps_est", "main_policy",
#     "alpha", "alpha_mode",
#
#     "baseline_mode", "configured_cascade_mode", "effective_cascade_mode",
#
#     "seq_solver_success", "seq_solver_reason",
#     "seq_solver_feasible_count", "seq_solver_total_count",
#     "seq_solver_latency_ms", "seq_solver_a", "seq_solver_b",
#     "seq_solver_theta", "seq_solver_accuracy", "seq_solver_error",
#     "seq_solver_accept_rate", "seq_solver_cost",
#     "seq_solver_expected_latency_ms",
#
#     "risk_selected_mode",
#     "risk_full", "risk_sequential", "risk_parallel_first_finish",
#     "risk_full_latency_s", "risk_sequential_latency_s",
#     "risk_parallel_first_finish_latency_s",
#     "risk_full_cost", "risk_sequential_cost",
#     "risk_parallel_first_finish_cost",
#     "risk_full_accuracy", "risk_sequential_accuracy",
#     "risk_parallel_first_finish_accuracy",
#
#     *PER_MODEL_LABEL_COLS,
#     *PER_MODEL_PROB_COLS,
#     *PER_MODEL_WAIT_COLS,
#
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "recent_latency_overhead_s",
# ]
#
#
# def _get_per_model_fields(response):
#     per_model = response.get("per_model", {}) or {}
#     labels, probs = [], []
#
#     for m in MODEL_LIST:
#         info = per_model.get(m, {}) or {}
#         labels.append(str(info.get("label", "")))
#
#         try:
#             probs.append(float(info.get("probability", -1.0)))
#         except Exception:
#             probs.append(-1.0)
#
#     return labels, probs
#
#
# def _get_per_model_waits(response):
#     waits = response.get("wait_times", {}) or {}
#     vals = []
#
#     for m in MODEL_LIST:
#         try:
#             vals.append(float(waits.get(m, -1.0)))
#         except Exception:
#             vals.append(-1.0)
#
#     return vals
#
#
# def _get_seq_solver_fields(response):
#     details = response.get("seq_solver_details", {}) or {}
#     selected_config = details.get("selected_config", {}) or {}
#
#     def get_float(d, key, default=-1.0):
#         try:
#             val = d.get(key, default)
#             return float(val) if val is not None else default
#         except Exception:
#             return default
#
#     def get_int(d, key, default=-1):
#         try:
#             val = d.get(key, default)
#             return int(val) if val is not None else default
#         except Exception:
#             return default
#
#     return [
#         bool(details.get("success", False)),
#         details.get("reason", ""),
#         get_int(details, "feasible_count"),
#         get_int(details, "total_count"),
#         get_float(details, "solver_latency_ms"),
#
#         selected_config.get("a", ""),
#         selected_config.get("b", ""),
#         get_float(selected_config, "theta"),
#         get_float(selected_config, "cascade_accuracy"),
#         get_float(selected_config, "error"),
#         get_float(selected_config, "a_accept_rate"),
#         get_float(selected_config, "relative_compute_cost"),
#         get_float(selected_config, "expected_latency_cost_ms"),
#     ]
#
#
# def _get_risk_fields(response):
#     risk_selected_mode = response.get("risk_selected_mode", "")
#     risk_details = response.get("risk_details", {}) or {}
#
#     def get_float(mode, field, default=-1.0):
#         try:
#             return float(risk_details.get(mode, {}).get(field, default))
#         except Exception:
#             return default
#
#     return [
#         risk_selected_mode,
#
#         get_float("full_parallel", "risk"),
#         get_float("sequential", "risk"),
#         get_float("parallel_first_finish", "risk"),
#
#         get_float("full_parallel", "latency_s"),
#         get_float("sequential", "latency_s"),
#         get_float("parallel_first_finish", "latency_s"),
#
#         get_float("full_parallel", "cost"),
#         get_float("sequential", "cost"),
#         get_float("parallel_first_finish", "cost"),
#
#         get_float("full_parallel", "accuracy"),
#         get_float("sequential", "accuracy"),
#         get_float("parallel_first_finish", "accuracy"),
#     ]
#
#
# def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#
#     log_path = (
#         "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_solver_details_step1.csv"
#     )
#
#     write_header = not os.path.exists(log_path)
#
#     selected_models = response.get("selected_models", [])
#     executed_models = response.get("executed_models", [])
#     label = response.get("label", -1)
#     accuracy = response.get("accuracy", None)
#     combiner = response.get("combiner_policy", "")
#     e2e_ms = response.get("e2e_time_ms", -1)
#     success = response.get("success", False)
#
#     selected_folder = response.get("selected_folder", "")
#     ensemble_size = response.get("ensemble_size", -1)
#     threshold = response.get("threshold", "")
#     threshold_stage = response.get("threshold_stage", "")
#     selected_theta = response.get("selected_theta", "")
#     parallel_first_n = response.get("parallel_first_n", "")
#     stop_reason = response.get("stop_reason", "")
#
#     qps_est = response.get("qps_est", "")
#     main_policy = response.get("main_policy", "")
#     alpha = response.get("alpha", "")
#     alpha_mode = response.get("alpha_mode", "")
#
#     baseline_mode = response.get("baseline_mode", "none")
#     configured_cascade_mode = response.get(
#         "configured_cascade_mode",
#         response.get("cascade_mode", ""),
#     )
#     effective_cascade_mode = response.get(
#         "effective_cascade_mode",
#         response.get("effective_mode", ""),
#     )
#
#     seq_solver_fields = _get_seq_solver_fields(response)
#     risk_fields = _get_risk_fields(response)
#
#     model_labels, model_probs = _get_per_model_fields(response)
#     model_waits = _get_per_model_waits(response)
#
#     selected_est_latency_s = response.get("selected_est_latency_s", -1)
#     raw_selected_est_latency_s = response.get("raw_selected_est_latency_s", -1)
#     recent_latency_overhead_s = response.get("recent_latency_overhead_s", -1)
#
#     row = [
#         req_id, deadline, iar, response_time, current_time_sec,
#         selected_models, executed_models, label, accuracy,
#         combiner, e2e_ms, success, selected_folder,
#         ensemble_size, threshold, threshold_stage, selected_theta,
#         parallel_first_n, stop_reason, qps_est, main_policy,
#         alpha, alpha_mode,
#
#         baseline_mode, configured_cascade_mode, effective_cascade_mode,
#
#         *seq_solver_fields,
#         *risk_fields,
#
#         *model_labels,
#         *model_probs,
#         *model_waits,
#
#         selected_est_latency_s,
#         raw_selected_est_latency_s,
#         recent_latency_overhead_s,
#     ]
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(LOG_HEADER)
#         writer.writerow(row)
#
#     table = Table(show_header=True, header_style="bold magenta")
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     table.add_row(*[str(x) for x in row])
#     console.print(table)


# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
# PER_MODEL_PROB_COLS = [f"{m}_prob" for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS = [f"{m}_wait" for m in MODEL_LIST]
#
# LOG_HEADER = [
#     "ID", "Deadline", "IAR", "RespTime", "RunTime",
#     "selected_models", "executed_models", "label", "Accuracy",
#     "combiner_policy", "e2e_time_ms", "Success", "selected_folder",
#     "ensemble_size", "threshold", "threshold_stage", "selected_theta",
#     "parallel_first_n", "stop_reason", "qps_est", "main_policy",
#     "alpha", "alpha_mode",
#
#     "baseline_mode", "configured_cascade_mode", "effective_cascade_mode",
#
#     "constraint_solver_success",
#     "constraint_solver_policy",
#     "constraint_solver_reason",
#     "constraint_solver_feasible_count",
#     "constraint_solver_total_count",
#     "constraint_solver_latency_ms",
#     "constraint_solver_time_mode",
#     "constraint_solver_max_cost",
#     "constraint_solver_target_accuracy",
#
#     "constraint_solver_models",
#     "constraint_solver_a",
#     "constraint_solver_b",
#     "constraint_solver_theta",
#     "constraint_solver_accuracy",
#     "constraint_solver_error",
#     "constraint_solver_accept_rate",
#     "constraint_solver_cost",
#     "constraint_solver_expected_latency_ms",
#     "constraint_solver_objective_latency_ms",
#
#     *PER_MODEL_LABEL_COLS,
#     *PER_MODEL_PROB_COLS,
#     *PER_MODEL_WAIT_COLS,
#
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "recent_latency_overhead_s",
# ]
#
#
# def _get_per_model_fields(response):
#     per_model = response.get("per_model", {}) or {}
#     labels, probs = [], []
#
#     for m in MODEL_LIST:
#         info = per_model.get(m, {}) or {}
#         labels.append(str(info.get("label", "")))
#
#         try:
#             probs.append(float(info.get("probability", -1.0)))
#         except Exception:
#             probs.append(-1.0)
#
#     return labels, probs
#
#
# def _get_per_model_waits(response):
#     waits = response.get("wait_times", {}) or {}
#     vals = []
#
#     for m in MODEL_LIST:
#         try:
#             vals.append(float(waits.get(m, -1.0)))
#         except Exception:
#             vals.append(-1.0)
#
#     return vals
#
#
# def _get_constraint_solver_fields(response):
#     details = response.get("constraint_solver_details", {}) or {}
#     selected_config = details.get("selected_config", {}) or {}
#
#     def get_float(d, key, default=-1.0):
#         try:
#             val = d.get(key, default)
#             return float(val) if val is not None else default
#         except Exception:
#             return default
#
#     def get_int(d, key, default=-1):
#         try:
#             val = d.get(key, default)
#             return int(val) if val is not None else default
#         except Exception:
#             return default
#
#     return [
#         bool(details.get("success", False)),
#         details.get("policy", ""),
#         details.get("reason", ""),
#         get_int(details, "feasible_count"),
#         get_int(details, "total_count"),
#         get_float(details, "solver_latency_ms"),
#         details.get("time_mode", ""),
#         get_float(details, "max_cost"),
#         get_float(details, "target_accuracy"),
#
#         selected_config.get("models", ""),
#         selected_config.get("a", ""),
#         selected_config.get("b", ""),
#         get_float(selected_config, "theta"),
#         get_float(selected_config, "accuracy"),
#         get_float(selected_config, "error"),
#         get_float(selected_config, "a_accept_rate"),
#         get_float(selected_config, "relative_compute_cost"),
#         get_float(selected_config, "expected_latency_cost_ms"),
#         get_float(selected_config, "solver_latency_ms_objective"),
#     ]
#
#
# def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
#     os.makedirs("data", exist_ok=True)
#
#     log_path = (
# "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/constrain_solver_latency_n_IAR_2.csv"
#     )
#
#     write_header = not os.path.exists(log_path)
#
#     selected_models = response.get("selected_models", [])
#     executed_models = response.get("executed_models", [])
#     label = response.get("label", -1)
#     accuracy = response.get("accuracy", None)
#     combiner = response.get("combiner_policy", "")
#     e2e_ms = response.get("e2e_time_ms", -1)
#     success = response.get("success", False)
#
#     selected_folder = response.get("selected_folder", "")
#     ensemble_size = response.get("ensemble_size", -1)
#     threshold = response.get("threshold", "")
#     threshold_stage = response.get("threshold_stage", "")
#     selected_theta = response.get("selected_theta", "")
#     parallel_first_n = response.get("parallel_first_n", "")
#     stop_reason = response.get("stop_reason", "")
#
#     qps_est = response.get("qps_est", "")
#     main_policy = response.get("main_policy", "")
#     alpha = response.get("alpha", "")
#     alpha_mode = response.get("alpha_mode", "")
#
#     baseline_mode = response.get("baseline_mode", "none")
#     configured_cascade_mode = response.get(
#         "configured_cascade_mode",
#         response.get("cascade_mode", ""),
#     )
#     effective_cascade_mode = response.get(
#         "effective_cascade_mode",
#         response.get("effective_mode", ""),
#     )
#
#     constraint_solver_fields = _get_constraint_solver_fields(response)
#     model_labels, model_probs = _get_per_model_fields(response)
#     model_waits = _get_per_model_waits(response)
#
#     selected_est_latency_s = response.get("selected_est_latency_s", -1)
#     raw_selected_est_latency_s = response.get("raw_selected_est_latency_s", -1)
#     recent_latency_overhead_s = response.get("recent_latency_overhead_s", -1)
#
#     row = [
#         req_id, deadline, iar, response_time, current_time_sec,
#         selected_models, executed_models, label, accuracy,
#         combiner, e2e_ms, success, selected_folder,
#         ensemble_size, threshold, threshold_stage, selected_theta,
#         parallel_first_n, stop_reason, qps_est, main_policy,
#         alpha, alpha_mode,
#
#         baseline_mode, configured_cascade_mode, effective_cascade_mode,
#
#         *constraint_solver_fields,
#
#         *model_labels,
#         *model_probs,
#         *model_waits,
#
#         selected_est_latency_s,
#         raw_selected_est_latency_s,
#         recent_latency_overhead_s,
#     ]
#
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#
#         if write_header:
#             writer.writerow(LOG_HEADER)
#
#         writer.writerow(row)
#
#     table = Table(show_header=True, header_style="bold magenta")
#
#     for col in LOG_HEADER:
#         table.add_column(col)
#
#     table.add_row(*[str(x) for x in row])
#     console.print(table)


import csv
import os
from rich.console import Console
from rich.table import Table

console = Console()

MODEL_LIST = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
]

PER_MODEL_LABEL_COLS = [f"{m}_label" for m in MODEL_LIST]
PER_MODEL_PROB_COLS = [f"{m}_prob" for m in MODEL_LIST]
PER_MODEL_WAIT_COLS = [f"{m}_wait" for m in MODEL_LIST]

LOG_HEADER = [
    "ID", "Deadline", "IAR", "RespTime", "RunTime",
    "selected_models", "executed_models", "label", "Accuracy",
    "combiner_policy", "e2e_time_ms", "Success", "selected_folder",
    "ensemble_size", "threshold", "threshold_stage", "selected_theta",
    "parallel_first_n", "parallel_batch_size", "stop_reason",
    "qps_est", "main_policy", "alpha", "alpha_mode",

    "baseline_mode", "configured_cascade_mode", "effective_cascade_mode",

    # Static optimal configuration fields
    "static_config_enabled",
    "static_cost",
    "static_error",
    "static_target_accuracy",
    "static_policy",
    "static_models",
    "static_theta",
    "static_parallel_first_n",

    # Constraint solver fields
    "constraint_solver_success",
    "constraint_solver_policy",
    "constraint_solver_reason",
    "constraint_solver_feasible_count",
    "constraint_solver_total_count",
    "constraint_solver_latency_ms",
    "constraint_solver_time_mode",
    "constraint_solver_max_cost",
    "constraint_solver_target_accuracy",

    "constraint_solver_models",
    "constraint_solver_a",
    "constraint_solver_b",
    "constraint_solver_theta",
    "constraint_solver_accuracy",
    "constraint_solver_error",
    "constraint_solver_accept_rate",
    "constraint_solver_cost",
    "constraint_solver_expected_latency_ms",
    "constraint_solver_objective_latency_ms",

    *PER_MODEL_LABEL_COLS,
    *PER_MODEL_PROB_COLS,
    *PER_MODEL_WAIT_COLS,

    "selected_est_latency_s",
    "raw_selected_est_latency_s",
    "recent_latency_overhead_s",
]


def _get_per_model_fields(response):
    per_model = response.get("per_model", {}) or {}
    labels, probs = [], []

    for m in MODEL_LIST:
        info = per_model.get(m, {}) or {}
        labels.append(str(info.get("label", "")))

        try:
            probs.append(float(info.get("probability", -1.0)))
        except Exception:
            probs.append(-1.0)

    return labels, probs


def _get_per_model_waits(response):
    waits = response.get("wait_times", {}) or {}
    vals = []

    for m in MODEL_LIST:
        try:
            vals.append(float(waits.get(m, -1.0)))
        except Exception:
            vals.append(-1.0)

    return vals


def _get_static_config_fields(response):
    details = response.get("static_config_details", {}) or {}

    return [
        bool(details.get("enabled", False)),
        details.get("cost", ""),
        details.get("error", ""),
        details.get("target_accuracy", ""),
        details.get("policy", ""),
        details.get("models", ""),
        details.get("theta", ""),
        details.get("parallel_first_n", ""),
    ]


def _get_constraint_solver_fields(response):
    details = response.get("constraint_solver_details", {}) or {}
    selected_config = details.get("selected_config", {}) or {}

    def get_float(d, key, default=-1.0):
        try:
            val = d.get(key, default)
            return float(val) if val is not None else default
        except Exception:
            return default

    def get_int(d, key, default=-1):
        try:
            val = d.get(key, default)
            return int(val) if val is not None else default
        except Exception:
            return default

    return [
        bool(details.get("success", False)),
        details.get("policy", ""),
        details.get("reason", ""),
        get_int(details, "feasible_count"),
        get_int(details, "total_count"),
        get_float(details, "solver_latency_ms"),
        details.get("time_mode", ""),
        get_float(details, "max_cost"),
        get_float(details, "target_accuracy"),

        selected_config.get("models", ""),
        selected_config.get("a", ""),
        selected_config.get("b", ""),
        get_float(selected_config, "theta"),
        get_float(selected_config, "accuracy"),
        get_float(selected_config, "error"),
        get_float(selected_config, "a_accept_rate"),
        get_float(selected_config, "relative_compute_cost"),
        get_float(selected_config, "expected_latency_cost_ms"),
        get_float(selected_config, "solver_latency_ms_objective"),
    ]


def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
    os.makedirs("data", exist_ok=True)

    log_path = (
"/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
    )

    write_header = not os.path.exists(log_path)

    selected_models = response.get("selected_models", [])
    executed_models = response.get("executed_models", [])
    label = response.get("label", -1)
    accuracy = response.get("accuracy", None)
    combiner = response.get("combiner_policy", "")
    e2e_ms = response.get("e2e_time_ms", -1)
    success = response.get("success", False)

    selected_folder = response.get("selected_folder", "")
    ensemble_size = response.get("ensemble_size", -1)
    threshold = response.get("threshold", "")
    threshold_stage = response.get("threshold_stage", "")
    selected_theta = response.get("selected_theta", "")
    parallel_first_n = response.get("parallel_first_n", "")
    parallel_batch_size = response.get("parallel_batch_size", "")
    stop_reason = response.get("stop_reason", "")

    qps_est = response.get("qps_est", "")
    main_policy = response.get("main_policy", "")
    alpha = response.get("alpha", "")
    alpha_mode = response.get("alpha_mode", "")

    baseline_mode = response.get("baseline_mode", "none")
    configured_cascade_mode = response.get(
        "configured_cascade_mode",
        response.get("cascade_mode", ""),
    )
    effective_cascade_mode = response.get(
        "effective_cascade_mode",
        response.get("effective_mode", ""),
    )

    static_config_fields = _get_static_config_fields(response)
    constraint_solver_fields = _get_constraint_solver_fields(response)

    model_labels, model_probs = _get_per_model_fields(response)
    model_waits = _get_per_model_waits(response)

    selected_est_latency_s = response.get("selected_est_latency_s", -1)
    raw_selected_est_latency_s = response.get("raw_selected_est_latency_s", -1)
    recent_latency_overhead_s = response.get("recent_latency_overhead_s", -1)

    row = [
        req_id, deadline, iar, response_time, current_time_sec,
        selected_models, executed_models, label, accuracy,
        combiner, e2e_ms, success, selected_folder,
        ensemble_size, threshold, threshold_stage, selected_theta,
        parallel_first_n, parallel_batch_size, stop_reason,
        qps_est, main_policy, alpha, alpha_mode,

        baseline_mode, configured_cascade_mode, effective_cascade_mode,

        *static_config_fields,
        *constraint_solver_fields,

        *model_labels,
        *model_probs,
        *model_waits,

        selected_est_latency_s,
        raw_selected_est_latency_s,
        recent_latency_overhead_s,
    ]

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow(LOG_HEADER)

        writer.writerow(row)

    table = Table(show_header=True, header_style="bold magenta")

    for col in LOG_HEADER:
        table.add_column(col)

    table.add_row(*[str(x) for x in row])
    console.print(table)