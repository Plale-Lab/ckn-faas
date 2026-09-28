# import ast
# import csv
# import json
# import itertools
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# LOG_HEADER = [
#     "req_id", "deadline", "iar", "response_time", "current_time_sec",
#     "selected_models", "executed_models", "label", "confidence", "combiner",
#     "e2e_ms", "success", "selected_folder", "ensemble_size", "threshold",
#     "threshold_stage", "parallel_first_n", "stop_reason", "qps_est",
#     "main_policy", "alpha", "alpha_mode",
#
#     "label_mobilenet_v3_small",
#     "label_resnet18",
#     "label_resnet34",
#     "label_resnet50",
#     "label_resnet101",
#     "label_vit_b_16",
#
#     "prob_mobilenet_v3_small",
#     "prob_resnet18",
#     "prob_resnet34",
#     "prob_resnet50",
#     "prob_resnet101",
#     "prob_vit_b_16",
#
#     "wait_mobilenet_v3_small",
#     "wait_resnet18",
#     "wait_resnet34",
#     "wait_resnet50",
#     "wait_resnet101",
#     "wait_vit_b_16",
#
#     "selected_est_latency_s",
# ]
#
#
# def safe_list(x):
#     try:
#         return ast.literal_eval(x)
#     except Exception:
#         return []
#
#
# def normalize_text(x):
#     if x is None or pd.isna(x):
#         return ""
#     return str(x).lower().replace("_", " ").replace("-", " ").strip()
#
#
# def load_label_json(path):
#     if path is None:
#         return None
#     with open(path, "r") as f:
#         return json.load(f)
#
#
# def prediction_matches_wnid(pred_text, wnid, label_json):
#     if label_json is None or wnid not in label_json:
#         return None
#
#     pred_norm = normalize_text(pred_text)
#     true_names = label_json[wnid]
#     true_options = [normalize_text(x) for x in true_names.split(",")]
#
#     return any(pred_norm == option for option in true_options)
#
#
# def weighted_majority(labels, probs):
#     scores = {}
#
#     for label, prob in zip(labels, probs):
#         if label is None or pd.isna(label):
#             continue
#
#         label = str(label).strip()
#         if label == "":
#             continue
#
#         try:
#             prob = float(prob)
#         except Exception:
#             continue
#
#         if prob < 0:
#             continue
#
#         scores[label] = scores.get(label, 0.0) + prob
#
#     if not scores:
#         return None, -1.0
#
#     total_score = sum(scores.values())
#     pred = max(scores, key=scores.get)
#     conf = scores[pred] / total_score if total_score > 0 else -1.0
#
#     return pred, conf
#
#
# def get_model_output(row, model):
#     label_col = f"label_{model}"
#     prob_col = f"prob_{model}"
#
#     label = row.get(label_col, None)
#     prob = row.get(prob_col, -1.0)
#
#     try:
#         prob = float(prob)
#     except Exception:
#         prob = -1.0
#
#     if label is None or pd.isna(label) or str(label).strip() == "":
#         return None, -1.0
#
#     return str(label).strip(), prob
#
#
# def get_model_wait_ms(row, model):
#     wait_col = f"wait_{model}"
#
#     try:
#         wait_s = float(row.get(wait_col, -1.0))
#     except Exception:
#         wait_s = -1.0
#
#     if wait_s < 0:
#         return 0.0
#
#     return wait_s * 1000.0
#
#
# def load_logs(log_path, has_header=False):
#     if has_header:
#         df = pd.read_csv(log_path)
#     else:
#         df = pd.read_csv(
#             log_path,
#             header=None,
#             names=LOG_HEADER,
#             quoting=csv.QUOTE_MINIMAL,
#         )
#
#     df["selected_models_list"] = df["selected_models"].apply(safe_list)
#     df["executed_models_list"] = df["executed_models"].apply(safe_list)
#
#     df["success"] = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
#     df["e2e_ms"] = pd.to_numeric(df["e2e_ms"], errors="coerce").fillna(0.0)
#
#     return df
#
#
# def get_stage_model_sequence(row, mode):
#     if mode == "power2":
#         return row["executed_models_list"]
#     return row["selected_models_list"]
#
#
# def build_stage_predictions(df, num_thresholds=3, mode="seq"):
#     """
#     stage1_conf = actual first model confidence
#     stage2_conf = weighted majority confidence of first 2 models
#     stage3_conf = weighted majority confidence of first 3 models
#     """
#
#     for stage in range(1, num_thresholds + 1):
#         preds = []
#         confs = []
#
#         for _, row in df.iterrows():
#             model_sequence = get_stage_model_sequence(row, mode)
#             stage_models = model_sequence[:stage]
#
#             if not stage_models:
#                 preds.append(None)
#                 confs.append(-1.0)
#                 continue
#
#             if stage == 1:
#                 model = stage_models[0]
#                 pred, conf = get_model_output(row, model)
#             else:
#                 labels = []
#                 probs = []
#
#                 for model in stage_models:
#                     label, prob = get_model_output(row, model)
#                     labels.append(label)
#                     probs.append(prob)
#
#                 pred, conf = weighted_majority(labels, probs)
#
#             preds.append(pred)
#             confs.append(conf)
#
#         df[f"stage{stage}_pred"] = preds
#         df[f"stage{stage}_conf"] = confs
#
#     df["final_pred"] = df["label"]
#
#     return df
#
#
# def estimate_stage_latency_ms(row, stage_num, mode="seq"):
#     model_sequence = get_stage_model_sequence(row, mode)
#
#     if not model_sequence:
#         return float(row["e2e_ms"])
#
#     stage_models = model_sequence[:stage_num]
#     waits = [get_model_wait_ms(row, m) for m in stage_models]
#
#     if not waits:
#         return float(row["e2e_ms"])
#
#     if mode == "seq":
#         return sum(waits)
#
#     if mode == "power2":
#         if stage_num == 1:
#             return waits[0]
#
#         if stage_num == 2:
#             return max(waits[:2]) if len(waits) >= 2 else waits[0]
#
#         if stage_num >= 3:
#             first_batch = max(waits[:2]) if len(waits) >= 2 else waits[0]
#             tail = sum(waits[2:])
#             return first_batch + tail
#
#     return float(row["e2e_ms"])
#
#
# def is_prediction_correct(row, pred, label_json):
#     if pred is None:
#         return False
#
#     if label_json is not None:
#         matched = prediction_matches_wnid(pred, row["selected_folder"], label_json)
#         if matched is not None:
#             return matched
#
#     return (pred == row["final_pred"]) and bool(row["success"])
#
#
# def get_threshold_candidates(df, conf_col, num_quantiles=25, min_value=0.0):
#     values = pd.to_numeric(df[conf_col], errors="coerce").dropna()
#     values = values[(values >= min_value) & (values >= 0)]
#
#     if values.empty:
#         return []
#
#     quantiles = np.linspace(0.05, 0.95, num_quantiles)
#     candidates = values.quantile(quantiles).values
#     candidates = sorted(set(np.round(candidates, 4)))
#
#     return candidates
#
#
# def simulate_thresholds(df, thresholds, label_json=None, mode="seq"):
#     threshold1, threshold2, threshold3 = thresholds
#
#     correct = 0
#     total_latency = 0.0
#
#     stage1_exit = 0
#     stage2_exit = 0
#     stage3_exit = 0
#     ran_all = 0
#
#     stage1_wrong = 0
#     stage2_wrong = 0
#     stage3_wrong = 0
#
#     for _, row in df.iterrows():
#
#         if row["stage1_conf"] >= threshold1:
#             pred = row["stage1_pred"]
#             latency = estimate_stage_latency_ms(row, 1, mode)
#             is_correct = is_prediction_correct(row, pred, label_json)
#
#             stage1_exit += 1
#             if not is_correct:
#                 stage1_wrong += 1
#
#         elif row["stage2_conf"] >= threshold2:
#             pred = row["stage2_pred"]
#             latency = estimate_stage_latency_ms(row, 2, mode)
#             is_correct = is_prediction_correct(row, pred, label_json)
#
#             stage2_exit += 1
#             if not is_correct:
#                 stage2_wrong += 1
#
#         elif row["stage3_conf"] >= threshold3:
#             pred = row["stage3_pred"]
#             latency = estimate_stage_latency_ms(row, 3, mode)
#             is_correct = is_prediction_correct(row, pred, label_json)
#
#             stage3_exit += 1
#             if not is_correct:
#                 stage3_wrong += 1
#
#         else:
#             pred = row["final_pred"]
#             latency = float(row["e2e_ms"])
#
#             if label_json is not None:
#                 is_correct = is_prediction_correct(row, pred, label_json)
#             else:
#                 is_correct = bool(row["success"])
#
#             ran_all += 1
#
#         if is_correct:
#             correct += 1
#
#         total_latency += latency
#
#     n = len(df)
#
#     stage1_error = stage1_wrong / stage1_exit if stage1_exit else 0.0
#     stage2_error = stage2_wrong / stage2_exit if stage2_exit else 0.0
#     stage3_error = stage3_wrong / stage3_exit if stage3_exit else 0.0
#
#     return {
#         "threshold1": threshold1,
#         "threshold2": threshold2,
#         "threshold3": threshold3,
#
#         "accuracy": correct / n if n else 0.0,
#         "avg_latency_ms": total_latency / n if n else 0.0,
#
#         "stage1_exit_rate": stage1_exit / n if n else 0.0,
#         "stage2_exit_rate": stage2_exit / n if n else 0.0,
#         "stage3_exit_rate": stage3_exit / n if n else 0.0,
#         "ran_all_rate": ran_all / n if n else 0.0,
#
#         "stage1_error": stage1_error,
#         "stage2_error": stage2_error,
#         "stage3_error": stage3_error,
#     }
#
#
# def find_best_thresholds(
#     df,
#     target_accuracy,
#     label_json=None,
#     mode="seq",
#     min_threshold1=0.90,
#     min_threshold2=0.55,
#     min_threshold3=0.55,
#     max_stage1_error=0.10,
#     max_stage2_error=0.10,
#     max_stage3_error=0.10,
# ):
#     threshold1_candidates = get_threshold_candidates(
#         df, "stage1_conf", min_value=min_threshold1
#     )
#     threshold2_candidates = get_threshold_candidates(
#         df, "stage2_conf", min_value=min_threshold2
#     )
#     threshold3_candidates = get_threshold_candidates(
#         df, "stage3_conf", min_value=min_threshold3
#     )
#
#     print("\nCandidate counts:")
#     print("threshold1:", len(threshold1_candidates), threshold1_candidates[:5], "...", threshold1_candidates[-5:])
#     print("threshold2:", len(threshold2_candidates), threshold2_candidates[:5], "...", threshold2_candidates[-5:])
#     print("threshold3:", len(threshold3_candidates), threshold3_candidates[:5], "...", threshold3_candidates[-5:])
#
#     results = []
#
#     for t1, t2, t3 in itertools.product(
#         threshold1_candidates,
#         threshold2_candidates,
#         threshold3_candidates,
#     ):
#         result = simulate_thresholds(
#             df,
#             thresholds=[t1, t2, t3],
#             label_json=label_json,
#             mode=mode,
#         )
#         results.append(result)
#
#     summary = pd.DataFrame(results)
#
#     valid = summary[
#         (summary["accuracy"] >= target_accuracy)
#         & (summary["stage1_error"] <= max_stage1_error)
#         & (summary["stage2_error"] <= max_stage2_error)
#         & (summary["stage3_error"] <= max_stage3_error)
#     ].copy()
#
#     if valid.empty:
#         return summary, None
#
#     best = valid.sort_values(
#         ["avg_latency_ms", "accuracy"],
#         ascending=[True, False],
#     ).iloc[0]
#
#     return summary, best
#
#
# def plot_accuracy_latency(summary, best=None):
#     plt.figure(figsize=(7, 5))
#     plt.scatter(summary["avg_latency_ms"], summary["accuracy"], alpha=0.6)
#
#     if best is not None:
#         plt.scatter(
#             [best["avg_latency_ms"]],
#             [best["accuracy"]],
#             marker="*",
#             s=260,
#             label=(
#                 f"Selected: "
#                 f"T1={best['threshold1']}, "
#                 f"T2={best['threshold2']}, "
#                 f"T3={best['threshold3']}"
#             ),
#         )
#         plt.legend()
#
#     plt.xlabel("Average Latency (ms)")
#     plt.ylabel("Accuracy")
#     plt.title("Threshold Search: Accuracy vs Latency")
#     plt.tight_layout()
#     plt.savefig("threshold_accuracy_latency.png", dpi=300)
#     plt.close()
#
#
# def plot_exit_distribution(best):
#     labels = ["Stage 1 Exit", "Stage 2 Exit", "Stage 3 Exit", "Ran All"]
#     values = [
#         best["stage1_exit_rate"],
#         best["stage2_exit_rate"],
#         best["stage3_exit_rate"],
#         best["ran_all_rate"],
#     ]
#
#     plt.figure(figsize=(7, 4))
#     plt.bar(labels, values)
#     plt.ylabel("Fraction of Requests")
#     plt.title("Exit Distribution for Selected Thresholds")
#     plt.tight_layout()
#     plt.savefig("selected_threshold_exit_distribution.png", dpi=300)
#     plt.close()
#
#
# def main():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument(
#         "--log",
#         default="/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_power2_M3.csv",
#     )
#
#     parser.add_argument(
#         "--label-json",
#         default="/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json",
#     )
#
#     parser.add_argument("--target-accuracy", type=float, default=0.90)
#     parser.add_argument("--mode", choices=["seq", "power2"], default="power2")
#     parser.add_argument("--has-header", action="store_true")
#
#     # parser.add_argument("--min-threshold1", type=float, default=0.80)
#     # parser.add_argument("--min-threshold2", type=float, default=0.70)
#     # parser.add_argument("--min-threshold3", type=float, default=0.60)
#     #
#     # parser.add_argument("--max-stage1-error", type=float, default=0.05)
#     # parser.add_argument("--max-stage2-error", type=float, default=0.05)
#     # parser.add_argument("--max-stage3-error", type=float, default=0.05)
#     parser.add_argument("--min-threshold1", type=float, default=0.90)
#     parser.add_argument("--min-threshold2", type=float, default=0.55)
#     parser.add_argument("--min-threshold3", type=float, default=0.55)
#
#     parser.add_argument("--max-stage1-error", type=float, default=0.10)
#     parser.add_argument("--max-stage2-error", type=float, default=0.10)
#     parser.add_argument("--max-stage3-error", type=float, default=0.10)
#
#     args = parser.parse_args()
#
#     label_json = load_label_json(args.label_json)
#     df = load_logs(args.log, has_header=args.has_header)
#
#     df = build_stage_predictions(
#         df,
#         num_thresholds=3,
#         mode=args.mode,
#     )
#
#     print("\nStage confidence preview:")
#     print(df[[
#         "req_id",
#         "executed_models",
#         "stage1_pred",
#         "stage1_conf",
#         "stage2_pred",
#         "stage2_conf",
#         "stage3_pred",
#         "stage3_conf",
#         "confidence",
#         "success",
#     ]].head(20).to_string(index=False))
#
#     print("\nStage1 confidence min/max:")
#     print(df["stage1_conf"].min(), df["stage1_conf"].max())
#
#     print("\nFinal logged success rate:")
#     print(df["success"].mean())
#
#     summary, best = find_best_thresholds(
#         df,
#         target_accuracy=args.target_accuracy,
#         label_json=label_json,
#         mode=args.mode,
#         min_threshold1=args.min_threshold1,
#         min_threshold2=args.min_threshold2,
#         min_threshold3=args.min_threshold3,
#         max_stage1_error=args.max_stage1_error,
#         max_stage2_error=args.max_stage2_error,
#         max_stage3_error=args.max_stage3_error,
#     )
#
#     summary.to_csv("three_threshold_search_summary.csv", index=False)
#
#     print("\n===== Three-Threshold Search Finished =====")
#     print(f"Total threshold combinations tested: {len(summary)}")
#     print(f"Mode: {args.mode}")
#     print(f"Target accuracy: {args.target_accuracy}")
#     print(f"Minimum thresholds: {args.min_threshold1}, {args.min_threshold2}, {args.min_threshold3}")
#     print(f"Max stage errors: {args.max_stage1_error}, {args.max_stage2_error}, {args.max_stage3_error}")
#
#     if best is None:
#         print("\nNo threshold combination satisfied all constraints.")
#     else:
#         print("\n===== Best Thresholds =====")
#         print(best.to_string())
#
#     plot_accuracy_latency(summary, best)
#
#     if best is not None:
#         plot_exit_distribution(best)
#
#     print("\nSaved:")
#     print("three_threshold_search_summary.csv")
#     print("threshold_accuracy_latency.png")
#     if best is not None:
#         print("selected_threshold_exit_distribution.png")
#
#
# if __name__ == "__main__":
#     main()

import ast
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def safe_list(x):
    try:
        if isinstance(x, list):
            return x
        return ast.literal_eval(str(x))
    except Exception:
        return []


def normalize_text(x):
    if x is None or pd.isna(x):
        return ""
    return str(x).lower().replace("_", " ").replace("-", " ").strip()


def load_label_json(path):
    if not path:
        return None
    with open(path, "r") as f:
        return json.load(f)


def prediction_matches_wnid(pred_text, wnid, label_json):
    if label_json is None or wnid not in label_json:
        return None

    pred_norm = normalize_text(pred_text)
    true_names = label_json[wnid]
    true_options = [normalize_text(x) for x in true_names.split(",")]

    return any(pred_norm == option for option in true_options)


def weighted_majority(labels, probs):
    scores = {}

    for label, prob in zip(labels, probs):
        if label is None or pd.isna(label):
            continue

        label = str(label).strip()
        if label == "":
            continue

        try:
            prob = float(prob)
        except Exception:
            continue

        if prob < 0:
            continue

        scores[label] = scores.get(label, 0.0) + prob

    if not scores:
        return None, -1.0

    total = sum(scores.values())
    pred = max(scores, key=scores.get)
    conf = scores[pred] / total if total > 0 else -1.0

    return pred, conf


def load_logs(log_path):
    df = pd.read_csv(log_path)

    df["selected_models_list"] = df["selected_models"].apply(safe_list)
    df["executed_models_list"] = df["executed_models"].apply(safe_list)

    df["Success"] = df["Success"].astype(str).str.lower().isin(["true", "1", "yes"])
    df["e2e_time_ms"] = pd.to_numeric(df["e2e_time_ms"], errors="coerce").fillna(0.0)

    return df


def get_model_output(row, model):
    label_col = f"{model}_label"
    prob_col = f"{model}_prob"

    label = row.get(label_col, "")
    prob = row.get(prob_col, -1.0)

    try:
        prob = float(prob)
    except Exception:
        prob = -1.0

    if label is None or pd.isna(label) or str(label).strip() == "":
        return None, -1.0

    return str(label).strip(), prob


def get_model_wait_ms(row, model):
    wait_col = f"{model}_wait"

    try:
        wait_s = float(row.get(wait_col, -1.0))
    except Exception:
        wait_s = -1.0

    if wait_s < 0:
        return 0.0

    return wait_s * 1000.0


def get_stage_model_sequence(row, mode):
    if mode in ["power2", "parallel_first", "parallel_first_finish"]:
        return row["executed_models_list"]

    return row["selected_models_list"]


def build_stage_predictions(df, mode, num_stages=3):
    for stage in range(1, num_stages + 1):
        preds = []
        confs = []

        for _, row in df.iterrows():
            model_sequence = get_stage_model_sequence(row, mode)
            stage_models = model_sequence[:stage]

            if not stage_models:
                preds.append(None)
                confs.append(-1.0)
                continue

            if stage == 1:
                pred, conf = get_model_output(row, stage_models[0])
            else:
                labels = []
                probs = []

                for model in stage_models:
                    label, prob = get_model_output(row, model)
                    labels.append(label)
                    probs.append(prob)

                pred, conf = weighted_majority(labels, probs)

            preds.append(pred)
            confs.append(conf)

        df[f"stage{stage}_pred"] = preds
        df[f"stage{stage}_conf"] = confs

    df["final_pred"] = df["label"]
    return df


def estimate_stage_latency_ms(row, stage_num, mode):
    model_sequence = get_stage_model_sequence(row, mode)

    if not model_sequence:
        return float(row["e2e_time_ms"])

    stage_models = model_sequence[:stage_num]
    waits = [get_model_wait_ms(row, m) for m in stage_models]

    if not waits:
        return float(row["e2e_time_ms"])

    if mode == "seq":
        return sum(waits)

    if mode in ["power2", "parallel_first", "parallel_first_finish"]:
        if stage_num == 1:
            return waits[0]

        if stage_num == 2:
            return max(waits[:2]) if len(waits) >= 2 else waits[0]

        first_batch = max(waits[:2]) if len(waits) >= 2 else waits[0]
        tail = sum(waits[2:])
        return first_batch + tail

    return float(row["e2e_time_ms"])


def is_prediction_correct(row, pred, label_json):
    if pred is None:
        return False

    if label_json is not None:
        matched = prediction_matches_wnid(pred, row["selected_folder"], label_json)
        if matched is not None:
            return matched

    return normalize_text(pred) == normalize_text(row["final_pred"]) and bool(row["Success"])


def get_threshold_candidates(df, conf_col, min_value, max_value, num_steps=20):
    values = pd.to_numeric(df[conf_col], errors="coerce").dropna()
    values = values[(values >= 0) & (values >= min_value) & (values <= max_value)]

    if values.empty:
        return []

    # Use evenly spaced values, easier to control than quantiles
    candidates = np.linspace(min_value, max_value, num_steps)
    candidates = sorted(set(np.round(candidates, 4)))

    return candidates


def simulate_thresholds(df, thresholds, label_json, mode):
    threshold1, threshold2, threshold3 = thresholds

    correct = 0
    total_latency = 0.0

    stage1_exit = 0
    stage2_exit = 0
    stage3_exit = 0
    ran_all = 0

    stage1_wrong = 0
    stage2_wrong = 0
    stage3_wrong = 0

    for _, row in df.iterrows():

        if row["stage1_conf"] >= threshold1:
            pred = row["stage1_pred"]
            latency = estimate_stage_latency_ms(row, 1, mode)
            ok = is_prediction_correct(row, pred, label_json)

            stage1_exit += 1
            if not ok:
                stage1_wrong += 1

        elif row["stage2_conf"] >= threshold2:
            pred = row["stage2_pred"]
            latency = estimate_stage_latency_ms(row, 2, mode)
            ok = is_prediction_correct(row, pred, label_json)

            stage2_exit += 1
            if not ok:
                stage2_wrong += 1

        elif row["stage3_conf"] >= threshold3:
            pred = row["stage3_pred"]
            latency = estimate_stage_latency_ms(row, 3, mode)
            ok = is_prediction_correct(row, pred, label_json)

            stage3_exit += 1
            if not ok:
                stage3_wrong += 1

        else:
            pred = row["final_pred"]
            latency = float(row["e2e_time_ms"])
            ok = is_prediction_correct(row, pred, label_json)
            ran_all += 1

        if ok:
            correct += 1

        total_latency += latency

    n = len(df)

    return {
        "threshold1": threshold1,
        "threshold2": threshold2,
        "threshold3": threshold3,

        "accuracy": correct / n if n else 0.0,
        "avg_latency_ms": total_latency / n if n else 0.0,

        "stage1_exit_rate": stage1_exit / n if n else 0.0,
        "stage2_exit_rate": stage2_exit / n if n else 0.0,
        "stage3_exit_rate": stage3_exit / n if n else 0.0,
        "ran_all_rate": ran_all / n if n else 0.0,

        "stage1_error": stage1_wrong / stage1_exit if stage1_exit else 0.0,
        "stage2_error": stage2_wrong / stage2_exit if stage2_exit else 0.0,
        "stage3_error": stage3_wrong / stage3_exit if stage3_exit else 0.0,
    }


def find_best_thresholds_score_based(
    df,
    label_json,
    mode,
    min_threshold1,
    max_threshold1,
    min_threshold2,
    max_threshold2,
    min_threshold3,
    max_threshold3,
    threshold_steps,
    latency_weight,
    error_weight,
):
    t1_candidates = get_threshold_candidates(
        df, "stage1_conf", min_threshold1, max_threshold1, threshold_steps
    )
    t2_candidates = get_threshold_candidates(
        df, "stage2_conf", min_threshold2, max_threshold2, threshold_steps
    )
    t3_candidates = get_threshold_candidates(
        df, "stage3_conf", min_threshold3, max_threshold3, threshold_steps
    )

    print("\nCandidate counts:")
    print("threshold1:", len(t1_candidates), t1_candidates[:5], "...", t1_candidates[-5:])
    print("threshold2:", len(t2_candidates), t2_candidates[:5], "...", t2_candidates[-5:])
    print("threshold3:", len(t3_candidates), t3_candidates[:5], "...", t3_candidates[-5:])

    results = []

    for t1, t2, t3 in itertools.product(t1_candidates, t2_candidates, t3_candidates):
        results.append(
            simulate_thresholds(
                df=df,
                thresholds=[t1, t2, t3],
                label_json=label_json,
                mode=mode,
            )
        )

    summary = pd.DataFrame(results)

    if summary.empty:
        return summary, None

    lat_min = summary["avg_latency_ms"].min()
    lat_max = summary["avg_latency_ms"].max()

    summary["latency_norm"] = (
        (summary["avg_latency_ms"] - lat_min) /
        max(lat_max - lat_min, 1e-9)
    )

    summary["avg_stage_error"] = (
        summary["stage1_error"]
        + summary["stage2_error"]
        + summary["stage3_error"]
    ) / 3.0

    summary["total_exit_rate"] = (
        summary["stage1_exit_rate"]
        + summary["stage2_exit_rate"]
        + summary["stage3_exit_rate"]
    )

    # Higher is better:
    # accuracy is rewarded
    # latency is penalized
    # wrong early-exit error is penalized
    summary["score"] = (
        summary["accuracy"]
        - latency_weight * summary["latency_norm"]
        - error_weight * summary["avg_stage_error"]
    )

    best = summary.sort_values(
        ["score", "accuracy", "avg_latency_ms"],
        ascending=[False, False, True],
    ).iloc[0]

    return summary, best


def plot_accuracy_latency(summary, best):
    plt.figure(figsize=(7, 5))
    plt.scatter(summary["avg_latency_ms"], summary["accuracy"], alpha=0.55)

    if best is not None:
        plt.scatter(
            [best["avg_latency_ms"]],
            [best["accuracy"]],
            marker="*",
            s=280,
            label=(
                f"Best: "
                f"T1={best['threshold1']:.3f}, "
                f"T2={best['threshold2']:.3f}, "
                f"T3={best['threshold3']:.3f}"
            ),
        )
        plt.legend()

    plt.xlabel("Average Latency (ms)")
    plt.ylabel("Accuracy")
    plt.title("Score-Based Threshold Search")
    plt.tight_layout()
    plt.savefig("threshold_accuracy_latency_score_based.png", dpi=300)
    plt.close()


def plot_exit_distribution(best):
    labels = ["Stage 1 Exit", "Stage 2 Exit", "Stage 3 Exit", "Ran All"]
    values = [
        best["stage1_exit_rate"],
        best["stage2_exit_rate"],
        best["stage3_exit_rate"],
        best["ran_all_rate"],
    ]

    plt.figure(figsize=(7, 4))
    plt.bar(labels, values)
    plt.ylabel("Fraction of Requests")
    plt.title("Exit Distribution for Selected Thresholds")
    plt.tight_layout()
    plt.savefig("selected_threshold_exit_distribution_score_based.png", dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--log",
        default="/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_modes_baselines.csv",
    )

    parser.add_argument(
        "--label-json",
        default="/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json",
    )

    parser.add_argument(
        "--mode",
        choices=["seq", "power2", "parallel_first", "parallel_first_finish"],
        default="power2",
    )

    parser.add_argument(
        "--filter-mode",
        default="",
        help="Optional: filter effective_cascade_mode, e.g., sequential or parallel_first_finish",
    )

    parser.add_argument("--min-threshold1", type=float, default=0.85)
    parser.add_argument("--max-threshold1", type=float, default=1.00)

    parser.add_argument("--min-threshold2", type=float, default=0.90)
    parser.add_argument("--max-threshold2", type=float, default=1.00)

    parser.add_argument("--min-threshold3", type=float, default=0.90)
    parser.add_argument("--max-threshold3", type=float, default=1.00)

    parser.add_argument("--threshold-steps", type=int, default=20)

    # Tune these:
    # higher latency_weight -> prefer faster thresholds
    # higher error_weight -> avoid wrong early exits
    parser.add_argument("--latency-weight", type=float, default=0.20)
    parser.add_argument("--error-weight", type=float, default=0.30)

    args = parser.parse_args()

    label_json = load_label_json(args.label_json)
    df = load_logs(args.log)

    if args.filter_mode:
        df = df[df["effective_cascade_mode"] == args.filter_mode].copy()

    df = build_stage_predictions(df, mode=args.mode, num_stages=3)

    print("\nRows used:", len(df))
    print("Mode:", args.mode)

    print("\nThreshold ranges:")
    print(f"T1: {args.min_threshold1} to {args.max_threshold1}")
    print(f"T2: {args.min_threshold2} to {args.max_threshold2}")
    print(f"T3: {args.min_threshold3} to {args.max_threshold3}")
    print(f"Latency weight: {args.latency_weight}")
    print(f"Error weight: {args.error_weight}")

    confidence_col = "Confidence" if "Confidence" in df.columns else "Accuracy"

    print("\nStage confidence preview:")
    print(df[[
        "ID",
        "effective_cascade_mode",
        "selected_models",
        "executed_models",
        "stage1_pred",
        "stage1_conf",
        "stage2_pred",
        "stage2_conf",
        "stage3_pred",
        "stage3_conf",
        "label",
        confidence_col,
        "Success",
    ]].head(20).to_string(index=False))

    summary, best = find_best_thresholds_score_based(
        df=df,
        label_json=label_json,
        mode=args.mode,
        min_threshold1=args.min_threshold1,
        max_threshold1=args.max_threshold1,
        min_threshold2=args.min_threshold2,
        max_threshold2=args.max_threshold2,
        min_threshold3=args.min_threshold3,
        max_threshold3=args.max_threshold3,
        threshold_steps=args.threshold_steps,
        latency_weight=args.latency_weight,
        error_weight=args.error_weight,
    )

    summary.to_csv("threshold_search_score_based_summary.csv", index=False)

    print("\n===== Score-Based Threshold Search Finished =====")
    print("Total combinations tested:", len(summary))

    if best is None:
        print("\nNo threshold combination found.")
    else:
        print("\n===== Best Thresholds =====")
        print(best.to_string())

        print("\nUse this in config:")
        print(
            f"THRESHOLD_STAGE = "
            f"[{best['threshold1']:.4f}, {best['threshold2']:.4f}, {best['threshold3']:.4f}]"
        )

    if not summary.empty:
        plot_accuracy_latency(summary, best)

    if best is not None:
        plot_exit_distribution(best)

    print("\nSaved:")
    print("threshold_search_score_based_summary.csv")
    print("threshold_accuracy_latency_score_based.png")
    if best is not None:
        print("selected_threshold_exit_distribution_score_based.png")


if __name__ == "__main__":
    main()