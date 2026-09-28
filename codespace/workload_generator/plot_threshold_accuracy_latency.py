# import ast
# import json
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/threshold_accuracy_latency.csv"
# LABEL_JSON_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json"
#
# OUT_PATH = "threshold_accuracy_latency_tradeoff.png"
# SUMMARY_OUT_PATH = "threshold_accuracy_latency_summary.csv"
#
#
# def normalize_text(x):
#     return str(x).lower().replace("_", " ").replace("-", " ").strip()
#
#
# def load_label_json(path):
#     with open(path, "r") as f:
#         return json.load(f)
#
#
# def prediction_matches_wnid(pred_text, wnid, label_json):
#     if wnid not in label_json:
#         return False
#
#     pred_norm = normalize_text(pred_text)
#     true_options = [
#         normalize_text(x)
#         for x in label_json[wnid].split(",")
#     ]
#
#     return pred_norm in true_options
#
#
# def clean_threshold_stage(x):
#     """
#     Supports:
#     - 0.90
#     - "0.90"
#     - [0.80, 0.87, 0.90]
#     - "[0.80, 0.87, 0.90]"
#
#     Single value 0.90 is converted to:
#     [0.90, 0.90, 0.90]
#     """
#     if pd.isna(x):
#         return "unknown"
#
#     text = str(x).strip()
#
#     try:
#         parsed = ast.literal_eval(text)
#
#         if isinstance(parsed, (list, tuple)):
#             return "[" + ", ".join(f"{float(v):.2f}" for v in parsed) + "]"
#
#         v = float(parsed)
#         return f"[{v:.2f}, {v:.2f}, {v:.2f}]"
#
#     except Exception:
#         try:
#             v = float(text)
#             return f"[{v:.2f}, {v:.2f}, {v:.2f}]"
#         except Exception:
#             return text
#
#
# def get_method(row):
#     baseline = str(row.get("baseline_mode", "none")).lower()
#
#     if baseline != "none":
#         return baseline
#
#     mode = str(row.get("effective_cascade_mode", "unknown")).lower()
#
#     if mode == "parallel_first_finish":
#         return "MODE-S Power2"
#
#     if mode == "sequential":
#         return "MODE-S Sequential"
#
#     if mode in ["mode_s", "full_parallel"]:
#         return "MODE-S Full Parallel"
#
#     return f"MODE-S {mode}"
#
#
# def get_group(method):
#     m = str(method).lower()
#
#     if "mode-s" in m:
#         return "MODE-S"
#
#     if any(x in m for x in ["smallest", "random", "power2"]):
#         return "Baselines"
#
#     return "Other"
#
#
# label_json = load_label_json(LABEL_JSON_PATH)
# # df = pd.read_csv(CSV_PATH)
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
#     "baseline_mode",
#     "configured_cascade_mode",
#     "effective_cascade_mode",
#
#     *[f"{m}_label" for m in MODEL_LIST],
#     *[f"{m}_prob" for m in MODEL_LIST],
#     *[f"{m}_wait" for m in MODEL_LIST],
#
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "recent_latency_overhead_s",
# ]
#
#
# def load_log_csv(path):
#     df = pd.read_csv(path)
#
#     if "Success" not in df.columns:
#         print("[WARN] CSV header not found. Loading with manual LOG_HEADER.")
#         df = pd.read_csv(path, header=None)
#         df = df.iloc[:, :len(LOG_HEADER)]
#         df.columns = LOG_HEADER
#
#     print("Loaded columns:")
#     print(df.columns.tolist())
#
#     return df
#
#
# df = load_log_csv(CSV_PATH)
#
# # -----------------------------
# # Clean fields
# # -----------------------------
# df["Success"] = df["Success"].astype(str).str.lower().isin(["true", "1", "yes"])
#
# # RespTime is true client-side end-to-end latency in seconds
# df["RespTime"] = pd.to_numeric(df["RespTime"], errors="coerce")
# df["latency_ms"] = df["RespTime"] * 1000.0
#
# df = df[df["Success"] == True].copy()
# df = df.dropna(subset=["latency_ms"])
#
# # -----------------------------
# # Build grouping labels
# # -----------------------------
# df["method"] = df.apply(get_method, axis=1)
# df["threshold_key"] = df["threshold_stage"].apply(clean_threshold_stage)
#
# # Optional: group by both threshold and method
# df["method_threshold"] = df["method"] + " | " + df["threshold_key"]
#
# # -----------------------------
# # Compute true accuracy
# # -----------------------------
# df["correct"] = df.apply(
#     lambda row: prediction_matches_wnid(
#         row["label"],
#         row["selected_folder"],
#         label_json,
#     ),
#     axis=1,
# )
#
# # -----------------------------
# # Group by threshold only
# # -----------------------------
# summary = (
#     df.groupby("threshold_key")
#     .agg(
#         avg_latency_ms=("latency_ms", "mean"),
#         accuracy=("correct", "mean"),
#         requests=("correct", "count"),
#     )
#     .reset_index()
# )
#
# summary["accuracy_percent"] = summary["accuracy"] * 100
# summary = summary.sort_values("avg_latency_ms")
#
# print("\n===== Accuracy-Latency by Threshold =====")
# print(summary.to_string(index=False))
#
# summary.to_csv(SUMMARY_OUT_PATH, index=False)
#
# # -----------------------------
# # Plot: one point per threshold
# # -----------------------------
# plt.figure(figsize=(10, 6))
#
# plt.scatter(
#     summary["avg_latency_ms"],
#     summary["accuracy_percent"],
#     s=130,
# )
#
# for _, row in summary.iterrows():
#     plt.annotate(
#         row["threshold_key"],
#         (row["avg_latency_ms"], row["accuracy_percent"]),
#         textcoords="offset points",
#         xytext=(6, 5),
#         fontsize=8,
#     )
#
# plt.xlabel("Average End-to-End Latency (ms)")
# plt.ylabel("Average Accuracy (%)")
# plt.title("Average Accuracy–Latency Trade-off Across Threshold Values")
# plt.grid(True, alpha=0.35)
# plt.tight_layout()
#
# plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.show()
#
# print(f"\nSaved graph to: {OUT_PATH}")
# print(f"Saved summary to: {SUMMARY_OUT_PATH}")



import ast
import json
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/threshold_accuracy_latency.csv"
LABEL_JSON_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json"

OUT_PATH = "threshold_accuracy_latency_tradeoff_by_mode.png"
SUMMARY_OUT_PATH = "threshold_accuracy_latency_summary_by_mode.csv"


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

LOG_HEADER = [
    "ID", "Deadline", "IAR", "RespTime", "RunTime",
    "selected_models", "executed_models", "label", "Accuracy",
    "combiner_policy", "e2e_time_ms", "Success", "selected_folder",
    "ensemble_size", "threshold", "threshold_stage", "parallel_first_n",
    "stop_reason", "qps_est", "main_policy", "alpha", "alpha_mode",
    "baseline_mode", "configured_cascade_mode", "effective_cascade_mode",

    *[f"{m}_label" for m in MODEL_LIST],
    *[f"{m}_prob" for m in MODEL_LIST],
    *[f"{m}_wait" for m in MODEL_LIST],

    "selected_est_latency_s",
    "raw_selected_est_latency_s",
    "recent_latency_overhead_s",
]


def load_log_csv(path):
    df = pd.read_csv(path)

    if "Success" not in df.columns:
        print("[WARN] CSV header not found. Loading with manual LOG_HEADER.")
        df = pd.read_csv(path, header=None)
        df = df.iloc[:, :len(LOG_HEADER)]
        df.columns = LOG_HEADER

    return df


def normalize_text(x):
    return str(x).lower().replace("_", " ").replace("-", " ").strip()


def load_label_json(path):
    with open(path, "r") as f:
        return json.load(f)


def prediction_matches_wnid(pred_text, wnid, label_json):
    if wnid not in label_json:
        return False

    pred_norm = normalize_text(pred_text)
    true_options = [
        normalize_text(x)
        for x in label_json[wnid].split(",")
    ]

    return pred_norm in true_options


def is_empty_value(x):
    if pd.isna(x):
        return True

    text = str(x).strip().lower()
    return text in ["", "[]", "nan", "none", "null"]


def format_single_threshold(x):
    try:
        return f"T={float(x):.2f}"
    except Exception:
        return str(x)


def format_stage_threshold(x):
    try:
        parsed = ast.literal_eval(str(x).strip())

        if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
            return "[" + ", ".join(f"{float(v):.2f}" for v in parsed) + "]"

        return format_single_threshold(parsed)

    except Exception:
        return format_single_threshold(x)


def get_threshold_key(row):
    threshold_stage = row.get("threshold_stage", "")
    threshold = row.get("threshold", "")

    if not is_empty_value(threshold_stage):
        return format_stage_threshold(threshold_stage)

    if not is_empty_value(threshold):
        return format_single_threshold(threshold)

    return "unknown"


def get_mode_label(mode):
    mode = str(mode).lower()

    if mode == "parallel_first_finish":
        return "Power2"

    if mode == "sequential":
        return "Sequential"

    if mode in ["mode_s", "full_parallel"]:
        return "Full Parallel"

    return mode


label_json = load_label_json(LABEL_JSON_PATH)
df = load_log_csv(CSV_PATH)

print("Unique threshold_stage values:")
print(df["threshold_stage"].dropna().unique()[:20])

print("\nUnique threshold values:")
print(df["threshold"].dropna().unique()[:20])

print("\nUnique effective_cascade_mode values:")
print(df["effective_cascade_mode"].dropna().unique()[:20])

df["Success"] = df["Success"].astype(str).str.lower().isin(["true", "1", "yes"])

# RespTime is true client-side end-to-end latency in seconds
df["RespTime"] = pd.to_numeric(df["RespTime"], errors="coerce")
df["latency_ms"] = df["RespTime"] * 1000.0

df = df[df["Success"] == True].copy()
df = df.dropna(subset=["latency_ms"])

df["threshold_key"] = df.apply(get_threshold_key, axis=1)
df["mode_label"] = df["effective_cascade_mode"].apply(get_mode_label)

df["correct"] = df.apply(
    lambda row: prediction_matches_wnid(
        row["label"],
        row["selected_folder"],
        label_json,
    ),
    axis=1,
)

summary = (
    df.groupby(["mode_label", "threshold_key"])
    .agg(
        avg_latency_ms=("latency_ms", "mean"),
        accuracy=("correct", "mean"),
        requests=("correct", "count"),
    )
    .reset_index()
)

summary["accuracy_percent"] = summary["accuracy"] * 100
summary = summary.sort_values(["mode_label", "avg_latency_ms"])

print("\n===== Accuracy-Latency by Mode and Threshold =====")
print(summary.to_string(index=False))

summary.to_csv(SUMMARY_OUT_PATH, index=False)

plt.figure(figsize=(10, 6))

mode_markers = {
    "Sequential": "o",
    "Power2": "^",
    "Full Parallel": "s",
}

for mode, gdf in summary.groupby("mode_label"):
    plt.scatter(
        gdf["avg_latency_ms"],
        gdf["accuracy_percent"],
        marker=mode_markers.get(mode, "o"),
        s=130,
        label=mode,
    )

    for _, row in gdf.iterrows():
        plt.annotate(
            row["threshold_key"],
            (row["avg_latency_ms"], row["accuracy_percent"]),
            textcoords="offset points",
            xytext=(8, 5),
            fontsize=8,
        )

plt.xlabel("Average End-to-End Latency (ms)")
plt.ylabel("Average Accuracy (%)")
plt.title("Accuracy–Latency Trade-off Across Threshold Values")
plt.grid(True, alpha=0.35)
plt.legend(title="Execution Mode")
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"\nSaved graph to: {OUT_PATH}")
print(f"Saved summary to: {SUMMARY_OUT_PATH}")