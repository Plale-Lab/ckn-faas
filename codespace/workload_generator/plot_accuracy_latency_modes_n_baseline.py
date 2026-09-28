# import pandas as pd
# import matplotlib.pyplot as plt
# import ast
#
# # =========================
# # CONFIG
# # =========================
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_latency_all_methods_n_baselines.csv"   # change this
# OUT_PATH = "accuracy_latency_tradeoff.png"
#
# COLS = [
#     "request_id",
#     "deadline_ms",
#     "sleep_ms",
#     "e2e_latency_s",        # this is your real e2e latency: 0.775568...
#     "client_overhead_ms",
#     "selected_models",
#     "executed_models",
#     "label",
#     "confidence",
#     "combiner_policy",
#     "server_e2e_time_ms",
#     "success",
#     "selected_folder",
#     "ensemble_size",
#     "threshold",
#     "threshold_stage",
#     "parallel_first_n",
#     "stop_reason",
#     "response_time_s",
#     "main_policy",
#     "alpha",
#     "alpha_mode",
#     "image_name",
# ]
#
# # =========================
# # LOAD CSV
# # =========================
# df = pd.read_csv(CSV_PATH, header=None)
#
# # keep only known columns
# df = df.iloc[:, :len(COLS)]
# df.columns = COLS
#
# print("Loaded columns:")
# print(df.columns.tolist())
#
# # =========================
# # CLEAN DATA
# # =========================
# df["success"] = df["success"].astype(str).str.lower().isin(["true", "1", "yes"])
# df["e2e_latency_s"] = pd.to_numeric(df["e2e_latency_s"], errors="coerce")
# df["e2e_latency_ms"] = df["e2e_latency_s"] * 1000
# df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
# df["parallel_first_n"] = pd.to_numeric(df["parallel_first_n"], errors="coerce").fillna(0).astype(int)
#
# df = df[df["success"] == True].copy()
# df = df.dropna(subset=["e2e_latency_ms"])
#
# # =========================
# # HELPERS
# # =========================
# def parse_list(x):
#     try:
#         return ast.literal_eval(str(x))
#     except Exception:
#         return []
#
#
# def is_correct(row):
#     pred = str(row["label"]).lower()
#     gt = str(row["selected_folder"]).lower()
#
#     if not pred or not gt or gt == "unknown":
#         return False
#
#     # Your prediction is text label, ground truth is folder wnid.
#     # This simple method only works if selected_folder already contains text.
#     # If selected_folder is wnid like n01753488, use label.json mapping later.
#     return gt in pred or pred in gt
#
#
# def get_method(row):
#     selected = parse_list(row["selected_models"])
#     executed = parse_list(row["executed_models"])
#     stop_reason = str(row["stop_reason"]).lower()
#     combiner = str(row["combiner_policy"]).lower()
#
#     if len(selected) == 1:
#         return selected[0]
#
#     if row["parallel_first_n"] == 2 or "first_finished" in stop_reason:
#         return "Power2 MODE-S"
#
#     if combiner == "weighted_majority" and len(executed) == len(selected):
#         return "Full Parallel MODE-S"
#
#     return "Sequential MODE-S"
#
#
# def get_group(method):
#     m = method.lower()
#
#     if "mode-s" in m:
#         return "MODE-S"
#
#     if any(x in m for x in ["smallest", "random", "power2"]):
#         return "Baselines"
#
#     return "Single Models"
#
#
# # =========================
# # BUILD METHOD + ACCURACY
# # =========================
# df["method"] = df.apply(get_method, axis=1)
# df["correct"] = df.apply(is_correct, axis=1)
#
# summary = (
#     df.groupby("method")
#     .agg(
#         avg_latency_ms=("e2e_latency_ms", "mean"),
#         accuracy=("correct", "mean"),
#         requests=("correct", "count"),
#     )
#     .reset_index()
# )
#
# summary["accuracy_percent"] = summary["accuracy"] * 100
# summary["group"] = summary["method"].apply(get_group)
#
# print("\nSummary:")
# print(summary)
#
# # =========================
# # PLOT
# # =========================
# markers = {
#     "Single Models": "o",
#     "Baselines": "^",
#     "MODE-S": "*",
# }
#
# plt.figure(figsize=(10, 6))
#
# for group, gdf in summary.groupby("group"):
#     plt.scatter(
#         gdf["avg_latency_ms"],
#         gdf["accuracy_percent"],
#         marker=markers.get(group, "o"),
#         s=180 if group == "MODE-S" else 100,
#         label=group,
#     )
#
# for _, row in summary.iterrows():
#     plt.annotate(
#         row["method"],
#         (row["avg_latency_ms"], row["accuracy_percent"]),
#         textcoords="offset points",
#         xytext=(6, 5),
#         fontsize=8,
#     )
#
# plt.xlabel("Average End-to-End Latency (ms)")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy–Latency Trade-off Across MODE-S")
# plt.grid(True, alpha=0.35)
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.show()
#
# print(f"\nSaved graph to: {OUT_PATH}")



# import pandas as pd
# import matplotlib.pyplot as plt
# import json
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_latency_all_methods_n_baselines.csv"
# LABEL_JSON_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json"
# OUT_PATH = "accuracy_latency_tradeoff.png"
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
#     m = method.lower()
#
#     if "mode-s" in m:
#         return "MODE-S"
#
#     if any(x in m for x in ["smallest", "random", "power2"]):
#         return "Baselines"
#
#     return "Single Models"
#
#
# label_json = load_label_json(LABEL_JSON_PATH)
# df = pd.read_csv(CSV_PATH)
#
# df["Success"] = df["Success"].astype(str).str.lower().isin(["true", "1", "yes"])
# df["e2e_time_ms"] = pd.to_numeric(df["e2e_time_ms"], errors="coerce")
#
# df = df[df["Success"] == True].copy()
# df = df.dropna(subset=["e2e_time_ms"])
#
# df["method"] = df.apply(get_method, axis=1)
#
# df["correct"] = df.apply(
#     lambda row: prediction_matches_wnid(
#         row["label"],
#         row["selected_folder"],
#         label_json,
#     ),
#     axis=1,
# )
#
# summary = (
#     df.groupby("method")
#     .agg(
#         avg_latency_ms=("e2e_time_ms", "mean"),
#         accuracy=("correct", "mean"),
#         requests=("correct", "count"),
#     )
#     .reset_index()
# )
#
# summary["accuracy_percent"] = summary["accuracy"] * 100
# summary["group"] = summary["method"].apply(get_group)
#
# print(summary)
#
# markers = {
#     "MODE-S": "*",
#     "Baselines": "^",
#     "Single Models": "o",
# }
#
# plt.figure(figsize=(10, 6))
#
# for group, gdf in summary.groupby("group"):
#     plt.scatter(
#         gdf["avg_latency_ms"],
#         gdf["accuracy_percent"],
#         marker=markers.get(group, "o"),
#         s=180 if group == "MODE-S" else 100,
#         label=group,
#     )
#
# for _, row in summary.iterrows():
#     plt.annotate(
#         row["method"],
#         (row["avg_latency_ms"], row["accuracy_percent"]),
#         textcoords="offset points",
#         xytext=(6, 5),
#         fontsize=8,
#     )
#
# plt.xlabel("Average End-to-End Latency (ms)")
# plt.ylabel("Accuracy (%)")
# plt.title("Accuracy–Latency Trade-off Across MODE-S and Baselines")
# plt.grid(True, alpha=0.35)
# plt.legend()
# plt.tight_layout()
#
# plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
# plt.show()
#
# print(f"Saved graph to: {OUT_PATH}")



import pandas as pd
import matplotlib.pyplot as plt
import json

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_latency_all_methods_n_baselines.csv"
LABEL_JSON_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json"
OUT_PATH = "accuracy_latency_tradeoff.png"


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


def get_method(row):
    baseline = str(row.get("baseline_mode", "none")).lower()

    if baseline != "none":
        return baseline

    mode = str(row.get("effective_cascade_mode", "unknown")).lower()

    if mode == "parallel_first_finish":
        return "MODE-S Power2"

    if mode == "sequential":
        return "MODE-S Sequential"

    if mode in ["mode_s", "full_parallel"]:
        return "MODE-S Full Parallel"

    return f"MODE-S {mode}"


def get_group(method):
    m = method.lower()

    if "mode-s" in m:
        return "MODE-S"

    if any(x in m for x in ["smallest", "random", "power2"]):
        return "Baselines"

    return "Single Models"


label_json = load_label_json(LABEL_JSON_PATH)
df = pd.read_csv(CSV_PATH)

df["Success"] = df["Success"].astype(str).str.lower().isin(["true", "1", "yes"])

# IMPORTANT:
# RespTime is your true client-side end-to-end latency in seconds.
df["RespTime"] = pd.to_numeric(df["RespTime"], errors="coerce")
df["latency_ms"] = df["RespTime"] * 1000.0

df = df[df["Success"] == True].copy()
df = df.dropna(subset=["latency_ms"])

df["method"] = df.apply(get_method, axis=1)

df["correct"] = df.apply(
    lambda row: prediction_matches_wnid(
        row["label"],
        row["selected_folder"],
        label_json,
    ),
    axis=1,
)

summary = (
    df.groupby("method")
    .agg(
        avg_latency_ms=("latency_ms", "mean"),
        accuracy=("correct", "mean"),
        requests=("correct", "count"),
    )
    .reset_index()
)

summary["accuracy_percent"] = summary["accuracy"] * 100
summary["group"] = summary["method"].apply(get_group)

print(summary)

markers = {
    "MODE-S": "*",
    "Baselines": "^",
    "Single Models": "o",
}

plt.figure(figsize=(10, 6))

for group, gdf in summary.groupby("group"):
    plt.scatter(
        gdf["avg_latency_ms"],
        gdf["accuracy_percent"],
        marker=markers.get(group, "o"),
        s=180 if group == "MODE-S" else 100,
        label=group,
    )

for _, row in summary.iterrows():
    plt.annotate(
        row["method"],
        (row["avg_latency_ms"], row["accuracy_percent"]),
        textcoords="offset points",
        xytext=(6, 5),
        fontsize=8,
    )

plt.xlabel("Average End-to-End Latency (ms)")
plt.ylabel("Accuracy (%)")
plt.title("Accuracy–Latency Trade-off Across MODE-S and Baselines")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()

plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.show()

print(f"Saved graph to: {OUT_PATH}")