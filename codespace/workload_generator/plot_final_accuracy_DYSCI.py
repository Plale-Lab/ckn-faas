# import os
# import json
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # ============================================================
# # PATHS
# # ============================================================
#
# CSV_PATH = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/"
#     "seq_latency_IAT_table.csv"
# )
#
# LABEL_JSON_PATH = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/Labels.json"
# )
#
# OUT_DIR = "average_accuracy_plots"
# os.makedirs(OUT_DIR, exist_ok=True)
#
#
# # ============================================================
# # LABELS
# # ============================================================
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "sequential": "Seq. Cascade",
#     "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 Smallest to Largest",
#     "random_to_random": "Random to Random",
#     "smallest_to_largest": "Smallest to Largest",
#     "spec_b_smallest_to_largest": "Smallest and Largest",
#     "spec_b_random_to_random": "Random Model Pair",
#     "full_ensemble_edge": "3-Model Edge Ensemble",
#     "full_ensemble_cloud": "Full Cloud Ensemble",
#     "frugalml": "FrugalML",
# }
#
#
# # ============================================================
# # COLORS
# # ============================================================
#
# ERROR_COLORS = {
#     0.090: "#1f77b4",
#     0.100: "#ff7f0e",
#     0.120: "#2ca02c",
#     0.140: "#8c564b",
# }
#
# # BASELINE_COLORS = {
# #     "frugalml": "#d62728",
# #     "random_to_random": "#d62728",
# #     "smallest_to_largest": "#9467bd",
# #     "power2_smallest_to_largest": "#9467bd",
# #     "spec_b_smallest_to_largest": "#9467bd",
# #     "spec_b_random_to_random": "#d62728",
# #     "full_ensemble_edge": "#7f7f7f",
# #     "full_ensemble_cloud": "#bcbd22",
# # }
#
# BASELINE_COLORS = {
#     # Dark Red
#     "frugalml": "#8B0000",
#     "random_to_random": "#8B0000",
#     "spec_b_random_to_random": "#8B0000",
#
#     # Dark Purple
#     "smallest_to_largest": "#4B0082",
#     "power2_smallest_to_largest": "#4B0082",
#     "spec_b_smallest_to_largest": "#4B0082",
#
#     # Dark Gray
#     "full_ensemble_edge": "#404040",
#
#     # Dark Olive Green
#     "full_ensemble_cloud": "#556B2F",
# }
#
#
# # ============================================================
# # HELPERS
# # ============================================================
#
# def convert_to_bool(value):
#     """Convert common CSV Boolean representations."""
#     if pd.isna(value):
#         return pd.NA
#
#     normalized = str(value).strip().lower()
#
#     if normalized in {"true", "1", "yes"}:
#         return True
#
#     if normalized in {"false", "0", "no"}:
#         return False
#
#     return pd.NA
#
#
# def normalize_text(value):
#     """
#     Normalize model predictions and ImageNet labels.
#
#     Example:
#         "Leatherback_Turtle" -> "leatherback turtle"
#     """
#     return (
#         str(value)
#         .lower()
#         .replace("_", " ")
#         .replace("-", " ")
#         .strip()
#     )
#
#
# def load_label_json(path):
#     with open(path, "r", encoding="utf-8") as file:
#         return json.load(file)
#
#
# def prediction_matches_wnid(pred_text, wnid, label_json):
#     """
#     Compare a predicted text label with the possible labels associated
#     with the ground-truth ImageNet WNID.
#     """
#     if pd.isna(pred_text) or pd.isna(wnid):
#         return False
#
#     wnid = str(wnid).strip()
#     pred_norm = normalize_text(pred_text)
#
#     if not pred_norm:
#         return False
#
#     if wnid not in label_json:
#         return False
#
#     label_value = label_json[wnid]
#
#     # Handle:
#     # "albatross, mollymawk"
#     #
#     # and also possible JSON lists:
#     # ["albatross", "mollymawk"]
#     if isinstance(label_value, list):
#         true_options = [
#             normalize_text(option)
#             for option in label_value
#         ]
#     else:
#         true_options = [
#             normalize_text(option)
#             for option in str(label_value).split(",")
#         ]
#
#     # Exact normalized match
#     return pred_norm in true_options
#
#
# def add_confidence_interval(summary):
#     """
#     Approximate 95% confidence interval for a binary correctness value.
#     """
#     summary = summary.copy()
#
#     p = summary["accuracy_fraction"]
#     n = summary["request_count"].clip(lower=1)
#
#     standard_error = np.sqrt(
#         p * (1.0 - p) / n
#     )
#
#     margin = 1.96 * standard_error
#
#     summary["accuracy_percent"] = p * 100.0
#     summary["ci_margin_percent"] = margin * 100.0
#
#     return summary
#
#
# # ============================================================
# # LOAD LABEL MAPPING
# # ============================================================
#
# label_json = load_label_json(LABEL_JSON_PATH)
#
# print(f"Loaded {len(label_json)} WNID mappings from Labels.json")
#
#
# # ============================================================
# # LOAD CSV
# # ============================================================
#
# df = pd.read_csv(CSV_PATH)
#
# # Normalize headings:
# # Success -> success
# # IAR -> iar
# # RespTime -> resptime
# df.columns = (
#     df.columns
#     .str.strip()
#     .str.lower()
# )
#
# print("\nCSV columns:")
# print(df.columns.tolist())
#
#
# # ============================================================
# # CHECK REQUIRED COLUMNS
# # ============================================================
#
# required_columns = [
#     "label",
#     "selected_folder",
#     "success",
#     "static_config_enabled",
#     "static_policy",
#     "static_cost",
#     "static_error",
#     "baseline_mode",
# ]
#
# missing_columns = [
#     column
#     for column in required_columns
#     if column not in df.columns
# ]
#
# if missing_columns:
#     raise ValueError(
#         f"Missing required columns: {missing_columns}"
#     )
#
#
# # ============================================================
# # CLEAN DATA
# # ============================================================
#
# df["static_cost"] = pd.to_numeric(
#     df["static_cost"],
#     errors="coerce",
# )
#
# df["static_error"] = pd.to_numeric(
#     df["static_error"],
#     errors="coerce",
# )
#
# df["success_bool"] = df["success"].apply(
#     convert_to_bool
# )
#
# df["is_static"] = df["static_config_enabled"].apply(
#     convert_to_bool
# )
#
# df["baseline_mode"] = (
#     df["baseline_mode"]
#     .fillna("none")
#     .astype(str)
#     .str.strip()
#     .str.lower()
# )
#
# df["static_policy"] = (
#     df["static_policy"]
#     .fillna("none")
#     .astype(str)
#     .str.strip()
#     .str.lower()
# )
#
#
# # ============================================================
# # FILTER ONLY SUCCESSFULLY COMPLETED REQUESTS
# # ============================================================
#
# # Success means the request completed successfully.
# # It is NOT classification correctness.
# df = df[
#     df["success_bool"] == True
# ].copy()
#
# df = df.dropna(
#     subset=[
#         "label",
#         "selected_folder",
#         "is_static",
#     ]
# ).copy()
#
#
# # ============================================================
# # CALCULATE TRUE CLASSIFICATION CORRECTNESS
# # ============================================================
#
# df["correct"] = df.apply(
#     lambda row: prediction_matches_wnid(
#         pred_text=row["label"],
#         wnid=row["selected_folder"],
#         label_json=label_json,
#     ),
#     axis=1,
# )
#
# df["correct_numeric"] = df["correct"].astype(float)
#
#
# # ============================================================
# # DEBUG CHECKS
# # ============================================================
#
# print("\nCorrectness distribution:")
# print(df["correct"].value_counts(dropna=False))
#
# print("\nOverall classification accuracy:")
# print(f"{df['correct'].mean() * 100:.2f}%")
#
# unknown_wnids = sorted(
#     set(df["selected_folder"].astype(str))
#     - set(label_json.keys())
# )
#
# print("\nNumber of WNIDs missing from Labels.json:")
# print(len(unknown_wnids))
#
# if unknown_wnids:
#     print("First missing WNIDs:")
#     print(unknown_wnids[:10])
#
#
# # Show a few examples
# debug_examples = df[
#     [
#         "label",
#         "selected_folder",
#         "correct",
#     ]
# ].head(15)
#
# print("\nSample prediction comparisons:")
# print(debug_examples.to_string(index=False))
#
#
# # ============================================================
# # SEPARATE DYSCI AND BASELINES
# # ============================================================
#
# static_df = df[
#     df["is_static"] == True
# ].copy()
#
# baseline_df = df[
#     (df["is_static"] == False)
#     & (~df["baseline_mode"].isin(
#         ["none", "", "nan"]
#     ))
# ].copy()
#
#
# SELECTED_BASELINES = [
#     "frugalml",
#     "random_to_random",
#     "smallest_to_largest",
# ]
#
# baseline_df = baseline_df[
#     baseline_df["baseline_mode"].isin(
#         SELECTED_BASELINES
#     )
# ].copy()
#
#
# # ============================================================
# # CALCULATE AVERAGE ACCURACY
# # ============================================================
#
# static_summary = (
#     static_df.groupby(
#         [
#             "static_policy",
#             "static_cost",
#             "static_error",
#         ],
#         as_index=False,
#     )
#     .agg(
#         accuracy_fraction=(
#             "correct_numeric",
#             "mean",
#         ),
#         correct_predictions=(
#             "correct_numeric",
#             "sum",
#         ),
#         request_count=(
#             "correct_numeric",
#             "count",
#         ),
#     )
# )
#
# baseline_summary = (
#     baseline_df.groupby(
#         ["baseline_mode"],
#         as_index=False,
#     )
#     .agg(
#         accuracy_fraction=(
#             "correct_numeric",
#             "mean",
#         ),
#         correct_predictions=(
#             "correct_numeric",
#             "sum",
#         ),
#         request_count=(
#             "correct_numeric",
#             "count",
#         ),
#     )
# )
#
#
# static_summary = add_confidence_interval(
#     static_summary
# )
#
# baseline_summary = add_confidence_interval(
#     baseline_summary
# )
#
#
# # ============================================================
# # PRINT SUMMARY
# # ============================================================
#
# print("\nDYSCI average accuracy:")
# print(
#     static_summary[
#         [
#             "static_policy",
#             "static_cost",
#             "static_error",
#             "correct_predictions",
#             "request_count",
#             "accuracy_percent",
#         ]
#     ].to_string(index=False)
# )
#
# print("\nBaseline average accuracy:")
# print(
#     baseline_summary[
#         [
#             "baseline_mode",
#             "correct_predictions",
#             "request_count",
#             "accuracy_percent",
#         ]
#     ].to_string(index=False)
# )
#
#
# # ============================================================
# # CREATE BAR CHART
# # ============================================================
#
# for policy, policy_data in static_summary.groupby(
#     "static_policy"
# ):
#
#     policy_display = POLICY_LABELS.get(
#         policy,
#         policy,
#     )
#
#     costs = sorted(
#         policy_data["static_cost"]
#         .dropna()
#         .unique()
#     )
#
#     for cost in costs:
#
#         cost_data = policy_data[
#             policy_data["static_cost"] == cost
#         ].sort_values("static_error")
#
#         labels = []
#         accuracies = []
#         # confidence_intervals = []
#         colors = []
#         request_counts = []
#
#         # ----------------------------------------------------
#         # DYSCI bars
#         # ----------------------------------------------------
#
#         for _, row in cost_data.iterrows():
#
#             error_value = float(
#                 row["static_error"]
#             )
#
#             labels.append(
#                 f"{policy_display}\nE={error_value:.3f}"
#             )
#
#             accuracies.append(
#                 float(row["accuracy_percent"])
#             )
#
#             # confidence_intervals.append(
#             #     float(row["ci_margin_percent"])
#             # )
#
#             request_counts.append(
#                 int(row["request_count"])
#             )
#
#             colors.append(
#                 ERROR_COLORS.get(
#                     round(error_value, 3),
#                     "#7f7f7f",
#                 )
#             )
#
#         # ----------------------------------------------------
#         # Baseline bars
#         # ----------------------------------------------------
#
#         for _, row in baseline_summary.iterrows():
#
#             baseline_name = row["baseline_mode"]
#
#             labels.append(
#                 BASELINE_LABELS.get(
#                     baseline_name,
#                     baseline_name,
#                 )
#             )
#
#             accuracies.append(
#                 float(row["accuracy_percent"])
#             )
#
#             # confidence_intervals.append(
#             #     float(row["ci_margin_percent"])
#             # )
#
#             request_counts.append(
#                 int(row["request_count"])
#             )
#
#             colors.append(
#                 BASELINE_COLORS.get(
#                     baseline_name,
#                     "#7f7f7f",
#                 )
#             )
#
#         if not accuracies:
#             continue
#
#         fig, ax = plt.subplots(
#             figsize=(12, 6)
#         )
#
#         x_positions = np.arange(
#             len(labels)
#         )
#
#         bars = ax.bar(
#             x_positions,
#             accuracies,
#             # yerr=confidence_intervals,
#             # capsize=5,
#             color=colors,
#             edgecolor="black",
#             linewidth=0.8,
#         )
#
#         # Accuracy value above each bar
#         for bar, accuracy, count in zip(
#             bars,
#             accuracies,
#             request_counts,
#         ):
#             ax.text(
#                 bar.get_x() + bar.get_width() / 2,
#                 bar.get_height() + 0.5,
#                 f"{accuracy:.1f}%",
#                 ha="center",
#                 va="bottom",
#                 fontsize=11,
#             )
#
#         ax.set_xticks(
#             x_positions
#         )
#
#         ax.set_xticklabels(
#             labels,
#             fontsize=12,
#         )
#
#         ax.set_ylabel(
#             "Average Classification Accuracy (%)",
#             fontsize=16,
#         )
#
#         ax.set_title(
#             f"Average Accuracy — "
#             f"{policy_display}, Cost Constraint C={cost:g}",
#             fontsize=18,
#         )
#
#         # Automatically zoom while leaving enough space for labels
#         minimum_accuracy = min(accuracies)
#
#         lower_limit = max(
#             0,
#             np.floor(minimum_accuracy / 5.0) * 5.0 - 5.0,
#         )
#
#         ax.set_ylim(
#             lower_limit,
#             103,
#         )
#
#         ax.grid(
#             axis="y",
#             alpha=0.3,
#         )
#
#         ax.tick_params(
#             axis="y",
#             labelsize=13,
#         )
#
#         plt.tight_layout()
#
#         safe_policy = (
#             policy_display
#             .lower()
#             .replace(" ", "_")
#             .replace(".", "")
#             .replace("-", "_")
#         )
#
#         out_base = os.path.join(
#             OUT_DIR,
#             (
#                 f"{safe_policy}_"
#                 f"cost_{cost:g}_"
#                 "average_accuracy"
#             ),
#         )
#
#         plt.savefig(
#             f"{out_base}.png",
#             dpi=300,
#             bbox_inches="tight",
#             facecolor="white",
#         )
#
#         plt.savefig(
#             f"{out_base}.pdf",
#             bbox_inches="tight",
#             facecolor="white",
#         )
#
#         print(
#             f"\nSaved:\n"
#             f"  {out_base}.png\n"
#             f"  {out_base}.pdf"
#         )
#
#         plt.show()
#         plt.close(fig)
#
#
# print(
#     f"\nAll figures saved in: {OUT_DIR}"
# )



import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# PATHS
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/"
    "seq_latency_IAT_table_final.csv"
)

LABEL_JSON_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/Labels.json"
)

OUT_DIR = "professional_average_accuracy_plots"
os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# DISPLAY LABELS
# ============================================================

POLICY_LABELS = {
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "sequential": "Sequential Cascade",
    "spec_b": "Cloud Speculation",
}

BASELINE_LABELS = {
    "power2_smallest_to_largest": "2 Smallest→Largest",
    "random_to_random": "Random→Random",
    "smallest_to_largest": "Smallest→Largest",
    "spec_b_smallest_to_largest": "Smallest→Largest",
    "spec_b_random_to_random": "Random→Random",
    "full_ensemble_edge": "3-Model Edge Ensemble",
    "full_ensemble_cloud": "Full Cloud Ensemble",
    "frugalml": "FrugalML",
}


# ============================================================
# DARKER PAPER-STYLE COLORS
# ============================================================

ERROR_COLORS = {
    0.090: "#1F4E79",   # dark blue
    0.100: "#B45F06",   # dark orange
    0.120: "#38761D",   # dark green
    0.140: "#654321",   # dark brown
}

BASELINE_COLORS = {
    "frugalml": "#8B0000",                     # dark red
    "random_to_random": "#8B0000",             # dark red
    "smallest_to_largest": "#4B0082",          # dark purple
    "power2_smallest_to_largest": "#4B0082",   # dark purple
    "spec_b_smallest_to_largest": "#4B0082",   # dark purple
    "spec_b_random_to_random": "#8B0000",      # dark red
    "full_ensemble_edge": "#404040",            # dark gray
    "full_ensemble_cloud": "#556B2F",           # dark olive
}


# ============================================================
# BASELINES TO INCLUDE
# ============================================================

SELECTED_BASELINES = [
    "frugalml",
    "smallest_to_largest",
    # "random_to_random",
]


# ============================================================
# HELPERS
# ============================================================

def convert_to_bool(value):
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    return pd.NA


def normalize_text(value):
    return (
        str(value)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


def load_label_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def prediction_matches_wnid(pred_text, wnid, label_json):
    """
    Compare the predicted text label against the possible ImageNet
    labels associated with the ground-truth WNID.
    """
    if pd.isna(pred_text) or pd.isna(wnid):
        return False

    wnid = str(wnid).strip()
    pred_norm = normalize_text(pred_text)

    if not pred_norm or wnid not in label_json:
        return False

    mapped_labels = label_json[wnid]

    if isinstance(mapped_labels, list):
        valid_labels = [
            normalize_text(label)
            for label in mapped_labels
        ]
    else:
        valid_labels = [
            normalize_text(label)
            for label in str(mapped_labels).split(",")
        ]

    return pred_norm in valid_labels


def sanitize_filename(value):
    return (
        str(value)
        .lower()
        .replace(" ", "_")
        .replace(".", "")
        .replace("-", "_")
        .replace("→", "_to_")
    )


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

label_json = load_label_json(LABEL_JSON_PATH)

print(
    f"Loaded {len(label_json)} class mappings "
    f"from {LABEL_JSON_PATH}"
)


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

# Normalize column names:
# Success -> success
# IAR -> iar
# RespTime -> resptime
df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)

required_columns = [
    "label",
    "selected_folder",
    "success",
    "static_config_enabled",
    "static_policy",
    "static_cost",
    "static_error",
    "baseline_mode",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in CSV: {missing_columns}"
    )


# ============================================================
# CLEAN DATA
# ============================================================

df["static_cost"] = pd.to_numeric(
    df["static_cost"],
    errors="coerce",
)

df["static_error"] = pd.to_numeric(
    df["static_error"],
    errors="coerce",
)

df["success_bool"] = df["success"].apply(
    convert_to_bool
)

df["is_static"] = df["static_config_enabled"].apply(
    convert_to_bool
)

df["baseline_mode"] = (
    df["baseline_mode"]
    .fillna("none")
    .astype(str)
    .str.strip()
    .str.lower()
)

df["static_policy"] = (
    df["static_policy"]
    .fillna("none")
    .astype(str)
    .str.strip()
    .str.lower()
)


# ============================================================
# FILTER SUCCESSFULLY COMPLETED REQUESTS
# ============================================================

# Success here means the request completed.
# True classification correctness is calculated using Labels.json.
df = df[df["success_bool"] == True].copy()

df = df.dropna(
    subset=[
        "label",
        "selected_folder",
        "is_static",
    ]
).copy()


# ============================================================
# CALCULATE TRUE CLASSIFICATION CORRECTNESS
# ============================================================

df["correct"] = df.apply(
    lambda row: prediction_matches_wnid(
        pred_text=row["label"],
        wnid=row["selected_folder"],
        label_json=label_json,
    ),
    axis=1,
)

df["correct_numeric"] = df["correct"].astype(float)


# ============================================================
# DEBUG INFORMATION
# ============================================================

print("\nCorrectness counts:")
print(df["correct"].value_counts(dropna=False))

print(
    "\nOverall accuracy: "
    f"{df['correct_numeric'].mean() * 100:.2f}%"
)

missing_wnids = sorted(
    set(df["selected_folder"].astype(str))
    - set(label_json.keys())
)

print(
    f"\nWNIDs missing from Labels.json: "
    f"{len(missing_wnids)}"
)

if missing_wnids:
    print("First missing WNIDs:")
    print(missing_wnids[:10])


# ============================================================
# SEPARATE DYSCI CONFIGURATIONS AND BASELINES
# ============================================================

static_df = df[
    df["is_static"] == True
].copy()

baseline_df = df[
    (df["is_static"] == False)
    & (~df["baseline_mode"].isin(
        ["none", "", "nan"]
    ))
].copy()

baseline_df = baseline_df[
    baseline_df["baseline_mode"].isin(
        SELECTED_BASELINES
    )
].copy()


# ============================================================
# CALCULATE AVERAGE ACCURACY
# ============================================================

static_summary = (
    static_df.groupby(
        [
            "static_policy",
            "static_cost",
            "static_error",
        ],
        as_index=False,
    )
    .agg(
        accuracy_fraction=(
            "correct_numeric",
            "mean",
        ),
        correct_predictions=(
            "correct_numeric",
            "sum",
        ),
        request_count=(
            "correct_numeric",
            "count",
        ),
    )
)

baseline_summary = (
    baseline_df.groupby(
        "baseline_mode",
        as_index=False,
    )
    .agg(
        accuracy_fraction=(
            "correct_numeric",
            "mean",
        ),
        correct_predictions=(
            "correct_numeric",
            "sum",
        ),
        request_count=(
            "correct_numeric",
            "count",
        ),
    )
)

static_summary["accuracy_percent"] = (
    static_summary["accuracy_fraction"] * 100.0
)

baseline_summary["accuracy_percent"] = (
    baseline_summary["accuracy_fraction"] * 100.0
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\nDYSCI accuracy summary:")
print(
    static_summary[
        [
            "static_policy",
            "static_cost",
            "static_error",
            "correct_predictions",
            "request_count",
            "accuracy_percent",
        ]
    ].to_string(index=False)
)

print("\nBaseline accuracy summary:")
print(
    baseline_summary[
        [
            "baseline_mode",
            "correct_predictions",
            "request_count",
            "accuracy_percent",
        ]
    ].to_string(index=False)
)


# ============================================================
# PROFESSIONAL PLOTTING STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 14.5,
    "ytick.labelsize": 14,
    "legend.fontsize": 10,
})


# ============================================================
# CREATE ONE FIGURE PER POLICY AND COST
# ============================================================

for policy, policy_data in static_summary.groupby(
    "static_policy"
):

    policy_display = POLICY_LABELS.get(
        policy,
        policy,
    )

    costs = sorted(
        policy_data["static_cost"]
        .dropna()
        .unique()
    )

    for cost in costs:

        cost_data = (
            policy_data[
                policy_data["static_cost"] == cost
            ]
            .sort_values("static_error")
        )

        labels = []
        accuracies = []
        colors = []
        counts = []

        # ----------------------------------------------------
        # Add DYSCI bars
        # ----------------------------------------------------

        for _, row in cost_data.iterrows():

            error_value = float(
                row["static_error"]
            )

            labels.append(
                f"DYSCI\nE={error_value:.3f}"
            )

            accuracies.append(
                float(row["accuracy_percent"])
            )

            counts.append(
                int(row["request_count"])
            )

            colors.append(
                ERROR_COLORS.get(
                    round(error_value, 3),
                    "#666666",
                )
            )

        # ----------------------------------------------------
        # Add baseline bars in selected order
        # ----------------------------------------------------

        for baseline_name in SELECTED_BASELINES:

            matching_rows = baseline_summary[
                baseline_summary["baseline_mode"]
                == baseline_name
            ]

            if matching_rows.empty:
                continue

            row = matching_rows.iloc[0]

            labels.append(
                BASELINE_LABELS.get(
                    baseline_name,
                    baseline_name,
                )
            )

            accuracies.append(
                float(row["accuracy_percent"])
            )

            counts.append(
                int(row["request_count"])
            )

            colors.append(
                BASELINE_COLORS.get(
                    baseline_name,
                    "#666666",
                )
            )

        if not accuracies:
            continue

        # ----------------------------------------------------
        # Draw chart
        # ----------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(9.2, 5.2)
        )

        x_positions = np.arange(
            len(labels)
        )

        bars = ax.bar(
            x_positions,
            accuracies,
            width=0.62,
            color=colors,
            edgecolor="#222222",
            linewidth=0.7,
        )

        # ----------------------------------------------------
        # Add values above bars
        # ----------------------------------------------------

        max_accuracy = max(accuracies)
        min_accuracy = min(accuracies)

        value_offset = max(
            0.20,
            (max_accuracy - min_accuracy) * 0.04,
        )

        for bar, accuracy in zip(
            bars,
            accuracies,
        ):
            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                bar.get_height() + value_offset,
                f"{accuracy:.1f}",
                ha="center",
                va="bottom",
                fontsize=14.5,
                fontweight="medium",
            )

        # ----------------------------------------------------
        # Axis labels
        # ----------------------------------------------------

        ax.set_xticks(
            x_positions
        )

        ax.set_xticklabels(
            labels
        )

        ax.set_ylabel(
            "Accuracy (%)"
        )

        # ax.set_title(
        #     f"Accuracy under "
        #     f"Cost Constraint C={cost:g}",
        #     pad=12,
        # )

        # ----------------------------------------------------
        # Professional y-axis range
        # ----------------------------------------------------

        lower_limit = max(
            0,
            np.floor(min_accuracy / 2.0) * 2.0 - 2.0,
        )

        upper_limit = min(
            100,
            np.ceil(max_accuracy) + 2.0,
        )

        # Allow enough room above a 100% bar if needed.
        if upper_limit <= max_accuracy + value_offset:
            upper_limit = min(
                101,
                max_accuracy + 1.5,
            )

        ax.set_ylim(
            lower_limit,
            upper_limit,
        )

        ax.yaxis.set_major_locator(
            MultipleLocator(2)
        )

        # ----------------------------------------------------
        # Clean paper-style axes
        # ----------------------------------------------------

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.9)
            spine.set_color("#222222")

        ax.grid(
            axis="y",
            linestyle="--",
            linewidth=0.65,
            alpha=0.28,
        )

        ax.set_axisbelow(True)

        ax.tick_params(
            axis="x",
            length=0,
            pad=8,
        )

        ax.tick_params(
            axis="y",
            width=0.8,
        )

        # Add slight horizontal margins
        ax.margins(
            x=0.05
        )

        plt.tight_layout()

        safe_policy = sanitize_filename(
            policy_display
        )

        out_base = os.path.join(
            OUT_DIR,
            (
                f"{safe_policy}_"
                f"cost_{cost:g}_"
                "professional_accuracy"
            ),
        )

        plt.savefig(
            f"{out_base}.png",
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
        )

        plt.savefig(
            f"{out_base}.pdf",
            bbox_inches="tight",
            facecolor="white",
        )

        print(
            f"\nSaved:\n"
            f"  {out_base}.png\n"
            f"  {out_base}.pdf"
        )

        plt.show()
        plt.close(fig)


print(
    f"\nAll professional figures saved in: "
    f"{OUT_DIR}"
)