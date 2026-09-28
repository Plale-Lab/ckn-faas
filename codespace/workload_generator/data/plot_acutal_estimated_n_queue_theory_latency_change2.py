# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ============================================================
# # Configuration
# # ============================================================
#
# CSV_PATH = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/tem_delete16.csv"
# )
#
# OUT_DIR = "dysci_estimator_comparison"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # IMPORTANT:
# # Change these to the IAR values actually used in the new experiment.
# # Your example CSV row has IAR = 500.
# IAT_VALUES = [750, 200]
#
# IAT_LABELS = {
#     2000: "Low Load",
#     1000: "Low Load",
#     750: "Medium Load",
#     500: "Medium Load",
#     200: "High Load",
# }
#
# # Models for which the new CSV contains estimator columns
# MODELS = [
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
# # ------------------------------------------------------------
# # Which model do you want to evaluate?
# # ------------------------------------------------------------
# MODEL_TO_PLOT = "resnet34"
#
# # 1 = raw values
# # 3 or 5 = visual smoothing only
# ROLLING_WINDOW = 1
#
# AXIS_LABEL_FONTSIZE = 20
# TITLE_FONTSIZE = 18
# TICK_FONTSIZE = 17
# LEGEND_FONTSIZE = 16
#
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
#
# # ============================================================
# # Helpers
# # ============================================================
#
# def parse_model_list(value):
#     if pd.isna(value):
#         return []
#
#     if isinstance(value, list):
#         return value
#
#     try:
#         parsed = ast.literal_eval(str(value))
#         return parsed if isinstance(parsed, list) else []
#     except Exception:
#         return []
#
#
# def safe_numeric(series):
#     return pd.to_numeric(series, errors="coerce")
#
#
# def rolling(series):
#     return series.rolling(
#         window=ROLLING_WINDOW,
#         min_periods=1
#     ).mean()
#
#
# def save_fig(fig, base_path):
#     fig.savefig(
#         f"{base_path}.png",
#         dpi=300,
#         bbox_inches="tight"
#     )
#
#     fig.savefig(
#         f"{base_path}.pdf",
#         bbox_inches="tight"
#     )
#
#
# # ============================================================
# # Load CSV
# # ============================================================
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print(f"Loaded {len(df)} rows")
#
# # ------------------------------------------------------------
# # Convert basic columns
# # ------------------------------------------------------------
#
# for col in [
#     "IAR",
#     "RunTime",
#     "RespTime",
#     "e2e_time_ms",
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "actual_execution_latency_s",
# ]:
#     if col in df.columns:
#         df[col] = safe_numeric(df[col])
#
#
# # ============================================================
# # Check model-specific columns
# # ============================================================
#
# dysci_col = f"{MODEL_TO_PLOT}_dysci_est"
# iluvatar_col = f"{MODEL_TO_PLOT}_iluvatar_est"
# actual_col = f"{MODEL_TO_PLOT}_actual"
# source_col = f"{MODEL_TO_PLOT}_est_source"
#
# required = [
#     "IAR",
#     "RunTime",
#     dysci_col,
#     iluvatar_col,
#     actual_col,
# ]
#
# missing = [c for c in required if c not in df.columns]
#
# if missing:
#     raise ValueError(
#         f"Missing required columns for {MODEL_TO_PLOT}: {missing}"
#     )
#
#
# # Convert estimator columns to numeric
# df[dysci_col] = safe_numeric(df[dysci_col])
# df[iluvatar_col] = safe_numeric(df[iluvatar_col])
# df[actual_col] = safe_numeric(df[actual_col])
#
#
# # ============================================================
# # Replace logger sentinel values with NaN
# # ============================================================
# #
# # Your logger uses -1 when a model was not executed / unavailable.
# # These values must NOT be plotted.
# #
#
# for col in [
#     dysci_col,
#     iluvatar_col,
#     actual_col,
# ]:
#     df.loc[df[col] < 0, col] = np.nan
#
#
# # ============================================================
# # Convert seconds -> milliseconds
# # ============================================================
#
# df["dysci_ms"] = df[dysci_col] * 1000.0
# df["iluvatar_ms"] = df[iluvatar_col] * 1000.0
# df["actual_ms"] = df[actual_col] * 1000.0
#
#
# # ============================================================
# # Show available IAR values
# # ============================================================
#
# print("\nAvailable IAR values:")
# print(
#     sorted(
#         df["IAR"]
#         .dropna()
#         .round()
#         .astype(int)
#         .unique()
#     )
# )
#
#
# # ============================================================
# # Prepare data for each IAT
# # ============================================================
#
# plot_data = {}
#
# for iat in IAT_VALUES:
#
#     d = df[
#         df["IAR"].round().astype("Int64") == int(iat)
#     ].copy()
#
#     if d.empty:
#         print(f"\nNo rows found for IAT={iat} ms")
#         continue
#
#     # --------------------------------------------------------
#     # Keep rows where this model has an actual execution
#     # --------------------------------------------------------
#     #
#     # This is important for a fair comparison:
#     # compare both estimators against the same actual invocation.
#     #
#
#     d = d.dropna(
#         subset=[
#             "actual_ms",
#             "dysci_ms",
#         ]
#     ).copy()
#
#     if d.empty:
#         print(
#             f"No valid {MODEL_TO_PLOT} executions "
#             f"for IAT={iat}"
#         )
#         continue
#
#     # --------------------------------------------------------
#     # Sort chronologically
#     # --------------------------------------------------------
#
#     d = d.sort_values("RunTime").reset_index(drop=True)
#
#     # Start each panel from t = 0
#     d["plot_time_sec"] = (
#         d["RunTime"] - d["RunTime"].min()
#     )
#
#     # --------------------------------------------------------
#     # Smooth only for visualization
#     # --------------------------------------------------------
#
#     d["actual_plot_ms"] = rolling(d["actual_ms"])
#     d["dysci_plot_ms"] = rolling(d["dysci_ms"])
#     d["iluvatar_plot_ms"] = rolling(d["iluvatar_ms"])
#
#     plot_data[iat] = d
#
#
# # ============================================================
# # Calculate estimator errors
# # ============================================================
#
# print("\n" + "=" * 70)
# print(f"Estimator evaluation for model: {MODEL_TO_PLOT}")
# print("=" * 70)
#
# for iat, d in plot_data.items():
#
#     # --------------------------------------------------------
#     # DYSCI error
#     # --------------------------------------------------------
#
#     dysci_valid = d[
#         d["actual_ms"].notna()
#         & d["dysci_ms"].notna()
#     ]
#
#     dysci_error = (
#         dysci_valid["dysci_ms"]
#         - dysci_valid["actual_ms"]
#     )
#
#     dysci_mae = np.mean(np.abs(dysci_error))
#
#     dysci_rmse = np.sqrt(
#         np.mean(dysci_error ** 2)
#     )
#
#     # --------------------------------------------------------
#     # Iluvatar error
#     # --------------------------------------------------------
#
#     iluvatar_valid = d[
#         d["actual_ms"].notna()
#         & d["iluvatar_ms"].notna()
#     ]
#
#     if len(iluvatar_valid) > 0:
#
#         iluvatar_error = (
#             iluvatar_valid["iluvatar_ms"]
#             - iluvatar_valid["actual_ms"]
#         )
#
#         iluvatar_mae = np.mean(
#             np.abs(iluvatar_error)
#         )
#
#         iluvatar_rmse = np.sqrt(
#             np.mean(iluvatar_error ** 2)
#         )
#
#     else:
#         iluvatar_mae = np.nan
#         iluvatar_rmse = np.nan
#
#     print(f"\nIAT = {iat} ms")
#     print(f"Number of actual executions: {len(d)}")
#
#     print(
#         f"DYSCI     MAE  = {dysci_mae:.2f} ms"
#     )
#
#     print(
#         f"DYSCI     RMSE = {dysci_rmse:.2f} ms"
#     )
#
#     print(
#         f"Ilúvatar  MAE  = {iluvatar_mae:.2f} ms"
#     )
#
#     print(
#         f"Ilúvatar  RMSE = {iluvatar_rmse:.2f} ms"
#     )
#
#
# # ============================================================
# # Determine common Y-axis
# # ============================================================
#
# all_values = []
#
# for d in plot_data.values():
#
#     all_values.extend(
#         d["actual_plot_ms"].dropna().tolist()
#     )
#
#     all_values.extend(
#         d["dysci_plot_ms"].dropna().tolist()
#     )
#
#     all_values.extend(
#         d["iluvatar_plot_ms"].dropna().tolist()
#     )
#
#
# if not all_values:
#     raise ValueError(
#         "No valid data available for plotting."
#     )
#
#
# y_min = max(
#     0,
#     np.nanpercentile(all_values, 1) * 0.85
# )
#
# y_max = (
#     np.nanpercentile(all_values, 99) * 1.10
# )
#
#
# # ============================================================
# # Individual plots
# # ============================================================
#
# for iat, d in plot_data.items():
#
#     fig, ax = plt.subplots(
#         figsize=(9, 5.5)
#     )
#
#     # Actual
#     ax.plot(
#         d["plot_time_sec"],
#         d["actual_plot_ms"],
#         linewidth=2.8,
#         label="Actual latency",
#     )
#
#     # DYSCI
#     ax.plot(
#         d["plot_time_sec"],
#         d["dysci_plot_ms"],
#         linewidth=2.8,
#         linestyle="--",
#         label="DYSCI estimate",
#     )
#
#     # Iluvatar
#     ax.plot(
#         d["plot_time_sec"],
#         d["iluvatar_plot_ms"],
#         linewidth=2.8,
#         linestyle=":",
#         label="Ilúvatar estimate",
#     )
#
#     ax.set_xlabel(
#         "Time (s)",
#         fontsize=AXIS_LABEL_FONTSIZE,
#     )
#
#     ax.set_ylabel(
#         "Invocation Latency (ms)",
#         fontsize=AXIS_LABEL_FONTSIZE,
#     )
#
#     label = IAT_LABELS.get(
#         iat,
#         f"IAT={iat} ms"
#     )
#
#     ax.set_title(
#         f"{label} (IAT={iat} ms)",
#         fontsize=TITLE_FONTSIZE,
#         fontweight="bold",
#     )
#
#     ax.set_ylim(y_min, y_max)
#
#     ax.grid(
#         True,
#         alpha=0.28,
#     )
#
#     ax.tick_params(
#         axis="both",
#         labelsize=TICK_FONTSIZE,
#     )
#
#     ax.legend(
#         fontsize=LEGEND_FONTSIZE,
#         loc="best",
#     )
#
#     plt.tight_layout()
#
#     out_base = os.path.join(
#         OUT_DIR,
#         f"{MODEL_TO_PLOT}_iat_{iat}"
#     )
#
#     save_fig(fig, out_base)
#
#     plt.show()
#
#
# # ============================================================
# # Combined figure
# # ============================================================
#
# available_iats = [
#     iat
#     for iat in IAT_VALUES
#     if iat in plot_data
# ]
#
# if available_iats:
#
#     fig, axes = plt.subplots(
#         1,
#         len(available_iats),
#         figsize=(6 * len(available_iats), 5.8),
#         sharey=True,
#     )
#
#     if len(available_iats) == 1:
#         axes = [axes]
#
#     handles = None
#     labels = None
#
#     for ax, iat in zip(
#         axes,
#         available_iats,
#     ):
#
#         d = plot_data[iat]
#
#         line1, = ax.plot(
#             d["plot_time_sec"],
#             d["actual_plot_ms"],
#             linewidth=2.8,
#             label="Actual latency",
#         )
#
#         line2, = ax.plot(
#             d["plot_time_sec"],
#             d["dysci_plot_ms"],
#             linewidth=2.8,
#             linestyle="--",
#             label="DYSCI estimate",
#         )
#
#         line3, = ax.plot(
#             d["plot_time_sec"],
#             d["iluvatar_plot_ms"],
#             linewidth=2.8,
#             linestyle=":",
#             label="Ilúvatar estimate",
#         )
#
#         handles = [
#             line1,
#             line2,
#             line3,
#         ]
#
#         labels = [
#             "Actual latency",
#             "DYSCI estimate",
#             "Ilúvatar estimate",
#         ]
#
#         load_label = IAT_LABELS.get(
#             iat,
#             ""
#         )
#
#         ax.set_title(
#             f"{load_label}\n(IAT={iat} ms)",
#             fontsize=TITLE_FONTSIZE,
#             fontweight="bold",
#         )
#
#         ax.set_ylim(
#             y_min,
#             y_max,
#         )
#
#         ax.grid(
#             True,
#             alpha=0.28,
#         )
#
#         ax.tick_params(
#             axis="both",
#             labelsize=TICK_FONTSIZE,
#         )
#
#
#     axes[0].set_ylabel(
#         "Invocation Latency (ms)",
#         fontsize=AXIS_LABEL_FONTSIZE,
#     )
#
#     fig.supxlabel(
#         "Time (s)",
#         fontsize=AXIS_LABEL_FONTSIZE,
#         y=0.04,
#     )
#
#     fig.legend(
#         handles=handles,
#         labels=labels,
#         loc="upper center",
#         ncol=3,
#         fontsize=LEGEND_FONTSIZE,
#         frameon=True,
#         bbox_to_anchor=(0.5, 0.99),
#     )
#
#     plt.tight_layout(
#         rect=[0, 0.07, 1, 0.86]
#     )
#
#     combined_base = os.path.join(
#         OUT_DIR,
#         f"{MODEL_TO_PLOT}_estimator_comparison_combined"
#     )
#
#     save_fig(
#         fig,
#         combined_base,
#     )
#
#     plt.show()
#
#     print(
#         f"\nCombined figure saved to:\n"
#         f"{combined_base}.png\n"
#         f"{combined_base}.pdf"
#     )



import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/tem_delete21.csv"
)

OUT_DIR = "queue_theory_latency_by_iat"
os.makedirs(OUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# Select the IAT values used in your experiment
# ------------------------------------------------------------

IAT_VALUES = [750, 100]

IAT_LABELS = {
    2000: "Low Load",
    1000: "Low Load",
    750: "Medium Load",
    500: "Medium Load",
    # 200: "High Load",
    100: "High Load",
}


# ============================================================
# Model baseline service times
# ============================================================

MODEL_INFERENCE_LATENCY = {
    "mobilenet_v3_small": 39.91 / 1000,
    "mobilenet_v2": 63.46 / 1000,
    "shufflenet_v2_x1_0": 47.08 / 1000,
    "resnet18": 67.85 / 1000,
    "resnet34": 95.56 / 1000,
    "resnet50": 54.94 / 1000,
    "resnet101": 59.78 / 1000,
    "efficientnet_b3": 67.52 / 1000,
    "convnext_tiny": 39.44 / 1000,
    "vit_b_16": 39.59 / 1000,
}


# ============================================================
# Plot settings
# ============================================================

# Use 1 first to inspect the raw measurements.
# Change to 3 or 5 later if you want visual smoothing.
ROLLING_WINDOW = 5

MM1_MIN_GAP = 0.05
MM1_CLIP_MS = 20000

AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 17
LEGEND_FONTSIZE = 18

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Helper functions
# ============================================================

def parse_model_list(value):
    """
    Convert CSV strings such as:
        "['resnet34', 'convnext_tiny']"
    into Python lists.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

        return []

    except Exception:
        return []


def get_service_time_s(row):
    """
    Estimate the service time represented by this request.

    First use executed_models because these are the models that
    actually executed.

    If unavailable, fall back to selected_models.
    """

    executed = parse_model_list(
        row.get("executed_models", "")
    )

    times = [
        MODEL_INFERENCE_LATENCY[m]
        for m in executed
        if m in MODEL_INFERENCE_LATENCY
    ]

    if times:
        return sum(times)

    # --------------------------------------------------------
    # Fallback to selected models
    # --------------------------------------------------------

    selected = parse_model_list(
        row.get("selected_models", "")
    )

    times = [
        MODEL_INFERENCE_LATENCY[m]
        for m in selected
        if m in MODEL_INFERENCE_LATENCY
    ]

    if times:
        return sum(times)

    return np.nan


def compute_constant_mm1_latency_ms(d, iat_value):
    """
    M/M/1:

        lambda = arrival rate
        mu     = service rate

        T = 1 / (mu - lambda)

    Returns milliseconds.
    """

    service_time_s = d["service_time_s"].mean()

    if pd.isna(service_time_s) or service_time_s <= 0:
        return np.nan

    # Service rate
    mu = 1.0 / service_time_s

    # IAT is milliseconds
    # Example:
    # IAT = 200 ms -> 5 req/s
    lam = 1000.0 / iat_value

    # Unstable M/M/1
    if lam >= mu:
        return MM1_CLIP_MS

    latency_s = 1.0 / max(
        mu - lam,
        MM1_MIN_GAP
    )

    return latency_s * 1000.0


def save_fig(fig, base_path):

    fig.savefig(
        f"{base_path}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        f"{base_path}.pdf",
        bbox_inches="tight",
    )


# ============================================================
# Load new CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

df.columns = df.columns.str.strip()

print(f"\nLoaded {len(df)} rows.")


# ============================================================
# Convert required columns
# ============================================================

numeric_columns = [
    "IAR",
    "RunTime",
    "RespTime",
    "e2e_time_ms",
    "selected_est_latency_s",
    "raw_selected_est_latency_s",
    "actual_execution_latency_s",
]

for col in numeric_columns:

    if col in df.columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )


# ============================================================
# Check required columns
# ============================================================

required_columns = [
    "IAR",
    "RunTime",
    "RespTime",
    "selected_est_latency_s",
]

missing_columns = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing_columns:

    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Show available IAT values
# ============================================================

print("\nAvailable IAT values in CSV:")

print(
    sorted(
        df["IAR"]
        .dropna()
        .round()
        .astype(int)
        .unique()
    )
)


# ============================================================
# Build latency columns
# ============================================================

# ------------------------------------------------------------
# Actual request E2E latency
#
# RespTime is in seconds
# ------------------------------------------------------------

df["actual_latency_ms"] = (
    df["RespTime"] * 1000.0
)


# ------------------------------------------------------------
# REAL DYSCI estimate from the new estimator
#
# No synthetic generation.
# No target_ratio.
# No noise.
# No wave.
# ------------------------------------------------------------

df["dysci_estimated_latency_ms"] = (
    df["selected_est_latency_s"] * 1000.0
)


# ------------------------------------------------------------
# Service time needed for M/M/1
# ------------------------------------------------------------

df["service_time_s"] = df.apply(
    get_service_time_s,
    axis=1,
)


# ============================================================
# Remove invalid values
# ============================================================

df.loc[
    df["actual_latency_ms"] < 0,
    "actual_latency_ms"
] = np.nan

df.loc[
    df["dysci_estimated_latency_ms"] < 0,
    "dysci_estimated_latency_ms"
] = np.nan

df.loc[
    df["service_time_s"] <= 0,
    "service_time_s"
] = np.nan


# ============================================================
# Prepare each IAT
# ============================================================

plot_data = {}

for iat in IAT_VALUES:

    d = df[
        df["IAR"].round().astype("Int64")
        == int(iat)
    ].copy()

    if d.empty:

        print(
            f"\nNo data found for IAT={iat} ms"
        )

        continue


    # --------------------------------------------------------
    # Remove rows without required measurements
    # --------------------------------------------------------

    d = d.dropna(
        subset=[
            "RunTime",
            "actual_latency_ms",
            "dysci_estimated_latency_ms",
            "service_time_s",
        ]
    ).copy()


    if d.empty:

        print(
            f"No valid rows after filtering "
            f"for IAT={iat} ms"
        )

        continue


    # --------------------------------------------------------
    # Sort chronologically
    # --------------------------------------------------------

    d = d.sort_values(
        "RunTime"
    ).reset_index(drop=True)


    # --------------------------------------------------------
    # Start time from zero
    # --------------------------------------------------------

    d["plot_time_sec"] = (
        d["RunTime"]
        - d["RunTime"].min()
    )


    # ========================================================
    # Smooth actual latency
    # ========================================================

    d["actual_smooth_ms"] = (
        d["actual_latency_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )


    # ========================================================
    # Smooth REAL DYSCI estimate
    # ========================================================

    d["dysci_smooth_ms"] = (
        d["dysci_estimated_latency_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )


    # ========================================================
    # Calculate M/M/1
    # ========================================================

    mm1_value = compute_constant_mm1_latency_ms(
        d,
        iat,
    )

    d["mm1_smooth_ms"] = mm1_value


    plot_data[iat] = d


# ============================================================
# Print M/M/1 information
# ============================================================

print("\n" + "=" * 70)
print("M/M/1 latency values")
print("=" * 70)

for iat, d in plot_data.items():

    service_time_ms = (
        d["service_time_s"].mean()
        * 1000.0
    )

    arrival_rate = (
        1000.0 / iat
    )

    service_rate = (
        1.0 /
        d["service_time_s"].mean()
    )

    mm1_value = (
        d["mm1_smooth_ms"].iloc[0]
    )

    print(
        f"\nIAT = {iat} ms"
    )

    print(
        f"Arrival rate λ = "
        f"{arrival_rate:.2f} req/s"
    )

    print(
        f"Average service time = "
        f"{service_time_ms:.2f} ms"
    )

    print(
        f"Service rate μ = "
        f"{service_rate:.2f} req/s"
    )

    print(
        f"M/M/1 latency = "
        f"{mm1_value:.2f} ms"
    )


# ============================================================
# Print DYSCI estimation error
# ============================================================

print("\n" + "=" * 70)
print("DYSCI estimator accuracy")
print("=" * 70)

for iat, d in plot_data.items():

    error = (
        d["dysci_estimated_latency_ms"]
        - d["actual_latency_ms"]
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    print(
        f"\nIAT={iat} ms"
    )

    print(
        f"MAE  = {mae:.2f} ms"
    )

    print(
        f"RMSE = {rmse:.2f} ms"
    )


# ============================================================
# Common Y-axis
# ============================================================

all_values = []

for d in plot_data.values():

    all_values.extend(
        d["actual_smooth_ms"]
        .dropna()
        .tolist()
    )

    all_values.extend(
        d["dysci_smooth_ms"]
        .dropna()
        .tolist()
    )

    all_values.extend(
        d["mm1_smooth_ms"]
        .dropna()
        .tolist()
    )


if not all_values:

    raise ValueError(
        "No valid data available for plotting."
    )


y_min = max(
    0,
    np.nanpercentile(
        all_values,
        2,
    ) * 0.75,
)

y_max = (
    np.nanpercentile(
        all_values,
        98,
    )
    * 1.20
)


# ============================================================
# Individual figures
# ============================================================

for iat, d in plot_data.items():

    fig, ax = plt.subplots(
        figsize=(9, 5.5)
    )


    # --------------------------------------------------------
    # Actual
    # --------------------------------------------------------

    ax.plot(
        d["plot_time_sec"],
        d["actual_smooth_ms"],
        linewidth=2.8,
        label="Actual latency",
    )


    # --------------------------------------------------------
    # DYSCI
    # --------------------------------------------------------

    ax.plot(
        d["plot_time_sec"],
        d["dysci_smooth_ms"],
        linewidth=2.8,
        linestyle="--",
        label="DYSCI estimated latency",
    )


    # --------------------------------------------------------
    # M/M/1
    # --------------------------------------------------------

    ax.plot(
        d["plot_time_sec"],
        d["mm1_smooth_ms"],
        linewidth=3.0,
        linestyle=":",
        label="M/M/1 latency",
    )


    ax.set_xlabel(
        "Time (s)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    ax.set_ylabel(
        "Edge End-to-End Latency (ms)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    load_label = IAT_LABELS.get(
        iat,
        ""
    )


    ax.set_title(
        f"{load_label} (IAT={iat} ms)",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )


    ax.set_ylim(
        y_min,
        y_max,
    )


    ax.grid(
        True,
        alpha=0.28,
    )


    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
    )


    ax.legend(
        fontsize=LEGEND_FONTSIZE,
        loc="best",
    )


    plt.tight_layout()


    safe_label = (
        load_label
        .lower()
        .replace(" ", "_")
    )


    out_base = os.path.join(
        OUT_DIR,
        f"latency_comparison_{safe_label}_iat_{iat}",
    )


    save_fig(
        fig,
        out_base,
    )


    plt.show()


# ============================================================
# Combined figure
# ============================================================

available_iats = [
    iat
    for iat in IAT_VALUES
    if iat in plot_data
]


if available_iats:

    fig, axes = plt.subplots(
        1,
        len(available_iats),
        figsize=(
            6 * len(available_iats),
            5.8,
        ),
        sharey=True,
    )


    # One IAT case
    if len(available_iats) == 1:
        axes = [axes]


    handles = None
    labels = None


    for ax, iat in zip(
        axes,
        available_iats,
    ):

        d = plot_data[iat]


        # ----------------------------------------------------
        # Actual
        # ----------------------------------------------------

        line1, = ax.plot(
            d["plot_time_sec"],
            d["actual_smooth_ms"],
            linewidth=2.8,
            label="Actual latency",
        )


        # ----------------------------------------------------
        # DYSCI
        # ----------------------------------------------------

        line2, = ax.plot(
            d["plot_time_sec"],
            d["dysci_smooth_ms"],
            linewidth=2.8,
            linestyle="--",
            label="DYSCI estimated latency",
        )


        # ----------------------------------------------------
        # M/M/1
        # ----------------------------------------------------

        line3, = ax.plot(
            d["plot_time_sec"],
            d["mm1_smooth_ms"],
            linewidth=3.0,
            linestyle=":",
            label="M/M/1 latency",
        )


        handles = [
            line1,
            line2,
            line3,
        ]

        labels = [
            "Actual latency",
            "DYSCI estimated latency",
            "M/M/1 latency",
        ]


        load_label = IAT_LABELS.get(
            iat,
            ""
        )


        ax.set_title(
            f"{load_label}\n(IAT={iat} ms)",
            fontsize=TITLE_FONTSIZE,
            fontweight="bold",
        )


        ax.grid(
            True,
            alpha=0.28,
        )


        ax.tick_params(
            axis="both",
            labelsize=TICK_FONTSIZE,
        )


        ax.set_ylim(
            y_min,
            y_max,
        )


    # --------------------------------------------------------
    # Shared Y label
    # --------------------------------------------------------

    axes[0].set_ylabel(
        "Edge End-to-End Latency (ms)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )


    # --------------------------------------------------------
    # Shared X label
    # --------------------------------------------------------

    fig.supxlabel(
        "Time (s)",
        fontsize=AXIS_LABEL_FONTSIZE,
        y=0.04,
    )


    # --------------------------------------------------------
    # Shared legend
    # --------------------------------------------------------

    fig.legend(
        handles=handles,
        labels=labels,
        loc="upper center",
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        bbox_to_anchor=(0.5, 0.99),
    )


    plt.tight_layout(
        rect=[
            0,
            0.07,
            1,
            0.86,
        ]
    )


    combined_base = os.path.join(
        OUT_DIR,
        "latency_comparison_medium_high_iat_combined",
    )


    save_fig(
        fig,
        combined_base,
    )


    plt.show()


    print(
        f"\nSaved combined figure:"
    )

    print(
        f"{combined_base}.png"
    )

    print(
        f"{combined_base}.pdf"
    )



