import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# File paths
# ============================================================

BASE_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data"
)

SEQ_CSV = os.path.join(
    BASE_DIR,
    "azure_multi_node_latency_seq.csv",
)

PO2_CSV = os.path.join(
    BASE_DIR,
    "azure_multi_node_latency_po2.csv",
)

SPEC_B_CSV = os.path.join(
    BASE_DIR,
    "azure_multi_node_latency_spec_b.csv",
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "multinode_plots",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# RespTime unit
#
# True:
#   RespTime is in seconds -> convert to milliseconds
#
# False:
#   RespTime is already milliseconds
# ============================================================

RESPTIME_IS_SECONDS = True


# ============================================================
# Deployment configuration
# ============================================================

DEPLOYMENT_ORDER = [
    "1E+1C",
    "2E+1C",
    "1E+2C",
    "2E+2C",
]

DEPLOYMENT_LABELS = {
    "1E+1C": "1 Edge + 1 Cloud",
    "2E+1C": "2 Edges + 1 Cloud",
    "1E+2C": "1 Edge + 2 Clouds",
    "2E+2C": "2 Edges + 2 Clouds",
}


# ============================================================
# Policy order
# ============================================================

POLICY_ORDER = [
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]


# ============================================================
# Colors
#
# Each deployment always uses the same color.
# ============================================================

DEPLOYMENT_COLORS = {
    "1E+1C": "#4C78A8",
    "2E+1C": "#F58518",
    "1E+2C": "#54A24B",
    "2E+2C": "#B279A2",
}


# ============================================================
# Load one policy file
# ============================================================

def load_policy_file(
    csv_path,
    policy_name,
):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File not found: {csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_columns = [
        "deployment_config",
        "RespTime",
        "Success",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{csv_path} is missing columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Successful requests only
    # --------------------------------------------------------

    df = df[
        df["Success"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "true"
    ].copy()

    # --------------------------------------------------------
    # RespTime numeric conversion
    # --------------------------------------------------------

    df["RespTime"] = pd.to_numeric(
        df["RespTime"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "RespTime",
            "deployment_config",
        ]
    )

    # --------------------------------------------------------
    # Clean deployment names
    # --------------------------------------------------------

    df["deployment_config"] = (
        df["deployment_config"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # Assign policy explicitly
    #
    # This is useful because each CSV corresponds to one policy.
    # --------------------------------------------------------

    df["policy_name"] = (
        policy_name
    )

    # --------------------------------------------------------
    # RespTime -> milliseconds
    # --------------------------------------------------------

    if RESPTIME_IS_SECONDS:

        df["latency_ms"] = (
            df["RespTime"]
            * 1000.0
        )

    else:

        df["latency_ms"] = (
            df["RespTime"]
        )

    return df


# ============================================================
# Load all three Azure workload files
# ============================================================

seq_df = load_policy_file(
    SEQ_CSV,
    "Sequential",
)

po2_df = load_policy_file(
    PO2_CSV,
    "Edge Speculation",
)

spec_b_df = load_policy_file(
    SPEC_B_CSV,
    "Cloud Speculation",
)


# ============================================================
# Combine all policy data
# ============================================================

df = pd.concat(
    [
        seq_df,
        po2_df,
        spec_b_df,
    ],
    ignore_index=True,
)


# ============================================================
# Debug information
# ============================================================

print("\n======================================")
print("Loaded Azure Multi-Node Data")
print("======================================")

print(
    f"Total successful requests: "
    f"{len(df)}"
)


print("\nPolicies:")
print(
    df["policy_name"].unique()
)


print("\nDeployment configurations:")
print(
    df["deployment_config"].unique()
)


print("\nRequest counts:")
print(
    df.groupby(
        [
            "policy_name",
            "deployment_config",
        ]
    )
    .size()
)


# ============================================================
# Mean + standard deviation
#
# IMPORTANT:
# This considers the FULL Azure workload in each file.
# It does NOT filter by IAR.
# ============================================================

summary = (
    df.groupby(
        [
            "policy_name",
            "deployment_config",
        ]
    )["latency_ms"]
    .agg(
        mean_latency_ms="mean",
        std_latency_ms="std",
        count="count",
    )
    .reset_index()
)


summary["std_latency_ms"] = (
    summary["std_latency_ms"]
    .fillna(0.0)
)


print("\n======================================")
print("Average Latency Summary")
print("======================================")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Mean pivot
#
# Rows    -> Policy
# Columns -> Deployment
# ============================================================

mean_pivot = summary.pivot_table(
    index="policy_name",
    columns="deployment_config",
    values="mean_latency_ms",
    aggfunc="first",
)


# ============================================================
# STD pivot
# ============================================================

std_pivot = summary.pivot_table(
    index="policy_name",
    columns="deployment_config",
    values="std_latency_ms",
    aggfunc="first",
)


# ============================================================
# Keep expected policy/deployment order
# ============================================================

available_policies = [
    policy
    for policy in POLICY_ORDER
    if policy in mean_pivot.index
]


available_deployments = [
    deployment
    for deployment in DEPLOYMENT_ORDER
    if deployment in mean_pivot.columns
]


if not available_policies:
    raise ValueError(
        "No expected policies found."
    )


if not available_deployments:
    raise ValueError(
        "No expected deployment configurations found."
    )


mean_pivot = mean_pivot.reindex(
    index=available_policies,
    columns=available_deployments,
)


std_pivot = (
    std_pivot
    .reindex(
        index=available_policies,
        columns=available_deployments,
    )
    .fillna(0.0)
)


# ============================================================
# Print final pivot
# ============================================================

print("\n======================================")
print("Mean Latency Pivot")
print("======================================")

print(
    mean_pivot
)


# ============================================================
# Create grouped bar chart
# ============================================================

fig, ax = plt.subplots(
    figsize=(12.5, 7.0)
)


num_policies = len(
    available_policies
)

num_deployments = len(
    available_deployments
)


# Policy positions:
#
# 0 -> Sequential
# 1 -> Edge Speculation
# 2 -> Cloud Speculation
x = np.arange(
    num_policies
)


# ============================================================
# Bar dimensions
# ============================================================

total_group_width = 0.82

bar_width = (
    total_group_width
    / num_deployments
)


offsets = (
    np.arange(num_deployments)
    - (num_deployments - 1) / 2
) * bar_width


# ============================================================
# Draw deployment bars
# ============================================================

for i, deployment in enumerate(
    available_deployments
):

    means = mean_pivot[
        deployment
    ].to_numpy(
        dtype=float
    )


    stds = std_pivot[
        deployment
    ].to_numpy(
        dtype=float
    )


    bars = ax.bar(
        x + offsets[i],

        means,

        width=(
            bar_width
            * 0.90
        ),

        # ------------------------------------------
        # Standard deviation error bars
        # ------------------------------------------

        yerr=stds,
        capsize=5,

        color=DEPLOYMENT_COLORS[
            deployment
        ],

        label=DEPLOYMENT_LABELS[
            deployment
        ],

        alpha=0.92,

        edgecolor="white",
        linewidth=0.8,

        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },
    )


    # ========================================================
    # Value above each bar
    # ========================================================

    for (
        bar,
        mean_value,
        std_value,
    ) in zip(
        bars,
        means,
        stds,
    ):

        if not np.isfinite(
            mean_value
        ):
            continue


        # Put value above error bar
        label_y = (
            mean_value
            + std_value
        )


        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            label_y,

            f"{mean_value:.0f}",

            ha="center",
            va="bottom",

            fontsize=8,
        )


# ============================================================
# X-axis
# ============================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(
    available_policies,
    fontsize=12,
)


ax.set_xlabel(
    "Execution Policy",
    fontsize=13,
)


# ============================================================
# Y-axis
# ============================================================

ax.set_ylabel(
    "Average End-to-End Latency (ms)",
    fontsize=13,
)


# Start from zero
ax.set_ylim(
    bottom=0
)


# ============================================================
# Calculate clean Y-axis scale
# ============================================================

all_upper_values = (
    mean_pivot
    + std_pivot
).to_numpy()


max_value = np.nanmax(
    all_upper_values
)


# ------------------------------------------------------------
# Choose clean interval
# ------------------------------------------------------------

if max_value <= 3000:

    Y_INTERVAL = 500

elif max_value <= 6000:

    Y_INTERVAL = 1000

elif max_value <= 12000:

    Y_INTERVAL = 2000

else:

    Y_INTERVAL = 5000


# ------------------------------------------------------------
# Round upper limit
# ------------------------------------------------------------

y_upper = (
    np.ceil(
        max_value
        / Y_INTERVAL
    )
    * Y_INTERVAL
)


# Give one extra interval for labels
y_upper = (
    y_upper
    + Y_INTERVAL
)


ax.set_ylim(
    0,
    y_upper,
)


ax.yaxis.set_major_locator(
    MultipleLocator(
        Y_INTERVAL
    )
)


# ============================================================
# Title
# ============================================================

ax.set_title(
    "Multi-Node Performance Under Azure Workload",
    fontsize=16,
    fontweight="bold",
    pad=14,
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="y",
    linestyle="-",
    linewidth=0.6,
    alpha=0.20,
)


ax.set_axisbelow(
    True
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    title="Deployment Configuration",

    fontsize=9,
    title_fontsize=10,

    ncol=2,

    loc="upper left",

    frameon=True,
)


# ============================================================
# Tick formatting
# ============================================================

ax.tick_params(
    axis="y",
    labelsize=11,
)


# ============================================================
# Keep clean border
# ============================================================

ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


# ============================================================
# Error-bar note
# ============================================================

ax.text(
    0.99,
    0.02,

    "Error bars = ±1 standard deviation",

    transform=ax.transAxes,

    horizontalalignment="right",
    verticalalignment="bottom",

    fontsize=9,
    alpha=0.7,
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save figure
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "azure_policy_deployment_average_latency.png",
)


plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


print("\n======================================")
print("Finished")
print("======================================")

print(
    f"Saved graph to:\n{OUTPUT_PATH}"
)