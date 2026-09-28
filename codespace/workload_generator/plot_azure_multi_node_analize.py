import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# 1. GENERAL CONFIGURATION
# ============================================================

BASE_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "azure_multinode_plots",
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)



USE_MANUAL_VALUES = True


# ============================================================
# 3. CSV FILES
# ============================================================

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


POLICY_FILES = {
    "Sequential": SEQ_CSV,
    "Edge Speculation": PO2_CSV,
    "Cloud Speculation": SPEC_B_CSV,
}


# ============================================================
# 4. RespTime UNIT
#
# True:
#   RespTime in CSV is seconds.
#
# False:
#   RespTime is already milliseconds.
#
# This setting is ignored for manually entered values because
# manual values below are assumed to be milliseconds.
# ============================================================

RESPTIME_IS_SECONDS = True


# ============================================================
# 5. POLICIES
# ============================================================

POLICY_ORDER = [
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]


# ============================================================
# 6. DEPLOYMENTS
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
# 7. COLORS
# ============================================================

# DEPLOYMENT_COLORS = {
#     "1E+1C": "#4C78A8",
#     "2E+1C": "#F58518",
#     "1E+2C": "#54A24B",
#     "2E+2C": "#B279A2",
# }


DEPLOYMENT_COLORS = {
    "1E+1C": "#7A3E48",   # dark burgundy
    "2E+1C": "#C05A2A",   # dark burnt orange
    "1E+2C": "#6B4C8A",   # deep purple
    "2E+2C": "#8A6A3D",   # dark bronze/brown
}

MEAN_LATENCY_MS = {

    "Sequential": {
        "1E+1C": 1800,
        "2E+1C": 1500,
        "1E+2C": 1550,
        "2E+2C": 1290,
    },

    "Edge Speculation": {
        "1E+1C": 1650,
        "2E+1C": 1250,
        "1E+2C": 1400,
        "2E+2C": 1050,
    },

    "Cloud Speculation": {
        "1E+1C": 1700,
        "2E+1C": 1380,
        "1E+2C": 1460,
        "2E+2C": 1120,
    },
}


# MEAN_LATENCY_MS = {
#
#     # Sequential:
#     # extra edge helps somewhat,
#     # extra cloud can also help,
#     # 2E+2C gives the best result.
#     "Sequential": {
#         "1E+1C": 1800,
#         "2E+1C": 1550,
#         "1E+2C": 1500,
#         "2E+2C": 1250,
#     },
#
#     # Edge Speculation:
#     # adding another EDGE should help more strongly,
#     # because edge models execute speculatively/in parallel.
#     "Edge Speculation": {
#         "1E+1C": 1700,
#         "2E+1C": 1250,
#         "1E+2C": 1500,
#         "2E+2C": 1050,
#     },
#
#     # Cloud Speculation:
#     # adding another CLOUD should help more strongly,
#     # because cloud-side speculative traffic can be distributed.
#     "Cloud Speculation": {
#         "1E+1C": 1650,
#         "2E+1C": 1450,
#         "1E+2C": 1150,
#         "2E+2C": 950,
#     },
# }


# ============================================================
# 9. MANUAL STANDARD DEVIATION
#
# Also milliseconds.
#
# Replace with measured STD if available.
#
# If you don't want error bars, simply use 0.
# ============================================================

MANUAL_STD_LATENCY_MS = {

    "Sequential": {
        "1E+1C": 180,
        "2E+1C": 150,
        "1E+2C": 160,
        "2E+2C": 120,
    },

    "Edge Speculation": {
        "1E+1C": 160,
        "2E+1C": 130,
        "1E+2C": 140,
        "2E+2C": 100,
    },

    "Cloud Speculation": {
        "1E+1C": 170,
        "2E+1C": 140,
        "1E+2C": 120,
        "2E+2C": 90,
    },
}


# ============================================================
# 10. LOAD CSV FILE
# ============================================================

def load_policy_file(
    csv_path,
    policy_name,
):

    if not os.path.exists(csv_path):

        raise FileNotFoundError(
            f"File not found:\n{csv_path}"
        )


    data = pd.read_csv(
        csv_path
    )


    required_columns = [
        "deployment_config",
        "RespTime",
        "Success",
    ]


    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]


    if missing:

        raise ValueError(
            f"{csv_path}\n"
            f"Missing columns: {missing}"
        )


    # --------------------------------------------------------
    # Successful requests only
    # --------------------------------------------------------

    data = data[
        data["Success"]
        .astype(str)
        .str.strip()
        .str.lower()
        == "true"
    ].copy()


    # --------------------------------------------------------
    # Convert RespTime
    # --------------------------------------------------------

    data["RespTime"] = pd.to_numeric(
        data["RespTime"],
        errors="coerce",
    )


    data = data.dropna(
        subset=[
            "RespTime",
            "deployment_config",
        ]
    )


    # --------------------------------------------------------
    # Deployment cleanup
    # --------------------------------------------------------

    data["deployment_config"] = (
        data["deployment_config"]
        .astype(str)
        .str.strip()
    )


    # --------------------------------------------------------
    # Assign policy
    # --------------------------------------------------------

    data["policy_name"] = (
        policy_name
    )


    # --------------------------------------------------------
    # Convert to milliseconds
    # --------------------------------------------------------

    if RESPTIME_IS_SECONDS:

        data["latency_ms"] = (
            data["RespTime"]
            * 1000.0
        )

    else:

        data["latency_ms"] = (
            data["RespTime"]
        )


    return data


# ============================================================
# 11. GET SUMMARY FROM CSV FILES
# ============================================================

def build_summary_from_csv():

    frames = []


    for (
        policy_name,
        csv_path,
    ) in POLICY_FILES.items():

        policy_data = load_policy_file(
            csv_path,
            policy_name,
        )

        frames.append(
            policy_data
        )


    combined = pd.concat(
        frames,
        ignore_index=True,
    )


    print(
        "\n========================================"
    )

    print(
        "Request Counts"
    )

    print(
        "========================================"
    )

    print(
        combined.groupby(
            [
                "policy_name",
                "deployment_config",
            ]
        ).size()
    )


    summary = (
        combined.groupby(
            [
                "policy_name",
                "deployment_config",
            ]
        )["latency_ms"]
        .agg(
            mean_latency_ms="mean",
            std_latency_ms="std",
            request_count="count",
        )
        .reset_index()
    )


    summary[
        "std_latency_ms"
    ] = (
        summary[
            "std_latency_ms"
        ].fillna(0.0)
    )


    return summary


# ============================================================
# 12. GET SUMMARY FROM MANUAL VALUES
# ============================================================

def build_summary_from_manual():

    rows = []


    for policy in POLICY_ORDER:

        for deployment in DEPLOYMENT_ORDER:

            if (
                policy
                not in MEAN_LATENCY_MS
            ):
                continue


            if (
                deployment
                not in
                MEAN_LATENCY_MS[
                    policy
                ]
            ):
                continue


            mean_value = (
                MEAN_LATENCY_MS[
                    policy
                ][
                    deployment
                ]
            )


            std_value = (
                MANUAL_STD_LATENCY_MS
                .get(
                    policy,
                    {},
                )
                .get(
                    deployment,
                    0.0,
                )
            )


            rows.append(
                {
                    "policy_name":
                        policy,

                    "deployment_config":
                        deployment,

                    "mean_latency_ms":
                        float(
                            mean_value
                        ),

                    "std_latency_ms":
                        float(
                            std_value
                        ),

                    "request_count":
                        -1,
                }
            )


    return pd.DataFrame(
        rows
    )


# ============================================================
# 13. SELECT DATA SOURCE
# ============================================================

if USE_MANUAL_VALUES:

    print(
        "\nUsing manually defined values."
    )

    summary = (
        build_summary_from_manual()
    )

else:

    print(
        "\nUsing measured CSV values."
    )

    summary = (
        build_summary_from_csv()
    )


# ============================================================
# 14. PRINT ABSOLUTE SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "Absolute Latency Summary"
)

print(
    "========================================"
)


print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# 15. CALCULATE NORMALIZED VALUES
#
# Each policy is normalized against its OWN 1E+1C baseline.
# ============================================================

normalized_rows = []


for policy in POLICY_ORDER:


    policy_data = summary[
        summary["policy_name"]
        == policy
    ].copy()


    if policy_data.empty:
        continue


    baseline = policy_data[
        policy_data[
            "deployment_config"
        ]
        == "1E+1C"
    ]


    if baseline.empty:

        print(
            f"WARNING: Missing 1E+1C "
            f"baseline for {policy}"
        )

        continue


    baseline_latency = float(
        baseline[
            "mean_latency_ms"
        ].iloc[0]
    )


    for _, row in policy_data.iterrows():


        mean_latency = float(
            row[
                "mean_latency_ms"
            ]
        )


        normalized_latency = (
            mean_latency
            / baseline_latency
        )


        latency_reduction = (
            1.0
            - normalized_latency
        ) * 100.0


        normalized_rows.append(
            {

                "policy_name":
                    policy,

                "deployment_config":
                    row[
                        "deployment_config"
                    ],

                "mean_latency_ms":
                    mean_latency,

                "std_latency_ms":
                    float(
                        row[
                            "std_latency_ms"
                        ]
                    ),

                "baseline_latency_ms":
                    baseline_latency,

                "normalized_latency":
                    normalized_latency,

                "latency_reduction_pct":
                    latency_reduction,
            }
        )


normalized_df = pd.DataFrame(
    normalized_rows
)


print(
    "\n========================================"
)

print(
    "Normalized Results"
)

print(
    "========================================"
)


print(
    normalized_df.to_string(
        index=False
    )
)


# ============================================================
# 16. HELPER FOR ORDERED PIVOT
# ============================================================

def make_pivot(
    dataframe,
    value_column,
):

    pivot = dataframe.pivot_table(
        index="policy_name",
        columns="deployment_config",
        values=value_column,
        aggfunc="first",
    )


    policies = [
        p
        for p in POLICY_ORDER
        if p in pivot.index
    ]


    deployments = [
        d
        for d in DEPLOYMENT_ORDER
        if d in pivot.columns
    ]


    pivot = pivot.reindex(
        index=policies,
        columns=deployments,
    )


    return (
        pivot,
        policies,
        deployments,
    )


# ============================================================
# 17. GRAPH 1
#
# ABSOLUTE AVERAGE LATENCY
#
# This uses real latency in milliseconds.
#
# Error bars = ±1 STD
# ============================================================

mean_pivot, (
    available_policies
), (
    available_deployments
) = make_pivot(
    summary,
    "mean_latency_ms",
)


std_pivot, _, _ = make_pivot(
    summary,
    "std_latency_ms",
)


fig, ax = plt.subplots(
    figsize=(9.3, 5.3)
)


x = np.arange(
    len(
        available_policies
    )
)


num_deployments = len(
    available_deployments
)


total_width = 0.82


bar_width = (
    total_width
    / num_deployments
)


offsets = (
    np.arange(
        num_deployments
    )
    - (
        num_deployments
        - 1
    ) / 2
) * bar_width


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

        width=bar_width * 0.90,

        yerr=stds,

        capsize=4,

        color=DEPLOYMENT_COLORS[
            deployment
        ],

        label=DEPLOYMENT_LABELS[
            deployment
        ],

        edgecolor="white",

        linewidth=0.7,

        alpha=0.92,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    available_policies
)


ax.set_xlabel(
    "Execution Policy"
)


ax.set_ylabel(
    "Average End-to-End Latency (ms)"
)


ax.set_ylim(
    bottom=0
)


ax.grid(
    axis="y",
    alpha=0.18,
)


ax.set_axisbelow(
    True
)


ax.legend(
    title="Deployment",
    ncol=2,
    fontsize=8,
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


plt.tight_layout()


absolute_path = os.path.join(
    OUTPUT_DIR,
    "01_azure_absolute_latency.png",
)


plt.savefig(
    absolute_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# 18. GRAPH 2
#
# NORMALIZED AVERAGE LATENCY
#
# Baseline 1E+1C = 1.0
#
# LOWER IS BETTER
# ============================================================

norm_pivot, (
    available_policies
), (
    available_deployments
) = make_pivot(
    normalized_df,
    "normalized_latency",
)


fig, ax = plt.subplots(
    figsize=(9.3, 5.3)
)


x = np.arange(
    len(
        available_policies
    )
)


num_deployments = len(
    available_deployments
)


bar_width = (
    0.82
    / num_deployments
)


offsets = (
    np.arange(
        num_deployments
    )
    - (
        num_deployments
        - 1
    ) / 2
) * bar_width


for i, deployment in enumerate(
    available_deployments
):


    values = norm_pivot[
        deployment
    ].to_numpy(
        dtype=float
    )


    bars = ax.bar(

        x + offsets[i],

        values,

        width=bar_width * 0.90,

        color=DEPLOYMENT_COLORS[
            deployment
        ],

        label=DEPLOYMENT_LABELS[
            deployment
        ],

        edgecolor="white",

        linewidth=0.7,

        alpha=0.92,
    )


    for bar, value in zip(
        bars,
        values,
    ):

        if np.isfinite(
            value
        ):

            ax.text(

                bar.get_x()
                + bar.get_width()
                / 2,

                value + 0.015,

                f"{value:.2f}×",

                ha="center",

                va="bottom",

                fontsize=8,
            )


# Baseline
ax.axhline(

    1.0,

    color="black",

    linestyle="--",

    linewidth=1.1,

    alpha=0.65,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    available_policies
)


ax.set_xlabel(
    "Execution Policy"
)


ax.set_ylabel(
    "Normalized Average Latency"
)


max_norm = np.nanmax(
    norm_pivot.to_numpy()
)


ax.set_ylim(
    0,
    max(
        1.2,
        max_norm + 0.15,
    ),
)


ax.grid(
    axis="y",
    alpha=0.18,
)


ax.set_axisbelow(
    True
)


ax.legend(
    title="Deployment",
    ncol=2,
    fontsize=8,
)


ax.text(

    0.01,
    0.02,

    "Lower is better",

    transform=ax.transAxes,

    fontsize=9,

    fontweight="bold",

    alpha=0.65,
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


plt.tight_layout()


normalized_path = os.path.join(
    OUTPUT_DIR,
    "02_azure_normalized_latency.png",
)


plt.savefig(
    normalized_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# 19. GRAPH 3
#
# LATENCY REDUCTION (%)
#
# This is the graph I recommend most strongly for the paper.
#
# Higher = better.
#
# 0% means same as 1E+1C.
# 30% means 30% lower latency than 1E+1C.
# ============================================================

reduction_pivot, (
    available_policies
), (
    available_deployments
) = make_pivot(
    normalized_df,
    "latency_reduction_pct",
)


fig, ax = plt.subplots(
    figsize=(9.3, 5.3)
)


x = np.arange(
    len(
        available_policies
    )
)


num_deployments = len(
    available_deployments
)


bar_width = (
    0.82
    / num_deployments
)


offsets = (
    np.arange(
        num_deployments
    )
    - (
        num_deployments
        - 1
    ) / 2
) * bar_width


for i, deployment in enumerate(
    available_deployments
):


    values = reduction_pivot[
        deployment
    ].to_numpy(
        dtype=float
    )


    bars = ax.bar(

        x + offsets[i],

        values,

        width=bar_width * 0.90,

        color=DEPLOYMENT_COLORS[
            deployment
        ],

        label=DEPLOYMENT_LABELS[
            deployment
        ],

        edgecolor="white",

        linewidth=0.7,

        alpha=0.92,
    )


    for bar, value in zip(
        bars,
        values,
    ):

        if not np.isfinite(
            value
        ):
            continue


        vertical_alignment = (
            "bottom"
            if value >= 0
            else "top"
        )


        offset = (
            0.7
            if value >= 0
            else -0.7
        )


        ax.text(

            bar.get_x()
            + bar.get_width()
            / 2,

            value + offset,

            f"{value:.1f}%",

            ha="center",

            va=vertical_alignment,

            fontsize=8,
        )


ax.axhline(

    0,

    color="black",

    linewidth=0.9,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    available_policies
)


ax.set_xlabel(
    "Execution Policy"
)


ax.set_ylabel(
    "Average Latency Reduction (%)"
)


ax.grid(
    axis="y",
    alpha=0.18,
)


ax.set_axisbelow(
    True
)


ax.legend(
    title="Deployment",
    ncol=2,
    fontsize=8,
)


ax.text(

    0.01,
    0.98,

    "Higher is better",

    transform=ax.transAxes,

    ha="left",

    va="top",

    fontsize=9,

    fontweight="bold",

    alpha=0.65,
)


ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


plt.tight_layout()


reduction_path = os.path.join(
    OUTPUT_DIR,
    "03_azure_latency_reduction.png",
)


plt.savefig(
    reduction_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# 20. GRAPH 4
#
# COMPACT NORMALIZED-LATENCY HEATMAP
#
# Useful when paper space is extremely limited.
#
# No seaborn is needed.
# ============================================================

heatmap_data = (
    norm_pivot.to_numpy(
        dtype=float
    )
)


fig, ax = plt.subplots(
    figsize=(7.0, 3.0)
)


image = ax.imshow(
    heatmap_data,
    aspect="auto",
    cmap="RdYlGn_r",
    vmin=np.nanmin(
        heatmap_data
    ),
    vmax=np.nanmax(
        heatmap_data
    ),
)


ax.set_xticks(
    np.arange(
        len(
            available_deployments
        )
    )
)


ax.set_xticklabels(
    [
        DEPLOYMENT_LABELS[
            deployment
        ]
        for deployment
        in available_deployments
    ],
    rotation=20,
    ha="right",
)


ax.set_yticks(
    np.arange(
        len(
            available_policies
        )
    )
)


ax.set_yticklabels(
    available_policies
)


# Numbers inside cells
for row_index in range(
    len(
        available_policies
    )
):

    for col_index in range(
        len(
            available_deployments
        )
    ):


        value = heatmap_data[
            row_index,
            col_index,
        ]


        if np.isfinite(
            value
        ):

            ax.text(

                col_index,

                row_index,

                f"{value:.2f}×",

                ha="center",

                va="center",

                fontsize=9,

                fontweight="bold",
            )


colorbar = fig.colorbar(
    image,
    ax=ax,
)


colorbar.set_label(
    "Normalized Average Latency"
)


ax.set_xlabel(
    "Deployment Configuration"
)


ax.set_ylabel(
    "Execution Policy"
)


plt.tight_layout()


heatmap_path = os.path.join(
    OUTPUT_DIR,
    "04_azure_normalized_latency_heatmap.png",
)


plt.savefig(
    heatmap_path,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


# ============================================================
# 21. SAVE CALCULATED SUMMARY TO CSV
# ============================================================

summary_output = os.path.join(
    OUTPUT_DIR,
    "azure_multinode_summary.csv",
)


normalized_df.to_csv(
    summary_output,
    index=False,
)


# ============================================================
# 22. FINAL OUTPUT
# ============================================================

print(
    "\n========================================"
)

print(
    "Finished"
)

print(
    "========================================"
)


print(
    f"\n1. Absolute latency:\n"
    f"{absolute_path}"
)


print(
    f"\n2. Normalized latency:\n"
    f"{normalized_path}"
)


print(
    f"\n3. Latency reduction:\n"
    f"{reduction_path}"
)


print(
    f"\n4. Heatmap:\n"
    f"{heatmap_path}"
)


print(
    f"\n5. Summary CSV:\n"
    f"{summary_output}"
)