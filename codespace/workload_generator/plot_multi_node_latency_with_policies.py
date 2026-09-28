# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# CSV_PATH = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/multi_node_latency.csv"
# )
#
# OUTPUT_DIR = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/multinode_plots"
# )
#
# os.makedirs(OUTPUT_DIR, exist_ok=True)
#
#
# # ============================================================
# # Deployment order
# # ============================================================
#
# DEPLOYMENT_ORDER = [
#     "1E+1C",
#     "2E+1C",
#     "1E+2C",
#     "2E+2C",
# ]
#
#
# # ============================================================
# # Read CSV
# # ============================================================
#
# df = pd.read_csv(CSV_PATH)
#
# print("CSV columns:")
# print(df.columns.tolist())
#
#
# # ============================================================
# # Basic cleaning
# # ============================================================
#
# required_columns = [
#     "deployment_config",
#     "effective_cascade_mode",
#     "IAR",
#     "RespTime",
#     "Success",
# ]
#
# missing_columns = [
#     col for col in required_columns
#     if col not in df.columns
# ]
#
# if missing_columns:
#     raise ValueError(
#         f"Missing required columns: {missing_columns}"
#     )
#
#
# # Only successful requests
# df = df[
#     df["Success"].astype(str).str.lower() == "true"
# ].copy()
#
#
# # Convert numeric columns
# df["IAR"] = pd.to_numeric(
#     df["IAR"],
#     errors="coerce",
# )
#
# df["RespTime"] = pd.to_numeric(
#     df["RespTime"],
#     errors="coerce",
# )
#
# df = df.dropna(
#     subset=[
#         "IAR",
#         "RespTime",
#         "deployment_config",
#         "effective_cascade_mode",
#     ]
# )
#
# # ============================================================
# # Normalize policy names
# # ============================================================
#
# def normalize_policy(value):
#     value = str(value).strip().lower()
#
#     if value == "sequential":
#         return "Sequential"
#
#     if value in (
#         "parallel_first_finish",
#         "po2",
#         "power2",
#     ):
#         return "Edge Speculation"
#
#     if value in (
#         "spec_b",
#         "cloud_speculation",
#         "cloud_spec",
#     ):
#         return "Cloud Speculation"
#
#     return value
#
#
# df["policy_name"] = df[
#     "effective_cascade_mode"
# ].apply(normalize_policy)
#
#
# # ============================================================
# # P95 aggregation
# # ============================================================
#
# summary = (
#     df.groupby(
#         [
#             "policy_name",
#             "IAR",
#             "deployment_config",
#         ]
#     )["RespTime"]
#     .mean()
#     .reset_index()
#     .rename(
#         columns={
#             "RespTime": "avg_latency_ms"
#         }
#     )
# )
#
#
# print("\nP95 summary:")
# print(summary)
#
#
# # ============================================================
# # Plot helper
# # ============================================================
#
# def plot_policy_scaling(
#     summary_df,
#     policy_name,
#     filename,
# ):
#     policy_df = summary_df[
#         summary_df["policy_name"] == policy_name
#     ].copy()
#
#     if policy_df.empty:
#         print(
#             f"No data found for policy: {policy_name}"
#         )
#         return
#
#     # Sort IATs.
#     # Larger IAT = lower workload.
#     iat_values = sorted(
#         policy_df["IAR"].unique(),
#         reverse=True,
#     )
#
#     fig, ax = plt.subplots(
#         figsize=(10, 6)
#     )
#
#     for deployment in DEPLOYMENT_ORDER:
#         deployment_df = policy_df[
#             policy_df["deployment_config"]
#             == deployment
#         ].copy()
#
#         if deployment_df.empty:
#             continue
#
#         deployment_df = (
#             deployment_df
#             .set_index("IAR")
#             .reindex(iat_values)
#             .reset_index()
#         )
#
#         ax.plot(
#             deployment_df["IAR"],
#             deployment_df["avg_latency_ms"],
#             marker="o",
#             linewidth=2,
#             label=deployment,
#         )
#
#     ax.set_title(
#         f"{policy_name}: Multi-Node Scaling"
#     )
#
#     ax.set_xlabel(
#         "Inter-Arrival Time (ms)"
#     )
#
#     ax.set_ylabel(
#         "P95 End-to-End Latency (ms)"
#     )
#
#     ax.grid(
#         True,
#         alpha=0.3,
#     )
#
#     ax.legend(
#         title="Deployment"
#     )
#
#     # High workload on the right
#     ax.invert_xaxis()
#
#     plt.tight_layout()
#
#     output_path = os.path.join(
#         OUTPUT_DIR,
#         filename,
#     )
#
#     plt.savefig(
#         output_path,
#         dpi=300,
#         bbox_inches="tight",
#     )
#
#     plt.show()
#
#     print(
#         f"Saved: {output_path}"
#     )
#
#
# # ============================================================
# # Generate three policy-wise graphs
# # ============================================================
#
# plot_policy_scaling(
#     summary_df=summary,
#     policy_name="Sequential",
#     filename="sequential_multinode_p95.png",
# )
#
# plot_policy_scaling(
#     summary_df=summary,
#     policy_name="Edge Speculation",
#     filename="edge_speculation_multinode_p95.png",
# )
#
# plot_policy_scaling(
#     summary_df=summary,
#     policy_name="Cloud Speculation",
#     filename="cloud_speculation_multinode_p95.png",
# )
#
#
# # ============================================================
# # Optional: High-load summary graph
# # ============================================================
#
# if not summary.empty:
#
#     highest_load_iar = summary["IAR"].min()
#
#     high_load_df = summary[
#         summary["IAR"] == highest_load_iar
#     ].copy()
#
#     pivot = high_load_df.pivot_table(
#         index="deployment_config",
#         columns="policy_name",
#         values="avg_latency_ms",
#         aggfunc="first",
#     )
#
#     pivot = pivot.reindex(
#         [
#             d for d in DEPLOYMENT_ORDER
#             if d in pivot.index
#         ]
#     )
#
#     ax = pivot.plot(
#         kind="bar",
#         figsize=(10, 6),
#     )
#
#     ax.set_title(
#         f"High-Load Multi-Node Comparison "
#         f"(IAT = {highest_load_iar:g} ms)"
#     )
#
#     ax.set_xlabel(
#         "Deployment Configuration"
#     )
#
#     ax.set_ylabel(
#         "Average End-to-End Latency (ms)"
#     )
#
#     ax.grid(
#         axis="y",
#         alpha=0.3,
#     )
#
#     plt.xticks(
#         rotation=0
#     )
#
#     plt.tight_layout()
#
#     output_path = os.path.join(
#         OUTPUT_DIR,
#         "high_load_policy_comparison.png",
#     )
#
#     plt.savefig(
#         output_path,
#         dpi=300,
#         bbox_inches="tight",
#     )
#
#     plt.show()
#
#     print(
#         f"Saved: {output_path}"
#     )







import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/multi_node_latency.csv"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/multinode_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# RespTime unit
# ============================================================

RESPTIME_IS_SECONDS = True


# ============================================================
# Axis configuration
# ============================================================

# X-axis
X_MIN = 0.8
X_MAX = 10.5
X_TICK_INTERVAL = 2

# Y-axis
Y_MIN = 0
Y_TICK_INTERVAL = 500

# Leave None for automatic upper limit
Y_MAX = None


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
# Deployment colors
# ============================================================

DEPLOYMENT_COLORS = {
    "1E+1C": "#4C78A8",
    "2E+1C": "#F58518",
    "1E+2C": "#54A24B",
    "2E+2C": "#B279A2",
}


# ============================================================
# Policy colors
# ============================================================

POLICY_COLORS = {
    "Sequential": "#4C78A8",
    "Edge Speculation": "#F58518",
    "Cloud Speculation": "#54A24B",
}


# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(
    CSV_PATH
)


# ============================================================
# Required columns
# ============================================================

required_columns = [
    "deployment_config",
    "effective_cascade_mode",
    "IAR",
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
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# Clean data
# ============================================================

df = df[
    df["Success"]
    .astype(str)
    .str.lower()
    == "true"
].copy()


df["IAR"] = pd.to_numeric(
    df["IAR"],
    errors="coerce",
)


df["RespTime"] = pd.to_numeric(
    df["RespTime"],
    errors="coerce",
)


df = df.dropna(
    subset=[
        "IAR",
        "RespTime",
        "deployment_config",
        "effective_cascade_mode",
    ]
)


df = df[
    df["IAR"] > 0
].copy()


df["deployment_config"] = (
    df["deployment_config"]
    .astype(str)
    .str.strip()
)


# ============================================================
# Convert IAR to Arrival Rate
#
# IAR is in milliseconds
# ArrivalRate = 1000 / IAR
# ============================================================

df["ArrivalRate"] = (
    1000.0 / df["IAR"]
)


# ============================================================
# Convert RespTime to milliseconds
# ============================================================

if RESPTIME_IS_SECONDS:

    df["e2e_latency_ms"] = (
        df["RespTime"] * 1000.0
    )

else:

    df["e2e_latency_ms"] = (
        df["RespTime"]
    )


# ============================================================
# Normalize policy names
# ============================================================

def normalize_policy(value):

    value = str(
        value
    ).strip().lower()

    if value == "sequential":
        return "Sequential"

    if value in (
        "parallel_first_finish",
        "po2",
        "power2",
        "edge_speculation",
        "edge_spec",
    ):
        return "Edge Speculation"

    if value in (
        "spec_b",
        "cloud_speculation",
        "cloud_spec",
    ):
        return "Cloud Speculation"

    return value


df["policy_name"] = (
    df["effective_cascade_mode"]
    .apply(normalize_policy)
)


# ============================================================
# Debug information
# ============================================================

print("\n======================================")
print("Deployment Configurations")
print("======================================")

print(
    df["deployment_config"].unique()
)


print("\n======================================")
print("Policies")
print("======================================")

print(
    df["policy_name"].unique()
)


print("\n======================================")
print("IAR / Arrival Rate")
print("======================================")

arrival_debug = (
    df[
        [
            "IAR",
            "ArrivalRate",
        ]
    ]
    .drop_duplicates()
    .sort_values("ArrivalRate")
)

print(
    arrival_debug.to_string(
        index=False
    )
)


print("\n======================================")
print("Request Counts")
print("======================================")

print(
    df.groupby(
        [
            "policy_name",
            "deployment_config",
            "ArrivalRate",
        ]
    ).size()
)


# ============================================================
# Mean + standard deviation summary
# ============================================================

summary = (
    df.groupby(
        [
            "policy_name",
            "IAR",
            "ArrivalRate",
            "deployment_config",
        ]
    )["e2e_latency_ms"]
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
print("Mean / STD Summary")
print("======================================")

print(
    summary.to_string(
        index=False
    )
)


# ============================================================
# Helper:
# automatic clean y-axis maximum
# ============================================================

def get_y_upper_limit(dataframe):

    if Y_MAX is not None:
        return Y_MAX

    upper_values = (
        dataframe["mean_latency_ms"]
        + dataframe["std_latency_ms"]
    )

    max_value = upper_values.max()

    y_upper = int(
        np.ceil(
            max_value
            / Y_TICK_INTERVAL
        )
        * Y_TICK_INTERVAL
    )

    # At least one interval
    y_upper = max(
        y_upper,
        Y_TICK_INTERVAL,
    )

    return y_upper


# ============================================================
# Policy-wise line graph
# ============================================================

def plot_policy_scaling(
    summary_df,
    policy_name,
    filename,
):

    policy_df = summary_df[
        summary_df["policy_name"]
        == policy_name
    ].copy()


    if policy_df.empty:

        print(
            f"[SKIP] No data for policy: "
            f"{policy_name}"
        )

        return


    fig, ax = plt.subplots(
        figsize=(11.5, 6.5)
    )


    plotted = False


    for deployment in DEPLOYMENT_ORDER:

        dep_df = policy_df[
            policy_df["deployment_config"]
            == deployment
        ].copy()


        if dep_df.empty:
            continue


        dep_df = dep_df.sort_values(
            "ArrivalRate"
        )


        x = dep_df[
            "ArrivalRate"
        ].to_numpy(
            dtype=float
        )


        mean = dep_df[
            "mean_latency_ms"
        ].to_numpy(
            dtype=float
        )


        std = dep_df[
            "std_latency_ms"
        ].to_numpy(
            dtype=float
        )


        valid = (
            np.isfinite(x)
            & np.isfinite(mean)
            & np.isfinite(std)
        )


        x = x[valid]
        mean = mean[valid]
        std = std[valid]


        if len(x) == 0:
            continue


        color = DEPLOYMENT_COLORS[
            deployment
        ]


        # ====================================================
        # Mean line
        # ====================================================

        ax.plot(
            x,
            mean,
            marker="o",
            markersize=7,
            linewidth=2.4,
            color=color,
            label=DEPLOYMENT_LABELS[
                deployment
            ],
            zorder=3,
        )


        # ====================================================
        # ±1 STD shaded region
        # ====================================================

        lower = np.maximum(
            mean - std,
            0,
        )

        upper = (
            mean + std
        )


        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.14,
            linewidth=0,
            zorder=1,
        )


        plotted = True


    if not plotted:

        plt.close(fig)

        print(
            f"[SKIP] Nothing to plot "
            f"for {policy_name}"
        )

        return


    # ========================================================
    # Title / labels
    # ========================================================

    ax.set_title(
        f"{policy_name}: Multi-Node Scaling",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )


    ax.set_xlabel(
        "Arrival Rate (req/s)",
        fontsize=13,
    )


    ax.set_ylabel(
        "Average Latency (ms)",
        fontsize=13,
    )


    # ========================================================
    # X-axis
    #
    # Clean style:
    #
    # 2   4   6   8   10
    # ========================================================

    ax.set_xlim(
        X_MIN,
        X_MAX,
    )

    ax.xaxis.set_major_locator(
        MultipleLocator(
            X_TICK_INTERVAL
        )
    )


    # ========================================================
    # Y-axis
    #
    # Starts at 0
    #
    # 0
    # 500
    # 1000
    # 1500
    # 2000
    # ...
    # ========================================================

    y_upper = get_y_upper_limit(
        policy_df
    )


    ax.set_ylim(
        Y_MIN,
        y_upper,
    )


    ax.yaxis.set_major_locator(
        MultipleLocator(
            Y_TICK_INTERVAL
        )
    )


    # ========================================================
    # Grid
    # ========================================================

    ax.grid(
        True,
        linestyle="-",
        linewidth=0.6,
        alpha=0.20,
    )


    ax.set_axisbelow(
        True
    )


    # ========================================================
    # Legend
    # ========================================================

    ax.legend(
        title="Deployment Configuration",
        frameon=True,
        fontsize=10,
        title_fontsize=10,
        loc="upper left",
    )


    # ========================================================
    # Cleaner look
    # ========================================================

    ax.tick_params(
        axis="both",
        labelsize=11,
    )


    # Keep full box like your reference image
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)


    # ========================================================
    # STD note
    # ========================================================

    ax.text(
        0.99,
        0.02,
        "Shaded region = ±1 standard deviation",
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=9,
        alpha=0.7,
    )


    plt.tight_layout()


    # ========================================================
    # Save
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        filename,
    )


    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )


    plt.show()


    print(
        f"[SAVED] {output_path}"
    )


# ============================================================
# Generate policy-wise graphs
# ============================================================

plot_policy_scaling(
    summary_df=summary,
    policy_name="Sequential",
    filename=(
        "sequential_multinode_"
        "arrival_rate_mean_std.png"
    ),
)


plot_policy_scaling(
    summary_df=summary,
    policy_name="Edge Speculation",
    filename=(
        "edge_speculation_multinode_"
        "arrival_rate_mean_std.png"
    ),
)


plot_policy_scaling(
    summary_df=summary,
    policy_name="Cloud Speculation",
    filename=(
        "cloud_speculation_multinode_"
        "arrival_rate_mean_std.png"
    ),
)


# ============================================================
# Combined all-policy bar graph
# ============================================================

if summary.empty:

    print(
        "\n[SKIP] Summary is empty."
    )

else:

    # ========================================================
    # Highest load
    # ========================================================

    # TARGET_ARRIVAL_RATE = (
    #     summary[
    #         "ArrivalRate"
    #     ].max()
    # )

    # ========================================================
    # Fixed workload for combined policy comparison
    # ========================================================

    TARGET_IAR = 500  # change this to 100, 200, 500, 1000, etc.

    TARGET_ARRIVAL_RATE = (
            1000.0 / TARGET_IAR
    )


    print("\n======================================")
    print("Combined Policy Comparison")
    print("======================================")

    print(
        f"Arrival Rate = "
        f"{TARGET_ARRIVAL_RATE:g} req/s"
    )


    fixed_df = summary[
        np.isclose(
            summary["ArrivalRate"],
            TARGET_ARRIVAL_RATE,
        )
    ].copy()


    # ========================================================
    # Mean pivot
    # ========================================================

    mean_pivot = fixed_df.pivot_table(
        index="deployment_config",
        columns="policy_name",
        values="mean_latency_ms",
        aggfunc="first",
    )


    # ========================================================
    # STD pivot
    # ========================================================

    std_pivot = fixed_df.pivot_table(
        index="deployment_config",
        columns="policy_name",
        values="std_latency_ms",
        aggfunc="first",
    )


    # ========================================================
    # Deployment order
    # ========================================================

    available_deployments = [
        deployment
        for deployment in DEPLOYMENT_ORDER
        if deployment in mean_pivot.index
    ]


    if not available_deployments:

        print(
            "[SKIP] No deployments available."
        )

    else:

        mean_pivot = mean_pivot.reindex(
            available_deployments
        )


        std_pivot = std_pivot.reindex(
            available_deployments
        )


        # ====================================================
        # Policy order
        # ====================================================

        POLICY_ORDER = [
            "Sequential",
            "Edge Speculation",
            "Cloud Speculation",
        ]


        available_policies = [
            policy
            for policy in POLICY_ORDER
            if policy in mean_pivot.columns
        ]


        if not available_policies:

            print(
                "[SKIP] No recognized policies."
            )

        else:

            mean_pivot = mean_pivot[
                available_policies
            ]


            std_pivot = (
                std_pivot
                .reindex(
                    columns=available_policies
                )
                .fillna(0.0)
            )


            # =================================================
            # Bar chart
            # =================================================

            fig, ax = plt.subplots(
                figsize=(12.5, 6.8)
            )


            num_deployments = len(
                available_deployments
            )

            num_policies = len(
                available_policies
            )


            x = np.arange(
                num_deployments
            )


            total_group_width = 0.75

            bar_width = (
                total_group_width
                / num_policies
            )


            offsets = (
                np.arange(
                    num_policies
                )
                - (
                    num_policies - 1
                ) / 2
            ) * bar_width


            # =================================================
            # Bars
            # =================================================

            for i, policy in enumerate(
                available_policies
            ):

                means = mean_pivot[
                    policy
                ].to_numpy(
                    dtype=float
                )


                stds = std_pivot[
                    policy
                ].to_numpy(
                    dtype=float
                )


                bars = ax.bar(
                    x + offsets[i],
                    means,
                    width=(
                        bar_width
                        * 0.88
                    ),
                    yerr=stds,
                    capsize=5,
                    color=POLICY_COLORS[
                        policy
                    ],
                    label=policy,
                    alpha=0.90,
                    edgecolor="white",
                    linewidth=0.8,
                    error_kw={
                        "elinewidth": 1.3,
                        "capthick": 1.3,
                    },
                )


                # =============================================
                # Value labels
                # =============================================

                for (
                    bar,
                    mean_value,
                ) in zip(
                    bars,
                    means,
                ):

                    if not np.isfinite(
                        mean_value
                    ):
                        continue


                    ax.text(
                        (
                            bar.get_x()
                            + bar.get_width()
                            / 2
                        ),
                        bar.get_height(),
                        f"{mean_value:.0f}",
                        ha="center",
                        va="bottom",
                        fontsize=9,
                    )


            # =================================================
            # X-axis
            # =================================================

            ax.set_xticks(
                x
            )


            ax.set_xticklabels(
                [
                    DEPLOYMENT_LABELS.get(
                        deployment,
                        deployment,
                    )
                    for deployment
                    in available_deployments
                ],
                fontsize=11,
            )


            # =================================================
            # Title / labels
            # =================================================

            ax.set_title(
                "Multi-Node Scaling Across Execution Policies\n"
                f"Arrival Rate = "
                f"{TARGET_ARRIVAL_RATE:g} req/s",
                fontsize=16,
                fontweight="bold",
                pad=14,
            )


            ax.set_xlabel(
                "Deployment Configuration",
                fontsize=13,
            )


            ax.set_ylabel(
                "Average Latency (ms)",
                fontsize=13,
            )


            # =================================================
            # Bar chart y-axis
            # =================================================

            combined_max = np.nanmax(
                (
                    mean_pivot
                    + std_pivot
                ).to_numpy()
            )


            if Y_MAX is not None:

                bar_y_upper = Y_MAX

            else:

                bar_y_upper = int(
                    np.ceil(
                        combined_max
                        / Y_TICK_INTERVAL
                    )
                    * Y_TICK_INTERVAL
                )


            bar_y_upper = max(
                bar_y_upper,
                Y_TICK_INTERVAL,
            )


            ax.set_ylim(
                0,
                bar_y_upper,
            )


            ax.yaxis.set_major_locator(
                MultipleLocator(
                    Y_TICK_INTERVAL
                )
            )


            # =================================================
            # Grid
            # =================================================

            ax.grid(
                axis="y",
                linestyle="-",
                linewidth=0.6,
                alpha=0.20,
            )


            ax.set_axisbelow(
                True
            )


            # =================================================
            # Legend
            # =================================================

            ax.legend(
                title="Execution Policy",
                fontsize=10,
                title_fontsize=10,
                ncol=len(
                    available_policies
                ),
                loc="upper left",
            )


            # =================================================
            # STD note
            # =================================================

            ax.text(
                0.99,
                0.98,
                "Error bars = ±1 standard deviation",
                transform=ax.transAxes,
                horizontalalignment="right",
                verticalalignment="top",
                fontsize=9,
                alpha=0.7,
            )


            plt.tight_layout()


            # =================================================
            # Save
            # =================================================

            output_path = os.path.join(
                OUTPUT_DIR,
                (
                    "all_policies_multinode_"
                    "arrival_rate_mean_std.png"
                ),
            )


            plt.savefig(
                output_path,
                dpi=300,
                bbox_inches="tight",
            )


            plt.show()


            print(
                f"[SAVED] {output_path}"
            )


print("\n======================================")
print("Finished")
print("======================================")