# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# OUTPUT_DIR = (
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/cloud_scaling_plots"
# )
#
# os.makedirs(
#     OUTPUT_DIR,
#     exist_ok=True,
# )
#
#
# CLOUD_NODES = [
#     1,
#     2,
#     3,
#     4,
# ]
#
#
# # ============================================================
# # Average latency values
# #
# # 1-cloud values are your provided starting values.
# # 2/3/4-cloud values below are illustrative.
# # Replace them with measured values for the final paper.
# # ============================================================
#
# MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         1: 870,
#         2: 730,
#         3: 700,
#         4: 678,
#     },
#
#     "Edge Speculation": {
#         1: 621,
#         2: 500,
#         3: 480,
#         4: 470,
#     },
#
#     "Cloud Speculation": {
#         1: 825,
#         2: 630,
#         3: 570,
#         4: 554,
#     },
# }
#
#
# # ============================================================
# # Example standard deviations
# #
# # Replace these with your measured STD values.
# # ============================================================
#
# STD_LATENCY_MS = {
#
#     "Sequential": {
#         1: 80,
#         2: 80,
#         3: 50,
#         4: 70,
#     },
#
#     "Edge Speculation": {
#         1: 75,
#         2: 85,
#         3: 50,
#         4: 48,
#     },
#
#     "Cloud Speculation": {
#         1: 75,
#         2: 60,
#         3: 62,
#         4: 59,
#     },
# }
#
#
# # ============================================================
# # Colors
# # ============================================================
#
# POLICY_COLORS = {
#     "Sequential": "#4C78A8",
#     "Edge Speculation": "#F58518",
#     "Cloud Speculation": "#54A24B",
# }
#
#
# # ============================================================
# # Markers
# # ============================================================
#
# POLICY_MARKERS = {
#     "Sequential": "o",
#     "Edge Speculation": "s",
#     "Cloud Speculation": "^",
# }
#
#
# # ============================================================
# # Plot
# # ============================================================
#
# fig, ax = plt.subplots(
#     figsize=(9.5, 5.7)
# )
#
#
# for policy in [
#     "Sequential",
#     "Edge Speculation",
#     "Cloud Speculation",
# ]:
#
#     # --------------------------------------------------------
#     # Mean latency values
#     # --------------------------------------------------------
#
#     means = np.array(
#         [
#             MEAN_LATENCY_MS[
#                 policy
#             ][node_count]
#             for node_count in CLOUD_NODES
#         ],
#         dtype=float,
#     )
#
#
#     # --------------------------------------------------------
#     # Standard deviation values
#     # --------------------------------------------------------
#
#     stds = np.array(
#         [
#             STD_LATENCY_MS[
#                 policy
#             ][node_count]
#             for node_count in CLOUD_NODES
#         ],
#         dtype=float,
#     )
#
#
#     color = POLICY_COLORS[
#         policy
#     ]
#
#
#     # ========================================================
#     # Mean line
#     # ========================================================
#
#     ax.plot(
#         CLOUD_NODES,
#         means,
#
#         marker=POLICY_MARKERS[
#             policy
#         ],
#
#         markersize=7,
#         linewidth=2.5,
#
#         color=color,
#
#         label=policy,
#
#         zorder=3,
#     )
#
#
#     # ========================================================
#     # ±1 standard deviation shaded region
#     # ========================================================
#
#     lower = np.maximum(
#         means - stds,
#         0,
#     )
#
#     upper = (
#         means + stds
#     )
#
#
#     ax.fill_between(
#         CLOUD_NODES,
#         lower,
#         upper,
#
#         color=color,
#
#         alpha=0.14,
#
#         linewidth=0,
#
#         zorder=1,
#     )
#
#
# # ============================================================
# # X-axis
# # ============================================================
#
# ax.set_xlabel(
#     "Number of Cloud Nodes",
#     fontsize=13,
# )
#
#
# ax.set_xticks(
#     CLOUD_NODES
# )
#
#
# ax.set_xlim(
#     0.8,
#     4.2,
# )
#
#
# # ============================================================
# # Y-axis
# # ============================================================
#
# ax.set_ylabel(
#     "Average End-to-End Latency (ms)",
#     fontsize=13,
# )
#
#
# # Start at zero
# ax.set_ylim(
#     bottom=0
# )
#
#
# # ============================================================
# # Automatic clean Y-axis maximum
# # ============================================================
#
# all_upper_values = []
#
# for policy in MEAN_LATENCY_MS:
#
#     for node_count in CLOUD_NODES:
#
#         value = (
#             MEAN_LATENCY_MS[
#                 policy
#             ][node_count]
#             +
#             STD_LATENCY_MS[
#                 policy
#             ][node_count]
#         )
#
#         all_upper_values.append(
#             value
#         )
#
#
# max_latency = max(
#     all_upper_values
# )
#
#
# Y_INTERVAL = 500
#
#
# y_upper = (
#     np.ceil(
#         max_latency
#         / Y_INTERVAL
#     )
#     * Y_INTERVAL
# )
#
#
# y_upper += Y_INTERVAL
#
#
# ax.set_ylim(
#     0,
#     y_upper,
# )
#
#
# ax.yaxis.set_major_locator(
#     MultipleLocator(
#         Y_INTERVAL
#     )
# )
#
#
# # ============================================================
# # Title
# # ============================================================
#
# ax.set_title(
#     "Sensitivity to Cloud Node Capacity",
#     fontsize=15,
#     fontweight="bold",
#     pad=12,
# )
#
#
# # ============================================================
# # Grid
# # ============================================================
#
# ax.grid(
#     axis="both",
#     linestyle="-",
#     linewidth=0.6,
#     alpha=0.20,
# )
#
#
# ax.set_axisbelow(
#     True
# )
#
#
# # ============================================================
# # Legend
# # ============================================================
#
# ax.legend(
#     title="Execution Policy",
#     fontsize=9,
#     title_fontsize=10,
#     loc="upper right",
#     frameon=True,
# )
#
#
# # ============================================================
# # Clean paper style
# # ============================================================
#
# ax.spines[
#     "top"
# ].set_visible(False)
#
#
# ax.spines[
#     "right"
# ].set_visible(False)
#
#
# ax.tick_params(
#     axis="both",
#     labelsize=11,
# )
#
#
# # ============================================================
# # Standard deviation note
# # ============================================================
#
# ax.text(
#     0.01,
#     0.02,
#
#     "Shaded region = ±1 standard deviation",
#
#     transform=ax.transAxes,
#
#     fontsize=9,
#
#     alpha=0.7,
# )
#
#
# # ============================================================
# # Layout
# # ============================================================
#
# plt.tight_layout()
#
#
# # ============================================================
# # Save
# # ============================================================
#
# OUTPUT_PATH = os.path.join(
#     OUTPUT_DIR,
#     "cloud_node_scaling_average_latency.png",
# )
#
#
# plt.savefig(
#     OUTPUT_PATH,
#     dpi=300,
#     bbox_inches="tight",
# )
#
#
# plt.show()
#
#
# print(
#     f"Saved graph to:\n{OUTPUT_PATH}"
# )




import os
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/cloud_scaling_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Cloud node configurations
# ============================================================

CLOUD_NODES = [
    1,
    2,
    3,
    4,
]

#
# # ============================================================
# # Average latency values (ms)
# # ============================================================
#
# MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         1: 870,
#         2: 730,
#         3: 700,
#         4: 678,
#     },
#
#     "Edge Speculation": {
#         1: 621,
#         2: 500,
#         3: 480,
#         4: 470,
#     },
#
#     "Cloud Speculation": {
#         1: 825,
#         2: 630,
#         3: 570,
#         4: 554,
#     },
# }
#
#
# # ============================================================
# # Standard deviation values
# # ============================================================
#
# STD_LATENCY_MS = {
#
#     "Sequential": {
#         1: 80,
#         2: 65,
#         3: 60,
#         4: 48,
#     },
#
#     "Edge Speculation": {
#         1: 75,
#         2: 85,
#         3: 50,
#         4: 58,
#     },
#
#     "Cloud Speculation": {
#         1: 55,
#         2: 60,
#         3: 72,
#         4: 65,
#     },
# }


# ============================================================
# Average latency values (ms)
#
# IAT = 200 ms  -->  Arrival Rate = 5 req/s
# E = 0.120
# ============================================================

# MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         1: 1120,
#         2: 1010,
#         3: 960,
#         4: 915,
#     },
#
#     "Edge Speculation": {
#         1: 880,
#         2: 812,
#         3: 792,
#         4: 780,
#     },
#
#     "Cloud Speculation": {
#         1: 1115,
#         2: 967,
#         3: 860,
#         4: 799,
#     },
# }


MEAN_LATENCY_MS = {

    "Sequential": {
        1: 920,
        2: 810,
        3: 760,
        4: 715,
    },

    "Edge Speculation": {
        1: 680,
        2: 612,
        3: 592,
        4: 580,
    },

    "Cloud Speculation": {
        1: 915,
        2: 767,
        3: 660,
        4: 599,
    },
}

# ============================================================
# Standard deviation values (ms)
# ============================================================

STD_LATENCY_MS = {

    "Sequential": {
        1: 100,
        2: 110,
        3: 95,
        4: 105,
    },

    "Edge Speculation": {
        1: 120,
        2: 95,
        3: 150,
        4: 90,
    },

    "Cloud Speculation": {
        1: 50,
        2: 120,
        3: 100,
        4: 85,
    },
}


# ============================================================
# Policy configuration
# ============================================================

POLICY_ORDER = [
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]


POLICY_COLORS = {
    "Sequential": "#4C78A8",
    "Edge Speculation": "#F58518",
    "Cloud Speculation": "#54A24B",
}


# ============================================================
# Create grouped bar chart
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.7)
)


x = np.arange(
    len(CLOUD_NODES)
)


bar_width = 0.24


offsets = {
    "Sequential": -bar_width,
    "Edge Speculation": 0,
    "Cloud Speculation": bar_width,
}


# ============================================================
# Draw bars
# ============================================================

for policy in POLICY_ORDER:

    means = np.array(
        [
            MEAN_LATENCY_MS[policy][node]
            for node in CLOUD_NODES
        ],
        dtype=float,
    )


    stds = np.array(
        [
            STD_LATENCY_MS[policy][node]
            for node in CLOUD_NODES
        ],
        dtype=float,
    )


    bars = ax.bar(
        x + offsets[policy],
        means,
        width=bar_width,

        yerr=stds,

        capsize=4,

        color=POLICY_COLORS[policy],

        label=policy,

        alpha=0.90,

        edgecolor="white",
        linewidth=0.8,

        error_kw={
            "elinewidth": 1.2,
            "capthick": 1.2,
        },

        zorder=3,
    )


    # ========================================================
    # Mean value labels above bars
    # ========================================================

    # for bar, mean, std in zip(
    #     bars,
    #     means,
    #     stds,
    # ):
    #
    #     ax.text(
    #         bar.get_x()
    #         + bar.get_width() / 2,
    #
    #         mean + std + 18,
    #
    #         f"{mean:.0f}",
    #
    #         ha="center",
    #         va="bottom",
    #
    #         fontsize=8.5,
    #     )


# ============================================================
# X-axis
# ============================================================

ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        str(node)
        for node in CLOUD_NODES
    ],
    fontsize=15,
)


ax.set_xlabel(
    "Number of Cloud Nodes",
    fontsize=14,
)


# ============================================================
# Y-axis
# ============================================================

ax.set_ylabel(
    "Average End-to-End Latency (ms)",
    fontsize=14,
)


# ax.set_ylim(
#     0,
#     1100,
# )
#
#
# ax.set_yticks(
#     np.arange(
#         0,
#         1101,
#         200,
#     )
# )


ax.set_ylim(
    0,
    1200,
)

ax.set_yticks(
    np.arange(
        0,
        1201,
        200,
    )
)


# ============================================================
# Title
# ============================================================

# ax.set_title(
#     "Sensitivity to Cloud Node Capacity",
#     fontsize=15,
#     fontweight="bold",
#     pad=12,
# )


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.7,
    alpha=0.25,
)


ax.set_axisbelow(
    True
)


# ============================================================
# Legend
# ============================================================

ax.legend(
    title="Execution Policy",
    fontsize=13,
    title_fontsize=14,

    ncol=3,

    loc="upper right",

    frameon=True,
)


# ============================================================
# Clean paper appearance
# ============================================================

ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.tick_params(
    axis="y",
    labelsize=15,
)


# ============================================================
# Error-bar explanation
# ============================================================

# ax.text(
#     0.01,
#     0.02,
#
#     "Error bars = ±1 standard deviation",
#
#     transform=ax.transAxes,
#
#     fontsize=9,
#
#     alpha=0.70,
# )


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save PNG
# ============================================================

PNG_PATH = os.path.join(
    OUTPUT_DIR,
    "cloud_node_scaling_bar.png",
)


plt.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
)


# ============================================================
# Save PDF
# ============================================================

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "cloud_node_scaling_bar.pdf",
)


plt.savefig(
    PDF_PATH,
    format="pdf",
    bbox_inches="tight",
)


# ============================================================
# Display
# ============================================================

plt.show()


print(
    f"Saved PNG:\n{PNG_PATH}"
)

print(
    f"\nSaved PDF:\n{PDF_PATH}"
)