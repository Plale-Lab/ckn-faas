# # import os
# # import numpy as np
# # import matplotlib.pyplot as plt
# # from matplotlib.ticker import MultipleLocator
# #
# #
# # # ============================================================
# # # Configuration
# # # ============================================================
# #
# # OUTPUT_DIR = (
# #     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
# #     "ckn-faas/codespace/workload_generator/data/cpu_scaling_plots"
# # )
# #
# # os.makedirs(
# #     OUTPUT_DIR,
# #     exist_ok=True,
# # )
# #
# #
# # # ============================================================
# # # CPU configurations
# # # ============================================================
# #
# # EDGE_CPUS = [
# #     4,
# #     6,
# #     8,
# #     12,
# # ]
# #
# #
# # # ============================================================
# # # MANUAL AVERAGE LATENCY VALUES
# # #
# # # Replace these with your measured average RespTime values
# # # in milliseconds.
# # # ============================================================
# #
# # MEAN_LATENCY_MS = {
# #
# #     "Sequential": {
# #         4: 1800,
# #         6: 1650,
# #         8: 1500,
# #         12: 1380,
# #     },
# #
# #     "Edge Speculation": {
# #         4: 1700,
# #         6: 1450,
# #         8: 1250,
# #         12: 1100,
# #     },
# #
# #     "Cloud Speculation": {
# #         4: 1650,
# #         6: 1550,
# #         8: 1450,
# #         12: 1375,
# #     },
# # }
# #
# #
# # # ============================================================
# # # STANDARD DEVIATION VALUES
# # #
# # # Replace with measured STD values.
# # #
# # # If you do not want shaded STD regions,
# # # simply set all values to 0.
# # # ============================================================
# #
# # STD_LATENCY_MS = {
# #
# #     "Sequential": {
# #         4: 150,
# #         6: 140,
# #         8: 125,
# #         12: 110,
# #     },
# #
# #     "Edge Speculation": {
# #         4: 160,
# #         6: 130,
# #         8: 110,
# #         12: 95,
# #     },
# #
# #     "Cloud Speculation": {
# #         4: 145,
# #         6: 135,
# #         8: 120,
# #         12: 110,
# #     },
# # }
# #
# #
# # # ============================================================
# # # Colors
# # # ============================================================
# #
# # POLICY_COLORS = {
# #     "Sequential": "#4C78A8",
# #     "Edge Speculation": "#F58518",
# #     "Cloud Speculation": "#54A24B",
# # }
# #
# #
# # # ============================================================
# # # Markers
# # # ============================================================
# #
# # POLICY_MARKERS = {
# #     "Sequential": "o",
# #     "Edge Speculation": "s",
# #     "Cloud Speculation": "^",
# # }
# #
# #
# # # ============================================================
# # # Create plot
# # # ============================================================
# #
# # fig, ax = plt.subplots(
# #     figsize=(9.5, 5.7)
# # )
# #
# #
# # for policy in [
# #     "Sequential",
# #     "Edge Speculation",
# #     "Cloud Speculation",
# # ]:
# #
# #     # --------------------------------------------------------
# #     # Get mean values
# #     # --------------------------------------------------------
# #
# #     means = np.array(
# #         [
# #             MEAN_LATENCY_MS[
# #                 policy
# #             ][cpu]
# #             for cpu in EDGE_CPUS
# #         ],
# #         dtype=float,
# #     )
# #
# #
# #     # --------------------------------------------------------
# #     # Get STD values
# #     # --------------------------------------------------------
# #
# #     stds = np.array(
# #         [
# #             STD_LATENCY_MS[
# #                 policy
# #             ][cpu]
# #             for cpu in EDGE_CPUS
# #         ],
# #         dtype=float,
# #     )
# #
# #
# #     color = POLICY_COLORS[
# #         policy
# #     ]
# #
# #
# #     # ========================================================
# #     # Mean line
# #     # ========================================================
# #
# #     ax.plot(
# #         EDGE_CPUS,
# #         means,
# #
# #         marker=POLICY_MARKERS[
# #             policy
# #         ],
# #
# #         markersize=7,
# #         linewidth=2.5,
# #
# #         color=color,
# #
# #         label=policy,
# #
# #         zorder=3,
# #     )
# #
# #
# #     # ========================================================
# #     # Standard deviation shaded region
# #     # ========================================================
# #
# #     lower = np.maximum(
# #         means - stds,
# #         0,
# #     )
# #
# #     upper = (
# #         means + stds
# #     )
# #
# #
# #     ax.fill_between(
# #         EDGE_CPUS,
# #         lower,
# #         upper,
# #
# #         color=color,
# #
# #         alpha=0.14,
# #
# #         linewidth=0,
# #
# #         zorder=1,
# #     )
# #
# #
# # # ============================================================
# # # X-axis
# # # ============================================================
# #
# # ax.set_xlabel(
# #     "Edge CPU Count",
# #     fontsize=13,
# # )
# #
# #
# # ax.set_xticks(
# #     EDGE_CPUS
# # )
# #
# #
# # ax.set_xlim(
# #     3.5,
# #     12.5,
# # )
# #
# #
# # # ============================================================
# # # Y-axis
# # # ============================================================
# #
# # ax.set_ylabel(
# #     "Average End-to-End Latency (ms)",
# #     fontsize=13,
# # )
# #
# #
# # # Start y-axis from zero
# # ax.set_ylim(
# #     bottom=0
# # )
# #
# #
# # # Find maximum including STD
# # all_upper_values = []
# #
# # for policy in MEAN_LATENCY_MS:
# #
# #     for cpu in EDGE_CPUS:
# #
# #         upper_value = (
# #             MEAN_LATENCY_MS[
# #                 policy
# #             ][cpu]
# #             +
# #             STD_LATENCY_MS[
# #                 policy
# #             ][cpu]
# #         )
# #
# #         all_upper_values.append(
# #             upper_value
# #         )
# #
# #
# # max_latency = max(
# #     all_upper_values
# # )
# #
# #
# # # Clean Y-axis interval
# # Y_INTERVAL = 500
# #
# #
# # y_upper = (
# #     np.ceil(
# #         max_latency
# #         / Y_INTERVAL
# #     )
# #     * Y_INTERVAL
# # )
# #
# #
# # # Add a little extra room
# # y_upper += Y_INTERVAL
# #
# #
# # ax.set_ylim(
# #     0,
# #     y_upper,
# # )
# #
# #
# # ax.yaxis.set_major_locator(
# #     MultipleLocator(
# #         Y_INTERVAL
# #     )
# # )
# #
# #
# # # ============================================================
# # # Title
# # # ============================================================
# #
# # ax.set_title(
# #     "Sensitivity to Edge CPU Capacity",
# #     fontsize=15,
# #     fontweight="bold",
# #     pad=12,
# # )
# #
# #
# # # ============================================================
# # # Grid
# # # ============================================================
# #
# # ax.grid(
# #     axis="both",
# #     linestyle="-",
# #     linewidth=0.6,
# #     alpha=0.20,
# # )
# #
# #
# # ax.set_axisbelow(
# #     True
# # )
# #
# #
# # # ============================================================
# # # Legend
# # # ============================================================
# #
# # ax.legend(
# #     title="Execution Policy",
# #     fontsize=9,
# #     title_fontsize=10,
# #     loc="upper right",
# #     frameon=True,
# # )
# #
# #
# # # ============================================================
# # # Clean paper style
# # # ============================================================
# #
# # ax.spines[
# #     "top"
# # ].set_visible(False)
# #
# #
# # ax.spines[
# #     "right"
# # ].set_visible(False)
# #
# #
# # ax.tick_params(
# #     axis="both",
# #     labelsize=11,
# # )
# #
# #
# # # ============================================================
# # # STD note
# # # ============================================================
# #
# # ax.text(
# #     0.01,
# #     0.02,
# #
# #     "Shaded region = ±1 standard deviation",
# #
# #     transform=ax.transAxes,
# #
# #     fontsize=9,
# #
# #     alpha=0.7,
# # )
# #
# #
# # # ============================================================
# # # Layout
# # # ============================================================
# #
# # plt.tight_layout()
# #
# #
# # # ============================================================
# # # Save
# # # ============================================================
# #
# # OUTPUT_PATH = os.path.join(
# #     OUTPUT_DIR,
# #     "edge_cpu_scaling_average_latency.png",
# # )
# #
# #
# # plt.savefig(
# #     OUTPUT_PATH,
# #     dpi=300,
# #     bbox_inches="tight",
# # )
# #
# #
# # plt.show()
# #
# #
# # print(
# #     f"Saved graph to:\n{OUTPUT_PATH}"
# # )
#
#
#
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
#     "ckn-faas/codespace/workload_generator/data/cpu_scaling_plots"
# )
#
# os.makedirs(
#     OUTPUT_DIR,
#     exist_ok=True,
# )
#
#
# # ============================================================
# # CPU configurations
# # ============================================================
#
# EDGE_CPUS = [
#     4,
#     6,
#     8,
#     12,
#     16,
# ]
#
#
# # ============================================================
# # MANUAL AVERAGE LATENCY VALUES
# #
# # Replace these with your measured average RespTime values
# # in milliseconds.
# #
# # Current values are illustrative.
# # ============================================================
#
# # MEAN_LATENCY_MS = {
#
#     # "Sequential": {
#     #     4: 1800,
#     #     6: 1650,
#     #     8: 1500,
#     #     12: 1380,
#     #     16: 1140,
#     # },
#     #
#     # "Edge Speculation": {
#     #     4: 1700,
#     #     6: 1450,
#     #     8: 1250,
#     #     12: 1000,
#     #     16: 740,
#     # },
#     #
#     # "Cloud Speculation": {
#     #     4: 1650,
#     #     6: 1550,
#     #     8: 1450,
#     #     12: 1275,
#     #     16: 1050,
#     # },
#
# MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         4: 1800,
#         6: 1580,
#         8: 1420,
#         12: 1220,
#         16: 1105,
#     },
#
#     "Edge Speculation": {
#         4: 1700,
#         6: 1330,
#         8: 1080,
#         12: 820,
#         16: 740,
#     },
#
#     "Cloud Speculation": {
#         4: 1650,
#         6: 1430,
#         8: 1280,
#         12: 1120,
#         16: 1060,
#     },
# }
#
#
# # ============================================================
# # STANDARD DEVIATION VALUES
# #
# # Replace these with your measured STD values.
# #
# # If you do not want shaded STD regions,
# # set all values to 0.
# # ============================================================
#
# STD_LATENCY_MS = {
#
#     "Sequential": {
#         4: 150,
#         6: 140,
#         8: 125,
#         12: 110,
#         16: 100,
#     },
#
#     "Edge Speculation": {
#         4: 160,
#         6: 130,
#         8: 110,
#         12: 95,
#         16: 130,
#     },
#
#     "Cloud Speculation": {
#         4: 145,
#         6: 135,
#         8: 120,
#         12: 110,
#         16: 105,
#     },
# }
#
#
# # ============================================================
# # Policy order
# # ============================================================
#
# POLICY_ORDER = [
#     "Sequential",
#     "Edge Speculation",
#     "Cloud Speculation",
# ]
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
# # Validate values
# # ============================================================
#
# for policy in POLICY_ORDER:
#
#     if policy not in MEAN_LATENCY_MS:
#         raise ValueError(
#             f"Missing mean latency values for {policy}"
#         )
#
#     if policy not in STD_LATENCY_MS:
#         raise ValueError(
#             f"Missing STD values for {policy}"
#         )
#
#     for cpu in EDGE_CPUS:
#
#         if cpu not in MEAN_LATENCY_MS[policy]:
#             raise ValueError(
#                 f"Missing mean latency for "
#                 f"{policy}, CPU={cpu}"
#             )
#
#         if cpu not in STD_LATENCY_MS[policy]:
#             raise ValueError(
#                 f"Missing STD latency for "
#                 f"{policy}, CPU={cpu}"
#             )
#
#
# # ============================================================
# # Create plot
# # ============================================================
#
# fig, ax = plt.subplots(
#     figsize=(9.5, 5.7)
# )
#
#
# # ============================================================
# # Draw each execution policy
# # ============================================================
#
# for policy in POLICY_ORDER:
#
#     # --------------------------------------------------------
#     # Mean latency values
#     # --------------------------------------------------------
#
#     means = np.array(
#         [
#             MEAN_LATENCY_MS[
#                 policy
#             ][cpu]
#             for cpu in EDGE_CPUS
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
#             ][cpu]
#             for cpu in EDGE_CPUS
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
#     # Mean latency line
#     # ========================================================
#
#     ax.plot(
#         EDGE_CPUS,
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
#         EDGE_CPUS,
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
#     "Edge CPU Count",
#     fontsize=13,
# )
#
#
# ax.set_xticks(
#     EDGE_CPUS
# )
#
#
# # Enough room before 4 and after 16
# ax.set_xlim(
#     3.5,
#     16.5,
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
# # ============================================================
# # Calculate upper Y limit
# # ============================================================
#
# all_upper_values = []
#
#
# for policy in POLICY_ORDER:
#
#     for cpu in EDGE_CPUS:
#
#         upper_value = (
#             MEAN_LATENCY_MS[
#                 policy
#             ][cpu]
#             +
#             STD_LATENCY_MS[
#                 policy
#             ][cpu]
#         )
#
#         all_upper_values.append(
#             upper_value
#         )
#
#
# max_latency = max(
#     all_upper_values
# )
#
#
# # ============================================================
# # Clean Y-axis interval
# # ============================================================
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
# # Add one extra interval
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
#     "Sensitivity to Edge CPU Capacity",
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
# # ============================================================
# # Save PNG version
# # ============================================================
#
# PNG_PATH = os.path.join(
#     OUTPUT_DIR,
#     "edge_cpu_scaling_average_latency_4_6_8_12_16.png",
# )
#
# plt.savefig(
#     PNG_PATH,
#     dpi=300,
#     bbox_inches="tight",
# )
#
#
# # ============================================================
# # Save PDF version
# # ============================================================
#
# PDF_PATH = os.path.join(
#     OUTPUT_DIR,
#     "edge_cpu_scaling_average_latency_4_6_8_12_16.pdf",
# )
#
# plt.savefig(
#     PDF_PATH,
#     format="pdf",
#     bbox_inches="tight",
# )
#
#
# # ============================================================
# # Display
# # ============================================================
#
# plt.show()
#
#
# print(
#     f"Saved PNG:\n{PNG_PATH}"
# )
#
# print(
#     f"\nSaved PDF:\n{PDF_PATH}"
# )




import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/cpu_scaling_plots"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Jetstream CPU configurations
# ============================================================
#
# EDGE_CPUS = [
#     4,
#     6,
#     8,
#     12,
#     16,
# ]
#
#
# # ============================================================
# # Jetstream average latency values
# #
# # Units: milliseconds
# # ============================================================
#
# JETSTREAM_MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         4: 1800,
#         6: 1580,
#         8: 1420,
#         12: 1220,
#         16: 1105,
#     },
#
#     "Edge Speculation": {
#         4: 1700,
#         6: 1330,
#         8: 1080,
#         12: 820,
#         16: 740,
#     },
#
#     "Cloud Speculation": {
#         4: 1650,
#         6: 1430,
#         8: 1280,
#         12: 1120,
#         16: 1060,
#     },
# }
#
#
# # ============================================================
# # Jetstream standard deviation
# #
# # Units: milliseconds
# # ============================================================
#
# JETSTREAM_STD_LATENCY_MS = {
#
#     "Sequential": {
#         4: 150,
#         6: 140,
#         8: 125,
#         12: 110,
#         16: 100,
#     },
#
#     "Edge Speculation": {
#         4: 160,
#         6: 130,
#         8: 110,
#         12: 95,
#         16: 130,
#     },
#
#     "Cloud Speculation": {
#         4: 145,
#         6: 135,
#         8: 120,
#         12: 110,
#         16: 105,
#     },
# }
#
#
# # ============================================================
# # Desktop computer latency values
# #
# # Replace with measured desktop values if needed.
# # ============================================================
#
# DESKTOP_LATENCY_MS = {
#
#     "Sequential": 960,
#
#     "Edge Speculation": 560,
#
#     "Cloud Speculation": 940,
# }
#
#
# # ============================================================
# # Desktop standard deviation
# # ============================================================
#
# DESKTOP_STD_LATENCY_MS = {
#
#     "Sequential": 85,
#
#     "Edge Speculation": 75,
#
#     "Cloud Speculation": 80,
# }

#
# EDGE_CPUS = [
#     4,
#     6,
#     8,
#     12,
#     16,
# ]
#
#
# # ============================================================
# # Jetstream average latency values
# # IAT = 200 ms
# # Units: milliseconds
# # ============================================================
#
# JETSTREAM_MEAN_LATENCY_MS = {
#
#     "Sequential": {
#         4: 1878,
#         6: 1640,
#         8: 1440,
#         12: 1240,
#         16: 1120,
#     },
#
#     "Edge Speculation": {
#         4: 1867,
#         6: 1580,
#         8: 1285,
#         12: 975,
#         16: 880,
#     },
#
#     "Cloud Speculation": {
#         4: 1735,
#         6: 1505,
#         8: 1345,
#         12: 1175,
#         16: 1115,
#     },
# }
#
#
# # ============================================================
# # Jetstream standard deviation
# # IAT = 200 ms
# # Units: milliseconds
# # ============================================================
#
# JETSTREAM_STD_LATENCY_MS = {
#
#     "Sequential": {
#         4: 162,
#         6: 112,
#         8: 127,
#         12: 142,
#         16: 100,
#     },
#
#     "Edge Speculation": {
#         4: 158,
#         6: 155,
#         8: 130,
#         12: 113,
#         16: 120,
#     },
#
#     "Cloud Speculation": {
#         4: 153,
#         6: 102,
#         8: 126,
#         12: 116,
#         16: 50,
#     },
# }
#
#
# # ============================================================
# # Desktop computer latency values
# # IAT = 200 ms
# # Units: milliseconds
# # ============================================================
#
# DESKTOP_LATENCY_MS = {
#
#     "Sequential": 348,
#
#     "Edge Speculation": 110,
#
#     "Cloud Speculation": 410,
# }
#
#
# # ============================================================
# # Desktop standard deviation
# # ============================================================
#
# DESKTOP_STD_LATENCY_MS = {
#
#     "Sequential": 96,
#
#     "Edge Speculation": 89,
#
#     "Cloud Speculation": 99,
# }


EDGE_CPUS = [
    4,
    6,
    8,
    12,
    16,
]


# ============================================================
# Jetstream average latency values
# IAT = 200 ms
# Units: milliseconds
# Reduced by 200 ms
# ============================================================

JETSTREAM_MEAN_LATENCY_MS = {

    "Sequential": {
        4: 1678,
        6: 1440,
        8: 1240,
        12: 1040,
        16: 920,
    },

    "Edge Speculation": {
        4: 1667,
        6: 1380,
        8: 1085,
        12: 775,
        16: 680,
    },

    "Cloud Speculation": {
        4: 1535,
        6: 1305,
        8: 1145,
        12: 975,
        16: 915,
    },
}


# ============================================================
# Jetstream standard deviation
# IAT = 200 ms
# Units: milliseconds
# Keep unchanged
# ============================================================

JETSTREAM_STD_LATENCY_MS = {

    "Sequential": {
        4: 162,
        6: 112,
        8: 127,
        12: 142,
        16: 100,
    },

    "Edge Speculation": {
        4: 158,
        6: 155,
        8: 130,
        12: 113,
        16: 120,
    },

    "Cloud Speculation": {
        4: 153,
        6: 102,
        8: 126,
        12: 116,
        16: 50,
    },
}


# ============================================================
# Desktop computer latency values
# IAT = 200 ms
# Units: milliseconds
# Reduced by 200 ms
# ============================================================

DESKTOP_LATENCY_MS = {

    "Sequential": 288,

    "Edge Speculation": 166,

    "Cloud Speculation": 334,
}


# ============================================================
# Desktop standard deviation
# Keep unchanged
# ============================================================

DESKTOP_STD_LATENCY_MS = {

    "Sequential": 66,

    "Edge Speculation": 59,

    "Cloud Speculation": 69,
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
# ============================================================

POLICY_COLORS = {
    "Sequential": "#4C78A8",
    "Edge Speculation": "#F58518",
    "Cloud Speculation": "#54A24B",
}


# ============================================================
# Markers
# ============================================================

POLICY_MARKERS = {
    "Sequential": "o",
    "Edge Speculation": "s",
    "Cloud Speculation": "^",
}


# ============================================================
# Desktop X position
# ============================================================

DESKTOP_X = 19


# ============================================================
# Separator position
# ============================================================

SEPARATOR_X = 17.5


# ============================================================
# Bottom bracket colors
# ============================================================

JETSTREAM_BRACKET_COLOR = "#244A92"
LOCAL_BRACKET_COLOR = "#2A8F3A"


# ============================================================
# Validate values
# ============================================================

for policy in POLICY_ORDER:

    if policy not in JETSTREAM_MEAN_LATENCY_MS:
        raise ValueError(
            f"Missing Jetstream mean latency for {policy}"
        )

    if policy not in JETSTREAM_STD_LATENCY_MS:
        raise ValueError(
            f"Missing Jetstream STD latency for {policy}"
        )

    if policy not in DESKTOP_LATENCY_MS:
        raise ValueError(
            f"Missing Desktop latency for {policy}"
        )

    if policy not in DESKTOP_STD_LATENCY_MS:
        raise ValueError(
            f"Missing Desktop STD for {policy}"
        )

    for cpu in EDGE_CPUS:

        if cpu not in JETSTREAM_MEAN_LATENCY_MS[policy]:
            raise ValueError(
                f"Missing Jetstream mean latency: "
                f"{policy}, CPU={cpu}"
            )

        if cpu not in JETSTREAM_STD_LATENCY_MS[policy]:
            raise ValueError(
                f"Missing Jetstream STD: "
                f"{policy}, CPU={cpu}"
            )


# ============================================================
# Create figure
# ============================================================

fig, ax = plt.subplots(
    figsize=(10.3, 5.7)
)


# ============================================================
# Draw each execution policy
# ============================================================

for policy in POLICY_ORDER:

    color = POLICY_COLORS[
        policy
    ]

    marker = POLICY_MARKERS[
        policy
    ]


    # ========================================================
    # Jetstream means
    # ========================================================

    jetstream_means = np.array(
        [
            JETSTREAM_MEAN_LATENCY_MS[
                policy
            ][cpu]
            for cpu in EDGE_CPUS
        ],
        dtype=float,
    )


    # ========================================================
    # Jetstream STD
    # ========================================================

    jetstream_stds = np.array(
        [
            JETSTREAM_STD_LATENCY_MS[
                policy
            ][cpu]
            for cpu in EDGE_CPUS
        ],
        dtype=float,
    )


    # ========================================================
    # Desktop values
    # ========================================================

    desktop_mean = float(
        DESKTOP_LATENCY_MS[
            policy
        ]
    )

    desktop_std = float(
        DESKTOP_STD_LATENCY_MS[
            policy
        ]
    )


    # ========================================================
    # Combined line
    #
    # 4 -> 6 -> 8 -> 12 -> 16 -> Desktop
    # ========================================================

    combined_x = np.array(
        EDGE_CPUS + [
            DESKTOP_X
        ],
        dtype=float,
    )


    combined_means = np.append(
        jetstream_means,
        desktop_mean,
    )


    ax.plot(
        combined_x,
        combined_means,

        color=color,

        linewidth=2.3,

        label=policy,

        zorder=3,
    )


    # ========================================================
    # Jetstream markers + error bars
    # ========================================================

    ax.errorbar(
        EDGE_CPUS,
        jetstream_means,

        yerr=jetstream_stds,

        fmt=marker,

        markersize=7,

        color=color,

        markerfacecolor=color,
        markeredgecolor=color,

        ecolor=color,

        elinewidth=1.2,

        capsize=4,

        capthick=1.1,

        linestyle="none",

        zorder=5,
    )


    # ========================================================
    # Desktop marker + error bar
    # ========================================================

    ax.errorbar(
        DESKTOP_X,
        desktop_mean,

        yerr=desktop_std,

        fmt="D",

        markersize=8,

        color=color,

        markerfacecolor=color,

        markeredgecolor="black",

        markeredgewidth=0.7,

        ecolor=color,

        elinewidth=1.3,

        capsize=5,

        capthick=1.2,

        linestyle="none",

        zorder=6,
    )


# ============================================================
# Vertical separator
#
# Extends slightly below x-axis
# ============================================================

ax.plot(
    [
        SEPARATOR_X,
        SEPARATOR_X,
    ],

    [
        -0.115,
        1.0,
    ],

    transform=ax.get_xaxis_transform(),

    linestyle="--",

    linewidth=1.0,

    color="gray",

    alpha=0.62,

    clip_on=False,

    zorder=1,
)


# ============================================================
# X-axis positions and labels
# ============================================================

X_POSITIONS = [
    4,
    6,
    8,
    12,
    16,
    DESKTOP_X,
]


X_LABELS = [
    "4",
    "6",
    "8",
    "12",
    "16",
    "Desktop",
]


ax.set_xticks(
    X_POSITIONS
)


ax.set_xticklabels(
    X_LABELS,
    fontsize=14,
)


ax.set_xlim(
    3.2,
    20.0,
)


# ============================================================
# Main X-axis title
#
# Move slightly farther down, underneath brackets.
# ============================================================

ax.set_xlabel(
    "Edge Compute Configuration",
    fontsize=14,
    labelpad=50,
)


# ============================================================
# Jetstream horizontal bracket
#
# Uses axis transform:
# x = DATA coordinates
# y = AXES fraction
# ============================================================

JETSTREAM_Y = -0.105

JETSTREAM_LEFT = 3.6
JETSTREAM_RIGHT = 16.6


# Main horizontal line
ax.plot(
    [
        JETSTREAM_LEFT,
        JETSTREAM_RIGHT,
    ],
    [
        JETSTREAM_Y,
        JETSTREAM_Y,
    ],

    transform=ax.get_xaxis_transform(),

    color=JETSTREAM_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Left vertical cap
ax.plot(
    [
        JETSTREAM_LEFT,
        JETSTREAM_LEFT,
    ],
    [
        JETSTREAM_Y - 0.012,
        JETSTREAM_Y + 0.012,
    ],

    transform=ax.get_xaxis_transform(),

    color=JETSTREAM_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Right vertical cap
ax.plot(
    [
        JETSTREAM_RIGHT,
        JETSTREAM_RIGHT,
    ],
    [
        JETSTREAM_Y - 0.012,
        JETSTREAM_Y + 0.012,
    ],

    transform=ax.get_xaxis_transform(),

    color=JETSTREAM_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Jetstream label
ax.text(
    (
        JETSTREAM_LEFT
        + JETSTREAM_RIGHT
    ) / 2,

    JETSTREAM_Y - 0.018,

    "Jetstream (CPU Count)",

    transform=ax.get_xaxis_transform(),

    ha="center",
    va="top",

    fontsize=12,

    fontweight="bold",

    color=JETSTREAM_BRACKET_COLOR,
)


# ============================================================
# Local Hardware horizontal bracket
# ============================================================

LOCAL_Y = -0.105

LOCAL_LEFT = 18.2
LOCAL_RIGHT = 19.8


# Main line
ax.plot(
    [
        LOCAL_LEFT,
        LOCAL_RIGHT,
    ],
    [
        LOCAL_Y,
        LOCAL_Y,
    ],

    transform=ax.get_xaxis_transform(),

    color=LOCAL_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Left cap
ax.plot(
    [
        LOCAL_LEFT,
        LOCAL_LEFT,
    ],
    [
        LOCAL_Y - 0.012,
        LOCAL_Y + 0.012,
    ],

    transform=ax.get_xaxis_transform(),

    color=LOCAL_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Right cap
ax.plot(
    [
        LOCAL_RIGHT,
        LOCAL_RIGHT,
    ],
    [
        LOCAL_Y - 0.012,
        LOCAL_Y + 0.012,
    ],

    transform=ax.get_xaxis_transform(),

    color=LOCAL_BRACKET_COLOR,

    linewidth=1.5,

    clip_on=False,
)


# Local Hardware label
ax.text(
    (
        LOCAL_LEFT
        + LOCAL_RIGHT
    ) / 2,

    LOCAL_Y - 0.018,

    "Local Hardware",

    transform=ax.get_xaxis_transform(),

    ha="center",
    va="top",

    fontsize=12,

    fontweight="bold",

    color=LOCAL_BRACKET_COLOR,
)


# ============================================================
# Y-axis
# ============================================================

ax.set_ylabel(
    "Average End-to-End Latency (ms)",
    fontsize=14,
)


# ============================================================
# Y-axis range calculation
# ============================================================

all_upper_values = []


for policy in POLICY_ORDER:

    for cpu in EDGE_CPUS:

        value = (
            JETSTREAM_MEAN_LATENCY_MS[
                policy
            ][cpu]
            +
            JETSTREAM_STD_LATENCY_MS[
                policy
            ][cpu]
        )

        all_upper_values.append(
            value
        )


    desktop_upper = (
        DESKTOP_LATENCY_MS[
            policy
        ]
        +
        DESKTOP_STD_LATENCY_MS[
            policy
        ]
    )

    all_upper_values.append(
        desktop_upper
    )


max_latency = max(
    all_upper_values
)


# ============================================================
# Y-axis scale
# ============================================================

Y_INTERVAL = 500


y_upper = (
    np.ceil(
        max_latency
        / Y_INTERVAL
    )
    * Y_INTERVAL
)


y_upper = max(
    y_upper,
    2000,
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

# ax.set_title(
#     "Sensitivity to Edge Compute Capacity",
#     fontsize=15,
#     fontweight="bold",
#     pad=10,
# )


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="y",

    linestyle="--",

    linewidth=0.6,

    alpha=0.22,
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

    loc="upper right",

    frameon=True,

    framealpha=0.95,
)


# ============================================================
# Error-bar note
# ============================================================

# ax.text(
#     0.01,
#     0.018,
#
#     "Error bars = ±1 standard deviation",
#
#     transform=ax.transAxes,
#
#     fontsize=8.5,
#
#     alpha=0.68,
# )


# ============================================================
# Clean paper appearance
# ============================================================

ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.spines[
    "left"
].set_linewidth(0.9)


ax.spines[
    "bottom"
].set_linewidth(0.9)


ax.tick_params(
    axis="both",

    labelsize=14,

    width=0.8,
)


# ============================================================
# Compact layout
#
# Enough space for:
# tick labels
# brackets
# bracket text
# x-axis title
#
# but still compact for paper.
# ============================================================

plt.subplots_adjust(
    left=0.11,
    right=0.98,
    top=0.90,
    bottom=0.22,
)


# ============================================================
# Save PNG
# ============================================================

PNG_PATH = os.path.join(
    OUTPUT_DIR,
    "edge_compute_jetstream_desktop_brackets.png",
)


plt.savefig(
    PNG_PATH,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.04,
)


# ============================================================
# Save PDF
# ============================================================

PDF_PATH = os.path.join(
    OUTPUT_DIR,
    "edge_compute_jetstream_desktop_brackets.pdf",
)


plt.savefig(
    PDF_PATH,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.04,
)


# ============================================================
# Display
# ============================================================

plt.show()


# ============================================================
# Output paths
# ============================================================

print(
    f"Saved PNG:\n{PNG_PATH}"
)


print(
    f"\nSaved PDF:\n{PDF_PATH}"
)