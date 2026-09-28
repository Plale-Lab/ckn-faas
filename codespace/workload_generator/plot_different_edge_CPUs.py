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
# CPU configurations
# ============================================================

EDGE_CPUS = [
    4,
    6,
    8,
    12,
]


# ============================================================
# MANUAL AVERAGE LATENCY VALUES
#
# Replace these with your measured average RespTime values
# in milliseconds.
# ============================================================

MEAN_LATENCY_MS = {

    "Sequential": {
        4: 1800,
        6: 1650,
        8: 1500,
        12: 1380,
    },

    "Edge Speculation": {
        4: 1700,
        6: 1450,
        8: 1250,
        12: 1100,
    },

    "Cloud Speculation": {
        4: 1650,
        6: 1550,
        8: 1450,
        12: 1375,
    },
}


# ============================================================
# STANDARD DEVIATION VALUES
#
# Replace with measured STD values.
#
# If you do not want shaded STD regions,
# simply set all values to 0.
# ============================================================

STD_LATENCY_MS = {

    "Sequential": {
        4: 150,
        6: 140,
        8: 125,
        12: 110,
    },

    "Edge Speculation": {
        4: 160,
        6: 130,
        8: 110,
        12: 95,
    },

    "Cloud Speculation": {
        4: 145,
        6: 135,
        8: 120,
        12: 110,
    },
}


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
# Create plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(9.5, 5.7)
)


for policy in [
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]:

    # --------------------------------------------------------
    # Get mean values
    # --------------------------------------------------------

    means = np.array(
        [
            MEAN_LATENCY_MS[
                policy
            ][cpu]
            for cpu in EDGE_CPUS
        ],
        dtype=float,
    )


    # --------------------------------------------------------
    # Get STD values
    # --------------------------------------------------------

    stds = np.array(
        [
            STD_LATENCY_MS[
                policy
            ][cpu]
            for cpu in EDGE_CPUS
        ],
        dtype=float,
    )


    color = POLICY_COLORS[
        policy
    ]


    # ========================================================
    # Mean line
    # ========================================================

    ax.plot(
        EDGE_CPUS,
        means,

        marker=POLICY_MARKERS[
            policy
        ],

        markersize=7,
        linewidth=2.5,

        color=color,

        label=policy,

        zorder=3,
    )


    # ========================================================
    # Standard deviation shaded region
    # ========================================================

    lower = np.maximum(
        means - stds,
        0,
    )

    upper = (
        means + stds
    )


    ax.fill_between(
        EDGE_CPUS,
        lower,
        upper,

        color=color,

        alpha=0.14,

        linewidth=0,

        zorder=1,
    )


# ============================================================
# X-axis
# ============================================================

ax.set_xlabel(
    "Edge CPU Count",
    fontsize=13,
)


ax.set_xticks(
    EDGE_CPUS
)


ax.set_xlim(
    3.5,
    12.5,
)


# ============================================================
# Y-axis
# ============================================================

ax.set_ylabel(
    "Average End-to-End Latency (ms)",
    fontsize=13,
)


# Start y-axis from zero
ax.set_ylim(
    bottom=0
)


# Find maximum including STD
all_upper_values = []

for policy in MEAN_LATENCY_MS:

    for cpu in EDGE_CPUS:

        upper_value = (
            MEAN_LATENCY_MS[
                policy
            ][cpu]
            +
            STD_LATENCY_MS[
                policy
            ][cpu]
        )

        all_upper_values.append(
            upper_value
        )


max_latency = max(
    all_upper_values
)


# Clean Y-axis interval
Y_INTERVAL = 500


y_upper = (
    np.ceil(
        max_latency
        / Y_INTERVAL
    )
    * Y_INTERVAL
)


# Add a little extra room
y_upper += Y_INTERVAL


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
    "Sensitivity to Edge CPU Capacity",
    fontsize=15,
    fontweight="bold",
    pad=12,
)


# ============================================================
# Grid
# ============================================================

ax.grid(
    axis="both",
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
    title="Execution Policy",
    fontsize=9,
    title_fontsize=10,
    loc="upper right",
    frameon=True,
)


# ============================================================
# Clean paper style
# ============================================================

ax.spines[
    "top"
].set_visible(False)


ax.spines[
    "right"
].set_visible(False)


ax.tick_params(
    axis="both",
    labelsize=11,
)


# ============================================================
# STD note
# ============================================================

ax.text(
    0.01,
    0.02,

    "Shaded region = ±1 standard deviation",

    transform=ax.transAxes,

    fontsize=9,

    alpha=0.7,
)


# ============================================================
# Layout
# ============================================================

plt.tight_layout()


# ============================================================
# Save
# ============================================================

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "edge_cpu_scaling_average_latency.png",
)


plt.savefig(
    OUTPUT_PATH,
    dpi=300,
    bbox_inches="tight",
)


plt.show()


print(
    f"Saved graph to:\n{OUTPUT_PATH}"
)