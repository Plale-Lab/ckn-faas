import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ============================================================
# Publication-ready DYSCI policy table
# Saves BOTH PNG and vector PDF
# ============================================================

# -----------------------------
# Output files
# -----------------------------
PNG_OUT = "dysci_policy_table.png"
PDF_OUT = "dysci_policy_table.pdf"

# -----------------------------
# Table dimensions / labels
# -----------------------------
C_VALUES = [20, 25, 35, 50, 75, 100]
E_VALUES = [0.18, 0.16, 0.14, 0.12, 0.10, 0.09]

CLOUD_LATENCIES = [0, 50, 100]

# -----------------------------
# Colors
# -----------------------------
COLORS = {
    "Sequential": "#A8D8ED",
    "Edge Speculation": "#F6E28A",
    "Cloud Speculation": "#EFAEBE",
    "none": "#D3D3D3",
}

# -----------------------------
# Percentage values
#
# Each matrix is:
# rows    -> E = [0.18, 0.16, 0.14, 0.12, 0.10, 0.09]
# columns -> C = [20, 25, 35, 50, 75, 100]
# -----------------------------
VALUES = {
    0: [
        [7.2,  6.1,  6.1,  6.1,  6.1,  6.1],
        [6.3,  5.8,  5.8,  5.8,  5.8,  5.8],
        [6.6,  9.1,  9.1,  9.1,  9.1,  9.1],
        [6.5, 13.5, 13.5, 13.5, 13.5, 13.5],
        [None, 3.7, 3.7, 3.7, 3.7, 9.4],
        [None, None, None, 48.2, 38.8, 32.3],
    ],

    50: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.6, 2.6, 2.6, 2.6, 2.6, 2.6],
        [4.9, 3.2, 3.2, 3.6, 3.6, 3.6],
        [12.7, 1.1, 1.1, 7.3, 7.3, 7.3],
        [None, 1.9, 1.9, 1.9, 1.9, 6.6],
        [None, None, None, 39.9, 39.9, 39.9],
    ],

    100: [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [6.8, 6.8, 6.8, 6.8, 6.8, 5.1],
        [4.3, 4.3, 4.3, 0.2, 0.2, 0.2],
        [7.4, 7.4, 7.4, 5.5, 5.5, 5.5],
        [None, 0.9, 0.9, 0.9, 0.9, 5.1],
        [None, None, None, 33.7, 33.7, 33.7],
    ],
}

# -----------------------------
# Optimal policy corresponding
# to every cell
# -----------------------------
POLICIES = {
    0: [
        ["Edge Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["Edge Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["Edge Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["Edge Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["none",
         "Sequential", "Sequential", "Sequential",
         "Sequential", "Cloud Speculation"],

        ["none", "none", "none",
         "Edge Speculation", "Edge Speculation",
         "Edge Speculation"],
    ],

    50: [
        ["Sequential"] * 6,

        ["Edge Speculation"] * 6,

        ["Cloud Speculation"] * 6,

        ["Sequential", "Sequential", "Sequential",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["none",
         "Sequential", "Sequential", "Sequential",
         "Sequential", "Cloud Speculation"],

        ["none", "none", "none",
         "Edge Speculation", "Edge Speculation",
         "Edge Speculation"],
    ],

    100: [
        ["Sequential"] * 6,

        ["Edge Speculation"] * 6,

        ["Edge Speculation", "Edge Speculation",
         "Edge Speculation",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["Sequential", "Sequential", "Sequential",
         "Cloud Speculation", "Cloud Speculation",
         "Cloud Speculation"],

        ["none",
         "Sequential", "Sequential", "Sequential",
         "Sequential", "Cloud Speculation"],

        ["none", "none", "none",
         "Edge Speculation", "Edge Speculation",
         "Edge Speculation"],
    ],
}


# ============================================================
# Figure setup
# ============================================================

# Good ratio for a full-width figure in a two-column paper
fig, ax = plt.subplots(figsize=(11.5, 4.15))

ax.axis("off")

n_rows = len(E_VALUES)
n_cols = len(C_VALUES)

# Geometry
cell_w = 1.0
cell_h = 0.72

label_w = 0.75
group_gap = 0.22

header_h = 0.65
subheader_h = 0.55

x_start = 0.0
y_start = 0.0

# Styling
grid_color = "0.45"
grid_lw = 0.55
outer_lw = 0.9

value_fontsize = 9.0
label_fontsize = 9.5
title_fontsize = 11.0

# ============================================================
# Draw each cloud-latency group
# ============================================================

for group_idx, cloud_latency in enumerate(CLOUD_LATENCIES):

    group_width = label_w + n_cols * cell_w

    gx = x_start + group_idx * (group_width + group_gap)

    # --------------------------------------------------------
    # Group title
    # --------------------------------------------------------
    title_y = (
        y_start
        + n_rows * cell_h
        + subheader_h
    )

    ax.add_patch(
        Rectangle(
            (gx, title_y),
            group_width,
            header_h,
            facecolor="white",
            edgecolor=grid_color,
            linewidth=outer_lw
        )
    )

    ax.text(
        gx + group_width / 2,
        title_y + header_h / 2,
        f"Cloud Latency = {cloud_latency} ms",
        ha="center",
        va="center",
        fontsize=title_fontsize,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # E \ C header
    # --------------------------------------------------------
    sub_y = y_start + n_rows * cell_h

    ax.add_patch(
        Rectangle(
            (gx, sub_y),
            label_w,
            subheader_h,
            facecolor="white",
            edgecolor=grid_color,
            linewidth=grid_lw
        )
    )

    ax.text(
        gx + label_w / 2,
        sub_y + subheader_h / 2,
        r"$E \backslash C$",
        ha="center",
        va="center",
        fontsize=label_fontsize,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # C headers
    # --------------------------------------------------------
    for j, C in enumerate(C_VALUES):

        x = gx + label_w + j * cell_w

        ax.add_patch(
            Rectangle(
                (x, sub_y),
                cell_w,
                subheader_h,
                facecolor="white",
                edgecolor=grid_color,
                linewidth=grid_lw
            )
        )

        ax.text(
            x + cell_w / 2,
            sub_y + subheader_h / 2,
            str(C),
            ha="center",
            va="center",
            fontsize=label_fontsize,
            fontweight="bold"
        )

    # --------------------------------------------------------
    # Data cells
    # --------------------------------------------------------
    for i, E in enumerate(E_VALUES):

        # Reverse vertical position so 0.18 is at top
        y = y_start + (n_rows - 1 - i) * cell_h

        # E label cell
        ax.add_patch(
            Rectangle(
                (gx, y),
                label_w,
                cell_h,
                facecolor="white",
                edgecolor=grid_color,
                linewidth=grid_lw
            )
        )

        # Display 0.10 as 0.10 instead of 0.1 for consistency
        e_label = f"{E:.2f}"

        ax.text(
            gx + label_w / 2,
            y + cell_h / 2,
            e_label,
            ha="center",
            va="center",
            fontsize=label_fontsize
        )

        # Values
        for j in range(n_cols):

            x = gx + label_w + j * cell_w

            policy = POLICIES[cloud_latency][i][j]
            value = VALUES[cloud_latency][i][j]

            ax.add_patch(
                Rectangle(
                    (x, y),
                    cell_w,
                    cell_h,
                    facecolor=COLORS[policy],
                    edgecolor=grid_color,
                    linewidth=grid_lw
                )
            )

            if value is None:
                text = "–"
            else:
                text = f"{value:.1f}%"

            ax.text(
                x + cell_w / 2,
                y + cell_h / 2,
                text,
                ha="center",
                va="center",
                fontsize=value_fontsize
            )

    # Strong outer border
    total_height = n_rows * cell_h + subheader_h + header_h

    ax.add_patch(
        Rectangle(
            (gx, y_start),
            group_width,
            total_height,
            fill=False,
            edgecolor="0.25",
            linewidth=outer_lw
        )
    )


# ============================================================
# Legend
# ============================================================

legend_y = (
    y_start
    + n_rows * cell_h
    + subheader_h
    + header_h
    + 0.42
)

legend_items = [
    ("Sequential", COLORS["Sequential"]),
    ("Edge Speculation", COLORS["Edge Speculation"]),
    ("Cloud Speculation", COLORS["Cloud Speculation"]),
    ("Infeasible", COLORS["none"]),
]

# Center legend across full table
total_group_width = (
    3 * (label_w + n_cols * cell_w)
    + 2 * group_gap
)

legend_spacing = 3.25
legend_total = legend_spacing * len(legend_items)

legend_x = total_group_width / 2 - legend_total / 2 + 0.35

for label, color in legend_items:

    box_size = 0.23

    ax.add_patch(
        Rectangle(
            (legend_x, legend_y),
            box_size,
            box_size,
            facecolor=color,
            edgecolor="0.4",
            linewidth=0.5
        )
    )

    ax.text(
        legend_x + box_size + 0.13,
        legend_y + box_size / 2,
        label,
        ha="left",
        va="center",
        fontsize=9.5
    )

    legend_x += legend_spacing


# ============================================================
# Limits / spacing
# ============================================================

ax.set_xlim(
    -0.05,
    total_group_width + 0.05
)

ax.set_ylim(
    -0.05,
    legend_y + 0.48
)

plt.tight_layout(pad=0.15)


# ============================================================
# Save files
# ============================================================

# PNG: high resolution for slides / preview
plt.savefig(
    PNG_OUT,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03
)

# PDF: VECTOR format -- recommended for your paper
plt.savefig(
    PDF_OUT,
    format="pdf",
    bbox_inches="tight",
    pad_inches=0.03
)

plt.show()

print(f"Saved PNG: {PNG_OUT}")
print(f"Saved PDF: {PDF_OUT}")