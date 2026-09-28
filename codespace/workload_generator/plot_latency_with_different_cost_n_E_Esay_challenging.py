# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # =========================
# # Config
# # =========================
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/latency_with_different_cost_n_E_Esay_challenging.csv"
# OUT_DIR = "policy_three_section_graph"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# Y_METRIC = "RespTime"   # uses RespTime column
# AGG = "mean"
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential Cascade",
#     "parallel_first_finish": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "po2": "Edge Speculation",
#     "spec_b": "Cloud Speculation",
#     "cloud_spec": "Cloud Speculation",
# }
#
# POLICY_ORDER = [
#     "Sequential Cascade",
#     "Edge Speculation",
#     "Cloud Speculation",
# ]
#
# POLICY_COLORS = {
#     "Sequential Cascade": "#1f77b4",
#     "Edge Speculation": "#2ca02c",
#     "Cloud Speculation": "#d62728",
# }
#
# POLICY_MARKERS = {
#     "Sequential Cascade": "o",
#     "Edge Speculation": "s",
#     "Cloud Speculation": "^",
# }
#
# # =========================
# # Load CSV
# # =========================
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = [
#     "RespTime",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["C"] = df["static_cost"]
# df["E"] = df["static_error"]
#
# # =========================
# # Policy name
# # =========================
# def get_policy(row):
#     for col in ["static_policy", "effective_cascade_mode", "configured_cascade_mode", "main_policy"]:
#         if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
#             return str(row[col]).strip()
#     return "unknown"
#
# df["policy_raw"] = df.apply(get_policy, axis=1)
# df["policy"] = df["policy_raw"].map(POLICY_NAME_MAP).fillna(df["policy_raw"])
#
# # =========================
# # Helper
# # =========================
# def summarize(data, group_name):
#     if data.empty:
#         print(f"WARNING: no data for {group_name}")
#         return []
#
#     summary = (
#         data.groupby("policy")["latency_ms"]
#         .agg(["mean", "std", "count"])
#         .reset_index()
#     )
#
#     rows = []
#     for _, row in summary.iterrows():
#         rows.append({
#             "x_group": group_name,
#             "policy": row["policy"],
#             "latency_mean_ms": row["mean"],
#             "latency_std_ms": 0.0 if pd.isna(row["std"]) else row["std"],
#             "count": row["count"],
#         })
#
#     return rows
#
# # =========================
# # Select graph sections
# # =========================
# plot_points = []
#
# # 1) Fixed: C=100, E=0.10
# fixed_df = df[
#     (df["C"].round(3) == 100) &
#     (df["E"].round(3) == 0.100)
# ]
#
# plot_points += summarize(
#     fixed_df,
#     "Fixed\nC=100, E=0.10"
# )
#
# # 2) Equal cost mix: C=100, E={0.10, 0.12, 0.14}
# equal_cost_errors = [0.10, 0.12, 0.14]
#
# equal_cost_df = df[
#     (df["C"].round(3) == 100) &
#     (df["E"].round(3).isin(equal_cost_errors))
# ]
#
# plot_points += summarize(
#     equal_cost_df,
#     "Equal cost mix\nC=100\nE={0.10,0.12,0.14}"
# )
#
# # 3) Easy: C=25, E=0.140
# easy_df = df[
#     (df["C"].round(3) == 25) &
#     (df["E"].round(3) == 0.140)
# ]
#
# plot_points += summarize(
#     easy_df,
#     "Easy\nC=25, E=0.14"
# )
#
# # 4) Challenging: C=100, E=0.090
# challenging_df = df[
#     (df["C"].round(3) == 100) &
#     (df["E"].round(3) == 0.090)
# ]
#
# plot_points += summarize(
#     challenging_df,
#     "Challenging\nC=100, E=0.09"
# )
#
# # =========================
# # Plot dataframe
# # =========================
# plot_df = pd.DataFrame(plot_points)
#
# print("\nPlot data:")
# print(plot_df)
#
# if plot_df.empty:
#     raise ValueError("No matching data found. Check static_cost/static_error values.")
#
# x_labels = [
#     "Fixed\nC=100, E=0.10",
#     "Equal cost mix\nC=100\nE={0.10,0.12,0.14}",
#     "Easy\nC=25, E=0.14",
#     "Challenging\nC=100, E=0.09",
# ]
#
# x_positions = list(range(len(x_labels)))
# x_map = dict(zip(x_labels, x_positions))
#
# available_policies = plot_df["policy"].unique().tolist()
# ordered_policies = [p for p in POLICY_ORDER if p in available_policies]
# ordered_policies += [p for p in available_policies if p not in ordered_policies]
#
# # =========================
# # Plot
# # =========================
# plt.figure(figsize=(13, 6))
#
# for policy in ordered_policies:
#     temp = plot_df[plot_df["policy"] == policy].copy()
#     temp["x"] = temp["x_group"].map(x_map)
#     temp = temp.dropna(subset=["x"]).sort_values("x")
#
#     x = temp["x"].values
#     y = temp["latency_mean_ms"].values
#     std = temp["latency_std_ms"].values
#
#     color = POLICY_COLORS.get(policy, None)
#
#     plt.plot(
#         x,
#         y,
#         marker=POLICY_MARKERS.get(policy, "o"),
#         linewidth=2.8,
#         markersize=8,
#         label=policy,
#         color=color
#     )
#
#     # shaded standard deviation
#     # plt.fill_between(
#     #     x,
#     #     y - std,
#     #     y + std,
#     #     alpha=0.18,
#     #     color=color
#     # )
#
# plt.xticks(x_positions, x_labels, fontsize=10)
# plt.ylabel("Response Time (ms)", fontsize=12)
# plt.xlabel("Experiment Setting", fontsize=12)
#
# plt.grid(axis="y", linestyle="--", alpha=0.35)
# plt.legend(title="Policy", fontsize=10)
#
# plt.title("Policy Response Time Across Fixed, Equal-Cost, Easy, and Challenging Settings")
#
# plt.tight_layout()
#
# out_path = os.path.join(OUT_DIR, "policy_three_section_resptime_with_std.png")
# plt.savefig(out_path, dpi=300, bbox_inches="tight")
# plt.show()
#
# print(f"Saved figure to: {out_path}")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config
# =========================
CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/latency_with_different_cost_n_E_Esay_challenging.csv"
OUT_DIR = "policy_three_section_graph"
os.makedirs(OUT_DIR, exist_ok=True)

POLICY_NAME_MAP = {
    "sequential": "Sequential Cascade",
    "parallel_first_finish": "Edge Speculation",
    "power2": "Edge Speculation",
    "po2": "Edge Speculation",
    "spec_b": "Cloud Speculation",
    "cloud_spec": "Cloud Speculation",
}

POLICY_ORDER = [
    "Sequential Cascade",
    "Edge Speculation",
    "Cloud Speculation",
]

POLICY_COLORS = {
    "Sequential Cascade": "#59A14F",  # soft blue
    "Edge Speculation": "#F28E2B",    # soft green
    "Cloud Speculation": "#4C78A8",   # orange
}

BAR_WIDTH = 0.24
SHOW_ERROR_BARS = True

AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 11
TITLE_FONTSIZE = 15

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# =========================
# Load CSV
# =========================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

for col in ["RespTime", "static_cost", "static_error"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["RespTime", "static_cost", "static_error"])

df["latency_ms"] = df["RespTime"] * 1000.0
df["C"] = df["static_cost"]
df["E"] = df["static_error"]

# =========================
# Policy name
# =========================
def get_policy(row):
    for col in [
        "static_policy",
        "effective_cascade_mode",
        "configured_cascade_mode",
        "main_policy",
    ]:
        if col in row and pd.notna(row[col]) and str(row[col]).strip() != "":
            return str(row[col]).strip()

    return "unknown"


df["policy_raw"] = df.apply(get_policy, axis=1)
df["policy"] = df["policy_raw"].map(POLICY_NAME_MAP).fillna(df["policy_raw"])

# =========================
# Helper
# =========================
def summarize(data, group_name):
    if data.empty:
        print(f"WARNING: no data for {group_name}")
        return []

    summary = (
        data.groupby("policy")["latency_ms"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    rows = []

    for _, row in summary.iterrows():
        rows.append(
            {
                "x_group": group_name,
                "policy": row["policy"],
                "latency_mean_ms": row["mean"],
                "latency_std_ms": 0.0 if pd.isna(row["std"]) else row["std"],
                "count": row["count"],
            }
        )

    return rows


# =========================
# Select graph sections
# =========================
plot_points = []

fixed_df = df[
    (df["C"].round(3) == 100)
    & (df["E"].round(3) == 0.100)
]

plot_points += summarize(
    fixed_df,
    "Fixed\nC=100, E=0.10",
)

equal_cost_errors = [0.10, 0.12, 0.14]

equal_cost_df = df[
    (df["C"].round(3) == 100)
    & (df["E"].round(3).isin(equal_cost_errors))
]

plot_points += summarize(
    equal_cost_df,
    "Equal cost mix\nC=100\nE={0.10,0.12,0.14}",
)

easy_df = df[
    (df["C"].round(3) == 25)
    & (df["E"].round(3) == 0.140)
]

plot_points += summarize(
    easy_df,
    "Easy\nC=25, E=0.14",
)

challenging_df = df[
    (df["C"].round(3) == 100)
    & (df["E"].round(3) == 0.090)
]

plot_points += summarize(
    challenging_df,
    "Challenging\nC=100, E=0.09",
)

plot_df = pd.DataFrame(plot_points)

print("\nPlot data:")
print(plot_df)

if plot_df.empty:
    raise ValueError("No matching data found. Check static_cost/static_error values.")

# =========================
# Plot setup
# =========================
x_labels = [
    "Fixed\nC=100, E=0.10",
    "Equal cost mix\nC=100\nE={0.10,0.12,0.14}",
    "Easy\nC=25, E=0.14",
    "Challenging\nC=100, E=0.09",
]

available_policies = plot_df["policy"].unique().tolist()
ordered_policies = [p for p in POLICY_ORDER if p in available_policies]
ordered_policies += [p for p in available_policies if p not in ordered_policies]

x = np.arange(len(x_labels))
num_policies = len(ordered_policies)

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# =========================
# Grouped bar chart
# =========================
for i, policy in enumerate(ordered_policies):
    temp = plot_df[plot_df["policy"] == policy].copy()

    means = []
    stds = []

    for group in x_labels:
        row = temp[temp["x_group"] == group]

        if row.empty:
            means.append(np.nan)
            stds.append(0.0)
        else:
            means.append(float(row["latency_mean_ms"].iloc[0]))
            stds.append(float(row["latency_std_ms"].iloc[0]))

    offset = (i - (num_policies - 1) / 2) * BAR_WIDTH

    if SHOW_ERROR_BARS:
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            label=policy,
            color=POLICY_COLORS.get(policy, None),
            yerr=stds,
            capsize=4,
            edgecolor="black",
            linewidth=0.6,
        )
    else:
        ax.bar(
            x + offset,
            means,
            width=BAR_WIDTH,
            label=policy,
            color=POLICY_COLORS.get(policy, None),
            edgecolor="black",
            linewidth=0.6,
        )

# =========================
# Style
# =========================
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=TICK_FONTSIZE)

ax.set_ylabel("Average Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_xlabel("Experiment Setting", fontsize=AXIS_LABEL_FONTSIZE)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

ax.legend(
    title="Policy",
    fontsize=LEGEND_FONTSIZE,
    title_fontsize=LEGEND_FONTSIZE,
    loc="upper left",
    frameon=True,
)

# ax.set_title(
#     "Policy Response Time Across Fixed, Equal-Cost, Easy, and Challenging Settings",
#     fontsize=TITLE_FONTSIZE,
#     fontweight="bold",
# )

plt.tight_layout()

# =========================
# Save PNG and PDF
# =========================
out_base = os.path.join(
    OUT_DIR,
    "policy_three_section_resptime_grouped_bar",
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

plt.show()

print(f"Saved figure to:")
print(f"{out_base}.png")
print(f"{out_base}.pdf")