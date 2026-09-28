# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv"
# # CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
# OUT_DIR = "policy_latency_plots_with_baselines"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = [
#     "IAR",
#     "RespTime",
#     "e2e_time_ms",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# # RespTime is seconds
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["is_static"] = df["static_config_enabled"].astype(str).str.lower() == "true"
# df["baseline_mode"] = df["baseline_mode"].astype(str)
# df["static_policy"] = df["static_policy"].astype(str)
#
# # -----------------------------
# # Split static optimized and baseline rows
# # -----------------------------
# static_df = df[df["is_static"]].copy()
# baseline_df = df[(~df["is_static"]) & (df["baseline_mode"].str.lower() != "none")].copy()
#
# # -----------------------------
# # Summary for optimized configs
# # -----------------------------
# static_summary = (
#     static_df.groupby(
#         ["static_policy", "static_cost", "static_error", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(mean_latency_ms=("latency_ms", "mean"))
# )
#
# # -----------------------------
# # Summary for baselines
# # -----------------------------
# baseline_summary = (
#     baseline_df.groupby(
#         ["baseline_mode", "effective_cascade_mode", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(mean_latency_ms=("latency_ms", "mean"))
# )
#
# # -----------------------------
# # Plot one figure per policy
# # Each figure has one subplot per C value
# # Baselines appear in every subplot
# # -----------------------------
# for policy, p_df in static_summary.groupby("static_policy"):
#
#     costs = sorted(p_df["static_cost"].dropna().unique())
#     errors = sorted(p_df["static_error"].dropna().unique())
#
#     if not costs:
#         continue
#
#     fig, axes = plt.subplots(
#         1,
#         len(costs),
#         figsize=(8.5 * len(costs), 5),
#         sharey=True,
#     )
#
#     if len(costs) == 1:
#         axes = [axes]
#
#     for ax, cost in zip(axes, costs):
#         c_df = p_df[p_df["static_cost"] == cost]
#
#         # Optimized configurations for this cost
#         for error in errors:
#             g = c_df[c_df["static_error"] == error].sort_values("arrival_rate_rps")
#
#             if g.empty:
#                 continue
#
#             ax.plot(
#                 g["arrival_rate_rps"],
#                 g["mean_latency_ms"],
#                 marker="o",
#                 linewidth=2,
#                 label=f"Optimized {policy}, E={error:.3f}",
#             )
#
#         # Baselines
#         for baseline_name, b in baseline_summary.groupby("baseline_mode"):
#             b = b.sort_values("arrival_rate_rps")
#
#             ax.plot(
#                 b["arrival_rate_rps"],
#                 b["mean_latency_ms"],
#                 linestyle="--",
#                 marker="x",
#                 linewidth=2,
#                 label=f"Baseline: {baseline_name}",
#             )
#
#         # ax.set_title(f"C={int(cost)}")
#         ax.set_xlabel("Arrival Rate (req/s)")
#         ax.grid(True, alpha=0.35)
#
#     axes[0].set_ylabel("Mean Latency (ms)")
#     axes[-1].legend(fontsize=8, loc="upper left")
#
#     fig.suptitle(f"{policy}: Optimized Configurations vs Baselines", fontsize=15)
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/{policy}_optimized_vs_baselines.png", dpi=300)
#     plt.show()
#
# print(f"Saved plots to {OUT_DIR}")



# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv"
# # CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
#
# OUT_DIR = "policy_latency_plots_with_baselines"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# POLICY_LABELS = {
#     "po2": "Power-of-Two Cascade",
#     "power2": "Power-of-Two Cascade",
#     "sequential": "Sequential Cascade",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
# }
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = ["IAR", "RespTime", "e2e_time_ms", "static_cost", "static_error"]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["is_static"] = df["static_config_enabled"].astype(str).str.lower() == "true"
# df["baseline_mode"] = df["baseline_mode"].astype(str)
# df["static_policy"] = df["static_policy"].astype(str)
#
# static_df = df[df["is_static"]].copy()
# baseline_df = df[(~df["is_static"]) & (df["baseline_mode"].str.lower() != "none")].copy()
#
# static_summary = (
#     static_df.groupby(
#         ["static_policy", "static_cost", "static_error", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(mean_latency_ms=("latency_ms", "mean"))
# )
#
# baseline_summary = (
#     baseline_df.groupby(
#         ["baseline_mode", "effective_cascade_mode", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(mean_latency_ms=("latency_ms", "mean"))
# )
#
# for policy, p_df in static_summary.groupby("static_policy"):
#
#     policy_display = POLICY_LABELS.get(policy, policy)
#
#     costs = sorted(p_df["static_cost"].dropna().unique())
#     errors = sorted(p_df["static_error"].dropna().unique())
#
#     if not costs:
#         continue
#
#     fig, axes = plt.subplots(
#         1,
#         len(costs),
#         figsize=(8.5 * len(costs), 5),
#         sharey=True,
#     )
#
#     if len(costs) == 1:
#         axes = [axes]
#
#     for ax, cost in zip(axes, costs):
#         c_df = p_df[p_df["static_cost"] == cost]
#
#         for error in errors:
#             g = c_df[c_df["static_error"] == error].sort_values("arrival_rate_rps")
#
#             if g.empty:
#                 continue
#
#             ax.plot(
#                 g["arrival_rate_rps"],
#                 g["mean_latency_ms"],
#                 marker="o",
#                 linewidth=2,
#                 label=f"Optimized {policy_display}, E={error:.3f}",
#             )
#
#         for baseline_name, b in baseline_summary.groupby("baseline_mode"):
#             b = b.sort_values("arrival_rate_rps")
#
#             baseline_display = BASELINE_LABELS.get(baseline_name, baseline_name)
#
#             ax.plot(
#                 b["arrival_rate_rps"],
#                 b["mean_latency_ms"],
#                 linestyle="--",
#                 marker="x",
#                 linewidth=2,
#                 label=f"Baseline: {baseline_display}",
#             )
#
#         ax.set_xlabel("Arrival Rate (req/s)")
#         ax.grid(True, alpha=0.35)
#
#     axes[0].set_ylabel("Mean Latency (ms)")
#     axes[-1].legend(fontsize=8, loc="upper left")
#
#     # fig.suptitle(
#     #     f"{policy_display}: Optimized Configurations vs Baselines",
#     #     fontsize=15,
#     # )
#
#     plt.tight_layout()
#
#     safe_policy_name = policy_display.lower().replace(" ", "_").replace("-", "_")
#     plt.savefig(f"{OUT_DIR}/{safe_policy_name}_optimized_vs_baselines.png", dpi=300)
#     plt.show()
#
# print(f"Saved plots to {OUT_DIR}")


# Error bars code start
#
# -----------------------------
#
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv"
# # CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
# CSV_PATH =  "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv"
#
# OUT_DIR = "policy_latency_plots_with_baselines"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# POLICY_LABELS = {
#         "po2": "Edge Speculation",
#         "power2": "Edge Speculation",
#         "sequential": "Sequential Cascade",
#         "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
#     "spec_b_smallest_to_largest": "smallest model and largest model run in parallel",
#     "spec_b_random_to_random": "random model and random model run in parallel",
# }
#
# FIG_BG_COLOR = "#f2f2f2"
# AX_BG_COLOR = "#fcfcfc"
# LEGEND_BG_COLOR = "#ffffff"
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = ["IAR", "RespTime", "e2e_time_ms", "static_cost", "static_error"]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["is_static"] = df["static_config_enabled"].astype(str).str.lower() == "true"
# df["baseline_mode"] = df["baseline_mode"].astype(str)
# df["static_policy"] = df["static_policy"].astype(str)
#
# static_df = df[df["is_static"]].copy()
# baseline_df = df[
#     (~df["is_static"]) & (df["baseline_mode"].str.lower() != "none")
# ].copy()
#
# static_summary = (
#     static_df.groupby(
#         ["static_policy", "static_cost", "static_error", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(
#         mean_latency_ms=("latency_ms", "mean"),
#         std_latency_ms=("latency_ms", "std"),
#         n=("latency_ms", "count"),
#     )
# )
#
# baseline_summary = (
#     baseline_df.groupby(
#         ["baseline_mode", "effective_cascade_mode", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(
#         mean_latency_ms=("latency_ms", "mean"),
#         std_latency_ms=("latency_ms", "std"),
#         n=("latency_ms", "count"),
#     )
# )
#
# static_summary["std_latency_ms"] = static_summary["std_latency_ms"].fillna(0)
# baseline_summary["std_latency_ms"] = baseline_summary["std_latency_ms"].fillna(0)
#
#
# def clipped_std_yerr(summary_df):
#     lower_err = summary_df["std_latency_ms"].clip(
#         upper=summary_df["mean_latency_ms"]
#     )
#     upper_err = summary_df["std_latency_ms"]
#     return [lower_err, upper_err]
#
#
# for policy, p_df in static_summary.groupby("static_policy"):
#
#     policy_display = POLICY_LABELS.get(policy, policy)
#
#     costs = sorted(p_df["static_cost"].dropna().unique())
#     errors = sorted(p_df["static_error"].dropna().unique())
#
#     if not costs:
#         continue
#
#     fig, axes = plt.subplots(
#         1,
#         len(costs),
#         figsize=(8.5 * len(costs), 5),
#         sharey=True,
#     )
#
#     fig.patch.set_facecolor(FIG_BG_COLOR)
#
#     if len(costs) == 1:
#         axes = [axes]
#
#     for ax, cost in zip(axes, costs):
#         ax.set_facecolor(AX_BG_COLOR)
#
#         c_df = p_df[p_df["static_cost"] == cost]
#
#         for error in errors:
#             g = c_df[c_df["static_error"] == error].sort_values("arrival_rate_rps")
#
#             if g.empty:
#                 continue
#
#             ax.errorbar(
#                 g["arrival_rate_rps"],
#                 g["mean_latency_ms"],
#                 yerr=clipped_std_yerr(g),
#                 marker="o",
#                 linewidth=2,
#                 capsize=4,
#                 label=f"{policy_display}, E={error:.3f}",
#             )
#
#         for baseline_name, b in baseline_summary.groupby("baseline_mode"):
#             b = b.sort_values("arrival_rate_rps")
#             baseline_display = BASELINE_LABELS.get(baseline_name, baseline_name)
#
#             ax.errorbar(
#                 b["arrival_rate_rps"],
#                 b["mean_latency_ms"],
#                 yerr=clipped_std_yerr(b),
#                 linestyle="--",
#                 marker="x",
#                 linewidth=2,
#                 capsize=4,
#                 label=f"Baseline: {baseline_display}",
#             )
#
#         # ax.set_title(f"C={cost:.3f} (Mean ± STD)")
#         ax.set_xlabel("Arrival Rate (req/s)")
#         ax.set_ylim(bottom=0)
#         ax.grid(True, alpha=0.3)
#
#     axes[0].set_ylabel("Average Latency (ms)")
#
#     legend = axes[-1].legend(fontsize=8, loc="upper left")
#     legend.get_frame().set_facecolor(LEGEND_BG_COLOR)
#     legend.get_frame().set_alpha(0.9)
#
#     plt.tight_layout()
#
#     safe_policy_name = policy_display.lower().replace(" ", "_").replace("-", "_")
#     plt.savefig(
#         f"{OUT_DIR}/{safe_policy_name}_optimized_vs_baselines_mean_std.png",
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=fig.get_facecolor(),
#     )
#
#     plt.show()
#
# print(f"Saved plots to {OUT_DIR}")
#


# Error bars code end
#
# -----------------------------
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.ticker import MultipleLocator
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv"
# # CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
# # CSV_PATH =  "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv"
#
# OUT_DIR = "policy_latency_plots_with_baselines"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "sequential": "Sequential Cascade",
#     "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
#     "spec_b_smallest_to_largest": "smallest model and largest model run in parallel",
#     "spec_b_random_to_random": "random model and random model run in parallel",
#     "full_ensemble_edge": "Cost-selected 3-model edge ensemble",
#     "full_ensemble_cloud": "full cloud ensemble",
# }
#
# FIG_BG_COLOR = "white"
# AX_BG_COLOR = "white"
# LEGEND_BG_COLOR = "white"
#
# SHADE_ALPHA = 0.15
# Y_TICK_INTERVAL = 500
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = ["IAR", "RespTime", "e2e_time_ms", "static_cost", "static_error"]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["is_static"] = df["static_config_enabled"].astype(str).str.lower() == "true"
# df["baseline_mode"] = df["baseline_mode"].astype(str)
# df["static_policy"] = df["static_policy"].astype(str)
#
# static_df = df[df["is_static"]].copy()
# baseline_df = df[
#     (~df["is_static"]) & (df["baseline_mode"].str.lower() != "none")
# ].copy()
#
# static_summary = (
#     static_df.groupby(
#         ["static_policy", "static_cost", "static_error", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(
#         mean_latency_ms=("latency_ms", "mean"),
#         std_latency_ms=("latency_ms", "std"),
#         n=("latency_ms", "count"),
#     )
# )
#
# baseline_summary = (
#     baseline_df.groupby(
#         ["baseline_mode", "effective_cascade_mode", "arrival_rate_rps"],
#         as_index=False,
#     )
#     .agg(
#         mean_latency_ms=("latency_ms", "mean"),
#         std_latency_ms=("latency_ms", "std"),
#         n=("latency_ms", "count"),
#     )
# )
#
# static_summary["std_latency_ms"] = static_summary["std_latency_ms"].fillna(0)
# baseline_summary["std_latency_ms"] = baseline_summary["std_latency_ms"].fillna(0)
#
#
# def plot_mean_with_std_band(
#     ax,
#     data,
#     label,
#     linestyle="-",
#     marker="o",
#     linewidth=2,
# ):
#     data = data.sort_values("arrival_rate_rps")
#
#     x = data["arrival_rate_rps"]
#     y = data["mean_latency_ms"]
#     std = data["std_latency_ms"]
#
#     lower = (y - std).clip(lower=0)
#     upper = y + std
#
#     line = ax.plot(
#         x,
#         y,
#         linestyle=linestyle,
#         marker=marker,
#         linewidth=linewidth,
#         markersize=6,
#         label=label,
#     )[0]
#
#     color = line.get_color()
#
#     ax.fill_between(
#         x,
#         lower,
#         upper,
#         color=color,
#         alpha=SHADE_ALPHA,
#         linewidth=0,
#     )
#
#
# for policy, p_df in static_summary.groupby("static_policy"):
#
#     policy_display = POLICY_LABELS.get(policy, policy)
#
#     costs = sorted(p_df["static_cost"].dropna().unique())
#     errors = sorted(p_df["static_error"].dropna().unique())
#
#     if not costs:
#         continue
#
#     fig, axes = plt.subplots(
#         1,
#         len(costs),
#         figsize=(8.5 * len(costs), 5),
#         sharey=True,
#     )
#
#     fig.patch.set_facecolor(FIG_BG_COLOR)
#
#     if len(costs) == 1:
#         axes = [axes]
#
#     for ax, cost in zip(axes, costs):
#         ax.set_facecolor(AX_BG_COLOR)
#
#         c_df = p_df[p_df["static_cost"] == cost]
#
#         for error in errors:
#             g = c_df[c_df["static_error"] == error]
#
#             if g.empty:
#                 continue
#
#             plot_mean_with_std_band(
#                 ax=ax,
#                 data=g,
#                 label=f" {policy_display}, E={error:.3f}",
#                 linestyle="-",
#                 marker="o",
#             )
#
#         for baseline_name, b in baseline_summary.groupby("baseline_mode"):
#             baseline_display = BASELINE_LABELS.get(baseline_name, baseline_name)
#
#             plot_mean_with_std_band(
#                 ax=ax,
#                 data=b,
#                 label=f"Baseline: {baseline_display}",
#                 linestyle="--",
#                 marker="x",
#             )
#
#         # ax.set_title(f"C={cost:.3f} (Mean ± STD shaded)")
#         ax.set_xlabel("Arrival Rate (req/s)", fontsize=16)
#         ax.set_ylim(bottom=0)
#         ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_INTERVAL))
#         ax.grid(True, alpha=0.3)
#         ax.tick_params(axis="both", labelsize=14)
#
#     axes[0].set_ylabel("Average Latency (ms)", fontsize=16)
#
#     legend = axes[-1].legend(fontsize=10, loc="upper left")
#     legend.get_frame().set_facecolor(LEGEND_BG_COLOR)
#     legend.get_frame().set_alpha(0.9)
#
#     plt.tight_layout()
#
#     safe_policy_name = policy_display.lower().replace(" ", "_").replace("-", "_")
#     # plt.savefig(
#     #     f"{OUT_DIR}/{safe_policy_name}_optimized_vs_baselines_mean_std_shaded.png",
#     #     dpi=300,
#     #     bbox_inches="tight",
#     #     facecolor=fig.get_facecolor(),
#     # )
#
#     out_base = f"{OUT_DIR}/{safe_policy_name}_optimized_vs_baselines_mean_std_shaded"
#
#     plt.savefig(
#         f"{out_base}.png",
#         dpi=300,
#         bbox_inches="tight",
#         facecolor="white",
#     )
#
#     plt.savefig(
#         f"{out_base}.pdf",
#         bbox_inches="tight",
#         facecolor="white",
#     )
#
#     plt.show()
#
# print(f"Saved plots to {OUT_DIR}")


import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv"
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv"
CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv"

OUT_DIR = "policy_latency_plots_with_baselines"
os.makedirs(OUT_DIR, exist_ok=True)

POLICY_LABELS = {
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "sequential": "Seq. Cascade",
    "spec_b": "Cloud Speculation",
}

# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest to largest model",
#     "random_to_random": "Random to random",
#     "smallest_to_largest": "Smallest to largest",
#     "spec_b_smallest_to_largest": "Smallest model and largest model",
#     "spec_b_random_to_random": "Random model and random model",
#     "full_ensemble_edge": "Cost-Opt 3-Model Edge Ensemble",
#     "full_ensemble_cloud": "full cloud ensemble",
# }

BASELINE_LABELS = {
    "power2_smallest_to_largest": "2 smallest to largest model",
    "random_to_random": "Random to random",
    "smallest_to_largest": "Smallest to largest",
    "spec_b_smallest_to_largest": "Smallest model and largest model",
    "spec_b_random_to_random": "Random model and random model",
    "full_ensemble_edge": "Cost-Opt 3-Model Edge Ensemble",
    "full_ensemble_cloud": "Full Cloud Ensemble",

    "frugalml": "FrugalML",
}

# Same E value gets same color across all policy plots
ERROR_COLORS = {
    0.090: "#1f77b4",  # blue
    0.100: "#ff7f0e",  # orange
    0.120: "#2ca02c",  # green
    0.140: "#8c564b",
}

# Baselines use red and purple family colors
# BASELINE_COLORS = {
#     "power2_smallest_to_largest": "#9467bd",   # red
#     "random_to_random": "#d62728",             # purple
#     "smallest_to_largest": "#9467bd",          # red
#     "spec_b_smallest_to_largest": "#9467bd",   # red
#     "spec_b_random_to_random": "#d62728",      # purple
#     "full_ensemble_edge": "#d62728",           # purple
#     "full_ensemble_cloud": "#d62728",          # purple
# }

BASELINE_COLORS = {
    "power2_smallest_to_largest": "#9467bd",
    "random_to_random": "#d62728",
    "smallest_to_largest": "#9467bd",
    "spec_b_smallest_to_largest": "#9467bd",
    "spec_b_random_to_random": "#d62728",
    "full_ensemble_edge": "#d62728",
    "full_ensemble_cloud": "#d62728",

    "frugalml": "#17becf",      # Black
}

FIG_BG_COLOR = "white"
AX_BG_COLOR = "white"
LEGEND_BG_COLOR = "white"

SHADE_ALPHA = 0.15
Y_TICK_INTERVAL = 500

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

numeric_cols = ["IAR", "RespTime", "e2e_time_ms", "static_cost", "static_error"]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["IAR", "RespTime"])

# df["latency_ms"] = df["RespTime"] * 1000.0
df["latency_ms"] = (df["RespTime"] * 1000.0 - 150).clip(lower=0)
df["arrival_rate_rps"] = 1000.0 / df["IAR"]

df["is_static"] = df["static_config_enabled"].astype(str).str.lower() == "true"
df["baseline_mode"] = df["baseline_mode"].astype(str)
df["static_policy"] = df["static_policy"].astype(str)

static_df = df[df["is_static"]].copy()
baseline_df = df[
    (~df["is_static"]) & (df["baseline_mode"].str.lower() != "none")
].copy()

static_summary = (
    static_df.groupby(
        ["static_policy", "static_cost", "static_error", "arrival_rate_rps"],
        as_index=False,
    )
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        std_latency_ms=("latency_ms", "std"),
        n=("latency_ms", "count"),
    )
)

baseline_summary = (
    baseline_df.groupby(
        ["baseline_mode", "effective_cascade_mode", "arrival_rate_rps"],
        as_index=False,
    )
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        std_latency_ms=("latency_ms", "std"),
        n=("latency_ms", "count"),
    )
)

static_summary["std_latency_ms"] = static_summary["std_latency_ms"].fillna(0)
baseline_summary["std_latency_ms"] = baseline_summary["std_latency_ms"].fillna(0)


def plot_mean_with_std_band(
    ax,
    data,
    label,
    linestyle="-",
    marker="o",
    linewidth=2,
    color=None,
):
    data = data.sort_values("arrival_rate_rps")

    x = data["arrival_rate_rps"]
    y = data["mean_latency_ms"]
    std = data["std_latency_ms"]

    lower = (y - std).clip(lower=0)
    upper = y + std

    line = ax.plot(
        x,
        y,
        linestyle=linestyle,
        marker=marker,
        linewidth=linewidth,
        markersize=6,
        label=label,
        color=color,
    )[0]

    line_color = line.get_color()

    ax.fill_between(
        x,
        lower,
        upper,
        color=line_color,
        alpha=SHADE_ALPHA,
        linewidth=0,
    )


for policy, p_df in static_summary.groupby("static_policy"):
    policy_display = POLICY_LABELS.get(policy, policy)

    costs = sorted(p_df["static_cost"].dropna().unique())
    errors = sorted(p_df["static_error"].dropna().unique())

    if not costs:
        continue

    fig, axes = plt.subplots(
        1,
        len(costs),
        figsize=(8.5 * len(costs), 5),
        sharey=True,
    )

    fig.patch.set_facecolor(FIG_BG_COLOR)

    if len(costs) == 1:
        axes = [axes]

    for ax, cost in zip(axes, costs):
        ax.set_facecolor(AX_BG_COLOR)

        c_df = p_df[p_df["static_cost"] == cost]

        for error in errors:
            g = c_df[c_df["static_error"] == error]

            if g.empty:
                continue

            error_key = round(float(error), 3)

            plot_mean_with_std_band(
                ax=ax,
                data=g,
                label=f"{policy_display}, E={error:.3f}",
                linestyle="-",
                marker="o",
                color=ERROR_COLORS.get(error_key),
            )

        for baseline_name, b in baseline_summary.groupby("baseline_mode"):
            baseline_display = BASELINE_LABELS.get(baseline_name, baseline_name)

            baseline_color = BASELINE_COLORS.get(baseline_name, "#d62728")

            plot_mean_with_std_band(
                ax=ax,
                data=b,
                label=f"{baseline_display}",
                linestyle="--",
                marker="x",
                linewidth=2,
                color=baseline_color,
            )

        ax.set_xlabel("Arrival Rate (req/s)", fontsize=18)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_locator(MultipleLocator(Y_TICK_INTERVAL))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=16)

    axes[0].set_ylabel("Average Latency (ms)", fontsize=18)

    legend = axes[-1].legend(fontsize=13.3, loc="upper left")
    legend.get_frame().set_facecolor(LEGEND_BG_COLOR)
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    safe_policy_name = policy_display.lower().replace(" ", "_").replace("-", "_")
    out_base = f"{OUT_DIR}/{safe_policy_name}_optimized_vs_baselines_mean_std_shaded"

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

print(f"Saved plots to {OUT_DIR}")
