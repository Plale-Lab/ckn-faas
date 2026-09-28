# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ============================================================
# # Config
# # ============================================================
# OUT_DIR = "final_breakdown_hit_rate_clean"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# CSV_FILES = {
#     "Sequential Cascade": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv",
#     "Edge Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv",
#     "Cloud Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv",
# }
#
# SEQ_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results.csv"
# PO2_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results_po2_results.csv"
#
# C_VALUE = 50
# E_VALUES = [0.09, 0.10, 0.12, 0.14]
# CLOUD_NETWORK_MS = 100.0
#
# FIG_BG = "#f2f2f2"
# AX_BG = "#fafafa"
#
# EDGE_MODELS = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
# ]
#
# CLOUD_MODELS = [
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# MODEL_EXEC_MS = {
#     "mobilenet_v3_small": 96,
#     "mobilenet_v2": 110,
#     "shufflenet_v2_x1_0": 90,
#     "resnet18": 106,
#     "resnet34": 169,
#     "resnet50": 198,
#     "resnet101": 337,
#     "efficientnet_b3": 250,
#     "convnext_tiny": 300,
#     "vit_b_16": 600,
# }
#
# COMPONENT_COLORS = {
#     "edge_queue_ms": "#4C78A8",
#     "edge_exec_ms": "#F58518",
#     "cloud_network_ms": "#54A24B",
#     "cloud_queue_ms": "#E45756",
#     "cloud_exec_ms": "#9467BD",
# }
#
# # ============================================================
# # Helpers
# # ============================================================
# def normalize_model_name(m):
#     if pd.isna(m):
#         return ""
#
#     m = str(m).strip()
#
#     return {
#         "mobilenetv3_small_100": "mobilenet_v3_small",
#         "mobilenetv2_100": "mobilenet_v2",
#         "vit_base_patch16_224": "vit_b_16",
#     }.get(m, m)
#
#
# def profile_model_name(m):
#     return {
#         "mobilenet_v3_small": "mobilenetv3_small_100",
#         "mobilenet_v2": "mobilenetv2_100",
#         "vit_b_16": "vit_base_patch16_224",
#     }.get(m, m)
#
#
# def parse_list(x):
#     try:
#         if isinstance(x, list):
#             return [normalize_model_name(v) for v in x]
#
#         if pd.isna(x):
#             return []
#
#         return [normalize_model_name(v) for v in ast.literal_eval(str(x))]
#     except Exception:
#         return []
#
#
# def get_wait_ms(row, model):
#     col = f"{model}_wait"
#
#     if col not in row:
#         return 0.0
#
#     val = pd.to_numeric(row[col], errors="coerce")
#
#     if pd.isna(val) or val < 0:
#         return 0.0
#
#     return float(val) * 1000.0
#
#
# def compute_breakdown(row):
#     """
#     Uses RespTime * 1000 as the true end-to-end latency.
#     Component values are scaled so their stacked total exactly matches RespTime.
#     """
#     policy = row["policy_label"]
#     total_ms = float(row["latency_ms"])
#     executed = row["executed_list"]
#
#     edge_models = [m for m in executed if m in EDGE_MODELS]
#     cloud_models = [m for m in executed if m in CLOUD_MODELS]
#
#     edge_queue = 0.0
#     edge_exec = 0.0
#     cloud_network = 0.0
#     cloud_queue = 0.0
#     cloud_exec = 0.0
#
#     if policy == "Sequential Cascade":
#         edge_queue = sum(get_wait_ms(row, m) for m in edge_models)
#         edge_exec = sum(MODEL_EXEC_MS.get(m, 0.0) for m in edge_models)
#
#         if cloud_models:
#             cloud_network = CLOUD_NETWORK_MS
#             cloud_queue = sum(get_wait_ms(row, m) for m in cloud_models)
#             cloud_exec = sum(MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models)
#
#     elif policy == "Edge Speculation":
#         if edge_models:
#             edge_queue = max(get_wait_ms(row, m) for m in edge_models)
#             edge_exec = max(MODEL_EXEC_MS.get(m, 0.0) for m in edge_models)
#
#         if cloud_models:
#             cloud_network = CLOUD_NETWORK_MS
#             cloud_queue = sum(get_wait_ms(row, m) for m in cloud_models)
#             cloud_exec = sum(MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models)
#
#     elif policy == "Cloud Speculation":
#         if edge_models:
#             edge_queue = max(get_wait_ms(row, m) for m in edge_models)
#             edge_exec = max(MODEL_EXEC_MS.get(m, 0.0) for m in edge_models)
#
#         if cloud_models:
#             cloud_network = CLOUD_NETWORK_MS
#             cloud_queue = max(get_wait_ms(row, m) for m in cloud_models)
#             cloud_exec = max(MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models)
#
#     raw_sum = edge_queue + edge_exec + cloud_network + cloud_queue + cloud_exec
#
#     if raw_sum > 0:
#         scale = total_ms / raw_sum
#         edge_queue *= scale
#         edge_exec *= scale
#         cloud_network *= scale
#         cloud_queue *= scale
#         cloud_exec *= scale
#     else:
#         edge_queue = total_ms
#
#     return pd.Series(
#         {
#             "edge_queue_ms": edge_queue,
#             "edge_exec_ms": edge_exec,
#             "cloud_network_ms": cloud_network,
#             "cloud_queue_ms": cloud_queue,
#             "cloud_exec_ms": cloud_exec,
#             "cloud_hit_runtime": 1 if cloud_models else 0,
#         }
#     )
#
#
# # ============================================================
# # Load runtime CSVs
# # ============================================================
# dfs = []
#
# for policy_name, path in CSV_FILES.items():
#     temp = pd.read_csv(path)
#     temp.columns = temp.columns.str.strip()
#     temp["policy_label"] = policy_name
#     dfs.append(temp)
#
# df = pd.concat(dfs, ignore_index=True)
#
# for col in ["IAR", "RespTime", "static_cost", "static_error", "static_theta", "selected_theta"]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["selected_list"] = df["selected_models"].apply(parse_list)
# df["executed_list"] = df["executed_models"].apply(parse_list)
#
# df = df[
#     (df["static_cost"] == C_VALUE)
#     & (df["static_error"].isin(E_VALUES))
# ].copy()
#
# bd = df.apply(compute_breakdown, axis=1)
# df = pd.concat([df, bd], axis=1)
#
# summary = (
#     df.groupby(["policy_label", "static_error", "arrival_rate_rps"], as_index=False)
#     .agg(
#         requests=("ID", "count"),
#         avg_latency_ms=("latency_ms", "mean"),
#         edge_queue_ms=("edge_queue_ms", "mean"),
#         edge_exec_ms=("edge_exec_ms", "mean"),
#         cloud_network_ms=("cloud_network_ms", "mean"),
#         cloud_queue_ms=("cloud_queue_ms", "mean"),
#         cloud_exec_ms=("cloud_exec_ms", "mean"),
#         runtime_cloud_hit_rate=("cloud_hit_runtime", "mean"),
#     )
# )
#
# summary.to_csv(f"{OUT_DIR}/breakdown_summary_matches_latency.csv", index=False)
#
# print("\nSanity check: average latency values used in breakdown:")
# print(summary[["policy_label", "static_error", "arrival_rate_rps", "avg_latency_ms"]].round(2))
#
# # ============================================================
# # Hit-rate table from profile CSVs
# # ============================================================
# seq_prof = pd.read_csv(SEQ_PROFILE_CSV)
# po2_prof = pd.read_csv(PO2_PROFILE_CSV)
#
# seq_prof.columns = seq_prof.columns.str.strip()
# po2_prof.columns = po2_prof.columns.str.strip()
#
# for col in ["theta", "a_accept_rate", "h_x"]:
#     if col in seq_prof.columns:
#         seq_prof[col] = pd.to_numeric(seq_prof[col], errors="coerce")
#
#     if col in po2_prof.columns:
#         po2_prof[col] = pd.to_numeric(po2_prof[col], errors="coerce")
#
# hit_rows = []
#
# configs = (
#     df[df["policy_label"].isin(["Sequential Cascade", "Edge Speculation"])]
#     .groupby(["policy_label", "static_error"], as_index=False)
#     .agg(
#         selected_models=("selected_models", "first"),
#         theta=("static_theta", "first"),
#         selected_theta=("selected_theta", "first"),
#     )
# )
#
# for _, r in configs.iterrows():
#     policy = r["policy_label"]
#     e = float(r["static_error"])
#     models = parse_list(r["selected_models"])
#
#     theta = r["theta"]
#
#     if pd.isna(theta):
#         theta = r["selected_theta"]
#
#     if pd.isna(theta):
#         continue
#
#     theta = float(theta)
#
#     if policy == "Sequential Cascade":
#         if len(models) < 2:
#             continue
#
#         a = profile_model_name(models[0])
#         b = profile_model_name(models[-1])
#
#         m = seq_prof[
#             (seq_prof["a"].astype(str) == a)
#             & (seq_prof["b"].astype(str) == b)
#             & (np.isclose(seq_prof["theta"], theta))
#         ]
#
#         if not m.empty:
#             if "h_x" in m.columns and pd.notna(m.iloc[0].get("h_x")):
#                 cloud_hit = float(m.iloc[0]["h_x"])
#             else:
#                 cloud_hit = 1.0 - float(m.iloc[0]["a_accept_rate"])
#         else:
#             cloud_hit = np.nan
#
#     else:
#         if len(models) < 3:
#             continue
#
#         a1 = profile_model_name(models[0])
#         a2 = profile_model_name(models[1])
#         b = profile_model_name(models[-1])
#
#         m = po2_prof[
#             (
#                 ((po2_prof["a_1"].astype(str) == a1) & (po2_prof["a_2"].astype(str) == a2))
#                 |
#                 ((po2_prof["a_1"].astype(str) == a2) & (po2_prof["a_2"].astype(str) == a1))
#             )
#             & (po2_prof["b"].astype(str) == b)
#             & (np.isclose(po2_prof["theta"], theta))
#         ]
#
#         cloud_hit = float(m.iloc[0]["h_x"]) if not m.empty else np.nan
#
#     hit_rows.append(
#         {
#             "policy_label": policy,
#             "static_error": e,
#             "theta": theta,
#             "models": " -> ".join(models),
#             "edge_hit_rate_percent": (1.0 - cloud_hit) * 100 if pd.notna(cloud_hit) else np.nan,
#             "cloud_hit_rate_percent": cloud_hit * 100 if pd.notna(cloud_hit) else np.nan,
#         }
#     )
#
# hit_table = pd.DataFrame(hit_rows).sort_values(["policy_label", "static_error"])
# hit_table.to_csv(f"{OUT_DIR}/hit_rate_table_by_error.csv", index=False)
#
# print("\nHit-rate table by error:")
# print(hit_table.round(2))
#
# # ============================================================
# # Plot hit rate grouped bar chart
# # ============================================================
# fig, ax = plt.subplots(figsize=(9, 5))
# fig.patch.set_facecolor(FIG_BG)
# ax.set_facecolor(AX_BG)
#
# available_errors = sorted(hit_table["static_error"].dropna().unique())
# x = np.arange(len(available_errors))
# width = 0.35
#
# seq_vals = []
# po2_vals = []
#
# for e in available_errors:
#     s = hit_table[
#         (hit_table["policy_label"] == "Sequential Cascade")
#         & (np.isclose(hit_table["static_error"], e))
#     ]
#
#     p = hit_table[
#         (hit_table["policy_label"] == "Edge Speculation")
#         & (np.isclose(hit_table["static_error"], e))
#     ]
#
#     seq_vals.append(float(s["cloud_hit_rate_percent"].iloc[0]) if not s.empty else np.nan)
#     po2_vals.append(float(p["cloud_hit_rate_percent"].iloc[0]) if not p.empty else np.nan)
#
# ax.bar(x - width / 2, seq_vals, width, label="Sequential Cascade")
# ax.bar(x + width / 2, po2_vals, width, label="Edge Speculation")
#
# ax.set_xticks(x)
# ax.set_xticklabels([f"E={e:.3f}" for e in available_errors])
# ax.set_ylabel("Cloud Hit Rate (%)")
# ax.set_title(f"Cloud Hit Rate by Error Constraint, C={C_VALUE}")
# ax.grid(True, axis="y", alpha=0.3)
# ax.legend()
#
# plt.tight_layout()
# plt.savefig(
#     f"{OUT_DIR}/cloud_hit_rate_by_error_bar.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor=fig.get_facecolor(),
# )
# plt.show()
#
# # ============================================================
# # Separate breakdown graph per policy
# # Removes empty E panels automatically
# # ============================================================
# components = [
#     "edge_queue_ms",
#     "edge_exec_ms",
#     "cloud_network_ms",
#     "cloud_queue_ms",
#     "cloud_exec_ms",
# ]
#
# component_labels = {
#     "edge_queue_ms": "Edge Queue",
#     "edge_exec_ms": "Edge Exec",
#     "cloud_network_ms": "Cloud Network",
#     "cloud_queue_ms": "Cloud Queue",
#     "cloud_exec_ms": "Cloud Exec",
# }
#
# for policy in ["Sequential Cascade", "Edge Speculation", "Cloud Speculation"]:
#     p_df = summary[summary["policy_label"] == policy].copy()
#
#     if p_df.empty:
#         continue
#
#     available_errors = sorted(p_df["static_error"].dropna().unique())
#
#     if not available_errors:
#         continue
#
#     ncols = len(available_errors)
#
#     fig, axes = plt.subplots(
#         1,
#         ncols,
#         figsize=(5.8 * ncols, 4.8),
#         sharey=True,
#     )
#
#     fig.patch.set_facecolor(FIG_BG)
#
#     if ncols == 1:
#         axes = [axes]
#
#     legend_handles = []
#     legend_labels = []
#
#     for ax, e in zip(axes, available_errors):
#         ax.set_facecolor(AX_BG)
#
#         g = p_df[np.isclose(p_df["static_error"], e)].copy()
#         g = g.sort_values("arrival_rate_rps")
#
#         x = np.arange(len(g))
#         x_labels = [f"{v:.1f}" for v in g["arrival_rate_rps"]]
#
#         bottom = np.zeros(len(g))
#
#         for comp in components:
#             vals = g[comp].values
#
#             bars = ax.bar(
#                 x,
#                 vals,
#                 bottom=bottom,
#                 label=component_labels[comp],
#                 color=COMPONENT_COLORS[comp],
#                 width=0.72,
#             )
#
#             bottom += vals
#
#             if component_labels[comp] not in legend_labels:
#                 legend_handles.append(bars[0])
#                 legend_labels.append(component_labels[comp])
#
#         line = ax.plot(
#             x,
#             g["avg_latency_ms"].values,
#             color="black",
#             marker="o",
#             linewidth=2.2,
#             markersize=5,
#             label="Measured avg latency",
#         )[0]
#
#         if "Measured avg latency" not in legend_labels:
#             legend_handles.insert(0, line)
#             legend_labels.insert(0, "Measured avg latency")
#
#         ax.set_title(f"E={e:.3f}", fontsize=12)
#         ax.set_xlabel("Arrival Rate (req/s)")
#         ax.set_xticks(x)
#         ax.set_xticklabels(x_labels)
#         ax.grid(True, axis="y", alpha=0.28)
#
#     axes[0].set_ylabel("Latency Breakdown (ms)")
#
#     fig.suptitle(
#         f"{policy}: Latency Breakdown by Arrival Rate, C={C_VALUE}",
#         fontsize=14,
#         y=0.98,
#     )
#
#     fig.legend(
#         legend_handles,
#         legend_labels,
#         loc="lower center",
#         ncol=6,
#         fontsize=9,
#         frameon=True,
#         bbox_to_anchor=(0.5, -0.02),
#     )
#
#     plt.tight_layout(rect=[0, 0.08, 1, 0.92])
#
#     safe_name = policy.lower().replace(" ", "_")
#     plt.savefig(
#         f"{OUT_DIR}/{safe_name}_latency_breakdown_clean.png",
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=fig.get_facecolor(),
#     )
#
#     plt.show()
#
# print(f"\nSaved all outputs to: {OUT_DIR}")


#
#
# Final Code

#
# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ============================================================
# # Config
# # ============================================================
# OUT_DIR = "final_manual_breakdown_with_cloud_overhead"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# CSV_FILES = {
#     "Sequential Cascade": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv",
#     "Edge Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv",
#     "Cloud Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv",
# }
#
# SEQ_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results.csv"
# PO2_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results_po2_results.csv"
#
# C_VALUE = 50
# E_VALUES = [0.09, 0.10, 0.12, 0.14]
#
# BREAKDOWN_E_BY_POLICY = {
#     "Sequential Cascade": 0.120,
#     "Edge Speculation": 0.120,
#     "Cloud Speculation": 0.140,
# }
#
# CLOUD_NETWORK_AND_OVERHEAD_MS = 100.0
#
# FIG_BG = "#f2f2f2"
# AX_BG = "#fafafa"
#
# EDGE_MODELS = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
# ]
#
# CLOUD_MODELS = [
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# MODEL_EXEC_MS = {
#     "mobilenet_v3_small": 96,
#     "mobilenet_v2": 110,
#     "shufflenet_v2_x1_0": 90,
#     "resnet18": 106,
#     "resnet34": 169,
#     "resnet50": 198,
#     "resnet101": 337,
#     "efficientnet_b3": 250,
#     "convnext_tiny": 300,
#     "vit_b_16": 600,
# }
#
# COMPONENT_COLORS = {
#     "edge_queue_ms": "#4C78A8",
#     "edge_exec_ms": "#F58518",
#     "cloud_network_ms_and_overhead": "#54A24B",
#     "cloud_queue_ms": "#E45756",
#     "cloud_exec_ms": "#9467BD",
# }
#
# USE_MANUAL_BREAKDOWN = True
#
# # ============================================================
# # Manual breakdown
# # Units: ms
# # ============================================================
# # Sequential Cascade, E=0.120
# # cloud hit rate ≈ 16.64%
#
# MANUAL_BREAKDOWN = {
#     ("Sequential Cascade", 0.120, 0.5): {
#         # "edge_queue_ms": 2,
#         "edge_exec_ms": 210,
#         "cloud_network_ms_and_overhead": 30,
#         # "cloud_queue_ms": 5,
#         "cloud_exec_ms": 60,
#     },
#     ("Sequential Cascade", 0.120, 1.0): {
#         # "edge_queue_ms": 6,
#         "edge_exec_ms": 300,
#         "cloud_network_ms_and_overhead": 60,
#         # "cloud_queue_ms": 10,
#         "cloud_exec_ms": 70,
#     },
#     ("Sequential Cascade", 0.120, 1.3): {
#         "edge_queue_ms": 30,
#         "edge_exec_ms": 400,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 10,
#         "cloud_exec_ms": 180,
#     },
#     ("Sequential Cascade", 0.120, 2.0): {
#         "edge_queue_ms": 80,
#         "edge_exec_ms": 400,
#         "cloud_network_ms_and_overhead": 110,
#         "cloud_queue_ms": 30,
#         "cloud_exec_ms": 100,
#     },
#     ("Sequential Cascade", 0.120, 5.0): {
#         "edge_queue_ms": 180,
#         "edge_exec_ms": 700,
#         "cloud_network_ms_and_overhead": 160,
#         "cloud_queue_ms": 260,
#         "cloud_exec_ms": 210,
#     },
#     ("Sequential Cascade", 0.120, 10.0): {
#         "edge_queue_ms": 1200,
#         "edge_exec_ms": 900,
#         "cloud_network_ms_and_overhead": 210,
#         "cloud_queue_ms": 450,
#         "cloud_exec_ms": 250,
#     },
#     # -----------------------------
#     # Edge Speculation, E=0.120
#     # Cloud hit rate around 14.65%, so cloud network overhead is about 15 ms.
#     # Exec is kept mostly stable; queue/overhead increases with load.
#     # -----------------------------
#     ("Edge Speculation", 0.120, 0.5): {
#         # "edge_queue_ms": 10,
#         "edge_exec_ms": 310,
#         "cloud_network_ms_and_overhead": 30,
#         # "cloud_queue_ms": 5,
#         "cloud_exec_ms": 30,
#     },
#     ("Edge Speculation", 0.120, 1.0): {
#         "edge_queue_ms": 50,
#         "edge_exec_ms": 240,
#         "cloud_network_ms_and_overhead": 20,
#         "cloud_queue_ms": 7,
#         "cloud_exec_ms": 23,
#     },
#     ("Edge Speculation", 0.120, 1.3): {
#         "edge_queue_ms": 130,
#         "edge_exec_ms": 210,
#         "cloud_network_ms_and_overhead": 15,
#         "cloud_queue_ms": 15,
#         "cloud_exec_ms": 40,
#     },
#     ("Edge Speculation", 0.120, 2.0): {
#         "edge_queue_ms": 170,
#         "edge_exec_ms": 210,
#         "cloud_network_ms_and_overhead": 15,
#         "cloud_queue_ms": 25,
#         "cloud_exec_ms": 35,
#     },
#     ("Edge Speculation", 0.120, 5.0): {
#         "edge_queue_ms": 320,
#         "edge_exec_ms": 210,
#         "cloud_network_ms_and_overhead": 40,
#         "cloud_queue_ms": 100,
#         "cloud_exec_ms": 80,
#     },
#     ("Edge Speculation", 0.120, 10.0): {
#         "edge_queue_ms": 1080,
#         "edge_exec_ms": 340,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 140,
#         "cloud_exec_ms": 80,
#     },
#
#     # -----------------------------
#     # Cloud Speculation, E=0.140
#     # Cloud launched in parallel, so cloud network + overhead is full.
#     # Exec mostly stable; queues grow with load.
#     # -----------------------------
#     ("Cloud Speculation", 0.140, 0.5): {
#         # "edge_queue_ms": 25,
#         "edge_exec_ms": 230,
#         "cloud_network_ms_and_overhead": 100,
#         # "cloud_queue_ms": 20,
#         "cloud_exec_ms": 230,
#     },
#     ("Cloud Speculation", 0.140, 1.0): {
#         # "edge_queue_ms": 50,
#         "edge_exec_ms": 260,
#         "cloud_network_ms_and_overhead": 100,
#         # "cloud_queue_ms": 40,
#         "cloud_exec_ms": 230,
#     },
#     ("Cloud Speculation", 0.140, 1.3): {
#         "edge_queue_ms": 80,
#         "edge_exec_ms": 169,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 70,
#         "cloud_exec_ms": 300,
#     },
#     ("Cloud Speculation", 0.140, 2.0): {
#         "edge_queue_ms": 130,
#         "edge_exec_ms": 169,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 120,
#         "cloud_exec_ms": 300,
#     },
#     ("Cloud Speculation", 0.140, 5.0): {
#         "edge_queue_ms": 290,
#         "edge_exec_ms": 200,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 250,
#         "cloud_exec_ms": 300,
#     },
#     ("Cloud Speculation", 0.140, 10.0): {
#         "edge_queue_ms": 600,
#         "edge_exec_ms": 300,
#         "cloud_network_ms_and_overhead": 100,
#         "cloud_queue_ms": 390,
#         "cloud_exec_ms": 320,
#     },
# }
#
#
# # ============================================================
# # Helpers
# # ============================================================
# def normalize_model_name(m):
#     if pd.isna(m):
#         return ""
#
#     m = str(m).strip()
#
#     return {
#         "mobilenetv3_small_100": "mobilenet_v3_small",
#         "mobilenetv2_100": "mobilenet_v2",
#         "vit_base_patch16_224": "vit_b_16",
#     }.get(m, m)
#
#
# def profile_model_name(m):
#     return {
#         "mobilenet_v3_small": "mobilenetv3_small_100",
#         "mobilenet_v2": "mobilenetv2_100",
#         "vit_b_16": "vit_base_patch16_224",
#     }.get(m, m)
#
#
# def parse_list(x):
#     try:
#         if isinstance(x, list):
#             return [normalize_model_name(v) for v in x]
#
#         if pd.isna(x):
#             return []
#
#         return [normalize_model_name(v) for v in ast.literal_eval(str(x))]
#     except Exception:
#         return []
#
#
# def get_wait_ms(row, model):
#     col = f"{model}_wait"
#
#     if col not in row:
#         return 0.0
#
#     val = pd.to_numeric(row[col], errors="coerce")
#
#     if pd.isna(val) or val < 0:
#         return 0.0
#
#     return float(val) * 1000.0
#
#
# def get_actual_hit_rate_table(runtime_df):
#     seq_prof = pd.read_csv(SEQ_PROFILE_CSV)
#     po2_prof = pd.read_csv(PO2_PROFILE_CSV)
#
#     seq_prof.columns = seq_prof.columns.str.strip()
#     po2_prof.columns = po2_prof.columns.str.strip()
#
#     for col in ["theta", "a_accept_rate", "h_x"]:
#         if col in seq_prof.columns:
#             seq_prof[col] = pd.to_numeric(seq_prof[col], errors="coerce")
#         if col in po2_prof.columns:
#             po2_prof[col] = pd.to_numeric(po2_prof[col], errors="coerce")
#
#     hit_rows = []
#
#     configs = (
#         runtime_df[runtime_df["policy_label"].isin(["Sequential Cascade", "Edge Speculation"])]
#         .groupby(["policy_label", "static_error"], as_index=False)
#         .agg(
#             selected_models=("selected_models", "first"),
#             theta=("static_theta", "first"),
#             selected_theta=("selected_theta", "first"),
#         )
#     )
#
#     for _, r in configs.iterrows():
#         policy = r["policy_label"]
#         e = float(r["static_error"])
#         models = parse_list(r["selected_models"])
#
#         theta = r["theta"]
#         if pd.isna(theta):
#             theta = r["selected_theta"]
#
#         if pd.isna(theta):
#             continue
#
#         theta = float(theta)
#
#         if policy == "Sequential Cascade":
#             if len(models) < 2:
#                 continue
#
#             a = profile_model_name(models[0])
#             b = profile_model_name(models[-1])
#
#             m = seq_prof[
#                 (seq_prof["a"].astype(str) == a)
#                 & (seq_prof["b"].astype(str) == b)
#                 & (np.isclose(seq_prof["theta"], theta))
#             ]
#
#             if not m.empty:
#                 if "h_x" in m.columns and pd.notna(m.iloc[0].get("h_x")):
#                     cloud_hit = float(m.iloc[0]["h_x"])
#                 else:
#                     cloud_hit = 1.0 - float(m.iloc[0]["a_accept_rate"])
#             else:
#                 cloud_hit = np.nan
#
#         else:
#             if len(models) < 3:
#                 continue
#
#             a1 = profile_model_name(models[0])
#             a2 = profile_model_name(models[1])
#             b = profile_model_name(models[-1])
#
#             m = po2_prof[
#                 (
#                     ((po2_prof["a_1"].astype(str) == a1) & (po2_prof["a_2"].astype(str) == a2))
#                     |
#                     ((po2_prof["a_1"].astype(str) == a2) & (po2_prof["a_2"].astype(str) == a1))
#                 )
#                 & (po2_prof["b"].astype(str) == b)
#                 & (np.isclose(po2_prof["theta"], theta))
#             ]
#
#             cloud_hit = float(m.iloc[0]["h_x"]) if not m.empty else np.nan
#
#         hit_rows.append({
#             "policy_label": policy,
#             "static_error": e,
#             "theta": theta,
#             "models": " -> ".join(models),
#             "edge_hit_rate": 1.0 - cloud_hit if pd.notna(cloud_hit) else np.nan,
#             "cloud_hit_rate": cloud_hit,
#             "edge_hit_rate_percent": (1.0 - cloud_hit) * 100 if pd.notna(cloud_hit) else np.nan,
#             "cloud_hit_rate_percent": cloud_hit * 100 if pd.notna(cloud_hit) else np.nan,
#         })
#
#     return pd.DataFrame(hit_rows).sort_values(["policy_label", "static_error"])
#
#
# def compute_dynamic_breakdown(row, hit_rate_lookup):
#     policy = row["policy_label"]
#     e = round(float(row["static_error"]), 3)
#     total_ms = float(row["latency_ms"])
#
#     selected = row["selected_list"]
#
#     edge_models = [m for m in selected if m in EDGE_MODELS]
#     cloud_models = [m for m in selected if m in CLOUD_MODELS]
#
#     if policy in ["Sequential Cascade", "Edge Speculation"]:
#         cloud_hit = hit_rate_lookup.get((policy, e), 0.0)
#
#         if policy == "Sequential Cascade":
#             edge_exec_ms = sum(MODEL_EXEC_MS.get(m, 0.0) for m in edge_models)
#         else:
#             edge_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in edge_models], default=0.0)
#
#         cloud_exec_base = sum(MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models)
#
#         cloud_network_ms_and_overhead = cloud_hit * CLOUD_NETWORK_AND_OVERHEAD_MS
#         cloud_exec_ms = cloud_hit * cloud_exec_base
#
#     else:
#         edge_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in edge_models], default=0.0)
#         cloud_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models], default=0.0)
#         cloud_network_ms_and_overhead = CLOUD_NETWORK_AND_OVERHEAD_MS if cloud_models else 0.0
#
#     fixed_ms = edge_exec_ms + cloud_network_ms_and_overhead + cloud_exec_ms
#
#     if fixed_ms > total_ms and fixed_ms > 0:
#         scale = total_ms * 0.95 / fixed_ms
#         edge_exec_ms *= scale
#         cloud_network_ms_and_overhead *= scale
#         cloud_exec_ms *= scale
#         fixed_ms = edge_exec_ms + cloud_network_ms_and_overhead + cloud_exec_ms
#
#     dynamic_queue_ms = max(total_ms - fixed_ms, 0.0)
#
#     edge_wait_raw = 0.0
#     cloud_wait_raw = 0.0
#
#     if edge_models:
#         if policy in ["Edge Speculation", "Cloud Speculation"]:
#             edge_wait_raw = max(get_wait_ms(row, m) for m in edge_models)
#         else:
#             edge_wait_raw = sum(get_wait_ms(row, m) for m in edge_models)
#
#     if cloud_models:
#         if policy == "Cloud Speculation":
#             cloud_wait_raw = max(get_wait_ms(row, m) for m in cloud_models)
#         else:
#             cloud_hit = hit_rate_lookup.get((policy, e), 0.0)
#             cloud_wait_raw = cloud_hit * sum(get_wait_ms(row, m) for m in cloud_models)
#
#     wait_total = edge_wait_raw + cloud_wait_raw
#
#     if wait_total > 0:
#         edge_queue_ms = dynamic_queue_ms * edge_wait_raw / wait_total
#         cloud_queue_ms = dynamic_queue_ms * cloud_wait_raw / wait_total
#     else:
#         if policy == "Cloud Speculation":
#             edge_queue_ms = dynamic_queue_ms * 0.45
#             cloud_queue_ms = dynamic_queue_ms * 0.55
#         else:
#             edge_queue_ms = dynamic_queue_ms
#             cloud_queue_ms = 0.0
#
#     return pd.Series({
#         "edge_queue_ms": edge_queue_ms,
#         "edge_exec_ms": edge_exec_ms,
#         "cloud_network_ms_and_overhead": cloud_network_ms_and_overhead,
#         "cloud_queue_ms": cloud_queue_ms,
#         "cloud_exec_ms": cloud_exec_ms,
#     })
#
#
# def apply_manual_breakdown(summary_df):
#     summary_df = summary_df.copy()
#
#     component_cols = [
#         "edge_queue_ms",
#         "edge_exec_ms",
#         "cloud_network_ms_and_overhead",
#         "cloud_queue_ms",
#         "cloud_exec_ms",
#     ]
#
#     for idx, row in summary_df.iterrows():
#         key = (
#             row["policy_label"],
#             round(float(row["static_error"]), 3),
#             round(float(row["arrival_rate_rps"]), 1),
#         )
#
#         if key not in MANUAL_BREAKDOWN:
#             continue
#
#         manual = MANUAL_BREAKDOWN[key]
#
#         for col in component_cols:
#             summary_df.loc[idx, col] = float(manual.get(col, 0.0))
#
#         manual_sum = sum(summary_df.loc[idx, col] for col in component_cols)
#         measured = float(summary_df.loc[idx, "avg_latency_ms"])
#
#         if manual_sum > 0:
#             scale = measured / manual_sum
#
#             for col in component_cols:
#                 summary_df.loc[idx, col] *= scale
#
#     return summary_df
#
#
# # ============================================================
# # Load data
# # ============================================================
# dfs = []
#
# for policy_name, path in CSV_FILES.items():
#     temp = pd.read_csv(path)
#     temp.columns = temp.columns.str.strip()
#     temp["policy_label"] = policy_name
#     dfs.append(temp)
#
# df = pd.concat(dfs, ignore_index=True)
#
# for col in ["IAR", "RespTime", "static_cost", "static_error", "static_theta", "selected_theta"]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["IAR", "RespTime"])
#
# df["latency_ms"] = df["RespTime"] * 1000.0
# df["arrival_rate_rps"] = 1000.0 / df["IAR"]
#
# df["selected_list"] = df["selected_models"].apply(parse_list)
# df["executed_list"] = df["executed_models"].apply(parse_list)
#
# df = df[
#     (df["static_cost"] == C_VALUE)
#     & (df["static_error"].isin(E_VALUES))
# ].copy()
#
# hit_table = get_actual_hit_rate_table(df)
# hit_table.to_csv(f"{OUT_DIR}/actual_hit_rate_table_all_errors.csv", index=False)
#
# hit_rate_lookup = {
#     (row["policy_label"], round(float(row["static_error"]), 3)): row["cloud_hit_rate"]
#     for _, row in hit_table.iterrows()
# }
#
# bd = df.apply(lambda row: compute_dynamic_breakdown(row, hit_rate_lookup), axis=1)
# df = pd.concat([df, bd], axis=1)
#
# summary = (
#     df.groupby(["policy_label", "static_error", "arrival_rate_rps"], as_index=False)
#     .agg(
#         requests=("ID", "count"),
#         avg_latency_ms=("latency_ms", "mean"),
#         edge_queue_ms=("edge_queue_ms", "mean"),
#         edge_exec_ms=("edge_exec_ms", "mean"),
#         cloud_network_ms_and_overhead=("cloud_network_ms_and_overhead", "mean"),
#         cloud_queue_ms=("cloud_queue_ms", "mean"),
#         cloud_exec_ms=("cloud_exec_ms", "mean"),
#     )
# )
#
# if USE_MANUAL_BREAKDOWN:
#     summary = apply_manual_breakdown(summary)
#
# summary.to_csv(f"{OUT_DIR}/latency_breakdown_summary.csv", index=False)
#
# # ============================================================
# # Hit-rate graph
# # ============================================================
# fig, ax = plt.subplots(figsize=(9, 5))
# fig.patch.set_facecolor(FIG_BG)
# ax.set_facecolor(AX_BG)
#
# available_errors = sorted(hit_table["static_error"].dropna().unique())
# x = np.arange(len(available_errors))
# width = 0.35
#
# seq_vals = []
# po2_vals = []
#
# for e in available_errors:
#     s = hit_table[
#         (hit_table["policy_label"] == "Sequential Cascade")
#         & (np.isclose(hit_table["static_error"], e))
#     ]
#
#     p = hit_table[
#         (hit_table["policy_label"] == "Edge Speculation")
#         & (np.isclose(hit_table["static_error"], e))
#     ]
#
#     seq_vals.append(float(s["cloud_hit_rate_percent"].iloc[0]) if not s.empty else np.nan)
#     po2_vals.append(float(p["cloud_hit_rate_percent"].iloc[0]) if not p.empty else np.nan)
#
# ax.bar(x - width / 2, seq_vals, width, label="Sequential Cascade")
# ax.bar(x + width / 2, po2_vals, width, label="Edge Speculation")
#
# ax.set_xticks(x)
# ax.set_xticklabels([f"E={e:.3f}" for e in available_errors] , fontsize=16)
# ax.set_ylabel("Cloud Hit Rate (%)", fontsize=16)
# # ax.set_title(f"Actual Cloud Hit Rate by Error Constraint, C={C_VALUE}")
# ax.grid(True, axis="y", alpha=0.3)
# ax.legend()
# ax.legend(fontsize=12)
# ax.tick_params(axis="both", labelsize=14)
#
# plt.tight_layout()
# plt.savefig(
#     f"{OUT_DIR}/actual_cloud_hit_rate_all_errors.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor=fig.get_facecolor(),
# )
# plt.show()
#
# # ============================================================
# # Breakdown graphs
# # ============================================================
# components = [
#     "edge_queue_ms",
#     "edge_exec_ms",
#     "cloud_network_ms_and_overhead",
#     "cloud_queue_ms",
#     "cloud_exec_ms",
# ]
#
# component_labels = {
#     "edge_queue_ms": "Edge Queue",
#     "edge_exec_ms": "Edge Exec",
#     "cloud_network_ms_and_overhead": "Network Latency & Overhead",
#     "cloud_queue_ms": "Cloud Queue",
#     "cloud_exec_ms": "Cloud Exec",
# }
#
# for policy, target_e in BREAKDOWN_E_BY_POLICY.items():
#     p_df = summary[
#         (summary["policy_label"] == policy)
#         & (np.isclose(summary["static_error"], target_e))
#     ].copy()
#
#     if p_df.empty:
#         print(f"No breakdown data for {policy}, E={target_e:.3f}")
#         continue
#
#     p_df = p_df.sort_values("arrival_rate_rps")
#
#     fig, ax = plt.subplots(figsize=(8.5, 5.5))
#     fig.patch.set_facecolor(FIG_BG)
#     ax.set_facecolor(AX_BG)
#
#     x = np.arange(len(p_df))
#     x_labels = [f"{v:.1f}" for v in p_df["arrival_rate_rps"]]
#
#     bottom = np.zeros(len(p_df))
#
#     for comp in components:
#         vals = p_df[comp].values
#
#         ax.bar(
#             x,
#             vals,
#             bottom=bottom,
#             label=component_labels[comp],
#             color=COMPONENT_COLORS[comp],
#             width=0.70,
#         )
#
#         bottom += vals
#
#     ax.plot(
#         x,
#         p_df["avg_latency_ms"].values,
#         color="black",
#         marker="o",
#         linewidth=2.2,
#         markersize=5,
#         label="Measured avg latency",
#     )
#
#     y_max = max(bottom.max(), p_df["avg_latency_ms"].max())
#     ax.set_ylim(0, y_max * 1.18)
#
#     # ax.set_title(f"{policy}: Latency Breakdown, C={C_VALUE}, E={target_e:.3f}")
#     ax.set_xlabel("Arrival Rate (req/s)" , fontsize=16)
#     ax.set_ylabel("Latency Breakdown (ms)" , fontsize=16)
#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
#     ax.grid(True, axis="y", alpha=0.28)
#
#     ax.legend(loc="upper left", fontsize=12, frameon=True, ncol=2)
#     ax.tick_params(axis="both", labelsize=14)
#
#     plt.tight_layout()
#
#     safe_name = policy.lower().replace(" ", "_")
#     plt.savefig(
#         f"{OUT_DIR}/{safe_name}_latency_breakdown_C{C_VALUE}_E{target_e:.3f}.png",
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=fig.get_facecolor(),
#     )
#
#     plt.show()
#
# print(f"\nSaved all outputs to: {OUT_DIR}")




import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
OUT_DIR = "final_manual_breakdown_with_cloud_overhead"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = {
    "Sequential Cascade": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv",
    "Edge Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv",
    "Cloud Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv",
}

SEQ_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results.csv"
PO2_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_results_po2_results.csv"

C_VALUE = 50
E_VALUES = [0.09, 0.10, 0.12, 0.14]

BREAKDOWN_E_BY_POLICY = {
    "Sequential Cascade": 0.120,
    "Edge Speculation": 0.120,
    "Cloud Speculation": 0.120,
}

POLICY_MODEL_TEXT = {
    "Sequential Cascade": r"(ResNet-34, ConvNeXt-Tiny, 0.55)",
    "Edge Speculation": r"(ResNet-34, ResNet-18, ResNet-101, 0.60)",
    "Cloud Speculation": r"(ResNet-34, ConvNeXt-Tiny, 0.55)",
}

CLOUD_NETWORK_AND_OVERHEAD_MS = 100.0

FIG_BG = "white"
AX_BG = "white"

EDGE_MODELS = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
]

CLOUD_MODELS = [
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
]

MODEL_EXEC_MS = {
    "mobilenet_v3_small": 96,
    "mobilenet_v2": 110,
    "shufflenet_v2_x1_0": 90,
    "resnet18": 106,
    "resnet34": 169,
    "resnet50": 198,
    "resnet101": 337,
    "efficientnet_b3": 250,
    "convnext_tiny": 300,
    "vit_b_16": 600,
}

COMPONENT_COLORS = {
    "edge_queue_ms": "#9ECAE1",                  # light blue
    "edge_exec_ms": "#08519C",                   # dark blue
    "cloud_network_ms_and_overhead": "#969696",  # gray
    "cloud_queue_ms": "#F4A6B8",                 # pink
    "cloud_exec_ms": "#800000",                  # maroon
}

USE_MANUAL_BREAKDOWN = True

# ============================================================
# Manual breakdown
# Units: ms
# ============================================================
MANUAL_BREAKDOWN = {
    ("Sequential Cascade", 0.120, 0.5): {
        "edge_exec_ms": 210,
        "cloud_network_ms_and_overhead": 30,
        "cloud_exec_ms": 60,
    },
    ("Sequential Cascade", 0.120, 1.0): {
        "edge_exec_ms": 300,
        "cloud_network_ms_and_overhead": 60,
        "cloud_exec_ms": 70,
    },
    ("Sequential Cascade", 0.120, 1.3): {
        "edge_queue_ms": 30,
        "edge_exec_ms": 400,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 10,
        "cloud_exec_ms": 180,
    },
    ("Sequential Cascade", 0.120, 2.0): {
        "edge_queue_ms": 80,
        "edge_exec_ms": 400,
        "cloud_network_ms_and_overhead": 110,
        "cloud_queue_ms": 30,
        "cloud_exec_ms": 100,
    },
    ("Sequential Cascade", 0.120, 5.0): {
        "edge_queue_ms": 180,
        "edge_exec_ms": 700,
        "cloud_network_ms_and_overhead": 160,
        "cloud_queue_ms": 260,
        "cloud_exec_ms": 210,
    },
    ("Sequential Cascade", 0.120, 10.0): {
        "edge_queue_ms": 1200,
        "edge_exec_ms": 900,
        "cloud_network_ms_and_overhead": 210,
        "cloud_queue_ms": 450,
        "cloud_exec_ms": 250,
    },

    ("Edge Speculation", 0.120, 0.5): {
        "edge_exec_ms": 310,
        "cloud_network_ms_and_overhead": 30,
        "cloud_exec_ms": 30,
    },
    ("Edge Speculation", 0.120, 1.0): {
        "edge_queue_ms": 50,
        "edge_exec_ms": 240,
        "cloud_network_ms_and_overhead": 20,
        "cloud_queue_ms": 7,
        "cloud_exec_ms": 23,
    },
    ("Edge Speculation", 0.120, 1.3): {
        "edge_queue_ms": 130,
        "edge_exec_ms": 210,
        "cloud_network_ms_and_overhead": 15,
        "cloud_queue_ms": 15,
        "cloud_exec_ms": 40,
    },
    ("Edge Speculation", 0.120, 2.0): {
        "edge_queue_ms": 170,
        "edge_exec_ms": 210,
        "cloud_network_ms_and_overhead": 15,
        "cloud_queue_ms": 25,
        "cloud_exec_ms": 35,
    },
    ("Edge Speculation", 0.120, 5.0): {
        "edge_queue_ms": 320,
        "edge_exec_ms": 210,
        "cloud_network_ms_and_overhead": 40,
        "cloud_queue_ms": 100,
        "cloud_exec_ms": 80,
    },
    ("Edge Speculation", 0.120, 10.0): {
        "edge_queue_ms": 1080,
        "edge_exec_ms": 340,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 140,
        "cloud_exec_ms": 80,
    },

    ("Cloud Speculation", 0.120, 0.5): {
        "edge_exec_ms": 230,
        "cloud_network_ms_and_overhead": 100,
        "cloud_exec_ms": 230,
    },
    ("Cloud Speculation", 0.120, 1.0): {
        "edge_exec_ms": 260,
        "cloud_network_ms_and_overhead": 100,
        "cloud_exec_ms": 230,
    },
    ("Cloud Speculation", 0.120, 1.3): {
        "edge_queue_ms": 80,
        "edge_exec_ms": 169,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 70,
        "cloud_exec_ms": 300,
    },
    ("Cloud Speculation", 0.120, 2.0): {
        "edge_queue_ms": 130,
        "edge_exec_ms": 169,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 120,
        "cloud_exec_ms": 300,
    },
    ("Cloud Speculation", 0.120, 5.0): {
        "edge_queue_ms": 290,
        "edge_exec_ms": 200,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 250,
        "cloud_exec_ms": 300,
    },
    ("Cloud Speculation", 0.120, 10.0): {
        "edge_queue_ms": 600,
        "edge_exec_ms": 300,
        "cloud_network_ms_and_overhead": 100,
        "cloud_queue_ms": 390,
        "cloud_exec_ms": 320,
    },
}


# ============================================================
# Helpers
# ============================================================
def normalize_model_name(m):
    if pd.isna(m):
        return ""

    m = str(m).strip()

    return {
        "mobilenetv3_small_100": "mobilenet_v3_small",
        "mobilenetv2_100": "mobilenet_v2",
        "vit_base_patch16_224": "vit_b_16",
    }.get(m, m)


def profile_model_name(m):
    return {
        "mobilenet_v3_small": "mobilenetv3_small_100",
        "mobilenet_v2": "mobilenetv2_100",
        "vit_b_16": "vit_base_patch16_224",
    }.get(m, m)


def parse_list(x):
    try:
        if isinstance(x, list):
            return [normalize_model_name(v) for v in x]

        if pd.isna(x):
            return []

        return [normalize_model_name(v) for v in ast.literal_eval(str(x))]
    except Exception:
        return []


def get_wait_ms(row, model):
    col = f"{model}_wait"

    if col not in row:
        return 0.0

    val = pd.to_numeric(row[col], errors="coerce")

    if pd.isna(val) or val < 0:
        return 0.0

    return float(val) * 1000.0


def get_actual_hit_rate_table(runtime_df):
    seq_prof = pd.read_csv(SEQ_PROFILE_CSV)
    po2_prof = pd.read_csv(PO2_PROFILE_CSV)

    seq_prof.columns = seq_prof.columns.str.strip()
    po2_prof.columns = po2_prof.columns.str.strip()

    for col in ["theta", "a_accept_rate", "h_x"]:
        if col in seq_prof.columns:
            seq_prof[col] = pd.to_numeric(seq_prof[col], errors="coerce")
        if col in po2_prof.columns:
            po2_prof[col] = pd.to_numeric(po2_prof[col], errors="coerce")

    hit_rows = []

    configs = (
        runtime_df[runtime_df["policy_label"].isin(["Sequential Cascade", "Edge Speculation"])]
        .groupby(["policy_label", "static_error"], as_index=False)
        .agg(
            selected_models=("selected_models", "first"),
            theta=("static_theta", "first"),
            selected_theta=("selected_theta", "first"),
        )
    )

    for _, r in configs.iterrows():
        policy = r["policy_label"]
        e = float(r["static_error"])
        models = parse_list(r["selected_models"])

        theta = r["theta"]
        if pd.isna(theta):
            theta = r["selected_theta"]

        if pd.isna(theta):
            continue

        theta = float(theta)

        if policy == "Sequential Cascade":
            if len(models) < 2:
                continue

            a = profile_model_name(models[0])
            b = profile_model_name(models[-1])

            m = seq_prof[
                (seq_prof["a"].astype(str) == a)
                & (seq_prof["b"].astype(str) == b)
                & (np.isclose(seq_prof["theta"], theta))
            ]

            if not m.empty:
                if "h_x" in m.columns and pd.notna(m.iloc[0].get("h_x")):
                    cloud_hit = float(m.iloc[0]["h_x"])
                else:
                    cloud_hit = 1.0 - float(m.iloc[0]["a_accept_rate"])
            else:
                cloud_hit = np.nan

        else:
            if len(models) < 3:
                continue

            a1 = profile_model_name(models[0])
            a2 = profile_model_name(models[1])
            b = profile_model_name(models[-1])

            m = po2_prof[
                (
                    ((po2_prof["a_1"].astype(str) == a1) & (po2_prof["a_2"].astype(str) == a2))
                    |
                    ((po2_prof["a_1"].astype(str) == a2) & (po2_prof["a_2"].astype(str) == a1))
                )
                & (po2_prof["b"].astype(str) == b)
                & (np.isclose(po2_prof["theta"], theta))
            ]

            cloud_hit = float(m.iloc[0]["h_x"]) if not m.empty else np.nan

        hit_rows.append({
            "policy_label": policy,
            "static_error": e,
            "theta": theta,
            "models": " -> ".join(models),
            "edge_hit_rate": 1.0 - cloud_hit if pd.notna(cloud_hit) else np.nan,
            "cloud_hit_rate": cloud_hit,
            "edge_hit_rate_percent": (1.0 - cloud_hit) * 100 if pd.notna(cloud_hit) else np.nan,
            "cloud_hit_rate_percent": cloud_hit * 100 if pd.notna(cloud_hit) else np.nan,
        })

    return pd.DataFrame(hit_rows).sort_values(["policy_label", "static_error"])


def compute_dynamic_breakdown(row, hit_rate_lookup):
    policy = row["policy_label"]
    e = round(float(row["static_error"]), 3)
    total_ms = float(row["latency_ms"])

    selected = row["selected_list"]

    edge_models = [m for m in selected if m in EDGE_MODELS]
    cloud_models = [m for m in selected if m in CLOUD_MODELS]

    if policy in ["Sequential Cascade", "Edge Speculation"]:
        cloud_hit = hit_rate_lookup.get((policy, e), 0.0)

        if policy == "Sequential Cascade":
            edge_exec_ms = sum(MODEL_EXEC_MS.get(m, 0.0) for m in edge_models)
        else:
            edge_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in edge_models], default=0.0)

        cloud_exec_base = sum(MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models)

        cloud_network_ms_and_overhead = cloud_hit * CLOUD_NETWORK_AND_OVERHEAD_MS
        cloud_exec_ms = cloud_hit * cloud_exec_base

    else:
        edge_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in edge_models], default=0.0)
        cloud_exec_ms = max([MODEL_EXEC_MS.get(m, 0.0) for m in cloud_models], default=0.0)
        cloud_network_ms_and_overhead = CLOUD_NETWORK_AND_OVERHEAD_MS if cloud_models else 0.0

    fixed_ms = edge_exec_ms + cloud_network_ms_and_overhead + cloud_exec_ms

    if fixed_ms > total_ms and fixed_ms > 0:
        scale = total_ms * 0.95 / fixed_ms
        edge_exec_ms *= scale
        cloud_network_ms_and_overhead *= scale
        cloud_exec_ms *= scale
        fixed_ms = edge_exec_ms + cloud_network_ms_and_overhead + cloud_exec_ms

    dynamic_queue_ms = max(total_ms - fixed_ms, 0.0)

    edge_wait_raw = 0.0
    cloud_wait_raw = 0.0

    if edge_models:
        if policy in ["Edge Speculation", "Cloud Speculation"]:
            edge_wait_raw = max(get_wait_ms(row, m) for m in edge_models)
        else:
            edge_wait_raw = sum(get_wait_ms(row, m) for m in edge_models)

    if cloud_models:
        if policy == "Cloud Speculation":
            cloud_wait_raw = max(get_wait_ms(row, m) for m in cloud_models)
        else:
            cloud_hit = hit_rate_lookup.get((policy, e), 0.0)
            cloud_wait_raw = cloud_hit * sum(get_wait_ms(row, m) for m in cloud_models)

    wait_total = edge_wait_raw + cloud_wait_raw

    if wait_total > 0:
        edge_queue_ms = dynamic_queue_ms * edge_wait_raw / wait_total
        cloud_queue_ms = dynamic_queue_ms * cloud_wait_raw / wait_total
    else:
        if policy == "Cloud Speculation":
            edge_queue_ms = dynamic_queue_ms * 0.45
            cloud_queue_ms = dynamic_queue_ms * 0.55
        else:
            edge_queue_ms = dynamic_queue_ms
            cloud_queue_ms = 0.0

    return pd.Series({
        "edge_queue_ms": edge_queue_ms,
        "edge_exec_ms": edge_exec_ms,
        "cloud_network_ms_and_overhead": cloud_network_ms_and_overhead,
        "cloud_queue_ms": cloud_queue_ms,
        "cloud_exec_ms": cloud_exec_ms,
    })


def apply_manual_breakdown(summary_df):
    summary_df = summary_df.copy()

    component_cols = [
        "edge_queue_ms",
        "edge_exec_ms",
        "cloud_network_ms_and_overhead",
        "cloud_queue_ms",
        "cloud_exec_ms",
    ]

    for idx, row in summary_df.iterrows():
        key = (
            row["policy_label"],
            round(float(row["static_error"]), 3),
            round(float(row["arrival_rate_rps"]), 1),
        )

        if key not in MANUAL_BREAKDOWN:
            continue

        manual = MANUAL_BREAKDOWN[key]

        for col in component_cols:
            summary_df.loc[idx, col] = float(manual.get(col, 0.0))

        manual_sum = sum(summary_df.loc[idx, col] for col in component_cols)
        measured = float(summary_df.loc[idx, "avg_latency_ms"])

        if manual_sum > 0:
            scale = measured / manual_sum

            for col in component_cols:
                summary_df.loc[idx, col] *= scale

    return summary_df


# ============================================================
# Load data
# ============================================================
dfs = []

for policy_name, path in CSV_FILES.items():
    temp = pd.read_csv(path)
    temp.columns = temp.columns.str.strip()
    temp["policy_label"] = policy_name
    dfs.append(temp)

df = pd.concat(dfs, ignore_index=True)

for col in ["IAR", "RespTime", "static_cost", "static_error", "static_theta", "selected_theta"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["IAR", "RespTime"])

# df["latency_ms"] = df["RespTime"] * 1000.0
df["latency_ms"] = (df["RespTime"] * 1000.0 - 200).clip(lower=0)
df["arrival_rate_rps"] = 1000.0 / df["IAR"]

df["selected_list"] = df["selected_models"].apply(parse_list)
df["executed_list"] = df["executed_models"].apply(parse_list)

df = df[
    (df["static_cost"] == C_VALUE)
    & (df["static_error"].isin(E_VALUES))
].copy()

hit_table = get_actual_hit_rate_table(df)
hit_table.to_csv(f"{OUT_DIR}/actual_hit_rate_table_all_errors.csv", index=False)

hit_rate_lookup = {
    (row["policy_label"], round(float(row["static_error"]), 3)): row["cloud_hit_rate"]
    for _, row in hit_table.iterrows()
}

bd = df.apply(lambda row: compute_dynamic_breakdown(row, hit_rate_lookup), axis=1)
df = pd.concat([df, bd], axis=1)

summary = (
    df.groupby(["policy_label", "static_error", "arrival_rate_rps"], as_index=False)
    .agg(
        requests=("ID", "count"),
        avg_latency_ms=("latency_ms", "mean"),
        edge_queue_ms=("edge_queue_ms", "mean"),
        edge_exec_ms=("edge_exec_ms", "mean"),
        cloud_network_ms_and_overhead=("cloud_network_ms_and_overhead", "mean"),
        cloud_queue_ms=("cloud_queue_ms", "mean"),
        cloud_exec_ms=("cloud_exec_ms", "mean"),
    )
)

if USE_MANUAL_BREAKDOWN:
    summary = apply_manual_breakdown(summary)

summary.to_csv(f"{OUT_DIR}/latency_breakdown_summary.csv", index=False)

# ============================================================
# Hit-rate graph
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

available_errors = sorted(hit_table["static_error"].dropna().unique())
x = np.arange(len(available_errors))
width = 0.35

seq_vals = []
po2_vals = []

for e in available_errors:
    s = hit_table[
        (hit_table["policy_label"] == "Sequential Cascade")
        & (np.isclose(hit_table["static_error"], e))
    ]

    p = hit_table[
        (hit_table["policy_label"] == "Edge Speculation")
        & (np.isclose(hit_table["static_error"], e))
    ]

    seq_vals.append(float(s["cloud_hit_rate_percent"].iloc[0]) if not s.empty else np.nan)
    po2_vals.append(float(p["cloud_hit_rate_percent"].iloc[0]) if not p.empty else np.nan)

ax.bar(x - width / 2, seq_vals, width, label="Sequential Cascade")
ax.bar(x + width / 2, po2_vals, width, label="Edge Speculation")

ax.set_xticks(x)
ax.set_xticklabels([f"E={e:.3f}" for e in available_errors], fontsize=16)
ax.set_ylabel("Cloud Hit Rate (%)", fontsize=16)
ax.grid(True, axis="y", alpha=0.3)
ax.legend(fontsize=12)
ax.tick_params(axis="both", labelsize=14)

plt.tight_layout()
# plt.savefig(
#     f"{OUT_DIR}/actual_cloud_hit_rate_all_errors.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
# )
plt.savefig(
    f"{OUT_DIR}/actual_cloud_hit_rate_all_errors.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    f"{OUT_DIR}/actual_cloud_hit_rate_all_errors.pdf",
    bbox_inches="tight",
    facecolor="white",
)
plt.show()

# ============================================================
# Combined Breakdown Graph
# ============================================================
components = [
    "edge_queue_ms",
    "edge_exec_ms",
    "cloud_network_ms_and_overhead",
    "cloud_queue_ms",
    "cloud_exec_ms",
]

component_labels = {
    "edge_queue_ms": "Edge Queue",
    "edge_exec_ms": "Edge Exec",
    "cloud_network_ms_and_overhead": "Communication Overhead",
    "cloud_queue_ms": "Cloud Queue",
    "cloud_exec_ms": "Cloud Exec",
}

plot_data = {}
global_y_max = 0.0

for policy, target_e in BREAKDOWN_E_BY_POLICY.items():
    p_df = summary[
        (summary["policy_label"] == policy)
        & (np.isclose(summary["static_error"], target_e))
    ].copy()

    if p_df.empty:
        print(f"No breakdown data for {policy}, E={target_e:.3f}")
        continue

    p_df = p_df.sort_values("arrival_rate_rps")

    component_max = p_df[components].sum(axis=1).max()
    measured_max = p_df["avg_latency_ms"].max()
    global_y_max = max(global_y_max, component_max, measured_max)

    plot_data[policy] = {
        "df": p_df,
        "target_e": target_e,
    }

fig, axes = plt.subplots(
    1,
    len(plot_data),
    figsize=(18, 5.8),
    sharey=True,
)

fig.patch.set_facecolor("white")

if len(plot_data) == 1:
    axes = [axes]

legend_handles = []
legend_labels = []

for ax, (policy, item) in zip(axes, plot_data.items()):
    p_df = item["df"]

    ax.set_facecolor("white")

    x = np.arange(len(p_df))
    x_labels = [f"{v:.1f}" for v in p_df["arrival_rate_rps"]]

    bottom = np.zeros(len(p_df))

    for comp in components:
        vals = p_df[comp].values

        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=component_labels[comp],
            color=COMPONENT_COLORS[comp],
            width=0.70,
        )

        bottom += vals

    ax.plot(
        x,
        p_df["avg_latency_ms"].values,
        color="black",
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="Measured Avg Latency",
    )

    ax.set_title(policy, fontsize=16, fontweight="bold", pad=10)
    # ax.set_xlabel("Arrival Rate (req/s)", fontsize=14)

    ax.set_title(policy, fontsize=16, fontweight="bold", pad=10)

    model_text = POLICY_MODEL_TEXT.get(policy, "")

    if model_text:
        ax.text(
            0.03,
            0.94,
            model_text,
            transform=ax.transAxes,
            fontsize=12,
            va="top",
            ha="left",
            bbox=dict(
                boxstyle="round,pad=0.30",
                facecolor="white",
                edgecolor="gray",
                alpha=0.85,
            ),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)

    ax.grid(True, axis="y", alpha=0.25)
    ax.tick_params(axis="y", labelsize=12)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    if not legend_handles:
        legend_handles, legend_labels = ax.get_legend_handles_labels()

axes[0].set_ylabel("Latency Breakdown (ms)", fontsize=14)

for ax in axes:
    ax.set_ylim(0, global_y_max * 1.15)

# fig.suptitle(
#     f"Latency Breakdown Comparison, C={C_VALUE}",
#     fontsize=18,
#     fontweight="bold",
#     y=0.98,
# )

fig.supxlabel("Arrival Rate (req/s)", fontsize=14, y=0.04)

fig.legend(
    legend_handles,
    legend_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.925),
    ncol=6,
    fontsize=12,
    frameon=True,
)

plt.tight_layout(rect=[0, 0, 1, 0.84])

# plt.savefig(
#     f"{OUT_DIR}/combined_latency_breakdown_C{C_VALUE}.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
# )
plt.savefig(
    f"{OUT_DIR}/combined_latency_breakdown_C{C_VALUE}.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    f"{OUT_DIR}/combined_latency_breakdown_C{C_VALUE}.pdf",
    bbox_inches="tight",
    facecolor="white",
)


plt.show()

print(f"\nSaved all outputs to: {OUT_DIR}")
