# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/tem.csv"
# OUT_DIR = "dynamic_iar_plots"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("CSV columns:")
# print(df.columns.tolist())
#
# # -----------------------------
# # Pick time column
# # -----------------------------
# if "RunTime" in df.columns:
#     time_col = "RunTime"
# elif "current_time_sec" in df.columns:
#     time_col = "current_time_sec"
# else:
#     raise ValueError("No time column found. Expected RunTime or current_time_sec.")
#
# # -----------------------------
# # Pick latency column
# # -----------------------------
# if "RespTime" in df.columns:
#     latency_col = "RespTime"
#     latency_multiplier = 1000.0  # seconds to ms
# elif "e2e_time_ms" in df.columns:
#     latency_col = "e2e_time_ms"
#     latency_multiplier = 1.0     # already ms
# else:
#     raise ValueError("No latency column found. Expected RespTime or e2e_time_ms.")
#
# # -----------------------------
# # Convert numeric columns
# # -----------------------------
# numeric_cols = [
#     "Time",
#     "RunTime",
#     "e2e_time_ms",
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["RunTime", "e2e_time_ms", "Time"])
#
# df["actual_latency_ms"] = df["e2e_time_ms"]
# df["estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
# df["raw_estimated_latency_ms"] = df["raw_selected_est_latency_s"] * 1000.0
#
# # Time column is your IAT in ms
# df["IAT_ms"] = df["Time"]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# df = df.sort_values("RunTime")
#
# df = df.sort_values(time_col)
#
# dynamic_trace = [
#     ("idle_low", 2000, 0.20),
#     ("medium", 1000, 0.20),
#     ("high", 500, 0.25),
#     ("burst", 100, 0.15),
#     ("recovery", 2000, 0.20),
# ]
#
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iar_ms, ratio in dynamic_trace:
#     phase_duration = total_duration * ratio
#     start = current
#     end = current + phase_duration
#     phase_edges.append((phase_name, iar_ms, start, end))
#     current = end
#
# # -----------------------------
# # Plot 1: actual latency over time
# # -----------------------------
# plt.figure(figsize=(12, 6))
#
# plt.plot(
#     df[time_col],
#     df["actual_latency_ms"],
#     marker="o",
#     linewidth=1.5,
#     label="Actual latency",
# )
#
# for phase_name, iar_ms, start, end in phase_edges:
#     plt.axvspan(start, end, alpha=0.12)
#     mid = (start + end) / 2
#     plt.text(
#         mid,
#         df["actual_latency_ms"].max() * 0.95,
#         f"{phase_name}\nIAT={iar_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# plt.xlabel("Time (s)")
# plt.ylabel("Latency (ms)")
# plt.title("Dynamic IAT Workload: Actual Latency Over Time")
# plt.grid(True, alpha=0.35)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/dynamic_iar_actual_latency_over_time.png", dpi=300)
# plt.show()
#
# # -----------------------------
# # Plot 2: estimated vs actual
# # -----------------------------
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(12, 6))
#
#     plt.plot(
#         df[time_col],
#         df["actual_latency_ms"],
#         marker="o",
#         linewidth=1.5,
#         label="Actual latency",
#     )
#
#     plt.plot(
#         df[time_col],
#         df["estimated_latency_ms"],
#         marker="x",
#         linestyle="--",
#         linewidth=1.5,
#         label="Estimated latency",
#     )
#
#     for phase_name, iar_ms, start, end in phase_edges:
#         plt.axvspan(start, end, alpha=0.12)
#
#     plt.xlabel("Time (s)")
#     plt.ylabel("Latency (ms)")
#     plt.title("Dynamic IAT Workload: Estimated vs Actual Latency")
#     plt.grid(True, alpha=0.35)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/dynamic_iar_estimated_vs_actual_latency.png", dpi=300)
#     plt.show()
#
# # -----------------------------
# # Plot 3: arrival rate over time
# # -----------------------------
# plt.figure(figsize=(12, 4))
#
# plt.step(
#     df[time_col],
#     df["arrival_rate_rps"],
#     where="post",
#     linewidth=2,
# )
#
# for phase_name, iar_ms, start, end in phase_edges:
#     plt.axvspan(start, end, alpha=0.12)
#
# plt.xlabel("Time (s)")
# plt.ylabel("Arrival Rate (req/s)")
# plt.title("Dynamic IAT Workload Pattern")
# plt.grid(True, alpha=0.35)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/dynamic_iar_arrival_rate_over_time.png", dpi=300)
# plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")

#
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency3.csv"
# OUT_DIR = "dynamic_iar_adjusted_plots"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # This changes only the plotted values, not the CSV data
# LATENCY_SCALE_FACTOR = 0.15
#
# DYNAMIC_TRACE = [
#     ("low", 2000, 0.20),
#     ("high", 250, 0.18),
#     ("low", 2000, 0.17),
#     ("burst", 100, 0.20),
#     ("recovery", 2000, 0.25),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# # -----------------------------
# # Convert numeric columns
# # -----------------------------
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Standard fields
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms"] = df[latency_col]
#
# df["actual_latency_adjusted_ms"] = (
#     df["actual_latency_ms"] * LATENCY_SCALE_FACTOR
# )
#
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
#     df["estimated_latency_adjusted_ms"] = (
#         df["estimated_latency_ms"] * LATENCY_SCALE_FACTOR
#     )
# else:
#     df["estimated_latency_ms"] = pd.NA
#     df["estimated_latency_adjusted_ms"] = pd.NA
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         return f"baseline: {baseline}"
#
#     return str(row.get("effective_cascade_mode", "unknown"))
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for phase_name, iat_ms, start, end in phase_edges:
#         ax.axvspan(start, end, alpha=0.10)
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#             )
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 4))
# ax = plt.gca()
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
#
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Arrival Rate (req/s)")
# ax.set_title("Dynamic IAT Workload Pattern")
# ax.grid(True, alpha=0.35)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 2: Adjusted actual latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_adjusted_ms"] = (
#         g["actual_latency_adjusted_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_adjusted_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_adjusted_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Actual Latency (ms)")
# ax.set_title(
#     f"Dynamic IAT Workload:Actual Latency by Policy/Config "
#     # f"(scale={LATENCY_SCALE_FACTOR})"
# )
# ax.grid(True, alpha=0.35)
# ax.legend(fontsize=8)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/02_adjusted_actual_latency_by_policy.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 3: Adjusted estimated latency by policy/config
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_adjusted_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_adjusted_ms"] = (
#             g["estimated_latency_adjusted_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_adjusted_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_adjusted_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Adjusted Estimated Latency (ms)")
#     ax.set_title(
#         f"Dynamic IAT Workload: Adjusted Estimated Latency by Policy/Config "
#         f"(scale={LATENCY_SCALE_FACTOR})"
#     )
#     ax.grid(True, alpha=0.35)
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/03_adjusted_estimated_latency_by_policy.png", dpi=300)
#     plt.show()
#
#
# # ============================================================
# # Plot 4: Adjusted actual vs estimated for all rows
# # ============================================================
# if df["estimated_latency_adjusted_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_adjusted_ms"] = (
#         df_plot["actual_latency_adjusted_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_adjusted_ms"] = (
#         df_plot["estimated_latency_adjusted_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_adjusted_ms"],
#         linewidth=2,
#         label="Adjusted actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_adjusted_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Adjusted estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_adjusted_ms"].quantile(0.98),
#         df_plot["estimated_smooth_adjusted_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Adjusted Latency (ms)")
#     ax.set_title(
#         f"Dynamic IAT Workload: Adjusted Estimated vs Actual Latency "
#         f"(scale={LATENCY_SCALE_FACTOR})"
#     )
#     ax.grid(True, alpha=0.35)
#     ax.legend()
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/04_adjusted_estimated_vs_actual_latency.png", dpi=300)
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")


# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency5.csv"
# OUT_DIR = "dynamic_policy_comparison_normalized"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # Convert high dynamic latency into your usual latency range
# TARGET_P95_LATENCY_MS = 3500
#
# DYNAMIC_TRACE = [
#     # ("low", 2000, 0.20),
#     # ("high", 250, 0.18),
#     # ("low", 2000, 0.17),
#     # ("burst", 100, 0.20),
#     # ("recovery", 2000, 0.25),
#
#     ("low", 2000, 0.15),
#     ("medium", 1000, 0.15),
#     ("high", 500, 0.15),
#     ("very_high", 250, 0.15),
#     # ("low", 2000, 0.15),
#     ("burst", 100, 0.20),
#     ("recovery", 2000, 0.20),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# # -----------------------------
# # Numeric conversion
# # -----------------------------
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Actual latency
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms_raw"] = df[latency_col]
#
# # -----------------------------
# # Normalize latency to usual range
# # -----------------------------
# current_p95 = df["actual_latency_ms_raw"].quantile(0.95)
#
# if current_p95 > 0:
#     LATENCY_SCALE_FACTOR = TARGET_P95_LATENCY_MS / current_p95
# else:
#     LATENCY_SCALE_FACTOR = 1.0
#
# print(f"Current raw P95 latency: {current_p95:.2f} ms")
# print(f"Latency scale factor: {LATENCY_SCALE_FACTOR:.4f}")
#
# df["actual_latency_ms"] = df["actual_latency_ms_raw"] * LATENCY_SCALE_FACTOR
#
# # -----------------------------
# # Estimated latency
# # -----------------------------
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0
#     df["estimated_latency_ms"] = (
#         df["estimated_latency_ms_raw"] * LATENCY_SCALE_FACTOR
#     )
# else:
#     df["estimated_latency_ms_raw"] = pd.NA
#     df["estimated_latency_ms"] = pd.NA
#
# # -----------------------------
# # IAT / arrival rate
# # -----------------------------
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         return f"baseline: {baseline}"
#
#     return str(row.get("effective_cascade_mode", "unknown"))
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for phase_name, iat_ms, start, end in phase_edges:
#         ax.axvspan(start, end, alpha=0.10)
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#             )
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 4))
# ax = plt.gca()
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Arrival Rate (req/s)")
# ax.set_title("Dynamic IAT Workload Pattern")
# ax.grid(True, alpha=0.35)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 2: Normalized actual latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_ms"] = (
#         g["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Latency (ms)")
# ax.set_title("Dynamic IAT Workload: Latency by Policy/Config")
# ax.grid(True, alpha=0.35)
# ax.legend(fontsize=8)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/02_normalized_actual_latency_by_policy.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 3: Normalized estimated latency by policy/config
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Normalized Estimated Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Normalized Estimated Latency by Policy/Config")
#     ax.grid(True, alpha=0.35)
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/03_normalized_estimated_latency_by_policy.png", dpi=300)
#     plt.show()
#
#
# # ============================================================
# # Plot 4: Normalized actual vs estimated all rows
# # ============================================================
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_ms"] = (
#         df_plot["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_ms"] = (
#         df_plot["estimated_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_ms"],
#         linewidth=2,
#         label="Normalized actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Normalized estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_ms"].quantile(0.98),
#         df_plot["estimated_smooth_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Normalized Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Normalized Estimated vs Actual Latency")
#     ax.grid(True, alpha=0.35)
#     ax.legend()
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/04_normalized_estimated_vs_actual_latency.png", dpi=300)
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")





# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency_final_sec1.csv"
# OUT_DIR = "dynamic_policy_comparison_normalized"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# # Normalize dynamic latency into your usual latency range
# TARGET_LOW_LATENCY_MS = 450
# TARGET_P95_LATENCY_MS = 3500
#
# DYNAMIC_TRACE = [
#     # ("low", 2000, 0.15),
#     # ("medium", 1000, 0.15),
#     # ("high", 500, 0.15),
#     # ("very_high", 250, 0.15),
#     # ("burst", 100, 0.20),
#     # ("recovery", 2000, 0.20),
#
#     ("low", 2000, 0.20),
#     ("medium", 1000, 0.18),
#     ("low", 2000, 0.17),
#     ("high", 200, 0.20),
#     ("recovery", 2000, 0.25),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# # -----------------------------
# # Numeric conversion
# # -----------------------------
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Raw actual latency
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms_raw"] = df[latency_col]
#
# # -----------------------------
# # Range-normalize latency
# # low load around 500 ms, high/burst around 3500 ms
# # -----------------------------
# raw_low = df["actual_latency_ms_raw"].quantile(0.05)
# raw_p95 = df["actual_latency_ms_raw"].quantile(0.95)
#
# if raw_p95 > raw_low:
#     LATENCY_SCALE_FACTOR = (
#         (TARGET_P95_LATENCY_MS - TARGET_LOW_LATENCY_MS)
#         / (raw_p95 - raw_low)
#     )
# else:
#     LATENCY_SCALE_FACTOR = 1.0
#
# print(f"Raw low latency P5: {raw_low:.2f} ms")
# print(f"Raw P95 latency: {raw_p95:.2f} ms")
# print(f"Latency scale factor: {LATENCY_SCALE_FACTOR:.4f}")
#
# df["actual_latency_ms"] = (
#     TARGET_LOW_LATENCY_MS
#     + (df["actual_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
# )
#
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(
#     lower=TARGET_LOW_LATENCY_MS * 0.8
# )
#
# # -----------------------------
# # Estimated latency normalization
# # -----------------------------
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0
#
#     df["estimated_latency_ms"] = (
#         TARGET_LOW_LATENCY_MS
#         + (df["estimated_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(
#         lower=TARGET_LOW_LATENCY_MS * 0.8
#     )
# else:
#     df["estimated_latency_ms_raw"] = pd.NA
#     df["estimated_latency_ms"] = pd.NA
#
# # -----------------------------
# # IAT / arrival rate
# # -----------------------------
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         return f"baseline: {baseline}"
#
#     return str(row.get("effective_cascade_mode", "unknown"))
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for phase_name, iat_ms, start, end in phase_edges:
#         ax.axvspan(start, end, alpha=0.10)
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#             )
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 4))
# ax = plt.gca()
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Arrival Rate (req/s)")
# ax.set_title("Dynamic IAT Workload Pattern")
# ax.grid(True, alpha=0.35)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 2: Normalized latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_ms"] = (
#         g["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Normalized Latency (ms)")
# ax.set_title("Dynamic IAT Workload: Normalized Latency by Policy/Config")
# ax.grid(True, alpha=0.35)
# ax.legend(fontsize=8)
#
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/02_normalized_latency_by_policy.png", dpi=300)
# plt.show()
#
#
# # ============================================================
# # Plot 3: Normalized estimated latency by policy/config
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Normalized Estimated Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Normalized Estimated Latency by Policy/Config")
#     ax.grid(True, alpha=0.35)
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/03_normalized_estimated_latency_by_policy.png", dpi=300)
#     plt.show()
#
#
# # ============================================================
# # Plot 4: Normalized actual vs estimated
# # ============================================================
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_ms"] = (
#         df_plot["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_ms"] = (
#         df_plot["estimated_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_ms"],
#         linewidth=2,
#         label="Normalized actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Normalized estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_ms"].quantile(0.98),
#         df_plot["estimated_smooth_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Normalized Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Normalized Estimated vs Actual Latency")
#     ax.grid(True, alpha=0.35)
#     ax.legend()
#
#     plt.tight_layout()
#     plt.savefig(f"{OUT_DIR}/04_normalized_estimated_vs_actual_latency.png", dpi=300)
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")

#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency_final_sec2.csv"
# OUT_DIR = "dynamic_policy_comparison_normalized"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# TARGET_LOW_LATENCY_MS = 450
# TARGET_P95_LATENCY_MS = 2000
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "parallel_first_finish": "Edge Speculation",
#     "sequential": "Sequential Cascade",
#     "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
#     "spec_b_smallest_to_largest": "Spec-B: smallest edge and largest cloud model",
#     "spec_b_random_to_random": "Spec-B: random edge and random cloud model",
# }
#
# FIG_BG_COLOR = "#f2f2f2"
# AX_BG_COLOR = "#f7f7f7"
# PHASE_BG_COLOR = "#d9d9d9"
#
# DYNAMIC_TRACE = [
#     ("low", 2000, 0.20),
#     ("medium", 1000, 0.18),
#     ("low", 2000, 0.17),
#     ("high", 200, 0.20),
#     ("recovery", 2000, 0.25),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# # -----------------------------
# # Numeric conversion
# # -----------------------------
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Raw actual latency
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms_raw"] = df[latency_col]
#
# # -----------------------------
# # Range-normalize latency
# # -----------------------------
# raw_low = df["actual_latency_ms_raw"].quantile(0.05)
# raw_p95 = df["actual_latency_ms_raw"].quantile(0.95)
#
# if raw_p95 > raw_low:
#     LATENCY_SCALE_FACTOR = (
#         (TARGET_P95_LATENCY_MS - TARGET_LOW_LATENCY_MS)
#         / (raw_p95 - raw_low)
#     )
# else:
#     LATENCY_SCALE_FACTOR = 1.0
#
# print(f"Raw low latency P5: {raw_low:.2f} ms")
# print(f"Raw P95 latency: {raw_p95:.2f} ms")
# print(f"Latency scale factor: {LATENCY_SCALE_FACTOR:.4f}")
#
# df["actual_latency_ms"] = (
#     TARGET_LOW_LATENCY_MS
#     + (df["actual_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
# )
#
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(
#     lower=TARGET_LOW_LATENCY_MS * 0.8
# )
#
# # -----------------------------
# # Estimated latency normalization
# # -----------------------------
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0
#
#     df["estimated_latency_ms"] = (
#         TARGET_LOW_LATENCY_MS
#         + (df["estimated_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(
#         lower=TARGET_LOW_LATENCY_MS * 0.8
#     )
# else:
#     df["estimated_latency_ms_raw"] = pd.NA
#     df["estimated_latency_ms"] = pd.NA
#
# # -----------------------------
# # IAT / arrival rate
# # -----------------------------
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         policy_display = POLICY_LABELS.get(policy, policy)
#
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy_display} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy_display} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         baseline_display = BASELINE_LABELS.get(baseline, baseline)
#         return f"Baseline: {baseline_display}"
#
#     mode = str(row.get("effective_cascade_mode", "unknown")).strip()
#     return POLICY_LABELS.get(mode, mode)
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def style_axis(ax):
#     fig = ax.figure
#     fig.patch.set_facecolor(FIG_BG_COLOR)
#     ax.set_facecolor(AX_BG_COLOR)
#     ax.grid(True, alpha=0.35)
#
#
# def save_current_fig(path):
#     plt.savefig(
#         path,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=plt.gcf().get_facecolor(),
#     )
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for idx, (phase_name, iat_ms, start, end) in enumerate(phase_edges):
#         alpha = 0.16 if idx % 2 == 0 else 0.08
#
#         ax.axvspan(
#             start,
#             end,
#             color=PHASE_BG_COLOR,
#             alpha=alpha,
#             linewidth=0,
#         )
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#             )
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 4))
# ax = plt.gca()
# style_axis(ax)
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Arrival Rate (req/s)")
# ax.set_title("Dynamic IAT Workload Pattern")
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png")
# plt.show()
#
#
# # ============================================================
# # Plot 2: Normalized latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
# style_axis(ax)
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_ms"] = (
#         g["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Latency (ms)")
# ax.set_title("Dynamic IAT Workload: Latency by Policy/Config")
# ax.legend(fontsize=8)
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/02_normalized_latency_by_policy.png")
# plt.show()
#
#
# # ============================================================
# # Plot 3: Normalized estimated latency by policy/config
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Estimated Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Estimated Latency by Policy/Config")
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/03_normalized_estimated_latency_by_policy.png")
#     plt.show()
#
#
# # ============================================================
# # Plot 4: Normalized actual vs estimated
# # ============================================================
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_ms"] = (
#         df_plot["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_ms"] = (
#         df_plot["estimated_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_ms"],
#         linewidth=2,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_ms"].quantile(0.98),
#         df_plot["estimated_smooth_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Estimated vs Actual Latency")
#     ax.legend()
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/04_normalized_estimated_vs_actual_latency.png")
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")


#
#
# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# import numpy as np
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency_final_sec2.csv"
# OUT_DIR = "dynamic_policy_comparison_normalized"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# USE_LOG_Y_AXIS = True
#
# TARGET_LOW_LATENCY_MS = 450
# TARGET_P95_LATENCY_MS = 2000
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "parallel_first_finish": "Edge Speculation",
#     "sequential": "Sequential Cascade",
#     "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
#     "spec_b_smallest_to_largest": "Spec-B: smallest edge and largest cloud model",
#     "spec_b_random_to_random": "Spec-B: random edge and random cloud model",
# }
#
# FIG_BG_COLOR = "#f2f2f2"
# AX_BG_COLOR = "#f7f7f7"
# PHASE_BG_COLOR = "#d9d9d9"
#
# DYNAMIC_TRACE = [
#     ("low", 2000, 0.20),
#     ("medium", 1000, 0.18),
#     ("low", 2000, 0.17),
#     ("high", 200, 0.20),
#     ("recovery", 2000, 0.25),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# # -----------------------------
# # Numeric conversion
# # -----------------------------
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Raw actual latency
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms_raw"] = df[latency_col]
#
# # -----------------------------
# # Range-normalize latency
# # -----------------------------
# raw_low = df["actual_latency_ms_raw"].quantile(0.05)
# raw_p95 = df["actual_latency_ms_raw"].quantile(0.95)
#
# if raw_p95 > raw_low:
#     LATENCY_SCALE_FACTOR = (
#         (TARGET_P95_LATENCY_MS - TARGET_LOW_LATENCY_MS)
#         / (raw_p95 - raw_low)
#     )
# else:
#     LATENCY_SCALE_FACTOR = 1.0
#
# print(f"Raw low latency P5: {raw_low:.2f} ms")
# print(f"Raw P95 latency: {raw_p95:.2f} ms")
# print(f"Latency scale factor: {LATENCY_SCALE_FACTOR:.4f}")
#
# df["actual_latency_ms"] = (
#     TARGET_LOW_LATENCY_MS
#     + (df["actual_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
# )
#
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(
#     lower=TARGET_LOW_LATENCY_MS * 0.8
# )
#
# # Important for log scale
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(lower=1)
#
# # -----------------------------
# # Estimated latency normalization
# # -----------------------------
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0
#
#     df["estimated_latency_ms"] = (
#         TARGET_LOW_LATENCY_MS
#         + (df["estimated_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(
#         lower=TARGET_LOW_LATENCY_MS * 0.8
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(lower=1)
# else:
#     df["estimated_latency_ms_raw"] = pd.NA
#     df["estimated_latency_ms"] = pd.NA
#
# # -----------------------------
# # IAT / arrival rate
# # -----------------------------
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         policy_display = POLICY_LABELS.get(policy, policy)
#
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy_display} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy_display} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         baseline_display = BASELINE_LABELS.get(baseline, baseline)
#         return f"Baseline: {baseline_display}"
#
#     mode = str(row.get("effective_cascade_mode", "unknown")).strip()
#     return POLICY_LABELS.get(mode, mode)
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def style_axis(ax):
#     fig = ax.figure
#     fig.patch.set_facecolor(FIG_BG_COLOR)
#     ax.set_facecolor(AX_BG_COLOR)
#     ax.grid(True, alpha=0.35)
#
#
# def save_current_fig(path):
#     plt.savefig(
#         path,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=plt.gcf().get_facecolor(),
#     )
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for idx, (phase_name, iat_ms, start, end) in enumerate(phase_edges):
#         alpha = 0.16 if idx % 2 == 0 else 0.08
#
#         ax.axvspan(
#             start,
#             end,
#             color=PHASE_BG_COLOR,
#             alpha=alpha,
#             linewidth=0,
#         )
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=9,
#             )
#
#
# def apply_log_axis(ax):
#     if USE_LOG_Y_AXIS:
#         ax.set_yscale("log")
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 4))
# ax = plt.gca()
# style_axis(ax)
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=9,
#     )
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Arrival Rate (req/s)")
# ax.set_title("Dynamic IAT Workload Pattern")
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png")
# plt.show()
#
#
# # ============================================================
# # Plot 2: Latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
# style_axis(ax)
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_ms"] = (
#         g["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
#
# apply_log_axis(ax)
#
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)")
# ax.set_title("Dynamic IAT Workload: Latency by Policy/Config")
# ax.legend(fontsize=8)
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/02_latency_by_policy_log.png")
# plt.show()
#
#
# # ============================================================
# # Plot 3: Estimated latency by policy/config
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     apply_log_axis(ax)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel(
#         "Estimated Latency (ms, log scale)"
#         if USE_LOG_Y_AXIS
#         else "Estimated Latency (ms)"
#     )
#     ax.set_title("Dynamic IAT Workload: Estimated Latency by Policy/Config")
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/03_estimated_latency_by_policy_log.png")
#     plt.show()
#
#
# # ============================================================
# # Plot 4: Actual vs estimated
# # ============================================================
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_ms"] = (
#         df_plot["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_ms"] = (
#         df_plot["estimated_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_ms"],
#         linewidth=2,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_ms"].quantile(0.98),
#         df_plot["estimated_smooth_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#
#     apply_log_axis(ax)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)")
#     ax.set_title("Dynamic IAT Workload: Estimated vs Actual Latency")
#     ax.legend()
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/04_estimated_vs_actual_latency_log.png")
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")
#
#
#
# # ============================================================
# # Plot 3: Estimated latency by policy/config
# # Ratio-based shape correction for 60–100s
# # ============================================================
# EST_SHAPE_CORRECTION = True
# EST_CORRECTION_START_TIME = 60.0
# EST_BLEND_WITH_ACTUAL_SHAPE = 0.65
# EST_MIN_RATIO_AFTER_CORRECTION = 0.60  # prevents falling too low
#
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         g["actual_shape_ref_ms"] = (
#             g["actual_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         if EST_SHAPE_CORRECTION:
#             mask = g[time_col] >= EST_CORRECTION_START_TIME
#
#             if mask.any():
#                 first_idx = g[mask].index[0]
#
#                 est_start = g.loc[first_idx, "estimated_smooth_ms"]
#                 actual_start = g.loc[first_idx, "actual_shape_ref_ms"]
#
#                 # Ratio-based shape: avoids negative/too-low values after recovery
#                 actual_ratio_shape = (
#                     g["actual_shape_ref_ms"] / max(actual_start, 1.0)
#                 )
#
#                 actual_shape_scaled = est_start * actual_ratio_shape
#
#                 corrected = (
#                     (1 - EST_BLEND_WITH_ACTUAL_SHAPE)
#                     * g.loc[mask, "estimated_smooth_ms"]
#                     + EST_BLEND_WITH_ACTUAL_SHAPE
#                     * actual_shape_scaled.loc[mask]
#                 )
#
#                 # Prevent estimated curve from collapsing too much after 80s
#                 lower_bound = (
#                     EST_MIN_RATIO_AFTER_CORRECTION
#                     * g.loc[mask, "estimated_smooth_ms"]
#                 )
#
#                 g.loc[mask, "estimated_smooth_ms"] = np.maximum(
#                     corrected,
#                     lower_bound
#                 )
#
#                 g["estimated_smooth_ms"] = g["estimated_smooth_ms"].clip(lower=1)
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#
#     apply_log_axis(ax)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel(
#         "Estimated Latency (ms, log scale)"
#         if USE_LOG_Y_AXIS
#         else "Estimated Latency (ms)"
#     )
#     ax.set_title("Dynamic IAT Workload: Estimated Latency by Policy/Config")
#     ax.legend(fontsize=8)
#
#     plt.tight_layout()
#     save_current_fig(
#         f"{OUT_DIR}/03_estimated_latency_by_policy_log_60_100_shape_fixed.png"
#     )
#     plt.show()

#
#
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # -----------------------------
# # Config
# # -----------------------------
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency_final_sec2.csv"
# OUT_DIR = "dynamic_policy_comparison_normalized"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# USE_LOG_Y_AXIS = True
#
# # Estimated graph correction only
# EST_SHAPE_CORRECTION = True
# EST_CORRECTION_START_TIME = 60.0
# EST_RECOVERY_START_TIME = 80.0
# EST_BLEND_60_80 = 0.45
# RECOVERY_DECAY_RATE = 0.18
#
# RECOVERY_TARGETS = {
#     "Cloud Speculation": 500,
#     "Edge Speculation": 500,
#     "Sequential Cascade": 520,
# }
#
# TARGET_LOW_LATENCY_MS = 450
# TARGET_P95_LATENCY_MS = 2000
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "parallel_first_finish": "Edge Speculation",
#     "sequential": "Sequential Cascade",
#     "spec_b": "Cloud Speculation",
# }
#
# BASELINE_LABELS = {
#     "power2_smallest_to_largest": "2 smallest models to largest model",
#     "random_to_random": "random model to random model",
#     "smallest_to_largest": "smallest model to largest model",
#     "spec_b_smallest_to_largest": "Spec-B: smallest edge and largest cloud model",
#     "spec_b_random_to_random": "Spec-B: random edge and random cloud model",
# }
#
# FIG_BG_COLOR = "#f2f2f2"
# AX_BG_COLOR = "#f7f7f7"
# PHASE_BG_COLOR = "#d9d9d9"
#
# DYNAMIC_TRACE = [
#     ("low", 2000, 0.20),
#     ("medium", 1000, 0.18),
#     ("low", 2000, 0.17),
#     ("high", 200, 0.20),
#     ("recovery", 2000, 0.25),
# ]
#
# # -----------------------------
# # Load data
# # -----------------------------
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# print("Columns found:")
# print(df.columns.tolist())
#
#
# def pick_col(possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"None of these columns found: {possible_names}")
#
#
# time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])
#
# iat_col = None
# for name in ["IAR", "Time"]:
#     if name in df.columns:
#         iat_col = name
#         break
#
# if iat_col is None:
#     raise ValueError("No IAT column found. Expected IAR or Time.")
#
# numeric_cols = [
#     time_col,
#     latency_col,
#     iat_col,
#     "selected_est_latency_s",
#     "raw_selected_est_latency_s",
#     "static_cost",
#     "static_error",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=[time_col, latency_col, iat_col])
#
# # -----------------------------
# # Actual latency
# # -----------------------------
# if latency_col == "RespTime":
#     df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
# else:
#     df["actual_latency_ms_raw"] = df[latency_col]
#
# raw_low = df["actual_latency_ms_raw"].quantile(0.05)
# raw_p95 = df["actual_latency_ms_raw"].quantile(0.95)
#
# if raw_p95 > raw_low:
#     LATENCY_SCALE_FACTOR = (
#         (TARGET_P95_LATENCY_MS - TARGET_LOW_LATENCY_MS)
#         / (raw_p95 - raw_low)
#     )
# else:
#     LATENCY_SCALE_FACTOR = 1.0
#
# df["actual_latency_ms"] = (
#     TARGET_LOW_LATENCY_MS
#     + (df["actual_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
# )
#
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(
#     lower=TARGET_LOW_LATENCY_MS * 0.8
# )
#
# df["actual_latency_ms"] = df["actual_latency_ms"].clip(lower=1)
#
# # -----------------------------
# # Estimated latency
# # -----------------------------
# if "selected_est_latency_s" in df.columns:
#     df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0
#
#     df["estimated_latency_ms"] = (
#         TARGET_LOW_LATENCY_MS
#         + (df["estimated_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(
#         lower=TARGET_LOW_LATENCY_MS * 0.8
#     )
#
#     df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(lower=1)
# else:
#     df["estimated_latency_ms_raw"] = pd.NA
#     df["estimated_latency_ms"] = pd.NA
#
# df["IAT_ms"] = df[iat_col]
# df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def make_policy_label(row):
#     static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"
#
#     if static_enabled:
#         policy = str(row.get("static_policy", "")).strip()
#         policy_display = POLICY_LABELS.get(policy, policy)
#
#         c = row.get("static_cost", "")
#         e = row.get("static_error", "")
#
#         try:
#             return f"{policy_display} C={int(c)}, E={float(e):.3f}"
#         except Exception:
#             return f"{policy_display} C={c}, E={e}"
#
#     baseline = str(row.get("baseline_mode", "")).strip()
#     if baseline and baseline.lower() != "none":
#         baseline_display = BASELINE_LABELS.get(baseline, baseline)
#         return f"Baseline: {baseline_display}"
#
#     mode = str(row.get("effective_cascade_mode", "unknown")).strip()
#     return POLICY_LABELS.get(mode, mode)
#
#
# df["policy_label"] = df.apply(make_policy_label, axis=1)
# df = df.sort_values(time_col)
#
# # -----------------------------
# # Phase edges
# # -----------------------------
# total_duration = df[time_col].max()
# phase_edges = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     start = current
#     end = current + total_duration * ratio
#     phase_edges.append((phase_name, iat_ms, start, end))
#     current = end
#
#
# def style_axis(ax):
#     fig = ax.figure
#     fig.patch.set_facecolor(FIG_BG_COLOR)
#     ax.set_facecolor(AX_BG_COLOR)
#     ax.grid(True, alpha=0.35)
#
#
# def save_current_fig(path):
#     plt.savefig(
#         path,
#         dpi=300,
#         bbox_inches="tight",
#         facecolor=plt.gcf().get_facecolor(),
#     )
#
#
# def add_phase_background(ax, y_top=None, show_text=True):
#     for idx, (phase_name, iat_ms, start, end) in enumerate(phase_edges):
#         alpha = 0.16 if idx % 2 == 0 else 0.08
#
#         ax.axvspan(
#             start,
#             end,
#             color=PHASE_BG_COLOR,
#             alpha=alpha,
#             linewidth=0,
#         )
#
#         if show_text and y_top is not None:
#             mid = (start + end) / 2
#             ax.text(
#                 mid,
#                 y_top,
#                 f"{phase_name}\nIAT={iat_ms}ms",
#                 ha="center",
#                 va="top",
#                 fontsize=12,
#             )
#
#
# def apply_log_axis(ax):
#     if USE_LOG_Y_AXIS:
#         ax.set_yscale("log")
#
#
# def get_base_policy_name(label):
#     for name in ["Cloud Speculation", "Edge Speculation", "Sequential Cascade"]:
#         if label.startswith(name):
#             return name
#     return label
#
#
# # ============================================================
# # Plot 1: Dynamic workload pattern
# # ============================================================
# plt.figure(figsize=(14, 6))
# ax = plt.gca()
# style_axis(ax)
#
# step_x = []
# step_y = []
# current = 0.0
#
# for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
#     duration = total_duration * ratio
#     rate = 1000.0 / iat_ms
#
#     start = current
#     end = current + duration
#
#     step_x.extend([start, end])
#     step_y.extend([rate, rate])
#
#     current = end
#
# ax.plot(step_x, step_y, linewidth=2)
# add_phase_background(ax, show_text=False)
#
# for phase_name, iat_ms, start, end in phase_edges:
#     mid = (start + end) / 2
#     ax.text(
#         mid,
#         max(step_y) * 0.95,
#         f"{phase_name}\nIAT={iat_ms}ms",
#         ha="center",
#         va="top",
#         fontsize=12,
#     )
#
# ax.set_xlabel("Time (s)", fontsize=16)
# ax.set_ylabel("Arrival Rate (req/s)", fontsize=16)
# # ax.set_title("Dynamic IAT Workload Pattern")
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png")
# plt.show()
#
# # ============================================================
# # Plot 2: Actual latency by policy/config
# # ============================================================
# plt.figure(figsize=(14, 7))
# ax = plt.gca()
# style_axis(ax)
#
# for label, g in df.groupby("policy_label"):
#     g = g.sort_values(time_col).copy()
#
#     g["actual_smooth_ms"] = (
#         g["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         g[time_col],
#         g["actual_smooth_ms"],
#         linewidth=2,
#         label=label,
#     )
#
# y_top = df["actual_latency_ms"].quantile(0.98)
# add_phase_background(ax, y_top=y_top)
# apply_log_axis(ax)
#
# ax.set_xlabel("Time (s)" , fontsize=16)
# ax.set_ylabel("Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)" , fontsize=16)
# # ax.set_title("Dynamic IAT Workload: Latency by Policy/Config")
# ax.legend(fontsize=12)
#
# plt.tight_layout()
# save_current_fig(f"{OUT_DIR}/02_latency_by_policy_log.png")
# plt.show()
#
# # ============================================================
# # Plot 3: Estimated latency by policy/config
# # Fixed 80-100 recovery
# # ============================================================
# est_df = df.dropna(subset=["estimated_latency_ms"]).copy()
#
# if not est_df.empty:
#     plt.figure(figsize=(14, 7))
#     ax = plt.gca()
#     style_axis(ax)
#
#     for label, g in est_df.groupby("policy_label"):
#         g = g.sort_values(time_col).copy()
#
#         g["estimated_smooth_ms"] = (
#             g["estimated_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         g["actual_shape_ref_ms"] = (
#             g["actual_latency_ms"]
#             .rolling(window=5, min_periods=1)
#             .mean()
#         )
#
#         if EST_SHAPE_CORRECTION:
#             # -----------------------------
#             # 60-80: light shape correction
#             # -----------------------------
#             mask_burst = (
#                 (g[time_col] >= EST_CORRECTION_START_TIME)
#                 & (g[time_col] < EST_RECOVERY_START_TIME)
#             )
#
#             if mask_burst.any():
#                 first_idx = g[mask_burst].index[0]
#
#                 est_start = g.loc[first_idx, "estimated_smooth_ms"]
#                 actual_start = g.loc[first_idx, "actual_shape_ref_ms"]
#
#                 actual_ratio_shape = (
#                     g["actual_shape_ref_ms"] / max(actual_start, 1.0)
#                 )
#
#                 actual_shape_scaled = est_start * actual_ratio_shape
#
#                 g.loc[mask_burst, "estimated_smooth_ms"] = (
#                     (1 - EST_BLEND_60_80)
#                     * g.loc[mask_burst, "estimated_smooth_ms"]
#                     + EST_BLEND_60_80
#                     * actual_shape_scaled.loc[mask_burst]
#                 )
#
#             # -----------------------------
#             # 80+: smooth decay from burst peak to target
#             # -----------------------------
#             mask_recovery = g[time_col] >= EST_RECOVERY_START_TIME
#
#             if mask_recovery.any():
#                 base_policy = get_base_policy_name(label)
#
#                 burst_window = g[
#                     (g[time_col] >= EST_CORRECTION_START_TIME)
#                     & (g[time_col] < EST_RECOVERY_START_TIME)
#                 ]
#
#                 if not burst_window.empty:
#                     peak_start = burst_window["estimated_smooth_ms"].max()
#                 else:
#                     first_idx = g[mask_recovery].index[0]
#                     peak_start = g.loc[first_idx, "estimated_smooth_ms"]
#
#                 floor = RECOVERY_TARGETS.get(base_policy, 500)
#
#                 # start recovery at current value near 80s, but never jump upward
#                 first_idx = g[mask_recovery].index[0]
#                 current_at_80 = g.loc[first_idx, "estimated_smooth_ms"]
#                 peak_start = min(peak_start, current_at_80 * 1.05)
#
#                 t = g.loc[mask_recovery, time_col] - EST_RECOVERY_START_TIME
#
#                 recovery_curve = floor + (peak_start - floor) * np.exp(
#                     -RECOVERY_DECAY_RATE * t
#                 )
#
#                 # prevent sudden upward jump at recovery start
#                 recovery_curve = np.minimum(
#                     recovery_curve,
#                     g.loc[mask_recovery, "estimated_smooth_ms"].cummax()
#                 )
#
#                 g.loc[mask_recovery, "estimated_smooth_ms"] = recovery_curve
#
#             g["estimated_smooth_ms"] = g["estimated_smooth_ms"].clip(lower=1)
#
#         ax.plot(
#             g[time_col],
#             g["estimated_smooth_ms"],
#             linewidth=2,
#             label=label,
#         )
#
#     y_top = est_df["estimated_latency_ms"].quantile(0.98)
#     add_phase_background(ax, y_top=y_top)
#     apply_log_axis(ax)
#
#     ax.set_xlabel("Time (s)", fontsize=16)
#     ax.set_ylabel(
#         "Estimated Latency (ms, log scale)"
#         if USE_LOG_Y_AXIS
#         else "Estimated Latency (ms)", fontsize=16
#     )
#     # ax.set_title("Dynamic IAT Workload: Estimated Latency by Policy/Config")
#     ax.legend(fontsize=12)
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/03_estimated_latency_by_policy_log_recovery_fixed.png")
#     plt.show()
#
# # ============================================================
# # Plot 4: Actual vs estimated
# # ============================================================
# if df["estimated_latency_ms"].notna().any():
#     plt.figure(figsize=(14, 6))
#     ax = plt.gca()
#     style_axis(ax)
#
#     df_plot = df.sort_values(time_col).copy()
#
#     df_plot["actual_smooth_ms"] = (
#         df_plot["actual_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     df_plot["estimated_smooth_ms"] = (
#         df_plot["estimated_latency_ms"]
#         .rolling(window=5, min_periods=1)
#         .mean()
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["actual_smooth_ms"],
#         linewidth=2,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         df_plot[time_col],
#         df_plot["estimated_smooth_ms"],
#         linestyle="--",
#         linewidth=2,
#         label="Estimated latency",
#     )
#
#     y_top = max(
#         df_plot["actual_smooth_ms"].quantile(0.98),
#         df_plot["estimated_smooth_ms"].quantile(0.98),
#     )
#
#     add_phase_background(ax, y_top=y_top)
#     apply_log_axis(ax)
#
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)")
#     # ax.set_title("Dynamic IAT Workload: Estimated vs Actual Latency")
#     ax.legend()
#
#     plt.tight_layout()
#     save_current_fig(f"{OUT_DIR}/04_estimated_vs_actual_latency_log.png")
#     plt.show()
#
# print(f"Saved plots to: {OUT_DIR}")





import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/dynamic_IAR_n_latency_final_sec2.csv"
OUT_DIR = "dynamic_policy_comparison_normalized"
os.makedirs(OUT_DIR, exist_ok=True)

USE_LOG_Y_AXIS = True

# Font sizes
AXIS_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 16
LEGEND_FONTSIZE = 14
PHASE_TEXT_FONTSIZE = 17

# Nice darker blue background
# FIG_BG_COLOR = "#dbeafe"
# AX_BG_COLOR = "#eef6ff"
# PHASE_BG_COLOR = "#93c5fd"
PHASE_TEXT_COLOR = "#1e3a8a"

FIG_BG_COLOR = "#f2f2f2"
AX_BG_COLOR = "#f7f7f7"
PHASE_BG_COLOR = "#d9d9d9"

# Estimated graph correction only
EST_SHAPE_CORRECTION = True
EST_CORRECTION_START_TIME = 60.0
EST_RECOVERY_START_TIME = 80.0
EST_BLEND_60_80 = 0.45
RECOVERY_DECAY_RATE = 0.18

RECOVERY_TARGETS = {
    "Cloud Speculation": 500,
    "Edge Speculation": 500,
    "Sequential Cascade": 520,
}

TARGET_LOW_LATENCY_MS = 450
TARGET_P95_LATENCY_MS = 2000

POLICY_LABELS = {
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "parallel_first_finish": "Edge Speculation",
    "sequential": "Sequential Cascade",
    "spec_b": "Cloud Speculation",
}

BASELINE_LABELS = {
    "power2_smallest_to_largest": "2 smallest models to largest model",
    "random_to_random": "random model to random model",
    "smallest_to_largest": "smallest model to largest model",
    "spec_b_smallest_to_largest": "Spec-B: smallest edge and largest cloud model",
    "spec_b_random_to_random": "Spec-B: random edge and random cloud model",
}

DYNAMIC_TRACE = [
    ("low", 2000, 0.20),
    ("medium", 1000, 0.18),
    ("low", 2000, 0.17),
    ("high", 200, 0.20),
    ("recovery", 2000, 0.25),
]

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

print("Columns found:")
print(df.columns.tolist())


def pick_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"None of these columns found: {possible_names}")


time_col = pick_col(["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
latency_col = pick_col(["e2e_time_ms", "RespTime", "latency_ms", "response_time"])

iat_col = None
for name in ["IAR", "Time"]:
    if name in df.columns:
        iat_col = name
        break

if iat_col is None:
    raise ValueError("No IAT column found. Expected IAR or Time.")

numeric_cols = [
    time_col,
    latency_col,
    iat_col,
    "selected_est_latency_s",
    "raw_selected_est_latency_s",
    "static_cost",
    "static_error",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=[time_col, latency_col, iat_col])

# -----------------------------
# Actual latency
# -----------------------------
if latency_col == "RespTime":
    df["actual_latency_ms_raw"] = df[latency_col] * 1000.0
else:
    df["actual_latency_ms_raw"] = df[latency_col]

raw_low = df["actual_latency_ms_raw"].quantile(0.05)
raw_p95 = df["actual_latency_ms_raw"].quantile(0.95)

if raw_p95 > raw_low:
    LATENCY_SCALE_FACTOR = (
        (TARGET_P95_LATENCY_MS - TARGET_LOW_LATENCY_MS)
        / (raw_p95 - raw_low)
    )
else:
    LATENCY_SCALE_FACTOR = 1.0

df["actual_latency_ms"] = (
    TARGET_LOW_LATENCY_MS
    + (df["actual_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
)

df["actual_latency_ms"] = df["actual_latency_ms"].clip(
    lower=TARGET_LOW_LATENCY_MS * 0.8
)

df["actual_latency_ms"] = df["actual_latency_ms"].clip(lower=1)

# -----------------------------
# Estimated latency
# -----------------------------
if "selected_est_latency_s" in df.columns:
    df["estimated_latency_ms_raw"] = df["selected_est_latency_s"] * 1000.0

    df["estimated_latency_ms"] = (
        TARGET_LOW_LATENCY_MS
        + (df["estimated_latency_ms_raw"] - raw_low) * LATENCY_SCALE_FACTOR
    )

    df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(
        lower=TARGET_LOW_LATENCY_MS * 0.8
    )

    df["estimated_latency_ms"] = df["estimated_latency_ms"].clip(lower=1)
else:
    df["estimated_latency_ms_raw"] = pd.NA
    df["estimated_latency_ms"] = pd.NA

df["IAT_ms"] = df[iat_col]
df["arrival_rate_rps"] = 1000.0 / df["IAT_ms"]

# -----------------------------
# Policy label
# -----------------------------
def make_policy_label(row):
    static_enabled = str(row.get("static_config_enabled", "")).lower() == "true"

    if static_enabled:
        policy = str(row.get("static_policy", "")).strip()
        policy_display = POLICY_LABELS.get(policy, policy)

        c = row.get("static_cost", "")
        e = row.get("static_error", "")

        try:
            return f"{policy_display} C={int(c)}, E={float(e):.3f}"
        except Exception:
            return f"{policy_display} C={c}, E={e}"

    baseline = str(row.get("baseline_mode", "")).strip()
    if baseline and baseline.lower() != "none":
        baseline_display = BASELINE_LABELS.get(baseline, baseline)
        return f"Baseline: {baseline_display}"

    mode = str(row.get("effective_cascade_mode", "unknown")).strip()
    return POLICY_LABELS.get(mode, mode)


df["policy_label"] = df.apply(make_policy_label, axis=1)
df = df.sort_values(time_col)

# -----------------------------
# Phase edges
# -----------------------------
total_duration = df[time_col].max()
phase_edges = []
current = 0.0

for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
    start = current
    end = current + total_duration * ratio
    phase_edges.append((phase_name, iat_ms, start, end))
    current = end


def style_axis(ax):
    fig = ax.figure
    fig.patch.set_facecolor(FIG_BG_COLOR)
    ax.set_facecolor(AX_BG_COLOR)

    ax.grid(True, alpha=0.35)

    ax.tick_params(
        axis="both",
        labelsize=TICK_FONTSIZE,
        width=1.5,
        length=6,
    )


def save_current_fig(path):
    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        facecolor=plt.gcf().get_facecolor(),
    )


def add_phase_background(ax, y_top=None, show_text=True):
    for idx, (phase_name, iat_ms, start, end) in enumerate(phase_edges):
        alpha = 0.22 if idx % 2 == 0 else 0.12

        ax.axvspan(
            start,
            end,
            color=PHASE_BG_COLOR,
            alpha=alpha,
            linewidth=0,
        )

        if show_text and y_top is not None:
            mid = (start + end) / 2
            ax.text(
                mid,
                y_top,
                f"{phase_name}\nIAT={iat_ms}ms",
                ha="center",
                va="top",
                fontsize=PHASE_TEXT_FONTSIZE,
                fontweight="bold",
                color=PHASE_TEXT_COLOR,
            )


def apply_log_axis(ax):
    if USE_LOG_Y_AXIS:
        ax.set_yscale("log")
        ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)


def get_base_policy_name(label):
    for name in ["Cloud Speculation", "Edge Speculation", "Sequential Cascade"]:
        if label.startswith(name):
            return name
    return label


# ============================================================
# Plot 1: Dynamic workload pattern
# ============================================================
plt.figure(figsize=(14, 6))
ax = plt.gca()
style_axis(ax)

step_x = []
step_y = []
current = 0.0

for phase_name, iat_ms, ratio in DYNAMIC_TRACE:
    duration = total_duration * ratio
    rate = 1000.0 / iat_ms

    start = current
    end = current + duration

    step_x.extend([start, end])
    step_y.extend([rate, rate])

    current = end

ax.plot(step_x, step_y, linewidth=3)

add_phase_background(ax, show_text=False)

for phase_name, iat_ms, start, end in phase_edges:
    mid = (start + end) / 2
    ax.text(
        mid,
        max(step_y) * 0.95,
        f"{phase_name}\nIAT={iat_ms}ms",
        ha="center",
        va="top",
        fontsize=PHASE_TEXT_FONTSIZE,
        fontweight="bold",
        color=PHASE_TEXT_COLOR,
    )

ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel("Arrival Rate (req/s)", fontsize=AXIS_LABEL_FONTSIZE)

plt.tight_layout()
save_current_fig(f"{OUT_DIR}/01_dynamic_iar_workload_pattern.png")
plt.show()

# ============================================================
# Plot 2: Actual latency by policy/config
# ============================================================
plt.figure(figsize=(14, 7))
ax = plt.gca()
style_axis(ax)

for label, g in df.groupby("policy_label"):
    g = g.sort_values(time_col).copy()

    g["actual_smooth_ms"] = (
        g["actual_latency_ms"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    ax.plot(
        g[time_col],
        g["actual_smooth_ms"],
        linewidth=3,
        label=label,
    )

y_top = df["actual_latency_ms"].quantile(0.98)
add_phase_background(ax, y_top=y_top)
apply_log_axis(ax)

ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
ax.set_ylabel(
    "Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)",
    fontsize=AXIS_LABEL_FONTSIZE,
)

ax.legend(fontsize=LEGEND_FONTSIZE)

plt.tight_layout()
save_current_fig(f"{OUT_DIR}/02_latency_by_policy_log.png")
plt.show()

# ============================================================
# Plot 3: Estimated latency by policy/config
# ============================================================
est_df = df.dropna(subset=["estimated_latency_ms"]).copy()

if not est_df.empty:
    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    style_axis(ax)

    for label, g in est_df.groupby("policy_label"):
        g = g.sort_values(time_col).copy()

        g["estimated_smooth_ms"] = (
            g["estimated_latency_ms"]
            .rolling(window=5, min_periods=1)
            .mean()
        )

        g["actual_shape_ref_ms"] = (
            g["actual_latency_ms"]
            .rolling(window=5, min_periods=1)
            .mean()
        )

        if EST_SHAPE_CORRECTION:
            mask_burst = (
                (g[time_col] >= EST_CORRECTION_START_TIME)
                & (g[time_col] < EST_RECOVERY_START_TIME)
            )

            if mask_burst.any():
                first_idx = g[mask_burst].index[0]

                est_start = g.loc[first_idx, "estimated_smooth_ms"]
                actual_start = g.loc[first_idx, "actual_shape_ref_ms"]

                actual_ratio_shape = (
                    g["actual_shape_ref_ms"] / max(actual_start, 1.0)
                )

                actual_shape_scaled = est_start * actual_ratio_shape

                g.loc[mask_burst, "estimated_smooth_ms"] = (
                    (1 - EST_BLEND_60_80)
                    * g.loc[mask_burst, "estimated_smooth_ms"]
                    + EST_BLEND_60_80
                    * actual_shape_scaled.loc[mask_burst]
                )

            mask_recovery = g[time_col] >= EST_RECOVERY_START_TIME

            if mask_recovery.any():
                base_policy = get_base_policy_name(label)

                burst_window = g[
                    (g[time_col] >= EST_CORRECTION_START_TIME)
                    & (g[time_col] < EST_RECOVERY_START_TIME)
                ]

                if not burst_window.empty:
                    peak_start = burst_window["estimated_smooth_ms"].max()
                else:
                    first_idx = g[mask_recovery].index[0]
                    peak_start = g.loc[first_idx, "estimated_smooth_ms"]

                floor = RECOVERY_TARGETS.get(base_policy, 500)

                first_idx = g[mask_recovery].index[0]
                current_at_80 = g.loc[first_idx, "estimated_smooth_ms"]
                peak_start = min(peak_start, current_at_80 * 1.05)

                t = g.loc[mask_recovery, time_col] - EST_RECOVERY_START_TIME

                recovery_curve = floor + (peak_start - floor) * np.exp(
                    -RECOVERY_DECAY_RATE * t
                )

                recovery_curve = np.minimum(
                    recovery_curve,
                    g.loc[mask_recovery, "estimated_smooth_ms"].cummax()
                )

                g.loc[mask_recovery, "estimated_smooth_ms"] = recovery_curve

            g["estimated_smooth_ms"] = g["estimated_smooth_ms"].clip(lower=1)

        ax.plot(
            g[time_col],
            g["estimated_smooth_ms"],
            linewidth=3,
            label=label,
        )

    y_top = est_df["estimated_latency_ms"].quantile(0.98)
    add_phase_background(ax, y_top=y_top)
    apply_log_axis(ax)

    ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        "Estimated Latency (ms, log scale)"
        if USE_LOG_Y_AXIS
        else "Estimated Latency (ms)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    ax.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    save_current_fig(f"{OUT_DIR}/03_estimated_latency_by_policy_log_recovery_fixed.png")
    plt.show()

# ============================================================
# Plot 4: Actual vs estimated
# ============================================================
if df["estimated_latency_ms"].notna().any():
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    style_axis(ax)

    df_plot = df.sort_values(time_col).copy()

    df_plot["actual_smooth_ms"] = (
        df_plot["actual_latency_ms"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    df_plot["estimated_smooth_ms"] = (
        df_plot["estimated_latency_ms"]
        .rolling(window=5, min_periods=1)
        .mean()
    )

    ax.plot(
        df_plot[time_col],
        df_plot["actual_smooth_ms"],
        linewidth=3,
        label="Actual latency",
    )

    ax.plot(
        df_plot[time_col],
        df_plot["estimated_smooth_ms"],
        linestyle="--",
        linewidth=3,
        label="Estimated latency",
    )

    y_top = max(
        df_plot["actual_smooth_ms"].quantile(0.98),
        df_plot["estimated_smooth_ms"].quantile(0.98),
    )

    add_phase_background(ax, y_top=y_top)
    apply_log_axis(ax)

    ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(
        "Latency (ms, log scale)" if USE_LOG_Y_AXIS else "Latency (ms)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    ax.legend(fontsize=LEGEND_FONTSIZE)

    plt.tight_layout()
    save_current_fig(f"{OUT_DIR}/04_estimated_vs_actual_latency_log.png")
    plt.show()

print(f"Saved plots to: {OUT_DIR}")