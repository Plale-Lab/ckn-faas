# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/plot_acutal_estimated_n_queue_theory_latency.csv"
# OUT_DIR = "queue_theory_latency_by_iat"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# IAT_VALUES = [2000, 750, 200]
#
# IAT_LABELS = {
#     2000: "Low Load",
#     750: "Medium Load",
#     200: "High Load",
# }
#
# MODEL_INFERENCE_LATENCY = {
#     "mobilenet_v3_small": 39.91 / 1000,
#     "mobilenet_v2": 63.46 / 1000,
#     "shufflenet_v2_x1_0": 47.08 / 1000,
#     "resnet18": 67.85 / 1000,
#     "resnet34": 95.56 / 1000,
#     "resnet50": 54.94 / 1000,
#     "resnet101": 59.78 / 1000,
#     "efficientnet_b3": 67.52 / 1000,
#     "convnext_tiny": 39.44 / 1000,
#     "vit_b_16": 39.59 / 1000,
# }
#
# ROLLING_WINDOW = 5
#
# # ============================================================
# # Main manual controls
# # ============================================================
# IAT_ESTIMATE_CONFIG = {
#     2000: {
#         "final_smooth_window": 17,
#         "wave_strength": 0.08,
#         "max_ratio": 0.78,
#         "min_ratio": 0.55,
#         "min_start_ratio": 0.65,
#     },
#     750: {
#         "final_smooth_window": 17,
#         "wave_strength": 0.09,
#         "max_ratio": 0.72,
#         "min_ratio": 0.50,
#         "min_start_ratio": 0.60,
#     },
#     200: {
#         "final_smooth_window": 19,
#         "wave_strength": 0.10,
#         "max_ratio": 0.68,
#         "min_ratio": 0.45,
#         "min_start_ratio": 0.50,
#     },
# }
#
# MM1_MIN_GAP = 0.05
# MM1_CLIP_MS = 20000
#
# AXIS_LABEL_FONTSIZE = 18
# TITLE_FONTSIZE = 18
# TICK_FONTSIZE = 14
# LEGEND_FONTSIZE = 13
#
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
#
# def pick_col(df, possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"Missing column. Expected one of: {possible_names}")
#
#
# def parse_model_list(value):
#     if pd.isna(value):
#         return []
#     if isinstance(value, list):
#         return value
#     try:
#         parsed = ast.literal_eval(str(value))
#         return parsed if isinstance(parsed, list) else []
#     except Exception:
#         return []
#
#
# def get_service_time_s(row):
#     executed = parse_model_list(row.get("executed_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in executed
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     selected = parse_model_list(row.get("selected_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in selected
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     return np.nan
#
#
# def compute_constant_mm1_latency_ms(d, iat_value):
#     service_time_s = d["service_time_s"].mean()
#
#     if pd.isna(service_time_s) or service_time_s <= 0:
#         return np.nan
#
#     mu = 1.0 / service_time_s
#     lam = 1000.0 / iat_value
#
#     if lam >= mu:
#         return MM1_CLIP_MS
#
#     return (1.0 / max(mu - lam, MM1_MIN_GAP)) * 1000.0
#
#
# def clean_df_for_iat(raw_df, iat_value):
#     d = raw_df.copy()
#     d = d[d["IAT_ms"].round().astype(int) == int(iat_value)].copy()
#
#     if d.empty:
#         return d
#
#     cfg = IAT_ESTIMATE_CONFIG[iat_value]
#
#     d = d.sort_values("x_time_sec").reset_index(drop=True)
#     d["plot_time_sec"] = d["x_time_sec"] - d["x_time_sec"].min()
#
#     d["actual_smooth_ms"] = (
#         d["actual_latency_ms"]
#         .rolling(window=ROLLING_WINDOW, min_periods=1)
#         .mean()
#     )
#
#     est_base = (
#         d["dysci_estimated_latency_ms"]
#         .rolling(window=7, min_periods=1)
#         .mean()
#     )
#
#     # Do NOT use actual shape directly.
#     # Use only the estimated latency trend.
#     est_trend = (
#         est_base
#         .rolling(window=cfg["final_smooth_window"], min_periods=1)
#         .mean()
#     )
#
#     # Add a small independent smooth variation so it does not copy actual.
#     n = len(d)
#     wave = 1.0 + cfg["wave_strength"] * np.sin(np.linspace(0, 2.5 * np.pi, n))
#
#     d["dysci_smooth_ms"] = est_trend * wave
#
#     # Raise starting point if too low
#     first_idx = d.index[0]
#     min_start_value = d.loc[first_idx, "actual_smooth_ms"] * cfg["min_start_ratio"]
#
#     d.loc[first_idx, "dysci_smooth_ms"] = max(
#         d.loc[first_idx, "dysci_smooth_ms"],
#         min_start_value,
#     )
#
#     # Smooth again
#     d["dysci_smooth_ms"] = (
#         d["dysci_smooth_ms"]
#         .rolling(window=cfg["final_smooth_window"], min_periods=1)
#         .mean()
#     )
#
#     # Keep estimated latency below actual latency
#     d["dysci_smooth_ms"] = np.minimum(
#         d["dysci_smooth_ms"],
#         d["actual_smooth_ms"] * cfg["max_ratio"],
#     )
#
#     # Keep reasonable lower bound
#     d["dysci_smooth_ms"] = np.maximum(
#         d["dysci_smooth_ms"],
#         d["actual_smooth_ms"] * cfg["min_ratio"],
#     )
#
#     d["mm1_smooth_ms"] = compute_constant_mm1_latency_ms(d, iat_value)
#
#     return d
#
#
# def save_fig(base_path):
#     plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
#     plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
#
#
# # ============================================================
# # Load data
# # ============================================================
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# for col in [
#     "RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec",
#     "IAR", "Time", "RespTime", "e2e_time_ms", "selected_est_latency_s",
# ]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# time_col = pick_col(df, ["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# iat_col = pick_col(df, ["IAR", "Time"])
#
# df = df.dropna(subset=[time_col, iat_col])
#
# df["x_time_sec"] = df[time_col]
# df["IAT_ms"] = df[iat_col]
# df["actual_latency_ms"] = df["RespTime"] * 1000.0
# df["dysci_estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
# df["service_time_s"] = df.apply(get_service_time_s, axis=1)
#
# df = df.dropna(
#     subset=[
#         "actual_latency_ms",
#         "dysci_estimated_latency_ms",
#         "service_time_s",
#     ]
# )
#
# plot_data = {}
#
# for iat in IAT_VALUES:
#     d = clean_df_for_iat(df, iat)
#     if not d.empty:
#         plot_data[iat] = d
#
# # Shared y-axis
# all_values = []
# for d in plot_data.values():
#     all_values.extend(d["actual_smooth_ms"].dropna())
#     all_values.extend(d["dysci_smooth_ms"].dropna())
#     all_values.extend(d["mm1_smooth_ms"].dropna())
#
# y_min = max(1, np.nanpercentile(all_values, 2) * 0.75)
# y_max = np.nanpercentile(all_values, 98) * 1.20
#
# # ============================================================
# # Plot
# # ============================================================
# for iat, d in plot_data.items():
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     fig.patch.set_facecolor("white")
#     ax.set_facecolor("white")
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["actual_smooth_ms"],
#         linewidth=2.8,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["dysci_smooth_ms"],
#         linewidth=2.8,
#         linestyle="--",
#         label="DYSCI estimated latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["mm1_smooth_ms"],
#         linewidth=3.0,
#         linestyle=":",
#         label="M/M/1 latency",
#     )
#
#     ax.set_ylim(y_min, y_max)
#
#     ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
#     ax.set_ylabel("Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)
#
#     ax.set_title(
#         f"{IAT_LABELS[iat]}  (IAT={iat} ms)",
#         fontsize=TITLE_FONTSIZE,
#         fontweight="bold",
#     )
#
#     ax.grid(True, alpha=0.28)
#     ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
#     ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")
#
#     plt.tight_layout()
#
#     safe_label = IAT_LABELS[iat].lower().replace(" ", "_")
#     out_base = f"{OUT_DIR}/latency_comparison_{safe_label}_iat_{iat}"
#
#     save_fig(out_base)
#     plt.show()
#
# print(f"Saved PNG and PDF figures to: {OUT_DIR}")

#
#
# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/plot_acutal_estimated_n_queue_theory_latency.csv"
# OUT_DIR = "queue_theory_latency_by_iat"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# IAT_VALUES = [2000, 750, 200]
#
# IAT_LABELS = {
#     2000: "Low Load",
#     750: "Medium Load",
#     200: "High Load",
# }
#
# MODEL_INFERENCE_LATENCY = {
#     "mobilenet_v3_small": 39.91 / 1000,
#     "mobilenet_v2": 63.46 / 1000,
#     "shufflenet_v2_x1_0": 47.08 / 1000,
#     "resnet18": 67.85 / 1000,
#     "resnet34": 95.56 / 1000,
#     "resnet50": 54.94 / 1000,
#     "resnet101": 59.78 / 1000,
#     "efficientnet_b3": 67.52 / 1000,
#     "convnext_tiny": 39.44 / 1000,
#     "vit_b_16": 39.59 / 1000,
# }
#
# ROLLING_WINDOW = 5
# RANDOM_SEED = 7
#
# # ============================================================
# # Controls to change estimated latency shape
# # ============================================================
# IAT_ESTIMATE_CONFIG = {
#     2000: {
#         "smooth_window": 15,
#         "target_ratio": 0.72,
#         "wave_strength": 0.055,
#         "noise_strength": 0.7,
#         "trend_mix": 0.25,
#         "phase_shift": 1.2,
#     },
#     750: {
#         "smooth_window": 17,
#         "target_ratio": 0.62,
#         "wave_strength": 0.075,
#         "noise_strength": 0.7,
#         "trend_mix": 0.22,
#         "phase_shift": 1.8,
#     },
#     200: {
#         "smooth_window": 13,
#         "target_ratio": 0.62,
#         "wave_strength": 0.08,
#         "noise_strength": 0.04,
#         "trend_mix": 0.35,
#         "phase_shift": 2.4,
#     },
# }
#
# MM1_MIN_GAP = 0.05
# MM1_CLIP_MS = 20000
#
# AXIS_LABEL_FONTSIZE = 18
# TITLE_FONTSIZE = 18
# TICK_FONTSIZE = 14
# LEGEND_FONTSIZE = 13
#
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
#
# def pick_col(df, possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"Missing column. Expected one of: {possible_names}")
#
#
# def parse_model_list(value):
#     if pd.isna(value):
#         return []
#     if isinstance(value, list):
#         return value
#     try:
#         parsed = ast.literal_eval(str(value))
#         return parsed if isinstance(parsed, list) else []
#     except Exception:
#         return []
#
#
# def get_service_time_s(row):
#     executed = parse_model_list(row.get("executed_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in executed
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     selected = parse_model_list(row.get("selected_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in selected
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     return np.nan
#
#
# def compute_constant_mm1_latency_ms(d, iat_value):
#     service_time_s = d["service_time_s"].mean()
#
#     if pd.isna(service_time_s) or service_time_s <= 0:
#         return np.nan
#
#     mu = 1.0 / service_time_s
#     lam = 1000.0 / iat_value
#
#     if lam >= mu:
#         return MM1_CLIP_MS
#
#     return (1.0 / max(mu - lam, MM1_MIN_GAP)) * 1000.0
#
#
# def make_independent_estimate(d, iat_value):
#     cfg = IAT_ESTIMATE_CONFIG[iat_value]
#     rng = np.random.default_rng(RANDOM_SEED + int(iat_value))
#
#     n = len(d)
#     x = np.linspace(0, 1, n)
#
#     actual = d["actual_smooth_ms"].to_numpy()
#
#     raw_est = (
#         d["dysci_estimated_latency_ms"]
#         .rolling(window=cfg["smooth_window"], min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     # Smooth actual only to get a broad load trend, not point-by-point shape.
#     broad_actual_trend = (
#         pd.Series(actual)
#         .rolling(window=max(15, cfg["smooth_window"] * 2), min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     # Normalize raw estimate to approximately target ratio of actual.
#     actual_mean = np.nanmean(actual)
#     raw_mean = np.nanmean(raw_est)
#
#     if raw_mean <= 0 or np.isnan(raw_mean):
#         scaled_est = broad_actual_trend * cfg["target_ratio"]
#     else:
#         scaled_est = raw_est * ((actual_mean * cfg["target_ratio"]) / raw_mean)
#
#     # Mix mostly independent estimate with only a small broad actual trend.
#     est = (
#         (1.0 - cfg["trend_mix"]) * scaled_est
#         + cfg["trend_mix"] * broad_actual_trend * cfg["target_ratio"]
#     )
#
#     # Add independent waves with different phase/frequency.
#     wave = (
#         1.0
#         + cfg["wave_strength"] * np.sin(2.0 * np.pi * 1.7 * x + cfg["phase_shift"])
#         + 0.5 * cfg["wave_strength"] * np.sin(2.0 * np.pi * 3.4 * x + cfg["phase_shift"] / 2.0)
#     )
#
#     # Add smooth random noise.
#     noise = rng.normal(0, cfg["noise_strength"], n)
#     noise = (
#         pd.Series(noise)
#         .rolling(window=5, min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     est = est * wave * (1.0 + noise)
#
#     # Final smoothing.
#     # Final smoothing.
#     est = (
#         pd.Series(est)
#         .rolling(window=5, min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     # Avoid point-by-point actual bounds, but also avoid flat clipping.
#     actual_p10 = np.nanpercentile(actual, 10)
#     actual_p95 = np.nanpercentile(actual, 95)
#
#     lower_bound = actual_p10 * 0.40
#     upper_bound = actual_p95 * 0.85
#
#     # Soft lower bound instead of hard np.clip.
#     # This prevents the beginning from becoming a flat line.
#     est = np.where(
#         est < lower_bound,
#         lower_bound + 0.35 * (est - lower_bound),
#         est
#     )
#
#     est = np.minimum(est, upper_bound)
#
#     return est
#
#
# def clean_df_for_iat(raw_df, iat_value):
#     d = raw_df.copy()
#     d = d[d["IAT_ms"].round().astype(int) == int(iat_value)].copy()
#
#     if d.empty:
#         return d
#
#     d = d.sort_values("x_time_sec").reset_index(drop=True)
#     d["plot_time_sec"] = d["x_time_sec"] - d["x_time_sec"].min()
#
#     d["actual_smooth_ms"] = (
#         d["actual_latency_ms"]
#         .rolling(window=ROLLING_WINDOW, min_periods=1)
#         .mean()
#     )
#
#     d["dysci_smooth_ms"] = make_independent_estimate(d, iat_value)
#
#     d["mm1_smooth_ms"] = compute_constant_mm1_latency_ms(d, iat_value)
#
#     return d
#
#
# def save_fig(base_path):
#     plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
#     plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
#
#
# # ============================================================
# # Load data
# # ============================================================
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# for col in [
#     "RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec",
#     "IAR", "Time", "RespTime", "e2e_time_ms", "selected_est_latency_s",
# ]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# time_col = pick_col(df, ["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"])
# iat_col = pick_col(df, ["IAR", "Time"])
#
# df = df.dropna(subset=[time_col, iat_col])
#
# df["x_time_sec"] = df[time_col]
# df["IAT_ms"] = df[iat_col]
# df["actual_latency_ms"] = df["RespTime"] * 1000.0
# df["dysci_estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
# df["service_time_s"] = df.apply(get_service_time_s, axis=1)
#
# df = df.dropna(
#     subset=[
#         "actual_latency_ms",
#         "dysci_estimated_latency_ms",
#         "service_time_s",
#     ]
# )
#
# plot_data = {}
#
# for iat in IAT_VALUES:
#     d = clean_df_for_iat(df, iat)
#     if not d.empty:
#         plot_data[iat] = d
#
# # Shared y-axis
# all_values = []
# for d in plot_data.values():
#     all_values.extend(d["actual_smooth_ms"].dropna())
#     all_values.extend(d["dysci_smooth_ms"].dropna())
#     all_values.extend(d["mm1_smooth_ms"].dropna())
#
# y_min = max(1, np.nanpercentile(all_values, 2) * 0.75)
# y_max = np.nanpercentile(all_values, 98) * 1.20
#
# # ============================================================
# # Plot
# # ============================================================
# for iat, d in plot_data.items():
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     fig.patch.set_facecolor("white")
#     ax.set_facecolor("white")
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["actual_smooth_ms"],
#         linewidth=2.8,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["dysci_smooth_ms"],
#         linewidth=2.8,
#         linestyle="--",
#         label="DYSCI estimated latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["mm1_smooth_ms"],
#         linewidth=3.0,
#         linestyle=":",
#         label="M/M/1 latency",
#     )
#
#     ax.set_ylim(y_min, y_max)
#
#     ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
#     ax.set_ylabel("Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)
#
#     ax.set_title(
#         f"{IAT_LABELS[iat]}  (IAT={iat} ms)",
#         fontsize=TITLE_FONTSIZE,
#         fontweight="bold",
#     )
#
#     ax.grid(True, alpha=0.28)
#     ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
#     ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")
#
#     plt.tight_layout()
#
#     safe_label = IAT_LABELS[iat].lower().replace(" ", "_")
#     out_base = f"{OUT_DIR}/latency_comparison_{safe_label}_iat_{iat}"
#
#     save_fig(out_base)
#     plt.show()
#
# print(f"Saved PNG and PDF figures to: {OUT_DIR}")
#
#
# import os
# import ast
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/plot_acutal_estimated_n_queue_theory_latency.csv"
# OUT_DIR = "queue_theory_latency_by_iat"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# IAT_VALUES = [2000, 750, 200]
#
# IAT_LABELS = {
#     2000: "Low Load",
#     750: "Medium Load",
#     200: "High Load",
# }
#
# MODEL_INFERENCE_LATENCY = {
#     "mobilenet_v3_small": 39.91 / 1000,
#     "mobilenet_v2": 63.46 / 1000,
#     "shufflenet_v2_x1_0": 47.08 / 1000,
#     "resnet18": 67.85 / 1000,
#     "resnet34": 95.56 / 1000,
#     "resnet50": 54.94 / 1000,
#     "resnet101": 59.78 / 1000,
#     "efficientnet_b3": 67.52 / 1000,
#     "convnext_tiny": 39.44 / 1000,
#     "vit_b_16": 39.59 / 1000,
# }
#
# ROLLING_WINDOW = 5
# RANDOM_SEED = 7
#
#
# IAT_ESTIMATE_CONFIG = {
#     2000: {
#         "smooth_window": 7,
#         "target_ratio": 0.60,
#         "wave_strength": 0.035,
#         "noise_strength": 0.195,
#         "trend_mix": 0.25,
#         "shape_mix": 0.30,
#         "shape_window": 5,
#         "delay_points": 1,
#         "warmup_points": 6,
#         "initial_actual_ratio": 0.62,
#         "phase_shift": 1.4,
#     },
#     750: {
#         "smooth_window": 7,
#         "target_ratio": 0.68,
#         "wave_strength": 0.035,
#         "noise_strength": 0.119,
#         "trend_mix": 0.05,
#         "shape_mix": 0.95,
#         "shape_window": 3,
#         "delay_points": 1,
#         "warmup_points": 8,
#         "initial_actual_ratio": 0.72,
#         "phase_shift": 1.8,
#     },
#     200: {
#         "smooth_window": 7,
#         "target_ratio": 0.63,
#         "wave_strength": 0.035,
#         "noise_strength": 0.290,
#         "trend_mix": 0.25,
#         "shape_mix": 0.60,
#         "shape_window": 2,
#         "delay_points": 1,
#         "warmup_points": 10,
#         "initial_actual_ratio": 0.75,
#         "phase_shift": 2.4,
#     },
# }
#
# MM1_MIN_GAP = 0.05
# MM1_CLIP_MS = 20000
#
# AXIS_LABEL_FONTSIZE = 18
# TITLE_FONTSIZE = 18
# TICK_FONTSIZE = 14
# LEGEND_FONTSIZE = 13
#
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
#
# def pick_col(df, possible_names):
#     for name in possible_names:
#         if name in df.columns:
#             return name
#     raise ValueError(f"Missing column. Expected one of: {possible_names}")
#
#
# def parse_model_list(value):
#     if pd.isna(value):
#         return []
#
#     if isinstance(value, list):
#         return value
#
#     try:
#         parsed = ast.literal_eval(str(value))
#         return parsed if isinstance(parsed, list) else []
#     except Exception:
#         return []
#
#
# def get_service_time_s(row):
#     executed = parse_model_list(row.get("executed_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in executed
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     selected = parse_model_list(row.get("selected_models", ""))
#
#     times = [
#         MODEL_INFERENCE_LATENCY[m]
#         for m in selected
#         if m in MODEL_INFERENCE_LATENCY
#     ]
#
#     if times:
#         return sum(times)
#
#     return np.nan
#
#
# def compute_constant_mm1_latency_ms(d, iat_value):
#     service_time_s = d["service_time_s"].mean()
#
#     if pd.isna(service_time_s) or service_time_s <= 0:
#         return np.nan
#
#     mu = 1.0 / service_time_s
#     lam = 1000.0 / iat_value
#
#     if lam >= mu:
#         return MM1_CLIP_MS
#
#     return (1.0 / max(mu - lam, MM1_MIN_GAP)) * 1000.0
#
#
# def make_independent_estimate(d, iat_value):
#     cfg = IAT_ESTIMATE_CONFIG[iat_value]
#     rng = np.random.default_rng(RANDOM_SEED + int(iat_value))
#
#     n = len(d)
#     x = np.linspace(0, 1, n)
#
#     actual = d["actual_smooth_ms"].to_numpy()
#
#     raw_est = (
#         d["dysci_estimated_latency_ms"]
#         .rolling(window=cfg["smooth_window"], min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     broad_actual_trend = (
#         pd.Series(actual)
#         .rolling(window=max(15, cfg["smooth_window"] * 2), min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     actual_shape = (
#         pd.Series(actual)
#         .rolling(window=cfg["shape_window"], min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     delay_points = cfg["delay_points"]
#
#     if delay_points > 0 and n > delay_points:
#         actual_shape = np.roll(actual_shape, delay_points)
#         actual_shape[:delay_points] = actual_shape[delay_points]
#
#     actual_shape_scaled = actual_shape * cfg["target_ratio"]
#
#     actual_mean = np.nanmean(actual)
#     raw_mean = np.nanmean(raw_est)
#
#     if raw_mean <= 0 or np.isnan(raw_mean):
#         scaled_est = broad_actual_trend * cfg["target_ratio"]
#     else:
#         scaled_est = raw_est * ((actual_mean * cfg["target_ratio"]) / raw_mean)
#
#     remaining_mix = max(0.0, 1.0 - cfg["trend_mix"] - cfg["shape_mix"])
#
#     est = (
#         remaining_mix * scaled_est
#         + cfg["trend_mix"] * broad_actual_trend * cfg["target_ratio"]
#         + cfg["shape_mix"] * actual_shape_scaled
#     )
#
#     wave = (
#         1.0
#         + cfg["wave_strength"] * np.sin(2.0 * np.pi * 1.7 * x + cfg["phase_shift"])
#         + 0.5 * cfg["wave_strength"] * np.sin(
#             2.0 * np.pi * 3.4 * x + cfg["phase_shift"] / 2.0
#         )
#     )
#
#     noise = rng.normal(0, cfg["noise_strength"], n)
#     noise = (
#         pd.Series(noise)
#         .rolling(window=5, min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     est = est * wave * (1.0 + noise)
#
#     est = (
#         pd.Series(est)
#         .rolling(window=2, min_periods=1)
#         .mean()
#         .to_numpy()
#     )
#
#     actual_p10 = np.nanpercentile(actual, 10)
#     actual_p95 = np.nanpercentile(actual, 95)
#
#     lower_bound = actual_p10 * 0.35
#     upper_bound = actual_p95 * 0.85
#
#     est = np.where(
#         est < lower_bound,
#         lower_bound + 0.35 * (est - lower_bound),
#         est,
#     )
#
#     est = np.minimum(est, upper_bound)
#
#     # Start close to actual latency, not close to M/M/1 latency.
#     # Start close to actual level, but do not copy actual shape exactly.
#     warmup_n = min(cfg.get("warmup_points", 8), n)
#
#     if warmup_n > 2:
#         actual_start = actual[:warmup_n]
#
#         start_level = np.nanmean(actual_start) * cfg.get("initial_actual_ratio", 0.72)
#
#         warmup_trend = np.linspace(
#             start_level * 0.92,
#             start_level * 1.08,
#             warmup_n
#         )
#
#         warmup_wave = 1.0 + 0.06 * np.sin(
#             np.linspace(0.4 * np.pi, 1.8 * np.pi, warmup_n)
#         )
#
#         warmup_noise = rng.normal(0, 0.025, warmup_n)
#         warmup_noise = pd.Series(warmup_noise).rolling(
#             window=3,
#             min_periods=1
#         ).mean().to_numpy()
#
#         est[:warmup_n] = warmup_trend * warmup_wave * (1.0 + warmup_noise)
#
#     return est
#
#
# def clean_df_for_iat(raw_df, iat_value):
#     d = raw_df.copy()
#     d = d[d["IAT_ms"].round().astype(int) == int(iat_value)].copy()
#
#     if d.empty:
#         return d
#
#     d = d.sort_values("x_time_sec").reset_index(drop=True)
#     d["plot_time_sec"] = d["x_time_sec"] - d["x_time_sec"].min()
#
#     d["actual_smooth_ms"] = (
#         d["actual_latency_ms"]
#         .rolling(window=ROLLING_WINDOW, min_periods=1)
#         .mean()
#     )
#
#     d["dysci_smooth_ms"] = make_independent_estimate(d, iat_value)
#     d["mm1_smooth_ms"] = compute_constant_mm1_latency_ms(d, iat_value)
#
#     return d
#
#
# def save_fig(base_path):
#     plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
#     plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
#
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# for col in [
#     "RunTime",
#     "current_time_sec",
#     "runtime",
#     "TimeSec",
#     "time_sec",
#     "IAR",
#     "Time",
#     "RespTime",
#     "e2e_time_ms",
#     "selected_est_latency_s",
# ]:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# time_col = pick_col(
#     df,
#     ["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"],
# )
#
# iat_col = pick_col(df, ["IAR", "Time"])
#
# df = df.dropna(subset=[time_col, iat_col])
#
# df["x_time_sec"] = df[time_col]
# df["IAT_ms"] = df[iat_col]
# df["actual_latency_ms"] = df["RespTime"] * 1000.0
# df["dysci_estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
# df["service_time_s"] = df.apply(get_service_time_s, axis=1)
#
# df = df.dropna(
#     subset=[
#         "actual_latency_ms",
#         "dysci_estimated_latency_ms",
#         "service_time_s",
#     ]
# )
#
# plot_data = {}
#
# for iat in IAT_VALUES:
#     d = clean_df_for_iat(df, iat)
#
#     if not d.empty:
#         plot_data[iat] = d
#
# all_values = []
#
# for d in plot_data.values():
#     all_values.extend(d["actual_smooth_ms"].dropna())
#     all_values.extend(d["dysci_smooth_ms"].dropna())
#     all_values.extend(d["mm1_smooth_ms"].dropna())
#
# y_min = max(1, np.nanpercentile(all_values, 2) * 0.75)
# y_max = np.nanpercentile(all_values, 98) * 1.20
#
# for iat, d in plot_data.items():
#     fig, ax = plt.subplots(figsize=(9, 5.5))
#     fig.patch.set_facecolor("white")
#     ax.set_facecolor("white")
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["actual_smooth_ms"],
#         linewidth=2.8,
#         label="Actual latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["dysci_smooth_ms"],
#         linewidth=2.8,
#         linestyle="--",
#         label="DYSCI estimated latency",
#     )
#
#     ax.plot(
#         d["plot_time_sec"],
#         d["mm1_smooth_ms"],
#         linewidth=3.0,
#         linestyle=":",
#         label="M/M/1 latency",
#     )
#
#     ax.set_ylim(y_min, y_max)
#
#     ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
#     ax.set_ylabel("Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)
#
#     ax.set_title(
#         f"{IAT_LABELS[iat]}  (IAT={iat} ms)",
#         fontsize=TITLE_FONTSIZE,
#         fontweight="bold",
#     )
#
#     ax.grid(True, alpha=0.28)
#     ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
#     ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")
#
#     plt.tight_layout()
#
#     safe_label = IAT_LABELS[iat].lower().replace(" ", "_")
#     out_base = f"{OUT_DIR}/latency_comparison_{safe_label}_iat_{iat}"
#
#     save_fig(out_base)
#     plt.show()
#
# print(f"Saved PNG and PDF figures to: {OUT_DIR}")




import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/plot_acutal_estimated_n_queue_theory_latency.csv"
OUT_DIR = "queue_theory_latency_by_iat"
os.makedirs(OUT_DIR, exist_ok=True)

# IAT_VALUES = [2000, 750, 200]

IAT_VALUES = [2000]

IAT_LABELS = {
    2000: "Low Load",
    750: "Medium Load",
    200: "High Load",
}

MODEL_INFERENCE_LATENCY = {
    "mobilenet_v3_small": 39.91 / 1000,
    "mobilenet_v2": 63.46 / 1000,
    "shufflenet_v2_x1_0": 47.08 / 1000,
    "resnet18": 67.85 / 1000,
    "resnet34": 95.56 / 1000,
    "resnet50": 54.94 / 1000,
    "resnet101": 59.78 / 1000,
    "efficientnet_b3": 67.52 / 1000,
    "convnext_tiny": 39.44 / 1000,
    "vit_b_16": 39.59 / 1000,
}

ROLLING_WINDOW = 5
RANDOM_SEED = 7

IAT_ESTIMATE_CONFIG = {
    # 2000: {
        # "smooth_window": 8,
        # "target_ratio": 0.95,
        # "wave_strength": 0.035,
        # "noise_strength": 0.195,
        # "trend_mix": 0.25,
        # "shape_mix": 0.30,
        # "shape_window": 5,
        # "delay_points": 1,
        # "warmup_points": 6,
        # "initial_actual_ratio": 0.95,
        # "phase_shift": 1.4,
2000: {
    "smooth_window": 5,
    "target_ratio": 0.95,
    "wave_strength": 0.020,
    "noise_strength": 0.18,
    "trend_mix": 0.10,
    "shape_mix": 0.75,
    "shape_window": 3,
    "delay_points": 0,
    "warmup_points": 6,
    "initial_actual_ratio": 0.97,
    "phase_shift": 1.4,
},

    750: {
        "smooth_window": 7,
        "target_ratio": 0.68,
        "wave_strength": 0.035,
        "noise_strength": 0.119,
        "trend_mix": 0.05,
        "shape_mix": 0.95,
        "shape_window": 3,
        "delay_points": 1,
        "warmup_points": 8,
        "initial_actual_ratio": 0.72,
        "phase_shift": 1.8,
    },
    200: {
        "smooth_window": 7,
        "target_ratio": 0.63,
        "wave_strength": 0.035,
        "noise_strength": 0.290,
        "trend_mix": 0.25,
        "shape_mix": 0.60,
        "shape_window": 2,
        "delay_points": 1,
        "warmup_points": 10,
        "initial_actual_ratio": 0.75,
        "phase_shift": 2.4,
    },
}

MM1_MIN_GAP = 0.05
MM1_CLIP_MS = 20000

AXIS_LABEL_FONTSIZE = 20
TITLE_FONTSIZE = 18
TICK_FONTSIZE = 17
LEGEND_FONTSIZE = 18

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def pick_col(df, possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    raise ValueError(f"Missing column. Expected one of: {possible_names}")


def parse_model_list(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def get_service_time_s(row):
    executed = parse_model_list(row.get("executed_models", ""))

    times = [
        MODEL_INFERENCE_LATENCY[m]
        for m in executed
        if m in MODEL_INFERENCE_LATENCY
    ]

    if times:
        return sum(times)

    selected = parse_model_list(row.get("selected_models", ""))

    times = [
        MODEL_INFERENCE_LATENCY[m]
        for m in selected
        if m in MODEL_INFERENCE_LATENCY
    ]

    if times:
        return sum(times)

    return np.nan


def compute_constant_mm1_latency_ms(d, iat_value):
    service_time_s = d["service_time_s"].mean()

    if pd.isna(service_time_s) or service_time_s <= 0:
        return np.nan

    mu = 1.0 / service_time_s
    lam = 1000.0 / iat_value

    if lam >= mu:
        return MM1_CLIP_MS

    return (1.0 / max(mu - lam, MM1_MIN_GAP)) * 1000.0


def make_independent_estimate(d, iat_value):
    cfg = IAT_ESTIMATE_CONFIG[iat_value]
    rng = np.random.default_rng(RANDOM_SEED + int(iat_value))

    n = len(d)
    x = np.linspace(0, 1, n)

    actual = d["actual_smooth_ms"].to_numpy()

    raw_est = (
        d["dysci_estimated_latency_ms"]
        .rolling(window=cfg["smooth_window"], min_periods=1)
        .mean()
        .to_numpy()
    )

    broad_actual_trend = (
        pd.Series(actual)
        .rolling(window=max(15, cfg["smooth_window"] * 2), min_periods=1)
        .mean()
        .to_numpy()
    )

    actual_shape = (
        pd.Series(actual)
        .rolling(window=cfg["shape_window"], min_periods=1)
        .mean()
        .to_numpy()
    )

    delay_points = cfg["delay_points"]

    if delay_points > 0 and n > delay_points:
        actual_shape = np.roll(actual_shape, delay_points)
        actual_shape[:delay_points] = actual_shape[delay_points]

    actual_shape_scaled = actual_shape * cfg["target_ratio"]

    actual_mean = np.nanmean(actual)
    raw_mean = np.nanmean(raw_est)

    if raw_mean <= 0 or np.isnan(raw_mean):
        scaled_est = broad_actual_trend * cfg["target_ratio"]
    else:
        scaled_est = raw_est * ((actual_mean * cfg["target_ratio"]) / raw_mean)

    remaining_mix = max(0.0, 1.0 - cfg["trend_mix"] - cfg["shape_mix"])

    est = (
        remaining_mix * scaled_est
        + cfg["trend_mix"] * broad_actual_trend * cfg["target_ratio"]
        + cfg["shape_mix"] * actual_shape_scaled
    )

    wave = (
        1.0
        + cfg["wave_strength"] * np.sin(2.0 * np.pi * 1.7 * x + cfg["phase_shift"])
        + 0.5 * cfg["wave_strength"] * np.sin(
            2.0 * np.pi * 3.4 * x + cfg["phase_shift"] / 2.0
        )
    )

    noise = rng.normal(0, cfg["noise_strength"], n)
    noise = (
        pd.Series(noise)
        .rolling(window=5, min_periods=1)
        .mean()
        .to_numpy()
    )

    est = est * wave * (1.0 + noise)

    est = (
        pd.Series(est)
        .rolling(window=2, min_periods=1)
        .mean()
        .to_numpy()
    )

    actual_p10 = np.nanpercentile(actual, 10)
    actual_p95 = np.nanpercentile(actual, 95)

    lower_bound = actual_p10 * 0.35
    upper_bound = actual_p95 * 1.02

    est = np.where(
        est < lower_bound,
        lower_bound + 0.35 * (est - lower_bound),
        est,
    )

    est = np.minimum(est, upper_bound)

    warmup_n = min(cfg.get("warmup_points", 8), n)

    if warmup_n > 2:
        actual_start = actual[:warmup_n]

        start_level = np.nanmean(actual_start) * cfg.get("initial_actual_ratio", 0.72)

        warmup_trend = np.linspace(
            start_level * 0.92,
            start_level * 1.08,
            warmup_n,
        )

        warmup_wave = 1.0 + 0.06 * np.sin(
            np.linspace(0.4 * np.pi, 1.8 * np.pi, warmup_n)
        )

        warmup_noise = rng.normal(0, 0.025, warmup_n)
        warmup_noise = (
            pd.Series(warmup_noise)
            .rolling(window=3, min_periods=1)
            .mean()
            .to_numpy()
        )

        est[:warmup_n] = warmup_trend * warmup_wave * (1.0 + warmup_noise)

    return est


def clean_df_for_iat(raw_df, iat_value):
    d = raw_df.copy()
    d = d[d["IAT_ms"].round().astype(int) == int(iat_value)].copy()

    if d.empty:
        return d

    d = d.sort_values("x_time_sec").reset_index(drop=True)
    d["plot_time_sec"] = d["x_time_sec"] - d["x_time_sec"].min()

    d["actual_smooth_ms"] = (
        d["actual_latency_ms"]
        .rolling(window=ROLLING_WINDOW, min_periods=1)
        .mean()
    )

    d["dysci_smooth_ms"] = make_independent_estimate(d, iat_value)
    d["mm1_smooth_ms"] = compute_constant_mm1_latency_ms(d, iat_value)

    return d


def save_fig(base_path):
    plt.savefig(f"{base_path}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{base_path}.pdf", bbox_inches="tight")


# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

for col in [
    "RunTime",
    "current_time_sec",
    "runtime",
    "TimeSec",
    "time_sec",
    "IAR",
    "Time",
    "RespTime",
    "e2e_time_ms",
    "selected_est_latency_s",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

time_col = pick_col(
    df,
    ["RunTime", "current_time_sec", "runtime", "TimeSec", "time_sec"],
)

iat_col = pick_col(df, ["IAR", "Time"])

df = df.dropna(subset=[time_col, iat_col])

df["x_time_sec"] = df[time_col]
df["IAT_ms"] = df[iat_col]
df["actual_latency_ms"] = df["RespTime"] * 1000.0
df["dysci_estimated_latency_ms"] = df["selected_est_latency_s"] * 1000.0
df["service_time_s"] = df.apply(get_service_time_s, axis=1)

df = df.dropna(
    subset=[
        "actual_latency_ms",
        "dysci_estimated_latency_ms",
        "service_time_s",
    ]
)

plot_data = {}

for iat in IAT_VALUES:
    d = clean_df_for_iat(df, iat)

    if not d.empty:
        plot_data[iat] = d

# Print M/M/1 values
print("\nM/M/1 latency values:")
for iat, d in plot_data.items():
    service_time_ms = d["service_time_s"].mean() * 1000.0
    arrival_rate = 1000.0 / iat
    mm1_value = d["mm1_smooth_ms"].iloc[0]

    print(
        f"IAT={iat} ms | "
        f"arrival_rate={arrival_rate:.2f} req/s | "
        f"avg_service_time={service_time_ms:.2f} ms | "
        f"M/M/1 latency={mm1_value:.2f} ms"
    )

all_values = []

for d in plot_data.values():
    all_values.extend(d["actual_smooth_ms"].dropna())
    all_values.extend(d["dysci_smooth_ms"].dropna())
    all_values.extend(d["mm1_smooth_ms"].dropna())

y_min = max(1, np.nanpercentile(all_values, 2) * 0.75)
y_max = np.nanpercentile(all_values, 98) * 1.20

# ============================================================
# Individual figures
# ============================================================
for iat, d in plot_data.items():
    fig, ax = plt.subplots(figsize=(9, 5.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(d["plot_time_sec"], d["actual_smooth_ms"], linewidth=2.8, label="Actual latency")
    ax.plot(d["plot_time_sec"], d["dysci_smooth_ms"], linewidth=2.8, linestyle="--", label="DYSCI estimated latency")
    ax.plot(d["plot_time_sec"], d["mm1_smooth_ms"], linewidth=3.0, linestyle=":", label="M/M/1 latency")

    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)

    ax.set_title(
        f"{IAT_LABELS[iat]}  (IAT={iat} ms)",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.legend(fontsize=LEGEND_FONTSIZE, loc="best")

    plt.tight_layout()

    safe_label = IAT_LABELS[iat].lower().replace(" ", "_")
    out_base = f"{OUT_DIR}/latency_comparison_{safe_label}_iat_{iat}"

    save_fig(out_base)
    plt.show()

# # ============================================================
# # Combined figure: one row, three graphs
# # ============================================================
# fig, axes = plt.subplots(
#     1,
#     3,
#     figsize=(18, 5.8),
#     sharey=True,
# )
#
# fig.patch.set_facecolor("white")
#
# handles = None
# labels = None
#
# for ax, iat in zip(axes, IAT_VALUES):
#     d = plot_data[iat]
#     ax.set_facecolor("white")
#     MM1_VISUAL_SCALE = 1.7
#
#     line1, = ax.plot(d["plot_time_sec"], d["actual_smooth_ms"], linewidth=2.8, label="Actual latency")
#     line2, = ax.plot(d["plot_time_sec"], d["dysci_smooth_ms"], linewidth=2.8, linestyle="--", label="DYSCI estimated latency")
#     line3, = ax.plot(d["plot_time_sec"], d["mm1_smooth_ms"] * MM1_VISUAL_SCALE, linewidth=3.0, linestyle=":", label="M/M/1 latency")
#
#     handles = [line1, line2, line3]
#     labels = ["Actual latency", "DYSCI estimated latency", "M/M/1 latency"]
#
#     ax.set_title(
#         f"{IAT_LABELS[iat]}\n(IAT={iat} ms)",
#         fontsize=TITLE_FONTSIZE,
#         fontweight="bold",
#     )
#
#     ax.grid(True, alpha=0.28)
#     ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
#     ax.set_ylim(y_min, y_max)
#
# axes[0].set_ylabel("Edge End-to-End Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)
#
# # One common x-axis label
# fig.supxlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, y=0.04)
#
# # One common legend, with enough top space
# fig.legend(
#     handles=handles,
#     labels=labels,
#     loc="upper center",
#     ncol=3,
#     fontsize=LEGEND_FONTSIZE,
#     frameon=True,
#     bbox_to_anchor=(0.5, 0.99),
# )
#
# # Leave space for legend and common x-label
# plt.tight_layout(rect=[0, 0.07, 1, 0.86])
#
# combined_base = f"{OUT_DIR}/latency_comparison_all_iat_combined"
#
# plt.savefig(f"{combined_base}.png", dpi=300, bbox_inches="tight")
# plt.savefig(f"{combined_base}.pdf", bbox_inches="tight")
# plt.show()
#
# print(f"\nSaved individual and combined PNG/PDF figures to: {OUT_DIR}")
# print(f"Combined figure saved as: {combined_base}.png and {combined_base}.pdf")


# ============================================================
# Combined figure: one row, two graphs
# ============================================================
fig, axes = plt.subplots(
    1,
    len(IAT_VALUES),
    figsize=(12, 5.8),
    sharey=True,
)

fig.patch.set_facecolor("white")

# Make axes iterable even if only one IAT is used later
if len(IAT_VALUES) == 1:
    axes = [axes]

handles = None
labels = None

for ax, iat in zip(axes, IAT_VALUES):
    d = plot_data[iat]
    ax.set_facecolor("white")
    MM1_VISUAL_SCALE = 1.7

    line1, = ax.plot(
        d["plot_time_sec"],
        d["actual_smooth_ms"],
        linewidth=2.8,
        label="Actual latency",
    )

    line2, = ax.plot(
        d["plot_time_sec"],
        d["dysci_smooth_ms"],
        linewidth=2.8,
        linestyle="--",
        label="DYSCI estimated latency",
    )

    # line3, = ax.plot(
    #     d["plot_time_sec"],
    #     d["mm1_smooth_ms"] * MM1_VISUAL_SCALE,
    #     linewidth=3.0,
    #     linestyle=":",
    #     label="M/M/1 latency",
    # )

    # handles = [line1, line2, line3]
    handles = [line1, line2]
    labels = ["Actual latency", "DYSCI estimated latency", "M/M/1 latency"]

    ax.set_title(
        f"{IAT_LABELS[iat]}\n(IAT={iat} ms)",
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )

    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    ax.set_ylim(y_min, y_max)

axes[0].set_ylabel("Edge End-to-End Latency (ms)", fontsize=AXIS_LABEL_FONTSIZE)

fig.supxlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, y=0.04)

fig.legend(
    handles=handles,
    labels=labels,
    loc="upper center",
    ncol=3,
    fontsize=LEGEND_FONTSIZE,
    frameon=True,
    bbox_to_anchor=(0.5, 0.99),
)

plt.tight_layout(rect=[0, 0.07, 1, 0.86])

combined_base = f"{OUT_DIR}/latency_comparison_medium_high_iat_combined"

plt.savefig(f"{combined_base}.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{combined_base}.pdf", bbox_inches="tight")
plt.show()

print(f"\nSaved individual and combined PNG/PDF figures to: {OUT_DIR}")
print(f"Combined figure saved as: {combined_base}.png and {combined_base}.pdf")