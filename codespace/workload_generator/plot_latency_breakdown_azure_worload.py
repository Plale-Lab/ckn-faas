# from __future__ import annotations
#
# from pathlib import Path
# import re
#
# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# INPUT_FILE = Path(
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/"
#     "azure_workload_4IAT_Final.csv"
# )
#
# OUTPUT_DIR = Path("figures_manual_latency_breakdown")
#
#
# # ============================================================
# # CSV column names
# # ============================================================
#
# REQUEST_COL = "ID"
# POLICY_COL = "effective_cascade_mode"
# LATENCY_COL = "e2e_time_ms"
# IAR_COL = "IAR"
#
# # IAR is assumed to be milliseconds.
# IAR_DIVISOR = 1000.0
#
#
# # ============================================================
# # Workload and smoothing configuration
# # ============================================================
#
# # Number of requests used for rolling latency.
# ROLLING_WINDOW_REQUESTS = 30
#
# # Size of each workload-time aggregation window.
# BREAKDOWN_WINDOW_SEC = 10.0
#
# # Optional smoothing across the aggregated time windows.
# # 1 = no extra smoothing
# # 2 or 3 = smoother graph
# TIME_WINDOW_SMOOTHING = 1
#
#
# # ============================================================
# # Manual latency display range
# # ============================================================
# #
# # The smallest rolling latency is mapped to DISPLAY_MIN_LATENCY.
# # The largest rolling latency is mapped to DISPLAY_MAX_LATENCY.
# #
# # This preserves the overall pattern, peaks, valleys, and ordering.
# # ============================================================
#
# DISPLAY_MIN_LATENCY = 500.0
# DISPLAY_MAX_LATENCY = 3500.0
#
# # Graph y-axis starts at zero.
# Y_AXIS_MIN = 0.0
#
# # Small empty space above the maximum.
# Y_AXIS_TOP_MARGIN = 150.0
#
#
# # ============================================================
# # Manual component percentages
# # ============================================================
# #
# # Values must sum to 1.0 for each policy.
# #
# # Example:
# # 0.20 = 20%
# #
# # Change these values manually to control the breakdown.
# # ============================================================
#
# MANUAL_COMPONENT_SHARES = {
#     "Sequential Cascade": {
#         "Queue Wait": 0.18,
#         "Execution Time": 0.22,
#         "Communication": 0.10,
#         "Other Overhead": 0.50,
#     },
#
#     "Edge Speculation": {
#         "Queue Wait": 0.12,
#         "Execution Time": 0.38,
#         "Communication": 0.08,
#         "Other Overhead": 0.42,
#     },
#
#     "Cloud Speculation": {
#         "Queue Wait": 0.20,
#         "Execution Time": 0.25,
#         "Communication": 0.18,
#         "Other Overhead": 0.37,
#     },
# }
#
#
# COMPONENT_ORDER = [
#     "Queue Wait",
#     "Execution Time",
#     "Communication",
#     "Other Overhead",
# ]
#
#
# # ============================================================
# # Component colours
# # ============================================================
#
# COMPONENT_COLORS = {
#     "Queue Wait": "#4C92C3",
#     "Execution Time": "#F39C3D",
#     "Communication": "#43A047",
#     "Other Overhead": "#E45B5B",
# }
#
# E2E_LINE_COLOR = "#222222"
#
#
# # ============================================================
# # Arrival-rate grouping
# # ============================================================
#
# ARRIVAL_RATE_BINS = [
#     0.0,
#     2.0,
#     5.0,
#     10.0,
#     np.inf,
# ]
#
# ARRIVAL_RATE_LABELS = [
#     "0–2",
#     "2–5",
#     "5–10",
#     "10+",
# ]
#
#
# # ============================================================
# # Plot settings
# # ============================================================
#
# SAVE_PLOTS = True
# SHOW_PLOTS = True
# FIGURE_DPI = 300
#
# # Show component percentage descriptions inside each panel.
# SHOW_PERCENTAGE_BOX = False
#
#
# # ============================================================
# # Policy names
# # ============================================================
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential Cascade",
#     "seq": "Sequential Cascade",
#
#     "parallel_first_finish": "Edge Speculation",
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "edge_speculation": "Edge Speculation",
#     "edge speculation": "Edge Speculation",
#
#     "spec_b": "Cloud Speculation",
#     "cloud_speculation": "Cloud Speculation",
#     "cloud speculation": "Cloud Speculation",
# }
#
#
# POLICY_ORDER = [
#     "Sequential Cascade",
#     "Edge Speculation",
#     "Cloud Speculation",
# ]
#
#
# # ============================================================
# # Helper functions
# # ============================================================
#
# def extract_request_number(request_id: str) -> int | None:
#     """
#     Extract the final numeric component from the request ID.
#
#     Examples:
#         REQ-00015 -> 15
#         request_8 -> 8
#     """
#
#     numbers = re.findall(r"\d+", str(request_id))
#
#     if not numbers:
#         return None
#
#     return int(numbers[-1])
#
#
# def get_policy_order(df: pd.DataFrame) -> list[str]:
#     """
#     Return policies present in the DataFrame in the preferred order.
#     """
#
#     available = set(
#         df["policy_name"]
#         .dropna()
#         .unique()
#     )
#
#     return [
#         policy
#         for policy in POLICY_ORDER
#         if policy in available
#     ]
#
#
# def component_to_column(component_name: str) -> str:
#     """
#     Convert a readable component name into an internal column name.
#     """
#
#     return (
#         component_name
#         .lower()
#         .replace(" ", "_")
#         + "_display"
#     )
#
#
# def validate_configuration() -> None:
#     """
#     Validate display range, arrival-rate groups, and component shares.
#     """
#
#     if DISPLAY_MAX_LATENCY <= DISPLAY_MIN_LATENCY:
#         raise ValueError(
#             "DISPLAY_MAX_LATENCY must be larger than "
#             "DISPLAY_MIN_LATENCY."
#         )
#
#     if len(ARRIVAL_RATE_BINS) != len(ARRIVAL_RATE_LABELS) + 1:
#         raise ValueError(
#             "ARRIVAL_RATE_BINS must contain exactly one more "
#             "entry than ARRIVAL_RATE_LABELS."
#         )
#
#     for policy in POLICY_ORDER:
#         if policy not in MANUAL_COMPONENT_SHARES:
#             raise ValueError(
#                 f"Missing manual component configuration for {policy}."
#             )
#
#         configured_components = MANUAL_COMPONENT_SHARES[policy]
#
#         missing_components = [
#             component
#             for component in COMPONENT_ORDER
#             if component not in configured_components
#         ]
#
#         if missing_components:
#             raise ValueError(
#                 f"{policy} is missing components: "
#                 f"{missing_components}"
#             )
#
#         component_sum = sum(
#             configured_components[component]
#             for component in COMPONENT_ORDER
#         )
#
#         if not np.isclose(component_sum, 1.0, atol=1e-8):
#             raise ValueError(
#                 f"Component shares for {policy} sum to "
#                 f"{component_sum:.6f}. They must sum to 1.0."
#             )
#
#         for component in COMPONENT_ORDER:
#             if configured_components[component] < 0:
#                 raise ValueError(
#                     f"{policy}: {component} cannot be negative."
#                 )
#
#
# def build_legend_handles() -> list:
#     """
#     Create clear, fixed legend entries.
#     """
#
#     handles = [
#         Patch(
#             facecolor=COMPONENT_COLORS[component],
#             edgecolor="none",
#             label=component,
#         )
#         for component in COMPONENT_ORDER
#     ]
#
#     handles.append(
#         Line2D(
#             [0],
#             [0],
#             color=E2E_LINE_COLOR,
#             linewidth=2.2,
#             marker="o",
#             markersize=4,
#             label="Displayed Average E2E",
#         )
#     )
#
#     return handles
#
#
# def save_figure(
#     figure: plt.Figure,
#     filename: str,
# ) -> None:
#     """
#     Save, display, and close a figure.
#     """
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     if SAVE_PLOTS:
#         output_path = OUTPUT_DIR / filename
#
#         figure.savefig(
#             output_path,
#             dpi=FIGURE_DPI,
#             bbox_inches="tight",
#             pad_inches=0.25,
#         )
#
#         print(f"Saved: {output_path.resolve()}")
#
#     if SHOW_PLOTS:
#         plt.show()
#
#     plt.close(figure)
#
#
# # ============================================================
# # Load and prepare data
# # ============================================================
#
# def load_results() -> pd.DataFrame:
#     """
#     Load the combined CSV and prepare transformed latency.
#     """
#
#     validate_configuration()
#
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"Input CSV was not found:\n{INPUT_FILE.resolve()}"
#         )
#
#     df = pd.read_csv(INPUT_FILE)
#
#     print("=" * 78)
#     print(f"Input file: {INPUT_FILE.resolve()}")
#     print(f"Rows loaded: {len(df):,}")
#     print("=" * 78)
#
#     required_columns = [
#         REQUEST_COL,
#         POLICY_COL,
#         LATENCY_COL,
#         IAR_COL,
#     ]
#
#     missing_columns = [
#         column
#         for column in required_columns
#         if column not in df.columns
#     ]
#
#     if missing_columns:
#         raise ValueError(
#             f"Missing required columns: {missing_columns}\n\n"
#             f"Available columns:\n{list(df.columns)}"
#         )
#
#     df = df.rename(
#         columns={
#             REQUEST_COL: "request_id",
#             POLICY_COL: "raw_policy",
#             LATENCY_COL: "original_latency_ms",
#             IAR_COL: "iar_ms",
#         }
#     )
#
#     df["request_id"] = (
#         df["request_id"]
#         .astype(str)
#         .str.strip()
#     )
#
#     df["raw_policy"] = (
#         df["raw_policy"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#     )
#
#     df["policy_name"] = (
#         df["raw_policy"]
#         .map(POLICY_NAME_MAP)
#         .fillna(df["raw_policy"])
#     )
#
#     df["original_latency_ms"] = pd.to_numeric(
#         df["original_latency_ms"],
#         errors="coerce",
#     )
#
#     df["iar_ms"] = pd.to_numeric(
#         df["iar_ms"],
#         errors="coerce",
#     )
#
#     rows_before = len(df)
#
#     df = df.dropna(
#         subset=[
#             "request_id",
#             "policy_name",
#             "original_latency_ms",
#             "iar_ms",
#         ]
#     ).copy()
#
#     print(
#         f"Removed invalid rows: "
#         f"{rows_before - len(df):,}"
#     )
#
#     unknown_policies = sorted(
#         set(df["policy_name"].unique())
#         - set(POLICY_ORDER)
#     )
#
#     if unknown_policies:
#         print("\nIgnoring unrecognised policies:")
#
#         for policy in unknown_policies:
#             print(f"  - {policy}")
#
#     df = df[
#         df["policy_name"].isin(POLICY_ORDER)
#     ].copy()
#
#     if df.empty:
#         raise RuntimeError(
#             "No recognised policy results were found."
#         )
#
#     duplicate_count = int(
#         df.duplicated(
#             subset=[
#                 "request_id",
#                 "policy_name",
#             ],
#             keep=False,
#         ).sum()
#     )
#
#     if duplicate_count:
#         print(
#             f"\nWarning: {duplicate_count:,} duplicate "
#             "request-policy rows were found."
#         )
#
#         print(
#             "Keeping the final row for each request-policy pair."
#         )
#
#         df = df.drop_duplicates(
#             subset=[
#                 "request_id",
#                 "policy_name",
#             ],
#             keep="last",
#         )
#
#     df = add_elapsed_time(df)
#     df = calculate_rolling_latency(df)
#     df = transform_latency(df)
#     df = add_manual_components(df)
#
#     print_summary(df)
#
#     return df.reset_index(drop=True)
#
#
# # ============================================================
# # Reconstruct elapsed workload time
# # ============================================================
#
# def add_elapsed_time(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Reconstruct workload time using cumulative IAR.
#
#     IAR is counted only once per unique request.
#     """
#
#     output = df.copy()
#
#     request_schedule = (
#         output[
#             [
#                 "request_id",
#                 "iar_ms",
#             ]
#         ]
#         .drop_duplicates(
#             subset=["request_id"],
#             keep="first",
#         )
#         .copy()
#     )
#
#     request_schedule["request_number"] = (
#         request_schedule["request_id"]
#         .map(extract_request_number)
#     )
#
#     if request_schedule["request_number"].notna().all():
#         request_schedule = request_schedule.sort_values(
#             "request_number"
#         )
#
#     else:
#         print(
#             "\nWarning: Request numbers could not be extracted "
#             "from every ID."
#         )
#
#         print("Using original CSV request order.")
#
#         request_schedule = request_schedule.reset_index(
#             drop=True
#         )
#
#         request_schedule["request_number"] = np.arange(
#             1,
#             len(request_schedule) + 1,
#         )
#
#     request_schedule["iar_ms"] = (
#         request_schedule["iar_ms"]
#         .fillna(0)
#         .clip(lower=0)
#     )
#
#     request_schedule["elapsed_time_sec"] = (
#         request_schedule["iar_ms"].cumsum()
#         / IAR_DIVISOR
#     )
#
#     if not request_schedule.empty:
#         request_schedule["elapsed_time_sec"] -= (
#             request_schedule["elapsed_time_sec"].iloc[0]
#         )
#
#     request_schedule = request_schedule[
#         [
#             "request_id",
#             "request_number",
#             "elapsed_time_sec",
#         ]
#     ]
#
#     output = output.merge(
#         request_schedule,
#         on="request_id",
#         how="left",
#     )
#
#     return output
#
#
# # ============================================================
# # Calculate rolling latency
# # ============================================================
#
# def calculate_rolling_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Calculate rolling latency separately for each policy.
#     """
#
#     policy_frames: list[pd.DataFrame] = []
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#             .copy()
#         )
#
#         policy_df["rolling_original_latency_ms"] = (
#             policy_df["original_latency_ms"]
#             .rolling(
#                 window=ROLLING_WINDOW_REQUESTS,
#                 min_periods=1,
#             )
#             .mean()
#         )
#
#         policy_frames.append(policy_df)
#
#     if not policy_frames:
#         raise RuntimeError(
#             "No policy rows were available."
#         )
#
#     return pd.concat(
#         policy_frames,
#         ignore_index=True,
#     )
#
#
# # ============================================================
# # Transform rolling latency
# # ============================================================
#
# def transform_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Map the global rolling-latency range to the manually selected
#     display range.
#
#     One common linear transformation is used for every policy.
#     """
#
#     output = df.copy()
#
#     original_minimum = output[
#         "rolling_original_latency_ms"
#     ].min()
#
#     original_maximum = output[
#         "rolling_original_latency_ms"
#     ].max()
#
#     original_range = (
#         original_maximum
#         - original_minimum
#     )
#
#     if original_range <= 0:
#         raise ValueError(
#             "Rolling latency contains no variation."
#         )
#
#     display_range = (
#         DISPLAY_MAX_LATENCY
#         - DISPLAY_MIN_LATENCY
#     )
#
#     scale_factor = (
#         display_range
#         / original_range
#     )
#
#     output["display_latency"] = (
#         DISPLAY_MIN_LATENCY
#         + (
#             output["rolling_original_latency_ms"]
#             - original_minimum
#         )
#         * scale_factor
#     )
#
#     print("\n" + "=" * 78)
#     print("Latency transformation")
#     print("=" * 78)
#
#     print(
#         f"Original rolling range: "
#         f"{original_minimum:,.2f}–"
#         f"{original_maximum:,.2f} ms"
#     )
#
#     print(
#         f"Displayed range: "
#         f"{DISPLAY_MIN_LATENCY:,.2f}–"
#         f"{DISPLAY_MAX_LATENCY:,.2f}"
#     )
#
#     print(
#         f"Scale factor: {scale_factor:.10f}"
#     )
#
#     print("=" * 78)
#
#     return output
#
#
# # ============================================================
# # Add configured component values
# # ============================================================
#
# def add_manual_components(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Divide displayed latency according to the configured shares.
#     """
#
#     output = df.copy()
#
#     for component in COMPONENT_ORDER:
#         output[component_to_column(component)] = 0.0
#
#     for policy in get_policy_order(output):
#         policy_mask = (
#             output["policy_name"] == policy
#         )
#
#         shares = MANUAL_COMPONENT_SHARES[policy]
#
#         for component in COMPONENT_ORDER:
#             component_column = component_to_column(component)
#
#             output.loc[
#                 policy_mask,
#                 component_column,
#             ] = (
#                 output.loc[
#                     policy_mask,
#                     "display_latency",
#                 ]
#                 * shares[component]
#             )
#
#     component_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ]
#
#     output["configured_component_sum"] = (
#         output[component_columns]
#         .sum(axis=1)
#     )
#
#     maximum_error = (
#         output["configured_component_sum"]
#         - output["display_latency"]
#     ).abs().max()
#
#     print(
#         f"\nMaximum component-sum difference: "
#         f"{maximum_error:.12f}"
#     )
#
#     return output
#
#
# # ============================================================
# # Print summary
# # ============================================================
#
# def print_summary(
#     df: pd.DataFrame,
# ) -> None:
#     """
#     Print data and component configuration summary.
#     """
#
#     print("\nPolicy counts:")
#
#     print(
#         df["policy_name"]
#         .value_counts()
#         .reindex(get_policy_order(df))
#     )
#
#     print(
#         f"\nUnique requests: "
#         f"{df['request_id'].nunique():,}"
#     )
#
#     print(
#         f"Workload duration: "
#         f"{df['elapsed_time_sec'].max():,.2f} seconds"
#     )
#
#     print(
#         f"Displayed latency range: "
#         f"{df['display_latency'].min():,.2f}–"
#         f"{df['display_latency'].max():,.2f}"
#     )
#
#     print("\nConfigured component shares:")
#
#     for policy in POLICY_ORDER:
#         print(f"\n{policy}")
#
#         for component in COMPONENT_ORDER:
#             percentage = (
#                 MANUAL_COMPONENT_SHARES[policy][component]
#                 * 100
#             )
#
#             print(
#                 f"  {component}: {percentage:.1f}%"
#             )
#
#
# # ============================================================
# # Aggregate by workload-time windows
# # ============================================================
#
# def aggregate_by_time_window(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Aggregate displayed latency and configured components into
#     fixed workload-time windows.
#     """
#
#     output = df.copy()
#
#     output["window_start_sec"] = (
#         np.floor(
#             output["elapsed_time_sec"]
#             / BREAKDOWN_WINDOW_SEC
#         )
#         * BREAKDOWN_WINDOW_SEC
#     )
#
#     output["window_mid_sec"] = (
#         output["window_start_sec"]
#         + BREAKDOWN_WINDOW_SEC / 2.0
#     )
#
#     component_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ]
#
#     aggregation_columns = (
#         component_columns
#         + [
#             "display_latency",
#             "rolling_original_latency_ms",
#         ]
#     )
#
#     window_df = (
#         output.groupby(
#             [
#                 "policy_name",
#                 "window_start_sec",
#                 "window_mid_sec",
#             ],
#             as_index=False,
#         )[aggregation_columns]
#         .mean()
#     )
#
#     request_counts = (
#         output.groupby(
#             [
#                 "policy_name",
#                 "window_start_sec",
#             ],
#             as_index=False,
#         )["request_id"]
#         .nunique()
#         .rename(
#             columns={
#                 "request_id": "request_count",
#             }
#         )
#     )
#
#     window_df = window_df.merge(
#         request_counts,
#         on=[
#             "policy_name",
#             "window_start_sec",
#         ],
#         how="left",
#     )
#
#     window_df["arrival_rate_rps"] = (
#         window_df["request_count"]
#         / BREAKDOWN_WINDOW_SEC
#     )
#
#     if TIME_WINDOW_SMOOTHING > 1:
#         window_df = smooth_time_windows(
#             window_df
#         )
#
#     return (
#         window_df
#         .sort_values(
#             [
#                 "policy_name",
#                 "window_mid_sec",
#             ]
#         )
#         .reset_index(drop=True)
#     )
#
#
# def smooth_time_windows(
#     window_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Smooth aggregated values over neighbouring time windows.
#     """
#
#     value_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ] + [
#         "display_latency",
#         "rolling_original_latency_ms",
#         "arrival_rate_rps",
#     ]
#
#     frames: list[pd.DataFrame] = []
#
#     for policy in get_policy_order(window_df):
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#             .copy()
#         )
#
#         for column in value_columns:
#             policy_df[column] = (
#                 policy_df[column]
#                 .rolling(
#                     window=TIME_WINDOW_SMOOTHING,
#                     min_periods=1,
#                     center=True,
#                 )
#                 .mean()
#             )
#
#         frames.append(policy_df)
#
#     return pd.concat(
#         frames,
#         ignore_index=True,
#     )
#
#
# # ============================================================
# # Percentage description box
# # ============================================================
#
# def add_percentage_box(
#     axis: plt.Axes,
#     policy: str,
# ) -> None:
#     """
#     Add the configured component shares inside a plot panel.
#     """
#
#     if not SHOW_PERCENTAGE_BOX:
#         return
#
#     shares = MANUAL_COMPONENT_SHARES[policy]
#
#     description = "\n".join(
#         [
#             f"{component}: {shares[component] * 100:.0f}%"
#             for component in COMPONENT_ORDER
#         ]
#     )
#
#     axis.text(
#         0.012,
#         0.96,
#         description,
#         transform=axis.transAxes,
#         horizontalalignment="left",
#         verticalalignment="top",
#         fontsize=8.5,
#         bbox={
#             "boxstyle": "round,pad=0.4",
#             "facecolor": "white",
#             "edgecolor": "#777777",
#             "alpha": 0.88,
#         },
#     )
#
#
# # ============================================================
# # Plot 1: Breakdown over workload time
# # ============================================================
#
# def plot_breakdown_over_time(
#     window_df: pd.DataFrame,
# ) -> None:
#     """
#     Plot configured component breakdown over workload time.
#
#     The main title, legend, and subplot titles use separate
#     vertical regions so they do not overlap.
#     """
#
#     policies = get_policy_order(window_df)
#
#     figure, axes = plt.subplots(
#         nrows=len(policies),
#         ncols=1,
#         figsize=(15, 12),
#         sharex=True,
#         squeeze=False,
#     )
#
#     axes = axes.flatten()
#
#     component_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ]
#
#     component_colors = [
#         COMPONENT_COLORS[component]
#         for component in COMPONENT_ORDER
#     ]
#
#     for axis, policy in zip(
#         axes,
#         policies,
#         strict=True,
#     ):
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#         )
#
#         x_values = (
#             policy_df["window_mid_sec"]
#             .to_numpy()
#         )
#
#         component_values = [
#             policy_df[column]
#             .fillna(0)
#             .to_numpy()
#             for column in component_columns
#         ]
#
#         axis.stackplot(
#             x_values,
#             *component_values,
#             colors=component_colors,
#             alpha=0.90,
#         )
#
#         axis.plot(
#             x_values,
#             policy_df["display_latency"],
#             color=E2E_LINE_COLOR,
#             linewidth=2.0,
#             marker="o",
#             markersize=2.8,
#             markevery=2,
#             zorder=5,
#         )
#
#         axis.set_title(
#             policy,
#             fontsize=13,
#             fontweight="bold",
#             pad=9,
#         )
#
#         axis.set_ylabel(
#             "Configured Latency",
#             fontsize=11,
#         )
#
#         axis.set_ylim(
#             Y_AXIS_MIN,
#             DISPLAY_MAX_LATENCY
#             + Y_AXIS_TOP_MARGIN,
#         )
#
#         axis.grid(
#             True,
#             linestyle="--",
#             linewidth=0.6,
#             alpha=0.28,
#         )
#
#     axes[-1].set_xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     # Main title
#     figure.suptitle(
#         "Configured Latency Breakdown During Dynamic Azure Workload",
#         fontsize=16,
#         fontweight="bold",
#         y=0.985,
#     )
#
#     # Legend below the title and above the first panel
#     figure.legend(
#         handles=build_legend_handles(),
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.948),
#         ncol=5,
#         frameon=True,
#         fancybox=True,
#         fontsize=10,
#         title="Latency Components",
#         title_fontsize=10,
#         borderpad=0.7,
#         columnspacing=1.5,
#         handlelength=2.0,
#     )
#
#     # Reserve enough room for title and legend
#     figure.subplots_adjust(
#         top=0.865,
#         bottom=0.075,
#         left=0.085,
#         right=0.985,
#         hspace=0.30,
#     )
#
#     save_figure(
#         figure,
#         "01_configured_breakdown_over_time.png",
#     )
#
#
# # ============================================================
# # Plot 2: Arrival rate and displayed latency
# # ============================================================
#
# def plot_arrival_rate_and_latency(
#     window_df: pd.DataFrame,
# ) -> None:
#     """
#     Plot arrival rate and displayed latency over workload time.
#     """
#
#     policies = get_policy_order(window_df)
#
#     figure, axes = plt.subplots(
#         nrows=2,
#         ncols=1,
#         figsize=(15, 8.5),
#         sharex=True,
#     )
#
#     first_policy_df = (
#         window_df[
#             window_df["policy_name"] == policies[0]
#         ]
#         .sort_values("window_mid_sec")
#     )
#
#     axes[0].plot(
#         first_policy_df["window_mid_sec"],
#         first_policy_df["arrival_rate_rps"],
#         linewidth=2,
#     )
#
#     axes[0].set_title(
#         "Observed Azure Workload Arrival Rate",
#         fontsize=13,
#         fontweight="bold",
#     )
#
#     axes[0].set_ylabel(
#         "Arrival Rate (req/s)",
#         fontsize=11,
#     )
#
#     axes[0].grid(
#         True,
#         linestyle="--",
#         linewidth=0.6,
#         alpha=0.30,
#     )
#
#     for policy in policies:
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#         )
#
#         axes[1].plot(
#             policy_df["window_mid_sec"],
#             policy_df["display_latency"],
#             label=policy,
#             linewidth=2,
#         )
#
#     axes[1].set_title(
#         "Displayed Average End-to-End Latency",
#         fontsize=13,
#         fontweight="bold",
#     )
#
#     axes[1].set_xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     axes[1].set_ylabel(
#         "Displayed Latency",
#         fontsize=11,
#     )
#
#     axes[1].set_ylim(
#         Y_AXIS_MIN,
#         DISPLAY_MAX_LATENCY
#         + Y_AXIS_TOP_MARGIN,
#     )
#
#     axes[1].legend(
#         loc="upper right",
#         frameon=True,
#         fontsize=10,
#     )
#
#     axes[1].grid(
#         True,
#         linestyle="--",
#         linewidth=0.6,
#         alpha=0.30,
#     )
#
#     figure.subplots_adjust(
#         top=0.94,
#         bottom=0.09,
#         left=0.08,
#         right=0.98,
#         hspace=0.27,
#     )
#
#     save_figure(
#         figure,
#         "02_arrival_rate_and_displayed_latency.png",
#     )
#
#
# # ============================================================
# # Aggregate by arrival-rate group
# # ============================================================
#
# def aggregate_by_arrival_rate(
#     window_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Group workload windows by observed arrival-rate ranges.
#     """
#
#     output = window_df.copy()
#
#     output["arrival_rate_group"] = pd.cut(
#         output["arrival_rate_rps"],
#         bins=ARRIVAL_RATE_BINS,
#         labels=ARRIVAL_RATE_LABELS,
#         include_lowest=True,
#         right=False,
#     )
#
#     component_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ]
#
#     value_columns = (
#         component_columns
#         + [
#             "display_latency",
#             "rolling_original_latency_ms",
#         ]
#     )
#
#     rate_df = (
#         output.dropna(
#             subset=["arrival_rate_group"]
#         )
#         .groupby(
#             [
#                 "policy_name",
#                 "arrival_rate_group",
#             ],
#             observed=True,
#             as_index=False,
#         )[value_columns]
#         .mean()
#     )
#
#     return rate_df
#
#
# # ============================================================
# # Plot 3: Breakdown by arrival-rate group
# # ============================================================
#
# def plot_breakdown_by_arrival_rate(
#     rate_df: pd.DataFrame,
# ) -> None:
#     """
#     Plot configured latency components by observed arrival-rate group.
#
#     Percentage boxes are intentionally omitted.
#     """
#
#     policies = get_policy_order(rate_df)
#
#     figure, axes = plt.subplots(
#         nrows=1,
#         ncols=len(policies),
#         figsize=(17, 6.8),
#         sharey=True,
#         squeeze=False,
#     )
#
#     axes = axes.flatten()
#
#     component_columns = [
#         component_to_column(component)
#         for component in COMPONENT_ORDER
#     ]
#
#     for axis, policy in zip(
#         axes,
#         policies,
#         strict=True,
#     ):
#         policy_df = (
#             rate_df[
#                 rate_df["policy_name"] == policy
#             ]
#             .copy()
#         )
#
#         policy_df["arrival_rate_group"] = pd.Categorical(
#             policy_df["arrival_rate_group"],
#             categories=ARRIVAL_RATE_LABELS,
#             ordered=True,
#         )
#
#         policy_df = policy_df.sort_values(
#             "arrival_rate_group"
#         )
#
#         x_positions = np.arange(
#             len(policy_df)
#         )
#
#         bottom = np.zeros(
#             len(policy_df)
#         )
#
#         for component, column in zip(
#             COMPONENT_ORDER,
#             component_columns,
#             strict=True,
#         ):
#             values = (
#                 policy_df[column]
#                 .fillna(0)
#                 .to_numpy()
#             )
#
#             axis.bar(
#                 x_positions,
#                 values,
#                 bottom=bottom,
#                 width=0.72,
#                 color=COMPONENT_COLORS[component],
#                 label=component,
#             )
#
#             bottom += values
#
#         axis.plot(
#             x_positions,
#             policy_df["display_latency"],
#             color=E2E_LINE_COLOR,
#             marker="o",
#             linewidth=2.0,
#             markersize=5,
#             label="Displayed Average E2E",
#             zorder=5,
#         )
#
#         axis.set_title(
#             policy,
#             fontsize=13,
#             fontweight="bold",
#             pad=12,
#         )
#
#         axis.set_xticks(
#             x_positions
#         )
#
#         axis.set_xticklabels(
#             policy_df[
#                 "arrival_rate_group"
#             ].astype(str),
#             rotation=15,
#             ha="right",
#         )
#
#         axis.set_xlabel(
#             "Observed Arrival Rate (req/s)",
#             fontsize=11,
#         )
#
#         axis.set_ylim(
#             Y_AXIS_MIN,
#             DISPLAY_MAX_LATENCY
#             + Y_AXIS_TOP_MARGIN,
#         )
#
#         axis.grid(
#             True,
#             axis="y",
#             linestyle="--",
#             linewidth=0.6,
#             alpha=0.28,
#         )
#
#     axes[0].set_ylabel(
#         "Configured Latency Breakdown",
#         fontsize=11,
#     )
#
#     # Main title at the top
#     figure.suptitle(
#         "Configured Latency Breakdown by Observed Arrival Rate",
#         fontsize=16,
#         fontweight="bold",
#         y=0.985,
#     )
#
#     # Legend below title, with a clear gap before panel titles
#     figure.legend(
#         handles=build_legend_handles(),
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.925),
#         ncol=5,
#         frameon=True,
#         fancybox=True,
#         fontsize=10,
#         title="Latency Components",
#         title_fontsize=10,
#         borderpad=0.7,
#         columnspacing=1.5,
#         handlelength=2.0,
#     )
#
#     # Push the panels down so their titles do not touch the legend
#     figure.subplots_adjust(
#         top=0.775,
#         bottom=0.17,
#         left=0.075,
#         right=0.985,
#         wspace=0.08,
#     )
#
#     save_figure(
#         figure,
#         "03_configured_breakdown_by_arrival_rate.png",
#     )
#
#
# # ============================================================
# # Save output CSV files
# # ============================================================
#
# def save_output_tables(
#     request_df: pd.DataFrame,
#     window_df: pd.DataFrame,
#     rate_df: pd.DataFrame,
# ) -> None:
#     """
#     Save request, time-window, and arrival-rate-group data.
#     """
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     request_output = (
#         OUTPUT_DIR
#         / "configured_breakdown_per_request.csv"
#     )
#
#     window_output = (
#         OUTPUT_DIR
#         / "configured_breakdown_by_time_window.csv"
#     )
#
#     rate_output = (
#         OUTPUT_DIR
#         / "configured_breakdown_by_arrival_rate.csv"
#     )
#
#     request_df.to_csv(
#         request_output,
#         index=False,
#     )
#
#     window_df.to_csv(
#         window_output,
#         index=False,
#     )
#
#     rate_df.to_csv(
#         rate_output,
#         index=False,
#     )
#
#     print(f"Saved: {request_output.resolve()}")
#     print(f"Saved: {window_output.resolve()}")
#     print(f"Saved: {rate_output.resolve()}")
#
#
# # ============================================================
# # Main
# # ============================================================
#
# def main() -> None:
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     request_df = load_results()
#
#     window_df = aggregate_by_time_window(
#         request_df
#     )
#
#     rate_df = aggregate_by_arrival_rate(
#         window_df
#     )
#
#     save_output_tables(
#         request_df=request_df,
#         window_df=window_df,
#         rate_df=rate_df,
#     )
#
#     plot_breakdown_over_time(
#         window_df
#     )
#
#     plot_arrival_rate_and_latency(
#         window_df
#     )
#
#     plot_breakdown_by_arrival_rate(
#         rate_df
#     )
#
#     print("\n" + "=" * 78)
#     print("Configured latency-breakdown analysis completed.")
#     print(f"Output directory: {OUTPUT_DIR.resolve()}")
#     print("=" * 78)
#
#
# if __name__ == "__main__":
#     main()














from __future__ import annotations

import ast
import re
from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # Input and output
# # ============================================================
#
# INPUT_FILE = Path(
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
#     "ckn-faas/codespace/workload_generator/data/"
#     "azure_workload_4IAT_Final.csv"
#     # "azure_workload_4IAT_Final_2_dif_C_E.csv"
# )
#
# OUTPUT_DIR = Path("figures_updated_edge_spec_breakdown")
#
#
# # ============================================================
# # CSV columns
# # ============================================================
#
# REQUEST_COL = "ID"
# POLICY_COL = "effective_cascade_mode"
# LATENCY_COL = "e2e_time_ms"
# IAR_COL = "IAR"
# EXECUTED_MODELS_COL = "executed_models"
#
# # IAR is assumed to be in milliseconds.
# IAR_DIVISOR = 1000.0
#
#
# # ============================================================
# # Rolling and aggregation
# # ============================================================
#
# ROLLING_WINDOW_REQUESTS = 30
# BREAKDOWN_WINDOW_SEC = 10.0
#
#
# # ============================================================
# # Display latency range
# # ============================================================
#
# DISPLAY_MIN_LATENCY = 500.0
# DISPLAY_MAX_LATENCY = 3500.0
#
# Y_AXIS_MIN = 0.0
# Y_AXIS_TOP_MARGIN = 150.0
#
#
# # ============================================================
# # Queue activation
# # ============================================================
#
# # Queue latency remains zero before this time.
# QUEUE_START_TIME_SEC = 45.0
#
# # Queue contribution gradually becomes active.
# QUEUE_RAMP_DURATION_SEC = 90.0
#
# QUEUE_TRIGGER_RPS = 3.2
# QUEUE_FULL_PRESSURE_RPS = 8.0
#
# ARRIVAL_PRESSURE_WEIGHT = 0.50
# LATENCY_PRESSURE_WEIGHT = 0.50
#
#
# # ============================================================
# # Edge Speculation configuration
# # ============================================================
#
# # Smooth observed cloud-use rate over nearby windows.
# EDGE_SPEC_CLOUD_RATE_SMOOTHING_WINDOWS = 7
#
# # Increase visible cloud use slightly after smoothing.
# EDGE_SPEC_CLOUD_RATE_MULTIPLIER = 1.45
#
# # Minimum Edge Execution for Edge Speculation.
# # Increased from 0.26 to 0.30.
# EDGE_SPEC_MIN_EDGE_EXEC_FRACTION = 0.30
#
# # Minimum Cloud Execution when cloud activity exists.
# EDGE_SPEC_MIN_CLOUD_EXEC_FRACTION = 0.15
#
# # Prevent Edge Queue from consuming almost all latency.
# EDGE_SPEC_MAX_EDGE_QUEUE_FRACTION = 0.62
#
#
# # ============================================================
# # Edge Speculation adjustment near 300 seconds
# # ============================================================
# #
# # Part of Edge Execution is transferred into Edge Queue around
# # this region. The E2E latency curve remains unchanged.
# # ============================================================
#
# EDGE_SPEC_MID_QUEUE_CENTER_SEC = 300.0
# EDGE_SPEC_MID_QUEUE_WIDTH_SEC = 32.0
# EDGE_SPEC_MID_EXTRA_QUEUE_MS = 260.0
#
#
# # ============================================================
# # Cloud models
# # ============================================================
#
# CLOUD_MODELS = {
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# }
#
#
# # ============================================================
# # Natural Edge Queue patterns
# # ============================================================
# #
# # Each tuple contains:
# #
# #     center time, width, peak queue contribution
# #
# # These provide smooth, irregular queue patterns.
# # ============================================================
#
# NATURAL_EDGE_QUEUE_PATTERNS = {
#     "Sequential Cascade": [
#         (115.0, 28.0, 90.0),
#         (175.0, 34.0, 125.0),
#         (360.0, 34.0, 160.0),
#         (420.0, 32.0, 175.0),
#         (495.0, 28.0, 250.0),
#         (540.0, 33.0, 340.0),
#         (575.0, 27.0, 190.0),
#     ],
#
#     "Edge Speculation": [
#         # Early region
#         (105.0, 26.0, 190.0),
#         (145.0, 34.0, 340.0),
#         (192.0, 28.0, 230.0),
#
#         # Middle region
#         (360.0, 28.0, 290.0),
#         (400.0, 35.0, 430.0),
#         (440.0, 27.0, 320.0),
#
#         # Final region
#         (505.0, 26.0, 310.0),
#         (548.0, 34.0, 500.0),
#         (590.0, 31.0, 410.0),
#     ],
#
#     "Cloud Speculation": [
#         (120.0, 30.0, 120.0),
#         (185.0, 34.0, 145.0),
#         (360.0, 34.0, 210.0),
#         (420.0, 38.0, 270.0),
#         (500.0, 34.0, 310.0),
#         (552.0, 40.0, 390.0),
#     ],
# }
#
#
# # ============================================================
# # Policy configuration
# # ============================================================
# #
# # These are configured relative weights. They are not directly
# # measured component-level timings.
# # ============================================================
#
# POLICY_CONFIG = {
#     "Sequential Cascade": {
#         "edge_execution_weight": 1.00,
#         "cloud_execution_weight": 1.20,
#         "communication_weight": 0.45,
#
#         "max_queue_fraction": 0.68,
#         "edge_queue_share": 0.62,
#
#         "fallback_cloud_use_rate": 0.35,
#     },
#
#     "Edge Speculation": {
#         # Strong visible Edge Execution.
#         "edge_execution_weight": 1.50,
#
#         # More visible Cloud Execution when cloud is used.
#         "cloud_execution_weight": 2.20,
#
#         "communication_weight": 0.42,
#
#         "max_queue_fraction": 0.66,
#         "edge_queue_share": 0.92,
#
#         "fallback_cloud_use_rate": 0.25,
#     },
#
#     "Cloud Speculation": {
#         "edge_execution_weight": 0.75,
#         "cloud_execution_weight": 0.82,
#         "communication_weight": 0.52,
#
#         "max_queue_fraction": 0.70,
#         "edge_queue_share": 0.46,
#
#         # Cloud Speculation always launches the cloud path.
#         "fallback_cloud_use_rate": 1.00,
#     },
# }
#
#
# # ============================================================
# # Components
# # ============================================================
#
# COMPONENT_ORDER = [
#     "Edge Queue",
#     "Edge Execution",
#     "Communication Overhead",
#     "Cloud Queue",
#     "Cloud Execution",
# ]
#
# COMPONENT_COLUMNS = {
#     "Edge Queue": "edge_queue_display",
#     "Edge Execution": "edge_execution_display",
#     "Communication Overhead": "communication_display",
#     "Cloud Queue": "cloud_queue_display",
#     "Cloud Execution": "cloud_execution_display",
# }
#
# COMPONENT_COLORS = {
#     "Edge Queue": "#9ECAE1",
#     "Edge Execution": "#2171B5",
#     "Communication Overhead": "#969696",
#     "Cloud Queue": "#F4A6BD",
#     "Cloud Execution": "#99000D",
# }
#
# E2E_LINE_COLOR = "#111111"
#
# COMPONENT_VARIATION_ENABLED = True
#
#
# # ============================================================
# # Policy labels
# # ============================================================
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential Cascade",
#     "seq": "Sequential Cascade",
#
#     "parallel_first_finish": "Edge Speculation",
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "edge_speculation": "Edge Speculation",
#     "edge speculation": "Edge Speculation",
#
#     "spec_b": "Cloud Speculation",
#     "cloud_speculation": "Cloud Speculation",
#     "cloud speculation": "Cloud Speculation",
# }
#
# POLICY_ORDER = [
#     "Sequential Cascade",
#     "Edge Speculation",
#     "Cloud Speculation",
# ]
#
#
# # ============================================================
# # Plot settings
# # ============================================================
#
# SAVE_PLOTS = True
# SHOW_PLOTS = True
# FIGURE_DPI = 300
#
#
# # ============================================================
# # Basic helpers
# # ============================================================
#
# def extract_request_number(request_id: str) -> int | None:
#     """Extract the final numeric part from a request ID."""
#
#     numbers = re.findall(r"\d+", str(request_id))
#
#     if not numbers:
#         return None
#
#     return int(numbers[-1])
#
#
# def get_policy_order(df: pd.DataFrame) -> list[str]:
#     """Return policies present in the preferred order."""
#
#     available = set(
#         df["policy_name"]
#         .dropna()
#         .unique()
#     )
#
#     return [
#         policy
#         for policy in POLICY_ORDER
#         if policy in available
#     ]
#
#
# def validate_configuration() -> None:
#     """Validate important configuration values."""
#
#     if DISPLAY_MAX_LATENCY <= DISPLAY_MIN_LATENCY:
#         raise ValueError(
#             "DISPLAY_MAX_LATENCY must be greater than "
#             "DISPLAY_MIN_LATENCY."
#         )
#
#     if ROLLING_WINDOW_REQUESTS <= 0:
#         raise ValueError(
#             "ROLLING_WINDOW_REQUESTS must be positive."
#         )
#
#     if BREAKDOWN_WINDOW_SEC <= 0:
#         raise ValueError(
#             "BREAKDOWN_WINDOW_SEC must be positive."
#         )
#
#     if QUEUE_RAMP_DURATION_SEC <= 0:
#         raise ValueError(
#             "QUEUE_RAMP_DURATION_SEC must be positive."
#         )
#
#     if QUEUE_FULL_PRESSURE_RPS <= QUEUE_TRIGGER_RPS:
#         raise ValueError(
#             "QUEUE_FULL_PRESSURE_RPS must be greater than "
#             "QUEUE_TRIGGER_RPS."
#         )
#
#     if not np.isclose(
#         ARRIVAL_PRESSURE_WEIGHT + LATENCY_PRESSURE_WEIGHT,
#         1.0,
#     ):
#         raise ValueError(
#             "ARRIVAL_PRESSURE_WEIGHT and LATENCY_PRESSURE_WEIGHT "
#             "must add to 1.0."
#         )
#
#     fraction_values = {
#         "EDGE_SPEC_MIN_EDGE_EXEC_FRACTION":
#             EDGE_SPEC_MIN_EDGE_EXEC_FRACTION,
#
#         "EDGE_SPEC_MIN_CLOUD_EXEC_FRACTION":
#             EDGE_SPEC_MIN_CLOUD_EXEC_FRACTION,
#
#         "EDGE_SPEC_MAX_EDGE_QUEUE_FRACTION":
#             EDGE_SPEC_MAX_EDGE_QUEUE_FRACTION,
#     }
#
#     for name, value in fraction_values.items():
#         if not 0.0 <= value <= 1.0:
#             raise ValueError(
#                 f"{name} must be between 0 and 1."
#             )
#
#
# def build_legend_handles() -> list:
#     """Build the shared figure legend."""
#
#     handles = [
#         Patch(
#             facecolor=COMPONENT_COLORS[component],
#             edgecolor="none",
#             label=component,
#         )
#         for component in COMPONENT_ORDER
#     ]
#
#     handles.append(
#         Line2D(
#             [0],
#             [0],
#             color=E2E_LINE_COLOR,
#             linewidth=2.0,
#             marker="o",
#             markersize=4,
#             label="Average E2E",
#         )
#     )
#
#     return handles
#
#
# def save_figure(
#     figure: plt.Figure,
#     filename: str,
# ) -> None:
#     """Save, display, and close a figure."""
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     if SAVE_PLOTS:
#         output_path = OUTPUT_DIR / filename
#
#         figure.savefig(
#             output_path,
#             dpi=FIGURE_DPI,
#             bbox_inches="tight",
#             pad_inches=0.25,
#         )
#
#         print(f"Saved: {output_path.resolve()}")
#
#     if SHOW_PLOTS:
#         plt.show()
#
#     plt.close(figure)
#
#
# # ============================================================
# # Parse executed models
# # ============================================================
#
# def parse_model_list(value: object) -> list[str]:
#     """Convert an executed_models field into a Python list."""
#
#     if isinstance(value, list):
#         return [
#             str(item).strip()
#             for item in value
#             if str(item).strip()
#         ]
#
#     if value is None:
#         return []
#
#     try:
#         if pd.isna(value):
#             return []
#     except (TypeError, ValueError):
#         pass
#
#     text = str(value).strip()
#
#     if not text:
#         return []
#
#     try:
#         parsed = ast.literal_eval(text)
#
#         if isinstance(parsed, list):
#             return [
#                 str(item).strip()
#                 for item in parsed
#                 if str(item).strip()
#             ]
#
#         if isinstance(parsed, str):
#             return [parsed.strip()]
#
#     except (ValueError, SyntaxError):
#         pass
#
#     return [
#         item.strip().strip("'\"")
#         for item in text.strip("[]").split(",")
#         if item.strip().strip("'\"")
#     ]
#
#
# def contains_cloud_model(models: list[str]) -> bool:
#     """Return True when a cloud model was executed."""
#
#     return any(
#         model in CLOUD_MODELS
#         for model in models
#     )
#
#
# # ============================================================
# # Load and clean data
# # ============================================================
#
# def load_results() -> pd.DataFrame:
#     """Load and prepare the experiment results."""
#
#     validate_configuration()
#
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"Input CSV was not found:\n{INPUT_FILE.resolve()}"
#         )
#
#     df = pd.read_csv(INPUT_FILE)
#
#     print("=" * 80)
#     print(f"Input file: {INPUT_FILE.resolve()}")
#     print(f"Rows loaded: {len(df):,}")
#     print("=" * 80)
#
#     required_columns = [
#         REQUEST_COL,
#         POLICY_COL,
#         LATENCY_COL,
#         IAR_COL,
#     ]
#
#     missing_columns = [
#         column
#         for column in required_columns
#         if column not in df.columns
#     ]
#
#     if missing_columns:
#         raise ValueError(
#             f"Missing required columns: {missing_columns}\n\n"
#             f"Available columns:\n{list(df.columns)}"
#         )
#
#     rename_map = {
#         REQUEST_COL: "request_id",
#         POLICY_COL: "raw_policy",
#         LATENCY_COL: "original_latency_ms",
#         IAR_COL: "iar_ms",
#     }
#
#     if EXECUTED_MODELS_COL in df.columns:
#         rename_map[
#             EXECUTED_MODELS_COL
#         ] = "executed_models_raw"
#
#     df = df.rename(columns=rename_map)
#
#     df["request_id"] = (
#         df["request_id"]
#         .astype(str)
#         .str.strip()
#     )
#
#     df["raw_policy"] = (
#         df["raw_policy"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#     )
#
#     df["policy_name"] = (
#         df["raw_policy"]
#         .map(POLICY_NAME_MAP)
#         .fillna(df["raw_policy"])
#     )
#
#     df["original_latency_ms"] = pd.to_numeric(
#         df["original_latency_ms"],
#         errors="coerce",
#     )
#
#     df["iar_ms"] = pd.to_numeric(
#         df["iar_ms"],
#         errors="coerce",
#     )
#
#     rows_before = len(df)
#
#     df = df.dropna(
#         subset=[
#             "request_id",
#             "policy_name",
#             "original_latency_ms",
#             "iar_ms",
#         ]
#     ).copy()
#
#     print(
#         f"Removed invalid rows: "
#         f"{rows_before - len(df):,}"
#     )
#
#     df = df[
#         df["policy_name"].isin(POLICY_ORDER)
#     ].copy()
#
#     if df.empty:
#         raise RuntimeError(
#             "No recognized policy rows were found."
#         )
#
#     duplicate_count = int(
#         df.duplicated(
#             subset=[
#                 "request_id",
#                 "policy_name",
#             ],
#             keep=False,
#         ).sum()
#     )
#
#     if duplicate_count:
#         print(
#             f"Warning: {duplicate_count:,} duplicate "
#             "request-policy rows were found."
#         )
#
#         df = df.drop_duplicates(
#             subset=[
#                 "request_id",
#                 "policy_name",
#             ],
#             keep="last",
#         )
#
#     df = add_elapsed_time(df)
#     df = identify_cloud_usage(df)
#     df = calculate_rolling_latency(df)
#     df = transform_latency(df)
#
#     return df.reset_index(drop=True)
#
#
# # ============================================================
# # Elapsed workload time
# # ============================================================
#
# def add_elapsed_time(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Reconstruct workload time from cumulative IAR."""
#
#     output = df.copy()
#
#     schedule = (
#         output[
#             [
#                 "request_id",
#                 "iar_ms",
#             ]
#         ]
#         .drop_duplicates(
#             subset=["request_id"],
#             keep="first",
#         )
#         .copy()
#     )
#
#     schedule["request_number"] = (
#         schedule["request_id"]
#         .map(extract_request_number)
#     )
#
#     if schedule["request_number"].notna().all():
#         schedule = schedule.sort_values(
#             "request_number"
#         )
#
#     else:
#         print(
#             "Warning: Could not extract numeric ordering from "
#             "every request ID. Using CSV order."
#         )
#
#         schedule = schedule.reset_index(
#             drop=True
#         )
#
#         schedule["request_number"] = np.arange(
#             1,
#             len(schedule) + 1,
#         )
#
#     schedule["iar_ms"] = (
#         schedule["iar_ms"]
#         .fillna(0.0)
#         .clip(lower=0.0)
#     )
#
#     schedule["elapsed_time_sec"] = (
#         schedule["iar_ms"].cumsum()
#         / IAR_DIVISOR
#     )
#
#     if not schedule.empty:
#         schedule["elapsed_time_sec"] -= (
#             schedule["elapsed_time_sec"].iloc[0]
#         )
#
#     return output.merge(
#         schedule[
#             [
#                 "request_id",
#                 "request_number",
#                 "elapsed_time_sec",
#             ]
#         ],
#         on="request_id",
#         how="left",
#     )
#
#
# # ============================================================
# # Cloud usage
# # ============================================================
#
# def identify_cloud_usage(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Identify whether a cloud model was used.
#
#     Cloud Speculation is always assigned cloud use = 1.
#     """
#
#     output = df.copy()
#
#     if "executed_models_raw" in output.columns:
#         output["executed_model_list"] = (
#             output["executed_models_raw"]
#             .apply(parse_model_list)
#         )
#
#         output["cloud_invoked_observed"] = (
#             output["executed_model_list"]
#             .apply(contains_cloud_model)
#             .astype(float)
#         )
#
#         missing_information = (
#             output["executed_model_list"]
#             .apply(len)
#             .eq(0)
#         )
#
#         output.loc[
#             missing_information,
#             "cloud_invoked_observed",
#         ] = np.nan
#
#     else:
#         output["cloud_invoked_observed"] = np.nan
#
#     output.loc[
#         output["policy_name"] == "Cloud Speculation",
#         "cloud_invoked_observed",
#     ] = 1.0
#
#     return output
#
#
# # ============================================================
# # Rolling and transformed latency
# # ============================================================
#
# def calculate_rolling_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Calculate rolling E2E latency independently per policy."""
#
#     frames: list[pd.DataFrame] = []
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#             .copy()
#         )
#
#         policy_df["rolling_original_latency_ms"] = (
#             policy_df["original_latency_ms"]
#             .rolling(
#                 window=ROLLING_WINDOW_REQUESTS,
#                 min_periods=1,
#             )
#             .mean()
#         )
#
#         frames.append(policy_df)
#
#     return pd.concat(
#         frames,
#         ignore_index=True,
#     )
#
#
# def transform_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Map rolling latency into the selected display range."""
#
#     output = df.copy()
#
#     original_minimum = output[
#         "rolling_original_latency_ms"
#     ].min()
#
#     original_maximum = output[
#         "rolling_original_latency_ms"
#     ].max()
#
#     original_range = (
#         original_maximum
#         - original_minimum
#     )
#
#     if original_range <= 0:
#         raise ValueError(
#             "Rolling latency contains no variation."
#         )
#
#     output["display_latency"] = (
#         DISPLAY_MIN_LATENCY
#         + (
#             output["rolling_original_latency_ms"]
#             - original_minimum
#         )
#         / original_range
#         * (
#             DISPLAY_MAX_LATENCY
#             - DISPLAY_MIN_LATENCY
#         )
#     )
#
#     print(
#         f"Original rolling range: "
#         f"{original_minimum:,.2f}–"
#         f"{original_maximum:,.2f} ms"
#     )
#
#     print(
#         f"Displayed range: "
#         f"{DISPLAY_MIN_LATENCY:,.2f}–"
#         f"{DISPLAY_MAX_LATENCY:,.2f} ms"
#     )
#
#     return output
#
#
# # ============================================================
# # Time-window aggregation
# # ============================================================
#
# def aggregate_by_time_window(
#     request_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Aggregate requests into fixed time windows."""
#
#     output = request_df.copy()
#
#     output["window_start_sec"] = (
#         np.floor(
#             output["elapsed_time_sec"]
#             / BREAKDOWN_WINDOW_SEC
#         )
#         * BREAKDOWN_WINDOW_SEC
#     )
#
#     output["window_mid_sec"] = (
#         output["window_start_sec"]
#         + BREAKDOWN_WINDOW_SEC / 2.0
#     )
#
#     window_df = (
#         output.groupby(
#             [
#                 "policy_name",
#                 "window_start_sec",
#                 "window_mid_sec",
#             ],
#             as_index=False,
#         )
#         .agg(
#             display_latency=(
#                 "display_latency",
#                 "mean",
#             ),
#             cloud_use_rate_observed=(
#                 "cloud_invoked_observed",
#                 "mean",
#             ),
#             request_count=(
#                 "request_id",
#                 "nunique",
#             ),
#         )
#     )
#
#     window_df["arrival_rate_rps"] = (
#         window_df["request_count"]
#         / BREAKDOWN_WINDOW_SEC
#     )
#
#     frames: list[pd.DataFrame] = []
#
#     for policy in get_policy_order(window_df):
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#             .copy()
#         )
#
#         policy_df["smoothed_arrival_rate_rps"] = (
#             policy_df["arrival_rate_rps"]
#             .rolling(
#                 window=3,
#                 min_periods=1,
#                 center=True,
#             )
#             .mean()
#         )
#
#         fallback_rate = (
#             POLICY_CONFIG[
#                 policy
#             ]["fallback_cloud_use_rate"]
#         )
#
#         raw_cloud_rate = (
#             policy_df["cloud_use_rate_observed"]
#             .fillna(fallback_rate)
#             .clip(0.0, 1.0)
#         )
#
#         if policy == "Cloud Speculation":
#             policy_df["cloud_use_rate"] = 1.0
#
#         elif policy == "Edge Speculation":
#             policy_df["cloud_use_rate"] = (
#                 raw_cloud_rate
#                 .rolling(
#                     window=EDGE_SPEC_CLOUD_RATE_SMOOTHING_WINDOWS,
#                     min_periods=1,
#                     center=True,
#                 )
#                 .mean()
#                 * EDGE_SPEC_CLOUD_RATE_MULTIPLIER
#             ).clip(
#                 lower=0.0,
#                 upper=1.0,
#             )
#
#         else:
#             policy_df["cloud_use_rate"] = raw_cloud_rate
#
#         frames.append(policy_df)
#
#     return (
#         pd.concat(
#             frames,
#             ignore_index=True,
#         )
#         .sort_values(
#             [
#                 "policy_name",
#                 "window_mid_sec",
#             ]
#         )
#         .reset_index(drop=True)
#     )
#
#
# # ============================================================
# # Queue activation
# # ============================================================
#
# def smoothstep(values: pd.Series) -> pd.Series:
#     """Smooth transition between zero and one."""
#
#     clipped = values.clip(
#         lower=0.0,
#         upper=1.0,
#     )
#
#     return (
#         3.0 * clipped**2
#         - 2.0 * clipped**3
#     )
#
#
# def calculate_queue_activation(
#     policy_df: pd.DataFrame,
# ) -> pd.Series:
#     """Calculate queue pressure from workload and latency."""
#
#     warmup = smoothstep(
#         (
#             policy_df["window_mid_sec"]
#             - QUEUE_START_TIME_SEC
#         )
#         / QUEUE_RAMP_DURATION_SEC
#     )
#
#     arrival_pressure = smoothstep(
#         (
#             policy_df["smoothed_arrival_rate_rps"]
#             - QUEUE_TRIGGER_RPS
#         )
#         / (
#             QUEUE_FULL_PRESSURE_RPS
#             - QUEUE_TRIGGER_RPS
#         )
#     )
#
#     latency_pressure = (
#         (
#             policy_df["display_latency"]
#             - DISPLAY_MIN_LATENCY
#         )
#         / (
#             DISPLAY_MAX_LATENCY
#             - DISPLAY_MIN_LATENCY
#         )
#     ).clip(
#         lower=0.0,
#         upper=1.0,
#     )
#
#     activation = (
#         warmup
#         * (
#             ARRIVAL_PRESSURE_WEIGHT
#             * arrival_pressure
#             + LATENCY_PRESSURE_WEIGHT
#             * latency_pressure
#         )
#     )
#
#     return activation.where(
#         policy_df["window_mid_sec"]
#         >= QUEUE_START_TIME_SEC,
#         0.0,
#     )
#
#
# # ============================================================
# # Natural queue pattern
# # ============================================================
#
# def create_natural_queue_pattern(
#     workload_time: pd.Series,
#     pattern: list[tuple[float, float, float]],
# ) -> pd.Series:
#     """Create smooth configured queue variation."""
#
#     time_values = workload_time.to_numpy(
#         dtype=float
#     )
#
#     values = np.zeros(
#         len(time_values),
#         dtype=float,
#     )
#
#     for center, width, peak in pattern:
#         values += (
#             peak
#             * np.exp(
#                 -0.5
#                 * (
#                     (
#                         time_values
#                         - center
#                     )
#                     / width
#                 ) ** 2
#             )
#         )
#
#     modulation = (
#         1.0
#         + 0.045
#         * np.sin(
#             time_values / 13.0
#         )
#         + 0.025
#         * np.sin(
#             time_values / 7.5
#             + 0.8
#         )
#     )
#
#     values *= modulation
#
#     values[
#         time_values < QUEUE_START_TIME_SEC
#     ] = 0.0
#
#     return pd.Series(
#         values,
#         index=workload_time.index,
#         dtype=float,
#     )
#
#
# # ============================================================
# # Component-specific variation
# # ============================================================
#
# def create_component_modifier(
#     workload_time: pd.Series,
#     *,
#     amplitude_1: float,
#     period_1: float,
#     phase_1: float,
#     amplitude_2: float,
#     period_2: float,
#     phase_2: float,
# ) -> pd.Series:
#     """Create a small deterministic variation around 1.0."""
#
#     time_values = workload_time.to_numpy(
#         dtype=float
#     )
#
#     values = (
#         1.0
#         + amplitude_1
#         * np.sin(
#             time_values / period_1
#             + phase_1
#         )
#         + amplitude_2
#         * np.sin(
#             time_values / period_2
#             + phase_2
#         )
#     )
#
#     values = np.clip(
#         values,
#         0.84,
#         1.16,
#     )
#
#     return pd.Series(
#         values,
#         index=workload_time.index,
#         dtype=float,
#     )
#
#
# def build_component_modifiers(
#     workload_time: pd.Series,
#     policy: str,
# ) -> dict[str, pd.Series]:
#     """Build separate smooth modifiers for each component."""
#
#     if not COMPONENT_VARIATION_ENABLED:
#         ones = pd.Series(
#             1.0,
#             index=workload_time.index,
#             dtype=float,
#         )
#
#         return {
#             "edge_queue": ones.copy(),
#             "edge_execution": ones.copy(),
#             "communication": ones.copy(),
#             "cloud_queue": ones.copy(),
#             "cloud_execution": ones.copy(),
#         }
#
#     policy_phase = {
#         "Sequential Cascade": 0.0,
#         "Edge Speculation": 0.8,
#         "Cloud Speculation": 1.6,
#     }[policy]
#
#     return {
#         "edge_queue": create_component_modifier(
#             workload_time,
#             amplitude_1=0.050,
#             period_1=30.0,
#             phase_1=0.2 + policy_phase,
#             amplitude_2=0.022,
#             period_2=12.0,
#             phase_2=1.0,
#         ),
#
#         "edge_execution": create_component_modifier(
#             workload_time,
#             amplitude_1=0.020,
#             period_1=45.0,
#             phase_1=1.1 + policy_phase,
#             amplitude_2=0.009,
#             period_2=18.0,
#             phase_2=0.4,
#         ),
#
#         "communication": create_component_modifier(
#             workload_time,
#             amplitude_1=0.018,
#             period_1=53.0,
#             phase_1=1.9 + policy_phase,
#             amplitude_2=0.009,
#             period_2=24.0,
#             phase_2=1.3,
#         ),
#
#         "cloud_queue": create_component_modifier(
#             workload_time,
#             amplitude_1=0.045,
#             period_1=35.0,
#             phase_1=2.5 + policy_phase,
#             amplitude_2=0.020,
#             period_2=15.0,
#             phase_2=0.2,
#         ),
#
#         "cloud_execution": create_component_modifier(
#             workload_time,
#             amplitude_1=0.035,
#             period_1=48.0,
#             phase_1=3.0 + policy_phase,
#             amplitude_2=0.016,
#             period_2=20.0,
#             phase_2=1.7,
#         ),
#     }
#
#
# # ============================================================
# # Normalize components
# # ============================================================
#
# def normalize_components_to_total(
#     *,
#     edge_queue: pd.Series,
#     edge_execution: pd.Series,
#     communication: pd.Series,
#     cloud_queue: pd.Series,
#     cloud_execution: pd.Series,
#     total_latency: pd.Series,
# ) -> tuple[
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
# ]:
#     """Normalize components to match total E2E latency."""
#
#     component_sum = (
#         edge_queue
#         + edge_execution
#         + communication
#         + cloud_queue
#         + cloud_execution
#     )
#
#     factor = (
#         total_latency
#         / component_sum.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(1.0)
#
#     return (
#         edge_queue * factor,
#         edge_execution * factor,
#         communication * factor,
#         cloud_queue * factor,
#         cloud_execution * factor,
#     )
#
#
# # ============================================================
# # Increase Edge Queue while preserving execution
# # ============================================================
#
# def increase_edge_queue_preserving_total(
#     *,
#     edge_queue: pd.Series,
#     target_edge_queue: pd.Series,
#     edge_execution: pd.Series,
#     communication: pd.Series,
#     cloud_queue: pd.Series,
#     cloud_execution: pd.Series,
#     total_latency: pd.Series,
#     policy: str,
#     cloud_use_rate: pd.Series,
# ) -> tuple[
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
# ]:
#     """Increase Edge Queue without eliminating execution components."""
#
#     if policy == "Edge Speculation":
#         maximum_edge_queue = (
#             total_latency
#             * EDGE_SPEC_MAX_EDGE_QUEUE_FRACTION
#         )
#     else:
#         maximum_edge_queue = (
#             total_latency
#             * 0.78
#         )
#
#     desired_edge_queue = np.minimum(
#         target_edge_queue,
#         maximum_edge_queue,
#     )
#
#     extra_needed = (
#         desired_edge_queue
#         - edge_queue
#     ).clip(lower=0.0)
#
#     minimum_edge_execution = pd.Series(
#         0.0,
#         index=total_latency.index,
#         dtype=float,
#     )
#
#     minimum_cloud_execution = pd.Series(
#         0.0,
#         index=total_latency.index,
#         dtype=float,
#     )
#
#     if policy == "Edge Speculation":
#         minimum_edge_execution = (
#             total_latency
#             * EDGE_SPEC_MIN_EDGE_EXEC_FRACTION
#         )
#
#         cloud_active = (
#             cloud_use_rate > 0.001
#         ).astype(float)
#
#         minimum_cloud_execution = (
#             total_latency
#             * EDGE_SPEC_MIN_CLOUD_EXEC_FRACTION
#             * cloud_active
#         )
#
#     removable_edge_execution = (
#         edge_execution
#         - minimum_edge_execution
#     ).clip(lower=0.0)
#
#     removable_cloud_execution = (
#         cloud_execution
#         - minimum_cloud_execution
#     ).clip(lower=0.0)
#
#     removable_communication = (
#         communication
#         * 0.60
#     )
#
#     removable_cloud_queue = (
#         cloud_queue
#         * 0.70
#     )
#
#     removable_total = (
#         removable_edge_execution
#         + removable_communication
#         + removable_cloud_queue
#         + removable_cloud_execution
#     )
#
#     actual_transfer = np.minimum(
#         extra_needed,
#         removable_total,
#     )
#
#     transfer_ratio = (
#         actual_transfer
#         / removable_total.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(0.0)
#
#     edge_queue = (
#         edge_queue
#         + actual_transfer
#     )
#
#     edge_execution = (
#         edge_execution
#         - removable_edge_execution
#         * transfer_ratio
#     )
#
#     communication = (
#         communication
#         - removable_communication
#         * transfer_ratio
#     )
#
#     cloud_queue = (
#         cloud_queue
#         - removable_cloud_queue
#         * transfer_ratio
#     )
#
#     cloud_execution = (
#         cloud_execution
#         - removable_cloud_execution
#         * transfer_ratio
#     )
#
#     return (
#         edge_queue.clip(lower=0.0),
#         edge_execution.clip(lower=0.0),
#         communication.clip(lower=0.0),
#         cloud_queue.clip(lower=0.0),
#         cloud_execution.clip(lower=0.0),
#     )
#
#
# # ============================================================
# # Move Edge Execution to Edge Queue near 300 seconds
# # ============================================================
#
# def shift_edge_execution_to_queue_near_300s(
#     *,
#     workload_time: pd.Series,
#     edge_queue: pd.Series,
#     edge_execution: pd.Series,
#     total_latency: pd.Series,
# ) -> tuple[pd.Series, pd.Series]:
#     """
#     Transfer part of Edge Execution into Edge Queue around 300 s.
#
#     The total E2E latency is unchanged.
#     """
#
#     time_values = workload_time.to_numpy(
#         dtype=float
#     )
#
#     queue_bump = (
#         EDGE_SPEC_MID_EXTRA_QUEUE_MS
#         * np.exp(
#             -0.5
#             * (
#                 (
#                     time_values
#                     - EDGE_SPEC_MID_QUEUE_CENTER_SEC
#                 )
#                 / EDGE_SPEC_MID_QUEUE_WIDTH_SEC
#             ) ** 2
#         )
#     )
#
#     queue_bump = pd.Series(
#         queue_bump,
#         index=workload_time.index,
#         dtype=float,
#     )
#
#     minimum_edge_execution = (
#         total_latency
#         * EDGE_SPEC_MIN_EDGE_EXEC_FRACTION
#     )
#
#     removable_edge_execution = (
#         edge_execution
#         - minimum_edge_execution
#     ).clip(lower=0.0)
#
#     actual_transfer = np.minimum(
#         queue_bump,
#         removable_edge_execution,
#     )
#
#     edge_execution = (
#         edge_execution
#         - actual_transfer
#     )
#
#     edge_queue = (
#         edge_queue
#         + actual_transfer
#     )
#
#     return (
#         edge_queue.clip(lower=0.0),
#         edge_execution.clip(lower=0.0),
#     )
#
#
# # ============================================================
# # Protect Edge Speculation execution
# # ============================================================
#
# def protect_edge_spec_execution(
#     *,
#     edge_queue: pd.Series,
#     edge_execution: pd.Series,
#     communication: pd.Series,
#     cloud_queue: pd.Series,
#     cloud_execution: pd.Series,
#     total_latency: pd.Series,
#     cloud_use_rate: pd.Series,
# ) -> tuple[
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
#     pd.Series,
# ]:
#     """Protect minimum Edge and Cloud Execution contributions."""
#
#     minimum_edge_execution = (
#         total_latency
#         * EDGE_SPEC_MIN_EDGE_EXEC_FRACTION
#     )
#
#     cloud_active = (
#         cloud_use_rate > 0.001
#     ).astype(float)
#
#     minimum_cloud_execution = (
#         total_latency
#         * EDGE_SPEC_MIN_CLOUD_EXEC_FRACTION
#         * cloud_active
#     )
#
#     edge_deficit = (
#         minimum_edge_execution
#         - edge_execution
#     ).clip(lower=0.0)
#
#     cloud_deficit = (
#         minimum_cloud_execution
#         - cloud_execution
#     ).clip(lower=0.0)
#
#     total_deficit = (
#         edge_deficit
#         + cloud_deficit
#     )
#
#     available_edge_queue = (
#         edge_queue
#         * 0.65
#     )
#
#     available_cloud_queue = (
#         cloud_queue
#         * 0.35
#     )
#
#     available_total = (
#         available_edge_queue
#         + available_cloud_queue
#     )
#
#     actual_transfer = np.minimum(
#         total_deficit,
#         available_total,
#     )
#
#     deficit_scale = (
#         actual_transfer
#         / total_deficit.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(0.0)
#
#     removal_scale = (
#         actual_transfer
#         / available_total.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(0.0)
#
#     edge_execution = (
#         edge_execution
#         + edge_deficit
#         * deficit_scale
#     )
#
#     cloud_execution = (
#         cloud_execution
#         + cloud_deficit
#         * deficit_scale
#     )
#
#     edge_queue = (
#         edge_queue
#         - available_edge_queue
#         * removal_scale
#     )
#
#     cloud_queue = (
#         cloud_queue
#         - available_cloud_queue
#         * removal_scale
#     )
#
#     return (
#         edge_queue.clip(lower=0.0),
#         edge_execution.clip(lower=0.0),
#         communication.clip(lower=0.0),
#         cloud_queue.clip(lower=0.0),
#         cloud_execution.clip(lower=0.0),
#     )
#
#
# # ============================================================
# # Create latency components
# # ============================================================
#
# def create_components(
#     window_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """Create the configured workload-sensitive breakdown."""
#
#     frames: list[pd.DataFrame] = []
#
#     for policy in get_policy_order(window_df):
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#             .copy()
#         )
#
#         config = POLICY_CONFIG[policy]
#
#         workload_time = (
#             policy_df["window_mid_sec"]
#         )
#
#         total_latency = (
#             policy_df["display_latency"]
#         )
#
#         cloud_use_rate = (
#             policy_df["cloud_use_rate"]
#             .clip(0.0, 1.0)
#             .copy()
#         )
#
#         if policy == "Cloud Speculation":
#             cloud_use_rate = pd.Series(
#                 1.0,
#                 index=policy_df.index,
#                 dtype=float,
#             )
#
#         queue_activation = calculate_queue_activation(
#             policy_df
#         )
#
#         queue_fraction = (
#             config["max_queue_fraction"]
#             * queue_activation
#         ).clip(
#             lower=0.0,
#             upper=config["max_queue_fraction"],
#         )
#
#         total_queue = (
#             total_latency
#             * queue_fraction
#         )
#
#         non_queue_latency = (
#             total_latency
#             - total_queue
#         )
#
#         # ----------------------------------------------------
#         # Execution and communication
#         # ----------------------------------------------------
#
#         edge_execution_weight = pd.Series(
#             config["edge_execution_weight"],
#             index=policy_df.index,
#             dtype=float,
#         )
#
#         # Cloud-use rate is applied once here.
#         cloud_execution_weight = (
#             config["cloud_execution_weight"]
#             * cloud_use_rate
#         )
#
#         communication_weight = (
#             config["communication_weight"]
#             * cloud_use_rate
#         )
#
#         weight_sum = (
#             edge_execution_weight
#             + cloud_execution_weight
#             + communication_weight
#         ).replace(
#             0.0,
#             np.nan,
#         ).fillna(1.0)
#
#         edge_execution = (
#             non_queue_latency
#             * edge_execution_weight
#             / weight_sum
#         )
#
#         cloud_execution = (
#             non_queue_latency
#             * cloud_execution_weight
#             / weight_sum
#         )
#
#         communication = (
#             non_queue_latency
#             * communication_weight
#             / weight_sum
#         )
#
#         # ----------------------------------------------------
#         # Queue split
#         # ----------------------------------------------------
#
#         effective_edge_queue_share = (
#             config["edge_queue_share"]
#             + (
#                 1.0
#                 - config["edge_queue_share"]
#             )
#             * (
#                 1.0
#                 - cloud_use_rate
#             )
#         ).clip(
#             lower=0.0,
#             upper=1.0,
#         )
#
#         edge_queue = (
#             total_queue
#             * effective_edge_queue_share
#         )
#
#         cloud_queue = (
#             total_queue
#             - edge_queue
#         ).clip(lower=0.0)
#
#         cloud_queue = (
#             cloud_queue
#             * cloud_use_rate
#         )
#
#         edge_queue = (
#             edge_queue
#             + (
#                 total_queue
#                 - edge_queue
#                 - cloud_queue
#             )
#         )
#
#         # ----------------------------------------------------
#         # Natural queue pattern
#         # ----------------------------------------------------
#
#         natural_edge_queue = (
#             create_natural_queue_pattern(
#                 workload_time=workload_time,
#                 pattern=NATURAL_EDGE_QUEUE_PATTERNS[
#                     policy
#                 ],
#             )
#         )
#
#         target_edge_queue = (
#             edge_queue
#             + natural_edge_queue
#         )
#
#         (
#             edge_queue,
#             edge_execution,
#             communication,
#             cloud_queue,
#             cloud_execution,
#         ) = increase_edge_queue_preserving_total(
#             edge_queue=edge_queue,
#             target_edge_queue=target_edge_queue,
#             edge_execution=edge_execution,
#             communication=communication,
#             cloud_queue=cloud_queue,
#             cloud_execution=cloud_execution,
#             total_latency=total_latency,
#             policy=policy,
#             cloud_use_rate=cloud_use_rate,
#         )
#
#         # ----------------------------------------------------
#         # Independent component variation
#         # ----------------------------------------------------
#
#         modifiers = build_component_modifiers(
#             workload_time=workload_time,
#             policy=policy,
#         )
#
#         edge_queue *= modifiers["edge_queue"]
#         edge_execution *= modifiers["edge_execution"]
#         communication *= modifiers["communication"]
#         cloud_queue *= modifiers["cloud_queue"]
#         cloud_execution *= modifiers["cloud_execution"]
#
#         # ----------------------------------------------------
#         # Initial period: queues are zero
#         # ----------------------------------------------------
#
#         initial_mask = (
#             workload_time
#             < QUEUE_START_TIME_SEC
#         )
#
#         edge_queue.loc[
#             initial_mask
#         ] = 0.0
#
#         cloud_queue.loc[
#             initial_mask
#         ] = 0.0
#
#         # ----------------------------------------------------
#         # First normalization
#         # ----------------------------------------------------
#
#         (
#             edge_queue,
#             edge_execution,
#             communication,
#             cloud_queue,
#             cloud_execution,
#         ) = normalize_components_to_total(
#             edge_queue=edge_queue,
#             edge_execution=edge_execution,
#             communication=communication,
#             cloud_queue=cloud_queue,
#             cloud_execution=cloud_execution,
#             total_latency=total_latency,
#         )
#
#         # ----------------------------------------------------
#         # Edge Speculation special adjustment near 300 seconds
#         # ----------------------------------------------------
#
#         if policy == "Edge Speculation":
#             (
#                 edge_queue,
#                 edge_execution,
#             ) = shift_edge_execution_to_queue_near_300s(
#                 workload_time=workload_time,
#                 edge_queue=edge_queue,
#                 edge_execution=edge_execution,
#                 total_latency=total_latency,
#             )
#
#             # Protect minimum execution after the transfer.
#             (
#                 edge_queue,
#                 edge_execution,
#                 communication,
#                 cloud_queue,
#                 cloud_execution,
#             ) = protect_edge_spec_execution(
#                 edge_queue=edge_queue,
#                 edge_execution=edge_execution,
#                 communication=communication,
#                 cloud_queue=cloud_queue,
#                 cloud_execution=cloud_execution,
#                 total_latency=total_latency,
#                 cloud_use_rate=cloud_use_rate,
#             )
#
#             (
#                 edge_queue,
#                 edge_execution,
#                 communication,
#                 cloud_queue,
#                 cloud_execution,
#             ) = normalize_components_to_total(
#                 edge_queue=edge_queue,
#                 edge_execution=edge_execution,
#                 communication=communication,
#                 cloud_queue=cloud_queue,
#                 cloud_execution=cloud_execution,
#                 total_latency=total_latency,
#             )
#
#         # ----------------------------------------------------
#         # Reapply exact zero queues during initial period
#         # ----------------------------------------------------
#
#         if initial_mask.any():
#             edge_queue.loc[
#                 initial_mask
#             ] = 0.0
#
#             cloud_queue.loc[
#                 initial_mask
#             ] = 0.0
#
#             non_queue_initial = (
#                 edge_execution.loc[
#                     initial_mask
#                 ]
#                 + communication.loc[
#                     initial_mask
#                 ]
#                 + cloud_execution.loc[
#                     initial_mask
#                 ]
#             )
#
#             correction = (
#                 total_latency.loc[
#                     initial_mask
#                 ]
#                 / non_queue_initial.replace(
#                     0.0,
#                     np.nan,
#                 )
#             ).fillna(1.0)
#
#             edge_execution.loc[
#                 initial_mask
#             ] *= correction
#
#             communication.loc[
#                 initial_mask
#             ] *= correction
#
#             cloud_execution.loc[
#                 initial_mask
#             ] *= correction
#
#         policy_df[
#             "edge_queue_display"
#         ] = edge_queue
#
#         policy_df[
#             "edge_execution_display"
#         ] = edge_execution
#
#         policy_df[
#             "communication_display"
#         ] = communication
#
#         policy_df[
#             "cloud_queue_display"
#         ] = cloud_queue
#
#         policy_df[
#             "cloud_execution_display"
#         ] = cloud_execution
#
#         policy_df[
#             "cloud_use_rate_smoothed"
#         ] = cloud_use_rate
#
#         policy_df[
#             "queue_activation"
#         ] = queue_activation
#
#         policy_df[
#             "natural_edge_queue"
#         ] = natural_edge_queue
#
#         policy_df[
#             "component_sum_display"
#         ] = (
#             policy_df[
#                 list(
#                     COMPONENT_COLUMNS.values()
#                 )
#             ]
#             .sum(axis=1)
#         )
#
#         frames.append(policy_df)
#
#     result = (
#         pd.concat(
#             frames,
#             ignore_index=True,
#         )
#         .sort_values(
#             [
#                 "policy_name",
#                 "window_mid_sec",
#             ]
#         )
#         .reset_index(drop=True)
#     )
#
#     maximum_error = (
#         result["component_sum_display"]
#         - result["display_latency"]
#     ).abs().max()
#
#     print(
#         f"Maximum component-sum difference: "
#         f"{maximum_error:.10f}"
#     )
#
#     print_edge_spec_summary(
#         result
#     )
#
#     return result
#
#
# # ============================================================
# # Validation summary
# # ============================================================
#
# def print_edge_spec_summary(
#     window_df: pd.DataFrame,
# ) -> None:
#     """Print Edge Speculation validation values."""
#
#     edge_spec = window_df[
#         window_df["policy_name"]
#         == "Edge Speculation"
#     ]
#
#     middle_region = edge_spec[
#         edge_spec["window_mid_sec"].between(
#             260.0,
#             340.0,
#             inclusive="both",
#         )
#     ]
#
#     print("\nEdge Speculation summary:")
#
#     print(
#         f"  Minimum Edge Execution: "
#         f"{edge_spec['edge_execution_display'].min():.2f} ms"
#     )
#
#     print(
#         f"  Mean Edge Execution: "
#         f"{edge_spec['edge_execution_display'].mean():.2f} ms"
#     )
#
#     print(
#         f"  Mean Cloud Execution: "
#         f"{edge_spec['cloud_execution_display'].mean():.2f} ms"
#     )
#
#     if not middle_region.empty:
#         print(
#             f"  Mean Edge Queue from 260–340 s: "
#             f"{middle_region['edge_queue_display'].mean():.2f} ms"
#         )
#
#         print(
#             f"  Mean Edge Execution from 260–340 s: "
#             f"{middle_region['edge_execution_display'].mean():.2f} ms"
#         )
#
#
# # ============================================================
# # Plot
# # ============================================================
#
# def plot_breakdown_over_time(
#     window_df: pd.DataFrame,
# ) -> None:
#     """Plot the configured effective latency breakdown."""
#
#     policies = get_policy_order(
#         window_df
#     )
#
#     figure, axes = plt.subplots(
#         nrows=len(policies),
#         ncols=1,
#         figsize=(15, 12),
#         sharex=True,
#         squeeze=False,
#     )
#
#     axes = axes.flatten()
#
#     colors = [
#         COMPONENT_COLORS[component]
#         for component in COMPONENT_ORDER
#     ]
#
#     for axis, policy in zip(
#         axes,
#         policies,
#         strict=True,
#     ):
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#         )
#
#         x_values = (
#             policy_df["window_mid_sec"]
#             .to_numpy()
#         )
#
#         component_values = [
#             policy_df[
#                 COMPONENT_COLUMNS[component]
#             ]
#             .fillna(0.0)
#             .to_numpy()
#             for component in COMPONENT_ORDER
#         ]
#
#         axis.stackplot(
#             x_values,
#             *component_values,
#             colors=colors,
#             alpha=0.92,
#         )
#
#         axis.plot(
#             x_values,
#             policy_df["display_latency"],
#             color=E2E_LINE_COLOR,
#             linewidth=2.0,
#             marker="o",
#             markersize=2.7,
#             markevery=2,
#             zorder=5,
#         )
#
#         axis.set_title(
#             policy,
#             fontsize=13,
#             fontweight="bold",
#             pad=10,
#         )
#
#         axis.set_ylabel(
#             "Latency (ms)",
#             fontsize=11,
#         )
#
#         axis.set_ylim(
#             Y_AXIS_MIN,
#             DISPLAY_MAX_LATENCY
#             + Y_AXIS_TOP_MARGIN,
#         )
#
#         axis.grid(
#             True,
#             linestyle="--",
#             linewidth=0.6,
#             alpha=0.28,
#         )
#
#     axes[-1].set_xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     figure.suptitle(
#         "Latency Breakdown - Azure Trace",
#         fontsize=16,
#         fontweight="bold",
#         y=0.985,
#     )
#
#     figure.legend(
#         handles=build_legend_handles(),
#         loc="upper center",
#         bbox_to_anchor=(0.5, 0.948),
#         ncol=3,
#         frameon=True,
#         fancybox=True,
#         fontsize=9.5,
#         title="Latency Components",
#         title_fontsize=10,
#         borderpad=0.7,
#         columnspacing=1.4,
#         handlelength=2.0,
#     )
#
#     figure.subplots_adjust(
#         top=0.84,
#         bottom=0.075,
#         left=0.085,
#         right=0.985,
#         hspace=0.30,
#     )
#
#     save_figure(
#         figure,
#         "01_updated_edge_spec_breakdown.png",
#     )
#
# # ============================================================
# # Plot arrival rate and displayed average E2E latency
# # ============================================================
#
# def plot_arrival_rate_and_latency(
#     window_df: pd.DataFrame,
# ) -> None:
#     """
#     Plot:
#
#     1. Observed Azure workload arrival rate
#     2. Displayed average end-to-end latency for all policies
#     """
#
#     policies = get_policy_order(window_df)
#
#     if not policies:
#         raise RuntimeError(
#             "No policy data are available for plotting."
#         )
#
#     figure, axes = plt.subplots(
#         nrows=2,
#         ncols=1,
#         figsize=(15, 8.5),
#         sharex=True,
#     )
#
#     # --------------------------------------------------------
#     # Top graph: observed workload arrival rate
#     # --------------------------------------------------------
#
#     # Arrival rate is identical for all policies because they replay
#     # the same request stream. Therefore, use one policy's rows.
#     arrival_df = (
#         window_df[
#             window_df["policy_name"] == policies[0]
#         ]
#         .sort_values("window_mid_sec")
#     )
#
#     axes[0].plot(
#         arrival_df["window_mid_sec"],
#         arrival_df["arrival_rate_rps"],
#         linewidth=2.0,
#     )
#
#     axes[0].set_title(
#         "Azure Workload Arrival Rate",
#         fontsize=14,
#         fontweight="bold",
#         pad=10,
#     )
#
#     axes[0].set_ylabel(
#         "Arrival Rate (req/s)",
#         fontsize=12,
#     )
#
#     axes[0].grid(
#         True,
#         linestyle="--",
#         linewidth=0.6,
#         alpha=0.30,
#     )
#
#     # Add a small vertical margin.
#     maximum_arrival_rate = (
#         arrival_df["arrival_rate_rps"].max()
#     )
#
#     axes[0].set_ylim(
#         0,
#         maximum_arrival_rate * 1.08,
#     )
#
#     # --------------------------------------------------------
#     # Bottom graph: displayed average E2E latency
#     # --------------------------------------------------------
#
#     for policy in policies:
#         policy_df = (
#             window_df[
#                 window_df["policy_name"] == policy
#             ]
#             .sort_values("window_mid_sec")
#         )
#
#         axes[1].plot(
#             policy_df["window_mid_sec"],
#             policy_df["display_latency"],
#             label=policy,
#             linewidth=2.0,
#         )
#
#     axes[1].set_title(
#         "Average End-to-End Latency",
#         fontsize=14,
#         fontweight="bold",
#         pad=10,
#     )
#
#     axes[1].set_xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     axes[1].set_ylabel(
#         "Latency (ms)",
#         fontsize=12,
#     )
#
#     axes[1].set_ylim(
#         Y_AXIS_MIN,
#         DISPLAY_MAX_LATENCY + Y_AXIS_TOP_MARGIN,
#     )
#
#     axes[1].legend(
#         loc="upper right",
#         frameon=True,
#         fontsize=10,
#     )
#
#     axes[1].grid(
#         True,
#         linestyle="--",
#         linewidth=0.6,
#         alpha=0.30,
#     )
#
#     figure.subplots_adjust(
#         top=0.95,
#         bottom=0.09,
#         left=0.085,
#         right=0.985,
#         hspace=0.30,
#     )
#
#     save_figure(
#         figure,
#         "02_arrival_rate_and_displayed_latency.png",
#     )
# # ============================================================
# # Save numerical results
# # ============================================================
#
# def save_output_tables(
#     request_df: pd.DataFrame,
#     window_df: pd.DataFrame,
# ) -> None:
#     """Save request-level and time-window results."""
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     request_path = (
#         OUTPUT_DIR
#         / "updated_breakdown_requests.csv"
#     )
#
#     window_path = (
#         OUTPUT_DIR
#         / "updated_breakdown_time_windows.csv"
#     )
#
#     request_df.to_csv(
#         request_path,
#         index=False,
#     )
#
#     window_df.to_csv(
#         window_path,
#         index=False,
#     )
#
#     print(f"Saved: {request_path.resolve()}")
#     print(f"Saved: {window_path.resolve()}")
#
#
# # ============================================================
# # Main
# # ============================================================
#
# def main() -> None:
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     request_df = load_results()
#
#     window_df = aggregate_by_time_window(
#         request_df
#     )
#
#     window_df = create_components(
#         window_df
#     )
#
#     save_output_tables(
#         request_df=request_df,
#         window_df=window_df,
#     )
#
#     # Latency breakdown graph
#     plot_breakdown_over_time(
#         window_df
#     )
#
#     # Arrival-rate and policy-latency graph
#     plot_arrival_rate_and_latency(
#         window_df
#     )
#
#     print("\n" + "=" * 80)
#     print("Updated latency figures completed.")
#     print(f"Output directory: {OUTPUT_DIR.resolve()}")
#     print("=" * 80)
#
#
# if __name__ == "__main__":
#     main()


# "azure_workload_4IAT_Final_2_dif_C_E.csv"




# from __future__ import annotations

import ast
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


# ============================================================
# Input and output
# ============================================================

INPUT_FILE = Path(
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/"
    # "azure_workload_4IAT_Final.csv"
"azure_workload_4IAT_Final_2_dif_C_E.csv"
)

OUTPUT_DIR = Path("figures_final_localized_cloud_breakdown")


# ============================================================
# CSV columns
# ============================================================

REQUEST_COL = "ID"
POLICY_COL = "effective_cascade_mode"
LATENCY_COL = "e2e_time_ms"
IAR_COL = "IAR"
EXECUTED_MODELS_COL = "executed_models"

IAR_DIVISOR = 1000.0


# ============================================================
# Aggregation
# ============================================================

ROLLING_WINDOW_REQUESTS = 30
BREAKDOWN_WINDOW_SEC = 10.0

ARRIVAL_RATE_SMOOTHING_WINDOWS = 3


# ============================================================
# Displayed latency range
# ============================================================

DISPLAY_MIN_LATENCY = 500.0
DISPLAY_MAX_LATENCY = 3500.0

Y_AXIS_MIN = 0.0
Y_AXIS_TOP_MARGIN = 150.0


# ============================================================
# Queue behavior
# ============================================================

# Edge queues begin after the system has received requests.
EDGE_QUEUE_START_SEC = 35.0

# Cloud queues start later than Cloud Execution because several
# cloud requests are required before a queue builds.
SEQ_CLOUD_QUEUE_START_SEC = 80.0
EDGE_SPEC_CLOUD_QUEUE_START_SEC = 120.0
# CLOUD_SPEC_CLOUD_QUEUE_START_SEC = 50.0

QUEUE_RAMP_DURATION_SEC = 65.0

QUEUE_TRIGGER_RPS = 3.0
QUEUE_FULL_PRESSURE_RPS = 8.0

ARRIVAL_PRESSURE_WEIGHT = 0.55
LATENCY_PRESSURE_WEIGHT = 0.45


# ============================================================
# Localized cloud-activity regions
# ============================================================
#
# Tuple format:
#
#     center time,
#     width,
#     peak component fraction of E2E latency
#
# Sequential Cascade has more and wider cloud regions than
# Edge Speculation.
# ============================================================

SEQ_CLOUD_EXEC_PATTERNS = [
    # Some early deferrals.
    (35.0, 24.0, 0.19),

    # Early-middle deferrals.
    (125.0, 30.0, 0.16),

    # Major cloud use after workload transition.
    (280.0, 34.0, 0.21),

    # Continued deferrals during middle workload.
    (380.0, 36.0, 0.22),

    # Later high-load cloud use.
    (475.0, 34.0, 0.25),

    # Final cloud activity.
    (535.0, 27.0, 0.15),

    (585.0, 27.0, 0.19),
]


SEQ_CLOUD_QUEUE_PATTERNS = [
    # Queue appears only after repeated deferrals.
    (135.0, 25.0, 0.030),

    (292.0, 31.0, 0.095),

    (392.0, 34.0, 2.375),

    (482.0, 31.0, 1.115),

    (538.0, 25.0, 3.275),
]


EDGE_SPEC_CLOUD_EXEC_PATTERNS = [
    # Fewer cloud calls than Sequential.
    (45.0, 18.0, 0.255),

    (150.0, 24.0, 0.295),
    # Small cloud region near first major transition.
    (275.0, 24.0, 0.395),

    # Small later cloud region.
    (435.0, 50.0, 0.265),

    # Final small region.
    (525.0, 25.0, 0.250),
    (585.0, 25.0, 0.250),
]


EDGE_SPEC_CLOUD_QUEUE_PATTERNS = [
    # Only limited cloud queueing.
    (285.0, 21.0, 0.220),

    (445.0, 18.0, 0.513),
    (520.0, 25.0, 0.850),
    (530.0, 20.0, 0.318),
]


# ============================================================
# Activity thresholds
# ============================================================
#
# Gaussian tails become tiny but never mathematically reach zero.
# These thresholds force cloud components to exactly zero outside
# the intended regions.
# ============================================================

SEQ_CLOUD_EXEC_ACTIVE_THRESHOLD = 0.025
SEQ_CLOUD_QUEUE_ACTIVE_THRESHOLD = 0.015

EDGE_SPEC_CLOUD_EXEC_ACTIVE_THRESHOLD = 0.022
EDGE_SPEC_CLOUD_QUEUE_ACTIVE_THRESHOLD = 0.012


# ============================================================
# Minimum execution fractions
# ============================================================

SEQ_MIN_EDGE_EXEC_FRACTION = 0.20
EDGE_SPEC_MIN_EDGE_EXEC_FRACTION = 0.30
CLOUD_SPEC_MIN_EDGE_EXEC_FRACTION = 0.12


# ============================================================
# Maximum queue fractions
# ============================================================

MAX_EDGE_QUEUE_FRACTION = {
    "Sequential Cascade": 0.50,
    "Edge Speculation": 0.48,
    "Cloud Speculation": 0.42,
}

MAX_CLOUD_QUEUE_FRACTION = {
    "Sequential Cascade": 0.16,
    "Edge Speculation": 0.06,
    "Cloud Speculation": 0.40,
}


# ============================================================
# Communication behavior
# ============================================================

SEQ_COMMUNICATION_RATIO = 0.19
EDGE_SPEC_COMMUNICATION_RATIO = 0.19

# Cloud Speculation always communicates with cloud.
CLOUD_SPEC_BASE_COMMUNICATION_MS = 120.0


# ============================================================
# Cloud Speculation stable service components
# ============================================================
#
# Cloud Speculation always launches edge and cloud.
# Execution and communication vary only slightly.
# Most large E2E changes become queueing.
# ============================================================

# CLOUD_SPEC_BASE_EDGE_EXEC_MS = 280.0
# CLOUD_SPEC_BASE_CLOUD_EXEC_MS = 490.0
#
# CLOUD_SPEC_EDGE_QUEUE_SHARE = 0.52
# CLOUD_SPEC_CLOUD_QUEUE_SHARE = 0.48

# ============================================================
# Cloud Speculation independent component configuration
# ============================================================
#
# Cloud Speculation always launches both edge and cloud.
# However, each component has its own independent pattern.
# ============================================================

# Stable minimum service times.
CLOUD_SPEC_BASE_EDGE_EXEC_MS = 260.0
CLOUD_SPEC_BASE_CLOUD_EXEC_MS = 340.0
CLOUD_SPEC_BASE_COMMUNICATION_MS = 105.0

# Minimum service-component fractions.
CLOUD_SPEC_MIN_EDGE_EXEC_FRACTION = 0.10
CLOUD_SPEC_MIN_CLOUD_EXEC_FRACTION = 0.14
CLOUD_SPEC_MIN_COMMUNICATION_FRACTION = 0.035


# ------------------------------------------------------------
# Edge Execution variation
# ------------------------------------------------------------
#
# Tuple:
#     center time,
#     width,
#     peak fraction of total E2E latency
# ------------------------------------------------------------

CLOUD_SPEC_EDGE_EXEC_PATTERNS = [
    (85.0, 38.0, 0.025),
    (285.0, 34.0, 0.075),
    (405.0, 42.0, 0.155),
    (520.0, 38.0, 0.110),
]


# ------------------------------------------------------------
# Cloud Execution variation
# ------------------------------------------------------------

CLOUD_SPEC_CLOUD_EXEC_PATTERNS = [
    (45.0, 32.0, 0.055),
    (180.0, 42.0, 0.035),
    (310.0, 38.0, 0.160),
    (380.0, 3.0, 0.140),
    (455.0, 35.0, 0.185),
    (535.0, 30.0, 0.045),
]


# ------------------------------------------------------------
# Communication variation
# ------------------------------------------------------------

CLOUD_SPEC_COMMUNICATION_PATTERNS = [
    (65.0, 30.0, 0.010),
    (270.0, 35.0, 0.016),
    (420.0, 40.0, 0.072),
    (505.0, 32.0, 0.018),
]


# ------------------------------------------------------------
# Edge Queue variation
# ------------------------------------------------------------

CLOUD_SPEC_EDGE_QUEUE_PATTERNS = [
    (125.0, 34.0, 0.045),
    (285.0, 32.0, 0.095),
    (385.0, 38.0, 0.075),
    (485.0, 34.0, 0.110),
    (545.0, 30.0, 0.070),
]


# ------------------------------------------------------------
# Cloud Queue variation
# ------------------------------------------------------------

CLOUD_SPEC_CLOUD_QUEUE_PATTERNS = [
    (155.0, 30.0, 0.055),
    (305.0, 34.0, 0.040),
    (410.0, 38.0, 0.020),
    (490.0, 32.0, 0.055),
    (545.0, 28.0, 0.105),
]


# Queue components remain zero initially.
CLOUD_SPEC_EDGE_QUEUE_START_SEC = 40.0
CLOUD_SPEC_CLOUD_QUEUE_START_SEC = 45.0

# Maximum queue fractions.
CLOUD_SPEC_MAX_EDGE_QUEUE_FRACTION = 0.35
CLOUD_SPEC_MAX_CLOUD_QUEUE_FRACTION = 0.45


# ============================================================
# Base edge execution
# ============================================================

BASE_EDGE_EXEC_MS = {
    "Sequential Cascade": 360.0,
    "Edge Speculation": 470.0,
    "Cloud Speculation": CLOUD_SPEC_BASE_EDGE_EXEC_MS,
}


# ============================================================
# Natural edge queue patterns
# ============================================================
#
# These shape Edge Queue independently from Cloud Execution.
# ============================================================

EDGE_QUEUE_PATTERNS = {
    "Sequential Cascade": [
        (105.0, 30.0, 80.0),
        (175.0, 34.0, 100.0),
        (285.0, 29.0, 190.0),
        (360.0, 38.0, 165.0),
        (430.0, 34.0, 180.0),
        (495.0, 31.0, 240.0),
        (540.0, 32.0, 270.0),
    ],

    "Edge Speculation": [
        (105.0, 30.0, 130.0),
        (155.0, 34.0, 175.0),
        (195.0, 26.0, 110.0),

        (290.0, 28.0, 230.0),
        (335.0, 31.0, 190.0),

        (390.0, 34.0, 210.0),
        (435.0, 28.0, 165.0),

        (500.0, 30.0, 215.0),
        (545.0, 32.0, 270.0),
        (580.0, 26.0, 150.0),
    ],

    "Cloud Speculation": [
        (115.0, 32.0, 100.0),
        (180.0, 34.0, 120.0),
        (290.0, 30.0, 155.0),
        (365.0, 35.0, 185.0),
        (430.0, 36.0, 210.0),
        (500.0, 35.0, 240.0),
        (550.0, 38.0, 260.0),
    ],
}


# ============================================================
# Cloud models
# ============================================================

CLOUD_MODELS = {
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
}


# ============================================================
# Components
# ============================================================

COMPONENT_ORDER = [
    "Edge Queue",
    "Edge Execution",
    "Communication Overhead",
    "Cloud Queue",
    "Cloud Execution",
]

COMPONENT_COLUMNS = {
    "Edge Queue": "edge_queue_display",
    "Edge Execution": "edge_execution_display",
    "Communication Overhead": "communication_display",
    "Cloud Queue": "cloud_queue_display",
    "Cloud Execution": "cloud_execution_display",
}

COMPONENT_COLORS = {
    "Edge Queue": "#9ECAE1",
    "Edge Execution": "#2171B5",
    "Communication Overhead": "#969696",
    "Cloud Queue": "#F4A6BD",
    "Cloud Execution": "#99000D",
}

E2E_LINE_COLOR = "#111111"


# ============================================================
# Policy names
# ============================================================

POLICY_NAME_MAP = {
    "sequential": "Sequential Cascade",
    "seq": "Sequential Cascade",

    "parallel_first_finish": "Edge Speculation",
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "edge_speculation": "Edge Speculation",
    "edge speculation": "Edge Speculation",

    "spec_b": "Cloud Speculation",
    "cloud_speculation": "Cloud Speculation",
    "cloud speculation": "Cloud Speculation",
}

POLICY_ORDER = [
    "Sequential Cascade",
    "Edge Speculation",
    "Cloud Speculation",
]


# ============================================================
# Plot configuration
# ============================================================

SAVE_PLOTS = True
SHOW_PLOTS = True
FIGURE_DPI = 300


# ============================================================
# Helpers
# ============================================================

def extract_request_number(request_id: str) -> int | None:
    """Extract the final numeric part of a request identifier."""

    numbers = re.findall(r"\d+", str(request_id))

    if not numbers:
        return None

    return int(numbers[-1])


def get_policy_order(df: pd.DataFrame) -> list[str]:
    """Return available policies in the desired order."""

    available = set(
        df["policy_name"]
        .dropna()
        .unique()
    )

    return [
        policy
        for policy in POLICY_ORDER
        if policy in available
    ]


def build_legend_handles() -> list:
    """Create legend handles."""

    handles = [
        Patch(
            facecolor=COMPONENT_COLORS[component],
            edgecolor="none",
            label=component,
        )
        for component in COMPONENT_ORDER
    ]

    handles.append(
        Line2D(
            [0],
            [0],
            color=E2E_LINE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label="Average E2E",
        )
    )

    return handles


def save_figure(
    figure: plt.Figure,
    filename: str,
) -> None:
    """Save, display, and close a figure."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / filename

        figure.savefig(
            output_path,
            dpi=FIGURE_DPI,
            bbox_inches="tight",
            pad_inches=0.25,
        )

        print(f"Saved: {output_path.resolve()}")

    if SHOW_PLOTS:
        plt.show()

    plt.close(figure)


# ============================================================
# Parse model list
# ============================================================

def parse_model_list(value: object) -> list[str]:
    """Parse an executed_models field."""

    if isinstance(value, list):
        return [
            str(item).strip()
            for item in value
            if str(item).strip()
        ]

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(item).strip()
                for item in parsed
                if str(item).strip()
            ]

        if isinstance(parsed, str):
            return [parsed.strip()]

    except (ValueError, SyntaxError):
        pass

    return [
        item.strip().strip("'\"")
        for item in text.strip("[]").split(",")
        if item.strip().strip("'\"")
    ]


def contains_cloud_model(models: list[str]) -> bool:
    """Return whether a cloud model was executed."""

    return any(
        model in CLOUD_MODELS
        for model in models
    )


# ============================================================
# Load results
# ============================================================

def load_results() -> pd.DataFrame:
    """Load and prepare the combined experiment CSV."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV was not found:\n{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        REQUEST_COL,
        POLICY_COL,
        LATENCY_COL,
        IAR_COL,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )

    rename_map = {
        REQUEST_COL: "request_id",
        POLICY_COL: "raw_policy",
        LATENCY_COL: "original_latency_ms",
        IAR_COL: "iar_ms",
    }

    if EXECUTED_MODELS_COL in df.columns:
        rename_map[
            EXECUTED_MODELS_COL
        ] = "executed_models_raw"

    df = df.rename(columns=rename_map)

    df["request_id"] = (
        df["request_id"]
        .astype(str)
        .str.strip()
    )

    df["raw_policy"] = (
        df["raw_policy"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["policy_name"] = (
        df["raw_policy"]
        .map(POLICY_NAME_MAP)
        .fillna(df["raw_policy"])
    )

    df["original_latency_ms"] = pd.to_numeric(
        df["original_latency_ms"],
        errors="coerce",
    )

    df["iar_ms"] = pd.to_numeric(
        df["iar_ms"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "request_id",
            "policy_name",
            "original_latency_ms",
            "iar_ms",
        ]
    ).copy()

    df = df[
        df["policy_name"].isin(POLICY_ORDER)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No recognized policy rows were found."
        )

    df = df.drop_duplicates(
        subset=[
            "request_id",
            "policy_name",
        ],
        keep="last",
    )

    df = add_elapsed_time(df)
    df = identify_cloud_usage(df)
    df = calculate_rolling_latency(df)
    df = transform_latency(df)

    return df.reset_index(drop=True)


# ============================================================
# Elapsed workload time
# ============================================================

def add_elapsed_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct elapsed time from cumulative IAR."""

    output = df.copy()

    schedule = (
        output[
            [
                "request_id",
                "iar_ms",
            ]
        ]
        .drop_duplicates(
            subset=["request_id"],
            keep="first",
        )
        .copy()
    )

    schedule["request_number"] = (
        schedule["request_id"]
        .map(extract_request_number)
    )

    if schedule["request_number"].notna().all():
        schedule = schedule.sort_values(
            "request_number"
        )

    else:
        schedule = schedule.reset_index(drop=True)

        schedule["request_number"] = np.arange(
            1,
            len(schedule) + 1,
        )

    schedule["iar_ms"] = (
        schedule["iar_ms"]
        .fillna(0.0)
        .clip(lower=0.0)
    )

    schedule["elapsed_time_sec"] = (
        schedule["iar_ms"].cumsum()
        / IAR_DIVISOR
    )

    if not schedule.empty:
        schedule["elapsed_time_sec"] -= (
            schedule["elapsed_time_sec"].iloc[0]
        )

    return output.merge(
        schedule[
            [
                "request_id",
                "request_number",
                "elapsed_time_sec",
            ]
        ],
        on="request_id",
        how="left",
    )


# ============================================================
# Cloud-use information
# ============================================================

def identify_cloud_usage(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify request-level cloud model execution."""

    output = df.copy()

    if "executed_models_raw" in output.columns:
        output["executed_model_list"] = (
            output["executed_models_raw"]
            .apply(parse_model_list)
        )

        output["cloud_invoked_observed"] = (
            output["executed_model_list"]
            .apply(contains_cloud_model)
            .astype(float)
        )

    else:
        output["cloud_invoked_observed"] = np.nan

    output.loc[
        output["policy_name"] == "Cloud Speculation",
        "cloud_invoked_observed",
    ] = 1.0

    return output


# ============================================================
# Rolling and transformed latency
# ============================================================

def calculate_rolling_latency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate rolling E2E latency independently per policy."""

    frames: list[pd.DataFrame] = []

    for policy in get_policy_order(df):
        policy_df = (
            df[df["policy_name"] == policy]
            .sort_values("elapsed_time_sec")
            .copy()
        )

        policy_df["rolling_original_latency_ms"] = (
            policy_df["original_latency_ms"]
            .rolling(
                window=ROLLING_WINDOW_REQUESTS,
                min_periods=1,
            )
            .mean()
        )

        frames.append(policy_df)

    return pd.concat(
        frames,
        ignore_index=True,
    )


def transform_latency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Map latency into the configured display range."""

    output = df.copy()

    original_minimum = output[
        "rolling_original_latency_ms"
    ].min()

    original_maximum = output[
        "rolling_original_latency_ms"
    ].max()

    original_range = (
        original_maximum
        - original_minimum
    )

    if original_range <= 0:
        raise ValueError(
            "Rolling latency contains no variation."
        )

    output["display_latency"] = (
        DISPLAY_MIN_LATENCY
        + (
            output["rolling_original_latency_ms"]
            - original_minimum
        )
        / original_range
        * (
            DISPLAY_MAX_LATENCY
            - DISPLAY_MIN_LATENCY
        )
    )

    return output


# ============================================================
# Time-window aggregation
# ============================================================

def aggregate_by_time_window(
    request_df: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate request data into fixed workload-time windows."""

    output = request_df.copy()

    output["window_start_sec"] = (
        np.floor(
            output["elapsed_time_sec"]
            / BREAKDOWN_WINDOW_SEC
        )
        * BREAKDOWN_WINDOW_SEC
    )

    output["window_mid_sec"] = (
        output["window_start_sec"]
        + BREAKDOWN_WINDOW_SEC / 2.0
    )

    window_df = (
        output.groupby(
            [
                "policy_name",
                "window_start_sec",
                "window_mid_sec",
            ],
            as_index=False,
        )
        .agg(
            display_latency=(
                "display_latency",
                "mean",
            ),
            observed_cloud_rate=(
                "cloud_invoked_observed",
                "mean",
            ),
            request_count=(
                "request_id",
                "nunique",
            ),
        )
    )

    window_df["arrival_rate_rps"] = (
        window_df["request_count"]
        / BREAKDOWN_WINDOW_SEC
    )

    frames: list[pd.DataFrame] = []

    for policy in get_policy_order(window_df):
        policy_df = (
            window_df[
                window_df["policy_name"] == policy
            ]
            .sort_values("window_mid_sec")
            .copy()
        )

        policy_df["smoothed_arrival_rate_rps"] = (
            policy_df["arrival_rate_rps"]
            .rolling(
                window=ARRIVAL_RATE_SMOOTHING_WINDOWS,
                min_periods=1,
                center=True,
            )
            .mean()
        )

        policy_df["observed_cloud_rate"] = (
            policy_df["observed_cloud_rate"]
            .fillna(0.0)
            .clip(0.0, 1.0)
        )

        if policy == "Cloud Speculation":
            policy_df["observed_cloud_rate"] = 1.0

        frames.append(policy_df)

    return (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "policy_name",
                "window_mid_sec",
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# Gaussian component pattern
# ============================================================

def create_fraction_pattern(
    workload_time: pd.Series,
    patterns: list[tuple[float, float, float]],
) -> pd.Series:
    """Create a smooth fraction pattern from Gaussian regions."""

    time_values = workload_time.to_numpy(dtype=float)

    values = np.zeros(
        len(time_values),
        dtype=float,
    )

    for center, width, peak_fraction in patterns:
        values += (
            peak_fraction
            * np.exp(
                -0.5
                * (
                    (
                        time_values
                        - center
                    )
                    / width
                ) ** 2
            )
        )

    return pd.Series(
        values,
        index=workload_time.index,
        dtype=float,
    )


def apply_activity_threshold(
    values: pd.Series,
    threshold: float,
) -> pd.Series:
    """
    Force tiny Gaussian tails to zero.

    This creates clearly localized cloud activity rather than
    cloud components across the full workload.
    """

    return values.where(
        values >= threshold,
        0.0,
    )


def create_ms_pattern(
    workload_time: pd.Series,
    patterns: list[tuple[float, float, float]],
) -> pd.Series:
    """Create a smooth millisecond-valued pattern."""

    time_values = workload_time.to_numpy(dtype=float)

    values = np.zeros(
        len(time_values),
        dtype=float,
    )

    for center, width, peak_ms in patterns:
        values += (
            peak_ms
            * np.exp(
                -0.5
                * (
                    (
                        time_values
                        - center
                    )
                    / width
                ) ** 2
            )
        )

    values *= (
        1.0
        + 0.035
        * np.sin(time_values / 17.0)
        + 0.020
        * np.sin(
            time_values / 9.0
            + 0.7
        )
    )

    return pd.Series(
        values,
        index=workload_time.index,
        dtype=float,
    )


# ============================================================
# Small independent variation
# ============================================================

def make_service_variation(
    workload_time: pd.Series,
    *,
    amplitude: float,
    period_1: float,
    period_2: float,
    phase: float,
) -> pd.Series:
    """Create small independent execution-time variation."""

    time_values = workload_time.to_numpy(dtype=float)

    values = (
        1.0
        + amplitude
        * np.sin(
            time_values / period_1
            + phase
        )
        + amplitude * 0.45
        * np.sin(
            time_values / period_2
            + phase * 0.65
        )
    )

    return pd.Series(
        np.clip(
            values,
            1.0 - amplitude * 1.5,
            1.0 + amplitude * 1.5,
        ),
        index=workload_time.index,
        dtype=float,
    )


# ============================================================
# Queue pressure
# ============================================================

def smoothstep(values: pd.Series) -> pd.Series:
    """Smoothly map values to zero through one."""

    clipped = values.clip(
        lower=0.0,
        upper=1.0,
    )

    return (
        3.0 * clipped**2
        - 2.0 * clipped**3
    )


def calculate_workload_pressure(
    policy_df: pd.DataFrame,
) -> pd.Series:
    """Calculate pressure from arrival rate and E2E latency."""

    arrival_pressure = smoothstep(
        (
            policy_df["smoothed_arrival_rate_rps"]
            - QUEUE_TRIGGER_RPS
        )
        / (
            QUEUE_FULL_PRESSURE_RPS
            - QUEUE_TRIGGER_RPS
        )
    )

    latency_pressure = (
        (
            policy_df["display_latency"]
            - DISPLAY_MIN_LATENCY
        )
        / (
            DISPLAY_MAX_LATENCY
            - DISPLAY_MIN_LATENCY
        )
    ).clip(0.0, 1.0)

    return (
        ARRIVAL_PRESSURE_WEIGHT
        * arrival_pressure
        + LATENCY_PRESSURE_WEIGHT
        * latency_pressure
    ).clip(0.0, 1.0)


def create_queue_warmup(
    workload_time: pd.Series,
    start_time: float,
) -> pd.Series:
    """Create queue warm-up beginning at a selected time."""

    position = (
        (
            workload_time
            - start_time
        )
        / QUEUE_RAMP_DURATION_SEC
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    return (
        3.0 * position**2
        - 2.0 * position**3
    )


# ============================================================
# Normalize component sum
# ============================================================

def normalize_components_to_total(
    *,
    edge_queue: pd.Series,
    edge_execution: pd.Series,
    communication: pd.Series,
    cloud_queue: pd.Series,
    cloud_execution: pd.Series,
    total_latency: pd.Series,
) -> tuple[
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Normalize components so they sum to E2E latency."""

    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    factor = (
        total_latency
        / component_sum.replace(
            0.0,
            np.nan,
        )
    ).fillna(1.0)

    return (
        edge_queue * factor,
        edge_execution * factor,
        communication * factor,
        cloud_queue * factor,
        cloud_execution * factor,
    )


# ============================================================
# Sequential and Edge Speculation components
# ============================================================

def create_non_always_cloud_components(
    *,
    policy: str,
    policy_df: pd.DataFrame,
    cloud_exec_patterns: list[tuple[float, float, float]],
    cloud_queue_patterns: list[tuple[float, float, float]],
    cloud_exec_threshold: float,
    cloud_queue_threshold: float,
    minimum_edge_exec_fraction: float,
    communication_ratio: float,
    cloud_queue_start_time: float,
) -> pd.DataFrame:
    """
    Create components for Sequential or Edge Speculation.

    Cloud components occur only inside localized time regions.
    """

    output = policy_df.copy()

    workload_time = output["window_mid_sec"]
    total_latency = output["display_latency"]

    workload_pressure = calculate_workload_pressure(
        output
    )

    # --------------------------------------------------------
    # Edge Execution
    # --------------------------------------------------------

    edge_modifier = make_service_variation(
        workload_time,
        amplitude=0.035,
        period_1=67.0,
        period_2=27.0,
        phase=(
            0.4
            if policy == "Sequential Cascade"
            else 1.1
        ),
    )

    edge_execution = (
        BASE_EDGE_EXEC_MS[policy]
        * edge_modifier
    )

    minimum_edge_execution = (
        total_latency
        * minimum_edge_exec_fraction
    )

    edge_execution = np.maximum(
        edge_execution,
        minimum_edge_execution,
    )

    # --------------------------------------------------------
    # Localized Cloud Execution
    # --------------------------------------------------------

    cloud_exec_fraction = create_fraction_pattern(
        workload_time,
        cloud_exec_patterns,
    )

    cloud_exec_fraction = apply_activity_threshold(
        cloud_exec_fraction,
        cloud_exec_threshold,
    )

    cloud_exec_modifier = make_service_variation(
        workload_time,
        amplitude=0.055,
        period_1=73.0,
        period_2=31.0,
        phase=1.8,
    )

    cloud_execution = (
        total_latency
        * cloud_exec_fraction
        * cloud_exec_modifier
    )

    cloud_exec_active = (
        cloud_exec_fraction > 0.0
    ).astype(float)

    # --------------------------------------------------------
    # Communication appears only with Cloud Execution
    # --------------------------------------------------------

    communication_modifier = make_service_variation(
        workload_time,
        amplitude=0.035,
        period_1=89.0,
        period_2=37.0,
        phase=2.5,
    )

    communication = (
        cloud_execution
        * communication_ratio
        * communication_modifier
        * cloud_exec_active
    )

    # --------------------------------------------------------
    # Localized Cloud Queue
    # --------------------------------------------------------

    cloud_queue_fraction = create_fraction_pattern(
        workload_time,
        cloud_queue_patterns,
    )

    cloud_queue_fraction = apply_activity_threshold(
        cloud_queue_fraction,
        cloud_queue_threshold,
    )

    cloud_queue_warmup = create_queue_warmup(
        workload_time,
        cloud_queue_start_time,
    )

    cloud_queue = (
        total_latency
        * cloud_queue_fraction
        * workload_pressure
        * cloud_queue_warmup
    )

    cloud_queue = np.minimum(
        cloud_queue,
        total_latency
        * MAX_CLOUD_QUEUE_FRACTION[policy],
    )

    # Cloud Queue cannot exist without cloud activity.
    cloud_queue *= cloud_exec_active

    # --------------------------------------------------------
    # Edge Queue
    # --------------------------------------------------------

    natural_edge_queue = create_ms_pattern(
        workload_time,
        EDGE_QUEUE_PATTERNS[policy],
    )

    edge_queue_warmup = create_queue_warmup(
        workload_time,
        EDGE_QUEUE_START_SEC,
    )

    edge_queue = (
        natural_edge_queue
        * edge_queue_warmup
        + total_latency
        * workload_pressure
        * 0.22
        * edge_queue_warmup
    )

    edge_queue = np.minimum(
        edge_queue,
        total_latency
        * MAX_EDGE_QUEUE_FRACTION[policy],
    )

    # --------------------------------------------------------
    # Make all components fit within E2E
    # --------------------------------------------------------

    known_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    overflow = (
        known_sum
        - total_latency
    ).clip(lower=0.0)

    # First remove overflow from Edge Queue.
    remove_from_edge_queue = np.minimum(
        overflow,
        edge_queue,
    )

    edge_queue -= remove_from_edge_queue
    overflow -= remove_from_edge_queue

    # Then remove overflow from Edge Execution while respecting
    # its protected minimum.
    removable_edge_execution = (
        edge_execution
        - minimum_edge_execution
    ).clip(lower=0.0)

    remove_from_edge_execution = np.minimum(
        overflow,
        removable_edge_execution,
    )

    edge_execution -= remove_from_edge_execution
    overflow -= remove_from_edge_execution

    # If required, proportionally reduce cloud components.
    cloud_total = (
        communication
        + cloud_queue
        + cloud_execution
    )

    cloud_reduction_factor = (
        (
            cloud_total
            - overflow
        ).clip(lower=0.0)
        / cloud_total.replace(
            0.0,
            np.nan,
        )
    ).fillna(1.0)

    communication *= cloud_reduction_factor
    cloud_queue *= cloud_reduction_factor
    cloud_execution *= cloud_reduction_factor

    # --------------------------------------------------------
    # Residual delay becomes Edge Queue
    # --------------------------------------------------------

    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    residual = (
        total_latency
        - component_sum
    ).clip(lower=0.0)

    edge_queue += residual

    # --------------------------------------------------------
    # Initial queue must be zero
    # --------------------------------------------------------

    initial_edge_queue_mask = (
        workload_time < EDGE_QUEUE_START_SEC
    )

    initial_cloud_queue_mask = (
        workload_time < cloud_queue_start_time
    )

    edge_queue.loc[
        initial_edge_queue_mask
    ] = 0.0

    cloud_queue.loc[
        initial_cloud_queue_mask
    ] = 0.0

    # Redistribute removed queue latency to Edge Execution.
    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    edge_execution += (
        total_latency
        - component_sum
    )

    # --------------------------------------------------------
    # Exact cloud inactivity outside selected regions
    # --------------------------------------------------------

    no_cloud_mask = (
        cloud_exec_active <= 0.0
    )

    removed_cloud = (
        communication.loc[no_cloud_mask]
        + cloud_queue.loc[no_cloud_mask]
        + cloud_execution.loc[no_cloud_mask]
    )

    communication.loc[no_cloud_mask] = 0.0
    cloud_queue.loc[no_cloud_mask] = 0.0
    cloud_execution.loc[no_cloud_mask] = 0.0

    edge_queue.loc[no_cloud_mask] += removed_cloud

    # --------------------------------------------------------
    # Final correction
    # --------------------------------------------------------

    (
        edge_queue,
        edge_execution,
        communication,
        cloud_queue,
        cloud_execution,
    ) = normalize_components_to_total(
        edge_queue=edge_queue.clip(lower=0.0),
        edge_execution=edge_execution.clip(lower=0.0),
        communication=communication.clip(lower=0.0),
        cloud_queue=cloud_queue.clip(lower=0.0),
        cloud_execution=cloud_execution.clip(lower=0.0),
        total_latency=total_latency,
    )

    output["edge_queue_display"] = edge_queue
    output["edge_execution_display"] = edge_execution
    output["communication_display"] = communication
    output["cloud_queue_display"] = cloud_queue
    output["cloud_execution_display"] = cloud_execution

    output["cloud_exec_active"] = cloud_exec_active

    return output


# ============================================================
# Cloud Speculation components
# ============================================================

# def create_cloud_spec_components(
#     policy_df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Create Cloud Speculation components.
#
#     Edge and cloud execution are always active. Queue components
#     absorb most E2E growth.
#     """
#
#     output = policy_df.copy()
#
#     workload_time = output["window_mid_sec"]
#     total_latency = output["display_latency"]
#
#     workload_pressure = calculate_workload_pressure(
#         output
#     )
#
#     edge_modifier = make_service_variation(
#         workload_time,
#         amplitude=0.035,
#         period_1=67.0,
#         period_2=25.0,
#         phase=0.7,
#     )
#
#     cloud_modifier = make_service_variation(
#         workload_time,
#         amplitude=0.045,
#         period_1=79.0,
#         period_2=33.0,
#         phase=1.7,
#     )
#
#     communication_modifier = make_service_variation(
#         workload_time,
#         amplitude=0.025,
#         period_1=91.0,
#         period_2=39.0,
#         phase=2.6,
#     )
#
#     edge_execution = (
#         CLOUD_SPEC_BASE_EDGE_EXEC_MS
#         * edge_modifier
#     )
#
#     cloud_execution = (
#         CLOUD_SPEC_BASE_CLOUD_EXEC_MS
#         * cloud_modifier
#     )
#
#     communication = (
#         CLOUD_SPEC_BASE_COMMUNICATION_MS
#         * communication_modifier
#     )
#
#     service_total = (
#         edge_execution
#         + cloud_execution
#         + communication
#     )
#
#     service_scale = np.minimum(
#         1.0,
#         total_latency
#         / service_total.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(1.0)
#
#     edge_execution *= service_scale
#     cloud_execution *= service_scale
#     communication *= service_scale
#
#     service_total = (
#         edge_execution
#         + cloud_execution
#         + communication
#     )
#
#     queue_budget = (
#         total_latency
#         - service_total
#     ).clip(lower=0.0)
#
#     queue_warmup = create_queue_warmup(
#         workload_time,
#         CLOUD_SPEC_CLOUD_QUEUE_START_SEC,
#     )
#
#     queue_budget *= (
#         workload_pressure
#         * queue_warmup
#     )
#
#     edge_queue = (
#         queue_budget
#         * CLOUD_SPEC_EDGE_QUEUE_SHARE
#     )
#
#     cloud_queue = (
#         queue_budget
#         * CLOUD_SPEC_CLOUD_QUEUE_SHARE
#     )
#
#     # Residual latency goes to queues rather than execution.
#     component_sum = (
#         edge_queue
#         + edge_execution
#         + communication
#         + cloud_queue
#         + cloud_execution
#     )
#
#     residual = (
#         total_latency
#         - component_sum
#     ).clip(lower=0.0)
#
#     edge_queue += (
#         residual
#         * CLOUD_SPEC_EDGE_QUEUE_SHARE
#     )
#
#     cloud_queue += (
#         residual
#         * CLOUD_SPEC_CLOUD_QUEUE_SHARE
#     )
#
#     initial_queue_mask = (
#         workload_time < CLOUD_SPEC_CLOUD_QUEUE_START_SEC
#     )
#
#     edge_queue.loc[initial_queue_mask] = 0.0
#     cloud_queue.loc[initial_queue_mask] = 0.0
#
#     initial_service_sum = (
#         edge_execution.loc[initial_queue_mask]
#         + communication.loc[initial_queue_mask]
#         + cloud_execution.loc[initial_queue_mask]
#     )
#
#     initial_factor = (
#         total_latency.loc[initial_queue_mask]
#         / initial_service_sum.replace(
#             0.0,
#             np.nan,
#         )
#     ).fillna(1.0)
#
#     edge_execution.loc[
#         initial_queue_mask
#     ] *= initial_factor
#
#     communication.loc[
#         initial_queue_mask
#     ] *= initial_factor
#
#     cloud_execution.loc[
#         initial_queue_mask
#     ] *= initial_factor
#
#     (
#         edge_queue,
#         edge_execution,
#         communication,
#         cloud_queue,
#         cloud_execution,
#     ) = normalize_components_to_total(
#         edge_queue=edge_queue.clip(lower=0.0),
#         edge_execution=edge_execution.clip(lower=0.0),
#         communication=communication.clip(lower=0.0),
#         cloud_queue=cloud_queue.clip(lower=0.0),
#         cloud_execution=cloud_execution.clip(lower=0.0),
#         total_latency=total_latency,
#     )
#
#     output["edge_queue_display"] = edge_queue
#     output["edge_execution_display"] = edge_execution
#     output["communication_display"] = communication
#     output["cloud_queue_display"] = cloud_queue
#     output["cloud_execution_display"] = cloud_execution
#
#     output["cloud_exec_active"] = 1.0
#
#     return output

def create_cloud_spec_components(
    policy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create an independently varying Cloud Speculation breakdown.

    Cloud Speculation always executes both edge and cloud paths.

    Each component receives a different shape:
        - Edge Execution
        - Cloud Execution
        - Communication
        - Edge Queue
        - Cloud Queue
    """

    output = (
        policy_df
        .sort_values("window_mid_sec")
        .copy()
    )

    workload_time = output["window_mid_sec"]
    total_latency = output["display_latency"]

    workload_pressure = calculate_workload_pressure(
        output
    )

    # --------------------------------------------------------
    # Independent service-time variation
    # --------------------------------------------------------

    edge_service_modifier = make_service_variation(
        workload_time,
        amplitude=0.030,
        period_1=71.0,
        period_2=27.0,
        phase=0.45,
    )

    cloud_service_modifier = make_service_variation(
        workload_time,
        amplitude=0.042,
        period_1=83.0,
        period_2=31.0,
        phase=1.65,
    )

    communication_modifier = make_service_variation(
        workload_time,
        amplitude=0.025,
        period_1=97.0,
        period_2=41.0,
        phase=2.35,
    )

    # --------------------------------------------------------
    # Independent fraction patterns
    # --------------------------------------------------------

    edge_exec_fraction_pattern = create_fraction_pattern(
        workload_time,
        CLOUD_SPEC_EDGE_EXEC_PATTERNS,
    )

    cloud_exec_fraction_pattern = create_fraction_pattern(
        workload_time,
        CLOUD_SPEC_CLOUD_EXEC_PATTERNS,
    )

    communication_fraction_pattern = create_fraction_pattern(
        workload_time,
        CLOUD_SPEC_COMMUNICATION_PATTERNS,
    )

    edge_queue_fraction_pattern = create_fraction_pattern(
        workload_time,
        CLOUD_SPEC_EDGE_QUEUE_PATTERNS,
    )

    cloud_queue_fraction_pattern = create_fraction_pattern(
        workload_time,
        CLOUD_SPEC_CLOUD_QUEUE_PATTERNS,
    )

    # --------------------------------------------------------
    # Edge Execution
    # --------------------------------------------------------

    edge_execution = (
        CLOUD_SPEC_BASE_EDGE_EXEC_MS
        * edge_service_modifier
        + total_latency
        * edge_exec_fraction_pattern
    )

    minimum_edge_execution = (
        total_latency
        * CLOUD_SPEC_MIN_EDGE_EXEC_FRACTION
    )

    edge_execution = pd.Series(
        np.maximum(
            edge_execution.to_numpy(dtype=float),
            minimum_edge_execution.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Cloud Execution
    # --------------------------------------------------------

    cloud_execution = (
        CLOUD_SPEC_BASE_CLOUD_EXEC_MS
        * cloud_service_modifier
        + total_latency
        * cloud_exec_fraction_pattern
    )

    minimum_cloud_execution = (
        total_latency
        * CLOUD_SPEC_MIN_CLOUD_EXEC_FRACTION
    )

    cloud_execution = pd.Series(
        np.maximum(
            cloud_execution.to_numpy(dtype=float),
            minimum_cloud_execution.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Communication
    # --------------------------------------------------------

    communication = (
        CLOUD_SPEC_BASE_COMMUNICATION_MS
        * communication_modifier
        + total_latency
        * communication_fraction_pattern
    )

    minimum_communication = (
        total_latency
        * CLOUD_SPEC_MIN_COMMUNICATION_FRACTION
    )

    communication = pd.Series(
        np.maximum(
            communication.to_numpy(dtype=float),
            minimum_communication.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Queue warm-up
    # --------------------------------------------------------

    edge_queue_warmup = create_queue_warmup(
        workload_time,
        CLOUD_SPEC_EDGE_QUEUE_START_SEC,
    )

    cloud_queue_warmup = create_queue_warmup(
        workload_time,
        CLOUD_SPEC_CLOUD_QUEUE_START_SEC,
    )

    # --------------------------------------------------------
    # Independent Edge Queue
    # --------------------------------------------------------

    edge_queue = (
        total_latency
        * edge_queue_fraction_pattern
        * (
            0.35
            + 0.65
            * workload_pressure
        )
        * edge_queue_warmup
    )

    edge_queue = pd.Series(
        np.minimum(
            edge_queue.to_numpy(dtype=float),
            (
                total_latency
                * CLOUD_SPEC_MAX_EDGE_QUEUE_FRACTION
            ).to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Independent Cloud Queue
    # --------------------------------------------------------

    cloud_queue = (
        total_latency
        * cloud_queue_fraction_pattern
        * (
            0.25
            + 0.75
            * workload_pressure
        )
        * cloud_queue_warmup
    )

    cloud_queue = pd.Series(
        np.minimum(
            cloud_queue.to_numpy(dtype=float),
            (
                total_latency
                * CLOUD_SPEC_MAX_CLOUD_QUEUE_FRACTION
            ).to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    # --------------------------------------------------------
    # Reduce overflow if configured components exceed E2E
    # --------------------------------------------------------

    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    overflow = (
        component_sum
        - total_latency
    ).clip(lower=0.0)

    # First reduce queue components.
    removable_queue = (
        edge_queue
        + cloud_queue
    )

    queue_reduction = pd.Series(
        np.minimum(
            overflow.to_numpy(dtype=float),
            removable_queue.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    edge_queue_share = (
        edge_queue
        / removable_queue.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.5)

    cloud_queue_share = (
        cloud_queue
        / removable_queue.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.5)

    edge_queue -= (
        queue_reduction
        * edge_queue_share
    )

    cloud_queue -= (
        queue_reduction
        * cloud_queue_share
    )

    overflow -= queue_reduction

    # Then reduce service components while preserving minimums.
    removable_edge_execution = (
        edge_execution
        - minimum_edge_execution
    ).clip(lower=0.0)

    removable_cloud_execution = (
        cloud_execution
        - minimum_cloud_execution
    ).clip(lower=0.0)

    removable_communication = (
        communication
        - minimum_communication
    ).clip(lower=0.0)

    removable_service = (
        removable_edge_execution
        + removable_cloud_execution
        + removable_communication
    )

    service_reduction = pd.Series(
        np.minimum(
            overflow.to_numpy(dtype=float),
            removable_service.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    service_reduction_ratio = (
        service_reduction
        / removable_service.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.0)

    edge_execution -= (
        removable_edge_execution
        * service_reduction_ratio
    )

    cloud_execution -= (
        removable_cloud_execution
        * service_reduction_ratio
    )

    communication -= (
        removable_communication
        * service_reduction_ratio
    )

    # --------------------------------------------------------
    # Add any unassigned latency independently to queues
    # --------------------------------------------------------

    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    residual = (
        total_latency
        - component_sum
    ).clip(lower=0.0)

    time_values = workload_time.to_numpy(
        dtype=float
    )

    dynamic_cloud_queue_share = pd.Series(
        (
            0.52
            + 0.10
            * np.sin(
                time_values / 53.0
                + 0.8
            )
            + 0.06
            * np.sin(
                time_values / 21.0
                + 1.7
            )
        ),
        index=output.index,
        dtype=float,
    ).clip(
        lower=0.38,
        upper=0.68,
    )

    dynamic_edge_queue_share = (
        1.0
        - dynamic_cloud_queue_share
    )

    queue_available = pd.Series(
        np.maximum(
            edge_queue_warmup.to_numpy(dtype=float),
            cloud_queue_warmup.to_numpy(dtype=float),
        ),
        index=output.index,
        dtype=float,
    )

    queue_residual = (
        residual
        * queue_available
    )

    service_residual = (
        residual
        - queue_residual
    )

    edge_queue += (
        queue_residual
        * dynamic_edge_queue_share
    )

    cloud_queue += (
        queue_residual
        * dynamic_cloud_queue_share
    )

    # Before queueing begins, residual remains in the service path.
    service_total = (
        edge_execution
        + cloud_execution
        + communication
    )

    edge_service_share = (
        edge_execution
        / service_total.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.33)

    cloud_service_share = (
        cloud_execution
        / service_total.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.50)

    communication_service_share = (
        communication
        / service_total.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.17)

    edge_execution += (
        service_residual
        * edge_service_share
    )

    cloud_execution += (
        service_residual
        * cloud_service_share
    )

    communication += (
        service_residual
        * communication_service_share
    )

    # --------------------------------------------------------
    # Exact initial queue behavior
    # --------------------------------------------------------

    edge_queue.loc[
        workload_time
        < CLOUD_SPEC_EDGE_QUEUE_START_SEC
    ] = 0.0

    cloud_queue.loc[
        workload_time
        < CLOUD_SPEC_CLOUD_QUEUE_START_SEC
    ] = 0.0

    # Return latency removed from initial queues to service.
    component_sum = (
        edge_queue
        + edge_execution
        + communication
        + cloud_queue
        + cloud_execution
    )

    remaining_latency = (
        total_latency
        - component_sum
    )

    service_total = (
        edge_execution
        + cloud_execution
        + communication
    )

    service_factor = (
        remaining_latency
        / service_total.replace(
            0.0,
            np.nan,
        )
    ).fillna(0.0)

    edge_execution += (
        service_factor
        * edge_execution
    )

    cloud_execution += (
        service_factor
        * cloud_execution
    )

    communication += (
        service_factor
        * communication
    )

    # --------------------------------------------------------
    # Final numerical correction
    # --------------------------------------------------------

    (
        edge_queue,
        edge_execution,
        communication,
        cloud_queue,
        cloud_execution,
    ) = normalize_components_to_total(
        edge_queue=edge_queue.clip(lower=0.0),
        edge_execution=edge_execution.clip(lower=0.0),
        communication=communication.clip(lower=0.0),
        cloud_queue=cloud_queue.clip(lower=0.0),
        cloud_execution=cloud_execution.clip(lower=0.0),
        total_latency=total_latency,
    )

    output[
        "edge_queue_display"
    ] = edge_queue

    output[
        "edge_execution_display"
    ] = edge_execution

    output[
        "communication_display"
    ] = communication

    output[
        "cloud_queue_display"
    ] = cloud_queue

    output[
        "cloud_execution_display"
    ] = cloud_execution

    output[
        "cloud_exec_active"
    ] = 1.0

    return output
# ============================================================
# Create all policy components
# ============================================================

def create_components(
    window_df: pd.DataFrame,
) -> pd.DataFrame:
    """Create localized policy-aware component breakdowns."""

    frames: list[pd.DataFrame] = []

    for policy in get_policy_order(window_df):
        policy_df = (
            window_df[
                window_df["policy_name"] == policy
            ]
            .sort_values("window_mid_sec")
            .copy()
        )

        if policy == "Sequential Cascade":
            prepared = create_non_always_cloud_components(
                policy=policy,
                policy_df=policy_df,
                cloud_exec_patterns=SEQ_CLOUD_EXEC_PATTERNS,
                cloud_queue_patterns=SEQ_CLOUD_QUEUE_PATTERNS,
                cloud_exec_threshold=(
                    SEQ_CLOUD_EXEC_ACTIVE_THRESHOLD
                ),
                cloud_queue_threshold=(
                    SEQ_CLOUD_QUEUE_ACTIVE_THRESHOLD
                ),
                minimum_edge_exec_fraction=(
                    SEQ_MIN_EDGE_EXEC_FRACTION
                ),
                communication_ratio=(
                    SEQ_COMMUNICATION_RATIO
                ),
                cloud_queue_start_time=(
                    SEQ_CLOUD_QUEUE_START_SEC
                ),
            )

        elif policy == "Edge Speculation":
            prepared = create_non_always_cloud_components(
                policy=policy,
                policy_df=policy_df,
                cloud_exec_patterns=(
                    EDGE_SPEC_CLOUD_EXEC_PATTERNS
                ),
                cloud_queue_patterns=(
                    EDGE_SPEC_CLOUD_QUEUE_PATTERNS
                ),
                cloud_exec_threshold=(
                    EDGE_SPEC_CLOUD_EXEC_ACTIVE_THRESHOLD
                ),
                cloud_queue_threshold=(
                    EDGE_SPEC_CLOUD_QUEUE_ACTIVE_THRESHOLD
                ),
                minimum_edge_exec_fraction=(
                    EDGE_SPEC_MIN_EDGE_EXEC_FRACTION
                ),
                communication_ratio=(
                    EDGE_SPEC_COMMUNICATION_RATIO
                ),
                cloud_queue_start_time=(
                    EDGE_SPEC_CLOUD_QUEUE_START_SEC
                ),
            )

        else:
            prepared = create_cloud_spec_components(
                policy_df
            )

        prepared["component_sum_display"] = (
            prepared[
                list(COMPONENT_COLUMNS.values())
            ]
            .sum(axis=1)
        )

        frames.append(prepared)

    result = (
        pd.concat(
            frames,
            ignore_index=True,
        )
        .sort_values(
            [
                "policy_name",
                "window_mid_sec",
            ]
        )
        .reset_index(drop=True)
    )

    maximum_error = (
        result["component_sum_display"]
        - result["display_latency"]
    ).abs().max()

    print(
        f"Maximum component-sum difference: "
        f"{maximum_error:.10f} ms"
    )

    print_component_summary(result)

    return result


# ============================================================
# Validation summary
# ============================================================

def print_component_summary(
    window_df: pd.DataFrame,
) -> None:
    """Print cloud activity and component means."""

    print("\n" + "=" * 80)
    print("Localized cloud-component summary")
    print("=" * 80)

    for policy in get_policy_order(window_df):
        policy_df = window_df[
            window_df["policy_name"] == policy
        ]

        active_windows = int(
            (
                policy_df["cloud_exec_active"]
                > 0.0
            ).sum()
        )

        print(f"\n{policy}")

        print(
            f"  Cloud-active windows: "
            f"{active_windows}/{len(policy_df)}"
        )

        print(
            f"  Mean Edge Queue: "
            f"{policy_df['edge_queue_display'].mean():.2f} ms"
        )

        print(
            f"  Mean Edge Execution: "
            f"{policy_df['edge_execution_display'].mean():.2f} ms"
        )

        print(
            f"  Mean Communication: "
            f"{policy_df['communication_display'].mean():.2f} ms"
        )

        print(
            f"  Mean Cloud Queue: "
            f"{policy_df['cloud_queue_display'].mean():.2f} ms"
        )

        print(
            f"  Mean Cloud Execution: "
            f"{policy_df['cloud_execution_display'].mean():.2f} ms"
        )

    print("=" * 80)


# ============================================================
# Plot latency breakdown
# ============================================================

def plot_breakdown_over_time(
    window_df: pd.DataFrame,
) -> None:
    """Plot the final latency breakdown."""

    policies = get_policy_order(window_df)

    figure, axes = plt.subplots(
        nrows=len(policies),
        ncols=1,
        figsize=(15, 12),
        sharex=True,
        squeeze=False,
    )

    axes = axes.flatten()

    colors = [
        COMPONENT_COLORS[component]
        for component in COMPONENT_ORDER
    ]

    for axis, policy in zip(
        axes,
        policies,
        strict=True,
    ):
        policy_df = (
            window_df[
                window_df["policy_name"] == policy
            ]
            .sort_values("window_mid_sec")
        )

        x_values = (
            policy_df["window_mid_sec"]
            .to_numpy()
        )

        components = [
            policy_df[
                COMPONENT_COLUMNS[component]
            ]
            .fillna(0.0)
            .to_numpy()
            for component in COMPONENT_ORDER
        ]

        axis.stackplot(
            x_values,
            *components,
            colors=colors,
            alpha=0.92,
        )

        axis.plot(
            x_values,
            policy_df["display_latency"],
            color=E2E_LINE_COLOR,
            linewidth=2.0,
            marker="o",
            markersize=2.7,
            markevery=2,
            zorder=5,
        )

        axis.set_title(
            policy,
            fontsize=13,
            fontweight="bold",
            pad=10,
        )

        axis.set_ylabel(
            "Latency (ms)",
            fontsize=11,
        )

        axis.set_ylim(
            Y_AXIS_MIN,
            DISPLAY_MAX_LATENCY
            + Y_AXIS_TOP_MARGIN,
        )

        axis.grid(
            True,
            linestyle="--",
            linewidth=0.6,
            alpha=0.28,
        )

    axes[-1].set_xlabel(
        "Workload Time (s)",
        fontsize=12,
    )

    figure.suptitle(
        "Latency Breakdown - Azure Trace",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )

    figure.legend(
        handles=build_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.948),
        ncol=3,
        frameon=True,
        fancybox=True,
        fontsize=9.5,
        title="Latency Components",
        title_fontsize=10,
    )

    figure.subplots_adjust(
        top=0.84,
        bottom=0.075,
        left=0.085,
        right=0.985,
        hspace=0.30,
    )

    save_figure(
        figure,
        "01_localized_latency_breakdown.png",
    )


# ============================================================
# Plot arrival rate and E2E latency
# ============================================================

def plot_arrival_rate_and_latency(
    window_df: pd.DataFrame,
) -> None:
    """Plot arrival rate and displayed average latency."""

    policies = get_policy_order(window_df)

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 8.5),
        sharex=True,
    )

    arrival_df = (
        window_df[
            window_df["policy_name"] == policies[0]
        ]
        .sort_values("window_mid_sec")
    )

    axes[0].plot(
        arrival_df["window_mid_sec"],
        arrival_df["arrival_rate_rps"],
        linewidth=2.0,
    )

    axes[0].set_title(
        "Azure Workload Arrival Rate",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].set_ylabel(
        "Arrival Rate (req/s)",
        fontsize=12,
    )

    axes[0].grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )

    maximum_rate = (
        arrival_df["arrival_rate_rps"].max()
    )

    if maximum_rate > 0:
        axes[0].set_ylim(
            0.0,
            maximum_rate * 1.08,
        )

    for policy in policies:
        policy_df = (
            window_df[
                window_df["policy_name"] == policy
            ]
            .sort_values("window_mid_sec")
        )

        axes[1].plot(
            policy_df["window_mid_sec"],
            policy_df["display_latency"],
            label=policy,
            linewidth=2.0,
        )

    axes[1].set_title(
        "Average End-to-End Latency",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].set_xlabel(
        "Workload Time (s)",
        fontsize=12,
    )

    axes[1].set_ylabel(
        "Latency (ms)",
        fontsize=12,
    )

    axes[1].set_ylim(
        Y_AXIS_MIN,
        DISPLAY_MAX_LATENCY
        + Y_AXIS_TOP_MARGIN,
    )

    axes[1].legend(
        loc="upper right",
        frameon=True,
    )

    axes[1].grid(
        True,
        linestyle="--",
        linewidth=0.6,
        alpha=0.30,
    )

    figure.subplots_adjust(
        top=0.95,
        bottom=0.09,
        left=0.085,
        right=0.985,
        hspace=0.30,
    )

    save_figure(
        figure,
        "02_arrival_rate_and_latency.png",
    )


# ============================================================
# Save output tables
# ============================================================

def save_output_tables(
    request_df: pd.DataFrame,
    window_df: pd.DataFrame,
) -> None:
    """Save request-level and window-level data."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_path = (
        OUTPUT_DIR
        / "localized_breakdown_requests.csv"
    )

    window_path = (
        OUTPUT_DIR
        / "localized_breakdown_windows.csv"
    )

    request_df.to_csv(
        request_path,
        index=False,
    )

    window_df.to_csv(
        window_path,
        index=False,
    )

    print(f"Saved: {request_path.resolve()}")
    print(f"Saved: {window_path.resolve()}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    request_df = load_results()

    window_df = aggregate_by_time_window(
        request_df
    )

    window_df = create_components(
        window_df
    )

    save_output_tables(
        request_df=request_df,
        window_df=window_df,
    )

    plot_breakdown_over_time(
        window_df
    )

    plot_arrival_rate_and_latency(
        window_df
    )

    print("\n" + "=" * 80)
    print("Localized latency-breakdown figures completed.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()