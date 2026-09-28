# from __future__ import annotations
#
# from pathlib import Path
# import re
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# INPUT_FILE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/azure_workload_4IAT_Final.csv")
# OUTPUT_DIR = Path("figures")
#
# REQUEST_COL = "ID"
# POLICY_COL = "effective_cascade_mode"
# LATENCY_COL = "e2e_time_ms"
# IAR_COL = "IAR"
#
# # IAR is assumed to be milliseconds.
# IAR_DIVISOR = 1000.0
#
# # Rolling mean window in requests.
# ROLLING_WINDOW = 30
#
# # Scale the maximum latency to this value.
# MAX_LATENCY = 600
#
# SAVE_PLOTS = True
# SHOW_PLOTS = True
#
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential",
#     "seq": "Sequential",
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
#     "Sequential",
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
#     """Extract the last numeric part of the request ID."""
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
#     """Return available policies in the preferred order."""
#
#     available = set(
#         df["policy_name"].dropna().unique()
#     )
#
#     return [
#         policy
#         for policy in POLICY_ORDER
#         if policy in available
#     ]
#
#
# def save_and_show(filename: str) -> None:
#     """Save, display, and close the current figure."""
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     plt.grid(
#         True,
#         linestyle="--",
#         linewidth=0.7,
#         alpha=0.30,
#     )
#
#     plt.tight_layout()
#
#     if SAVE_PLOTS:
#         output_path = OUTPUT_DIR / filename
#
#         plt.savefig(
#             output_path,
#             dpi=300,
#             bbox_inches="tight",
#         )
#
#         print(f"Saved: {output_path.resolve()}")
#
#     if SHOW_PLOTS:
#         plt.show()
#
#     plt.close()
#
#
# # ============================================================
# # Load and prepare data
# # ============================================================
#
# def load_results() -> pd.DataFrame:
#     """Load the CSV and prepare policy names and workload time."""
#
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"CSV file not found:\n{INPUT_FILE.resolve()}"
#         )
#
#     df = pd.read_csv(INPUT_FILE)
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
#             f"Missing columns: {missing_columns}\n"
#             f"Available columns: {list(df.columns)}"
#         )
#
#     df = df.rename(
#         columns={
#             REQUEST_COL: "request_id",
#             POLICY_COL: "raw_policy",
#             LATENCY_COL: "latency_ms",
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
#     df["latency_ms"] = pd.to_numeric(
#         df["latency_ms"],
#         errors="coerce",
#     )
#
#     df["iar_ms"] = pd.to_numeric(
#         df["iar_ms"],
#         errors="coerce",
#     )
#
#     df = df.dropna(
#         subset=[
#             "request_id",
#             "policy_name",
#             "latency_ms",
#             "iar_ms",
#         ]
#     ).copy()
#
#     df = df[
#         df["policy_name"].isin(POLICY_ORDER)
#     ].copy()
#
#     if df.empty:
#         raise RuntimeError(
#             "No recognised policy rows were found."
#         )
#
#     df = df.drop_duplicates(
#         subset=[
#             "request_id",
#             "policy_name",
#         ],
#         keep="last",
#     )
#
#     df = add_elapsed_time(df)
#     df = add_scaled_latency(df)
#
#     return df.reset_index(drop=True)
#
#
# # ============================================================
# # Reconstruct workload time
# # ============================================================
#
# def add_elapsed_time(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Reconstruct elapsed workload time from cumulative IAR.
#
#     IAR is counted once per unique request because every request
#     appears once for each policy.
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
#     else:
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
#     # Make the first request start at zero seconds.
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
# # Scale latency while preserving the original shape
# # ============================================================
#
# def add_scaled_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Scale all latency values using one common factor.
#
#     This preserves:
#         - the shape of every curve
#         - relative differences between policies
#         - the locations of peaks and valleys
#
#     The highest scaled latency becomes MAX_LATENCY.
#     """
#
#     output = df.copy()
#
#     original_maximum = output["latency_ms"].max()
#
#     if original_maximum <= 0:
#         raise ValueError(
#             "The maximum latency must be greater than zero."
#         )
#
#     scale_factor = (
#         MAX_LATENCY
#         / original_maximum
#     )
#
#     output["scaled_latency_ms"] = (
#         output["latency_ms"]
#         * scale_factor
#     )
#
#     print("=" * 65)
#     print(f"Original maximum latency: {original_maximum:,.2f} ms")
#     print(f"Target maximum latency:   {MAX_LATENCY:,.2f}")
#     print(f"Applied scale factor:     {scale_factor:.8f}")
#     print("=" * 65)
#
#     return output
#
#
# # ============================================================
# # Plot 1: Raw scaled end-to-end latency
# # ============================================================
#
# def plot_latency_timeline(
#     df: pd.DataFrame,
# ) -> None:
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#         )
#
#         plt.plot(
#             policy_df["elapsed_time_sec"],
#             policy_df["scaled_latency_ms"],
#             label=policy,
#             linewidth=1.5,
#             alpha=0.90,
#         )
#
#     plt.xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     plt.ylabel(
#         "Latency (ms)",
#         fontsize=12,
#     )
#
#     plt.title(
#         "End-to-End Latency – Azure Trace",
#         fontsize=14,
#     )
#
#     # No values are clipped because the data were scaled.
#     plt.ylim(
#         0,
#         MAX_LATENCY * 1.02,
#     )
#
#     plt.legend(
#         loc="upper left",
#         frameon=True,
#         fontsize=10,
#     )
#
#     save_and_show(
#         "01_scaled_latency_timeline.png"
#     )
#
#
# # ============================================================
# # Plot 2: Rolling scaled end-to-end latency
# # ============================================================
#
# def plot_rolling_latency(
#     df: pd.DataFrame,
# ) -> None:
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#             .copy()
#         )
#
#         policy_df["rolling_scaled_latency"] = (
#             policy_df["scaled_latency_ms"]
#             .rolling(
#                 window=ROLLING_WINDOW,
#                 min_periods=1,
#             )
#             .mean()
#         )
#
#         plt.plot(
#             policy_df["elapsed_time_sec"],
#             policy_df["rolling_scaled_latency"],
#             label=policy,
#             linewidth=2.2,
#             alpha=0.95,
#         )
#
#     plt.xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     plt.ylabel(
#         "Latency (ms)",
#         fontsize=12,
#     )
#
#     plt.title(
#         "End-to-End Latency – Azure Trace",
#         fontsize=14,
#     )
#
#     plt.ylim(
#         0,
#         MAX_LATENCY * 1.02,
#     )
#
#     plt.legend(
#         loc="best",
#         frameon=True,
#         fontsize=10,
#     )
#
#     save_and_show(
#         "02_scaled_rolling_latency.png"
#     )
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
#     df = load_results()
#
#     plot_latency_timeline(df)
#     plot_rolling_latency(df)
#
#     print("\nFinished generating both plots.")
#     print(f"Output directory: {OUTPUT_DIR.resolve()}")
#
#
# if __name__ == "__main__":
#     main()





#
#
# from __future__ import annotations
#
# from pathlib import Path
# import re
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
#
#
# # ============================================================
# # Configuration
# # ============================================================
#
# INPUT_FILE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/azure_workload_4IAT_Final.csv")
# OUTPUT_DIR = Path("figures")
#
# # Column names in the original CSV.
# REQUEST_COL = "ID"
# POLICY_COL = "effective_cascade_mode"
# LATENCY_COL = "e2e_time_ms"
# IAR_COL = "IAR"
#
# # IAR is assumed to be measured in milliseconds.
# IAR_DIVISOR = 1000.0
#
# # Number of consecutive requests used for the rolling average.
# ROLLING_WINDOW = 30
#
# # Target range for displaying normalised latency.
# MIN_DISPLAY_LATENCY = 500
# MAX_DISPLAY_LATENCY = 3500
#
# SAVE_PLOTS = True
# SHOW_PLOTS = True
#
#
# # ============================================================
# # Policy names
# # ============================================================
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential",
#     "seq": "Sequential",
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
#     "Sequential",
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
#     Extract the last numeric component from a request ID.
#
#     Examples:
#         REQ-00015 -> 15
#         request_8 -> 8
#         ID42      -> 42
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
#     Return policies that exist in the DataFrame using the
#     preferred plotting order.
#     """
#
#     available_policies = set(
#         df["policy_name"].dropna().unique()
#     )
#
#     return [
#         policy
#         for policy in POLICY_ORDER
#         if policy in available_policies
#     ]
#
#
# def save_and_show(filename: str) -> None:
#     """
#     Apply common formatting, save the figure, display it,
#     and then close it.
#     """
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     plt.grid(
#         True,
#         linestyle="--",
#         linewidth=0.7,
#         alpha=0.30,
#     )
#
#     plt.tight_layout()
#
#     if SAVE_PLOTS:
#         output_path = OUTPUT_DIR / filename
#
#         plt.savefig(
#             output_path,
#             dpi=300,
#             bbox_inches="tight",
#         )
#
#         print(f"Saved: {output_path.resolve()}")
#
#     if SHOW_PLOTS:
#         plt.show()
#
#     plt.close()
#
#
# # ============================================================
# # Load and clean data
# # ============================================================
#
# def load_results() -> pd.DataFrame:
#     """
#     Load the combined CSV, clean its columns, reconstruct workload
#     time, and add normalised latency.
#     """
#
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"Input CSV was not found:\n{INPUT_FILE.resolve()}"
#         )
#
#     df = pd.read_csv(INPUT_FILE)
#
#     print("=" * 70)
#     print(f"Input file: {INPUT_FILE.resolve()}")
#     print(f"Original rows: {len(df):,}")
#     print("=" * 70)
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
#     # Rename original CSV columns to simpler internal names.
#     df = df.rename(
#         columns={
#             REQUEST_COL: "request_id",
#             POLICY_COL: "raw_policy",
#             LATENCY_COL: "latency_ms",
#             IAR_COL: "iar_ms",
#         }
#     )
#
#     # Clean request IDs.
#     df["request_id"] = (
#         df["request_id"]
#         .astype(str)
#         .str.strip()
#     )
#
#     # Clean raw policy names.
#     df["raw_policy"] = (
#         df["raw_policy"]
#         .astype(str)
#         .str.strip()
#         .str.lower()
#     )
#
#     # Convert internal policy names to readable labels.
#     df["policy_name"] = (
#         df["raw_policy"]
#         .map(POLICY_NAME_MAP)
#         .fillna(df["raw_policy"])
#     )
#
#     # Convert latency and IAR to numeric values.
#     df["latency_ms"] = pd.to_numeric(
#         df["latency_ms"],
#         errors="coerce",
#     )
#
#     df["iar_ms"] = pd.to_numeric(
#         df["iar_ms"],
#         errors="coerce",
#     )
#
#     rows_before_cleaning = len(df)
#
#     # Remove rows missing required values.
#     df = df.dropna(
#         subset=[
#             "request_id",
#             "policy_name",
#             "latency_ms",
#             "iar_ms",
#         ]
#     ).copy()
#
#     removed_rows = rows_before_cleaning - len(df)
#
#     if removed_rows:
#         print(f"Removed invalid rows: {removed_rows:,}")
#
#     # Keep only the three policies being compared.
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
#             "No recognised Sequential, Edge Speculation, "
#             "or Cloud Speculation rows were found."
#         )
#
#     # Keep one result for each request-policy pair.
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
#             "Keeping the last row for each request-policy pair."
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
#     df = add_normalised_latency(df)
#
#     print_summary(df)
#
#     return df.reset_index(drop=True)
#
#
# # ============================================================
# # Reconstruct workload time from IAR
# # ============================================================
#
# def add_elapsed_time(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Reconstruct elapsed workload time using cumulative IAR.
#
#     Each request appears once for each policy, so IAR must be
#     counted only once for every unique request.
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
#     # Sort requests using the numeric part of their IDs.
#     if request_schedule["request_number"].notna().all():
#         request_schedule = request_schedule.sort_values(
#             "request_number"
#         )
#
#     else:
#         print(
#             "\nWarning: Numeric ordering could not be extracted "
#             "from every request ID."
#         )
#
#         print(
#             "The original CSV request order will be used."
#         )
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
#     # Replace invalid negative IAR values with zero.
#     negative_iar_count = int(
#         (request_schedule["iar_ms"] < 0).sum()
#     )
#
#     if negative_iar_count:
#         print(
#             f"\nWarning: {negative_iar_count:,} negative IAR "
#             "values were changed to zero."
#         )
#
#     request_schedule["iar_ms"] = (
#         request_schedule["iar_ms"]
#         .fillna(0)
#         .clip(lower=0)
#     )
#
#     # Reconstruct the request arrival timeline.
#     request_schedule["elapsed_time_sec"] = (
#         request_schedule["iar_ms"].cumsum()
#         / IAR_DIVISOR
#     )
#
#     # Place the first request at zero seconds.
#     if not request_schedule.empty:
#         first_time = request_schedule[
#             "elapsed_time_sec"
#         ].iloc[0]
#
#         request_schedule["elapsed_time_sec"] -= first_time
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
# # Normalise latency into the target display range
# # ============================================================
#
# def add_normalised_latency(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Apply global min-max normalisation.
#
#     The minimum original latency becomes MIN_DISPLAY_LATENCY.
#     The maximum original latency becomes MAX_DISPLAY_LATENCY.
#
#     One common transformation is applied to all policies so that
#     their relative ordering and curve shapes remain consistent.
#     """
#
#     output = df.copy()
#
#     if MAX_DISPLAY_LATENCY <= MIN_DISPLAY_LATENCY:
#         raise ValueError(
#             "MAX_DISPLAY_LATENCY must be greater than "
#             "MIN_DISPLAY_LATENCY."
#         )
#
#     original_minimum = output["latency_ms"].min()
#     original_maximum = output["latency_ms"].max()
#
#     original_range = (
#         original_maximum
#         - original_minimum
#     )
#
#     if original_range <= 0:
#         raise ValueError(
#             "Latency must contain at least two different values."
#         )
#
#     target_range = (
#         MAX_DISPLAY_LATENCY
#         - MIN_DISPLAY_LATENCY
#     )
#
#     output["normalised_latency"] = (
#         (
#             output["latency_ms"]
#             - original_minimum
#         )
#         / original_range
#         * target_range
#         + MIN_DISPLAY_LATENCY
#     )
#
#     print("\n" + "=" * 70)
#     print("Latency normalisation")
#     print("=" * 70)
#
#     print(
#         f"Original latency range: "
#         f"{original_minimum:,.2f} to "
#         f"{original_maximum:,.2f} ms"
#     )
#
#     print(
#         f"Normalised display range: "
#         f"{MIN_DISPLAY_LATENCY:,.2f} to "
#         f"{MAX_DISPLAY_LATENCY:,.2f}"
#     )
#
#     print("=" * 70)
#
#     return output
#
#
# # ============================================================
# # Print data summary
# # ============================================================
#
# def print_summary(df: pd.DataFrame) -> None:
#     """
#     Print validation information about the prepared dataset.
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
#     policy_counts_per_request = (
#         df.groupby("request_id")["policy_name"]
#         .nunique()
#     )
#
#     print("\nNumber of policies per request:")
#
#     print(
#         policy_counts_per_request
#         .value_counts()
#         .sort_index()
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
#         f"Original maximum latency: "
#         f"{df['latency_ms'].max():,.2f} ms"
#     )
#
#     print(
#         f"Normalised latency range: "
#         f"{df['normalised_latency'].min():,.2f} to "
#         f"{df['normalised_latency'].max():,.2f}"
#     )
#
#
# # ============================================================
# # Plot 1: Normalised end-to-end latency over time
# # ============================================================
#
# def plot_latency_timeline(
#     df: pd.DataFrame,
# ) -> None:
#     """
#     Plot normalised per-request latency over workload time.
#     """
#
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#         )
#
#         plt.plot(
#             policy_df["elapsed_time_sec"],
#             policy_df["normalised_latency"],
#             label=policy,
#             linewidth=1.7,
#             alpha=0.95,
#         )
#
#     plt.xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     plt.ylabel(
#         "Latency (ms)",
#         fontsize=12,
#     )
#
#     plt.title(
#         "End-to-End Latency – Azure Trace",
#         fontsize=14,
#     )
#
#     # Add a small visual margin around the configured range.
#     display_margin = (
#         MAX_DISPLAY_LATENCY
#         - MIN_DISPLAY_LATENCY
#     ) * 0.05
#
#     plt.ylim(
#         MIN_DISPLAY_LATENCY - display_margin,
#         MAX_DISPLAY_LATENCY + display_margin,
#     )
#
#     plt.legend(
#         loc="upper right",
#         frameon=True,
#         fontsize=10,
#     )
#
#     save_and_show(
#         "01_normalised_latency_timeline.png"
#     )
#
#
# # ============================================================
# # Plot 2: Rolling normalised latency over time
# # ============================================================
#
# def plot_rolling_latency(
#     df: pd.DataFrame,
# ) -> None:
#     """
#     Calculate rolling latency from the original latency values,
#     then normalise all rolling values into the configured range.
#
#     The minimum rolling latency becomes MIN_DISPLAY_LATENCY.
#     The maximum rolling latency becomes MAX_DISPLAY_LATENCY.
#     """
#
#     rolling_frames = []
#
#     # First calculate rolling latency using the original measurements.
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("elapsed_time_sec")
#             .copy()
#         )
#
#         policy_df["rolling_latency_ms"] = (
#             policy_df["latency_ms"]
#             .rolling(
#                 window=ROLLING_WINDOW,
#                 min_periods=1,
#             )
#             .mean()
#         )
#
#         rolling_frames.append(policy_df)
#
#     # Combine all policies so one common normalisation is applied.
#     rolling_df = pd.concat(
#         rolling_frames,
#         ignore_index=True,
#     )
#
#     rolling_minimum = rolling_df[
#         "rolling_latency_ms"
#     ].min()
#
#     rolling_maximum = rolling_df[
#         "rolling_latency_ms"
#     ].max()
#
#     rolling_range = (
#         rolling_maximum
#         - rolling_minimum
#     )
#
#     if rolling_range <= 0:
#         raise ValueError(
#             "Rolling latency contains no variation."
#         )
#
#     target_range = (
#         MAX_DISPLAY_LATENCY
#         - MIN_DISPLAY_LATENCY
#     )
#
#     # Normalise after calculating the rolling average.
#     rolling_df["normalised_rolling_latency"] = (
#         (
#             rolling_df["rolling_latency_ms"]
#             - rolling_minimum
#         )
#         / rolling_range
#         * target_range
#         + MIN_DISPLAY_LATENCY
#     )
#
#     print("\nRolling latency normalisation:")
#     print(
#         f"Original rolling range: "
#         f"{rolling_minimum:,.2f}–"
#         f"{rolling_maximum:,.2f} ms"
#     )
#     print(
#         f"Displayed rolling range: "
#         f"{rolling_df['normalised_rolling_latency'].min():.2f}–"
#         f"{rolling_df['normalised_rolling_latency'].max():.2f}"
#     )
#
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(rolling_df):
#         policy_df = (
#             rolling_df[
#                 rolling_df["policy_name"] == policy
#             ]
#             .sort_values("elapsed_time_sec")
#         )
#
#         plt.plot(
#             policy_df["elapsed_time_sec"],
#             policy_df["normalised_rolling_latency"],
#             label=policy,
#             linewidth=2.2,
#             alpha=0.95,
#         )
#
#     plt.xlabel(
#         "Workload Time (s)",
#         fontsize=12,
#     )
#
#     plt.ylabel(
#         "Latency (ms)",
#         fontsize=12,
#     )
#
#     plt.title(
#         "End-to-End Latency – Azure Trace",
#         fontsize=14,
#     )
#
#     display_margin = (
#         MAX_DISPLAY_LATENCY
#         - MIN_DISPLAY_LATENCY
#     ) * 0.03
#
#     plt.ylim(
#         MIN_DISPLAY_LATENCY - display_margin,
#         MAX_DISPLAY_LATENCY + display_margin,
#     )
#
#     plt.legend(
#         loc="upper right",
#         frameon=True,
#         fontsize=10,
#     )
#
#     save_and_show(
#         "02_rolling_normalised_latency.png"
#     )
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
#     df = load_results()
#
#     plot_latency_timeline(df)
#     plot_rolling_latency(df)
#
#     print("\n" + "=" * 70)
#     print("Finished generating both figures.")
#     print(f"Output directory: {OUTPUT_DIR.resolve()}")
#     print("=" * 70)
#
#
# if __name__ == "__main__":
#     main()

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = Path(
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/"
    "azure_workload_4IAT_Final_2_dif_C_E.csv"
    # "azure_workload_4IAT_Final.csv"
)

OUTPUT_DIR = Path("figures")

# Original CSV column names
REQUEST_COL = "ID"
POLICY_COL = "effective_cascade_mode"
LATENCY_COL = "e2e_time_ms"
IAR_COL = "IAR"

# IAR is assumed to be in milliseconds.
IAR_DIVISOR = 1000.0

# Number of consecutive requests used for the rolling mean.
ROLLING_WINDOW = 30

# Number of initial requests used to estimate the low-load baseline.
REFERENCE_REQUESTS = 30

# The initial low-load reference is displayed near this value.
INITIAL_TARGET_LATENCY = 500.0

# The maximum transformed latency is displayed near this value.
MAX_DISPLAY_LATENCY = 3500.0

# Prevent transformed values from becoming negative.
MIN_ALLOWED_LATENCY = 0.0

SAVE_PLOTS = True
SHOW_PLOTS = True


# ============================================================
# Policy names
# ============================================================

POLICY_NAME_MAP = {
    "sequential": "Sequential",
    "seq": "Sequential",

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
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]


# ============================================================
# Helper functions
# ============================================================

def extract_request_number(request_id: str) -> int | None:
    """
    Extract the last numeric component from a request ID.

    Examples:
        REQ-00015 -> 15
        request_8 -> 8
        ID42      -> 42
    """

    numbers = re.findall(r"\d+", str(request_id))

    if not numbers:
        return None

    return int(numbers[-1])


def get_policy_order(df: pd.DataFrame) -> list[str]:
    """
    Return the available policies using the preferred display order.
    """

    available_policies = set(
        df["policy_name"].dropna().unique()
    )

    return [
        policy
        for policy in POLICY_ORDER
        if policy in available_policies
    ]


def save_and_show(filename: str) -> None:
    """
    Format, save, show, and close the current figure.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.30,
    )

    plt.tight_layout()

    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / filename

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved: {output_path.resolve()}")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def get_initial_request_numbers(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Return the request numbers belonging to the initial reference period.
    """

    return (
        df[
            [
                "request_id",
                "request_number",
            ]
        ]
        .drop_duplicates(
            subset=["request_id"]
        )
        .sort_values("request_number")
        .head(REFERENCE_REQUESTS)["request_number"]
    )


# ============================================================
# Load and clean data
# ============================================================

def load_results() -> pd.DataFrame:
    """
    Load the CSV, clean the data, reconstruct workload time,
    and add transformed raw latency.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input CSV was not found:\n{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print(f"Input file: {INPUT_FILE.resolve()}")
    print(f"Original rows: {len(df):,}")
    print("=" * 70)

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

    df = df.rename(
        columns={
            REQUEST_COL: "request_id",
            POLICY_COL: "raw_policy",
            LATENCY_COL: "latency_ms",
            IAR_COL: "iar_ms",
        }
    )

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

    df["latency_ms"] = pd.to_numeric(
        df["latency_ms"],
        errors="coerce",
    )

    df["iar_ms"] = pd.to_numeric(
        df["iar_ms"],
        errors="coerce",
    )

    rows_before_cleaning = len(df)

    df = df.dropna(
        subset=[
            "request_id",
            "policy_name",
            "latency_ms",
            "iar_ms",
        ]
    ).copy()

    removed_rows = rows_before_cleaning - len(df)

    if removed_rows:
        print(f"Removed invalid rows: {removed_rows:,}")

    unknown_policies = sorted(
        set(df["policy_name"].unique())
        - set(POLICY_ORDER)
    )

    if unknown_policies:
        print("\nIgnoring unrecognised policies:")

        for policy in unknown_policies:
            print(f"  - {policy}")

    df = df[
        df["policy_name"].isin(POLICY_ORDER)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No recognised policy rows were found."
        )

    duplicate_count = int(
        df.duplicated(
            subset=[
                "request_id",
                "policy_name",
            ],
            keep=False,
        ).sum()
    )

    if duplicate_count:
        print(
            f"\nWarning: {duplicate_count:,} duplicate "
            "request-policy rows were found."
        )

        print(
            "Keeping the last row for each request-policy pair."
        )

        df = df.drop_duplicates(
            subset=[
                "request_id",
                "policy_name",
            ],
            keep="last",
        )

    df = add_elapsed_time(df)
    df = add_transformed_latency(df)

    print_summary(df)

    return df.reset_index(drop=True)


# ============================================================
# Reconstruct workload time from IAR
# ============================================================

def add_elapsed_time(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct elapsed workload time from cumulative IAR.

    Each request appears once for each policy. Therefore, IAR is
    counted only once per unique request.
    """

    output = df.copy()

    request_schedule = (
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

    request_schedule["request_number"] = (
        request_schedule["request_id"]
        .map(extract_request_number)
    )

    if request_schedule["request_number"].notna().all():
        request_schedule = request_schedule.sort_values(
            "request_number"
        )

    else:
        print(
            "\nWarning: Request numbers could not be extracted "
            "from every request ID."
        )

        print(
            "The original CSV request order will be used."
        )

        request_schedule = request_schedule.reset_index(
            drop=True
        )

        request_schedule["request_number"] = np.arange(
            1,
            len(request_schedule) + 1,
        )

    negative_iar_count = int(
        (request_schedule["iar_ms"] < 0).sum()
    )

    if negative_iar_count:
        print(
            f"\nWarning: {negative_iar_count:,} negative IAR "
            "values were replaced with zero."
        )

    request_schedule["iar_ms"] = (
        request_schedule["iar_ms"]
        .fillna(0)
        .clip(lower=0)
    )

    request_schedule["elapsed_time_sec"] = (
        request_schedule["iar_ms"].cumsum()
        / IAR_DIVISOR
    )

    if not request_schedule.empty:
        request_schedule["elapsed_time_sec"] -= (
            request_schedule["elapsed_time_sec"].iloc[0]
        )

    request_schedule = request_schedule[
        [
            "request_id",
            "request_number",
            "elapsed_time_sec",
        ]
    ]

    output = output.merge(
        request_schedule,
        on="request_id",
        how="left",
    )

    return output


# ============================================================
# Transform raw latency
# ============================================================

def add_transformed_latency(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply a reference-based linear transformation.

    The median latency during the initial reference period is mapped
    to INITIAL_TARGET_LATENCY.

    The maximum latency is mapped to MAX_DISPLAY_LATENCY.

    Values below the initial reference may fall below
    INITIAL_TARGET_LATENCY, allowing natural upward and downward
    movement around the initial baseline.
    """

    output = df.copy()

    if MAX_DISPLAY_LATENCY <= INITIAL_TARGET_LATENCY:
        raise ValueError(
            "MAX_DISPLAY_LATENCY must be greater than "
            "INITIAL_TARGET_LATENCY."
        )

    initial_request_numbers = get_initial_request_numbers(output)

    initial_df = output[
        output["request_number"].isin(
            initial_request_numbers
        )
    ]

    if initial_df.empty:
        raise ValueError(
            "The initial reference period contains no data."
        )

    initial_reference = initial_df[
        "latency_ms"
    ].median()

    original_maximum = output[
        "latency_ms"
    ].max()

    maximum_difference = (
        original_maximum
        - initial_reference
    )

    if maximum_difference <= 0:
        raise ValueError(
            "The maximum latency must be greater than the "
            "initial reference latency."
        )

    scale_factor = (
        MAX_DISPLAY_LATENCY
        - INITIAL_TARGET_LATENCY
    ) / maximum_difference

    output["display_latency"] = (
        INITIAL_TARGET_LATENCY
        + (
            output["latency_ms"]
            - initial_reference
        )
        * scale_factor
    )

    output["display_latency"] = (
        output["display_latency"]
        .clip(lower=MIN_ALLOWED_LATENCY)
    )

    print("\n" + "=" * 70)
    print("Raw latency transformation")
    print("=" * 70)

    print(
        f"Reference requests: first "
        f"{min(REFERENCE_REQUESTS, output['request_id'].nunique())}"
    )

    print(
        f"Initial median latency: "
        f"{initial_reference:,.2f} ms"
    )

    print(
        f"Initial target value: "
        f"{INITIAL_TARGET_LATENCY:,.2f}"
    )

    print(
        f"Original maximum latency: "
        f"{original_maximum:,.2f} ms"
    )

    print(
        f"Transformed maximum: "
        f"{output['display_latency'].max():,.2f}"
    )

    print(
        f"Transformed minimum: "
        f"{output['display_latency'].min():,.2f}"
    )

    print(
        f"Scale factor: {scale_factor:.8f}"
    )

    print("=" * 70)

    return output


# ============================================================
# Print summary
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    """
    Print validation information.
    """

    print("\nPolicy counts:")

    print(
        df["policy_name"]
        .value_counts()
        .reindex(get_policy_order(df))
    )

    policies_per_request = (
        df.groupby("request_id")["policy_name"]
        .nunique()
    )

    print("\nNumber of policies per request:")

    print(
        policies_per_request
        .value_counts()
        .sort_index()
    )

    print(
        f"\nUnique requests: "
        f"{df['request_id'].nunique():,}"
    )

    print(
        f"Workload duration: "
        f"{df['elapsed_time_sec'].max():,.2f} seconds"
    )

    print(
        f"Original latency range: "
        f"{df['latency_ms'].min():,.2f} to "
        f"{df['latency_ms'].max():,.2f} ms"
    )

    print(
        f"Transformed latency range: "
        f"{df['display_latency'].min():,.2f} to "
        f"{df['display_latency'].max():,.2f}"
    )


# ============================================================
# Plot 1: Transformed end-to-end latency
# ============================================================

def plot_latency_timeline(
    df: pd.DataFrame,
) -> None:
    """
    Plot transformed per-request latency over workload time.
    """

    plt.figure(figsize=(13, 5))

    for policy in get_policy_order(df):
        policy_df = (
            df[df["policy_name"] == policy]
            .sort_values("elapsed_time_sec")
        )

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["display_latency"],
            label=policy,
            linewidth=1.7,
            alpha=0.95,
        )

    plt.xlabel(
        "Workload Time (s)",
        fontsize=12,
    )

    plt.ylabel(
        "Transformed Latency",
        fontsize=12,
    )

    plt.title(
        "End-to-End Latency – Azure Trace",
        fontsize=14,
    )

    plt.ylim(
        MIN_ALLOWED_LATENCY,
        MAX_DISPLAY_LATENCY * 1.02,
    )

    plt.legend(
        loc="upper right",
        frameon=True,
        fontsize=10,
    )

    save_and_show(
        "01_transformed_latency_timeline.png"
    )


# ============================================================
# Plot 2: Transformed rolling latency
# ============================================================

def plot_rolling_latency(
    df: pd.DataFrame,
) -> None:
    """
    Calculate rolling latency from original measurements and then
    apply a separate reference-based transformation.

    This ensures the rolling graph also reaches approximately
    MAX_DISPLAY_LATENCY.
    """

    rolling_frames: list[pd.DataFrame] = []

    for policy in get_policy_order(df):
        policy_df = (
            df[df["policy_name"] == policy]
            .sort_values("elapsed_time_sec")
            .copy()
        )

        policy_df["rolling_latency_ms"] = (
            policy_df["latency_ms"]
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
            )
            .mean()
        )

        rolling_frames.append(policy_df)

    if not rolling_frames:
        raise RuntimeError(
            "No policy data were available for rolling latency."
        )

    rolling_df = pd.concat(
        rolling_frames,
        ignore_index=True,
    )

    initial_request_numbers = get_initial_request_numbers(
        rolling_df
    )

    initial_rolling_df = rolling_df[
        rolling_df["request_number"].isin(
            initial_request_numbers
        )
    ]

    if initial_rolling_df.empty:
        raise ValueError(
            "The initial rolling reference period contains no data."
        )

    initial_rolling_reference = initial_rolling_df[
        "rolling_latency_ms"
    ].median()

    rolling_maximum = rolling_df[
        "rolling_latency_ms"
    ].max()

    maximum_difference = (
        rolling_maximum
        - initial_rolling_reference
    )

    if maximum_difference <= 0:
        raise ValueError(
            "The maximum rolling latency must be greater than "
            "the initial rolling reference."
        )

    rolling_scale_factor = (
        MAX_DISPLAY_LATENCY
        - INITIAL_TARGET_LATENCY
    ) / maximum_difference

    rolling_df["display_rolling_latency"] = (
        INITIAL_TARGET_LATENCY
        + (
            rolling_df["rolling_latency_ms"]
            - initial_rolling_reference
        )
        * rolling_scale_factor
    )

    rolling_df["display_rolling_latency"] = (
        rolling_df["display_rolling_latency"]
        .clip(lower=MIN_ALLOWED_LATENCY)
    )

    print("\n" + "=" * 70)
    print("Rolling latency transformation")
    print("=" * 70)

    print(
        f"Initial rolling median: "
        f"{initial_rolling_reference:,.2f} ms"
    )

    print(
        f"Original rolling maximum: "
        f"{rolling_maximum:,.2f} ms"
    )

    print(
        f"Transformed rolling minimum: "
        f"{rolling_df['display_rolling_latency'].min():,.2f}"
    )

    print(
        f"Transformed rolling maximum: "
        f"{rolling_df['display_rolling_latency'].max():,.2f}"
    )

    print(
        f"Rolling scale factor: "
        f"{rolling_scale_factor:.8f}"
    )

    print("=" * 70)

    plt.figure(figsize=(13, 5))

    for policy in get_policy_order(rolling_df):
        policy_df = (
            rolling_df[
                rolling_df["policy_name"] == policy
            ]
            .sort_values("elapsed_time_sec")
        )

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["display_rolling_latency"],
            label=policy,
            linewidth=2.2,
            alpha=0.95,
        )

    plt.xlabel(
        "Workload Time (s)",
        fontsize=12,
    )

    plt.ylabel(
        "Transformed Rolling Latency",
        fontsize=12,
    )

    plt.title(
        "Rolling End-to-End Latency – Azure Trace",
        fontsize=14,
    )

    plt.ylim(
        MIN_ALLOWED_LATENCY,
        MAX_DISPLAY_LATENCY * 1.02,
    )

    plt.legend(
        loc="upper right",
        frameon=True,
        fontsize=10,
    )

    save_and_show(
        "02_transformed_rolling_latency.png"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    plot_latency_timeline(df)
    plot_rolling_latency(df)

    print("\n" + "=" * 70)
    print("Finished generating both figures.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()