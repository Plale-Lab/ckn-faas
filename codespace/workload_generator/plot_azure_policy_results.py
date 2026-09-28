#
# from __future__ import annotations
#
# from pathlib import Path
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
# INPUT_FILE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/azure_workload_1IAT_Final.csv")
# OUTPUT_DIR = Path("figures")
#
# REQUEST_COL = "ID"
# POLICY_COL = "effective_cascade_mode"
# LATENCY_COL = "e2e_time_ms"
# DEADLINE_COL = "Deadline"
# ACCURACY_COL = "Accuracy"
#
# ROLLING_WINDOW = 30
#
# # Set to True when Accuracy is confidence rather than correctness.
# # Based on your sample, Accuracy appears to be model confidence.
# # Therefore, the accuracy-latency graph is disabled by default.
# PLOT_ACCURACY_LATENCY = False
#
# POLICY_NAME_MAP = {
#     "sequential": "Sequential",
#     "seq": "Sequential",
#
#     "spec_b": "Cloud Speculation",
#     "cloud_speculation": "Cloud Speculation",
#
#     "parallel_first_finish": "Edge Speculation",
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "spec_b_power2": "Edge Speculation",
# }
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
# def load_results() -> pd.DataFrame:
#     """
#     Load the combined result CSV and create standard internal
#     column names.
#     """
#
#     if not INPUT_FILE.exists():
#         raise FileNotFoundError(
#             f"Input file was not found: {INPUT_FILE.resolve()}"
#         )
#
#     df = pd.read_csv(INPUT_FILE)
#
#     print(f"\nLoaded {len(df):,} rows")
#     print(f"Input file: {INPUT_FILE.resolve()}")
#
#     required_columns = [
#         REQUEST_COL,
#         POLICY_COL,
#         LATENCY_COL,
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
#             f"Missing required columns: {missing_columns}\n"
#             f"Available columns:\n{list(df.columns)}"
#         )
#
#     rename_map = {
#         REQUEST_COL: "request_id",
#         POLICY_COL: "raw_policy",
#         LATENCY_COL: "latency_ms",
#     }
#
#     if DEADLINE_COL in df.columns:
#         rename_map[DEADLINE_COL] = "deadline_ms"
#
#     if ACCURACY_COL in df.columns:
#         rename_map[ACCURACY_COL] = "accuracy_value"
#
#     df = df.rename(columns=rename_map)
#
#     # Clean request IDs.
#     df["request_id"] = (
#         df["request_id"]
#         .astype(str)
#         .str.strip()
#     )
#
#     # Clean policy names.
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
#     # Convert latency to numeric.
#     df["latency_ms"] = pd.to_numeric(
#         df["latency_ms"],
#         errors="coerce",
#     )
#
#     if "deadline_ms" in df.columns:
#         df["deadline_ms"] = pd.to_numeric(
#             df["deadline_ms"],
#             errors="coerce",
#         )
#
#     if "accuracy_value" in df.columns:
#         df["accuracy_value"] = pd.to_numeric(
#             df["accuracy_value"],
#             errors="coerce",
#         )
#
#     original_rows = len(df)
#
#     df = df.dropna(
#         subset=[
#             "request_id",
#             "policy_name",
#             "latency_ms",
#         ]
#     )
#
#     removed_rows = original_rows - len(df)
#
#     if removed_rows > 0:
#         print(
#             f"Removed {removed_rows:,} rows containing "
#             "missing IDs, policies, or latency values."
#         )
#
#     # Keep only the three policies of interest.
#     expected_policies = set(POLICY_ORDER)
#
#     unknown_policies = sorted(
#         set(df["policy_name"].unique()) - expected_policies
#     )
#
#     if unknown_policies:
#         print("\nIgnoring unrecognised policy values:")
#         for policy in unknown_policies:
#             print(f"  - {policy}")
#
#     df = df[
#         df["policy_name"].isin(expected_policies)
#     ].copy()
#
#     # Remove duplicate policy measurements for the same request.
#     duplicate_mask = df.duplicated(
#         subset=["request_id", "policy_name"],
#         keep=False,
#     )
#
#     duplicate_count = int(duplicate_mask.sum())
#
#     if duplicate_count > 0:
#         print(
#             f"\nWarning: {duplicate_count:,} rows belong to "
#             "duplicate request-policy combinations."
#         )
#         print(
#             "The script keeps the last row for each "
#             "request-policy combination."
#         )
#
#         df = df.drop_duplicates(
#             subset=["request_id", "policy_name"],
#             keep="last",
#         )
#
#     df = add_request_index(df)
#
#     print("\nPolicy counts:")
#     print(df["policy_name"].value_counts())
#
#     print("\nNumber of policy rows per request:")
#     print(
#         df.groupby("request_id")
#         .size()
#         .value_counts()
#         .sort_index()
#     )
#
#     complete_requests = (
#         df.groupby("request_id")["policy_name"]
#         .nunique()
#         .eq(3)
#         .sum()
#     )
#
#     total_requests = df["request_id"].nunique()
#
#     print(
#         f"\nComplete requests with all three policies: "
#         f"{complete_requests:,}/{total_requests:,}"
#     )
#
#     return df.reset_index(drop=True)
#
#
# def extract_request_number(request_id: str) -> int | None:
#     """
#     Extract the numeric portion of IDs such as REQ-00475.
#     """
#
#     digits = "".join(
#         character
#         for character in str(request_id)
#         if character.isdigit()
#     )
#
#     if not digits:
#         return None
#
#     return int(digits)
#
#
# def add_request_index(df: pd.DataFrame) -> pd.DataFrame:
#     """
#     Add a consistent numeric request index.
#
#     REQ-00001 becomes request_index=1, when possible.
#     Otherwise, request order is based on first appearance.
#     """
#
#     output = df.copy()
#
#     unique_requests = (
#         output[["request_id"]]
#         .drop_duplicates()
#         .copy()
#     )
#
#     unique_requests["parsed_request_number"] = (
#         unique_requests["request_id"]
#         .map(extract_request_number)
#     )
#
#     if unique_requests["parsed_request_number"].notna().all():
#         unique_requests = unique_requests.sort_values(
#             "parsed_request_number"
#         )
#
#         unique_requests["request_index"] = (
#             unique_requests["parsed_request_number"]
#         )
#     else:
#         unique_requests = unique_requests.reset_index(drop=True)
#
#         unique_requests["request_index"] = np.arange(
#             1,
#             len(unique_requests) + 1,
#         )
#
#     unique_requests = unique_requests[
#         ["request_id", "request_index"]
#     ]
#
#     output = output.merge(
#         unique_requests,
#         on="request_id",
#         how="left",
#     )
#
#     return output
#
#
# def get_policy_order(df: pd.DataFrame) -> list[str]:
#     """
#     Return only policies available in the current DataFrame,
#     following the desired presentation order.
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
# def save_figure(filename: str) -> None:
#     """
#     Save the current matplotlib figure.
#     """
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     plt.grid(
#         True,
#         alpha=0.25,
#         linewidth=0.7,
#     )
#
#     plt.tight_layout()
#
#     output_path = OUTPUT_DIR / filename
#
#     plt.savefig(
#         output_path,
#         dpi=300,
#         bbox_inches="tight",
#     )
#
#     plt.show()
#
#     plt.close()
#
#     print(f"Saved: {output_path}")
#
#
# def create_complete_latency_matrix(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Create one row per request and one latency column per policy.
#
#     Only requests containing results for all available policies
#     are retained.
#     """
#
#     matrix = df.pivot_table(
#         index=[
#             "request_id",
#             "request_index",
#         ],
#         columns="policy_name",
#         values="latency_ms",
#         aggfunc="mean",
#     )
#
#     policy_columns = get_policy_order(df)
#
#     missing_policy_columns = [
#         policy
#         for policy in policy_columns
#         if policy not in matrix.columns
#     ]
#
#     if missing_policy_columns:
#         raise ValueError(
#             "The following policies are missing from the "
#             f"latency matrix: {missing_policy_columns}"
#         )
#
#     matrix = matrix[
#         policy_columns
#     ].dropna()
#
#     matrix = matrix.reset_index()
#
#     matrix = matrix.sort_values(
#         "request_index"
#     ).reset_index(drop=True)
#
#     return matrix
#
#
# # ============================================================
# # 1. Raw latency versus request number
# # ============================================================
#
# def plot_latency_timeline(df: pd.DataFrame) -> None:
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("request_index")
#         )
#
#         plt.plot(
#             policy_df["request_index"],
#             policy_df["latency_ms"],
#             label=policy,
#             linewidth=1.0,
#             alpha=0.75,
#         )
#
#     plt.xlabel("Request Number")
#     plt.ylabel("End-to-End Latency (ms)")
#     plt.title(
#         "Per-Request Latency Under the Azure Workload"
#     )
#     plt.legend()
#
#     save_figure("01_latency_timeline.png")
#
#
# # ============================================================
# # 2. Rolling average latency
# # ============================================================
#
# def plot_rolling_latency(df: pd.DataFrame) -> None:
#     plt.figure(figsize=(13, 5))
#
#     for policy in get_policy_order(df):
#         policy_df = (
#             df[df["policy_name"] == policy]
#             .sort_values("request_index")
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
#         plt.plot(
#             policy_df["request_index"],
#             policy_df["rolling_latency_ms"],
#             label=policy,
#             linewidth=2,
#         )
#
#     plt.xlabel("Request Number")
#     plt.ylabel(
#         f"Rolling Mean Latency (ms), "
#         f"Window={ROLLING_WINDOW}"
#     )
#     plt.title(
#         "Rolling Latency During Azure Trace Replay"
#     )
#     plt.legend()
#
#     save_figure("02_rolling_latency.png")
#
#
# # ============================================================
# # 3. Latency CDF
# # ============================================================
#
# def plot_latency_cdf(df: pd.DataFrame) -> None:
#     plt.figure(figsize=(8, 5))
#
#     for policy in get_policy_order(df):
#         latency_values = (
#             df.loc[
#                 df["policy_name"] == policy,
#                 "latency_ms",
#             ]
#             .dropna()
#             .sort_values()
#             .to_numpy()
#         )
#
#         if len(latency_values) == 0:
#             continue
#
#         cdf = (
#             np.arange(
#                 1,
#                 len(latency_values) + 1,
#             )
#             / len(latency_values)
#         )
#
#         plt.plot(
#             latency_values,
#             cdf,
#             label=policy,
#             linewidth=2,
#         )
#
#     plt.xlabel("End-to-End Latency (ms)")
#     plt.ylabel("Cumulative Probability")
#     plt.title("End-to-End Latency CDF")
#     plt.legend()
#
#     save_figure("03_latency_cdf.png")
#
#
# # ============================================================
# # 4. Calculate offline best policy
# # ============================================================
#
# def calculate_best_policy(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     """
#     Identify the policy with minimum measured latency for
#     every request.
#
#     This is an offline oracle, not a runtime policy decision.
#     """
#
#     matrix = create_complete_latency_matrix(df)
#
#     policy_columns = get_policy_order(df)
#
#     matrix["best_policy"] = matrix[
#         policy_columns
#     ].idxmin(axis=1)
#
#     matrix["best_latency_ms"] = matrix[
#         policy_columns
#     ].min(axis=1)
#
#     matrix["worst_latency_ms"] = matrix[
#         policy_columns
#     ].max(axis=1)
#
#     matrix["oracle_saving_ms"] = (
#         matrix["worst_latency_ms"]
#         - matrix["best_latency_ms"]
#     )
#
#     return matrix
#
#
# # ============================================================
# # 5. Best-policy timeline
# # ============================================================
#
# def plot_best_policy_timeline(
#     best_df: pd.DataFrame,
# ) -> None:
#     available_policies = [
#         policy
#         for policy in POLICY_ORDER
#         if policy in best_df["best_policy"].unique()
#     ]
#
#     policy_to_code = {
#         policy: index
#         for index, policy in enumerate(
#             available_policies
#         )
#     }
#
#     plotting_df = best_df.copy()
#
#     plotting_df["policy_code"] = (
#         plotting_df["best_policy"]
#         .map(policy_to_code)
#     )
#
#     plt.figure(figsize=(13, 4))
#
#     plt.scatter(
#         plotting_df["request_index"],
#         plotting_df["policy_code"],
#         s=14,
#         alpha=0.8,
#     )
#
#     plt.yticks(
#         list(policy_to_code.values()),
#         list(policy_to_code.keys()),
#     )
#
#     plt.xlabel("Request Number")
#     plt.ylabel("Lowest-Latency Policy")
#     plt.title(
#         "Offline Lowest-Latency Policy for Each Request"
#     )
#
#     save_figure("04_best_policy_timeline.png")
#
#
# # ============================================================
# # 6. Best-policy region strip
# # ============================================================
#
# def plot_best_policy_regions(
#     best_df: pd.DataFrame,
# ) -> None:
#     available_policies = [
#         policy
#         for policy in POLICY_ORDER
#         if policy in best_df["best_policy"].unique()
#     ]
#
#     policy_to_code = {
#         policy: index
#         for index, policy in enumerate(
#             available_policies
#         )
#     }
#
#     policy_values = (
#         best_df["best_policy"]
#         .map(policy_to_code)
#         .to_numpy()
#         .reshape(1, -1)
#     )
#
#     plt.figure(figsize=(14, 2.5))
#
#     image = plt.imshow(
#         policy_values,
#         aspect="auto",
#         interpolation="nearest",
#     )
#
#     plt.yticks([])
#     plt.xlabel("Request Number")
#     plt.title("Offline Lowest-Latency Policy Regions")
#
#     colorbar = plt.colorbar(
#         image,
#         ticks=list(policy_to_code.values()),
#     )
#
#     colorbar.ax.set_yticklabels(
#         list(policy_to_code.keys())
#     )
#
#     save_figure("05_best_policy_regions.png")
#
#
# # ============================================================
# # 7. Policy win percentage
# # ============================================================
#
# def plot_policy_win_percentage(
#     best_df: pd.DataFrame,
# ) -> None:
#     win_counts = (
#         best_df["best_policy"]
#         .value_counts()
#         .reindex(
#             [
#                 policy
#                 for policy in POLICY_ORDER
#                 if policy
#                 in best_df["best_policy"].unique()
#             ]
#         )
#         .dropna()
#     )
#
#     percentages = (
#         win_counts
#         / win_counts.sum()
#         * 100
#     )
#
#     plt.figure(figsize=(8, 5))
#
#     bars = plt.bar(
#         percentages.index,
#         percentages.values,
#     )
#
#     for bar, percentage in zip(
#         bars,
#         percentages.values,
#     ):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{percentage:.1f}%",
#             ha="center",
#             va="bottom",
#         )
#
#     plt.xlabel("Policy")
#     plt.ylabel("Requests with Lowest Latency (%)")
#     plt.title("Per-Request Policy Win Rate")
#     plt.xticks(rotation=12)
#     plt.ylim(
#         0,
#         max(percentages.values) * 1.15,
#     )
#
#     save_figure("06_policy_win_percentage.png")
#
#
# # ============================================================
# # 8. Mean, median, P95 and P99 latency
# # ============================================================
#
# def plot_latency_summary(
#     df: pd.DataFrame,
# ) -> pd.DataFrame:
#     summary_rows = []
#
#     for policy in get_policy_order(df):
#         latency = df.loc[
#             df["policy_name"] == policy,
#             "latency_ms",
#         ].dropna()
#
#         summary_rows.append(
#             {
#                 "Policy": policy,
#                 "Mean": latency.mean(),
#                 "Median": latency.median(),
#                 "P95": latency.quantile(0.95),
#                 "P99": latency.quantile(0.99),
#             }
#         )
#
#     summary = pd.DataFrame(
#         summary_rows
#     ).set_index("Policy")
#
#     OUTPUT_DIR.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     summary.to_csv(
#         OUTPUT_DIR / "latency_summary.csv"
#     )
#
#     summary.plot(
#         kind="bar",
#         figsize=(10, 5),
#     )
#
#     plt.xlabel("Policy")
#     plt.ylabel("End-to-End Latency (ms)")
#     plt.title("Average and Tail Latency")
#     plt.xticks(rotation=12)
#     plt.legend(title="Metric")
#
#     save_figure("07_latency_summary.png")
#
#     print("\nLatency summary:")
#     print(summary.round(2))
#
#     return summary
#
#
# # ============================================================
# # 9. Deadline success rate
# # ============================================================
#
# def plot_deadline_success(
#     df: pd.DataFrame,
# ) -> None:
#     if "deadline_ms" not in df.columns:
#         print(
#             "\nSkipped deadline success graph: "
#             f"column '{DEADLINE_COL}' was not found."
#         )
#         return
#
#     working = df.dropna(
#         subset=[
#             "latency_ms",
#             "deadline_ms",
#         ]
#     ).copy()
#
#     if working.empty:
#         print(
#             "\nSkipped deadline success graph: "
#             "no valid deadline values were found."
#         )
#         return
#
#     working["deadline_met"] = (
#         working["latency_ms"]
#         <= working["deadline_ms"]
#     )
#
#     success_rate = (
#         working.groupby("policy_name")[
#             "deadline_met"
#         ]
#         .mean()
#         .mul(100)
#         .reindex(get_policy_order(working))
#     )
#
#     plt.figure(figsize=(8, 5))
#
#     bars = plt.bar(
#         success_rate.index,
#         success_rate.values,
#     )
#
#     for bar, value in zip(
#         bars,
#         success_rate.values,
#     ):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{value:.1f}%",
#             ha="center",
#             va="bottom",
#         )
#
#     plt.xlabel("Policy")
#     plt.ylabel("Deadline Success Rate (%)")
#     plt.title("Deadline Success Rate")
#     plt.ylim(0, 105)
#     plt.xticks(rotation=12)
#
#     save_figure("08_deadline_success.png")
#
#
# # ============================================================
# # 10. Latency reduction relative to Sequential
# # ============================================================
#
# def plot_improvement_over_sequential(
#     df: pd.DataFrame,
# ) -> None:
#     mean_latency = (
#         df.groupby("policy_name")[
#             "latency_ms"
#         ]
#         .mean()
#         .reindex(get_policy_order(df))
#     )
#
#     if "Sequential" not in mean_latency.index:
#         print(
#             "\nSkipped latency improvement graph: "
#             "Sequential results were not found."
#         )
#         return
#
#     sequential_mean = mean_latency["Sequential"]
#
#     improvement = (
#         (
#             sequential_mean
#             - mean_latency
#         )
#         / sequential_mean
#         * 100
#     )
#
#     improvement = improvement.drop(
#         labels=["Sequential"],
#         errors="ignore",
#     )
#
#     plt.figure(figsize=(8, 5))
#
#     bars = plt.bar(
#         improvement.index,
#         improvement.values,
#     )
#
#     plt.axhline(
#         0,
#         linewidth=1,
#     )
#
#     for bar, value in zip(
#         bars,
#         improvement.values,
#     ):
#         vertical_alignment = (
#             "bottom"
#             if value >= 0
#             else "top"
#         )
#
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{value:.1f}%",
#             ha="center",
#             va=vertical_alignment,
#         )
#
#     plt.xlabel("Policy")
#     plt.ylabel("Mean Latency Reduction (%)")
#     plt.title(
#         "Mean Latency Reduction Relative to Sequential"
#     )
#     plt.xticks(rotation=12)
#
#     save_figure(
#         "09_improvement_over_sequential.png"
#     )
#
#
# # ============================================================
# # 11. Oracle latency versus fixed policies
# # ============================================================
#
# def plot_oracle_latency(
#     df: pd.DataFrame,
#     best_df: pd.DataFrame,
# ) -> None:
#     fixed_means = (
#         df.groupby("policy_name")[
#             "latency_ms"
#         ]
#         .mean()
#         .reindex(get_policy_order(df))
#     )
#
#     oracle_mean = best_df[
#         "best_latency_ms"
#     ].mean()
#
#     comparison = fixed_means.copy()
#     comparison.loc["Offline Oracle"] = oracle_mean
#
#     plt.figure(figsize=(9, 5))
#
#     bars = plt.bar(
#         comparison.index,
#         comparison.values,
#     )
#
#     for bar, value in zip(
#         bars,
#         comparison.values,
#     ):
#         plt.text(
#             bar.get_x() + bar.get_width() / 2,
#             bar.get_height(),
#             f"{value:.1f}",
#             ha="center",
#             va="bottom",
#         )
#
#     plt.xlabel("Policy")
#     plt.ylabel("Mean End-to-End Latency (ms)")
#     plt.title(
#         "Fixed Policies Compared with the Offline Oracle"
#     )
#     plt.xticks(rotation=12)
#
#     save_figure(
#         "10_oracle_vs_fixed_policy_latency.png"
#     )
#
#
# # ============================================================
# # 12. Per-request oracle saving
# # ============================================================
#
# def plot_oracle_saving(
#     best_df: pd.DataFrame,
# ) -> None:
#     plotting_df = best_df.sort_values(
#         "request_index"
#     ).copy()
#
#     plotting_df["rolling_oracle_saving_ms"] = (
#         plotting_df["oracle_saving_ms"]
#         .rolling(
#             window=ROLLING_WINDOW,
#             min_periods=1,
#         )
#         .mean()
#     )
#
#     plt.figure(figsize=(13, 5))
#
#     plt.plot(
#         plotting_df["request_index"],
#         plotting_df[
#             "rolling_oracle_saving_ms"
#         ],
#         linewidth=2,
#     )
#
#     plt.xlabel("Request Number")
#     plt.ylabel(
#         "Rolling Latency Gap Between "
#         "Worst and Best Policy (ms)"
#     )
#     plt.title(
#         "Potential Benefit of Per-Request Policy Selection"
#     )
#
#     save_figure("11_oracle_saving_timeline.png")
#
#
# # ============================================================
# # 13. Optional accuracy-confidence versus latency
# # ============================================================
#
# def plot_accuracy_latency(
#     df: pd.DataFrame,
# ) -> None:
#     if not PLOT_ACCURACY_LATENCY:
#         print(
#             "\nSkipped accuracy-latency graph because "
#             "PLOT_ACCURACY_LATENCY=False."
#         )
#         return
#
#     if "accuracy_value" not in df.columns:
#         print(
#             "\nSkipped accuracy-latency graph: "
#             f"column '{ACCURACY_COL}' was not found."
#         )
#         return
#
#     summary = (
#         df.groupby("policy_name")
#         .agg(
#             mean_latency_ms=(
#                 "latency_ms",
#                 "mean",
#             ),
#             mean_accuracy_value=(
#                 "accuracy_value",
#                 "mean",
#             ),
#         )
#         .reindex(get_policy_order(df))
#         .reset_index()
#     )
#
#     plt.figure(figsize=(8, 6))
#
#     plt.scatter(
#         summary["mean_latency_ms"],
#         summary["mean_accuracy_value"],
#         s=120,
#     )
#
#     for _, row in summary.iterrows():
#         plt.annotate(
#             row["policy_name"],
#             (
#                 row["mean_latency_ms"],
#                 row["mean_accuracy_value"],
#             ),
#             xytext=(7, 7),
#             textcoords="offset points",
#         )
#
#     plt.xlabel("Mean End-to-End Latency (ms)")
#     plt.ylabel("Mean Accuracy or Confidence Value")
#     plt.title("Accuracy/Confidence–Latency Trade-off")
#
#     save_figure(
#         "12_accuracy_latency_tradeoff.png"
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
#     if df.empty:
#         raise RuntimeError(
#             "No recognised sequential, spec_b, or "
#             "parallel_first_finish rows were found."
#         )
#
#     plot_latency_timeline(df)
#     plot_rolling_latency(df)
#     plot_latency_cdf(df)
#
#     best_df = calculate_best_policy(df)
#
#     best_output_path = (
#         OUTPUT_DIR
#         / "best_policy_per_request.csv"
#     )
#
#     best_df.to_csv(
#         best_output_path,
#         index=False,
#     )
#
#     print(
#         f"\nSaved offline best-policy data: "
#         f"{best_output_path}"
#     )
#
#     plot_best_policy_timeline(best_df)
#     plot_best_policy_regions(best_df)
#     plot_policy_win_percentage(best_df)
#     plot_latency_summary(df)
#     plot_deadline_success(df)
#     plot_improvement_over_sequential(df)
#     plot_oracle_latency(df, best_df)
#     plot_oracle_saving(best_df)
#     plot_accuracy_latency(df)
#
#     print("\nFinished generating all available figures.")
#     print(f"Output directory: {OUTPUT_DIR.resolve()}")
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

INPUT_FILE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/azure_workload_4IAT_Final.csv")
OUTPUT_DIR = Path("figures")

# Actual CSV column names
REQUEST_COL = "ID"
POLICY_COL = "effective_cascade_mode"
LATENCY_COL = "e2e_time_ms"
RUNTIME_COL = "RunTime"
IAR_COL = "IAR"
DEADLINE_COL = "Deadline"
ACCURACY_COL = "Accuracy"

# IAR is assumed to be milliseconds.
IAR_DIVISOR = 1000.0

ROLLING_WINDOW = 30

# Show plots when running the script.
SHOW_PLOTS = True

# Save plots as PNG files.
SAVE_PLOTS = True

# Accuracy may represent confidence rather than real correctness.
PLOT_ACCURACY_LATENCY = False


POLICY_NAME_MAP = {
    "sequential": "Sequential",
    "seq": "Sequential",

    "spec_b": "Cloud Speculation",
    "cloud_speculation": "Cloud Speculation",
    "cloud speculation": "Cloud Speculation",

    "parallel_first_finish": "Edge Speculation",
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "edge_speculation": "Edge Speculation",
    "edge speculation": "Edge Speculation",
}


POLICY_ORDER = [
    "Sequential",
    "Edge Speculation",
    "Cloud Speculation",
]


# ============================================================
# General helper functions
# ============================================================

def extract_request_number(request_id: str) -> int | None:
    """
    Extract the numeric portion from request IDs.

    Examples:
        REQ-00015 -> 15
        request_37 -> 37
    """

    match = re.search(r"\d+", str(request_id))

    if match is None:
        return None

    return int(match.group())


def get_policy_order(df: pd.DataFrame) -> list[str]:
    """
    Return only policies present in the current DataFrame,
    following the requested presentation order.
    """

    available = set(df["policy_name"].dropna().unique())

    return [
        policy
        for policy in POLICY_ORDER
        if policy in available
    ]


def save_and_show(filename: str) -> None:
    """
    Apply final formatting, save the figure, and optionally show it.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    plt.grid(
        True,
        alpha=0.25,
        linewidth=0.7,
    )

    plt.tight_layout()

    if SAVE_PLOTS:
        output_path = OUTPUT_DIR / filename

        plt.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(f"Saved: {output_path}")

    if SHOW_PLOTS:
        plt.show()

    plt.close()


def add_value_labels(
    bars,
    decimals: int = 1,
    suffix: str = "",
) -> None:
    """
    Add numeric labels above or below bars.
    """

    for bar in bars:
        value = bar.get_height()

        vertical_alignment = (
            "bottom"
            if value >= 0
            else "top"
        )

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.{decimals}f}{suffix}",
            ha="center",
            va=vertical_alignment,
        )


# ============================================================
# Load and prepare data
# ============================================================

def load_results() -> pd.DataFrame:
    """
    Load and clean the combined policy result CSV.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found:\n"
            f"{INPUT_FILE.resolve()}"
        )

    df = pd.read_csv(INPUT_FILE)

    print("=" * 70)
    print(f"Loaded file: {INPUT_FILE.resolve()}")
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

    rename_map = {
        REQUEST_COL: "request_id",
        POLICY_COL: "raw_policy",
        LATENCY_COL: "latency_ms",
        IAR_COL: "iar_ms",
    }

    if RUNTIME_COL in df.columns:
        rename_map[RUNTIME_COL] = "runtime_ms"

    if DEADLINE_COL in df.columns:
        rename_map[DEADLINE_COL] = "deadline_ms"

    if ACCURACY_COL in df.columns:
        rename_map[ACCURACY_COL] = "accuracy_value"

    df = df.rename(columns=rename_map)

    # Clean IDs.
    df["request_id"] = (
        df["request_id"]
        .astype(str)
        .str.strip()
    )

    # Clean raw policy names.
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

    # Convert numeric columns.
    numeric_columns = [
        "latency_ms",
        "iar_ms",
    ]

    optional_numeric_columns = [
        "runtime_ms",
        "deadline_ms",
        "accuracy_value",
    ]

    for column in optional_numeric_columns:
        if column in df.columns:
            numeric_columns.append(column)

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    original_rows = len(df)

    df = df.dropna(
        subset=[
            "request_id",
            "policy_name",
            "latency_ms",
            "iar_ms",
        ]
    ).copy()

    removed_rows = original_rows - len(df)

    if removed_rows > 0:
        print(
            f"Removed {removed_rows:,} invalid rows."
        )

    # Keep only the three policies of interest.
    recognised_policies = set(POLICY_ORDER)

    unknown_policies = sorted(
        set(df["policy_name"].unique())
        - recognised_policies
    )

    if unknown_policies:
        print("\nIgnoring unrecognised policies:")

        for policy in unknown_policies:
            print(f"  - {policy}")

    df = df[
        df["policy_name"].isin(recognised_policies)
    ].copy()

    if df.empty:
        raise RuntimeError(
            "No recognised policy rows were found."
        )

    # Remove duplicate measurements for the same request and policy.
    duplicate_mask = df.duplicated(
        subset=[
            "request_id",
            "policy_name",
        ],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        print(
            f"\nWarning: {duplicate_count:,} duplicate "
            "request-policy rows were found."
        )

        print(
            "Keeping the last row for every "
            "request-policy combination."
        )

        df = df.drop_duplicates(
            subset=[
                "request_id",
                "policy_name",
            ],
            keep="last",
        )

    df = add_elapsed_time(df)

    print_dataset_summary(df)

    return df.reset_index(drop=True)


def add_elapsed_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct elapsed workload time using cumulative IAR.

    Each request appears once for each policy. Therefore, IAR
    must be counted only once for each unique request.
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

    # Use numeric request IDs when possible.
    if request_schedule["request_number"].notna().all():
        request_schedule = request_schedule.sort_values(
            "request_number"
        )
    else:
        request_schedule = request_schedule.reset_index(
            drop=True
        )

        request_schedule["request_number"] = np.arange(
            1,
            len(request_schedule) + 1,
        )

    request_schedule["iar_ms"] = (
        request_schedule["iar_ms"]
        .fillna(0)
        .clip(lower=0)
    )

    # The first request is placed at elapsed time zero.
    request_schedule["elapsed_time_sec"] = (
        request_schedule["iar_ms"]
        .cumsum()
        / IAR_DIVISOR
    )

    if not request_schedule.empty:
        first_time = (
            request_schedule["elapsed_time_sec"]
            .iloc[0]
        )

        request_schedule["elapsed_time_sec"] -= first_time

    request_schedule["elapsed_time_min"] = (
        request_schedule["elapsed_time_sec"]
        / 60.0
    )

    request_schedule = request_schedule[
        [
            "request_id",
            "request_number",
            "elapsed_time_sec",
            "elapsed_time_min",
        ]
    ]

    output = output.merge(
        request_schedule,
        on="request_id",
        how="left",
    )

    return output


def print_dataset_summary(df: pd.DataFrame) -> None:
    """
    Print important data validation information.
    """

    print("\nPolicy counts:")

    print(
        df["policy_name"]
        .value_counts()
        .reindex(get_policy_order(df))
    )

    rows_per_request = (
        df.groupby("request_id")
        .size()
    )

    print("\nNumber of policy rows per request:")

    print(
        rows_per_request
        .value_counts()
        .sort_index()
    )

    total_requests = df["request_id"].nunique()

    complete_requests = (
        df.groupby("request_id")["policy_name"]
        .nunique()
        .eq(len(POLICY_ORDER))
        .sum()
    )

    maximum_time = df["elapsed_time_sec"].max()

    print(
        f"\nUnique requests: {total_requests:,}"
    )

    print(
        "Requests containing all three policies: "
        f"{complete_requests:,}/{total_requests:,}"
    )

    print(
        f"Reconstructed workload duration: "
        f"{maximum_time:.2f} seconds "
        f"({maximum_time / 60.0:.2f} minutes)"
    )

    if "runtime_ms" not in df.columns:
        print(
            f"\nNote: '{RUNTIME_COL}' was not found. "
            "Runtime figures will be skipped."
        )


# ============================================================
# Create per-request policy matrix
# ============================================================

def create_complete_latency_matrix(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one row per request and one latency column per policy.

    Only requests with measurements for every available policy
    are retained.
    """

    policy_columns = get_policy_order(df)

    matrix = df.pivot_table(
        index=[
            "request_id",
            "request_number",
            "elapsed_time_sec",
            "elapsed_time_min",
        ],
        columns="policy_name",
        values="latency_ms",
        aggfunc="mean",
    )

    missing_policies = [
        policy
        for policy in policy_columns
        if policy not in matrix.columns
    ]

    if missing_policies:
        raise ValueError(
            f"Missing policies from latency matrix: "
            f"{missing_policies}"
        )

    matrix = (
        matrix[policy_columns]
        .dropna()
        .reset_index()
        .sort_values("elapsed_time_sec")
        .reset_index(drop=True)
    )

    return matrix


def calculate_best_policy(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine the minimum-latency policy for each request.

    This represents an offline oracle. It is not a runtime
    decision made by DYSCI.
    """

    matrix = create_complete_latency_matrix(df)

    policy_columns = get_policy_order(df)

    matrix["best_policy"] = (
        matrix[policy_columns]
        .idxmin(axis=1)
    )

    matrix["best_latency_ms"] = (
        matrix[policy_columns]
        .min(axis=1)
    )

    matrix["worst_latency_ms"] = (
        matrix[policy_columns]
        .max(axis=1)
    )

    matrix["oracle_saving_ms"] = (
        matrix["worst_latency_ms"]
        - matrix["best_latency_ms"]
    )

    if "Sequential" in policy_columns:
        matrix["saving_over_sequential_ms"] = (
            matrix["Sequential"]
            - matrix["best_latency_ms"]
        )

        matrix["saving_over_sequential_percent"] = (
            matrix["saving_over_sequential_ms"]
            / matrix["Sequential"]
            * 100
        )

    return matrix


# ============================================================
# Plot 1: Raw end-to-end latency over time
# ============================================================

def plot_latency_timeline(df: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 5))

    for policy in get_policy_order(df):
        policy_df = (
            df[df["policy_name"] == policy]
            .sort_values("elapsed_time_sec")
        )

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["latency_ms"],
            label=policy,
            linewidth=1.0,
            alpha=0.75,
        )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel("End-to-End Latency (ms)")
    plt.title(
        "Per-Request End-to-End Latency During Azure Trace Replay"
    )
    plt.legend()

    save_and_show("01_latency_timeline.png")


# ============================================================
# Plot 2: Rolling end-to-end latency over time
# ============================================================

def plot_rolling_latency(df: pd.DataFrame) -> None:
    plt.figure(figsize=(13, 5))

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

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["rolling_latency_ms"],
            label=policy,
            linewidth=2,
        )

    plt.xlabel("Workload Time (seconds)")
    plt.ylabel(
        f"Latency (ms)"
        # f"Window = {ROLLING_WINDOW} Requests"
    )
    plt.title(
        "End-to-End Latency - Azure Trace"
    )
    plt.legend()

    save_and_show("02_rolling_latency.png")


# ============================================================
# Plot 3: Runtime over workload time
# ============================================================

def plot_runtime_timeline(df: pd.DataFrame) -> None:
    if "runtime_ms" not in df.columns:
        print(
            "\nSkipped runtime timeline: "
            f"'{RUNTIME_COL}' was not found."
        )
        return

    runtime_df = df.dropna(
        subset=["runtime_ms"]
    )

    if runtime_df.empty:
        print(
            "\nSkipped runtime timeline: "
            "no valid runtime values were found."
        )
        return

    plt.figure(figsize=(13, 5))

    for policy in get_policy_order(runtime_df):
        policy_df = (
            runtime_df[
                runtime_df["policy_name"] == policy
            ]
            .sort_values("elapsed_time_sec")
        )

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["runtime_ms"],
            label=policy,
            linewidth=1,
            alpha=0.75,
        )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel("Request Run Time (ms)")
    plt.title(
        "Per-Request Execution Run Time During Azure Trace Replay"
    )
    plt.legend()

    save_and_show("03_runtime_timeline.png")


# ============================================================
# Plot 4: Rolling runtime over workload time
# ============================================================

def plot_rolling_runtime(df: pd.DataFrame) -> None:
    if "runtime_ms" not in df.columns:
        print(
            "\nSkipped rolling runtime: "
            f"'{RUNTIME_COL}' was not found."
        )
        return

    runtime_df = df.dropna(
        subset=["runtime_ms"]
    )

    if runtime_df.empty:
        print(
            "\nSkipped rolling runtime: "
            "no valid runtime values were found."
        )
        return

    plt.figure(figsize=(13, 5))

    for policy in get_policy_order(runtime_df):
        policy_df = (
            runtime_df[
                runtime_df["policy_name"] == policy
            ]
            .sort_values("elapsed_time_sec")
            .copy()
        )

        policy_df["rolling_runtime_ms"] = (
            policy_df["runtime_ms"]
            .rolling(
                window=ROLLING_WINDOW,
                min_periods=1,
            )
            .mean()
        )

        plt.plot(
            policy_df["elapsed_time_sec"],
            policy_df["rolling_runtime_ms"],
            label=policy,
            linewidth=2,
        )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel(
        f"Rolling Mean Run Time (ms), "
        f"Window = {ROLLING_WINDOW} Requests"
    )
    plt.title(
        "Rolling Execution Run Time During Azure Trace Replay"
    )
    plt.legend()

    save_and_show("04_rolling_runtime.png")


# ============================================================
# Plot 5: End-to-end latency CDF
# ============================================================

def plot_latency_cdf(df: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))

    for policy in get_policy_order(df):
        values = (
            df.loc[
                df["policy_name"] == policy,
                "latency_ms",
            ]
            .dropna()
            .sort_values()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        cdf = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        plt.plot(
            values,
            cdf,
            label=policy,
            linewidth=2,
        )

    plt.xlabel("End-to-End Latency (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("End-to-End Latency CDF")
    plt.legend()

    save_and_show("05_latency_cdf.png")


# ============================================================
# Plot 6: Runtime CDF
# ============================================================

def plot_runtime_cdf(df: pd.DataFrame) -> None:
    if "runtime_ms" not in df.columns:
        print(
            "\nSkipped runtime CDF: "
            f"'{RUNTIME_COL}' was not found."
        )
        return

    plt.figure(figsize=(8, 5))

    plotted = False

    for policy in get_policy_order(df):
        values = (
            df.loc[
                df["policy_name"] == policy,
                "runtime_ms",
            ]
            .dropna()
            .sort_values()
            .to_numpy()
        )

        if len(values) == 0:
            continue

        plotted = True

        cdf = (
            np.arange(1, len(values) + 1)
            / len(values)
        )

        plt.plot(
            values,
            cdf,
            label=policy,
            linewidth=2,
        )

    if not plotted:
        plt.close()
        print(
            "\nSkipped runtime CDF: "
            "no valid runtime values were found."
        )
        return

    plt.xlabel("Request Run Time (ms)")
    plt.ylabel("Cumulative Probability")
    plt.title("Request Execution Run Time CDF")
    plt.legend()

    save_and_show("06_runtime_cdf.png")


# ============================================================
# Plot 7: Offline best-policy timeline
# ============================================================

def plot_best_policy_timeline(
    best_df: pd.DataFrame,
) -> None:
    available_policies = [
        policy
        for policy in POLICY_ORDER
        if policy in best_df["best_policy"].unique()
    ]

    policy_to_code = {
        policy: index
        for index, policy in enumerate(
            available_policies
        )
    }

    plotting_df = best_df.copy()

    plotting_df["policy_code"] = (
        plotting_df["best_policy"]
        .map(policy_to_code)
    )

    plt.figure(figsize=(13, 4))

    plt.scatter(
        plotting_df["elapsed_time_sec"],
        plotting_df["policy_code"],
        s=18,
        alpha=0.8,
    )

    plt.yticks(
        list(policy_to_code.values()),
        list(policy_to_code.keys()),
    )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel("Lowest-Latency Policy")
    plt.title(
        "Offline Lowest-Latency Policy Over Workload Time"
    )

    save_and_show("07_best_policy_timeline.png")


# ============================================================
# Plot 8: Best-policy regions over actual elapsed time
# ============================================================

def plot_best_policy_regions(
    best_df: pd.DataFrame,
) -> None:
    available_policies = [
        policy
        for policy in POLICY_ORDER
        if policy in best_df["best_policy"].unique()
    ]

    policy_to_code = {
        policy: index
        for index, policy in enumerate(
            available_policies
        )
    }

    plotting_df = (
        best_df
        .sort_values("elapsed_time_sec")
        .copy()
    )

    plotting_df["policy_code"] = (
        plotting_df["best_policy"]
        .map(policy_to_code)
    )

    plt.figure(figsize=(14, 3))

    scatter = plt.scatter(
        plotting_df["elapsed_time_sec"],
        np.zeros(len(plotting_df)),
        c=plotting_df["policy_code"],
        marker="s",
        s=30,
        alpha=0.9,
    )

    plt.yticks([])
    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.title(
        "Offline Lowest-Latency Policy Regions Over Time"
    )

    colorbar = plt.colorbar(
        scatter,
        ticks=list(policy_to_code.values()),
    )

    colorbar.ax.set_yticklabels(
        list(policy_to_code.keys())
    )

    save_and_show("08_best_policy_regions.png")


# ============================================================
# Plot 9: Policy win percentage
# ============================================================

def plot_policy_win_percentage(
    best_df: pd.DataFrame,
) -> None:
    available_order = [
        policy
        for policy in POLICY_ORDER
        if policy in best_df["best_policy"].unique()
    ]

    win_counts = (
        best_df["best_policy"]
        .value_counts()
        .reindex(available_order)
        .fillna(0)
    )

    percentages = (
        win_counts
        / win_counts.sum()
        * 100
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        percentages.index,
        percentages.values,
    )

    add_value_labels(
        bars,
        decimals=1,
        suffix="%",
    )

    plt.xlabel("Policy")
    plt.ylabel("Requests with Lowest Latency (%)")
    plt.title("Per-Request Policy Win Rate")
    plt.xticks(rotation=12)

    maximum = percentages.max()

    plt.ylim(
        0,
        maximum * 1.15
        if maximum > 0
        else 1,
    )

    save_and_show("09_policy_win_percentage.png")


# ============================================================
# Plot 10: End-to-end latency summary
# ============================================================

def calculate_latency_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary_rows = []

    for policy in get_policy_order(df):
        values = df.loc[
            df["policy_name"] == policy,
            "latency_ms",
        ].dropna()

        summary_rows.append(
            {
                "Policy": policy,
                "Mean": values.mean(),
                "Median": values.median(),
                "P95": values.quantile(0.95),
                "P99": values.quantile(0.99),
            }
        )

    return (
        pd.DataFrame(summary_rows)
        .set_index("Policy")
    )


def plot_latency_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    summary = calculate_latency_summary(df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        OUTPUT_DIR / "latency_summary.csv"
    )

    summary.plot(
        kind="bar",
        figsize=(10, 5),
    )

    plt.xlabel("Policy")
    plt.ylabel("End-to-End Latency (ms)")
    plt.title("Average and Tail End-to-End Latency")
    plt.xticks(rotation=12)
    plt.legend(title="Metric")

    save_and_show("10_latency_summary.png")

    print("\nEnd-to-end latency summary:")
    print(summary.round(2))

    return summary


# ============================================================
# Plot 11: Runtime summary
# ============================================================

def plot_runtime_summary(
    df: pd.DataFrame,
) -> pd.DataFrame | None:
    if "runtime_ms" not in df.columns:
        print(
            "\nSkipped runtime summary: "
            f"'{RUNTIME_COL}' was not found."
        )
        return None

    summary_rows = []

    for policy in get_policy_order(df):
        values = df.loc[
            df["policy_name"] == policy,
            "runtime_ms",
        ].dropna()

        if values.empty:
            continue

        summary_rows.append(
            {
                "Policy": policy,
                "Mean": values.mean(),
                "Median": values.median(),
                "P95": values.quantile(0.95),
                "P99": values.quantile(0.99),
            }
        )

    if not summary_rows:
        print(
            "\nSkipped runtime summary: "
            "no valid runtime values were found."
        )
        return None

    summary = (
        pd.DataFrame(summary_rows)
        .set_index("Policy")
    )

    summary.to_csv(
        OUTPUT_DIR / "runtime_summary.csv"
    )

    summary.plot(
        kind="bar",
        figsize=(10, 5),
    )

    plt.xlabel("Policy")
    plt.ylabel("Request Run Time (ms)")
    plt.title("Average and Tail Execution Run Time")
    plt.xticks(rotation=12)
    plt.legend(title="Metric")

    save_and_show("11_runtime_summary.png")

    print("\nRuntime summary:")
    print(summary.round(2))

    return summary


# ============================================================
# Plot 12: Deadline success
# ============================================================

def plot_deadline_success(
    df: pd.DataFrame,
) -> None:
    if "deadline_ms" not in df.columns:
        print(
            "\nSkipped deadline success: "
            f"'{DEADLINE_COL}' was not found."
        )
        return

    working = df.dropna(
        subset=[
            "latency_ms",
            "deadline_ms",
        ]
    ).copy()

    if working.empty:
        print(
            "\nSkipped deadline success: "
            "no valid deadline values were found."
        )
        return

    working["deadline_met"] = (
        working["latency_ms"]
        <= working["deadline_ms"]
    )

    success_rate = (
        working.groupby("policy_name")[
            "deadline_met"
        ]
        .mean()
        .mul(100)
        .reindex(get_policy_order(working))
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        success_rate.index,
        success_rate.values,
    )

    add_value_labels(
        bars,
        decimals=1,
        suffix="%",
    )

    plt.xlabel("Policy")
    plt.ylabel("Deadline Success Rate (%)")
    plt.title("Deadline Success Rate")
    plt.xticks(rotation=12)
    plt.ylim(0, 105)

    save_and_show("12_deadline_success.png")


# ============================================================
# Plot 13: Improvement over Sequential
# ============================================================

def plot_improvement_over_sequential(
    df: pd.DataFrame,
) -> None:
    mean_latency = (
        df.groupby("policy_name")[
            "latency_ms"
        ]
        .mean()
        .reindex(get_policy_order(df))
    )

    if "Sequential" not in mean_latency.index:
        print(
            "\nSkipped improvement graph: "
            "Sequential results were not found."
        )
        return

    sequential_mean = mean_latency["Sequential"]

    improvement = (
        (
            sequential_mean
            - mean_latency
        )
        / sequential_mean
        * 100
    )

    improvement = improvement.drop(
        labels=["Sequential"],
        errors="ignore",
    )

    plt.figure(figsize=(8, 5))

    bars = plt.bar(
        improvement.index,
        improvement.values,
    )

    plt.axhline(
        0,
        linewidth=1,
    )

    add_value_labels(
        bars,
        decimals=1,
        suffix="%",
    )

    plt.xlabel("Policy")
    plt.ylabel(
        "Mean End-to-End Latency Reduction (%)"
    )
    plt.title(
        "Latency Reduction Relative to Sequential"
    )
    plt.xticks(rotation=12)

    save_and_show(
        "13_improvement_over_sequential.png"
    )


# ============================================================
# Plot 14: Offline oracle versus fixed policies
# ============================================================

def plot_oracle_latency(
    df: pd.DataFrame,
    best_df: pd.DataFrame,
) -> None:
    fixed_means = (
        df.groupby("policy_name")[
            "latency_ms"
        ]
        .mean()
        .reindex(get_policy_order(df))
    )

    comparison = fixed_means.copy()

    comparison.loc["Offline Oracle"] = (
        best_df["best_latency_ms"].mean()
    )

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        comparison.index,
        comparison.values,
    )

    add_value_labels(
        bars,
        decimals=1,
    )

    plt.xlabel("Policy")
    plt.ylabel("Mean End-to-End Latency (ms)")
    plt.title(
        "Fixed Policies Compared with the Offline Oracle"
    )
    plt.xticks(rotation=12)

    save_and_show(
        "14_oracle_vs_fixed_policy_latency.png"
    )


# ============================================================
# Plot 15: Oracle saving over workload time
# ============================================================

def plot_oracle_saving(
    best_df: pd.DataFrame,
) -> None:
    plotting_df = (
        best_df
        .sort_values("elapsed_time_sec")
        .copy()
    )

    plotting_df["rolling_oracle_saving_ms"] = (
        plotting_df["oracle_saving_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    plt.figure(figsize=(13, 5))

    plt.plot(
        plotting_df["elapsed_time_sec"],
        plotting_df["rolling_oracle_saving_ms"],
        linewidth=2,
    )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel(
        "Rolling Gap Between Worst and Best Policy (ms)"
    )
    plt.title(
        "Potential Benefit of Per-Request Policy Selection"
    )

    save_and_show(
        "15_oracle_saving_timeline.png"
    )


# ============================================================
# Plot 16: Arrival rate over workload time
# ============================================================

def plot_arrival_rate(
    df: pd.DataFrame,
    time_window_seconds: float = 10.0,
) -> None:
    """
    Plot request arrival rate using fixed elapsed-time windows.

    Each request is counted only once, not once per policy.
    """

    request_schedule = (
        df[
            [
                "request_id",
                "elapsed_time_sec",
            ]
        ]
        .drop_duplicates(
            subset=["request_id"]
        )
        .sort_values("elapsed_time_sec")
        .copy()
    )

    if request_schedule.empty:
        print(
            "\nSkipped arrival-rate graph: "
            "no request schedule was found."
        )
        return

    maximum_time = (
        request_schedule["elapsed_time_sec"]
        .max()
    )

    if maximum_time <= 0:
        print(
            "\nSkipped arrival-rate graph: "
            "workload duration is zero."
        )
        return

    bins = np.arange(
        0,
        maximum_time + time_window_seconds,
        time_window_seconds,
    )

    if len(bins) < 2:
        bins = np.array(
            [
                0,
                maximum_time + time_window_seconds,
            ]
        )

    counts, edges = np.histogram(
        request_schedule["elapsed_time_sec"],
        bins=bins,
    )

    arrival_rate = (
        counts
        / time_window_seconds
    )

    bin_centres = (
        edges[:-1]
        + edges[1:]
    ) / 2

    plt.figure(figsize=(13, 4))

    plt.step(
        bin_centres,
        arrival_rate,
        where="mid",
        linewidth=2,
    )

    plt.xlabel("Elapsed Workload Time (seconds)")
    plt.ylabel("Arrival Rate (requests/second)")
    plt.title(
        f"Azure Workload Arrival Rate "
        f"({time_window_seconds:g}-Second Windows)"
    )

    save_and_show("16_arrival_rate_timeline.png")


# ============================================================
# Plot 17: Runtime versus end-to-end latency
# ============================================================

def plot_runtime_vs_latency(
    df: pd.DataFrame,
) -> None:
    if "runtime_ms" not in df.columns:
        print(
            "\nSkipped runtime-versus-latency plot: "
            f"'{RUNTIME_COL}' was not found."
        )
        return

    working = df.dropna(
        subset=[
            "runtime_ms",
            "latency_ms",
        ]
    )

    if working.empty:
        print(
            "\nSkipped runtime-versus-latency plot: "
            "no valid data were found."
        )
        return

    plt.figure(figsize=(8, 6))

    for policy in get_policy_order(working):
        policy_df = working[
            working["policy_name"] == policy
        ]

        plt.scatter(
            policy_df["runtime_ms"],
            policy_df["latency_ms"],
            label=policy,
            s=22,
            alpha=0.6,
        )

    plt.xlabel("Request Run Time (ms)")
    plt.ylabel("End-to-End Latency (ms)")
    plt.title(
        "Execution Run Time versus End-to-End Latency"
    )
    plt.legend()

    save_and_show(
        "17_runtime_vs_e2e_latency.png"
    )


# ============================================================
# Optional accuracy/confidence-latency graph
# ============================================================

def plot_accuracy_latency(
    df: pd.DataFrame,
) -> None:
    if not PLOT_ACCURACY_LATENCY:
        print(
            "\nSkipped accuracy-latency graph because "
            "PLOT_ACCURACY_LATENCY=False."
        )
        return

    if "accuracy_value" not in df.columns:
        print(
            "\nSkipped accuracy-latency graph: "
            f"'{ACCURACY_COL}' was not found."
        )
        return

    summary = (
        df.groupby("policy_name")
        .agg(
            mean_latency_ms=(
                "latency_ms",
                "mean",
            ),
            mean_accuracy_value=(
                "accuracy_value",
                "mean",
            ),
        )
        .reindex(get_policy_order(df))
        .reset_index()
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        summary["mean_latency_ms"],
        summary["mean_accuracy_value"],
        s=120,
    )

    for _, row in summary.iterrows():
        plt.annotate(
            row["policy_name"],
            (
                row["mean_latency_ms"],
                row["mean_accuracy_value"],
            ),
            xytext=(7, 7),
            textcoords="offset points",
        )

    plt.xlabel("Mean End-to-End Latency (ms)")
    plt.ylabel("Mean Accuracy or Confidence")
    plt.title("Accuracy/Confidence–Latency Trade-off")

    save_and_show(
        "18_accuracy_latency_tradeoff.png"
    )


# ============================================================
# Main function
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_results()

    # Save the cleaned combined data.
    cleaned_output = (
        OUTPUT_DIR
        / "cleaned_combined_results.csv"
    )

    df.to_csv(
        cleaned_output,
        index=False,
    )

    print(
        f"\nSaved cleaned data: {cleaned_output}"
    )

    best_df = calculate_best_policy(df)

    best_output = (
        OUTPUT_DIR
        / "best_policy_per_request.csv"
    )

    best_df.to_csv(
        best_output,
        index=False,
    )

    print(
        f"Saved offline best-policy data: "
        f"{best_output}"
    )

    # Time-based figures
    plot_latency_timeline(df)
    plot_rolling_latency(df)
    plot_runtime_timeline(df)
    plot_rolling_runtime(df)

    # Distribution figures
    plot_latency_cdf(df)
    plot_runtime_cdf(df)

    # Policy-selection figures
    plot_best_policy_timeline(best_df)
    plot_best_policy_regions(best_df)
    plot_policy_win_percentage(best_df)

    # Summary figures
    plot_latency_summary(df)
    plot_runtime_summary(df)
    plot_deadline_success(df)
    plot_improvement_over_sequential(df)

    # Oracle figures
    plot_oracle_latency(df, best_df)
    plot_oracle_saving(best_df)

    # Workload and latency breakdown figures
    plot_arrival_rate(
        df,
        time_window_seconds=10.0,
    )

    plot_runtime_vs_latency(df)

    # Optional accuracy/confidence graph
    plot_accuracy_latency(df)

    print("\n" + "=" * 70)
    print("Finished generating all available figures.")
    print(f"Output directory: {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()