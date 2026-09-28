


from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# Change only this section
# ============================================================

INPUT_CSV = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/azure_workload.csv"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load/data/graph"
)


# Columns in your result CSV
IAR_COLUMN = "IAR"
LATENCY_COLUMN = "e2e_time_ms"
POLICY_COLUMN = "effective_cascade_mode"
SUCCESS_COLUMN = "Success"

# Your IAR column is likely stored in milliseconds.
# Use "milliseconds" or "seconds".
IAR_UNIT = "milliseconds"

# Aggregate requests into windows after idle-gap compression.
WINDOW_SECONDS = 10.0

# Long idle gaps are compressed to this value.
#
# Example:
# Original gap = 4 hours
# Compressed gap = 10 seconds
#
# Use None to preserve the complete original timing.
MAX_IDLE_GAP_SECONDS = 10.0

# Remove unsuccessful requests from latency calculations.
ONLY_SUCCESSFUL_REQUESTS = True

# Minimum number of requests required in a window before
# calculating P95 latency.
MIN_REQUESTS_FOR_P95 = 2

# Optional: plot only the first N requests.
# Use None to include all requests.
MAX_REQUESTS = None

# Optional smoothing for plotted latency values.
# 1 means no smoothing.
SMOOTHING_WINDOWS = 3

# ============================================================


POLICY_DISPLAY_NAMES = {
    "sequential": "Sequential",
    "seq": "Sequential",

    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "parallel_first_finish": "Edge Speculation",
    "edge_speculation": "Edge Speculation",
    "edge speculation": "Edge Speculation",

    "spec_b": "Cloud Speculation",
    "cloud_speculation": "Cloud Speculation",
    "cloud speculation": "Cloud Speculation",

    "full_parallel": "Full Parallel",
    "parallel": "Full Parallel",

    "dynamic": "DYSCI",
    "dysci": "DYSCI",
}


POLICY_NUMBERS = {
    "Sequential": 0,
    "Edge Speculation": 1,
    "Cloud Speculation": 2,
    "Full Parallel": 3,
    "DYSCI": 4,
    "Unknown": 5,
}


def normalize_policy(value) -> str:
    """
    Convert policy values into readable names.
    """

    if pd.isna(value):
        return "Unknown"

    text = str(value).strip()

    if not text:
        return "Unknown"

    normalized = text.lower()

    return POLICY_DISPLAY_NAMES.get(
        normalized,
        text,
    )


def parse_success_column(series: pd.Series) -> pd.Series:
    """
    Convert different success formats to Boolean values.
    """

    if series.dtype == bool:
        return series

    normalized = (
        series.astype(str)
        .str.strip()
        .str.lower()
    )

    return normalized.isin(
        {
            "true",
            "1",
            "yes",
            "success",
            "successful",
        }
    )


def load_results() -> pd.DataFrame:
    """
    Read and clean the Azure replay result file.
    """

    input_path = Path(INPUT_CSV)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input CSV was not found:\n{input_path}"
        )

    df = pd.read_csv(input_path)

    print("\nAvailable columns:")
    print(list(df.columns))

    required_columns = [
        IAR_COLUMN,
        LATENCY_COLUMN,
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"\nMissing required column: {column}\n"
                f"Available columns:\n{list(df.columns)}"
            )

    if MAX_REQUESTS is not None:
        df = df.head(MAX_REQUESTS).copy()

    df[IAR_COLUMN] = pd.to_numeric(
        df[IAR_COLUMN],
        errors="coerce",
    )

    df[LATENCY_COLUMN] = pd.to_numeric(
        df[LATENCY_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            IAR_COLUMN,
            LATENCY_COLUMN,
        ]
    ).copy()

    df = df[
        (df[IAR_COLUMN] >= 0)
        & (df[LATENCY_COLUMN] >= 0)
    ].copy()

    if df.empty:
        raise ValueError(
            "No valid rows remain after cleaning the CSV."
        )

    # --------------------------------------------------------
    # Convert IAR to seconds
    # --------------------------------------------------------

    if IAR_UNIT == "milliseconds":
        df["original_iar_sec"] = (
            df[IAR_COLUMN] / 1000.0
        )

    elif IAR_UNIT == "seconds":
        df["original_iar_sec"] = df[IAR_COLUMN]

    else:
        raise ValueError(
            'IAR_UNIT must be "milliseconds" or "seconds".'
        )

    # --------------------------------------------------------
    # Compress long idle periods
    # --------------------------------------------------------

    if MAX_IDLE_GAP_SECONDS is None:
        df["replay_iar_sec"] = df["original_iar_sec"]

    else:
        df["replay_iar_sec"] = np.minimum(
            df["original_iar_sec"],
            MAX_IDLE_GAP_SECONDS,
        )

    # --------------------------------------------------------
    # Reconstruct elapsed time
    # --------------------------------------------------------

    df["original_elapsed_sec"] = (
        df["original_iar_sec"].cumsum()
    )

    df["compressed_elapsed_sec"] = (
        df["replay_iar_sec"].cumsum()
    )

    df["original_elapsed_sec"] = (
        df["original_elapsed_sec"]
        - df["original_elapsed_sec"].iloc[0]
    )

    df["compressed_elapsed_sec"] = (
        df["compressed_elapsed_sec"]
        - df["compressed_elapsed_sec"].iloc[0]
    )

    df["latency_ms"] = df[LATENCY_COLUMN]

    # --------------------------------------------------------
    # Process success values
    # --------------------------------------------------------

    if SUCCESS_COLUMN in df.columns:
        df["is_success"] = parse_success_column(
            df[SUCCESS_COLUMN]
        )
    else:
        df["is_success"] = True

    # --------------------------------------------------------
    # Process policy names
    # --------------------------------------------------------

    if POLICY_COLUMN in df.columns:
        df["policy"] = df[POLICY_COLUMN].apply(
            normalize_policy
        )
    else:
        print(
            f"\nWarning: '{POLICY_COLUMN}' was not found. "
            "Policy graph will show Unknown."
        )
        df["policy"] = "Unknown"

    df = df.reset_index(drop=True)

    return df


def print_trace_information(df: pd.DataFrame) -> None:
    """
    Print information about the original and compressed trace.
    """

    original_duration = (
        df["original_elapsed_sec"].max()
    )

    compressed_duration = (
        df["compressed_elapsed_sec"].max()
    )

    original_long_gaps = 0

    if MAX_IDLE_GAP_SECONDS is not None:
        original_long_gaps = int(
            (
                df["original_iar_sec"]
                > MAX_IDLE_GAP_SECONDS
            ).sum()
        )

    print("\nAzure trace information")
    print("-----------------------")
    print(f"Requests: {len(df)}")

    print(
        "Original duration: "
        f"{original_duration:.2f} seconds "
        f"({original_duration / 3600:.2f} hours)"
    )

    print(
        "Compressed duration: "
        f"{compressed_duration:.2f} seconds "
        f"({compressed_duration / 60:.2f} minutes)"
    )

    if MAX_IDLE_GAP_SECONDS is not None:
        print(
            "Maximum retained idle gap: "
            f"{MAX_IDLE_GAP_SECONDS:.2f} seconds"
        )

        print(
            "Number of compressed idle gaps: "
            f"{original_long_gaps}"
        )

    print(
        "Median original IAR: "
        f"{df['original_iar_sec'].median():.6f} seconds"
    )

    print(
        "P95 original IAR: "
        f"{np.percentile(df['original_iar_sec'], 95):.6f} seconds"
    )

    print(
        "Maximum original IAR: "
        f"{df['original_iar_sec'].max():.2f} seconds"
    )


def calculate_window_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate workload and latency metrics for each time window.
    """

    work = df.copy()

    work["window_start_sec"] = (
        np.floor(
            work["compressed_elapsed_sec"]
            / WINDOW_SECONDS
        )
        * WINDOW_SECONDS
    )

    # Workload uses every request.
    workload_metrics = (
        work.groupby("window_start_sec")
        .agg(
            request_count=(
                "latency_ms",
                "size",
            ),
            successful_requests=(
                "is_success",
                "sum",
            ),
        )
        .reset_index()
    )

    workload_metrics["arrival_rate_rps"] = (
        workload_metrics["request_count"]
        / WINDOW_SECONDS
    )

    workload_metrics["success_rate_percent"] = (
        workload_metrics["successful_requests"]
        / workload_metrics["request_count"]
        * 100.0
    )

    # Latency can optionally use successful requests only.
    if ONLY_SUCCESSFUL_REQUESTS:
        latency_df = work[
            work["is_success"]
        ].copy()
    else:
        latency_df = work.copy()

    latency_metrics = (
        latency_df.groupby("window_start_sec")
        .agg(
            latency_request_count=(
                "latency_ms",
                "size",
            ),
            mean_latency_ms=(
                "latency_ms",
                "mean",
            ),
            median_latency_ms=(
                "latency_ms",
                "median",
            ),
            p95_latency_ms=(
                "latency_ms",
                lambda values: (
                    float(np.percentile(values, 95))
                    if len(values) >= MIN_REQUESTS_FOR_P95
                    else np.nan
                ),
            ),
            p99_latency_ms=(
                "latency_ms",
                lambda values: (
                    float(np.percentile(values, 99))
                    if len(values) >= MIN_REQUESTS_FOR_P95
                    else np.nan
                ),
            ),
        )
        .reset_index()
    )

    metrics = workload_metrics.merge(
        latency_metrics,
        on="window_start_sec",
        how="left",
    )

    metrics["window_center_sec"] = (
        metrics["window_start_sec"]
        + WINDOW_SECONDS / 2.0
    )

    # Create all missing windows so idle periods are visible.
    maximum_time = metrics[
        "window_start_sec"
    ].max()

    complete_windows = pd.DataFrame(
        {
            "window_start_sec": np.arange(
                0,
                maximum_time + WINDOW_SECONDS,
                WINDOW_SECONDS,
            )
        }
    )

    metrics = complete_windows.merge(
        metrics,
        on="window_start_sec",
        how="left",
    )

    zero_fill_columns = [
        "request_count",
        "successful_requests",
        "arrival_rate_rps",
        "success_rate_percent",
        "latency_request_count",
    ]

    for column in zero_fill_columns:
        if column in metrics.columns:
            metrics[column] = (
                metrics[column]
                .fillna(0)
            )

    metrics["window_center_sec"] = (
        metrics["window_start_sec"]
        + WINDOW_SECONDS / 2.0
    )

    # Smooth only the displayed latency.
    smoothing = max(1, SMOOTHING_WINDOWS)

    metrics["display_mean_latency_ms"] = (
        metrics["mean_latency_ms"]
        .rolling(
            window=smoothing,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    metrics["display_p95_latency_ms"] = (
        metrics["p95_latency_ms"]
        .rolling(
            window=smoothing,
            center=True,
            min_periods=1,
        )
        .mean()
    )

    return metrics


def determine_dominant_policy(
    values: pd.Series,
) -> str:
    """
    Return the policy used most frequently in a time window.
    """

    valid = values.dropna()

    if valid.empty:
        return "Unknown"

    mode = valid.mode()

    if mode.empty:
        return str(valid.iloc[0])

    return str(mode.iloc[0])


def calculate_policy_windows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Determine the most common selected policy in each window.
    """

    work = df.copy()

    work["window_start_sec"] = (
        np.floor(
            work["compressed_elapsed_sec"]
            / WINDOW_SECONDS
        )
        * WINDOW_SECONDS
    )

    policy_windows = (
        work.groupby("window_start_sec")
        .agg(
            dominant_policy=(
                "policy",
                determine_dominant_policy,
            ),
            policy_request_count=(
                "policy",
                "size",
            ),
        )
        .reset_index()
    )

    policy_windows["window_center_sec"] = (
        policy_windows["window_start_sec"]
        + WINDOW_SECONDS / 2.0
    )

    policy_windows["policy_number"] = (
        policy_windows["dominant_policy"]
        .map(POLICY_NUMBERS)
        .fillna(POLICY_NUMBERS["Unknown"])
    )

    return policy_windows


def plot_azure_workload(
    metrics: pd.DataFrame,
    policy_windows: pd.DataFrame,
) -> None:
    """
    Create a three-panel paper figure.
    """

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 8),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.0,
                1.5,
                0.55,
            ]
        },
    )

    workload_ax = axes[0]
    latency_ax = axes[1]
    policy_ax = axes[2]

    # ========================================================
    # Panel 1: Workload pattern
    # ========================================================

    workload_ax.step(
        metrics["window_center_sec"],
        metrics["arrival_rate_rps"],
        where="mid",
        linewidth=1.7,
    )

    workload_ax.fill_between(
        metrics["window_center_sec"],
        metrics["arrival_rate_rps"],
        step="mid",
        alpha=0.15,
    )

    workload_ax.set_ylabel(
        "Arrival rate\n(req/s)"
    )

    workload_ax.set_title(
        "DYSCI under an Azure Functions production workload"
    )

    workload_ax.grid(
        True,
        alpha=0.3,
    )

    # ========================================================
    # Panel 2: Latency
    # ========================================================

    latency_ax.plot(
        metrics["window_center_sec"],
        metrics["display_mean_latency_ms"],
        linewidth=1.8,
        label="Mean latency",
    )

    latency_ax.plot(
        metrics["window_center_sec"],
        metrics["display_p95_latency_ms"],
        linewidth=1.8,
        linestyle="--",
        label="P95 latency",
    )

    latency_ax.set_ylabel(
        "Latency (ms)"
    )

    latency_ax.grid(
        True,
        alpha=0.3,
    )

    latency_ax.legend(
        frameon=False,
        loc="upper left",
        ncol=2,
    )

    # ========================================================
    # Panel 3: Selected policy
    # ========================================================

    policy_ax.step(
        policy_windows["window_center_sec"],
        policy_windows["policy_number"],
        where="mid",
        linewidth=1.8,
    )

    used_policies = [
        policy
        for policy in POLICY_NUMBERS
        if policy in set(
            policy_windows["dominant_policy"]
        )
    ]

    used_numbers = [
        POLICY_NUMBERS[policy]
        for policy in used_policies
    ]

    policy_ax.set_yticks(
        used_numbers
    )

    policy_ax.set_yticklabels(
        used_policies
    )

    policy_ax.set_ylabel(
        "Selected\npolicy"
    )

    policy_ax.set_xlabel(
        "Compressed Azure trace time (seconds)"
    )

    policy_ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    policy_ax.set_ylim(
        min(used_numbers, default=0) - 0.5,
        max(used_numbers, default=1) + 0.5,
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / "azure_workload_latency_policy.png"
    )

    pdf_path = (
        output_dir
        / "azure_workload_latency_policy.pdf"
    )

    fig.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print("\nGraph saved successfully:")
    print(png_path)
    print(pdf_path)


def save_processed_data(
    df: pd.DataFrame,
    metrics: pd.DataFrame,
    policy_windows: pd.DataFrame,
) -> None:
    """
    Save processed trace data and window metrics.
    """

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_trace_path = (
        output_dir
        / "azure_compressed_trace.csv"
    )

    metrics_path = (
        output_dir
        / "azure_window_metrics.csv"
    )

    policy_path = (
        output_dir
        / "azure_policy_windows.csv"
    )

    processed_columns = [
        IAR_COLUMN,
        "original_iar_sec",
        "replay_iar_sec",
        "original_elapsed_sec",
        "compressed_elapsed_sec",
        "latency_ms",
        "is_success",
        "policy",
    ]

    available_columns = [
        column
        for column in processed_columns
        if column in df.columns
    ]

    df[available_columns].to_csv(
        processed_trace_path,
        index=False,
    )

    metrics.to_csv(
        metrics_path,
        index=False,
    )

    policy_windows.to_csv(
        policy_path,
        index=False,
    )

    print("\nProcessed data saved:")
    print(processed_trace_path)
    print(metrics_path)
    print(policy_path)


def print_latency_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print overall latency and policy information.
    """

    if ONLY_SUCCESSFUL_REQUESTS:
        latency_df = df[
            df["is_success"]
        ]
    else:
        latency_df = df

    latency = latency_df[
        "latency_ms"
    ].dropna()

    if latency.empty:
        print(
            "\nNo successful latency values are available."
        )
        return

    print("\nLatency summary")
    print("---------------")

    print(
        f"Successful requests: "
        f"{int(df['is_success'].sum())}/{len(df)}"
    )

    print(
        f"Success rate: "
        f"{df['is_success'].mean() * 100:.2f}%"
    )

    print(
        f"Mean latency: "
        f"{latency.mean():.2f} ms"
    )

    print(
        f"Median latency: "
        f"{latency.median():.2f} ms"
    )

    print(
        f"P95 latency: "
        f"{np.percentile(latency, 95):.2f} ms"
    )

    print(
        f"P99 latency: "
        f"{np.percentile(latency, 99):.2f} ms"
    )

    print("\nPolicy counts")
    print("-------------")

    print(
        df["policy"]
        .value_counts()
        .to_string()
    )


def main() -> None:
    print(
        "Reading Azure replay results..."
    )

    df = load_results()

    print_trace_information(df)

    print_latency_summary(df)

    metrics = calculate_window_metrics(df)

    policy_windows = calculate_policy_windows(df)

    save_processed_data(
        df,
        metrics,
        policy_windows,
    )

    plot_azure_workload(
        metrics,
        policy_windows,
    )


if __name__ == "__main__":
    main()

