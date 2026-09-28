from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# Change only this section
# ============================================================

AZURE_TRACE_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "selected_15min_workload"
)

# Length of workload segment to extract.
WINDOW_DURATION_MINUTES = 15

# Small sub-window used to measure workload variation.
# For example, the 15-minute interval is divided into
# 10-second workload-rate bins.
RATE_BIN_SECONDS = 10

# Set to None to automatically evaluate all applications.
#
# To analyze a specific Azure application, enter its name:
# SELECTED_APP = "app123"
SELECTED_APP = None

# Minimum number of requests required in a 15-minute segment.
# Increase this when you need a heavier workload.
MIN_REQUESTS_IN_WINDOW = 100

# Maximum number of applications to examine.
# Applications are first ranked by total invocation count.
#
# Use None to analyze every application, but this can take longer.
MAX_APPS_TO_ANALYZE = 30

# Type of workload segment to select:
#
# "dynamic" -> changing arrival rate with bursts and quiet periods
# "high"    -> highest request volume
# "medium"  -> medium request volume
# "low"     -> low request volume, but not completely idle
SELECTION_MODE = "dynamic"

# Azure timestamps and durations must use compatible units.
#
# Use:
# "seconds"
# "milliseconds"
# "microseconds"
#
# Start with "seconds" because your earlier extracted IATs
# appeared to be measured in seconds.
TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Some Azure records can produce identical start timestamps.
# Keep zero-IAT requests because they represent simultaneous arrivals.
KEEP_ZERO_IAT_REQUESTS = True

# ============================================================


def unit_divisor(unit: str) -> float:
    """
    Return the divisor needed to convert a value to seconds.
    """

    normalized = unit.strip().lower()

    if normalized == "seconds":
        return 1.0

    if normalized == "milliseconds":
        return 1_000.0

    if normalized == "microseconds":
        return 1_000_000.0

    raise ValueError(
        "Unit must be 'seconds', 'milliseconds', or 'microseconds'."
    )


def load_azure_trace() -> pd.DataFrame:
    """
    Read the Azure Functions trace and reconstruct request start times.
    """

    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace was not found:\n{trace_path}"
        )

    print("Reading Azure trace...")
    print(trace_path)

    # First try a normal CSV reader.
    df = pd.read_csv(trace_path)

    # Some copies of this dataset may be tab-separated or whitespace-separated.
    if len(df.columns) == 1:
        df = pd.read_csv(
            trace_path,
            sep=r"\s+",
            engine="python",
        )

    print("\nAvailable columns:")
    print(list(df.columns))

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"\nMissing required column: {column}\n"
                f"Available columns: {list(df.columns)}"
            )

    df["end_timestamp"] = pd.to_numeric(
        df["end_timestamp"],
        errors="coerce",
    )

    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "app",
            "func",
            "end_timestamp",
            "duration",
        ]
    ).copy()

    df = df[
        (df["end_timestamp"] >= 0)
        & (df["duration"] >= 0)
    ].copy()

    timestamp_divisor = unit_divisor(TIMESTAMP_UNIT)
    duration_divisor = unit_divisor(DURATION_UNIT)

    df["end_time_sec"] = (
        df["end_timestamp"] / timestamp_divisor
    )

    df["duration_sec"] = (
        df["duration"] / duration_divisor
    )

    # Reconstruct invocation start time.
    df["start_time_sec"] = (
        df["end_time_sec"] - df["duration_sec"]
    )

    df = df.sort_values(
        ["app", "start_time_sec"]
    ).reset_index(drop=True)

    print(f"\nValid trace records: {len(df):,}")
    print(f"Unique applications: {df['app'].nunique():,}")
    print(f"Unique functions: {df['func'].nunique():,}")

    return df


def get_candidate_apps(
    df: pd.DataFrame,
) -> list:
    """
    Return the applications that should be searched.
    """

    if SELECTED_APP is not None:
        if SELECTED_APP not in set(df["app"]):
            raise ValueError(
                f"Application '{SELECTED_APP}' was not found."
            )

        return [SELECTED_APP]

    app_counts = (
        df.groupby("app")
        .size()
        .sort_values(ascending=False)
    )

    if MAX_APPS_TO_ANALYZE is not None:
        app_counts = app_counts.head(
            MAX_APPS_TO_ANALYZE
        )

    print("\nApplications considered:")
    print(app_counts.to_string())

    return app_counts.index.tolist()


def calculate_window_statistics(
    app_df: pd.DataFrame,
    app_name: str,
) -> pd.DataFrame:
    """
    Calculate statistics for every contiguous 15-minute window.

    Windows advance one minute at a time.
    """

    window_seconds = (
        WINDOW_DURATION_MINUTES * 60
    )

    start_min = app_df["start_time_sec"].min()
    start_max = app_df["start_time_sec"].max()

    if start_max - start_min < window_seconds:
        return pd.DataFrame()

    # Search every one minute.
    candidate_starts = np.arange(
        start_min,
        start_max - window_seconds + 1,
        60.0,
    )

    start_times = (
        app_df["start_time_sec"]
        .sort_values()
        .to_numpy()
    )

    results = []

    for window_start in candidate_starts:
        window_end = (
            window_start + window_seconds
        )

        left_index = np.searchsorted(
            start_times,
            window_start,
            side="left",
        )

        right_index = np.searchsorted(
            start_times,
            window_end,
            side="left",
        )

        request_count = (
            right_index - left_index
        )

        if request_count < MIN_REQUESTS_IN_WINDOW:
            continue

        window_times = start_times[
            left_index:right_index
        ]

        relative_times = (
            window_times - window_start
        )

        rate_edges = np.arange(
            0,
            window_seconds + RATE_BIN_SECONDS,
            RATE_BIN_SECONDS,
        )

        bin_counts, _ = np.histogram(
            relative_times,
            bins=rate_edges,
        )

        request_rates = (
            bin_counts / RATE_BIN_SECONDS
        )

        mean_rate = float(
            np.mean(request_rates)
        )

        standard_deviation = float(
            np.std(request_rates)
        )

        rate_cv = (
            standard_deviation / mean_rate
            if mean_rate > 0
            else 0.0
        )

        peak_rate = float(
            np.max(request_rates)
        )

        median_rate = float(
            np.median(request_rates)
        )

        nonzero_rates = request_rates[
            request_rates > 0
        ]

        active_bin_fraction = float(
            np.mean(request_rates > 0)
        )

        idle_bin_fraction = float(
            np.mean(request_rates == 0)
        )

        burst_ratio = (
            peak_rate / mean_rate
            if mean_rate > 0
            else 0.0
        )

        if len(window_times) > 1:
            iats = np.diff(window_times)

            mean_iat = float(
                np.mean(iats)
            )

            median_iat = float(
                np.median(iats)
            )

            p95_iat = float(
                np.percentile(iats, 95)
            )

            iat_cv = (
                float(np.std(iats) / mean_iat)
                if mean_iat > 0
                else 0.0
            )
        else:
            mean_iat = np.nan
            median_iat = np.nan
            p95_iat = np.nan
            iat_cv = 0.0

        # Dynamic score:
        # - rewards changes in workload rate
        # - rewards bursts
        # - requires enough actual traffic
        # - avoids selecting a nearly empty interval
        dynamic_score = (
            np.log1p(request_count)
            * (1.0 + rate_cv)
            * (1.0 + np.log1p(burst_ratio))
            * max(active_bin_fraction, 0.05)
        )

        results.append(
            {
                "app": app_name,
                "window_start_sec": window_start,
                "window_end_sec": window_end,
                "request_count": request_count,
                "mean_rate_rps": mean_rate,
                "median_rate_rps": median_rate,
                "peak_rate_rps": peak_rate,
                "rate_std": standard_deviation,
                "rate_cv": rate_cv,
                "burst_ratio": burst_ratio,
                "active_bin_fraction": active_bin_fraction,
                "idle_bin_fraction": idle_bin_fraction,
                "mean_iat_sec": mean_iat,
                "median_iat_sec": median_iat,
                "p95_iat_sec": p95_iat,
                "iat_cv": iat_cv,
                "dynamic_score": dynamic_score,
            }
        )

    return pd.DataFrame(results)


def find_best_window(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Search applications and select the requested workload type.
    """

    candidate_apps = get_candidate_apps(df)

    all_statistics = []

    for index, app_name in enumerate(
        candidate_apps,
        start=1,
    ):
        print(
            f"Analyzing application {index}/"
            f"{len(candidate_apps)}: {app_name}"
        )

        app_df = df[
            df["app"] == app_name
        ].copy()

        statistics = calculate_window_statistics(
            app_df,
            app_name,
        )

        if not statistics.empty:
            all_statistics.append(statistics)

    if not all_statistics:
        raise ValueError(
            "No valid 15-minute workload window was found.\n"
            "Try reducing MIN_REQUESTS_IN_WINDOW."
        )

    candidates = pd.concat(
        all_statistics,
        ignore_index=True,
    )

    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates.to_csv(
        output_dir / "all_15min_candidate_windows.csv",
        index=False,
    )

    mode = SELECTION_MODE.strip().lower()

    if mode == "dynamic":
        selected = candidates.sort_values(
            "dynamic_score",
            ascending=False,
        ).iloc[0]

    elif mode == "high":
        selected = candidates.sort_values(
            [
                "request_count",
                "peak_rate_rps",
            ],
            ascending=False,
        ).iloc[0]

    elif mode == "low":
        # Choose a low-load window that still meets the minimum
        # request-count requirement.
        selected = candidates.sort_values(
            [
                "mean_rate_rps",
                "request_count",
            ],
            ascending=[
                True,
                True,
            ],
        ).iloc[0]

    elif mode == "medium":
        # Choose the candidate closest to the median request count.
        median_count = candidates[
            "request_count"
        ].median()

        candidates["distance_from_median_count"] = (
            candidates["request_count"]
            - median_count
        ).abs()

        selected = candidates.sort_values(
            [
                "distance_from_median_count",
                "dynamic_score",
            ],
            ascending=[
                True,
                False,
            ],
        ).iloc[0]

    else:
        raise ValueError(
            "SELECTION_MODE must be 'dynamic', 'high', "
            "'medium', or 'low'."
        )

    return selected


def extract_selected_window(
    df: pd.DataFrame,
    selected: pd.Series,
) -> pd.DataFrame:
    """
    Extract the exact invocation arrivals from the selected window.
    """

    app_name = selected["app"]
    window_start = selected["window_start_sec"]
    window_end = selected["window_end_sec"]

    selected_df = df[
        (df["app"] == app_name)
        & (df["start_time_sec"] >= window_start)
        & (df["start_time_sec"] < window_end)
    ].copy()

    selected_df = selected_df.sort_values(
        "start_time_sec"
    ).reset_index(drop=True)

    selected_df["relative_start_sec"] = (
        selected_df["start_time_sec"]
        - window_start
    )

    selected_df["iat_seconds"] = (
        selected_df["start_time_sec"].diff()
    )

    # The first request arrives relative to the beginning
    # of the selected 15-minute interval.
    selected_df.loc[
        0,
        "iat_seconds",
    ] = selected_df.loc[
        0,
        "relative_start_sec",
    ]

    if not KEEP_ZERO_IAT_REQUESTS:
        selected_df = selected_df[
            selected_df["iat_seconds"] > 0
        ].copy()

    selected_df["request_index"] = np.arange(
        len(selected_df)
    )

    selected_df["replay_time_sec"] = (
        selected_df["iat_seconds"].cumsum()
    )

    return selected_df


def save_replay_workload(
    selected_df: pd.DataFrame,
    selected_statistics: pd.Series,
) -> None:
    """
    Save the workload files used by the replay generator.
    """

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_columns = [
        "request_index",
        "app",
        "func",
        "relative_start_sec",
        "iat_seconds",
        "replay_time_sec",
        "duration_sec",
        "start_time_sec",
        "end_time_sec",
        "end_timestamp",
        "duration",
    ]

    available_columns = [
        column
        for column in replay_columns
        if column in selected_df.columns
    ]

    replay_path = (
        output_dir
        / f"azure_{SELECTION_MODE}_15min_replay.csv"
    )

    selected_df[
        available_columns
    ].to_csv(
        replay_path,
        index=False,
    )

    summary_path = (
        output_dir
        / f"azure_{SELECTION_MODE}_15min_summary.csv"
    )

    pd.DataFrame(
        [selected_statistics.to_dict()]
    ).to_csv(
        summary_path,
        index=False,
    )

    print("\nReplay workload saved:")
    print(replay_path)

    print("\nSelected-window summary saved:")
    print(summary_path)


def plot_selected_workload(
    selected_df: pd.DataFrame,
    selected_statistics: pd.Series,
) -> None:
    """
    Visualize the selected workload before running experiments.
    """

    output_dir = Path(OUTPUT_DIR)

    duration_seconds = (
        WINDOW_DURATION_MINUTES * 60
    )

    edges = np.arange(
        0,
        duration_seconds + RATE_BIN_SECONDS,
        RATE_BIN_SECONDS,
    )

    counts, _ = np.histogram(
        selected_df["relative_start_sec"],
        bins=edges,
    )

    rates = (
        counts / RATE_BIN_SECONDS
    )

    centers = (
        edges[:-1] + RATE_BIN_SECONDS / 2
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
    )

    axes[0].step(
        centers / 60.0,
        rates,
        where="mid",
        linewidth=1.7,
    )

    axes[0].set_ylabel(
        "Arrival rate\n(req/s)"
    )

    axes[0].set_title(
        f"Selected Azure {SELECTION_MODE} workload: "
        f"{selected_statistics['app']}"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[1].plot(
        selected_df["relative_start_sec"] / 60.0,
        selected_df["iat_seconds"],
        linewidth=1.0,
    )

    axes[1].set_xlabel(
        "Time within selected interval (minutes)"
    )

    axes[1].set_ylabel(
        "Inter-arrival\ntime (s)"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / f"azure_{SELECTION_MODE}_15min_pattern.png"
    )

    pdf_path = (
        output_dir
        / f"azure_{SELECTION_MODE}_15min_pattern.pdf"
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

    print("\nWorkload visualization saved:")
    print(png_path)
    print(pdf_path)


def print_selected_summary(
    selected: pd.Series,
    selected_df: pd.DataFrame,
) -> None:
    """
    Print information about the selected workload.
    """

    print("\nSelected Azure workload")
    print("-----------------------")

    print(
        f"Selection mode: {SELECTION_MODE}"
    )

    print(
        f"Application: {selected['app']}"
    )

    print(
        f"Duration: {WINDOW_DURATION_MINUTES} minutes"
    )

    print(
        f"Requests: {int(selected['request_count'])}"
    )

    print(
        f"Mean arrival rate: "
        f"{selected['mean_rate_rps']:.4f} req/s"
    )

    print(
        f"Peak arrival rate: "
        f"{selected['peak_rate_rps']:.4f} req/s"
    )

    print(
        f"Arrival-rate CV: "
        f"{selected['rate_cv']:.4f}"
    )

    print(
        f"Burst ratio: "
        f"{selected['burst_ratio']:.4f}"
    )

    print(
        f"Median IAT: "
        f"{selected['median_iat_sec']:.6f} seconds"
    )

    print(
        f"P95 IAT: "
        f"{selected['p95_iat_sec']:.6f} seconds"
    )

    print(
        f"Maximum replay time: "
        f"{selected_df['replay_time_sec'].max():.2f} seconds"
    )

    print(
        f"Maximum replay time: "
        f"{selected_df['replay_time_sec'].max() / 60:.2f} minutes"
    )


def main() -> None:
    trace_df = load_azure_trace()

    selected_statistics = find_best_window(
        trace_df
    )

    selected_df = extract_selected_window(
        trace_df,
        selected_statistics,
    )

    print_selected_summary(
        selected_statistics,
        selected_df,
    )

    save_replay_workload(
        selected_df,
        selected_statistics,
    )

    plot_selected_workload(
        selected_df,
        selected_statistics,
    )


if __name__ == "__main__":
    main()