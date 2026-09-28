#!/usr/bin/env python3

"""
visualize_azure_trace.py

Run this file directly after running extract_azure_iats.py.

Input:
    azure_original_iats.csv

Outputs:
    azure_trace_plots/
        azure_iat_sequence.png
        azure_arrival_timeline.png
        azure_request_rate.png
        azure_iat_cdf.png
        azure_iat_histogram.png

The program uses the original Azure inter-arrival times.
It does not scale or modify the workload timing.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# You usually only need to change this section
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIRECTORY / "azure_original_iats.csv"

OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "azure_trace_plots"

# Number of requests to show in sequence, timeline, and rate graphs.
# Use 1000 for a quick visualization.
MAX_REQUESTS_TO_PLOT = 1000

# Request-rate measurement window.
#
# Examples:
#   1.0  = requests per 1 second
#   5.0  = requests per 5 seconds
#   10.0 = requests per 10 seconds
RATE_BIN_SECONDS = 1.0

# Use logarithmic scale for the IAT sequence.
# This is recommended because Azure traces may contain both very small
# inter-arrival times and very long idle periods.
USE_LOG_SCALE_FOR_IAT_SEQUENCE = True

# Use logarithmic scale for the histogram x-axis.
USE_LOG_SCALE_FOR_HISTOGRAM = True

# Show each graph on the screen after saving it.
#
# Set to False when running on a remote server without a graphical display.
SHOW_GRAPHS = True

# Image quality.
FIGURE_DPI = 300


# ============================================================
# LOAD AND VALIDATE THE CSV
# ============================================================

def load_trace() -> pd.DataFrame:
    """
    Load the extracted Azure IAT file.
    """

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nInput file was not found:\n{INPUT_FILE}\n\n"
            "Run extract_azure_iats.py first. It should create "
            "azure_original_iats.csv in this directory."
        )

    print("=" * 72)
    print("Loading extracted Azure trace")
    print("=" * 72)
    print(f"Input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")

    required_columns = {
        "request_index",
        "start_time_seconds",
        "iat_seconds",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}\n\n"
            "Check that the file was created using extract_azure_iats.py."
        )

    numeric_columns = [
        "request_index",
        "start_time_seconds",
        "iat_seconds",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    original_count = len(df)

    df = df.dropna(
        subset=numeric_columns
    ).copy()

    removed_invalid = original_count - len(df)

    if removed_invalid > 0:
        print(
            f"Removed {removed_invalid:,} rows containing invalid "
            "numeric values."
        )

    negative_iat_count = int(
        (df["iat_seconds"] < 0).sum()
    )

    if negative_iat_count > 0:
        print(
            f"Removed {negative_iat_count:,} rows containing "
            "negative IAT values."
        )

        df = df.loc[
            df["iat_seconds"] >= 0
        ].copy()

    df = df.sort_values(
        "request_index"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "The input file does not contain any valid workload rows."
        )

    if len(df) < 2:
        raise ValueError(
            "At least two requests are required for visualization."
        )

    return df


def select_visible_segment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select the first MAX_REQUESTS_TO_PLOT requests for time-based graphs.
    """

    if MAX_REQUESTS_TO_PLOT is None:
        visible_df = df.copy()
    else:
        if MAX_REQUESTS_TO_PLOT < 2:
            raise ValueError(
                "MAX_REQUESTS_TO_PLOT must be at least 2 or None."
            )

        visible_df = df.head(
            MAX_REQUESTS_TO_PLOT
        ).copy()

    first_time = visible_df["start_time_seconds"].iloc[0]

    visible_df["relative_time_seconds"] = (
        visible_df["start_time_seconds"] - first_time
    )

    return visible_df


# ============================================================
# SUMMARY STATISTICS
# ============================================================

def print_summary(
    df: pd.DataFrame,
    visible_df: pd.DataFrame,
) -> None:
    """
    Print trace and IAT statistics.
    """

    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ]

    trace_duration = (
        df["start_time_seconds"].iloc[-1]
        - df["start_time_seconds"].iloc[0]
    )

    visible_duration = (
        visible_df["relative_time_seconds"].iloc[-1]
    )

    print()
    print("=" * 72)
    print("Azure trace summary")
    print("=" * 72)

    print(f"Total requests in CSV: {len(df):,}")
    print(
        f"Requests used in time-based graphs: "
        f"{len(visible_df):,}"
    )
    print(f"Complete trace duration: {trace_duration:.6f} seconds")
    print(
        f"Displayed segment duration: "
        f"{visible_duration:.6f} seconds"
    )

    if positive_iats.empty:
        print("No positive IAT values were found.")
        return

    print()
    print("Original IAT statistics")
    print("-" * 72)
    print(f"Mean:    {positive_iats.mean():.6f} seconds")
    print(f"Median:  {positive_iats.median():.6f} seconds")
    print(f"Minimum: {positive_iats.min():.6f} seconds")
    print(f"Maximum: {positive_iats.max():.6f} seconds")
    print(
        f"P90:     "
        f"{positive_iats.quantile(0.90):.6f} seconds"
    )
    print(
        f"P95:     "
        f"{positive_iats.quantile(0.95):.6f} seconds"
    )
    print(
        f"P99:     "
        f"{positive_iats.quantile(0.99):.6f} seconds"
    )

    if trace_duration > 0:
        average_rate = (
            len(df) - 1
        ) / trace_duration

        print(
            f"Average arrival rate: "
            f"{average_rate:.6f} requests/second"
        )

    zero_iat_fraction = (
        df["iat_seconds"].eq(0).mean()
    )

    print(
        f"Zero-IAT fraction: "
        f"{zero_iat_fraction:.4f}"
    )


# ============================================================
# SAVE AND DISPLAY HELPER
# ============================================================

def finish_figure(
    output_filename: str,
) -> None:
    """
    Save, optionally display, and close the current figure.
    """

    output_path = OUTPUT_DIRECTORY / output_filename

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
    )

    print(f"Saved: {output_path}")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# ============================================================
# GRAPH 1: IAT SEQUENCE
# ============================================================

def plot_iat_sequence(
    visible_df: pd.DataFrame,
) -> None:
    """
    Plot IAT against request index.

    This graph makes bursts and idle periods visible.
    Small IAT values indicate closely arriving requests.
    Large IAT values indicate idle gaps.
    """

    plot_df = visible_df.loc[
        visible_df["iat_seconds"] > 0
    ].copy()

    if plot_df.empty:
        print(
            "Skipping IAT sequence: no positive IAT values "
            "are available."
        )
        return

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        plot_df["request_index"],
        plot_df["iat_seconds"],
        linewidth=1.0,
    )

    plt.xlabel("Request index")

    if USE_LOG_SCALE_FOR_IAT_SEQUENCE:
        plt.yscale("log")
        plt.ylabel("Original IAT (seconds, log scale)")
    else:
        plt.ylabel("Original IAT (seconds)")

    plt.title(
        "Azure Inter-Arrival Time Sequence\n"
        f"First {len(visible_df):,} Requests"
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    finish_figure(
        "azure_iat_sequence.png"
    )


# ============================================================
# GRAPH 2: ARRIVAL TIMELINE
# ============================================================

def plot_arrival_timeline(
    visible_df: pd.DataFrame,
) -> None:
    """
    Plot each request as a vertical mark on the workload timeline.

    Dense marks indicate bursts.
    Empty regions indicate idle periods.
    """

    relative_times = visible_df[
        "relative_time_seconds"
    ].to_numpy()

    plt.figure(
        figsize=(12, 3.5)
    )

    plt.eventplot(
        relative_times,
        orientation="horizontal",
        lineoffsets=1,
        linelengths=0.8,
        linewidths=0.7,
    )

    plt.xlabel("Elapsed trace time (seconds)")
    plt.yticks([])

    plt.title(
        "Azure Request Arrival Timeline\n"
        f"First {len(visible_df):,} Requests"
    )

    plt.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    finish_figure(
        "azure_arrival_timeline.png"
    )


# ============================================================
# GRAPH 3: REQUEST RATE OVER TIME
# ============================================================

def plot_request_rate(
    visible_df: pd.DataFrame,
) -> None:
    """
    Calculate and plot the request rate in fixed time windows.
    """

    if RATE_BIN_SECONDS <= 0:
        raise ValueError(
            "RATE_BIN_SECONDS must be greater than zero."
        )

    relative_times = visible_df[
        "relative_time_seconds"
    ].to_numpy()

    maximum_time = float(
        relative_times.max()
    )

    if maximum_time <= 0:
        print(
            "Skipping request-rate graph because the displayed "
            "trace segment has zero duration."
        )
        return

    bin_edges = np.arange(
        0,
        maximum_time + RATE_BIN_SECONDS,
        RATE_BIN_SECONDS,
    )

    if len(bin_edges) < 2:
        bin_edges = np.array(
            [0.0, RATE_BIN_SECONDS],
            dtype=float,
        )

    # Ensure the final request belongs to a bin.
    if bin_edges[-1] <= maximum_time:
        bin_edges = np.append(
            bin_edges,
            bin_edges[-1] + RATE_BIN_SECONDS,
        )

    request_counts, bin_edges = np.histogram(
        relative_times,
        bins=bin_edges,
    )

    request_rates = (
        request_counts / RATE_BIN_SECONDS
    )

    bin_start_times = bin_edges[:-1]

    plt.figure(
        figsize=(12, 5)
    )

    plt.step(
        bin_start_times,
        request_rates,
        where="post",
        linewidth=1.2,
    )

    plt.xlabel("Elapsed trace time (seconds)")
    plt.ylabel("Request rate (requests/second)")

    plt.title(
        "Azure Request Rate Over Time\n"
        f"Window Size = {RATE_BIN_SECONDS:g} Seconds"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    finish_figure(
        "azure_request_rate.png"
    )


# ============================================================
# GRAPH 4: EMPIRICAL IAT CDF
# ============================================================

def plot_iat_cdf(
    df: pd.DataFrame,
) -> None:
    """
    Plot the empirical cumulative distribution of positive IAT values.
    """

    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ].to_numpy()

    if len(positive_iats) == 0:
        print(
            "Skipping IAT CDF: no positive IAT values are available."
        )
        return

    sorted_iats = np.sort(
        positive_iats
    )

    cumulative_probability = (
        np.arange(
            1,
            len(sorted_iats) + 1,
        )
        / len(sorted_iats)
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        sorted_iats,
        cumulative_probability,
        linewidth=2,
    )

    plt.xscale("log")

    plt.xlabel("Original IAT (seconds, log scale)")
    plt.ylabel("Cumulative probability")

    plt.title(
        "Empirical CDF of Azure Inter-Arrival Times"
    )

    plt.ylim(
        0,
        1.01,
    )

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    finish_figure(
        "azure_iat_cdf.png"
    )


# ============================================================
# GRAPH 5: IAT HISTOGRAM
# ============================================================

def plot_iat_histogram(
    df: pd.DataFrame,
) -> None:
    """
    Plot the distribution of positive IAT values.
    """

    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ].to_numpy()

    if len(positive_iats) == 0:
        print(
            "Skipping IAT histogram: no positive IAT values "
            "are available."
        )
        return

    minimum_iat = float(
        positive_iats.min()
    )

    maximum_iat = float(
        positive_iats.max()
    )

    plt.figure(
        figsize=(8, 5)
    )

    if (
        USE_LOG_SCALE_FOR_HISTOGRAM
        and minimum_iat > 0
        and maximum_iat > minimum_iat
    ):
        histogram_bins = np.logspace(
            np.log10(minimum_iat),
            np.log10(maximum_iat),
            50,
        )

        plt.hist(
            positive_iats,
            bins=histogram_bins,
            edgecolor="black",
            alpha=0.75,
        )

        plt.xscale("log")

        plt.xlabel(
            "Original IAT (seconds, log scale)"
        )
    else:
        plt.hist(
            positive_iats,
            bins=50,
            edgecolor="black",
            alpha=0.75,
        )

        plt.xlabel(
            "Original IAT (seconds)"
        )

    plt.ylabel("Number of requests")

    plt.title(
        "Distribution of Azure Inter-Arrival Times"
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.4,
    )

    finish_figure(
        "azure_iat_histogram.png"
    )


# ============================================================
# MAIN PROGRAM
# ============================================================

def main() -> None:
    """
    Run all visualization steps.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_trace()

    visible_df = select_visible_segment(
        df
    )

    print_summary(
        df,
        visible_df,
    )

    print()
    print("=" * 72)
    print("Creating Azure workload graphs")
    print("=" * 72)

    plot_iat_sequence(
        visible_df
    )

    plot_arrival_timeline(
        visible_df
    )

    plot_request_rate(
        visible_df
    )

    plot_iat_cdf(
        df
    )

    plot_iat_histogram(
        df
    )

    print()
    print("=" * 72)
    print("Completed successfully")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIRECTORY}")

    print()
    print("Created files:")
    print("  1. azure_iat_sequence.png")
    print("  2. azure_arrival_timeline.png")
    print("  3. azure_request_rate.png")
    print("  4. azure_iat_cdf.png")
    print("  5. azure_iat_histogram.png")
    print()
    print("No IAT scaling was applied.")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 72)
        print("ERROR")
        print("=" * 72)
        print(error)
        print()
        print(
            "Check that azure_original_iats.csv is in the same "
            "directory as this Python file."
        )

        input("\nPress Enter to close...")

        sys.exit(1)

    input("\nPress Enter to close...")