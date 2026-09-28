#!/usr/bin/env python3

"""
plot_azure_iats.py

Run this file directly after running extract_azure_iats.py.

It creates:

1. azure_iat_sequence.png
2. azure_arrival_timeline.png
3. azure_request_rate.png
4. azure_iat_cdf.png

The graphs use the original Azure IATs.
No scaling is applied.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# USER SETTINGS
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

INPUT_FILE = SCRIPT_DIRECTORY / "azure_original_iats.csv"

# Directory where the figures will be saved
OUTPUT_DIRECTORY = SCRIPT_DIRECTORY / "azure_iat_graphs"

# Number of requests shown in the first two figures
# Your CSV may contain 1,000 or more requests.
NUMBER_OF_REQUESTS_TO_PLOT = 1000

# Time-bin size used for the request-rate graph
# 1.0 means requests per second.
RATE_BIN_SECONDS = 1.0

# True gives a logarithmic y-axis for the IAT-sequence graph.
# This is useful when Azure contains both tiny IATs and large idle gaps.
USE_LOG_SCALE_FOR_IAT = True

# Set to True if you want the graphs to appear on the screen.
# Set to False if running on a remote server without a graphical display.
SHOW_GRAPHS = True


# ============================================================
# LOAD AND VALIDATE DATA
# ============================================================

def load_iat_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"\nThe input CSV was not found:\n{INPUT_FILE}\n\n"
            "Run extract_azure_iats.py first."
        )

    print("=" * 70)
    print("Loading extracted Azure workload")
    print("=" * 70)
    print(f"Input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required_columns = {
        "request_index",
        "start_time_seconds",
        "iat_seconds",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    df["request_index"] = pd.to_numeric(
        df["request_index"],
        errors="coerce",
    )

    df["start_time_seconds"] = pd.to_numeric(
        df["start_time_seconds"],
        errors="coerce",
    )

    df["iat_seconds"] = pd.to_numeric(
        df["iat_seconds"],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            "request_index",
            "start_time_seconds",
            "iat_seconds",
        ]
    ).copy()

    df = df.loc[df["iat_seconds"] >= 0].copy()

    df = df.sort_values("request_index").reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid workload rows were found.")

    print(f"Requests loaded: {len(df):,}")

    return df


# ============================================================
# FIGURE 1: IAT SEQUENCE
# ============================================================

def plot_iat_sequence(df: pd.DataFrame) -> None:
    visible_df = df.head(NUMBER_OF_REQUESTS_TO_PLOT).copy()

    # Zero cannot be shown on a logarithmic axis.
    plot_df = visible_df.loc[
        visible_df["iat_seconds"] > 0
    ].copy()

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_df["request_index"],
        plot_df["iat_seconds"],
        linewidth=1,
    )

    plt.xlabel("Request index")
    plt.ylabel("Original IAT (seconds)")

    plt.title(
        f"Azure Inter-Arrival Time Sequence "
        f"(First {len(visible_df):,} Requests)"
    )

    if USE_LOG_SCALE_FOR_IAT:
        plt.yscale("log")
        plt.ylabel("Original IAT (seconds, log scale)")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIRECTORY / "azure_iat_sequence.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_file}")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# ============================================================
# FIGURE 2: ARRIVAL TIMELINE
# ============================================================

def plot_arrival_timeline(df: pd.DataFrame) -> None:
    visible_df = df.head(NUMBER_OF_REQUESTS_TO_PLOT).copy()

    relative_time = (
        visible_df["start_time_seconds"]
        - visible_df["start_time_seconds"].iloc[0]
    )

    plt.figure(figsize=(12, 3.5))

    plt.eventplot(
        relative_time,
        orientation="horizontal",
        lineoffsets=1,
        linelengths=0.8,
    )

    plt.xlabel("Elapsed trace time (seconds)")
    plt.yticks([])

    plt.title(
        f"Azure Request Arrival Timeline "
        f"(First {len(visible_df):,} Requests)"
    )

    plt.grid(
        True,
        axis="x",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIRECTORY / "azure_arrival_timeline.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_file}")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# ============================================================
# FIGURE 3: REQUEST RATE
# ============================================================

def plot_request_rate(df: pd.DataFrame) -> None:
    visible_df = df.head(NUMBER_OF_REQUESTS_TO_PLOT).copy()

    relative_time = (
        visible_df["start_time_seconds"]
        - visible_df["start_time_seconds"].iloc[0]
    )

    maximum_time = float(relative_time.max())

    if maximum_time <= 0:
        raise ValueError(
            "The selected trace segment has no positive duration."
        )

    bins = np.arange(
        0,
        maximum_time + RATE_BIN_SECONDS,
        RATE_BIN_SECONDS,
    )

    if len(bins) < 2:
        bins = np.array(
            [0.0, RATE_BIN_SECONDS],
            dtype=float,
        )

    request_counts, bin_edges = np.histogram(
        relative_time,
        bins=bins,
    )

    request_rate = request_counts / RATE_BIN_SECONDS

    plt.figure(figsize=(12, 5))

    plt.step(
        bin_edges[:-1],
        request_rate,
        where="post",
        linewidth=1,
    )

    plt.xlabel("Elapsed trace time (seconds)")
    plt.ylabel("Request rate (requests/second)")

    plt.title(
        f"Azure Request Rate "
        f"(First {len(visible_df):,} Requests)"
    )

    plt.grid(
        True,
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIRECTORY / "azure_request_rate.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_file}")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# ============================================================
# FIGURE 4: IAT CDF
# ============================================================

def plot_iat_cdf(df: pd.DataFrame) -> None:
    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ].to_numpy()

    if len(positive_iats) == 0:
        raise ValueError(
            "No positive IAT values are available for the CDF."
        )

    sorted_iats = np.sort(positive_iats)

    cumulative_probability = (
        np.arange(1, len(sorted_iats) + 1)
        / len(sorted_iats)
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        sorted_iats,
        cumulative_probability,
        linewidth=2,
    )

    plt.xscale("log")

    plt.xlabel("Original IAT (seconds, log scale)")
    plt.ylabel("Cumulative probability")

    plt.title("CDF of Original Azure Inter-Arrival Times")

    plt.grid(
        True,
        which="both",
        linestyle="--",
        alpha=0.4,
    )

    plt.tight_layout()

    output_file = OUTPUT_DIRECTORY / "azure_iat_cdf.png"

    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )

    print(f"Saved: {output_file}")

    if SHOW_GRAPHS:
        plt.show()

    plt.close()


# ============================================================
# PRINT STATISTICS
# ============================================================

def print_summary(df: pd.DataFrame) -> None:
    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ]

    print()
    print("=" * 70)
    print("Workload summary")
    print("=" * 70)

    print(f"Total requests: {len(df):,}")

    if positive_iats.empty:
        print("No positive IAT values found.")
        return

    print(f"Mean IAT: {positive_iats.mean():.6f} seconds")
    print(f"Median IAT: {positive_iats.median():.6f} seconds")
    print(f"P90 IAT: {positive_iats.quantile(0.90):.6f} seconds")
    print(f"P95 IAT: {positive_iats.quantile(0.95):.6f} seconds")
    print(f"P99 IAT: {positive_iats.quantile(0.99):.6f} seconds")
    print(f"Maximum IAT: {positive_iats.max():.6f} seconds")

    trace_duration = (
        df["start_time_seconds"].iloc[-1]
        - df["start_time_seconds"].iloc[0]
    )

    print(f"Trace duration: {trace_duration:.6f} seconds")

    if trace_duration > 0:
        average_rate = (len(df) - 1) / trace_duration

        print(
            f"Average arrival rate: "
            f"{average_rate:.6f} requests/second"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_iat_data()

    print_summary(df)

    print()
    print("=" * 70)
    print("Creating graphs")
    print("=" * 70)

    plot_iat_sequence(df)
    plot_arrival_timeline(df)
    plot_request_rate(df)
    plot_iat_cdf(df)

    print()
    print("=" * 70)
    print("Completed successfully")
    print("=" * 70)
    print(f"Graphs directory: {OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(error)

        input("\nPress Enter to close...")

        sys.exit(1)

    input("\nPress Enter to close...")