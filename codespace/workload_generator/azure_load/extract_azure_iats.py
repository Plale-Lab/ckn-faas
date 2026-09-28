#!/usr/bin/env python3

"""
extract_azure_iats.py

Run this file directly.

What it does:
1. Reads the Azure Functions trace.
2. Selects one Azure application.
3. Reconstructs request start times.
4. Calculates the original inter-arrival times (IATs).
5. Saves the extracted workload to a CSV file.
6. Does not scale the IATs.

Important:
The script assumes that end_timestamp and duration use the same unit.
Based on your earlier analysis, this version assumes seconds.
"""

from pathlib import Path
import sys

import pandas as pd


# ============================================================
# USER SETTINGS
# Change only this section when necessary
# ============================================================

INPUT_FILE = Path(
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load/AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

# Bursty representative application selected earlier
APP_ID = (
    "6e42a4e8ea1df408e059ba37d0e04255fd8385235b310265a57091132d760a65"
)

# Output will be saved in the same directory as this Python file
SCRIPT_DIRECTORY = Path(__file__).resolve().parent

OUTPUT_FILE = SCRIPT_DIRECTORY / "azure_original_iats.csv"

# Use None to extract every request for the selected application.
# Use 1000 for a small test.
MAX_REQUESTS = 1000

# Input delimiter:
# Use "," if the source is comma-separated.
# Use "\t" if it is tab-separated.
SEPARATOR = ","

# Set this according to the timestamp and duration unit in the Azure file.
# Supported values:
# "seconds"
# "milliseconds"
# "microseconds"
TIME_UNIT = "seconds"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_seconds_conversion_factor(time_unit: str) -> float:
    """
    Convert the source timestamp unit into seconds.
    """

    conversion_factors = {
        "seconds": 1.0,
        "milliseconds": 0.001,
        "microseconds": 0.000001,
    }

    if time_unit not in conversion_factors:
        raise ValueError(
            f"Unsupported TIME_UNIT: {time_unit}. "
            "Use 'seconds', 'milliseconds', or 'microseconds'."
        )

    return conversion_factors[time_unit]


def load_trace(input_file: Path) -> pd.DataFrame:
    """
    Load the Azure Functions trace file.
    """

    if not input_file.exists():
        raise FileNotFoundError(
            f"\nAzure trace file was not found:\n{input_file}\n\n"
            "Check INPUT_FILE at the top of the script."
        )

    print("=" * 70)
    print("Reading Azure trace")
    print("=" * 70)
    print(f"Input file: {input_file}")

    separator = "\t" if SEPARATOR == r"\t" else SEPARATOR

    trace_df = pd.read_csv(
        input_file,
        sep=separator,
        low_memory=False,
    )

    print(f"Rows loaded: {len(trace_df):,}")
    print(f"Columns: {list(trace_df.columns)}")

    return trace_df


def validate_columns(trace_df: pd.DataFrame) -> None:
    """
    Check that the required source columns exist.
    """

    required_columns = {
        "app",
        "func",
        "end_timestamp",
        "duration",
    }

    missing_columns = required_columns.difference(trace_df.columns)

    if missing_columns:
        raise ValueError(
            f"\nMissing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(trace_df.columns)}\n\n"
            "The file delimiter may be incorrect. "
            "Check the SEPARATOR value at the top of the script."
        )


def select_application(
    trace_df: pd.DataFrame,
    app_id: str,
) -> pd.DataFrame:
    """
    Select all requests belonging to one Azure application.
    """

    app_df = trace_df.loc[
        trace_df["app"].astype(str) == str(app_id)
    ].copy()

    if app_df.empty:
        unique_apps = trace_df["app"].nunique()

        raise ValueError(
            f"\nNo requests were found for this application:\n{app_id}\n\n"
            f"The trace contains {unique_apps:,} unique applications.\n"
            "Check the APP_ID value at the top of the script."
        )

    print()
    print("=" * 70)
    print("Selected application")
    print("=" * 70)
    print(f"Application ID: {app_id}")
    print(f"Matching requests: {len(app_df):,}")

    return app_df


def clean_numeric_columns(app_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert timestamp and duration columns to numeric values.
    """

    app_df["end_timestamp"] = pd.to_numeric(
        app_df["end_timestamp"],
        errors="coerce",
    )

    app_df["duration"] = pd.to_numeric(
        app_df["duration"],
        errors="coerce",
    )

    original_count = len(app_df)

    app_df = app_df.dropna(
        subset=["end_timestamp", "duration"]
    ).copy()

    removed_count = original_count - len(app_df)

    if removed_count > 0:
        print(
            f"Removed {removed_count:,} rows with invalid "
            "timestamp or duration values."
        )

    if len(app_df) < 2:
        raise ValueError(
            "At least two valid requests are required to calculate IATs."
        )

    return app_df


def reconstruct_start_times(
    app_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Reconstruct an approximate execution start timestamp.

    start_timestamp = end_timestamp - duration
    """

    app_df["start_timestamp"] = (
        app_df["end_timestamp"] - app_df["duration"]
    )

    app_df = app_df.sort_values(
        "start_timestamp"
    ).reset_index(drop=True)

    return app_df


def limit_requests(
    app_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Optionally keep only the first MAX_REQUESTS requests.
    """

    if MAX_REQUESTS is None:
        print("Using all requests for the selected application.")
        return app_df

    if MAX_REQUESTS < 2:
        raise ValueError(
            "MAX_REQUESTS must be at least 2 or set to None."
        )

    limited_df = app_df.head(MAX_REQUESTS).copy()

    print(
        f"Using the first {len(limited_df):,} requests "
        f"because MAX_REQUESTS = {MAX_REQUESTS}."
    )

    return limited_df


def calculate_iats(
    app_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate original inter-arrival times without scaling.
    """

    factor = get_seconds_conversion_factor(TIME_UNIT)

    first_start_timestamp = app_df["start_timestamp"].iloc[0]

    app_df["start_time_seconds"] = (
        app_df["start_timestamp"] - first_start_timestamp
    ) * factor

    app_df["iat_seconds"] = (
        app_df["start_timestamp"].diff() * factor
    )

    # The first request has no previous request.
    # Set its IAT to zero so it can be sent immediately during replay.
    app_df.loc[0, "iat_seconds"] = 0.0

    negative_iat_count = int(
        (app_df["iat_seconds"] < 0).sum()
    )

    if negative_iat_count > 0:
        print(
            f"Warning: found {negative_iat_count} negative IAT values."
        )
        print("Those rows will be removed.")

        app_df = app_df.loc[
            app_df["iat_seconds"] >= 0
        ].copy()

    app_df = app_df.reset_index(drop=True)

    return app_df


def build_output_dataframe(
    app_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create the final output CSV structure.
    """

    output_df = pd.DataFrame(
        {
            "request_index": range(1, len(app_df) + 1),
            "app": app_df["app"].astype(str),
            "func": app_df["func"].astype(str),
            "start_time_seconds": app_df["start_time_seconds"],
            "iat_seconds": app_df["iat_seconds"],
            "original_end_timestamp": app_df["end_timestamp"],
            "original_duration": app_df["duration"],
        }
    )

    return output_df


def print_statistics(
    output_df: pd.DataFrame,
) -> None:
    """
    Print summary statistics for the extracted workload.
    """

    positive_iats = output_df.loc[
        output_df["iat_seconds"] > 0,
        "iat_seconds",
    ]

    print()
    print("=" * 70)
    print("Extracted workload summary")
    print("=" * 70)

    print(f"Requests: {len(output_df):,}")

    if positive_iats.empty:
        print("No positive IAT values were found.")
        return

    total_trace_time = output_df["start_time_seconds"].iloc[-1]

    print(f"Trace duration: {total_trace_time:.6f} seconds")
    print(f"Mean IAT: {positive_iats.mean():.6f} seconds")
    print(f"Median IAT: {positive_iats.median():.6f} seconds")
    print(f"Minimum IAT: {positive_iats.min():.6f} seconds")
    print(f"Maximum IAT: {positive_iats.max():.6f} seconds")
    print(
        f"P90 IAT: {positive_iats.quantile(0.90):.6f} seconds"
    )
    print(
        f"P95 IAT: {positive_iats.quantile(0.95):.6f} seconds"
    )
    print(
        f"P99 IAT: {positive_iats.quantile(0.99):.6f} seconds"
    )

    if total_trace_time > 0:
        average_rate = (
            len(output_df) - 1
        ) / total_trace_time

        print(
            f"Average arrival rate: "
            f"{average_rate:.6f} requests/second"
        )


def save_output(
    output_df: pd.DataFrame,
) -> None:
    """
    Save the extracted Azure workload.
    """

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("=" * 70)
    print("Completed successfully")
    print("=" * 70)
    print(f"Output file: {OUTPUT_FILE}")
    print()
    print("No IAT scaling was applied.")
    print("The saved IATs preserve the original Azure timing.")


def main() -> None:
    """
    Main program.
    """

    trace_df = load_trace(INPUT_FILE)

    validate_columns(trace_df)

    app_df = select_application(
        trace_df,
        APP_ID,
    )

    app_df = clean_numeric_columns(app_df)

    app_df = reconstruct_start_times(app_df)

    app_df = limit_requests(app_df)

    app_df = calculate_iats(app_df)

    output_df = build_output_dataframe(app_df)

    print_statistics(output_df)

    save_output(output_df)


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(error)
        print()
        print(
            "Please check INPUT_FILE, APP_ID, SEPARATOR, "
            "and TIME_UNIT at the top of the script."
        )

        # Keeps the terminal open when launched from some graphical editors.
        input("\nPress Enter to close...")

        sys.exit(1)

    input("\nPress Enter to close...")