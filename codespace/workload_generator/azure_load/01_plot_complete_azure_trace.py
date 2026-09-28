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
    "azure_15min_selection"
)

# Use one Azure application because an application represents
# one realistic workload stream.
#
# None means automatically use the application with the
# largest number of invocations.
SELECTED_APP = None

# Aggregate the complete trace into one-minute bins.
BIN_SECONDS = 60.0

# The Azure file used in your earlier analysis appears to use
# seconds for end_timestamp and duration.
TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# ============================================================


def unit_divisor(unit: str) -> float:
    unit = unit.strip().lower()

    if unit == "seconds":
        return 1.0

    if unit == "milliseconds":
        return 1_000.0

    if unit == "microseconds":
        return 1_000_000.0

    raise ValueError(
        "Unit must be seconds, milliseconds, or microseconds."
    )


def read_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace file was not found:\n{trace_path}"
        )

    print("Reading Azure trace...")
    print(trace_path)

    df = pd.read_csv(trace_path)

    # Fall back to whitespace-separated input if necessary.
    if len(df.columns) == 1:
        df = pd.read_csv(
            trace_path,
            sep=r"\s+",
            engine="python",
        )

    print("\nColumns:")
    print(list(df.columns))

    required = [
        "app",
        "func",
        "end_timestamp",
        "duration",
    ]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing column: {column}\n"
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

    return df


def select_application(df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    if SELECTED_APP is not None:
        selected_app = SELECTED_APP
    else:
        selected_app = (
            df.groupby("app")
            .size()
            .sort_values(ascending=False)
            .index[0]
        )

    app_df = df[
        df["app"] == selected_app
    ].copy()

    if app_df.empty:
        raise ValueError(
            f"No requests were found for app: {selected_app}"
        )

    print("\nSelected application:")
    print(selected_app)

    print(
        f"Number of invocations: {len(app_df):,}"
    )

    return str(selected_app), app_df


def calculate_complete_pattern(
    app_df: pd.DataFrame,
) -> pd.DataFrame:
    start_time = app_df["start_time_sec"].min()

    app_df = app_df.copy()

    app_df["relative_time_sec"] = (
        app_df["start_time_sec"] - start_time
    )

    total_duration = (
        app_df["relative_time_sec"].max()
    )

    edges = np.arange(
        0,
        total_duration + BIN_SECONDS,
        BIN_SECONDS,
    )

    counts, _ = np.histogram(
        app_df["relative_time_sec"],
        bins=edges,
    )

    pattern = pd.DataFrame(
        {
            "bin_start_sec": edges[:-1],
            "request_count": counts,
        }
    )

    pattern["time_hours"] = (
        pattern["bin_start_sec"] / 3600.0
    )

    pattern["arrival_rate_rps"] = (
        pattern["request_count"] / BIN_SECONDS
    )

    return pattern


def plot_pattern(
    app_name: str,
    pattern: pd.DataFrame,
) -> None:
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        pattern["time_hours"],
        pattern["arrival_rate_rps"],
        linewidth=1.0,
    )

    ax.set_title(
        f"Complete Azure workload for application {app_name}"
    )

    ax.set_xlabel(
        "Time from beginning of trace (hours)"
    )

    ax.set_ylabel(
        "Arrival rate (requests/second)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / "complete_azure_workload.png"
    )

    pdf_path = (
        output_dir
        / "complete_azure_workload.pdf"
    )

    csv_path = (
        output_dir
        / "complete_azure_workload_bins.csv"
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

    pattern.to_csv(
        csv_path,
        index=False,
    )

    plt.show()
    plt.close(fig)

    print("\nFiles saved:")
    print(png_path)
    print(pdf_path)
    print(csv_path)


def print_summary(
    app_name: str,
    app_df: pd.DataFrame,
    pattern: pd.DataFrame,
) -> None:
    duration_sec = (
        app_df["start_time_sec"].max()
        - app_df["start_time_sec"].min()
    )

    print("\nComplete Azure workload summary")
    print("--------------------------------")
    print(f"Application: {app_name}")
    print(f"Requests: {len(app_df):,}")

    print(
        f"Trace duration: "
        f"{duration_sec / 3600:.2f} hours"
    )

    print(
        f"Mean arrival rate: "
        f"{pattern['arrival_rate_rps'].mean():.4f} req/s"
    )

    print(
        f"Peak arrival rate: "
        f"{pattern['arrival_rate_rps'].max():.4f} req/s"
    )

    print(
        f"Idle-bin percentage: "
        f"{(pattern['request_count'] == 0).mean() * 100:.2f}%"
    )


def main() -> None:
    df = read_trace()

    app_name, app_df = select_application(df)

    pattern = calculate_complete_pattern(
        app_df
    )

    print_summary(
        app_name,
        app_df,
        pattern,
    )

    plot_pattern(
        app_name,
        pattern,
    )


if __name__ == "__main__":
    main()