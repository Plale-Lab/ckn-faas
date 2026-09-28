from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

TRACE_FILE = Path(
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

TOP_K_APPS = 5

# Remove extremely large IATs from only the displayed x-axis.
# The underlying statistics still use the full valid IAT data.
CDF_X_LIMIT_SECONDS = 10.0

OUTPUT_FIGURE = "azure_top5_iat_cdf.png"
OUTPUT_SUMMARY = "azure_top5_iat_summary5.csv"


# ============================================================
# Helper functions
# ============================================================

def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Return x and y values for an empirical CDF.
    """
    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values) &
        (values >= 0)
    ]

    if values.size == 0:
        raise ValueError("No valid values are available for the CDF.")

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def compute_iats_for_app(
    trace_df: pd.DataFrame,
    app_id: str,
) -> np.ndarray:
    """
    Compute invocation inter-arrival times for one Azure application.

    Arrival time is approximated as:
        arrival_time = end_timestamp - duration
    """
    app_df = trace_df[
        trace_df["app"] == app_id
    ].copy()

    app_df["arrival_time"] = (
        app_df["end_timestamp"] -
        app_df["duration"]
    )

    arrivals = (
        app_df["arrival_time"]
        .sort_values()
        .to_numpy(dtype=float)
    )

    if len(arrivals) < 2:
        return np.array([], dtype=float)

    iats = np.diff(arrivals)

    # Remove invalid negative or non-finite values.
    iats = iats[
        np.isfinite(iats) &
        (iats >= 0)
    ]

    return iats


def exponential_cdf(
    x: np.ndarray,
    mean_iat: float,
) -> np.ndarray:
    """
    CDF of an exponential distribution with a given mean IAT.

    F(t) = 1 - exp(-t / mean_iat)
    """
    if mean_iat <= 0:
        raise ValueError("Mean IAT must be greater than zero.")

    return 1.0 - np.exp(-x / mean_iat)


# ============================================================
# Load trace
# ============================================================

if not TRACE_FILE.exists():
    raise FileNotFoundError(
        f"Trace file was not found: {TRACE_FILE.resolve()}"
    )

print("Loading Azure trace...")

df = pd.read_csv(TRACE_FILE)

required_columns = {
    "app",
    "func",
    "end_timestamp",
    "duration",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# Convert timing columns safely.
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
        "end_timestamp",
        "duration",
    ]
)

print(f"Total invocations: {len(df):,}")
print(f"Total applications: {df['app'].nunique():,}")


# ============================================================
# Select the busiest applications
# ============================================================

app_counts = df["app"].value_counts()

top_apps = app_counts.head(TOP_K_APPS)

print("\nTop applications:")

for rank, (app_id, count) in enumerate(
    top_apps.items(),
    start=1,
):
    print(
        f"{rank}. {app_id[:12]}... "
        f"invocations={count:,}"
    )


# ============================================================
# Compute IATs and summaries
# ============================================================

app_iats = {}
summary_rows = []

for rank, (app_id, invocation_count) in enumerate(
    top_apps.items(),
    start=1,
):
    iats = compute_iats_for_app(
        trace_df=df,
        app_id=app_id,
    )

    if iats.size == 0:
        print(
            f"Skipping application {app_id[:12]}: "
            "not enough valid arrivals."
        )
        continue

    app_iats[app_id] = iats

    mean_iat = float(np.mean(iats))
    std_iat = float(np.std(iats))

    coefficient_of_variation = (
        std_iat / mean_iat
        if mean_iat > 0
        else np.nan
    )

    arrival_rate = (
        1.0 / mean_iat
        if mean_iat > 0
        else np.nan
    )

    summary_rows.append(
        {
            "rank": rank,
            "app_id": app_id,
            "invocations": int(invocation_count),
            "iat_count": int(len(iats)),
            "mean_iat_seconds": mean_iat,
            "median_iat_seconds": float(
                np.median(iats)
            ),
            "p95_iat_seconds": float(
                np.percentile(iats, 95)
            ),
            "p99_iat_seconds": float(
                np.percentile(iats, 99)
            ),
            "std_iat_seconds": std_iat,
            "coefficient_of_variation": (
                coefficient_of_variation
            ),
            "mean_arrival_rate_requests_per_second": (
                arrival_rate
            ),
            "zero_iat_fraction": float(
                np.mean(iats == 0)
            ),
        }
    )


summary_df = pd.DataFrame(summary_rows)

summary_df.to_csv(
    OUTPUT_SUMMARY,
    index=False,
)

print("\nIAT summary:")

print(
    summary_df[
        [
            "rank",
            "invocations",
            "mean_iat_seconds",
            "median_iat_seconds",
            "p95_iat_seconds",
            "coefficient_of_variation",
            "mean_arrival_rate_requests_per_second",
        ]
    ].to_string(index=False)
)


# ============================================================
# Plot empirical CDFs
# ============================================================

plt.figure(figsize=(10, 6))

for rank, app_id in enumerate(
    app_iats.keys(),
    start=1,
):
    iats = app_iats[app_id]

    x, y = empirical_cdf(iats)

    label = (
        f"App {rank} "
        f"({len(iats) + 1:,} invocations)"
    )

    plt.plot(
        x,
        y,
        linewidth=2,
        label=label,
    )


# ============================================================
# Add one Poisson/exponential reference
# ============================================================

# Use the mean IAT of the busiest application.
reference_app_id = next(iter(app_iats))

reference_iats = app_iats[reference_app_id]

reference_mean_iat = float(
    np.mean(reference_iats)
)

reference_x_max = (
    CDF_X_LIMIT_SECONDS
    if CDF_X_LIMIT_SECONDS is not None
    else float(np.percentile(reference_iats, 99.5))
)

reference_x = np.linspace(
    0,
    reference_x_max,
    1000,
)

reference_y = exponential_cdf(
    reference_x,
    reference_mean_iat,
)

plt.plot(
    reference_x,
    reference_y,
    linestyle="--",
    linewidth=2,
    label=(
        "Poisson reference "
        f"(mean IAT={reference_mean_iat:.3f}s)"
    ),
)


# ============================================================
# Figure formatting
# ============================================================

plt.xlabel("Inter-arrival time (seconds)")
plt.ylabel("Cumulative probability")
plt.title(
    "Inter-arrival-Time CDFs for Top Azure Applications"
)

if CDF_X_LIMIT_SECONDS is not None:
    plt.xlim(0, CDF_X_LIMIT_SECONDS)

plt.ylim(0, 1.01)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FIGURE,
    dpi=300,
)

plt.show()

print(f"\nSaved figure: {OUTPUT_FIGURE}")
print(f"Saved summary: {OUTPUT_SUMMARY}")