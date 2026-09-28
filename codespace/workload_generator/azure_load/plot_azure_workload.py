from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

WORKLOAD_FILE = Path("azure_bursty_500ms.csv")

# Display only the first N requests so the pattern is readable.
NUMBER_OF_REQUESTS_TO_PLOT = 200


# ------------------------------------------------------------
# Load workload
# ------------------------------------------------------------

if not WORKLOAD_FILE.exists():
    raise FileNotFoundError(
        f"Workload file not found: {WORKLOAD_FILE.resolve()}"
    )

df = pd.read_csv(WORKLOAD_FILE)

required_columns = {
    "request_id",
    "original_iat_seconds",
    "scaled_iat_ms",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

plot_df = df.head(NUMBER_OF_REQUESTS_TO_PLOT).copy()

plot_df["scaled_iat_seconds"] = (
    plot_df["scaled_iat_ms"] / 1000.0
)


# ------------------------------------------------------------
# Print summary
# ------------------------------------------------------------

print(f"Workload file: {WORKLOAD_FILE}")
print(f"Total IAT values: {len(df):,}")

print("\nOriginal IAT statistics:")
print(df["original_iat_seconds"].describe())

print("\nScaled IAT statistics in milliseconds:")
print(df["scaled_iat_ms"].describe())


# ------------------------------------------------------------
# Figure 1: Original IAT sequence
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    plot_df["request_id"],
    plot_df["original_iat_seconds"],
    marker=".",
    linewidth=1,
)

plt.xlabel("Request number")
plt.ylabel("Original IAT (seconds)")
plt.title(
    f"Original Azure IAT Pattern: First "
    f"{len(plot_df)} Requests"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "azure_original_iat_pattern.png",
    dpi=300,
)

plt.show()


# ------------------------------------------------------------
# Figure 2: Scaled IAT sequence
# ------------------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    plot_df["request_id"],
    plot_df["scaled_iat_ms"],
    marker=".",
    linewidth=1,
)

plt.xlabel("Request number")
plt.ylabel("Scaled IAT (ms)")
plt.title(
    f"Scaled Azure IAT Pattern: First "
    f"{len(plot_df)} Requests"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "azure_scaled_iat_pattern.png",
    dpi=300,
)

plt.show()


# ------------------------------------------------------------
# Figure 3: Cumulative request-arrival timeline
# ------------------------------------------------------------

plot_df["arrival_time_seconds"] = (
    plot_df["scaled_iat_seconds"].cumsum()
)

plt.figure(figsize=(12, 4))

plt.scatter(
    plot_df["arrival_time_seconds"],
    np.ones(len(plot_df)),
    marker="|",
    s=150,
)

plt.xlabel("Time since workload start (seconds)")
plt.yticks([])
plt.title(
    f"Scaled Azure Request-Arrival Timeline: First "
    f"{len(plot_df)} Requests"
)
plt.grid(True, axis="x", alpha=0.3)
plt.tight_layout()

plt.savefig(
    "azure_scaled_arrival_timeline.png",
    dpi=300,
)

plt.show()


# ------------------------------------------------------------
# Figure 4: Requests per second
# ------------------------------------------------------------

all_scaled_iat_seconds = (
    df["scaled_iat_ms"].to_numpy(dtype=float) / 1000.0
)

all_arrival_times = np.cumsum(
    all_scaled_iat_seconds
)

arrival_seconds = np.floor(
    all_arrival_times
).astype(int)

requests_per_second = (
    pd.Series(arrival_seconds)
    .value_counts()
    .sort_index()
)

# Include seconds with zero requests.
full_seconds = np.arange(
    0,
    arrival_seconds.max() + 1,
)

requests_per_second = (
    requests_per_second
    .reindex(full_seconds, fill_value=0)
)

plt.figure(figsize=(12, 5))

plt.plot(
    requests_per_second.index,
    requests_per_second.values,
    linewidth=1,
)

plt.xlabel("Time since workload start (seconds)")
plt.ylabel("Requests per second")
plt.title("Scaled Azure Workload over Time")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    "azure_scaled_requests_per_second.png",
    dpi=300,
)

plt.show()