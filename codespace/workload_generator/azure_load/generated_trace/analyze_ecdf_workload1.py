import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
CSV_PATH = "chosen-ecdf-5scal.csv"
OUTPUT_DIR = "overall_workload_plots_5scal"

# Arrival-rate window size in seconds
WINDOW_SEC = 5


# -----------------------------
# Load and prepare the trace
# -----------------------------
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Trace file not found: {CSV_PATH}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

required_columns = {"func_name", "invoke_time_ms"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

# Sort all requests together by invocation time
df = df.sort_values("invoke_time_ms").reset_index(drop=True)

# Convert milliseconds to seconds
df["invoke_time_sec"] = df["invoke_time_ms"] / 1000.0

print(f"Total requests: {len(df)}")
print(f"Unique functions: {df['func_name'].nunique()}")
print(
    f"Workload duration: "
    f"{df['invoke_time_sec'].max() - df['invoke_time_sec'].min():.2f} seconds"
)


# ============================================================
# Plot 1: Overall request arrival rate over time
# ============================================================

# Assign every request to a fixed time window
df["time_window"] = (
    np.floor(df["invoke_time_sec"] / WINDOW_SEC) * WINDOW_SEC
)

arrival_counts = (
    df.groupby("time_window")
    .size()
    .reset_index(name="request_count")
)

# Include empty windows so idle periods are visible
start_time = 0
end_time = np.ceil(df["invoke_time_sec"].max() / WINDOW_SEC) * WINDOW_SEC

all_windows = pd.DataFrame(
    {
        "time_window": np.arange(
            start_time,
            end_time + WINDOW_SEC,
            WINDOW_SEC
        )
    }
)

arrival_counts = all_windows.merge(
    arrival_counts,
    on="time_window",
    how="left"
)

arrival_counts["request_count"] = (
    arrival_counts["request_count"].fillna(0)
)

# Convert counts per window into requests per second
arrival_counts["arrival_rate_rps"] = (
    arrival_counts["request_count"] / WINDOW_SEC
)

plt.figure(figsize=(10, 5))
plt.plot(
    arrival_counts["time_window"],
    arrival_counts["arrival_rate_rps"],
    marker="o",
    markersize=3
)

plt.xlabel("Elapsed Time (seconds)")
plt.ylabel("Arrival Rate (requests/second)")
plt.title(
    f"Overall Workload Arrival Rate "
    # f"({WINDOW_SEC}-Second Windows)"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()

arrival_plot_path = os.path.join(
    OUTPUT_DIR,
    "overall_arrival_rate.png"
)

plt.savefig(arrival_plot_path, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Plot 2: Overall inter-arrival time CDF
# ============================================================

# Compute IAT between consecutive requests in the merged trace
df["iat_ms"] = df["invoke_time_ms"].diff()

iat_values = (
    df["iat_ms"]
    .dropna()
    .loc[lambda x: x >= 0]
    .sort_values()
    .to_numpy()
)

if len(iat_values) == 0:
    raise ValueError("No valid inter-arrival times were found.")

cdf = np.arange(1, len(iat_values) + 1) / len(iat_values)

plt.figure(figsize=(8, 5))
plt.plot(iat_values, cdf)

plt.xlabel("Overall Inter-Arrival Time (ms)")
plt.ylabel("CDF")
plt.title("Overall Workload Inter-Arrival Time CDF")
plt.grid(True, alpha=0.3)
plt.tight_layout()

iat_cdf_path = os.path.join(
    OUTPUT_DIR,
    "overall_iat_cdf.png"
)

plt.savefig(iat_cdf_path, dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# Print overall workload statistics
# ============================================================

duration_sec = (
    df["invoke_time_sec"].max()
    - df["invoke_time_sec"].min()
)

overall_rate = (
    len(df) / duration_sec
    if duration_sec > 0
    else float("nan")
)

print("\nOverall workload statistics")
print("---------------------------")
print(f"Total requests: {len(df)}")
print(f"Duration: {duration_sec:.2f} seconds")
print(f"Average arrival rate: {overall_rate:.3f} requests/second")
print(f"Mean IAT: {np.mean(iat_values):.3f} ms")
print(f"Median IAT: {np.median(iat_values):.3f} ms")
print(f"P90 IAT: {np.percentile(iat_values, 90):.3f} ms")
print(f"P95 IAT: {np.percentile(iat_values, 95):.3f} ms")
print(f"P99 IAT: {np.percentile(iat_values, 99):.3f} ms")

print("\nSaved plots:")
print(arrival_plot_path)
print(iat_cdf_path)