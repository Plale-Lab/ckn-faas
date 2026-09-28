import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "generated_trace/chosen-ecdf.csv"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "generated_trace/analysis5"
)

ARRIVAL_WINDOW_SEC = 10


# ============================================================
# Helper functions
# ============================================================

def validate_input(df: pd.DataFrame) -> None:
    required_columns = {"func_name", "invoke_time_ms"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}. "
            f"Found columns: {list(df.columns)}"
        )

    if df.empty:
        raise ValueError("The input workload file is empty.")


def calculate_burstiness(cv: float) -> float:
    """
    Burstiness index:
        B = (CV - 1) / (CV + 1)

    B approaches:
      -1 for highly regular arrivals
       0 for Poisson-like arrivals
       1 for highly bursty arrivals
    """
    if not np.isfinite(cv) or cv == -1:
        return np.nan

    return (cv - 1) / (cv + 1)


def save_current_figure(filename: str) -> None:
    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ============================================================
# Load and prepare workload
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
validate_input(df)

df["invoke_time_ms"] = pd.to_numeric(
    df["invoke_time_ms"],
    errors="raise",
)

df = (
    df.sort_values("invoke_time_ms")
    .reset_index(drop=True)
)

df["time_sec"] = df["invoke_time_ms"] / 1000.0

# Overall IAT across the combined workload.
# This measures the gap between any two consecutive requests.
df["global_iat_ms"] = df["invoke_time_ms"].diff()

# Per-function IAT.
# This is the correct IAT when analyzing each function individually.
df["function_iat_ms"] = (
    df.groupby("func_name")["invoke_time_ms"].diff()
)


# ============================================================
# Overall workload summary
# ============================================================

total_requests = len(df)
unique_functions = df["func_name"].nunique()

start_ms = float(df["invoke_time_ms"].min())
end_ms = float(df["invoke_time_ms"].max())
duration_sec = max((end_ms - start_ms) / 1000.0, 0.0)

if duration_sec > 0:
    mean_arrival_rate = total_requests / duration_sec
else:
    mean_arrival_rate = np.nan

global_iat = df["global_iat_ms"].dropna()

global_mean_iat = global_iat.mean()
global_median_iat = global_iat.median()
global_p90_iat = global_iat.quantile(0.90)
global_p95_iat = global_iat.quantile(0.95)
global_p99_iat = global_iat.quantile(0.99)
global_std_iat = global_iat.std()

if global_mean_iat > 0:
    global_cv = global_std_iat / global_mean_iat
else:
    global_cv = np.nan

global_burstiness = calculate_burstiness(global_cv)

summary_df = pd.DataFrame(
    {
        "metric": [
            "total_requests",
            "unique_functions",
            "duration_seconds",
            "mean_arrival_rate_rps",
            "global_mean_iat_ms",
            "global_median_iat_ms",
            "global_p90_iat_ms",
            "global_p95_iat_ms",
            "global_p99_iat_ms",
            "global_iat_std_ms",
            "global_iat_cv",
            "global_burstiness_index",
        ],
        "value": [
            total_requests,
            unique_functions,
            duration_sec,
            mean_arrival_rate,
            global_mean_iat,
            global_median_iat,
            global_p90_iat,
            global_p95_iat,
            global_p99_iat,
            global_std_iat,
            global_cv,
            global_burstiness,
        ],
    }
)

summary_path = os.path.join(
    OUTPUT_DIR,
    "workload_summary.csv",
)
summary_df.to_csv(summary_path, index=False)

print("\n========== Overall Workload Summary ==========")
print(summary_df.to_string(index=False))
print(f"\nSaved: {summary_path}")


# ============================================================
# Per-function statistics
# ============================================================

function_rows = []

for func_name, group in df.groupby("func_name"):
    group = group.sort_values("invoke_time_ms")
    iat = group["function_iat_ms"].dropna()

    request_count = len(group)

    first_time_ms = float(group["invoke_time_ms"].min())
    last_time_ms = float(group["invoke_time_ms"].max())
    active_duration_sec = max(
        (last_time_ms - first_time_ms) / 1000.0,
        0.0,
    )

    if active_duration_sec > 0:
        function_rate = request_count / active_duration_sec
    else:
        function_rate = np.nan

    mean_iat = iat.mean()
    median_iat = iat.median()
    p90_iat = iat.quantile(0.90)
    p95_iat = iat.quantile(0.95)
    p99_iat = iat.quantile(0.99)
    std_iat = iat.std()

    if mean_iat > 0:
        cv = std_iat / mean_iat
    else:
        cv = np.nan

    burstiness = calculate_burstiness(cv)

    function_rows.append(
        {
            "func_name": func_name,
            "request_count": request_count,
            "request_fraction": request_count / total_requests,
            "first_invoke_ms": first_time_ms,
            "last_invoke_ms": last_time_ms,
            "active_duration_sec": active_duration_sec,
            "mean_arrival_rate_rps": function_rate,
            "mean_iat_ms": mean_iat,
            "median_iat_ms": median_iat,
            "p90_iat_ms": p90_iat,
            "p95_iat_ms": p95_iat,
            "p99_iat_ms": p99_iat,
            "iat_std_ms": std_iat,
            "iat_cv": cv,
            "burstiness_index": burstiness,
        }
    )

function_stats = pd.DataFrame(function_rows)
function_stats = function_stats.sort_values(
    "request_count",
    ascending=False,
)

function_stats_path = os.path.join(
    OUTPUT_DIR,
    "per_function_statistics.csv",
)
function_stats.to_csv(
    function_stats_path,
    index=False,
)

print("\n========== Per-Function Statistics ==========")
print(function_stats.to_string(index=False))
print(f"\nSaved: {function_stats_path}")


# ============================================================
# Plot 1: Requests per function
# ============================================================

request_counts = (
    df["func_name"]
    .value_counts()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10, 5))
plt.bar(
    request_counts.index,
    request_counts.values,
)
plt.xlabel("Function")
plt.ylabel("Number of requests")
plt.title("Requests per Function")
plt.xticks(rotation=45, ha="right")

save_current_figure("requests_per_function.png")


# ============================================================
# Plot 2: Request timeline
# ============================================================

function_order = request_counts.index.tolist()

function_to_y = {
    func_name: index
    for index, func_name in enumerate(function_order)
}

timeline_y = df["func_name"].map(function_to_y)

plt.figure(figsize=(13, 5))
plt.scatter(
    df["time_sec"],
    timeline_y,
    s=10,
    alpha=0.7,
)

plt.yticks(
    range(len(function_order)),
    function_order,
)

plt.xlabel("Elapsed workload time (seconds)")
plt.ylabel("Function")
plt.title("Request Timeline")

save_current_figure("request_timeline.png")


# ============================================================
# Plot 3: Arrival rate in fixed windows
# ============================================================

max_time_sec = df["time_sec"].max()

bin_edges = np.arange(
    0,
    max_time_sec + ARRIVAL_WINDOW_SEC,
    ARRIVAL_WINDOW_SEC,
)

if len(bin_edges) < 2:
    bin_edges = np.array(
        [0, ARRIVAL_WINDOW_SEC],
        dtype=float,
    )

window_labels = bin_edges[:-1]

arrival_counts, _ = np.histogram(
    df["time_sec"],
    bins=bin_edges,
)

arrival_rate = (
    arrival_counts / ARRIVAL_WINDOW_SEC
)

plt.figure(figsize=(12, 5))
plt.plot(
    window_labels,
    arrival_rate,
    marker="o",
    markersize=3,
)

plt.xlabel("Elapsed workload time (seconds)")
plt.ylabel("Arrival rate (requests/second)")
plt.title(
    f"Arrival Rate Using {ARRIVAL_WINDOW_SEC}-Second Windows"
)

save_current_figure("arrival_rate_windowed.png")


# ============================================================
# Plot 4: Per-function IAT CDF
# ============================================================

plt.figure(figsize=(10, 6))

for func_name in function_order:
    iat = (
        df.loc[
            df["func_name"] == func_name,
            "function_iat_ms",
        ]
        .dropna()
        .to_numpy()
    )

    if len(iat) == 0:
        continue

    sorted_iat = np.sort(iat)
    cdf = (
        np.arange(1, len(sorted_iat) + 1)
        / len(sorted_iat)
    )

    plt.step(
        sorted_iat,
        cdf,
        where="post",
        label=func_name,
    )

plt.xlabel("Per-function inter-arrival time (ms)")
plt.ylabel("CDF")
plt.title("Per-Function CDF of Inter-Arrival Times")
plt.legend(title="Function")

save_current_figure("per_function_iat_cdf.png")


# ============================================================
# Plot 5: Global IAT CDF
# ============================================================

if len(global_iat) > 0:
    sorted_global_iat = np.sort(
        global_iat.to_numpy()
    )

    global_cdf = (
        np.arange(1, len(sorted_global_iat) + 1)
        / len(sorted_global_iat)
    )

    plt.figure(figsize=(9, 5))
    plt.step(
        sorted_global_iat,
        global_cdf,
        where="post",
    )

    plt.xlabel(
        "Inter-arrival time between consecutive workload requests (ms)"
    )
    plt.ylabel("CDF")
    plt.title("CDF of Combined Workload Inter-Arrival Times")

    save_current_figure("global_iat_cdf.png")


# ============================================================
# Plot 6: Per-function arrival rate over time
# ============================================================

plt.figure(figsize=(12, 6))

for func_name in function_order:
    function_times = df.loc[
        df["func_name"] == func_name,
        "time_sec",
    ]

    counts, _ = np.histogram(
        function_times,
        bins=bin_edges,
    )

    rate = counts / ARRIVAL_WINDOW_SEC

    plt.plot(
        window_labels,
        rate,
        label=func_name,
    )

plt.xlabel("Elapsed workload time (seconds)")
plt.ylabel("Arrival rate (requests/second)")
plt.title(
    f"Per-Function Arrival Rate Using "
    f"{ARRIVAL_WINDOW_SEC}-Second Windows"
)
plt.legend(title="Function")

save_current_figure("per_function_arrival_rate.png")


print("\nAnalysis completed successfully.")
print(f"All results are available in:\n{OUTPUT_DIR}")