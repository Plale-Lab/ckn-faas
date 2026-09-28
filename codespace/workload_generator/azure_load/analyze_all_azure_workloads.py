from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ============================================================
# Configuration
# ============================================================

TRACE_FILE = Path(
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

NUMBER_OF_CLUSTERS = 4

# Ignore applications with too few requests.
MIN_INVOCATIONS = 100

# Used only for calculating workload characteristics.
SHORT_IAT_SECONDS = 0.1
LONG_IDLE_SECONDS = 10.0

# Limit for the representative CDF figure.
CDF_X_LIMIT_SECONDS = 10.0

RANDOM_STATE = 42

FEATURES_OUTPUT = Path(
    "azure_app_workload_features.csv"
)

CLUSTER_SUMMARY_OUTPUT = Path(
    "azure_workload_cluster_summary.csv"
)

REPRESENTATIVE_OUTPUT = Path(
    "azure_representative_apps.csv"
)

CLUSTER_FIGURE = Path(
    "azure_workload_clusters.png"
)

CDF_FIGURE = Path(
    "azure_representative_cdfs.png"
)


# ============================================================
# Utility functions
# ============================================================

def coefficient_of_variation(values: pd.Series) -> float:
    """Calculate standard deviation divided by mean."""
    mean_value = values.mean()

    if mean_value <= 0:
        return np.nan

    return float(values.std(ddof=0) / mean_value)


def empirical_cdf(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return x and y values for an empirical CDF."""
    values = np.asarray(values, dtype=float)

    values = values[
        np.isfinite(values) &
        (values >= 0)
    ]

    if values.size == 0:
        raise ValueError("No valid values for CDF.")

    x = np.sort(values)
    y = np.arange(1, len(x) + 1) / len(x)

    return x, y


def shorten_app_id(app_id: str) -> str:
    """Shortened application ID for figures and console output."""
    app_id = str(app_id)
    return f"{app_id[:10]}..."


# ============================================================
# Load trace
# ============================================================

if not TRACE_FILE.exists():
    raise FileNotFoundError(
        f"Trace file not found: {TRACE_FILE.resolve()}"
    )

print("Loading Azure trace...")

df = pd.read_csv(
    TRACE_FILE,
    dtype={
        "app": "string",
        "func": "string",
    },
)

required_columns = {
    "app",
    "func",
    "end_timestamp",
    "duration",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing columns: {sorted(missing_columns)}"
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
        "end_timestamp",
        "duration",
    ]
).copy()

df = df[
    (df["end_timestamp"] >= 0) &
    (df["duration"] >= 0)
].copy()

print(f"Valid invocations: {len(df):,}")
print(f"Applications: {df['app'].nunique():,}")


# ============================================================
# Reconstruct invocation start times
# ============================================================

# Approximate invocation arrival/start time:
#
# start time = completion time - duration
#
# This is the invocation start time visible from the trace.
# It may not include queueing before Azure began execution.

df["arrival_time"] = (
    df["end_timestamp"] -
    df["duration"]
)

df = df[
    np.isfinite(df["arrival_time"]) &
    (df["arrival_time"] >= 0)
].copy()


# ============================================================
# Sort and calculate IATs
# ============================================================

print("Sorting invocations and calculating IATs...")

df = df.sort_values(
    by=["app", "arrival_time"],
    kind="mergesort",
)

df["iat_seconds"] = (
    df.groupby("app")["arrival_time"]
    .diff()
)

# Remove the first request of each application and invalid IATs.
iat_df = df.dropna(
    subset=["iat_seconds"]
).copy()

iat_df = iat_df[
    np.isfinite(iat_df["iat_seconds"]) &
    (iat_df["iat_seconds"] >= 0)
].copy()


# ============================================================
# Invocation counts
# ============================================================

invocation_counts = (
    df.groupby("app")
    .size()
    .rename("invocations")
)


# ============================================================
# Calculate workload features
# ============================================================

print("Calculating workload features...")

grouped_iats = iat_df.groupby("app")["iat_seconds"]

features_df = grouped_iats.agg(
    iat_count="count",
    mean_iat_seconds="mean",
    median_iat_seconds="median",
    std_iat_seconds="std",
    min_iat_seconds="min",
    max_iat_seconds="max",
).reset_index()

features_df["p90_iat_seconds"] = (
    grouped_iats.quantile(0.90).values
)

features_df["p95_iat_seconds"] = (
    grouped_iats.quantile(0.95).values
)

features_df["p99_iat_seconds"] = (
    grouped_iats.quantile(0.99).values
)

cv_values = grouped_iats.apply(
    coefficient_of_variation
).rename("iat_cv")

features_df = features_df.merge(
    cv_values,
    on="app",
    how="left",
)

features_df = features_df.merge(
    invocation_counts,
    on="app",
    how="left",
)


# ============================================================
# Additional workload characteristics
# ============================================================

zero_iat_fraction = (
    iat_df.assign(
        is_zero=iat_df["iat_seconds"] == 0
    )
    .groupby("app")["is_zero"]
    .mean()
    .rename("zero_iat_fraction")
)

short_iat_fraction = (
    iat_df.assign(
        is_short=(
            iat_df["iat_seconds"] <=
            SHORT_IAT_SECONDS
        )
    )
    .groupby("app")["is_short"]
    .mean()
    .rename("short_iat_fraction")
)

long_idle_fraction = (
    iat_df.assign(
        is_long_idle=(
            iat_df["iat_seconds"] >=
            LONG_IDLE_SECONDS
        )
    )
    .groupby("app")["is_long_idle"]
    .mean()
    .rename("long_idle_fraction")
)

features_df = features_df.merge(
    zero_iat_fraction,
    on="app",
    how="left",
)

features_df = features_df.merge(
    short_iat_fraction,
    on="app",
    how="left",
)

features_df = features_df.merge(
    long_idle_fraction,
    on="app",
    how="left",
)


# ============================================================
# Arrival-rate calculations
# ============================================================

features_df[
    "mean_arrival_rate_requests_per_second"
] = np.where(
    features_df["mean_iat_seconds"] > 0,
    1.0 / features_df["mean_iat_seconds"],
    np.nan,
)

# A simple burstiness index:
#
# B = (CV - 1) / (CV + 1)
#
# Approximately:
# B near -1 : very regular
# B near  0 : Poisson-like variability
# B near  1 : highly variable/bursty

features_df["burstiness_index"] = (
    (features_df["iat_cv"] - 1.0) /
    (features_df["iat_cv"] + 1.0)
)


# ============================================================
# Filter applications for clustering
# ============================================================

features_df = features_df[
    features_df["invocations"] >= MIN_INVOCATIONS
].copy()

features_df = features_df.replace(
    [np.inf, -np.inf],
    np.nan,
)

features_df = features_df.dropna(
    subset=[
        "mean_iat_seconds",
        "median_iat_seconds",
        "p95_iat_seconds",
        "iat_cv",
        "short_iat_fraction",
        "long_idle_fraction",
    ]
).copy()

if len(features_df) < NUMBER_OF_CLUSTERS:
    raise ValueError(
        "Not enough applications for the requested number "
        "of workload clusters."
    )

print(
    f"Applications retained for clustering: "
    f"{len(features_df):,}"
)


# ============================================================
# Prepare clustering features
# ============================================================

# Log transforms prevent very large values from dominating.
features_df["log_mean_iat"] = np.log1p(
    features_df["mean_iat_seconds"]
)

features_df["log_median_iat"] = np.log1p(
    features_df["median_iat_seconds"]
)

features_df["log_p95_iat"] = np.log1p(
    features_df["p95_iat_seconds"]
)

features_df["log_iat_cv"] = np.log1p(
    features_df["iat_cv"]
)

clustering_columns = [
    "log_mean_iat",
    "log_median_iat",
    "log_p95_iat",
    "log_iat_cv",
    "short_iat_fraction",
    "long_idle_fraction",
]

x = features_df[
    clustering_columns
].to_numpy(dtype=float)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# ============================================================
# K-means clustering
# ============================================================

print(
    f"Clustering applications into "
    f"{NUMBER_OF_CLUSTERS} workload classes..."
)

kmeans = KMeans(
    n_clusters=NUMBER_OF_CLUSTERS,
    random_state=RANDOM_STATE,
    n_init=20,
)

raw_cluster_ids = kmeans.fit_predict(x_scaled)

features_df["raw_cluster"] = raw_cluster_ids


# ============================================================
# Rename clusters A, B, C, D
# ============================================================

# K-means cluster numbers are arbitrary.
#
# We order clusters by average mean IAT:
#
# A = shortest average IAT / highest request rate
# D = longest average IAT / lowest request rate

cluster_order = (
    features_df.groupby("raw_cluster")[
        "mean_iat_seconds"
    ]
    .mean()
    .sort_values()
    .index
    .tolist()
)

class_names = [
    chr(ord("A") + index)
    for index in range(NUMBER_OF_CLUSTERS)
]

cluster_name_map = {
    raw_cluster: class_name
    for raw_cluster, class_name in zip(
        cluster_order,
        class_names,
    )
}

features_df["workload_class"] = (
    features_df["raw_cluster"]
    .map(cluster_name_map)
)


# ============================================================
# Find representative application for each cluster
# ============================================================

# Distance from each application to its assigned centroid.
distances = kmeans.transform(x_scaled)

assigned_distances = distances[
    np.arange(len(features_df)),
    raw_cluster_ids,
]

features_df["distance_to_centroid"] = (
    assigned_distances
)

representative_rows = []

for class_name in class_names:
    class_df = features_df[
        features_df["workload_class"] == class_name
    ]

    if class_df.empty:
        continue

    representative = class_df.loc[
        class_df["distance_to_centroid"].idxmin()
    ]

    representative_rows.append(
        representative
    )

representative_df = pd.DataFrame(
    representative_rows
).sort_values("workload_class")


# ============================================================
# Cluster summary
# ============================================================

cluster_summary = (
    features_df.groupby("workload_class")
    .agg(
        applications=("app", "count"),
        total_invocations=("invocations", "sum"),
        mean_iat_seconds=(
            "mean_iat_seconds",
            "mean",
        ),
        median_iat_seconds=(
            "median_iat_seconds",
            "median",
        ),
        mean_arrival_rate=(
            "mean_arrival_rate_requests_per_second",
            "mean",
        ),
        mean_iat_cv=("iat_cv", "mean"),
        mean_burstiness_index=(
            "burstiness_index",
            "mean",
        ),
        mean_short_iat_fraction=(
            "short_iat_fraction",
            "mean",
        ),
        mean_long_idle_fraction=(
            "long_idle_fraction",
            "mean",
        ),
    )
    .reset_index()
    .sort_values("workload_class")
)


# ============================================================
# Add simple descriptive labels
# ============================================================

description_map = {}

median_rate = cluster_summary[
    "mean_arrival_rate"
].median()

median_cv = cluster_summary[
    "mean_iat_cv"
].median()

for _, row in cluster_summary.iterrows():
    high_rate = (
        row["mean_arrival_rate"] >= median_rate
    )

    high_variability = (
        row["mean_iat_cv"] >= median_cv
    )

    if high_rate and high_variability:
        description = "High-rate, variable"
    elif high_rate and not high_variability:
        description = "High-rate, more regular"
    elif not high_rate and high_variability:
        description = "Low-rate, variable/idle-heavy"
    else:
        description = "Low-rate, more regular"

    description_map[
        row["workload_class"]
    ] = description

cluster_summary["description"] = (
    cluster_summary["workload_class"]
    .map(description_map)
)

features_df["workload_description"] = (
    features_df["workload_class"]
    .map(description_map)
)

representative_df["workload_description"] = (
    representative_df["workload_class"]
    .map(description_map)
)


# ============================================================
# Save CSV files
# ============================================================

output_columns = [
    "app",
    "workload_class",
    "workload_description",
    "invocations",
    "iat_count",
    "mean_iat_seconds",
    "median_iat_seconds",
    "p90_iat_seconds",
    "p95_iat_seconds",
    "p99_iat_seconds",
    "std_iat_seconds",
    "iat_cv",
    "burstiness_index",
    "mean_arrival_rate_requests_per_second",
    "zero_iat_fraction",
    "short_iat_fraction",
    "long_idle_fraction",
    "distance_to_centroid",
]

features_df[
    output_columns
].sort_values(
    by=[
        "workload_class",
        "distance_to_centroid",
    ]
).to_csv(
    FEATURES_OUTPUT,
    index=False,
)

cluster_summary.to_csv(
    CLUSTER_SUMMARY_OUTPUT,
    index=False,
)

representative_df[
    output_columns
].to_csv(
    REPRESENTATIVE_OUTPUT,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print("\nCluster summary:")

print(
    cluster_summary[
        [
            "workload_class",
            "description",
            "applications",
            "mean_iat_seconds",
            "mean_arrival_rate",
            "mean_iat_cv",
            "mean_burstiness_index",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)

print("\nRepresentative applications:")

print(
    representative_df[
        [
            "workload_class",
            "workload_description",
            "app",
            "invocations",
            "mean_iat_seconds",
            "mean_arrival_rate_requests_per_second",
            "iat_cv",
            "burstiness_index",
        ]
    ].to_string(
        index=False,
        float_format=lambda value: f"{value:.4f}",
    )
)


# ============================================================
# Figure 1: Cluster scatter plot
# ============================================================

plt.figure(figsize=(10, 6))

for class_name in class_names:
    class_df = features_df[
        features_df["workload_class"] == class_name
    ]

    plt.scatter(
        class_df[
            "mean_arrival_rate_requests_per_second"
        ],
        class_df["iat_cv"],
        label=(
            f"Class {class_name}: "
            f"{description_map.get(class_name, '')}"
        ),
        alpha=0.75,
        s=60,
    )

for _, row in representative_df.iterrows():
    plt.annotate(
        f"Representative {row['workload_class']}",
        (
            row[
                "mean_arrival_rate_requests_per_second"
            ],
            row["iat_cv"],
        ),
        xytext=(6, 6),
        textcoords="offset points",
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Mean arrival rate (requests/second)")
plt.ylabel("IAT coefficient of variation")
plt.title(
    "Azure Application Workload Classes"
)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CLUSTER_FIGURE,
    dpi=300,
)

plt.close()


# ============================================================
# Figure 2: Representative CDFs
# ============================================================

plt.figure(figsize=(10, 6))

for _, representative in representative_df.iterrows():
    app_id = representative["app"]
    class_name = representative["workload_class"]

    app_iats = iat_df.loc[
        iat_df["app"] == app_id,
        "iat_seconds",
    ].to_numpy(dtype=float)

    if len(app_iats) == 0:
        continue

    x_cdf, y_cdf = empirical_cdf(app_iats)

    plt.plot(
        x_cdf,
        y_cdf,
        linewidth=2,
        label=(
            f"Class {class_name}: "
            f"{description_map[class_name]}"
        ),
    )

plt.xlabel("Inter-arrival time (seconds)")
plt.ylabel("Cumulative probability")
plt.title(
    "Representative Azure Workload Classes"
)
plt.xlim(0, CDF_X_LIMIT_SECONDS)
plt.ylim(0, 1.01)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    CDF_FIGURE,
    dpi=300,
)

plt.close()


# ============================================================
# Completion message
# ============================================================

print("\nCreated files:")

for output_path in [
    FEATURES_OUTPUT,
    CLUSTER_SUMMARY_OUTPUT,
    REPRESENTATIVE_OUTPUT,
    CLUSTER_FIGURE,
    CDF_FIGURE,
]:
    print(f"  {output_path}")