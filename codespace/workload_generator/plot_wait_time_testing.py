import os
import re
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT CSV
# ============================================================

CSV_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/"
    # "/Users/agamage/Desktop/D2I/Codes Original/Fass Edge Cascades/"
    "2models_iluvatar_latency_wait_snapshot_iat_100ms_requests_100.csv"
)


# ============================================================
# OUTPUT FOLDER
# ============================================================

OUTPUT_DIR = "plots"

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# Read CSV
# ============================================================

df = pd.read_csv(
    CSV_FILE
)


# ============================================================
# Basic checks
# ============================================================

required_columns = [
    "request_number",
    "model_name",
    "wall_latency_ms",
    "rpc_latency_ms",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# Find monitored model prefixes
#
# Example column:
#
# edge_1__resnet34__cpu_wait_ms
#
# Prefix:
#
# edge_1__resnet34
# ============================================================

def find_wait_prefixes(
    dataframe,
):

    prefixes = set()

    suffixes = [
        "__cpu_wait_ms",
        "__gpu_wait_ms",
        "__selected_wait_ms",
    ]

    for column in dataframe.columns:

        for suffix in suffixes:

            if column.endswith(
                suffix
            ):

                prefix = column[
                    :-len(suffix)
                ]

                prefixes.add(
                    prefix
                )

    return sorted(
        prefixes
    )


WAIT_PREFIXES = (
    find_wait_prefixes(
        df
    )
)


print(
    "\nDetected monitored deployments:"
)

for prefix in WAIT_PREFIXES:

    print(
        f"  {prefix}"
    )


# ============================================================
# Human readable label
# ============================================================

def pretty_label(prefix):

    parts = prefix.split("__")

    if len(parts) >= 2:
        return "__".join(parts[1:])

    return prefix

# ============================================================
# Determine best wait column
#
# Prefer:
# selected_wait_ms
#
# otherwise CPU
#
# otherwise GPU
# ============================================================

def get_wait_column(
    dataframe,
    prefix,
):

    selected_column = (
        f"{prefix}"
        f"__selected_wait_ms"
    )

    cpu_column = (
        f"{prefix}"
        f"__cpu_wait_ms"
    )

    gpu_column = (
        f"{prefix}"
        f"__gpu_wait_ms"
    )

    if (
        selected_column
        in dataframe.columns
    ):

        return selected_column

    if (
        cpu_column
        in dataframe.columns
    ):

        return cpu_column

    if (
        gpu_column
        in dataframe.columns
    ):

        return gpu_column

    return None


# ============================================================
# Separate plots by actual executed model
#
# Example:
#
# resnet18 experiment
# resnet34 experiment
# resnet50 experiment
# ============================================================

EXECUTED_MODELS = (
    df[
        "model_name"
    ]
    .dropna()
    .unique()
)


# ============================================================
# GRAPH 1
#
# Estimated wait time of ALL candidate models
# vs request number
# ============================================================

for executed_model in (
    EXECUTED_MODELS
):

    model_df = (
        df[
            df[
                "model_name"
            ]
            == executed_model
        ]
        .sort_values(
            "request_number"
        )
    )

    plt.figure(
        figsize=(
            14,
            7,
        )
    )

    plotted = False

    for prefix in (
        WAIT_PREFIXES
    ):

        wait_column = (
            get_wait_column(
                model_df,
                prefix,
            )
        )

        if (
            wait_column is None
        ):
            continue

        values = pd.to_numeric(
            model_df[
                wait_column
            ],
            errors="coerce",
        )

        if values.notna().sum() == 0:
            continue

        plt.plot(
            model_df[
                "request_number"
            ],
            values,
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                pretty_label(
                    prefix
                )
            ),
        )

        plotted = True

    if plotted:

        plt.xlabel(
            "Request Number"
        )

        plt.ylabel(
            "Estimated Wait Time (ms)"
        )

        plt.title(
            f"Estimated Wait Time vs Request Number\n"
            f"Actual Workload Model: {executed_model}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend(
            fontsize=8,
            loc="best",
        )

        plt.tight_layout()

        output_file = os.path.join(
            OUTPUT_DIR,
            f"01_wait_time_vs_request_"
            f"{executed_model}.png",
        )

        plt.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

    plt.close()


# ============================================================
# GRAPH 2
#
# Actual E2E latency vs request number
# ============================================================

plt.figure(
    figsize=(
        13,
        7,
    )
)

for executed_model in (
    EXECUTED_MODELS
):

    model_df = (
        df[
            df[
                "model_name"
            ]
            == executed_model
        ]
        .sort_values(
            "request_number"
        )
    )

    plt.plot(
        model_df[
            "request_number"
        ],
        model_df[
            "wall_latency_ms"
        ],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=(
            executed_model
        ),
    )

plt.xlabel(
    "Request Number"
)

plt.ylabel(
    "Actual E2E Latency (ms)"
)

plt.title(
    "Actual End-to-End Latency vs Request Number"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "02_actual_e2e_latency_vs_request.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 3
#
# RPC latency vs request number
# ============================================================

plt.figure(
    figsize=(
        13,
        7,
    )
)

for executed_model in (
    EXECUTED_MODELS
):

    model_df = (
        df[
            df[
                "model_name"
            ]
            == executed_model
        ]
        .sort_values(
            "request_number"
        )
    )

    plt.plot(
        model_df[
            "request_number"
        ],
        model_df[
            "rpc_latency_ms"
        ],
        marker="o",
        markersize=3,
        linewidth=1.5,
        label=(
            executed_model
        ),
    )

plt.xlabel(
    "Request Number"
)

plt.ylabel(
    "RPC Latency (ms)"
)

plt.title(
    "Iluvatar RPC Latency vs Request Number"
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "03_rpc_latency_vs_request.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 4
#
# Average E2E latency by executed model
# ============================================================

avg_e2e = (
    df.groupby(
        "model_name"
    )[
        "wall_latency_ms"
    ]
    .mean()
    .sort_values()
)


plt.figure(
    figsize=(
        11,
        6,
    )
)

plt.bar(
    avg_e2e.index,
    avg_e2e.values,
)

plt.xlabel(
    "Executed Model"
)

plt.ylabel(
    "Average E2E Latency (ms)"
)

plt.title(
    "Average End-to-End Latency by Model"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "04_average_e2e_latency_by_model.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 5
#
# Average estimated wait time
# for each monitored model/deployment
# ============================================================

average_wait = {}

for prefix in (
    WAIT_PREFIXES
):

    wait_column = (
        get_wait_column(
            df,
            prefix,
        )
    )

    if wait_column is None:
        continue

    values = pd.to_numeric(
        df[
            wait_column
        ],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        continue

    average_wait[
        pretty_label(
            prefix
        )
    ] = values.mean()


average_wait_series = (
    pd.Series(
        average_wait
    )
    .sort_values()
)


plt.figure(
    figsize=(
        12,
        7,
    )
)

plt.barh(
    average_wait_series.index,
    average_wait_series.values,
)

plt.xlabel(
    "Average Estimated Wait Time (ms)"
)

plt.ylabel(
    "Monitored Model / Node"
)

plt.title(
    "Average Iluvatar Estimated Wait Time"
)

plt.grid(
    axis="x",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "05_average_wait_time_by_model.png",
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# GRAPH 6
#
# Average CPU waiting function count
# ============================================================

cpu_waiting_average = {}

for prefix in (
    WAIT_PREFIXES
):

    column = (
        f"{prefix}"
        f"__cpu_waiting"
    )

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        continue

    cpu_waiting_average[
        pretty_label(
            prefix
        )
    ] = values.mean()


if cpu_waiting_average:

    series = (
        pd.Series(
            cpu_waiting_average
        )
        .sort_values()
    )

    plt.figure(
        figsize=(
            12,
            7,
        )
    )

    plt.barh(
        series.index,
        series.values,
    )

    plt.xlabel(
        "Average CPU Waiting Functions"
    )

    plt.ylabel(
        "Monitored Model / Node"
    )

    plt.title(
        "Average CPU Queue Length by Model"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_average_cpu_waiting_functions.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# GRAPH 7
#
# Average CPU running function count
# ============================================================

cpu_running_average = {}

for prefix in (
    WAIT_PREFIXES
):

    column = (
        f"{prefix}"
        f"__cpu_running"
    )

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        continue

    cpu_running_average[
        pretty_label(
            prefix
        )
    ] = values.mean()


if cpu_running_average:

    series = (
        pd.Series(
            cpu_running_average
        )
        .sort_values()
    )

    plt.figure(
        figsize=(
            12,
            7,
        )
    )

    plt.barh(
        series.index,
        series.values,
    )

    plt.xlabel(
        "Average CPU Running Functions"
    )

    plt.ylabel(
        "Monitored Model / Node"
    )

    plt.title(
        "Average CPU Running Functions by Model"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "07_average_cpu_running_functions.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# GRAPH 8
#
# Actual E2E latency vs CURRENT EXECUTED MODEL'S
# estimated wait time
#
# This automatically finds the wait-time column
# corresponding to the model actually receiving the request.
# ============================================================

plt.figure(
    figsize=(
        10,
        7,
    )
)

scatter_created = False


for executed_model in (
    EXECUTED_MODELS
):

    model_df = (
        df[
            df[
                "model_name"
            ]
            == executed_model
        ]
        .copy()
    )

    matching_prefixes = [

        prefix

        for prefix in WAIT_PREFIXES

        if prefix.endswith(
            f"__{executed_model}"
        )
    ]

    if not matching_prefixes:
        continue


    # --------------------------------------------------------
    # If same model exists on multiple nodes, prefer the
    # node actually used for the inference request.
    # --------------------------------------------------------

    x_values = []
    y_values = []


    for _, row in (
        model_df.iterrows()
    ):

        actual_node = str(
            row[
                "node_id"
            ]
        )

        target_prefix = (
            f"{actual_node}"
            f"__{executed_model}"
        )

        if (
            target_prefix
            not in WAIT_PREFIXES
        ):

            target_prefix = (
                matching_prefixes[
                    0
                ]
            )

        wait_column = (
            get_wait_column(
                df,
                target_prefix,
            )
        )

        if wait_column is None:
            continue

        wait_value = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        wait_column
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        latency_value = pd.to_numeric(
            pd.Series(
                [
                    row.get(
                        "wall_latency_ms"
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]

        if (
            pd.isna(
                wait_value
            )
            or pd.isna(
                latency_value
            )
        ):
            continue

        x_values.append(
            wait_value
        )

        y_values.append(
            latency_value
        )


    if x_values:

        plt.scatter(
            x_values,
            y_values,
            alpha=0.7,
            label=(
                executed_model
            ),
        )

        scatter_created = True


if scatter_created:

    plt.xlabel(
        "Estimated Wait Time of Executed Model (ms)"
    )

    plt.ylabel(
        "Actual E2E Latency (ms)"
    )

    plt.title(
        "Estimated Wait Time vs Actual E2E Latency"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "08_wait_time_vs_actual_latency.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

plt.close()


# ============================================================
# GRAPH 9
#
# Estimator launch overhead vs request
# ============================================================

if (
    "estimator_launch_overhead_ms"
    in df.columns
):

    plt.figure(
        figsize=(
            13,
            7,
        )
    )

    for executed_model in (
        EXECUTED_MODELS
    ):

        model_df = (
            df[
                df[
                    "model_name"
                ]
                == executed_model
            ]
            .sort_values(
                "request_number"
            )
        )

        plt.plot(
            model_df[
                "request_number"
            ],
            model_df[
                "estimator_launch_overhead_ms"
            ],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=(
                executed_model
            ),
        )

    plt.xlabel(
        "Request Number"
    )

    plt.ylabel(
        "Estimator Launch Overhead (ms)"
    )

    plt.title(
        "Parallel Estimator Launch Overhead"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "09_estimator_launch_overhead.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# GRAPH 10
#
# Average estimator RPC query duration
# by monitored model
# ============================================================

query_average = {}

for prefix in (
    WAIT_PREFIXES
):

    column = (
        f"{prefix}"
        f"__estimator_query_ms"
    )

    if column not in df.columns:
        continue

    values = pd.to_numeric(
        df[
            column
        ],
        errors="coerce",
    )

    if values.notna().sum() == 0:
        continue

    query_average[
        pretty_label(
            prefix
        )
    ] = values.mean()


if query_average:

    query_series = (
        pd.Series(
            query_average
        )
        .sort_values()
    )

    plt.figure(
        figsize=(
            12,
            7,
        )
    )

    plt.barh(
        query_series.index,
        query_series.values,
    )

    plt.xlabel(
        "Average Estimator Query Time (ms)"
    )

    plt.ylabel(
        "Monitored Model / Node"
    )

    plt.title(
        "Average Iluvatar Estimator Query Time"
    )

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "10_average_estimator_query_time.png",
        ),
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Done
# ============================================================

print(
    "\nAll plots generated successfully."
)

print(
    f"Output directory: "
    f"{os.path.abspath(OUTPUT_DIR)}"
)

print(
    "\nGenerated plots:"
)

for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    if filename.endswith(
        ".png"
    ):

        print(
            f"  {filename}"
        )