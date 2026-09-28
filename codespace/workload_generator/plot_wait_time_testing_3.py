import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# INPUT
# ============================================================

CSV_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/"
    # "/Users/agamage/Desktop/D2I/Codes Original/Fass Edge Cascades/"
    "2models_iluvatar_latency_wait_snapshot_iat_100ms_requests_100.csv"
)

OUTPUT_DIR = "wait_resource_plots"

SHOW_PLOTS = True

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# READ DATA
# ============================================================

df = pd.read_csv(
    CSV_FILE
)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "request_number",
    "model_name",
    "node_id",
    "wall_latency_ms",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# FIND MONITORED PREFIXES
#
# Example:
#
# edge_1__resnet34__cpu_wait_ms
#
# prefix:
#
# edge_1__resnet34
# ============================================================

def find_prefixes(dataframe):

    suffixes = [
        "__cpu_wait_ms",
        "__gpu_wait_ms",
        "__cpu_waiting",
        "__cpu_running",
        "__gpu_waiting",
        "__gpu_running",
    ]

    prefixes = set()

    for column in dataframe.columns:

        for suffix in suffixes:

            if column.endswith(suffix):

                prefix = column[:-len(suffix)]

                prefixes.add(prefix)

    return sorted(prefixes)


PREFIXES = find_prefixes(df)


# ============================================================
# PRETTY LABEL
#
# edge_1__resnet34
#
# ->
#
# resnet34 (edge_1)
# ============================================================

def pretty_label(prefix):

    parts = prefix.split("__")

    if len(parts) >= 2:

        node = parts[0]

        model = "__".join(
            parts[1:]
        )

        return f"{model} ({node})"

    return prefix


# ============================================================
# SAFE NUMERIC COLUMN
# ============================================================

def numeric_column(
    dataframe,
    column,
):

    if column not in dataframe.columns:

        return None

    return pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )


# ============================================================
# EXECUTED MODELS
# ============================================================

executed_models = (
    df["model_name"]
    .dropna()
    .unique()
)


# ============================================================
# LOOP THROUGH EACH ACTUAL WORKLOAD MODEL
# ============================================================

for executed_model in executed_models:

    model_df = (
        df[
            df["model_name"]
            == executed_model
        ]
        .copy()
        .sort_values(
            "request_number"
        )
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Executed model: {executed_model}"
    )

    print(
        f"Requests: {len(model_df)}"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # GRAPH 1
    #
    # CPU WAIT TIME FOR ALL MONITORED MODELS
    # ========================================================

    plt.figure(
        figsize=(15, 8)
    )

    plotted = False


    for prefix in PREFIXES:

        column = (
            f"{prefix}"
            f"__cpu_wait_ms"
        )

        values = numeric_column(
            model_df,
            column,
        )

        if values is None:
            continue

        if values.notna().sum() == 0:
            continue


        plt.plot(
            model_df["request_number"],
            values,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=pretty_label(prefix),
        )

        plotted = True


    if plotted:

        plt.xlabel(
            "Request Number"
        )

        plt.ylabel(
            "CPU Estimated Wait Time (ms)"
        )

        plt.title(
            f"CPU Estimated Wait Time for Every Request\n"
            f"Actual Workload Model: {executed_model}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )

        plt.tight_layout()


        file_path = os.path.join(
            OUTPUT_DIR,
            f"01_cpu_wait_{executed_model}.png",
        )

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        if SHOW_PLOTS:
            plt.show()

    plt.close()


    # ========================================================
    # GRAPH 2
    #
    # GPU WAIT TIME FOR ALL MONITORED MODELS
    # ========================================================

    plt.figure(
        figsize=(15, 8)
    )

    plotted = False


    for prefix in PREFIXES:

        column = (
            f"{prefix}"
            f"__gpu_wait_ms"
        )

        values = numeric_column(
            model_df,
            column,
        )

        if values is None:
            continue

        if values.notna().sum() == 0:
            continue


        # Skip fully zero GPU data
        if (
            values.fillna(0)
            .abs()
            .sum()
            == 0
        ):
            continue


        plt.plot(
            model_df["request_number"],
            values,
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=pretty_label(prefix),
        )

        plotted = True


    if plotted:

        plt.xlabel(
            "Request Number"
        )

        plt.ylabel(
            "GPU Estimated Wait Time (ms)"
        )

        plt.title(
            f"GPU Estimated Wait Time for Every Request\n"
            f"Actual Workload Model: {executed_model}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend(
            fontsize=8,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
        )

        plt.tight_layout()


        file_path = os.path.join(
            OUTPUT_DIR,
            f"02_gpu_wait_{executed_model}.png",
        )

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        if SHOW_PLOTS:
            plt.show()

    plt.close()


    # ========================================================
    # GRAPH 3
    #
    # CPU WAITING / RUNNING
    #
    # ONLY FOR THE ACTUAL EXECUTED MODEL
    # ========================================================

    actual_prefix = (
        f"{model_df.iloc[0]['node_id']}"
        f"__"
        f"{executed_model}"
    )


    cpu_waiting_column = (
        f"{actual_prefix}"
        f"__cpu_waiting"
    )

    cpu_running_column = (
        f"{actual_prefix}"
        f"__cpu_running"
    )


    cpu_waiting = numeric_column(
        model_df,
        cpu_waiting_column,
    )

    cpu_running = numeric_column(
        model_df,
        cpu_running_column,
    )


    if (
        cpu_waiting is not None
        or cpu_running is not None
    ):

        plt.figure(
            figsize=(15, 7)
        )


        if cpu_waiting is not None:

            plt.plot(
                model_df["request_number"],
                cpu_waiting,
                marker="o",
                markersize=4,
                linewidth=2,
                label="CPU Waiting Functions",
            )


        if cpu_running is not None:

            plt.plot(
                model_df["request_number"],
                cpu_running,
                marker="s",
                markersize=4,
                linewidth=2,
                label="CPU Running Functions",
            )


        plt.xlabel(
            "Request Number"
        )

        plt.ylabel(
            "Function Count"
        )

        plt.title(
            f"CPU Waiting and Running Functions\n"
            f"Executed Model: {executed_model}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()


        file_path = os.path.join(
            OUTPUT_DIR,
            f"03_cpu_queue_running_{executed_model}.png",
        )

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        if SHOW_PLOTS:
            plt.show()

        plt.close()


    # ========================================================
    # GRAPH 4
    #
    # GPU WAITING / RUNNING
    #
    # ONLY FOR ACTUAL EXECUTED MODEL
    # ========================================================

    gpu_waiting_column = (
        f"{actual_prefix}"
        f"__gpu_waiting"
    )

    gpu_running_column = (
        f"{actual_prefix}"
        f"__gpu_running"
    )


    gpu_waiting = numeric_column(
        model_df,
        gpu_waiting_column,
    )

    gpu_running = numeric_column(
        model_df,
        gpu_running_column,
    )


    gpu_has_data = False


    if gpu_waiting is not None:

        gpu_has_data = (
            gpu_waiting
            .fillna(0)
            .abs()
            .sum()
            > 0
        )


    if gpu_running is not None:

        gpu_has_data = (
            gpu_has_data
            or
            (
                gpu_running
                .fillna(0)
                .abs()
                .sum()
                > 0
            )
        )


    if gpu_has_data:

        plt.figure(
            figsize=(15, 7)
        )


        if gpu_waiting is not None:

            plt.plot(
                model_df["request_number"],
                gpu_waiting,
                marker="o",
                markersize=4,
                linewidth=2,
                label="GPU Waiting Functions",
            )


        if gpu_running is not None:

            plt.plot(
                model_df["request_number"],
                gpu_running,
                marker="s",
                markersize=4,
                linewidth=2,
                label="GPU Running Functions",
            )


        plt.xlabel(
            "Request Number"
        )

        plt.ylabel(
            "Function Count"
        )

        plt.title(
            f"GPU Waiting and Running Functions\n"
            f"Executed Model: {executed_model}"
        )

        plt.grid(
            True,
            alpha=0.3,
        )

        plt.legend()

        plt.tight_layout()


        file_path = os.path.join(
            OUTPUT_DIR,
            f"04_gpu_queue_running_{executed_model}.png",
        )

        plt.savefig(
            file_path,
            dpi=300,
            bbox_inches="tight",
        )

        if SHOW_PLOTS:
            plt.show()

        plt.close()


    # ========================================================
    # GRAPH 5
    #
    # ACTUAL E2E LATENCY
    # VS CPU WAIT TIME
    # OF THE EXECUTED MODEL
    # ========================================================

    cpu_wait_column = (
        f"{actual_prefix}"
        f"__cpu_wait_ms"
    )

    gpu_wait_column = (
        f"{actual_prefix}"
        f"__gpu_wait_ms"
    )


    cpu_wait = numeric_column(
        model_df,
        cpu_wait_column,
    )

    gpu_wait = numeric_column(
        model_df,
        gpu_wait_column,
    )


    actual_latency = pd.to_numeric(
        model_df["wall_latency_ms"],
        errors="coerce",
    )


    plt.figure(
        figsize=(15, 7)
    )


    plt.plot(
        model_df["request_number"],
        actual_latency,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Actual E2E Latency",
    )


    if cpu_wait is not None:

        plt.plot(
            model_df["request_number"],
            cpu_wait,
            marker="s",
            markersize=4,
            linewidth=2,
            label="CPU Estimated Wait",
        )


    if (
        gpu_wait is not None
        and
        gpu_wait.fillna(0).abs().sum() > 0
    ):

        plt.plot(
            model_df["request_number"],
            gpu_wait,
            marker="^",
            markersize=4,
            linewidth=2,
            label="GPU Estimated Wait",
        )


    plt.xlabel(
        "Request Number"
    )

    plt.ylabel(
        "Time (ms)"
    )

    plt.title(
        f"Actual E2E Latency vs Estimated Wait Time\n"
        f"Executed Model: {executed_model}"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()


    file_path = os.path.join(
        OUTPUT_DIR,
        f"05_actual_vs_wait_{executed_model}.png",
    )

    plt.savefig(
        file_path,
        dpi=300,
        bbox_inches="tight",
    )

    if SHOW_PLOTS:
        plt.show()

    plt.close()


# ============================================================
# DONE
# ============================================================

print(
    "\nAll plots generated."
)

print(
    "Output folder:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)