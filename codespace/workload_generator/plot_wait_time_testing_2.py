





import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

CSV_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/"
    # "/Users/agamage/Desktop/D2I/Codes Original/Fass Edge Cascades/"
    "2models_iluvatar_latency_wait_snapshot_iat_100ms_requests_100.csv"
)

OUTPUT_DIR = "bug_report_plots"

SHOW_PLOTS = True


# ============================================================
# BUG DETECTION THRESHOLDS
#
# You can change these later.
# ============================================================

LOW_WAIT_THRESHOLD_MS = 10.0

HIGH_ACTUAL_LATENCY_MS = 250.0

MIN_CPU_WAITING = 1

MIN_CPU_RUNNING = 2

MIN_GPU_WAITING = 1

MIN_GPU_RUNNING = 2


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# READ CSV
# ============================================================

df = pd.read_csv(
    CSV_FILE
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "request_number",
    "node_id",
    "model_name",
    "wall_latency_ms",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Missing required column: {column}"
        )


# ============================================================
# SAFE NUMERIC CONVERSION
# ============================================================

def numeric_series(
    dataframe,
    column_name,
):

    if column_name not in dataframe.columns:

        return pd.Series(
            [float("nan")] * len(dataframe),
            index=dataframe.index,
        )

    return pd.to_numeric(
        dataframe[column_name],
        errors="coerce",
    )


# ============================================================
# FIND ALL EXECUTED MODELS
# ============================================================

executed_models = (
    df["model_name"]
    .dropna()
    .unique()
)


# ============================================================
# LOOP THROUGH EACH EXECUTED MODEL
# ============================================================

for executed_model in executed_models:

    # --------------------------------------------------------
    # Keep only requests where this model was actually invoked
    # --------------------------------------------------------

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


    if len(model_df) == 0:
        continue


    # ========================================================
    # GET EXECUTION NODE
    #
    # If model executes on edge_1, prefix becomes:
    #
    # edge_1__resnet34
    # ========================================================

    node_id = str(
        model_df.iloc[0]["node_id"]
    )

    prefix = (
        f"{node_id}"
        f"__"
        f"{executed_model}"
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Model: {executed_model}"
    )

    print(
        f"Node: {node_id}"
    )

    print(
        f"Requests: {len(model_df)}"
    )

    print(
        "=" * 80
    )


    # ========================================================
    # BUILD COLUMN NAMES
    # ========================================================

    cpu_wait_col = (
        f"{prefix}"
        f"__cpu_wait_ms"
    )

    gpu_wait_col = (
        f"{prefix}"
        f"__gpu_wait_ms"
    )

    cpu_waiting_col = (
        f"{prefix}"
        f"__cpu_waiting"
    )

    cpu_running_col = (
        f"{prefix}"
        f"__cpu_running"
    )

    gpu_waiting_col = (
        f"{prefix}"
        f"__gpu_waiting"
    )

    gpu_running_col = (
        f"{prefix}"
        f"__gpu_running"
    )


    # ========================================================
    # CONVERT DATA
    # ========================================================

    request_number = numeric_series(
        model_df,
        "request_number",
    )

    actual_e2e = numeric_series(
        model_df,
        "wall_latency_ms",
    )

    cpu_wait = numeric_series(
        model_df,
        cpu_wait_col,
    )

    gpu_wait = numeric_series(
        model_df,
        gpu_wait_col,
    )

    cpu_waiting = numeric_series(
        model_df,
        cpu_waiting_col,
    )

    cpu_running = numeric_series(
        model_df,
        cpu_running_col,
    )

    gpu_waiting = numeric_series(
        model_df,
        gpu_waiting_col,
    )

    gpu_running = numeric_series(
        model_df,
        gpu_running_col,
    )


    # ========================================================
    # ADD CLEAN VALUES TO DATAFRAME
    # ========================================================

    model_df[
        "bug_actual_e2e_ms"
    ] = actual_e2e

    model_df[
        "bug_cpu_wait_ms"
    ] = cpu_wait

    model_df[
        "bug_gpu_wait_ms"
    ] = gpu_wait

    model_df[
        "bug_cpu_waiting"
    ] = cpu_waiting

    model_df[
        "bug_cpu_running"
    ] = cpu_running

    model_df[
        "bug_gpu_waiting"
    ] = gpu_waiting

    model_df[
        "bug_gpu_running"
    ] = gpu_running


    # ========================================================
    # DETECT CPU SUSPICIOUS REQUESTS
    #
    # Condition:
    #
    # CPU queue/running state indicates contention
    # BUT estimator wait is still very low.
    # ========================================================

    cpu_contention = (
        (
            cpu_waiting >= MIN_CPU_WAITING
        )
        |
        (
            cpu_running >= MIN_CPU_RUNNING
        )
    )


    cpu_low_estimate = (
        cpu_wait <= LOW_WAIT_THRESHOLD_MS
    )


    cpu_suspicious = (
        cpu_contention
        &
        cpu_low_estimate
    )


    # ========================================================
    # OPTIONAL STRONGER CPU CONDITION
    #
    # Actual latency is high too.
    # ========================================================

    cpu_strong_suspicious = (
        cpu_suspicious
        &
        (
            actual_e2e
            >= HIGH_ACTUAL_LATENCY_MS
        )
    )


    # ========================================================
    # GPU SUSPICIOUS REQUESTS
    # ========================================================

    gpu_contention = (
        (
            gpu_waiting >= MIN_GPU_WAITING
        )
        |
        (
            gpu_running >= MIN_GPU_RUNNING
        )
    )


    gpu_low_estimate = (
        gpu_wait <= LOW_WAIT_THRESHOLD_MS
    )


    gpu_suspicious = (
        gpu_contention
        &
        gpu_low_estimate
    )


    gpu_strong_suspicious = (
        gpu_suspicious
        &
        (
            actual_e2e
            >= HIGH_ACTUAL_LATENCY_MS
        )
    )


    # ========================================================
    # COMBINED SUSPICIOUS CONDITION
    # ========================================================

    suspicious = (
        cpu_suspicious
        |
        gpu_suspicious
    )


    strong_suspicious = (
        cpu_strong_suspicious
        |
        gpu_strong_suspicious
    )


    model_df[
        "cpu_suspicious"
    ] = cpu_suspicious

    model_df[
        "gpu_suspicious"
    ] = gpu_suspicious

    model_df[
        "strong_suspicious"
    ] = strong_suspicious


    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print(
        f"CPU suspicious requests: "
        f"{cpu_suspicious.sum()}"
    )

    print(
        f"GPU suspicious requests: "
        f"{gpu_suspicious.sum()}"
    )

    print(
        f"Strong suspicious requests: "
        f"{strong_suspicious.sum()}"
    )


    # ========================================================
    # SAVE SUSPICIOUS REQUEST CSV
    # ========================================================

    suspicious_rows = (
        model_df[
            suspicious
        ]
        .copy()
    )


    suspicious_output_columns = [
        "request_number",
        "node_id",
        "model_name",
        "bug_actual_e2e_ms",
        "bug_cpu_wait_ms",
        "bug_gpu_wait_ms",
        "bug_cpu_waiting",
        "bug_cpu_running",
        "bug_gpu_waiting",
        "bug_gpu_running",
        "cpu_suspicious",
        "gpu_suspicious",
        "strong_suspicious",
    ]


    suspicious_csv = os.path.join(
        OUTPUT_DIR,
        (
            f"suspicious_requests_"
            f"{executed_model}.csv"
        ),
    )


    suspicious_rows[
        suspicious_output_columns
    ].to_csv(
        suspicious_csv,
        index=False,
    )


    print(
        f"Suspicious CSV saved: "
        f"{suspicious_csv}"
    )


    # ========================================================
    # ========================================================
    # GRAPH 1
    #
    # 3-PANEL BUG REPORT FIGURE
    # ========================================================
    # ========================================================

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(
            15,
            12,
        ),
        sharex=True,
    )


    # ========================================================
    # PANEL A
    #
    # ACTUAL E2E vs ESTIMATED WAIT
    # ========================================================

    axes[0].plot(
        request_number,
        actual_e2e,
        marker="o",
        markersize=4,
        linewidth=2,
        label="Actual E2E Latency",
    )


    if cpu_wait.notna().any():

        axes[0].plot(
            request_number,
            cpu_wait,
            marker="s",
            markersize=3,
            linewidth=1.7,
            label="CPU Estimated Wait",
        )


    if (
        gpu_wait.notna().any()
        and
        gpu_wait.fillna(0).abs().sum() > 0
    ):

        axes[0].plot(
            request_number,
            gpu_wait,
            marker="^",
            markersize=3,
            linewidth=1.7,
            label="GPU Estimated Wait",
        )


    # --------------------------------------------------------
    # Highlight suspicious requests
    # --------------------------------------------------------

    suspicious_requests = (
        request_number[
            suspicious
        ]
    )


    suspicious_actual = (
        actual_e2e[
            suspicious
        ]
    )


    if len(
        suspicious_requests
    ) > 0:

        axes[0].scatter(
            suspicious_requests,
            suspicious_actual,
            marker="x",
            s=80,
            linewidths=2,
            label="Suspicious Request",
        )


    axes[0].set_ylabel(
        "Time (ms)"
    )

    axes[0].set_title(
        f"Latency Estimator Behavior\n"
        f"Executed Model: {executed_model} "
        # f"Node: {node_id}"
    )

    axes[0].grid(
        True,
        alpha=0.3,
    )

    axes[0].legend(
        loc="best"
    )


    # ========================================================
    # PANEL B
    #
    # CPU WAITING / RUNNING
    # ========================================================

    axes[1].plot(
        request_number,
        cpu_waiting,
        marker="o",
        markersize=4,
        linewidth=2,
        label="CPU Waiting Functions",
    )


    axes[1].plot(
        request_number,
        cpu_running,
        marker="s",
        markersize=4,
        linewidth=2,
        label="CPU Running Functions",
    )


    axes[1].set_ylabel(
        "CPU Function Count"
    )

    axes[1].set_title(
        "CPU Queue / Running State"
    )

    axes[1].grid(
        True,
        alpha=0.3,
    )

    axes[1].legend(
        loc="best"
    )


    # ========================================================
    # PANEL C
    #
    # GPU WAITING / RUNNING
    # ========================================================

    axes[2].plot(
        request_number,
        gpu_waiting,
        marker="o",
        markersize=4,
        linewidth=2,
        label="GPU Waiting Functions",
    )


    axes[2].plot(
        request_number,
        gpu_running,
        marker="s",
        markersize=4,
        linewidth=2,
        label="GPU Running Functions",
    )


    axes[2].set_xlabel(
        "Request Number"
    )

    axes[2].set_ylabel(
        "GPU Function Count"
    )

    axes[2].set_title(
        "GPU Queue / Running State"
    )

    axes[2].grid(
        True,
        alpha=0.3,
    )

    axes[2].legend(
        loc="best"
    )


    # ========================================================
    # SAVE FULL BUG FIGURE
    # ========================================================

    plt.tight_layout()


    full_plot_file = os.path.join(
        OUTPUT_DIR,
        (
            f"bug_report_full_"
            f"{executed_model}.png"
        ),
    )


    plt.savefig(
        full_plot_file,
        dpi=300,
        bbox_inches="tight",
    )


    if SHOW_PLOTS:
        plt.show()


    plt.close()


    print(
        f"Full bug-report plot saved: "
        f"{full_plot_file}"
    )


    # ========================================================
    # ========================================================
    # GRAPH 2
    #
    # AUTOMATIC ZOOM AROUND SUSPICIOUS REQUESTS
    # ========================================================
    # ========================================================

    suspicious_request_numbers = (
        request_number[
            suspicious
        ]
        .dropna()
    )


    if len(
        suspicious_request_numbers
    ) > 0:

        # ----------------------------------------------------
        # First and last suspicious request
        # ----------------------------------------------------

        first_bad = int(
            suspicious_request_numbers.min()
        )

        last_bad = int(
            suspicious_request_numbers.max()
        )


        # ----------------------------------------------------
        # Add context around suspicious region
        # ----------------------------------------------------

        padding = 5


        start_request = max(
            1,
            first_bad - padding,
        )

        end_request = min(
            int(request_number.max()),
            last_bad + padding,
        )


        zoom_mask = (
            (
                request_number
                >= start_request
            )
            &
            (
                request_number
                <= end_request
            )
        )


        # ====================================================
        # ZOOM FIGURE
        # ====================================================

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(
                15,
                11,
            ),
            sharex=True,
        )


        # ====================================================
        # ZOOM PANEL A
        # ====================================================

        axes[0].plot(
            request_number[
                zoom_mask
            ],
            actual_e2e[
                zoom_mask
            ],
            marker="o",
            linewidth=2,
            label="Actual E2E Latency",
        )


        axes[0].plot(
            request_number[
                zoom_mask
            ],
            cpu_wait[
                zoom_mask
            ],
            marker="s",
            linewidth=2,
            label="CPU Estimated Wait",
        )


        if (
            gpu_wait
            .fillna(0)
            .abs()
            .sum()
            > 0
        ):

            axes[0].plot(
                request_number[
                    zoom_mask
                ],
                gpu_wait[
                    zoom_mask
                ],
                marker="^",
                linewidth=2,
                label="GPU Estimated Wait",
            )


        bad_zoom_mask = (
            zoom_mask
            &
            suspicious
        )


        axes[0].scatter(
            request_number[
                bad_zoom_mask
            ],
            actual_e2e[
                bad_zoom_mask
            ],
            marker="x",
            s=100,
            linewidths=2,
            label="Suspicious Request",
        )


        axes[0].set_ylabel(
            "Time (ms)"
        )

        axes[0].set_title(
            f"Suspicious Region: "
            f"Requests {start_request}–{end_request}\n"
            f"{executed_model}"
        )

        axes[0].grid(
            True,
            alpha=0.3,
        )

        axes[0].legend()


        # ====================================================
        # ZOOM PANEL B
        # ====================================================

        axes[1].plot(
            request_number[
                zoom_mask
            ],
            cpu_waiting[
                zoom_mask
            ],
            marker="o",
            linewidth=2,
            label="CPU Waiting",
        )


        axes[1].plot(
            request_number[
                zoom_mask
            ],
            cpu_running[
                zoom_mask
            ],
            marker="s",
            linewidth=2,
            label="CPU Running",
        )


        axes[1].set_ylabel(
            "CPU Count"
        )

        axes[1].grid(
            True,
            alpha=0.3,
        )

        axes[1].legend()


        # ====================================================
        # ZOOM PANEL C
        # ====================================================

        axes[2].plot(
            request_number[
                zoom_mask
            ],
            gpu_waiting[
                zoom_mask
            ],
            marker="o",
            linewidth=2,
            label="GPU Waiting",
        )


        axes[2].plot(
            request_number[
                zoom_mask
            ],
            gpu_running[
                zoom_mask
            ],
            marker="s",
            linewidth=2,
            label="GPU Running",
        )


        axes[2].set_xlabel(
            "Request Number"
        )

        axes[2].set_ylabel(
            "GPU Count"
        )

        axes[2].grid(
            True,
            alpha=0.3,
        )

        axes[2].legend()


        plt.tight_layout()


        zoom_plot_file = os.path.join(
            OUTPUT_DIR,
            (
                f"bug_report_zoom_"
                f"{executed_model}.png"
            ),
        )


        plt.savefig(
            zoom_plot_file,
            dpi=300,
            bbox_inches="tight",
        )


        if SHOW_PLOTS:
            plt.show()


        plt.close()


        print(
            f"Zoomed bug plot saved: "
            f"{zoom_plot_file}"
        )


# ============================================================
# DONE
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "Bug-report analysis completed."
)

print(
    "\nOutput directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)

print(
    "=" * 80
)