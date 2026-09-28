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
    "azure_15min_selection_improved"
)

TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Final workload duration.
WINDOW_MINUTES = 15

# Workload is divided into 15-second bins.
#
# 15 minutes = 900 seconds
# 900 / 15 = 60 bins
BIN_SECONDS = 15.0

# Sliding-window movement.
WINDOW_STEP_SECONDS = 60.0

# Search the most active applications.
TOP_APPS_TO_ANALYZE = 50

# Reject intervals with too few requests.
MIN_REQUESTS_IN_WINDOW = 100

# ------------------------------------------------------------
# Required total number of bins for each workload phase
# ------------------------------------------------------------

MIN_IDLE_BINS = 2
MIN_LOW_BINS = 4
MIN_MEDIUM_BINS = 4
MIN_HIGH_BINS = 3

# ------------------------------------------------------------
# Required longest consecutive run for each phase
# ------------------------------------------------------------
#
# With 15-second bins:
#
# 2 bins = 30 seconds
# 3 bins = 45 seconds
# 4 bins = 60 seconds

MIN_IDLE_CONSECUTIVE_BINS = 2
MIN_LOW_CONSECUTIVE_BINS = 3
MIN_MEDIUM_CONSECUTIVE_BINS = 3
MIN_HIGH_CONSECUTIVE_BINS = 3

# Require lower load after the main High phase.
MIN_RECOVERY_BINS = 3

# The High phase cannot begin too close to the end.
#
# Example:
# 0.85 means High must begin before 85% of the 15-minute trace.
MAX_HIGH_START_FRACTION = 0.85

# Require the High phase to end before this fraction so that
# a recovery region remains visible.
MAX_HIGH_END_FRACTION = 0.92

# Maximum fraction of completely idle bins.
#
# Prevents selecting a workload that is mostly idle.
MAX_IDLE_FRACTION = 0.35

# Minimum fraction of active bins.
MIN_ACTIVE_FRACTION = 0.65

# ============================================================


PHASE_TO_NUMBER = {
    "Idle": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def unit_divisor(unit: str) -> float:
    normalized = unit.strip().lower()

    if normalized == "seconds":
        return 1.0

    if normalized == "milliseconds":
        return 1_000.0

    if normalized == "microseconds":
        return 1_000_000.0

    raise ValueError(
        "Unit must be 'seconds', 'milliseconds', or 'microseconds'."
    )


def read_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace file was not found:\n{trace_path}"
        )

    print("Reading Azure Functions trace...")
    print(trace_path)

    df = pd.read_csv(trace_path)

    # Fallback for whitespace-separated input.
    if len(df.columns) == 1:
        df = pd.read_csv(
            trace_path,
            sep=r"\s+",
            engine="python",
        )

    print("\nAvailable columns:")
    print(list(df.columns))

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}\n"
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
        subset=required_columns
    ).copy()

    df = df[
        (df["end_timestamp"] >= 0)
        & (df["duration"] >= 0)
    ].copy()

    df["end_time_sec"] = (
        df["end_timestamp"]
        / unit_divisor(TIMESTAMP_UNIT)
    )

    df["duration_sec"] = (
        df["duration"]
        / unit_divisor(DURATION_UNIT)
    )

    # Reconstruct request start time.
    df["start_time_sec"] = (
        df["end_time_sec"]
        - df["duration_sec"]
    )

    df = df.sort_values(
        ["app", "start_time_sec"]
    ).reset_index(drop=True)

    print(f"\nValid records: {len(df):,}")
    print(f"Applications: {df['app'].nunique():,}")

    return df


def classify_rates(
    rates: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """
    Classify each bin as Idle, Low, Medium, or High.

    Idle:
        Zero requests.

    Low, Medium, High:
        Based on percentiles of the non-zero arrival rates
        within the candidate interval.
    """

    phases = np.full(
        len(rates),
        "Idle",
        dtype=object,
    )

    positive_rates = rates[rates > 0]

    if len(positive_rates) < 3:
        return phases, 0.0, 0.0

    low_threshold = float(
        np.percentile(
            positive_rates,
            35,
        )
    )

    high_threshold = float(
        np.percentile(
            positive_rates,
            70,
        )
    )

    positive_mask = rates > 0

    low_mask = (
        positive_mask
        & (rates <= low_threshold)
    )

    medium_mask = (
        positive_mask
        & (rates > low_threshold)
        & (rates <= high_threshold)
    )

    high_mask = (
        positive_mask
        & (rates > high_threshold)
    )

    phases[low_mask] = "Low"
    phases[medium_mask] = "Medium"
    phases[high_mask] = "High"

    return phases, low_threshold, high_threshold


def longest_consecutive_run(
    phases: np.ndarray,
    target_phase: str,
) -> int:
    longest = 0
    current = 0

    for phase in phases:
        if phase == target_phase:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def count_transitions(
    phases: np.ndarray,
) -> int:
    if len(phases) < 2:
        return 0

    return int(
        np.sum(phases[1:] != phases[:-1])
    )


def find_longest_phase_block(
    phases: np.ndarray,
    target_phase: str,
) -> tuple[int, int, int]:
    """
    Return:
        start_index,
        end_index,
        length

    end_index is inclusive.
    """

    best_start = -1
    best_end = -1
    best_length = 0

    current_start = -1
    current_length = 0

    for index, phase in enumerate(phases):
        if phase == target_phase:
            if current_length == 0:
                current_start = index

            current_length += 1

            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = index

        else:
            current_start = -1
            current_length = 0

    return best_start, best_end, best_length


def calculate_recovery_bins(
    phases: np.ndarray,
    high_end_index: int,
) -> int:
    """
    Count consecutive non-High bins after the main High block.

    Recovery can be Medium, Low, or Idle.
    """

    if high_end_index < 0:
        return 0

    recovery_count = 0

    for phase in phases[high_end_index + 1:]:
        if phase == "High":
            break

        recovery_count += 1

    return recovery_count


def evaluate_candidate_window(
    app_name: str,
    window_start: float,
    window_times: np.ndarray,
) -> dict | None:
    window_seconds = WINDOW_MINUTES * 60.0

    relative_times = (
        window_times - window_start
    )

    bin_edges = np.arange(
        0,
        window_seconds + BIN_SECONDS,
        BIN_SECONDS,
    )

    request_counts, _ = np.histogram(
        relative_times,
        bins=bin_edges,
    )

    arrival_rates = (
        request_counts / BIN_SECONDS
    )

    phases, low_threshold, high_threshold = classify_rates(
        arrival_rates
    )

    phase_counts = {
        phase: int(np.sum(phases == phase))
        for phase in PHASE_TO_NUMBER
    }

    total_bins = len(phases)

    idle_fraction = (
        phase_counts["Idle"] / total_bins
    )

    active_fraction = (
        1.0 - idle_fraction
    )

    # --------------------------------------------------------
    # Total phase requirements
    # --------------------------------------------------------

    if phase_counts["Idle"] < MIN_IDLE_BINS:
        return None

    if phase_counts["Low"] < MIN_LOW_BINS:
        return None

    if phase_counts["Medium"] < MIN_MEDIUM_BINS:
        return None

    if phase_counts["High"] < MIN_HIGH_BINS:
        return None

    if idle_fraction > MAX_IDLE_FRACTION:
        return None

    if active_fraction < MIN_ACTIVE_FRACTION:
        return None

    # --------------------------------------------------------
    # Consecutive-run requirements
    # --------------------------------------------------------

    longest_idle_run = longest_consecutive_run(
        phases,
        "Idle",
    )

    longest_low_run = longest_consecutive_run(
        phases,
        "Low",
    )

    longest_medium_run = longest_consecutive_run(
        phases,
        "Medium",
    )

    longest_high_run = longest_consecutive_run(
        phases,
        "High",
    )

    if longest_idle_run < MIN_IDLE_CONSECUTIVE_BINS:
        return None

    if longest_low_run < MIN_LOW_CONSECUTIVE_BINS:
        return None

    if longest_medium_run < MIN_MEDIUM_CONSECUTIVE_BINS:
        return None

    if longest_high_run < MIN_HIGH_CONSECUTIVE_BINS:
        return None

    # --------------------------------------------------------
    # High phase and recovery requirements
    # --------------------------------------------------------

    high_start, high_end, main_high_length = (
        find_longest_phase_block(
            phases,
            "High",
        )
    )

    if high_start < 0:
        return None

    high_start_fraction = (
        high_start / total_bins
    )

    high_end_fraction = (
        high_end / total_bins
    )

    if high_start_fraction > MAX_HIGH_START_FRACTION:
        return None

    if high_end_fraction > MAX_HIGH_END_FRACTION:
        return None

    recovery_bins = calculate_recovery_bins(
        phases,
        high_end,
    )

    if recovery_bins < MIN_RECOVERY_BINS:
        return None

    recovery_phases = phases[
        high_end + 1:
        high_end + 1 + recovery_bins
    ]

    # Require at least some actual active recovery,
    # not only an idle tail.
    active_recovery_bins = int(
        np.sum(
            np.isin(
                recovery_phases,
                ["Low", "Medium"],
            )
        )
    )

    if active_recovery_bins < 1:
        return None

    # --------------------------------------------------------
    # Workload statistics
    # --------------------------------------------------------

    mean_rate = float(
        np.mean(arrival_rates)
    )

    peak_rate = float(
        np.max(arrival_rates)
    )

    rate_std = float(
        np.std(arrival_rates)
    )

    rate_cv = (
        rate_std / mean_rate
        if mean_rate > 0
        else 0.0
    )

    burst_ratio = (
        peak_rate / mean_rate
        if mean_rate > 0
        else 0.0
    )

    transitions = count_transitions(
        phases
    )

    phase_values = np.array(
        list(phase_counts.values()),
        dtype=float,
    )

    phase_proportions = (
        phase_values / total_bins
    )

    # Higher when phases are reasonably balanced.
    phase_balance_score = float(
        1.0 - np.std(phase_proportions)
    )

    # Prefer the main High phase near the middle or later,
    # but not at the very end.
    ideal_high_center = 0.65

    high_center_fraction = (
        (high_start + high_end)
        / 2.0
        / total_bins
    )

    high_position_score = max(
        0.0,
        1.0
        - abs(
            high_center_fraction
            - ideal_high_center
        ),
    )

    # Prefer around 10% to 20% idle bins.
    ideal_idle_fraction = 0.15

    idle_score = max(
        0.0,
        1.0
        - abs(
            idle_fraction
            - ideal_idle_fraction
        ),
    )

    sustained_phase_score = (
        longest_idle_run
        + longest_low_run
        + longest_medium_run
        + 2.0 * longest_high_run
        + recovery_bins
    )

    # Main selection score.
    score = (
        8.0 * phase_balance_score
        + 5.0 * high_position_score
        + 4.0 * idle_score
        + 0.6 * sustained_phase_score
        + 0.15 * transitions
        + 1.5 * rate_cv
        + 1.0 * np.log1p(burst_ratio)
        + 1.0 * np.log1p(len(window_times))
    )

    return {
        "app": str(app_name),

        "window_start_sec": float(window_start),
        "window_end_sec": float(
            window_start + window_seconds
        ),

        "request_count": int(
            len(window_times)
        ),

        "mean_rate_rps": mean_rate,
        "peak_rate_rps": peak_rate,
        "rate_std": rate_std,
        "rate_cv": rate_cv,
        "burst_ratio": burst_ratio,

        "idle_bin_count": phase_counts["Idle"],
        "low_bin_count": phase_counts["Low"],
        "medium_bin_count": phase_counts["Medium"],
        "high_bin_count": phase_counts["High"],

        "idle_fraction": idle_fraction,
        "active_fraction": active_fraction,

        "longest_idle_run": longest_idle_run,
        "longest_low_run": longest_low_run,
        "longest_medium_run": longest_medium_run,
        "longest_high_run": longest_high_run,

        "main_high_start_bin": high_start,
        "main_high_end_bin": high_end,
        "main_high_length_bins": main_high_length,

        "high_start_fraction": high_start_fraction,
        "high_end_fraction": high_end_fraction,
        "high_center_fraction": high_center_fraction,

        "recovery_bin_count": recovery_bins,
        "active_recovery_bin_count": active_recovery_bins,

        "transition_count": transitions,

        "low_threshold_rps": low_threshold,
        "high_threshold_rps": high_threshold,

        "phase_balance_score": phase_balance_score,
        "high_position_score": high_position_score,
        "idle_score": idle_score,
        "sustained_phase_score": sustained_phase_score,

        "selection_score": score,
    }


def find_best_window(
    df: pd.DataFrame,
) -> pd.Series:
    app_counts = (
        df.groupby("app")
        .size()
        .sort_values(ascending=False)
        .head(TOP_APPS_TO_ANALYZE)
    )

    print("\nApplications being searched:")
    print(app_counts.to_string())

    window_seconds = (
        WINDOW_MINUTES * 60.0
    )

    candidate_results = []

    for app_number, app_name in enumerate(
        app_counts.index,
        start=1,
    ):
        print(
            f"\nAnalyzing application "
            f"{app_number}/{len(app_counts)}: "
            f"{app_name}"
        )

        app_times = (
            df.loc[
                df["app"] == app_name,
                "start_time_sec",
            ]
            .sort_values()
            .to_numpy()
        )

        if len(app_times) < MIN_REQUESTS_IN_WINDOW:
            continue

        first_time = float(
            app_times.min()
        )

        last_time = float(
            app_times.max()
        )

        if (
            last_time - first_time
            < window_seconds
        ):
            continue

        candidate_starts = np.arange(
            first_time,
            last_time - window_seconds + 1,
            WINDOW_STEP_SECONDS,
        )

        valid_for_app = 0

        for window_start in candidate_starts:
            window_end = (
                window_start + window_seconds
            )

            left = np.searchsorted(
                app_times,
                window_start,
                side="left",
            )

            right = np.searchsorted(
                app_times,
                window_end,
                side="left",
            )

            request_count = (
                right - left
            )

            if request_count < MIN_REQUESTS_IN_WINDOW:
                continue

            candidate = evaluate_candidate_window(
                app_name=str(app_name),
                window_start=float(window_start),
                window_times=app_times[left:right],
            )

            if candidate is not None:
                candidate_results.append(
                    candidate
                )
                valid_for_app += 1

        print(
            f"Valid candidate windows for this app: "
            f"{valid_for_app}"
        )

    if not candidate_results:
        raise ValueError(
            "\nNo valid 15-minute window was found.\n\n"
            "Try one or more of these changes:\n"
            "1. Reduce MIN_REQUESTS_IN_WINDOW.\n"
            "2. Reduce MIN_*_CONSECUTIVE_BINS.\n"
            "3. Increase TOP_APPS_TO_ANALYZE.\n"
            "4. Increase MAX_IDLE_FRACTION.\n"
        )

    candidates = pd.DataFrame(
        candidate_results
    )

    candidates = candidates.sort_values(
        "selection_score",
        ascending=False,
    ).reset_index(drop=True)

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_path = (
        output_dir
        / "improved_candidate_15min_windows.csv"
    )

    candidates.to_csv(
        candidate_path,
        index=False,
    )

    print(
        f"\nAll valid candidates saved to:\n"
        f"{candidate_path}"
    )

    return candidates.iloc[0]


def extract_selected_requests(
    df: pd.DataFrame,
    selected: pd.Series,
) -> pd.DataFrame:
    selected_requests = df[
        (df["app"] == selected["app"])
        & (
            df["start_time_sec"]
            >= selected["window_start_sec"]
        )
        & (
            df["start_time_sec"]
            < selected["window_end_sec"]
        )
    ].copy()

    selected_requests = (
        selected_requests
        .sort_values("start_time_sec")
        .reset_index(drop=True)
    )

    selected_requests[
        "relative_arrival_sec"
    ] = (
        selected_requests["start_time_sec"]
        - selected["window_start_sec"]
    )

    selected_requests["iat_seconds"] = (
        selected_requests[
            "relative_arrival_sec"
        ].diff()
    )

    # Preserve any delay from the beginning of the selected
    # interval to the first request.
    selected_requests.loc[
        0,
        "iat_seconds",
    ] = selected_requests.loc[
        0,
        "relative_arrival_sec",
    ]

    selected_requests["request_index"] = np.arange(
        len(selected_requests)
    )

    return selected_requests


def create_selected_pattern(
    selected_requests: pd.DataFrame,
) -> pd.DataFrame:
    window_seconds = (
        WINDOW_MINUTES * 60.0
    )

    edges = np.arange(
        0,
        window_seconds + BIN_SECONDS,
        BIN_SECONDS,
    )

    request_counts, _ = np.histogram(
        selected_requests[
            "relative_arrival_sec"
        ],
        bins=edges,
    )

    arrival_rates = (
        request_counts / BIN_SECONDS
    )

    phases, low_threshold, high_threshold = classify_rates(
        arrival_rates
    )

    pattern = pd.DataFrame(
        {
            "bin_index": np.arange(
                len(request_counts)
            ),

            "bin_start_sec": edges[:-1],
            "bin_end_sec": edges[1:],

            "request_count": request_counts,
            "arrival_rate_rps": arrival_rates,

            "phase": phases,
        }
    )

    pattern["time_minute"] = (
        pattern["bin_start_sec"]
        / 60.0
    )

    pattern["phase_number"] = (
        pattern["phase"]
        .map(PHASE_TO_NUMBER)
    )

    pattern["low_threshold_rps"] = (
        low_threshold
    )

    pattern["high_threshold_rps"] = (
        high_threshold
    )

    return pattern


def save_selected_data(
    selected: pd.Series,
    requests: pd.DataFrame,
    pattern: pd.DataFrame,
) -> None:
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_dir
        / "improved_selected_15min_summary.csv"
    )

    requests_path = (
        output_dir
        / "improved_selected_15min_requests.csv"
    )

    pattern_path = (
        output_dir
        / "improved_selected_15min_pattern.csv"
    )

    replay_path = (
        output_dir
        / "azure_15min_replay.csv"
    )

    pd.DataFrame(
        [selected.to_dict()]
    ).to_csv(
        summary_path,
        index=False,
    )

    request_columns = [
        "request_index",
        "app",
        "func",
        "relative_arrival_sec",
        "iat_seconds",
        "start_time_sec",
        "end_time_sec",
        "duration_sec",
        "end_timestamp",
        "duration",
    ]

    available_columns = [
        column
        for column in request_columns
        if column in requests.columns
    ]

    requests[
        available_columns
    ].to_csv(
        requests_path,
        index=False,
    )

    pattern.to_csv(
        pattern_path,
        index=False,
    )

    replay = pd.DataFrame(
        {
            "request_index": requests[
                "request_index"
            ],

            "scheduled_arrival_sec": requests[
                "relative_arrival_sec"
            ],

            "iat_seconds": requests[
                "iat_seconds"
            ],

            "azure_app": requests["app"],
            "azure_func": requests["func"],
        }
    )

    replay.to_csv(
        replay_path,
        index=False,
    )

    print("\nSelected workload files saved:")
    print(summary_path)
    print(requests_path)
    print(pattern_path)
    print(replay_path)


def plot_selected_pattern(
    requests: pd.DataFrame,
    pattern: pd.DataFrame,
    selected: pd.Series,
) -> None:
    output_dir = Path(OUTPUT_DIR)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.5,
                0.8,
                1.0,
            ]
        },
    )

    rate_ax = axes[0]
    phase_ax = axes[1]
    iat_ax = axes[2]

    # --------------------------------------------------------
    # Arrival rate
    # --------------------------------------------------------

    rate_ax.step(
        pattern["time_minute"],
        pattern["arrival_rate_rps"],
        where="post",
        linewidth=1.8,
    )

    rate_ax.fill_between(
        pattern["time_minute"],
        pattern["arrival_rate_rps"],
        step="post",
        alpha=0.15,
    )

    rate_ax.set_title(
        "Improved selected 15-minute Azure workload pattern"
    )

    rate_ax.set_ylabel(
        "Arrival rate\n(req/s)"
    )

    rate_ax.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Workload phase
    # --------------------------------------------------------

    phase_ax.step(
        pattern["time_minute"],
        pattern["phase_number"],
        where="post",
        linewidth=1.8,
    )

    phase_ax.set_yticks(
        list(PHASE_TO_NUMBER.values())
    )

    phase_ax.set_yticklabels(
        list(PHASE_TO_NUMBER.keys())
    )

    phase_ax.set_ylabel(
        "Workload\nphase"
    )

    phase_ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    # Highlight the longest High block.
    high_start_bin = int(
        selected["main_high_start_bin"]
    )

    high_end_bin = int(
        selected["main_high_end_bin"]
    )

    high_start_minute = (
        high_start_bin
        * BIN_SECONDS
        / 60.0
    )

    high_end_minute = (
        (high_end_bin + 1)
        * BIN_SECONDS
        / 60.0
    )

    for axis in axes:
        axis.axvspan(
            high_start_minute,
            high_end_minute,
            alpha=0.10,
        )

    # --------------------------------------------------------
    # Exact IAT values
    # --------------------------------------------------------

    iat_ax.plot(
        requests[
            "relative_arrival_sec"
        ] / 60.0,

        requests["iat_seconds"],

        linewidth=0.9,
    )

    iat_ax.set_xlabel(
        "Time within selected workload (minutes)"
    )

    iat_ax.set_ylabel(
        "Inter-arrival\ntime (s)"
    )

    iat_ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    png_path = (
        output_dir
        / "improved_selected_15min_workload.png"
    )

    pdf_path = (
        output_dir
        / "improved_selected_15min_workload.pdf"
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

    plt.show()
    plt.close(fig)

    print("\nSelected-workload graph saved:")
    print(png_path)
    print(pdf_path)


def print_selected_summary(
    selected: pd.Series,
    pattern: pd.DataFrame,
) -> None:
    print("\nSelected 15-minute workload")
    print("--------------------------------")

    print(
        f"Application: "
        f"{selected['app']}"
    )

    print(
        f"Requests: "
        f"{int(selected['request_count'])}"
    )

    print(
        f"Mean arrival rate: "
        f"{selected['mean_rate_rps']:.4f} req/s"
    )

    print(
        f"Peak arrival rate: "
        f"{selected['peak_rate_rps']:.4f} req/s"
    )

    print(
        f"Arrival-rate CV: "
        f"{selected['rate_cv']:.4f}"
    )

    print(
        f"Burst ratio: "
        f"{selected['burst_ratio']:.4f}"
    )

    print("\nPhase bins")
    print("----------")

    print(
        f"Idle: "
        f"{int(selected['idle_bin_count'])}"
    )

    print(
        f"Low: "
        f"{int(selected['low_bin_count'])}"
    )

    print(
        f"Medium: "
        f"{int(selected['medium_bin_count'])}"
    )

    print(
        f"High: "
        f"{int(selected['high_bin_count'])}"
    )

    print("\nLongest consecutive phase runs")
    print("------------------------------")

    print(
        f"Idle: "
        f"{int(selected['longest_idle_run'])} bins"
    )

    print(
        f"Low: "
        f"{int(selected['longest_low_run'])} bins"
    )

    print(
        f"Medium: "
        f"{int(selected['longest_medium_run'])} bins"
    )

    print(
        f"High: "
        f"{int(selected['longest_high_run'])} bins"
    )

    print(
        f"Recovery after High: "
        f"{int(selected['recovery_bin_count'])} bins"
    )

    print(
        f"Active recovery bins: "
        f"{int(selected['active_recovery_bin_count'])}"
    )

    print(
        f"Workload transitions: "
        f"{int(selected['transition_count'])}"
    )

    print(
        f"Selection score: "
        f"{selected['selection_score']:.4f}"
    )

    print("\nFinal phase composition:")
    print(
        pattern["phase"]
        .value_counts()
        .to_string()
    )


def main() -> None:
    trace_df = read_trace()

    selected = find_best_window(
        trace_df
    )

    selected_requests = extract_selected_requests(
        trace_df,
        selected,
    )

    selected_pattern = create_selected_pattern(
        selected_requests
    )

    print_selected_summary(
        selected,
        selected_pattern,
    )

    save_selected_data(
        selected,
        selected_requests,
        selected_pattern,
    )

    plot_selected_pattern(
        selected_requests,
        selected_pattern,
        selected,
    )


if __name__ == "__main__":
    main()