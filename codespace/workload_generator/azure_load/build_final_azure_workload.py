from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

AZURE_TRACE_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "final_trace_derived_workload"
)

SELECTED_APP = (
    "a9644719e5d173a27e098cffdd5e9227ba9c4a325d082c84d8bcadae6723f19c"
)

TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Each observed Azure bin covers this many seconds.
SOURCE_BIN_SECONDS = 5.0

# Final generated workload duration.
GENERATED_DURATION_SECONDS = 15 * 60

# Workload classification percentiles.
#
# Low:
#   positive bins up to P50
#
# Medium:
#   P50 to P80
#
# High:
#   P80 to P95
#
# Spike:
#   above P95
LOW_PERCENTILE = 50
MEDIUM_PERCENTILE = 80
HIGH_PERCENTILE = 95

PHASES = [
    "Idle",
    "Low",
    "Medium",
    "High",
    "Spike",
]

ACTIVE_PHASES = [
    "Low",
    "Medium",
    "High",
    "Spike",
]

PHASE_TO_NUMBER = {
    "Idle": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Spike": 4,
}

REQUIRED_PHASES = [
    "Idle",
    "Low",
    "Medium",
    "High",
    "Spike",
]

RANDOM_SEED = 42

# Number of phase-pattern candidates to examine.
PATTERN_GENERATION_ATTEMPTS = 50_000

# Workload-pattern constraints.
MAX_TOTAL_IDLE_FRACTION = 0.30
MAX_SINGLE_IDLE_DURATION_SECONDS = 120.0

MIN_SPIKE_DURATION_SECONDS = 10.0
MAX_SPIKE_DURATION_SECONDS = 90.0

MIN_PHASE_DURATION_SECONDS = SOURCE_BIN_SECONDS
MAX_PHASE_DURATION_SECONDS = 180.0

REQUIRE_INTERNAL_SPIKE = True
REQUIRE_ABRUPT_INCREASE = True
REQUIRE_ABRUPT_DECREASE = True

DEFAULT_DEADLINE_MS = 2000


# ============================================================
# BASIC HELPERS
# ============================================================

def unit_divisor(unit: str) -> float:
    divisors = {
        "seconds": 1.0,
        "milliseconds": 1_000.0,
        "microseconds": 1_000_000.0,
    }

    normalized = unit.strip().lower()

    if normalized not in divisors:
        raise ValueError(
            "Unit must be seconds, milliseconds, or microseconds."
        )

    return divisors[normalized]


def safe_percentile(
    values: np.ndarray,
    percentile: float,
) -> float:
    if len(values) == 0:
        return float("nan")

    return float(
        np.percentile(
            values,
            percentile,
        )
    )


# ============================================================
# READ AZURE TRACE
# ============================================================

def read_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace not found:\n{trace_path}"
        )

    print("Reading Azure Functions trace...")
    print(trace_path)

    df = pd.read_csv(trace_path)

    # Fallback for whitespace-separated traces.
    if len(df.columns) == 1:
        df = pd.read_csv(
            trace_path,
            sep=r"\s+",
            engine="python",
        )

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
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

    # Reconstruct invocation start time.
    df["start_time_sec"] = (
        df["end_time_sec"]
        - df["duration_sec"]
    )

    df = df.sort_values(
        [
            "app",
            "start_time_sec",
        ]
    ).reset_index(drop=True)

    print(f"Valid requests: {len(df):,}")
    print(f"Applications: {df['app'].nunique():,}")
    print(f"Functions: {df['func'].nunique():,}")

    return df


def get_selected_application(
    trace_df: pd.DataFrame,
) -> pd.DataFrame:
    app_df = trace_df[
        trace_df["app"].astype(str)
        == SELECTED_APP
    ].copy()

    if app_df.empty:
        raise ValueError(
            "Selected application was not found:\n"
            f"{SELECTED_APP}"
        )

    app_df = app_df.sort_values(
        "start_time_sec"
    ).reset_index(drop=True)

    duration_hours = (
        app_df["start_time_sec"].max()
        - app_df["start_time_sec"].min()
    ) / 3600.0

    print("\nSelected Azure application")
    print("--------------------------")
    print(SELECTED_APP)
    print(f"Requests: {len(app_df):,}")
    print(f"Duration: {duration_hours:.2f} hours")

    return app_df


# ============================================================
# CHARACTERIZE THE SELECTED TRACE
# ============================================================

def classify_rates(
    arrival_rates: np.ndarray,
) -> tuple[np.ndarray, dict]:
    phases = np.full(
        len(arrival_rates),
        "Idle",
        dtype=object,
    )

    positive_rates = arrival_rates[
        arrival_rates > 0
    ]

    if len(positive_rates) < 4:
        raise ValueError(
            "Not enough positive-rate bins to classify workload levels."
        )

    low_upper = safe_percentile(
        positive_rates,
        LOW_PERCENTILE,
    )

    medium_upper = safe_percentile(
        positive_rates,
        MEDIUM_PERCENTILE,
    )

    high_upper = safe_percentile(
        positive_rates,
        HIGH_PERCENTILE,
    )

    positive = arrival_rates > 0

    phases[
        positive
        & (arrival_rates <= low_upper)
    ] = "Low"

    phases[
        positive
        & (arrival_rates > low_upper)
        & (arrival_rates <= medium_upper)
    ] = "Medium"

    phases[
        positive
        & (arrival_rates > medium_upper)
        & (arrival_rates <= high_upper)
    ] = "High"

    phases[
        positive
        & (arrival_rates > high_upper)
    ] = "Spike"

    thresholds = {
        "low_upper_rps": low_upper,
        "medium_upper_rps": medium_upper,
        "high_upper_rps": high_upper,
    }

    return phases, thresholds


def create_observed_bins(
    app_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    requests = app_df.copy()

    first_start = float(
        requests["start_time_sec"].min()
    )

    requests["relative_time_sec"] = (
        requests["start_time_sec"]
        - first_start
    )

    requests["source_bin_index"] = np.floor(
        requests["relative_time_sec"]
        / SOURCE_BIN_SECONDS
    ).astype(int)

    requests["offset_within_bin_sec"] = (
        requests["relative_time_sec"]
        - (
            requests["source_bin_index"]
            * SOURCE_BIN_SECONDS
        )
    )

    maximum_relative_time = float(
        requests["relative_time_sec"].max()
    )

    number_of_bins = (
        int(
            math.floor(
                maximum_relative_time
                / SOURCE_BIN_SECONDS
            )
        )
        + 1
    )

    request_counts = np.zeros(
        number_of_bins,
        dtype=int,
    )

    grouped_counts = (
        requests.groupby("source_bin_index")
        .size()
    )

    request_counts[
        grouped_counts.index.to_numpy(dtype=int)
    ] = grouped_counts.to_numpy(dtype=int)

    arrival_rates = (
        request_counts
        / SOURCE_BIN_SECONDS
    )

    phases, thresholds = classify_rates(
        arrival_rates
    )

    observed_bins = pd.DataFrame(
        {
            "source_bin_index": np.arange(
                number_of_bins
            ),
            "bin_start_sec": (
                np.arange(number_of_bins)
                * SOURCE_BIN_SECONDS
            ),
            "bin_end_sec": (
                (
                    np.arange(number_of_bins)
                    + 1
                )
                * SOURCE_BIN_SECONDS
            ),
            "request_count": request_counts,
            "arrival_rate_rps": arrival_rates,
            "phase": phases,
        }
    )

    requests = requests.merge(
        observed_bins[
            [
                "source_bin_index",
                "arrival_rate_rps",
                "phase",
            ]
        ],
        on="source_bin_index",
        how="left",
    )

    return (
        observed_bins,
        requests,
        thresholds,
    )


# ============================================================
# BUILD THE THRESHOLD TABLE
# ============================================================

def calculate_phase_statistics(
    observed_bins: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for phase in PHASES:
        phase_bins = observed_bins[
            observed_bins["phase"] == phase
        ].copy()

        rates = phase_bins[
            "arrival_rate_rps"
        ].to_numpy(dtype=float)

        rows.append(
            {
                "phase": phase,
                "bin_count": int(
                    len(phase_bins)
                ),
                "request_count": int(
                    phase_bins["request_count"].sum()
                ),
                "mean_rate_rps": (
                    float(np.mean(rates))
                    if len(rates) > 0
                    else np.nan
                ),
                "median_rate_rps": (
                    float(np.median(rates))
                    if len(rates) > 0
                    else np.nan
                ),
                "p95_rate_rps": (
                    safe_percentile(
                        rates,
                        95,
                    )
                    if len(rates) > 0
                    else np.nan
                ),
                "maximum_rate_rps": (
                    float(np.max(rates))
                    if len(rates) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def build_threshold_table(
    thresholds: dict,
    phase_statistics: pd.DataFrame,
) -> pd.DataFrame:
    low_upper = float(
        thresholds["low_upper_rps"]
    )

    medium_upper = float(
        thresholds["medium_upper_rps"]
    )

    high_upper = float(
        thresholds["high_upper_rps"]
    )

    definitions = {
        "Idle": "rate = 0",
        "Low": (
            f"0 < rate <= {low_upper:.2f}"
        ),
        "Medium": (
            f"{low_upper:.2f} < rate <= "
            f"{medium_upper:.2f}"
        ),
        "High": (
            f"{medium_upper:.2f} < rate <= "
            f"{high_upper:.2f}"
        ),
        "Spike": (
            f"rate > {high_upper:.2f}"
        ),
    }

    rows = []

    for phase in PHASES:
        stats = phase_statistics[
            phase_statistics["phase"]
            == phase
        ].iloc[0]

        rows.append(
            {
                "workload_level": phase,
                "trace_derived_definition_rps": (
                    definitions[phase]
                ),
                "mean_observed_rate_rps": float(
                    stats["mean_rate_rps"]
                ),
                "median_observed_rate_rps": float(
                    stats["median_rate_rps"]
                ),
                "p95_observed_rate_rps": float(
                    stats["p95_rate_rps"]
                ),
                "maximum_observed_rate_rps": float(
                    stats["maximum_rate_rps"]
                ),
                "observed_bin_count": int(
                    stats["bin_count"]
                ),
            }
        )

    return pd.DataFrame(rows)


def save_threshold_table(
    threshold_table: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    threshold_table.to_csv(
        output_dir
        / "trace_derived_workload_thresholds.csv",
        index=False,
    )

    # Plain-text table.
    with open(
        output_dir
        / "trace_derived_workload_thresholds.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(
            threshold_table.to_string(
                index=False,
                float_format=lambda value: (
                    f"{value:.3f}"
                ),
            )
        )

        file.write("\n")

    # LaTeX table for the paper.
    latex_table = threshold_table[
        [
            "workload_level",
            "trace_derived_definition_rps",
            "mean_observed_rate_rps",
            "p95_observed_rate_rps",
            "maximum_observed_rate_rps",
        ]
    ].copy()

    latex_table.columns = [
        "Workload level",
        "Trace-derived definition (req/s)",
        "Mean rate (req/s)",
        "P95 rate (req/s)",
        "Maximum rate (req/s)",
    ]

    latex_text = latex_table.to_latex(
        index=False,
        float_format="%.2f",
        escape=True,
    )

    with open(
        output_dir
        / "trace_derived_workload_thresholds.tex",
        "w",
        encoding="utf-8",
    ) as file:
        file.write(latex_text)


# ============================================================
# LEARN TRANSITIONS AND PHASE DURATIONS
# ============================================================

def collapse_phases(
    phase_sequence: list[str],
) -> list[str]:
    if not phase_sequence:
        return []

    collapsed = [
        phase_sequence[0]
    ]

    for phase in phase_sequence[1:]:
        if phase != collapsed[-1]:
            collapsed.append(phase)

    return collapsed


def calculate_transition_counts(
    phase_sequence: list[str],
) -> Counter:
    collapsed = collapse_phases(
        phase_sequence
    )

    transitions = Counter()

    for current_phase, next_phase in zip(
        collapsed[:-1],
        collapsed[1:],
    ):
        transitions[
            (
                current_phase,
                next_phase,
            )
        ] += 1

    return transitions


def calculate_transition_probabilities(
    transitions: Counter,
) -> dict[str, dict[str, float]]:
    outgoing = defaultdict(Counter)

    for (
        current_phase,
        next_phase,
    ), count in transitions.items():
        outgoing[
            current_phase
        ][
            next_phase
        ] += count

    probabilities = {}

    for current_phase, next_counts in outgoing.items():
        total = sum(
            next_counts.values()
        )

        probabilities[current_phase] = {
            next_phase: count / total
            for next_phase, count
            in next_counts.items()
        }

    return probabilities


def calculate_phase_run_lengths(
    phase_sequence: list[str],
) -> dict[str, list[int]]:
    runs = defaultdict(list)

    if not phase_sequence:
        return {}

    current_phase = phase_sequence[0]
    current_length = 1

    for phase in phase_sequence[1:]:
        if phase == current_phase:
            current_length += 1
        else:
            runs[current_phase].append(
                current_length
            )

            current_phase = phase
            current_length = 1

    runs[current_phase].append(
        current_length
    )

    return dict(runs)


# ============================================================
# GENERATE THE 15-MINUTE PHASE PATTERN
# ============================================================

def weighted_choice(
    probabilities: dict[str, float],
) -> str:
    return random.choices(
        list(probabilities.keys()),
        weights=list(probabilities.values()),
        k=1,
    )[0]


def choose_initial_phase(
    phase_runs: dict[str, list[int]],
) -> str:
    available_phases = [
        phase
        for phase in PHASES
        if phase_runs.get(phase)
    ]

    if not available_phases:
        raise RuntimeError(
            "No phase runs are available."
        )

    weights = [
        len(phase_runs[phase])
        for phase in available_phases
    ]

    return random.choices(
        available_phases,
        weights=weights,
        k=1,
    )[0]


def sample_phase_run_bins(
    phase: str,
    phase_runs: dict[str, list[int]],
) -> int:
    observed_runs = phase_runs.get(
        phase,
        [1],
    )

    sampled_run = int(
        random.choice(
            observed_runs
        )
    )

    minimum_bins = max(
        1,
        int(
            MIN_PHASE_DURATION_SECONDS
            / SOURCE_BIN_SECONDS
        ),
    )

    maximum_bins = max(
        minimum_bins,
        int(
            MAX_PHASE_DURATION_SECONDS
            / SOURCE_BIN_SECONDS
        ),
    )

    if phase == "Idle":
        maximum_bins = min(
            maximum_bins,
            int(
                MAX_SINGLE_IDLE_DURATION_SECONDS
                / SOURCE_BIN_SECONDS
            ),
        )

    return int(
        np.clip(
            sampled_run,
            minimum_bins,
            maximum_bins,
        )
    )


def generate_candidate_blocks(
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
    phase_runs: dict[str, list[int]],
) -> list[dict]:
    target_bin_count = int(
        GENERATED_DURATION_SECONDS
        / SOURCE_BIN_SECONDS
    )

    current_phase = choose_initial_phase(
        phase_runs
    )

    current_bin = 0
    blocks = []

    while current_bin < target_bin_count:
        run_bins = sample_phase_run_bins(
            current_phase,
            phase_runs,
        )

        remaining_bins = (
            target_bin_count
            - current_bin
        )

        run_bins = min(
            run_bins,
            remaining_bins,
        )

        blocks.append(
            {
                "phase": current_phase,
                "start_bin": current_bin,
                "end_bin": (
                    current_bin
                    + run_bins
                ),
                "run_bins": run_bins,
                "start_sec": (
                    current_bin
                    * SOURCE_BIN_SECONDS
                ),
                "end_sec": (
                    (
                        current_bin
                        + run_bins
                    )
                    * SOURCE_BIN_SECONDS
                ),
                "duration_sec": (
                    run_bins
                    * SOURCE_BIN_SECONDS
                ),
            }
        )

        current_bin += run_bins

        if current_bin >= target_bin_count:
            break

        outgoing = transition_probabilities.get(
            current_phase,
            {},
        )

        if outgoing:
            current_phase = weighted_choice(
                outgoing
            )
        else:
            current_phase = choose_initial_phase(
                phase_runs
            )

    return blocks


def score_candidate_blocks(
    blocks: list[dict],
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
) -> float:
    if not blocks:
        return -float("inf")

    phases = [
        block["phase"]
        for block in blocks
    ]

    phase_durations = defaultdict(float)

    for block in blocks:
        phase_durations[
            block["phase"]
        ] += block["duration_sec"]

    # Require every workload level.
    for phase in REQUIRED_PHASES:
        if phase_durations.get(
            phase,
            0.0,
        ) <= 0:
            return -float("inf")

    idle_fraction = (
        phase_durations.get(
            "Idle",
            0.0,
        )
        / GENERATED_DURATION_SECONDS
    )

    if idle_fraction > MAX_TOTAL_IDLE_FRACTION:
        return -float("inf")

    spike_indices = [
        index
        for index, phase in enumerate(phases)
        if phase == "Spike"
    ]

    if not spike_indices:
        return -float("inf")

    if (
        REQUIRE_INTERNAL_SPIKE
        and all(
            index == 0
            for index in spike_indices
        )
    ):
        return -float("inf")

    spike_durations = [
        block["duration_sec"]
        for block in blocks
        if block["phase"] == "Spike"
    ]

    if not any(
        MIN_SPIKE_DURATION_SECONDS
        <= duration
        <= MAX_SPIKE_DURATION_SECONDS
        for duration in spike_durations
    ):
        return -float("inf")

    abrupt_increase = False
    abrupt_decrease = False
    transition_log_probability = 0.0

    for current_phase, next_phase in zip(
        phases[:-1],
        phases[1:],
    ):
        probability = (
            transition_probabilities
            .get(current_phase, {})
            .get(next_phase, 0.0)
        )

        if probability <= 0:
            return -float("inf")

        transition_log_probability += math.log(
            probability + 1e-12
        )

        current_level = PHASE_TO_NUMBER[
            current_phase
        ]

        next_level = PHASE_TO_NUMBER[
            next_phase
        ]

        if next_level - current_level >= 2:
            abrupt_increase = True

        if current_level - next_level >= 2:
            abrupt_decrease = True

    if (
        REQUIRE_ABRUPT_INCREASE
        and not abrupt_increase
    ):
        return -float("inf")

    if (
        REQUIRE_ABRUPT_DECREASE
        and not abrupt_decrease
    ):
        return -float("inf")

    active_duration_values = np.array(
        [
            phase_durations.get(
                phase,
                0.0,
            )
            for phase in ACTIVE_PHASES
        ],
        dtype=float,
    )

    active_duration_proportions = (
        active_duration_values
        / max(
            active_duration_values.sum(),
            1.0,
        )
    )

    balance_score = (
        1.0
        - float(
            np.std(
                active_duration_proportions
            )
        )
    )

    represented_phases = len(
        set(phases)
    )

    transition_count = (
        len(blocks)
        - 1
    )

    return (
        10.0 * represented_phases
        + 8.0 * balance_score
        + 0.5 * transition_count
        + 2.0 * float(abrupt_increase)
        + 2.0 * float(abrupt_decrease)
        + transition_log_probability
    )


def generate_best_pattern(
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
    phase_runs: dict[str, list[int]],
) -> pd.DataFrame:
    best_blocks = None
    best_score = -float("inf")

    for _ in range(
        PATTERN_GENERATION_ATTEMPTS
    ):
        candidate = generate_candidate_blocks(
            transition_probabilities,
            phase_runs,
        )

        score = score_candidate_blocks(
            candidate,
            transition_probabilities,
        )

        if score > best_score:
            best_score = score
            best_blocks = candidate

    if best_blocks is None:
        raise RuntimeError(
            "Could not generate a valid trace-derived pattern."
        )

    pattern = pd.DataFrame(
        best_blocks
    )

    pattern["block_index"] = np.arange(
        len(pattern)
    )

    pattern["pattern_score"] = (
        best_score
    )

    return pattern


def expand_pattern_to_bins(
    pattern: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    generated_bin_index = 0

    for _, block in pattern.iterrows():
        for _ in range(
            int(block["run_bins"])
        ):
            rows.append(
                {
                    "generated_bin_index": (
                        generated_bin_index
                    ),
                    "generated_bin_start_sec": (
                        generated_bin_index
                        * SOURCE_BIN_SECONDS
                    ),
                    "generated_bin_end_sec": (
                        (
                            generated_bin_index
                            + 1
                        )
                        * SOURCE_BIN_SECONDS
                    ),
                    "phase": str(
                        block["phase"]
                    ),
                    "phase_block_index": int(
                        block["block_index"]
                    ),
                }
            )

            generated_bin_index += 1

    return pd.DataFrame(rows)


# ============================================================
# SAMPLE REAL AZURE BINS
# ============================================================

def create_phase_bin_pools(
    observed_bins: pd.DataFrame,
) -> dict[str, np.ndarray]:
    pools = {}

    for phase in PHASES:
        pools[phase] = observed_bins.loc[
            observed_bins["phase"] == phase,
            "source_bin_index",
        ].to_numpy(dtype=int)

        print(
            f"{phase:7s}: "
            f"{len(pools[phase]):,} observed bins"
        )

    return pools


def sample_generated_bins(
    pattern_bins: pd.DataFrame,
    observed_bins: pd.DataFrame,
    phase_bin_pools: dict[str, np.ndarray],
) -> pd.DataFrame:
    observed_lookup = (
        observed_bins
        .set_index("source_bin_index")
    )

    rows = []

    for _, generated_bin in pattern_bins.iterrows():
        phase = str(
            generated_bin["phase"]
        )

        phase_pool = phase_bin_pools.get(
            phase,
            np.array([], dtype=int),
        )

        if len(phase_pool) == 0:
            raise RuntimeError(
                f"No observed bins are available for {phase}."
            )

        source_bin_index = int(
            random.choice(
                phase_pool.tolist()
            )
        )

        source_row = observed_lookup.loc[
            source_bin_index
        ]

        rows.append(
            {
                "generated_bin_index": int(
                    generated_bin[
                        "generated_bin_index"
                    ]
                ),
                "generated_bin_start_sec": float(
                    generated_bin[
                        "generated_bin_start_sec"
                    ]
                ),
                "generated_bin_end_sec": float(
                    generated_bin[
                        "generated_bin_end_sec"
                    ]
                ),
                "phase": phase,
                "phase_block_index": int(
                    generated_bin[
                        "phase_block_index"
                    ]
                ),
                "source_bin_index": (
                    source_bin_index
                ),
                "request_count": int(
                    source_row[
                        "request_count"
                    ]
                ),
                "arrival_rate_rps": float(
                    source_row[
                        "arrival_rate_rps"
                    ]
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# CREATE REPLAY REQUESTS
# ============================================================

def build_request_offset_lookup(
    observed_requests: pd.DataFrame,
) -> dict[int, np.ndarray]:
    return {
        int(source_bin): group[
            "offset_within_bin_sec"
        ].sort_values().to_numpy(dtype=float)
        for source_bin, group
        in observed_requests.groupby(
            "source_bin_index"
        )
    }


def create_replay(
    generated_bins: pd.DataFrame,
    observed_requests: pd.DataFrame,
) -> pd.DataFrame:
    offset_lookup = build_request_offset_lookup(
        observed_requests
    )

    rows = []

    for _, generated_bin in generated_bins.iterrows():
        generated_start = float(
            generated_bin[
                "generated_bin_start_sec"
            ]
        )

        source_bin_index = int(
            generated_bin[
                "source_bin_index"
            ]
        )

        offsets = offset_lookup.get(
            source_bin_index,
            np.array([], dtype=float),
        )

        for source_position, offset in enumerate(
            offsets
        ):
            rows.append(
                {
                    "scheduled_arrival_sec": (
                        generated_start
                        + float(offset)
                    ),
                    "phase": str(
                        generated_bin["phase"]
                    ),
                    "phase_block_index": int(
                        generated_bin[
                            "phase_block_index"
                        ]
                    ),
                    "generated_bin_index": int(
                        generated_bin[
                            "generated_bin_index"
                        ]
                    ),
                    "source_bin_index": (
                        source_bin_index
                    ),
                    "source_request_position": (
                        source_position
                    ),
                    "offset_within_bin_sec": (
                        float(offset)
                    ),
                    "deadline_ms": (
                        DEFAULT_DEADLINE_MS
                    ),
                }
            )

    replay = pd.DataFrame(
        rows
    )

    if replay.empty:
        raise RuntimeError(
            "Generated replay contains no requests."
        )

    replay = replay.sort_values(
        [
            "scheduled_arrival_sec",
            "generated_bin_index",
            "source_request_position",
        ]
    ).reset_index(drop=True)

    replay["request_index"] = np.arange(
        len(replay)
    )

    # Actual elapsed time between generated requests.
    replay["iat_seconds"] = (
        replay["scheduled_arrival_sec"]
        .diff()
    )

    # The first request has no previous request.
    replay.loc[
        0,
        "iat_seconds",
    ] = np.nan

    return replay


# ============================================================
# FINAL WORKLOAD FIGURE
# ============================================================

def calculate_pattern_rate(
    replay: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(
        0.0,
        GENERATED_DURATION_SECONDS
        + SOURCE_BIN_SECONDS,
        SOURCE_BIN_SECONDS,
    )

    request_counts, _ = np.histogram(
        replay["scheduled_arrival_sec"],
        bins=edges,
    )

    arrival_rates = (
        request_counts
        / SOURCE_BIN_SECONDS
    )

    time_minutes = (
        edges[:-1]
        + SOURCE_BIN_SECONDS / 2.0
    ) / 60.0

    return (
        time_minutes,
        arrival_rates,
    )


def plot_workload_pattern(
    replay: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    pattern_time, pattern_rate = (
        calculate_pattern_rate(
            replay
        )
    )

    valid_iats = replay.dropna(
        subset=["iat_seconds"]
    ).copy()

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11, 6.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.5,
                1.0,
            ]
        },
    )

    rate_ax = axes[0]
    iat_ax = axes[1]

    rate_ax.step(
        pattern_time,
        pattern_rate,
        where="mid",
        linewidth=1.8,
    )

    rate_ax.fill_between(
        pattern_time,
        pattern_rate,
        step="mid",
        alpha=0.15,
    )

    rate_ax.set_title(
        "Dynamic Workload Pattern"
    )

    rate_ax.set_ylabel(
        "Arrival rate\n(requests/second)"
    )

    rate_ax.grid(
        True,
        alpha=0.3,
    )

    iat_ax.scatter(
        valid_iats[
            "scheduled_arrival_sec"
        ] / 60.0,
        valid_iats[
            "iat_seconds"
        ],
        s=8,
        alpha=0.6,
    )

    iat_ax.set_yscale(
        "log"
    )

    iat_ax.set_xlabel(
        "Workload time (minutes)"
    )

    iat_ax.set_ylabel(
        "Inter-arrival time\n(seconds)"
    )

    iat_ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "final_trace_derived_workload_pattern.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "final_trace_derived_workload_pattern.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs(
    thresholds: dict,
    observed_bins: pd.DataFrame,
    phase_statistics: pd.DataFrame,
    threshold_table: pd.DataFrame,
    pattern: pd.DataFrame,
    generated_bins: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pd.DataFrame(
        [
            {
                "selected_app": SELECTED_APP,
                "source_bin_seconds": (
                    SOURCE_BIN_SECONDS
                ),
                "low_percentile": (
                    LOW_PERCENTILE
                ),
                "medium_percentile": (
                    MEDIUM_PERCENTILE
                ),
                "high_percentile": (
                    HIGH_PERCENTILE
                ),
                **thresholds,
            }
        ]
    ).to_csv(
        output_dir
        / "trace_derived_threshold_values.csv",
        index=False,
    )

    observed_bins.to_csv(
        output_dir
        / "selected_application_observed_bins.csv",
        index=False,
    )

    phase_statistics.to_csv(
        output_dir
        / "selected_application_phase_statistics.csv",
        index=False,
    )

    threshold_table.to_csv(
        output_dir
        / "trace_derived_workload_thresholds.csv",
        index=False,
    )

    pattern.to_csv(
        output_dir
        / "generated_workload_pattern.csv",
        index=False,
    )

    generated_bins.to_csv(
        output_dir
        / "generated_workload_bins.csv",
        index=False,
    )

    replay.to_csv(
        output_dir
        / "azure_trace_derived_15min_replay.csv",
        index=False,
    )


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(
    threshold_table: pd.DataFrame,
    pattern: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    print("\nTrace-derived workload threshold table")
    print("--------------------------------------")

    print(
        threshold_table[
            [
                "workload_level",
                "trace_derived_definition_rps",
                "mean_observed_rate_rps",
                "p95_observed_rate_rps",
                "maximum_observed_rate_rps",
            ]
        ].to_string(
            index=False,
            float_format=lambda value: (
                f"{value:.3f}"
            ),
        )
    )

    print("\nGenerated phase sequence")
    print("------------------------")

    for _, block in pattern.iterrows():
        print(
            f"{block['start_sec'] / 60.0:6.2f}–"
            f"{block['end_sec'] / 60.0:6.2f} min: "
            f"{block['phase']}"
        )

    print("\nGenerated replay")
    print("----------------")

    print(
        f"Duration: "
        f"{GENERATED_DURATION_SECONDS / 60:.2f} minutes"
    )

    print(
        f"Requests: {len(replay):,}"
    )

    print(
        "Overall mean arrival rate: "
        f"{len(replay) / GENERATED_DURATION_SECONDS:.3f} req/s"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_df = read_trace()

    selected_app_df = get_selected_application(
        trace_df
    )

    (
        observed_bins,
        observed_requests,
        thresholds,
    ) = create_observed_bins(
        selected_app_df
    )

    phase_statistics = calculate_phase_statistics(
        observed_bins
    )

    threshold_table = build_threshold_table(
        thresholds,
        phase_statistics,
    )

    phase_sequence = (
        observed_bins["phase"]
        .astype(str)
        .tolist()
    )

    transition_counts = (
        calculate_transition_counts(
            phase_sequence
        )
    )

    transition_probabilities = (
        calculate_transition_probabilities(
            transition_counts
        )
    )

    phase_runs = (
        calculate_phase_run_lengths(
            phase_sequence
        )
    )

    pattern = generate_best_pattern(
        transition_probabilities,
        phase_runs,
    )

    pattern_bins = expand_pattern_to_bins(
        pattern
    )

    phase_bin_pools = create_phase_bin_pools(
        observed_bins
    )

    generated_bins = sample_generated_bins(
        pattern_bins,
        observed_bins,
        phase_bin_pools,
    )

    replay = create_replay(
        generated_bins,
        observed_requests,
    )

    save_outputs(
        thresholds=thresholds,
        observed_bins=observed_bins,
        phase_statistics=phase_statistics,
        threshold_table=threshold_table,
        pattern=pattern,
        generated_bins=generated_bins,
        replay=replay,
    )

    save_threshold_table(
        threshold_table
    )

    print_summary(
        threshold_table,
        pattern,
        replay,
    )

    plot_workload_pattern(
        replay
    )

    print("\nOutputs saved to:")
    print(output_dir)

    print("\nMain workload figure:")
    print(
        output_dir
        / "final_trace_derived_workload_pattern.pdf"
    )

    print("\nThreshold table:")
    print(
        output_dir
        / "trace_derived_workload_thresholds.csv"
    )

    print("\nMain replay file:")
    print(
        output_dir
        / "azure_trace_derived_15min_replay.csv"
    )


if __name__ == "__main__":
    main()