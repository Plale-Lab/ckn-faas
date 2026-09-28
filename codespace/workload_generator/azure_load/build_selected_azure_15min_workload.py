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
    "selected_application_15min_workload"
)

# Recommended candidate from the previous analysis.
SELECTED_APP = (
    "a9644719e5d173a27e098cffdd5e9227ba9c4a325d082c84d8bcadae6723f19c"
)

TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Use the same 5-second interval used in the application analysis.
SOURCE_BIN_SECONDS = 5.0

# Final workload duration.
GENERATED_DURATION_SECONDS = 15 * 60

# Rate display resolutions.
FINE_RATE_BIN_SECONDS = 1.0
PHASE_RATE_BIN_SECONDS = SOURCE_BIN_SECONDS

# Phase classification from all positive-rate bins of the
# selected application.
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
PATTERN_GENERATION_ATTEMPTS = 50_000

# Workload-quality constraints.
MAX_TOTAL_IDLE_FRACTION = 0.30
MAX_SINGLE_IDLE_DURATION_SECONDS = 120.0

MIN_SPIKE_DURATION_SECONDS = 10.0
MAX_SPIKE_DURATION_SECONDS = 90.0

REQUIRE_INTERNAL_SPIKE = True
REQUIRE_ABRUPT_INCREASE = True
REQUIRE_ABRUPT_DECREASE = True

# Limit phase runs in the generated experiment.
MIN_PHASE_DURATION_SECONDS = SOURCE_BIN_SECONDS
MAX_PHASE_DURATION_SECONDS = 180.0

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
# READ TRACE
# ============================================================

def read_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace not found:\n{trace_path}"
        )

    print("Reading Azure trace...")
    print(trace_path)

    df = pd.read_csv(trace_path)

    # Fallback for whitespace-separated files.
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
            f"Missing columns: {missing_columns}\n"
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
            "The selected application was not found in the trace:\n"
            f"{SELECTED_APP}"
        )

    app_df = app_df.sort_values(
        "start_time_sec"
    ).reset_index(drop=True)

    duration_hours = (
        app_df["start_time_sec"].max()
        - app_df["start_time_sec"].min()
    ) / 3600.0

    print("\nSelected application")
    print("--------------------")
    print(SELECTED_APP)
    print(f"Requests: {len(app_df):,}")
    print(f"Duration: {duration_hours:.2f} hours")

    return app_df


# ============================================================
# CREATE OBSERVED 5-SECOND BINS
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
            "Not enough positive-rate bins to classify phases."
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
        - requests["source_bin_index"]
        * SOURCE_BIN_SECONDS
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

    bin_counts = (
        requests.groupby("source_bin_index")
        .size()
    )

    request_counts[
        bin_counts.index.to_numpy(dtype=int)
    ] = bin_counts.to_numpy(dtype=int)

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
# PHASE STATISTICS
# ============================================================

def calculate_phase_summary(
    observed_bins: pd.DataFrame,
    observed_requests: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for phase in PHASES:
        phase_bins = observed_bins[
            observed_bins["phase"] == phase
        ].copy()

        rates = phase_bins[
            "arrival_rate_rps"
        ].to_numpy(dtype=float)

        phase_requests = observed_requests[
            observed_requests["phase"] == phase
        ].copy()

        # IATs only within the same source bin.
        phase_requests = phase_requests.sort_values(
            [
                "source_bin_index",
                "offset_within_bin_sec",
            ]
        )

        phase_requests["previous_source_bin"] = (
            phase_requests["source_bin_index"]
            .shift(1)
        )

        phase_requests["previous_offset"] = (
            phase_requests["offset_within_bin_sec"]
            .shift(1)
        )

        same_bin = (
            phase_requests["source_bin_index"]
            == phase_requests["previous_source_bin"]
        )

        phase_requests["within_bin_iat_sec"] = (
            phase_requests["offset_within_bin_sec"]
            - phase_requests["previous_offset"]
        )

        iats = phase_requests.loc[
            same_bin
            & (
                phase_requests["within_bin_iat_sec"]
                > 0
            ),
            "within_bin_iat_sec",
        ].to_numpy(dtype=float)

        rows.append(
            {
                "phase": phase,
                "bin_count": int(
                    len(phase_bins)
                ),
                "total_requests": int(
                    phase_bins["request_count"].sum()
                ),
                "minimum_rate_rps": (
                    float(np.min(rates))
                    if len(rates) > 0
                    else np.nan
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
                    safe_percentile(rates, 95)
                    if len(rates) > 0
                    else np.nan
                ),
                "maximum_rate_rps": (
                    float(np.max(rates))
                    if len(rates) > 0
                    else np.nan
                ),
                "valid_within_bin_iats": int(
                    len(iats)
                ),
                "mean_within_bin_iat_sec": (
                    float(np.mean(iats))
                    if len(iats) > 0
                    else np.nan
                ),
                "median_within_bin_iat_sec": (
                    float(np.median(iats))
                    if len(iats) > 0
                    else np.nan
                ),
                "p95_within_bin_iat_sec": (
                    safe_percentile(iats, 95)
                    if len(iats) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# LEARN TRANSITIONS AND RUN LENGTHS
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
        outgoing[current_phase][next_phase] += count

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
# GENERATE A 15-MINUTE PHASE PATTERN
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

    run_bins = int(
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
            run_bins,
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
    total_target_bins = int(
        GENERATED_DURATION_SECONDS
        / SOURCE_BIN_SECONDS
    )

    current_phase = choose_initial_phase(
        phase_runs
    )

    current_bin = 0
    blocks = []

    while current_bin < total_target_bins:
        run_bins = sample_phase_run_bins(
            current_phase,
            phase_runs,
        )

        remaining_bins = (
            total_target_bins
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
                "end_bin": current_bin + run_bins,
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

        if current_bin >= total_target_bins:
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


def score_candidate(
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

    # Require all workload phases.
    for required_phase in REQUIRED_PHASES:
        if phase_durations.get(
            required_phase,
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

    duration_values = np.array(
        [
            phase_durations.get(
                phase,
                0.0,
            )
            for phase in REQUIRED_PHASES
        ],
        dtype=float,
    )

    proportions = (
        duration_values
        / max(
            duration_values.sum(),
            1.0,
        )
    )

    balance_score = (
        1.0
        - float(
            np.std(
                proportions
            )
        )
    )

    transition_count = (
        len(blocks) - 1
    )

    represented_phases = len(
        set(phases)
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

        candidate_score = score_candidate(
            candidate,
            transition_probabilities,
        )

        if candidate_score > best_score:
            best_score = candidate_score
            best_blocks = candidate

    if best_blocks is None:
        raise RuntimeError(
            "Could not generate a valid phase pattern.\n"
            "Try increasing PATTERN_GENERATION_ATTEMPTS or "
            "relaxing one pattern constraint."
        )

    pattern = pd.DataFrame(
        best_blocks
    )

    pattern["block_index"] = np.arange(
        len(pattern)
    )

    pattern["pattern_score"] = best_score

    return pattern


# ============================================================
# COPY ACTUAL AZURE BINS INTO THE NEW WORKLOAD
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
            f"{phase:7s} source bins: "
            f"{len(pools[phase]):,}"
        )

    return pools


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


def generate_workload_from_observed_bins(
    generated_bins: pd.DataFrame,
    observed_bins: pd.DataFrame,
    observed_requests: pd.DataFrame,
    phase_bin_pools: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    generated_bin_rows = []
    generated_request_rows = []

    observed_bin_lookup = (
        observed_bins
        .set_index("source_bin_index")
    )

    grouped_offsets = {
        int(source_bin): group[
            "offset_within_bin_sec"
        ].sort_values().to_numpy(dtype=float)
        for source_bin, group
        in observed_requests.groupby(
            "source_bin_index"
        )
    }

    for _, generated_bin in generated_bins.iterrows():
        phase = str(
            generated_bin["phase"]
        )

        generated_bin_index = int(
            generated_bin[
                "generated_bin_index"
            ]
        )

        generated_bin_start = float(
            generated_bin[
                "generated_bin_start_sec"
            ]
        )

        pool = phase_bin_pools.get(
            phase,
            np.array([], dtype=int),
        )

        if len(pool) == 0:
            raise RuntimeError(
                f"No observed bins exist for phase {phase}."
            )

        source_bin_index = int(
            random.choice(
                pool.tolist()
            )
        )

        source_row = observed_bin_lookup.loc[
            source_bin_index
        ]

        generated_bin_rows.append(
            {
                "generated_bin_index": (
                    generated_bin_index
                ),
                "generated_bin_start_sec": (
                    generated_bin_start
                ),
                "generated_bin_end_sec": (
                    generated_bin_start
                    + SOURCE_BIN_SECONDS
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
                "source_request_count": int(
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

        offsets = grouped_offsets.get(
            source_bin_index,
            np.array([], dtype=float),
        )

        for source_request_position, offset in enumerate(
            offsets
        ):
            scheduled_arrival_sec = (
                generated_bin_start
                + float(offset)
            )

            generated_request_rows.append(
                {
                    "scheduled_arrival_sec": (
                        scheduled_arrival_sec
                    ),
                    "phase": phase,
                    "phase_block_index": int(
                        generated_bin[
                            "phase_block_index"
                        ]
                    ),
                    "generated_bin_index": (
                        generated_bin_index
                    ),
                    "source_bin_index": (
                        source_bin_index
                    ),
                    "source_request_position": (
                        source_request_position
                    ),
                    "offset_within_bin_sec": (
                        float(offset)
                    ),
                    "deadline_ms": (
                        DEFAULT_DEADLINE_MS
                    ),
                }
            )

    generated_bin_df = pd.DataFrame(
        generated_bin_rows
    )

    replay = pd.DataFrame(
        generated_request_rows
    )

    if replay.empty:
        raise RuntimeError(
            "No requests were generated."
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

    replay["iat_seconds"] = (
        replay["scheduled_arrival_sec"]
        .diff()
    )

    replay.loc[
        0,
        "iat_seconds",
    ] = np.nan

    return generated_bin_df, replay


# ============================================================
# VALIDATE GENERATED WORKLOAD
# ============================================================

def calculate_generated_phase_summary(
    generated_bins: pd.DataFrame,
    replay: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for phase in PHASES:
        phase_bins = generated_bins[
            generated_bins["phase"] == phase
        ]

        rates = phase_bins[
            "arrival_rate_rps"
        ].to_numpy(dtype=float)

        phase_requests = replay[
            replay["phase"] == phase
        ].copy()

        # IATs only between requests in the same generated bin.
        phase_requests["previous_generated_bin"] = (
            phase_requests[
                "generated_bin_index"
            ].shift(1)
        )

        phase_requests["same_generated_bin"] = (
            phase_requests[
                "generated_bin_index"
            ]
            == phase_requests[
                "previous_generated_bin"
            ]
        )

        valid_iats = phase_requests.loc[
            phase_requests["same_generated_bin"]
            & (
                phase_requests["iat_seconds"]
                > 0
            ),
            "iat_seconds",
        ].to_numpy(dtype=float)

        rows.append(
            {
                "phase": phase,
                "generated_bin_count": int(
                    len(phase_bins)
                ),
                "duration_seconds": float(
                    len(phase_bins)
                    * SOURCE_BIN_SECONDS
                ),
                "request_count": int(
                    len(phase_requests)
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
                    safe_percentile(rates, 95)
                    if len(rates) > 0
                    else np.nan
                ),
                "maximum_rate_rps": (
                    float(np.max(rates))
                    if len(rates) > 0
                    else np.nan
                ),
                "valid_within_bin_iats": int(
                    len(valid_iats)
                ),
                "mean_within_bin_iat_sec": (
                    float(np.mean(valid_iats))
                    if len(valid_iats) > 0
                    else np.nan
                ),
                "median_within_bin_iat_sec": (
                    float(np.median(valid_iats))
                    if len(valid_iats) > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PLOTTING
# ============================================================

def calculate_rate_series(
    replay: pd.DataFrame,
    bin_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    edges = np.arange(
        0.0,
        GENERATED_DURATION_SECONDS
        + bin_seconds,
        bin_seconds,
    )

    counts, _ = np.histogram(
        replay["scheduled_arrival_sec"],
        bins=edges,
    )

    rates = counts / bin_seconds

    centers_minutes = (
        edges[:-1]
        + bin_seconds / 2.0
    ) / 60.0

    return centers_minutes, rates


def plot_workload(
    pattern: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    fine_x, fine_rates = calculate_rate_series(
        replay,
        FINE_RATE_BIN_SECONDS,
    )

    phase_x, phase_rates = calculate_rate_series(
        replay,
        PHASE_RATE_BIN_SECONDS,
    )

    phase_step_x = (
        pattern["start_sec"]
        .to_numpy(dtype=float)
        / 60.0
    )

    phase_step_y = (
        pattern["phase"]
        .map(PHASE_TO_NUMBER)
        .to_numpy(dtype=float)
    )

    phase_step_x = np.append(
        phase_step_x,
        pattern["end_sec"].iloc[-1]
        / 60.0,
    )

    phase_step_y = np.append(
        phase_step_y,
        phase_step_y[-1],
    )

    valid_iats = replay.dropna(
        subset=["iat_seconds"]
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.2,
                1.2,
                0.8,
                1.0,
            ]
        },
    )

    fine_rate_ax = axes[0]
    phase_rate_ax = axes[1]
    phase_ax = axes[2]
    iat_ax = axes[3]

    fine_rate_ax.step(
        fine_x,
        fine_rates,
        where="mid",
        linewidth=1.1,
    )

    fine_rate_ax.set_title(
        "Trace-derived 15-minute Azure workload"
    )

    fine_rate_ax.set_ylabel(
        "1-second rate\n(req/s)"
    )

    fine_rate_ax.grid(
        True,
        alpha=0.3,
    )

    phase_rate_ax.step(
        phase_x,
        phase_rates,
        where="mid",
        linewidth=1.6,
    )

    phase_rate_ax.fill_between(
        phase_x,
        phase_rates,
        step="mid",
        alpha=0.15,
    )

    phase_rate_ax.set_ylabel(
        "5-second rate\n(req/s)"
    )

    phase_rate_ax.grid(
        True,
        alpha=0.3,
    )

    phase_ax.step(
        phase_step_x,
        phase_step_y,
        where="post",
        linewidth=2.0,
    )

    phase_ax.set_yticks(
        list(
            PHASE_TO_NUMBER.values()
        )
    )

    phase_ax.set_yticklabels(
        list(
            PHASE_TO_NUMBER.keys()
        )
    )

    phase_ax.set_ylabel(
        "Phase"
    )

    phase_ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    iat_ax.scatter(
        valid_iats[
            "scheduled_arrival_sec"
        ] / 60.0,
        valid_iats["iat_seconds"],
        s=8,
        alpha=0.6,
    )

    iat_ax.set_yscale(
        "log"
    )

    iat_ax.set_ylabel(
        "Actual IAT\n(seconds)"
    )

    iat_ax.set_xlabel(
        "Generated workload time (minutes)"
    )

    iat_ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "selected_azure_15min_workload.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "selected_azure_15min_workload.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def plot_phase_rate_comparison(
    source_phase_summary: pd.DataFrame,
    generated_phase_summary: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    active_phases = [
        "Low",
        "Medium",
        "High",
        "Spike",
    ]

    source_values = [
        float(
            source_phase_summary.loc[
                source_phase_summary["phase"] == phase,
                "mean_rate_rps",
            ].iloc[0]
        )
        for phase in active_phases
    ]

    generated_values = [
        float(
            generated_phase_summary.loc[
                generated_phase_summary["phase"] == phase,
                "mean_rate_rps",
            ].iloc[0]
        )
        for phase in active_phases
    ]

    x = np.arange(
        len(active_phases)
    )

    width = 0.36

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.bar(
        x - width / 2,
        source_values,
        width,
        label="Complete selected application",
    )

    ax.bar(
        x + width / 2,
        generated_values,
        width,
        label="Generated 15-minute workload",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(active_phases)

    ax.set_xlabel(
        "Workload phase"
    )

    ax.set_ylabel(
        "Mean arrival rate (requests/second)"
    )

    ax.set_title(
        "Source versus generated phase arrival rates"
    )

    ax.legend()

    ax.grid(
        True,
        axis="y",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "source_vs_generated_phase_rates.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "source_vs_generated_phase_rates.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_outputs(
    thresholds: dict,
    observed_bins: pd.DataFrame,
    source_phase_summary: pd.DataFrame,
    pattern: pd.DataFrame,
    generated_bins: pd.DataFrame,
    replay: pd.DataFrame,
    generated_phase_summary: pd.DataFrame,
    transition_counts: Counter,
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
                **thresholds,
            }
        ]
    ).to_csv(
        output_dir
        / "selected_application_thresholds.csv",
        index=False,
    )

    observed_bins.to_csv(
        output_dir
        / "selected_application_observed_bins.csv",
        index=False,
    )

    source_phase_summary.to_csv(
        output_dir
        / "selected_application_phase_summary.csv",
        index=False,
    )

    pattern.to_csv(
        output_dir
        / "generated_15min_phase_pattern.csv",
        index=False,
    )

    generated_bins.to_csv(
        output_dir
        / "generated_15min_bins.csv",
        index=False,
    )

    replay.to_csv(
        output_dir
        / "azure_trace_derived_15min_replay.csv",
        index=False,
    )

    generated_phase_summary.to_csv(
        output_dir
        / "generated_15min_phase_summary.csv",
        index=False,
    )

    transition_rows = [
        {
            "current_phase": current_phase,
            "next_phase": next_phase,
            "count": count,
        }
        for (
            current_phase,
            next_phase,
        ), count in transition_counts.items()
    ]

    pd.DataFrame(
        transition_rows
    ).to_csv(
        output_dir
        / "selected_application_transition_counts.csv",
        index=False,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    thresholds: dict,
    source_phase_summary: pd.DataFrame,
    pattern: pd.DataFrame,
    replay: pd.DataFrame,
    generated_phase_summary: pd.DataFrame,
) -> None:
    print("\nActual phase thresholds")
    print("-----------------------")

    print("Idle: rate = 0 req/s")

    print(
        "Low: 0 < rate <= "
        f"{thresholds['low_upper_rps']:.2f} req/s"
    )

    print(
        "Medium: "
        f"{thresholds['low_upper_rps']:.2f} < rate <= "
        f"{thresholds['medium_upper_rps']:.2f} req/s"
    )

    print(
        "High: "
        f"{thresholds['medium_upper_rps']:.2f} < rate <= "
        f"{thresholds['high_upper_rps']:.2f} req/s"
    )

    print(
        "Spike: rate > "
        f"{thresholds['high_upper_rps']:.2f} req/s"
    )

    print("\nComplete selected application phase statistics")
    print("----------------------------------------------")

    print(
        source_phase_summary[
            [
                "phase",
                "bin_count",
                "mean_rate_rps",
                "median_rate_rps",
                "p95_rate_rps",
                "maximum_rate_rps",
                "median_within_bin_iat_sec",
            ]
        ].to_string(index=False)
    )

    print("\nGenerated phase sequence")
    print("------------------------")

    for _, block in pattern.iterrows():
        print(
            f"{block['start_sec'] / 60.0:6.2f}–"
            f"{block['end_sec'] / 60.0:6.2f} min: "
            f"{block['phase']}"
        )

    print("\nGenerated workload statistics")
    print("-----------------------------")

    print(
        f"Duration: "
        f"{GENERATED_DURATION_SECONDS / 60:.2f} minutes"
    )

    print(
        f"Requests: {len(replay):,}"
    )

    print(
        "Overall mean rate: "
        f"{len(replay) / GENERATED_DURATION_SECONDS:.2f} req/s"
    )

    print("\nGenerated phase statistics")
    print("--------------------------")

    print(
        generated_phase_summary[
            [
                "phase",
                "duration_seconds",
                "request_count",
                "mean_rate_rps",
                "median_rate_rps",
                "p95_rate_rps",
                "maximum_rate_rps",
                "median_within_bin_iat_sec",
            ]
        ].to_string(index=False)
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

    source_phase_summary = calculate_phase_summary(
        observed_bins,
        observed_requests,
    )

    phase_sequence = (
        observed_bins["phase"]
        .astype(str)
        .tolist()
    )

    transition_counts = calculate_transition_counts(
        phase_sequence
    )

    transition_probabilities = (
        calculate_transition_probabilities(
            transition_counts
        )
    )

    phase_runs = calculate_phase_run_lengths(
        phase_sequence
    )

    pattern = generate_best_pattern(
        transition_probabilities,
        phase_runs,
    )

    phase_bin_pools = create_phase_bin_pools(
        observed_bins
    )

    generated_pattern_bins = (
        expand_pattern_to_bins(
            pattern
        )
    )

    (
        generated_bins,
        replay,
    ) = generate_workload_from_observed_bins(
        generated_pattern_bins,
        observed_bins,
        observed_requests,
        phase_bin_pools,
    )

    generated_phase_summary = (
        calculate_generated_phase_summary(
            generated_bins,
            replay,
        )
    )

    save_outputs(
        thresholds=thresholds,
        observed_bins=observed_bins,
        source_phase_summary=(
            source_phase_summary
        ),
        pattern=pattern,
        generated_bins=generated_bins,
        replay=replay,
        generated_phase_summary=(
            generated_phase_summary
        ),
        transition_counts=transition_counts,
    )

    print_summary(
        thresholds=thresholds,
        source_phase_summary=(
            source_phase_summary
        ),
        pattern=pattern,
        replay=replay,
        generated_phase_summary=(
            generated_phase_summary
        ),
    )

    plot_workload(
        pattern,
        replay,
    )

    plot_phase_rate_comparison(
        source_phase_summary,
        generated_phase_summary,
    )

    print("\nOutputs saved to:")
    print(output_dir)

    print("\nMain replay file:")
    print(
        output_dir
        / "azure_trace_derived_15min_replay.csv"
    )


if __name__ == "__main__":
    main()