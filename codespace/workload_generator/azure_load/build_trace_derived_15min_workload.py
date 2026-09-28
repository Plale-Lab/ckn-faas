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
    "trace_derived_15min_workload"
)

TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Used for phase classification, transition analysis,
# and arrival-rate visualization.
ANALYSIS_BIN_SECONDS = 30.0

GENERATED_DURATION_SECONDS = 15 * 60

MIN_APP_REQUESTS = 500
MIN_ACTIVE_BINS = 20

# None means analyze every application.
MAX_APPS_TO_ANALYZE = None

PATTERN_GENERATION_ATTEMPTS = 20_000
RANDOM_SEED = 42

DEFAULT_DEADLINE_MS = 2000
MAX_GENERATED_REQUESTS = None

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

LOW_PERCENTILE = 35
HIGH_PERCENTILE = 70
SPIKE_PERCENTILE = 95

REQUIRED_PHASES = [
    "Idle",
    "Low",
    "Medium",
    "High",
]

REQUIRE_SPIKE_WHEN_AVAILABLE = True
MIN_SPIKE_BINS_TO_REQUIRE = 5

MAX_TOTAL_IDLE_FRACTION = 0.30
MAX_SINGLE_IDLE_DURATION_SECONDS = 180.0

MIN_SPIKE_DURATION_SECONDS = 30.0
MAX_SPIKE_DURATION_SECONDS = 120.0

MIN_PHASE_DURATION_SECONDS = ANALYSIS_BIN_SECONDS
MAX_PHASE_DURATION_SECONDS = 180.0

REQUIRE_ABRUPT_UPWARD_TRANSITION = True
REQUIRE_ABRUPT_DOWNWARD_TRANSITION = True
REQUIRE_INTERNAL_SPIKE = True


# ============================================================
# TRACE READING
# ============================================================

def unit_divisor(unit: str) -> float:
    values = {
        "seconds": 1.0,
        "milliseconds": 1_000.0,
        "microseconds": 1_000_000.0,
    }

    normalized = unit.strip().lower()

    if normalized not in values:
        raise ValueError(
            "Unit must be seconds, milliseconds, or microseconds."
        )

    return values[normalized]


def read_azure_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace not found:\n{trace_path}"
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

    required_columns = [
        "app",
        "func",
        "end_timestamp",
        "duration",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
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
        [
            "app",
            "start_time_sec",
        ]
    ).reset_index(drop=True)

    print(f"Valid requests: {len(df):,}")
    print(f"Applications: {df['app'].nunique():,}")
    print(f"Functions: {df['func'].nunique():,}")

    return df


# ============================================================
# PHASE CLASSIFICATION
# ============================================================

def classify_arrival_rates(
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
        return phases, {
            "low_threshold_rps": 0.0,
            "high_threshold_rps": 0.0,
            "spike_threshold_rps": 0.0,
        }

    low_threshold = float(
        np.percentile(
            positive_rates,
            LOW_PERCENTILE,
        )
    )

    high_threshold = float(
        np.percentile(
            positive_rates,
            HIGH_PERCENTILE,
        )
    )

    spike_threshold = float(
        np.percentile(
            positive_rates,
            SPIKE_PERCENTILE,
        )
    )

    positive = arrival_rates > 0

    phases[
        positive
        & (arrival_rates <= low_threshold)
    ] = "Low"

    phases[
        positive
        & (arrival_rates > low_threshold)
        & (arrival_rates <= high_threshold)
    ] = "Medium"

    phases[
        positive
        & (arrival_rates > high_threshold)
        & (arrival_rates <= spike_threshold)
    ] = "High"

    phases[
        positive
        & (arrival_rates > spike_threshold)
    ] = "Spike"

    thresholds = {
        "low_threshold_rps": low_threshold,
        "high_threshold_rps": high_threshold,
        "spike_threshold_rps": spike_threshold,
    }

    return phases, thresholds


def create_application_pattern(
    app_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    app_df = app_df.sort_values(
        "start_time_sec"
    ).copy()

    first_time = float(
        app_df["start_time_sec"].min()
    )

    last_time = float(
        app_df["start_time_sec"].max()
    )

    app_df["relative_time_sec"] = (
        app_df["start_time_sec"]
        - first_time
    )

    duration = max(
        last_time - first_time,
        ANALYSIS_BIN_SECONDS,
    )

    edges = np.arange(
        0.0,
        duration + ANALYSIS_BIN_SECONDS,
        ANALYSIS_BIN_SECONDS,
    )

    if len(edges) < 2:
        edges = np.array(
            [
                0.0,
                ANALYSIS_BIN_SECONDS,
            ]
        )

    counts, _ = np.histogram(
        app_df["relative_time_sec"],
        bins=edges,
    )

    arrival_rates = (
        counts / ANALYSIS_BIN_SECONDS
    )

    phases, thresholds = classify_arrival_rates(
        arrival_rates
    )

    pattern = pd.DataFrame(
        {
            "bin_index": np.arange(len(counts)),
            "bin_start_sec": edges[:-1],
            "bin_end_sec": edges[1:],
            "request_count": counts,
            "arrival_rate_rps": arrival_rates,
            "phase": phases,
        }
    )

    pattern["time_hours"] = (
        pattern["bin_start_sec"] / 3600.0
    )

    pattern["phase_number"] = (
        pattern["phase"]
        .map(PHASE_TO_NUMBER)
    )

    return pattern, thresholds


# ============================================================
# APPLICATION SELECTION
# ============================================================

def normalized_entropy(
    values: list[int],
) -> float:
    values_array = np.asarray(
        values,
        dtype=float,
    )

    total = float(
        values_array.sum()
    )

    if total <= 0:
        return 0.0

    probabilities = (
        values_array[values_array > 0]
        / total
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    maximum_entropy = math.log(
        len(values_array)
    )

    if maximum_entropy <= 0:
        return 0.0

    return entropy / maximum_entropy


def collapse_phase_sequence(
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


def calculate_phase_transitions(
    phase_sequence: list[str],
) -> Counter:
    collapsed = collapse_phase_sequence(
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


def calculate_phase_runs(
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


def analyze_application(
    app_name: str,
    app_df: pd.DataFrame,
) -> dict | None:
    if len(app_df) < MIN_APP_REQUESTS:
        return None

    pattern, thresholds = (
        create_application_pattern(
            app_df
        )
    )

    phase_counts = {
        phase: int(
            (
                pattern["phase"]
                == phase
            ).sum()
        )
        for phase in PHASES
    }

    active_bins = (
        len(pattern)
        - phase_counts["Idle"]
    )

    if active_bins < MIN_ACTIVE_BINS:
        return None

    if not all(
        phase_counts[phase] > 0
        for phase in REQUIRED_PHASES
    ):
        return None

    active_rates = pattern.loc[
        pattern["arrival_rate_rps"] > 0,
        "arrival_rate_rps",
    ]

    if active_rates.empty:
        return None

    mean_rate = float(
        active_rates.mean()
    )

    peak_rate = float(
        active_rates.max()
    )

    rate_cv = (
        float(
            active_rates.std()
            / mean_rate
        )
        if mean_rate > 0
        else 0.0
    )

    phase_sequence = (
        pattern["phase"]
        .astype(str)
        .tolist()
    )

    transitions = (
        calculate_phase_transitions(
            phase_sequence
        )
    )

    phase_entropy = normalized_entropy(
        [
            phase_counts[phase]
            for phase in PHASES
        ]
    )

    represented_phase_count = sum(
        phase_counts[phase] > 0
        for phase in PHASES
    )

    idle_fraction = (
        phase_counts["Idle"]
        / len(pattern)
    )

    useful_idle_score = max(
        0.0,
        1.0
        - abs(idle_fraction - 0.20),
    )

    selection_score = (
        8.0 * phase_entropy
        + 1.5 * represented_phase_count
        + 0.35 * len(transitions)
        + 1.5 * rate_cv
        + np.log1p(len(app_df))
        + 3.0 * useful_idle_score
    )

    return {
        "app": str(app_name),
        "request_count": int(len(app_df)),
        "trace_bins": int(len(pattern)),
        "active_bins": int(active_bins),
        "idle_bins": phase_counts["Idle"],
        "low_bins": phase_counts["Low"],
        "medium_bins": phase_counts["Medium"],
        "high_bins": phase_counts["High"],
        "spike_bins": phase_counts["Spike"],
        "represented_phase_count": represented_phase_count,
        "phase_entropy": phase_entropy,
        "unique_transition_count": len(transitions),
        "mean_active_rate_rps": mean_rate,
        "peak_rate_rps": peak_rate,
        "rate_cv": rate_cv,
        "idle_fraction": idle_fraction,
        "selection_score": selection_score,
        **thresholds,
    }


def select_best_application(
    df: pd.DataFrame,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    app_counts = (
        df.groupby("app")
        .size()
        .sort_values(
            ascending=False
        )
    )

    if MAX_APPS_TO_ANALYZE is not None:
        app_counts = app_counts.head(
            MAX_APPS_TO_ANALYZE
        )

    results = []

    print("\nAnalyzing applications...")

    for index, app_name in enumerate(
        app_counts.index,
        start=1,
    ):
        print(
            f"[{index}/{len(app_counts)}] "
            f"Application: {app_name}"
        )

        app_df = df[
            df["app"] == app_name
        ].copy()

        result = analyze_application(
            str(app_name),
            app_df,
        )

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            "No suitable application was found. "
            "Try decreasing MIN_APP_REQUESTS "
            "or MIN_ACTIVE_BINS."
        )

    ranking = pd.DataFrame(
        results
    ).sort_values(
        "selection_score",
        ascending=False,
    ).reset_index(drop=True)

    selected_app = str(
        ranking.iloc[0]["app"]
    )

    selected_app_df = df[
        df["app"].astype(str)
        == selected_app
    ].copy()

    print("\nSelected application:")
    print(selected_app)

    print("\nTop candidates:")
    print(
        ranking[
            [
                "app",
                "request_count",
                "idle_bins",
                "low_bins",
                "medium_bins",
                "high_bins",
                "spike_bins",
                "unique_transition_count",
                "rate_cv",
                "selection_score",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    return (
        selected_app,
        selected_app_df,
        ranking,
    )


# ============================================================
# TRANSITION MODEL
# ============================================================

def build_transition_probabilities(
    transitions: Counter,
) -> dict[str, dict[str, float]]:
    outgoing = defaultdict(Counter)

    for (
        current_phase,
        next_phase,
    ), count in transitions.items():
        outgoing[current_phase][next_phase] += count

    probabilities = {}

    for current_phase, counts in outgoing.items():
        total = sum(
            counts.values()
        )

        probabilities[current_phase] = {
            next_phase: count / total
            for next_phase, count
            in counts.items()
        }

    return probabilities


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
    possible = [
        phase
        for phase in PHASES
        if phase_runs.get(phase)
    ]

    if not possible:
        return "Low"

    weights = [
        len(phase_runs[phase])
        for phase in possible
    ]

    return random.choices(
        possible,
        weights=weights,
        k=1,
    )[0]


def sample_phase_duration(
    phase: str,
    phase_runs: dict[str, list[int]],
) -> float:
    observed_runs = phase_runs.get(
        phase,
        [],
    )

    run_bins = (
        random.choice(observed_runs)
        if observed_runs
        else 1
    )

    duration = (
        run_bins
        * ANALYSIS_BIN_SECONDS
    )

    duration = max(
        duration,
        MIN_PHASE_DURATION_SECONDS,
    )

    duration = min(
        duration,
        MAX_PHASE_DURATION_SECONDS,
    )

    if phase == "Idle":
        duration = min(
            duration,
            MAX_SINGLE_IDLE_DURATION_SECONDS,
        )

    return float(duration)


def generate_phase_candidate(
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
    phase_runs: dict[str, list[int]],
) -> list[dict]:
    phase = choose_initial_phase(
        phase_runs
    )

    current_time = 0.0
    blocks = []

    while (
        current_time
        < GENERATED_DURATION_SECONDS
    ):
        duration = sample_phase_duration(
            phase,
            phase_runs,
        )

        remaining = (
            GENERATED_DURATION_SECONDS
            - current_time
        )

        duration = min(
            duration,
            remaining,
        )

        blocks.append(
            {
                "phase": phase,
                "start_sec": current_time,
                "end_sec": (
                    current_time
                    + duration
                ),
                "duration_sec": duration,
            }
        )

        current_time += duration

        if (
            current_time
            >= GENERATED_DURATION_SECONDS
        ):
            break

        outgoing = (
            transition_probabilities.get(
                phase,
                {},
            )
        )

        if outgoing:
            phase = weighted_choice(
                outgoing
            )
        else:
            phase = choose_initial_phase(
                phase_runs
            )

    return blocks


def candidate_pattern_score(
    blocks: list[dict],
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
    required_phases: list[str],
) -> float:
    if not blocks:
        return -float("inf")

    phases = [
        block["phase"]
        for block in blocks
    ]

    total_duration = sum(
        block["duration_sec"]
        for block in blocks
    )

    durations = defaultdict(float)

    for block in blocks:
        durations[block["phase"]] += (
            block["duration_sec"]
        )

    for phase in required_phases:
        if durations.get(
            phase,
            0.0,
        ) <= 0:
            return -float("inf")

    idle_fraction = (
        durations.get(
            "Idle",
            0.0,
        )
        / max(total_duration, 1.0)
    )

    if (
        idle_fraction
        > MAX_TOTAL_IDLE_FRACTION
    ):
        return -float("inf")

    for block in blocks:
        if (
            block["phase"] == "Idle"
            and block["duration_sec"]
            > MAX_SINGLE_IDLE_DURATION_SECONDS
        ):
            return -float("inf")

    spike_indices = [
        index
        for index, block in enumerate(blocks)
        if block["phase"] == "Spike"
    ]

    spike_durations = [
        block["duration_sec"]
        for block in blocks
        if block["phase"] == "Spike"
    ]

    if "Spike" in required_phases:
        if not spike_durations:
            return -float("inf")

        valid_spike = any(
            MIN_SPIKE_DURATION_SECONDS
            <= duration
            <= MAX_SPIKE_DURATION_SECONDS
            for duration in spike_durations
        )

        if not valid_spike:
            return -float("inf")

    if (
        REQUIRE_INTERNAL_SPIKE
        and spike_indices
        and all(
            index == 0
            for index in spike_indices
        )
    ):
        return -float("inf")

    transition_log_probability = 0.0
    abrupt_upward = False
    abrupt_downward = False

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

        transition_log_probability += (
            math.log(
                probability + 1e-12
            )
        )

        current_level = (
            PHASE_TO_NUMBER[
                current_phase
            ]
        )

        next_level = (
            PHASE_TO_NUMBER[
                next_phase
            ]
        )

        if (
            next_level
            - current_level
            >= 2
        ):
            abrupt_upward = True

        if (
            current_level
            - next_level
            >= 2
        ):
            abrupt_downward = True

    if (
        REQUIRE_ABRUPT_UPWARD_TRANSITION
        and not abrupt_upward
    ):
        return -float("inf")

    if (
        REQUIRE_ABRUPT_DOWNWARD_TRANSITION
        and not abrupt_downward
    ):
        return -float("inf")

    represented_phases = len(
        set(phases)
    )

    required_duration_values = np.array(
        [
            durations.get(
                phase,
                0.0,
            )
            for phase in required_phases
        ],
        dtype=float,
    )

    duration_proportions = (
        required_duration_values
        / max(
            required_duration_values.sum(),
            1.0,
        )
    )

    balance_score = (
        1.0
        - float(
            np.std(
                duration_proportions
            )
        )
    )

    transition_count = max(
        len(phases) - 1,
        0,
    )

    return (
        10.0 * represented_phases
        + 8.0 * balance_score
        + 0.5 * transition_count
        + 2.0 * float(abrupt_upward)
        + 2.0 * float(abrupt_downward)
        + transition_log_probability
    )


def generate_best_phase_pattern(
    transition_probabilities: dict[
        str,
        dict[str, float],
    ],
    phase_runs: dict[str, list[int]],
    selected_pattern: pd.DataFrame,
) -> pd.DataFrame:
    required_phases = list(
        REQUIRED_PHASES
    )

    spike_bin_count = int(
        (
            selected_pattern["phase"]
            == "Spike"
        ).sum()
    )

    if (
        REQUIRE_SPIKE_WHEN_AVAILABLE
        and spike_bin_count
        >= MIN_SPIKE_BINS_TO_REQUIRE
    ):
        required_phases.append(
            "Spike"
        )

    print("\nRequired phases:")
    print(required_phases)

    best_blocks = None
    best_score = -float("inf")

    for _ in range(
        PATTERN_GENERATION_ATTEMPTS
    ):
        candidate = (
            generate_phase_candidate(
                transition_probabilities,
                phase_runs,
            )
        )

        score = candidate_pattern_score(
            candidate,
            transition_probabilities,
            required_phases,
        )

        if score > best_score:
            best_score = score
            best_blocks = candidate

    if best_blocks is None:
        raise RuntimeError(
            "No valid 15-minute pattern was found.\n"
            "Try increasing "
            "PATTERN_GENERATION_ATTEMPTS, "
            "increasing MAX_TOTAL_IDLE_FRACTION, "
            "or disabling one abrupt-transition "
            "requirement."
        )

    generated_pattern = pd.DataFrame(
        best_blocks
    )

    generated_pattern["block_index"] = (
        np.arange(
            len(generated_pattern)
        )
    )

    generated_pattern["phase_number"] = (
        generated_pattern["phase"]
        .map(PHASE_TO_NUMBER)
    )

    generated_pattern["pattern_score"] = (
        best_score
    )

    return generated_pattern


# ============================================================
# VALID PHASE-SPECIFIC IAT POOLS
# ============================================================

def assign_request_phases(
    app_df: pd.DataFrame,
    application_pattern: pd.DataFrame,
) -> pd.DataFrame:
    requests = app_df.sort_values(
        "start_time_sec"
    ).copy()

    first_time = float(
        requests[
            "start_time_sec"
        ].min()
    )

    requests["relative_time_sec"] = (
        requests["start_time_sec"]
        - first_time
    )

    requests["bin_index"] = np.floor(
        requests["relative_time_sec"]
        / ANALYSIS_BIN_SECONDS
    ).astype(int)

    phase_lookup = (
        application_pattern
        .set_index("bin_index")["phase"]
        .to_dict()
    )

    requests["phase"] = (
        requests["bin_index"]
        .map(phase_lookup)
        .fillna("Idle")
    )

    requests[
        "previous_start_time_sec"
    ] = requests[
        "start_time_sec"
    ].shift(1)

    requests["previous_phase"] = (
        requests["phase"].shift(1)
    )

    requests[
        "previous_bin_index"
    ] = requests[
        "bin_index"
    ].shift(1)

    requests[
        "original_iat_seconds"
    ] = (
        requests["start_time_sec"]
        - requests[
            "previous_start_time_sec"
        ]
    )

    same_phase = (
        requests["phase"]
        == requests["previous_phase"]
    )

    same_bin = (
        requests["bin_index"]
        == requests[
            "previous_bin_index"
        ]
    )

    # Important correction:
    # An IAT is used in a phase pool only when both
    # consecutive requests are in the same phase and
    # same source bin.
    requests["valid_phase_iat"] = (
        same_phase
        & same_bin
        & (
            requests["phase"]
            != "Idle"
        )
        & (
            requests[
                "original_iat_seconds"
            ] > 0
        )
        & np.isfinite(
            requests[
                "original_iat_seconds"
            ]
        )
    )

    return requests


def build_phase_iat_pools(
    requests_with_phases: pd.DataFrame,
) -> dict[str, np.ndarray]:
    pools = {}

    print("\nPhase IAT pools")
    print("----------------")

    for phase in [
        "Low",
        "Medium",
        "High",
        "Spike",
    ]:
        values = (
            requests_with_phases.loc[
                (
                    requests_with_phases[
                        "phase"
                    ]
                    == phase
                )
                & (
                    requests_with_phases[
                        "valid_phase_iat"
                    ]
                ),
                "original_iat_seconds",
            ]
            .dropna()
            .to_numpy(dtype=float)
        )

        values = values[
            np.isfinite(values)
            & (values > 0)
        ]

        if len(values) > 10:
            upper_limit = float(
                np.percentile(
                    values,
                    99,
                )
            )

            values = values[
                values <= upper_limit
            ]

        pools[phase] = values

        if len(values) > 0:
            print(
                f"{phase:7s}: "
                f"count={len(values):6d}, "
                f"median={np.median(values):8.4f}s, "
                f"mean={np.mean(values):8.4f}s, "
                f"p95={np.percentile(values, 95):8.4f}s"
            )
        else:
            print(
                f"{phase:7s}: "
                "no valid IAT samples"
            )

    return pools


def fallback_iat_for_phase(
    phase: str,
    application_pattern: pd.DataFrame,
) -> float:
    rates = application_pattern.loc[
        application_pattern["phase"]
        == phase,
        "arrival_rate_rps",
    ]

    rates = rates[
        rates > 0
    ]

    if not rates.empty:
        median_rate = float(
            rates.median()
        )

        if median_rate > 0:
            return (
                1.0
                / median_rate
            )

    fallback_values = {
        "Low": 10.0,
        "Medium": 3.0,
        "High": 1.0,
        "Spike": 0.25,
    }

    return fallback_values.get(
        phase,
        2.0,
    )


# ============================================================
# REQUEST GENERATION
# ============================================================

def generate_request_arrivals(
    generated_pattern: pd.DataFrame,
    iat_pools: dict[str, np.ndarray],
    application_pattern: pd.DataFrame,
) -> pd.DataFrame:
    arrivals = []

    for _, block in (
        generated_pattern.iterrows()
    ):
        phase = str(
            block["phase"]
        )

        block_start = float(
            block["start_sec"]
        )

        block_end = float(
            block["end_sec"]
        )

        # Idle blocks intentionally have no requests.
        if phase == "Idle":
            continue

        pool = iat_pools.get(
            phase,
            np.array(
                [],
                dtype=float,
            ),
        )

        fallback_iat = (
            fallback_iat_for_phase(
                phase,
                application_pattern,
            )
        )

        current_time = block_start

        while True:
            if len(pool) > 0:
                sampled_iat = float(
                    random.choice(
                        pool.tolist()
                    )
                )
            else:
                sampled_iat = (
                    fallback_iat
                )

            sampled_iat = max(
                sampled_iat,
                0.001,
            )

            next_arrival = (
                current_time
                + sampled_iat
            )

            if next_arrival >= block_end:
                break

            arrivals.append(
                {
                    "scheduled_arrival_sec": (
                        next_arrival
                    ),
                    "phase": phase,
                    "source_iat_seconds": (
                        sampled_iat
                    ),
                    "phase_block_index": int(
                        block[
                            "block_index"
                        ]
                    ),
                }
            )

            current_time = (
                next_arrival
            )

            if (
                MAX_GENERATED_REQUESTS
                is not None
                and len(arrivals)
                >= MAX_GENERATED_REQUESTS
            ):
                break

        if (
            MAX_GENERATED_REQUESTS
            is not None
            and len(arrivals)
            >= MAX_GENERATED_REQUESTS
        ):
            break

    replay = pd.DataFrame(
        arrivals
    )

    if replay.empty:
        raise RuntimeError(
            "No requests were generated."
        )

    replay = replay.sort_values(
        "scheduled_arrival_sec"
    ).reset_index(drop=True)

    replay["request_index"] = (
        np.arange(len(replay))
    )

    # Actual elapsed time between generated requests.
    # This includes idle periods.
    replay["iat_seconds"] = (
        replay[
            "scheduled_arrival_sec"
        ].diff()
    )

    # First request has no previous request.
    replay.loc[
        0,
        "iat_seconds",
    ] = np.nan

    replay["deadline_ms"] = (
        DEFAULT_DEADLINE_MS
    )

    return replay


# ============================================================
# TRANSITION TABLES
# ============================================================

def create_transition_tables(
    transitions: Counter,
    probability_dict: dict[
        str,
        dict[str, float],
    ],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    count_matrix = pd.DataFrame(
        0,
        index=PHASES,
        columns=PHASES,
        dtype=int,
    )

    probability_matrix = pd.DataFrame(
        0.0,
        index=PHASES,
        columns=PHASES,
    )

    for (
        current_phase,
        next_phase,
    ), count in transitions.items():
        count_matrix.loc[
            current_phase,
            next_phase,
        ] = count

    for (
        current_phase,
        next_probabilities,
    ) in probability_dict.items():
        for (
            next_phase,
            probability,
        ) in next_probabilities.items():
            probability_matrix.loc[
                current_phase,
                next_phase,
            ] = probability

    return (
        count_matrix,
        probability_matrix,
    )


# ============================================================
# PLOTTING
# ============================================================

def plot_selected_application(
    selected_app: str,
    pattern: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    ax.plot(
        pattern["time_hours"],
        pattern["arrival_rate_rps"],
        linewidth=0.9,
    )

    ax.set_title(
        "Complete workload of selected "
        f"Azure application: {selected_app}"
    )

    ax.set_xlabel(
        "Time from beginning of trace (hours)"
    )

    ax.set_ylabel(
        "Arrival rate (requests/second)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "selected_application_complete_workload.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "selected_application_complete_workload.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def plot_transition_matrix(
    probability_matrix: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        probability_matrix.to_numpy(),
        aspect="auto",
    )

    ax.set_xticks(
        np.arange(len(PHASES))
    )

    ax.set_yticks(
        np.arange(len(PHASES))
    )

    ax.set_xticklabels(
        PHASES,
        rotation=30,
        ha="right",
    )

    ax.set_yticklabels(
        PHASES
    )

    ax.set_xlabel(
        "Next phase"
    )

    ax.set_ylabel(
        "Current phase"
    )

    ax.set_title(
        "Observed Azure workload "
        "transition probabilities"
    )

    for row_index in range(
        len(PHASES)
    ):
        for column_index in range(
            len(PHASES)
        ):
            value = (
                probability_matrix.iloc[
                    row_index,
                    column_index,
                ]
            )

            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
        label="Transition probability",
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "observed_transition_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "observed_transition_matrix.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def plot_generated_workload(
    generated_pattern: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    # Use same bin size used for phase classification.
    bin_seconds = (
        ANALYSIS_BIN_SECONDS
    )

    edges = np.arange(
        0.0,
        GENERATED_DURATION_SECONDS
        + bin_seconds,
        bin_seconds,
    )

    request_counts, _ = np.histogram(
        replay[
            "scheduled_arrival_sec"
        ],
        bins=edges,
    )

    arrival_rates = (
        request_counts
        / bin_seconds
    )

    centers_minute = (
        edges[:-1]
        + bin_seconds / 2.0
    ) / 60.0

    phase_x = (
        generated_pattern[
            "start_sec"
        ].to_numpy(dtype=float)
        / 60.0
    )

    phase_y = (
        generated_pattern["phase"]
        .map(PHASE_TO_NUMBER)
        .to_numpy(dtype=float)
    )

    # Add final point so last block is visible.
    phase_x = np.append(
        phase_x,
        float(
            generated_pattern[
                "end_sec"
            ].iloc[-1]
        )
        / 60.0,
    )

    phase_y = np.append(
        phase_y,
        phase_y[-1],
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7.5),
        sharex=True,
        gridspec_kw={
            "height_ratios": [
                1.4,
                0.8,
                1.0,
            ]
        },
    )

    rate_ax = axes[0]
    phase_ax = axes[1]
    iat_ax = axes[2]

    # Arrival rate.
    rate_ax.step(
        centers_minute,
        arrival_rates,
        where="mid",
        linewidth=1.8,
    )

    rate_ax.fill_between(
        centers_minute,
        arrival_rates,
        step="mid",
        alpha=0.15,
    )

    rate_ax.set_title(
        "Trace-derived 15-minute "
        "Azure workload"
    )

    rate_ax.set_ylabel(
        "Arrival rate\n(req/s)"
    )

    rate_ax.grid(
        True,
        alpha=0.3,
    )

    # Workload phase.
    phase_ax.step(
        phase_x,
        phase_y,
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
        "Workload\nphase"
    )

    phase_ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    # Actual generated IAT observations.
    valid_iat = replay.dropna(
        subset=["iat_seconds"]
    ).copy()

    iat_ax.scatter(
        valid_iat[
            "scheduled_arrival_sec"
        ]
        / 60.0,
        valid_iat[
            "iat_seconds"
        ],
        s=12,
        alpha=0.7,
    )

    iat_ax.set_xlabel(
        "Generated workload time (minutes)"
    )

    iat_ax.set_ylabel(
        "Actual IAT\n(seconds)"
    )

    iat_ax.set_yscale(
        "log"
    )

    iat_ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "generated_trace_derived_15min_workload.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "generated_trace_derived_15min_workload.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# OUTPUT
# ============================================================

def save_outputs(
    selected_app: str,
    ranking: pd.DataFrame,
    selected_pattern: pd.DataFrame,
    thresholds: dict,
    transition_counts: pd.DataFrame,
    transition_probabilities: pd.DataFrame,
    generated_pattern: pd.DataFrame,
    replay: pd.DataFrame,
    requests_with_phases: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ranking.to_csv(
        output_dir
        / "azure_application_ranking.csv",
        index=False,
    )

    selected_pattern.to_csv(
        output_dir
        / "selected_application_complete_pattern.csv",
        index=False,
    )

    requests_with_phases.to_csv(
        output_dir
        / "selected_application_requests_with_valid_iats.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "selected_app": selected_app,
                "analysis_bin_seconds": (
                    ANALYSIS_BIN_SECONDS
                ),
                **thresholds,
            }
        ]
    ).to_csv(
        output_dir
        / "selected_application_thresholds.csv",
        index=False,
    )

    transition_counts.to_csv(
        output_dir
        / "observed_transition_counts.csv",
    )

    transition_probabilities.to_csv(
        output_dir
        / "observed_transition_probabilities.csv",
    )

    generated_pattern.to_csv(
        output_dir
        / "generated_15min_phase_pattern.csv",
        index=False,
    )

    replay.to_csv(
        output_dir
        / "azure_trace_derived_15min_replay.csv",
        index=False,
    )


def print_summary(
    selected_app: str,
    generated_pattern: pd.DataFrame,
    replay: pd.DataFrame,
) -> None:
    print("\nSelected application")
    print("--------------------")
    print(selected_app)

    print("\nGenerated phase sequence")
    print("------------------------")

    for _, block in (
        generated_pattern.iterrows()
    ):
        print(
            f"{block['start_sec'] / 60.0:6.2f}–"
            f"{block['end_sec'] / 60.0:6.2f} min: "
            f"{block['phase']}"
        )

    valid_iats = (
        replay["iat_seconds"]
        .dropna()
    )

    print("\nGenerated workload summary")
    print("--------------------------")

    print(
        "Duration: "
        f"{GENERATED_DURATION_SECONDS / 60:.2f} "
        "minutes"
    )

    print(
        f"Requests: {len(replay):,}"
    )

    print(
        "Mean arrival rate: "
        f"{len(replay) / GENERATED_DURATION_SECONDS:.4f} "
        "req/s"
    )

    if not valid_iats.empty:
        print(
            "Median actual IAT: "
            f"{valid_iats.median():.6f} seconds"
        )

        print(
            "P95 actual IAT: "
            f"{np.percentile(valid_iats, 95):.6f} "
            "seconds"
        )

        print(
            "Maximum actual IAT: "
            f"{valid_iats.max():.6f} seconds"
        )

    print("\nRequests by phase")
    print("-----------------")

    print(
        replay["phase"]
        .value_counts()
        .to_string()
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

    trace_df = read_azure_trace()

    (
        selected_app,
        selected_app_df,
        ranking,
    ) = select_best_application(
        trace_df
    )

    (
        selected_pattern,
        thresholds,
    ) = create_application_pattern(
        selected_app_df
    )

    phase_sequence = (
        selected_pattern["phase"]
        .astype(str)
        .tolist()
    )

    transitions = (
        calculate_phase_transitions(
            phase_sequence
        )
    )

    phase_runs = (
        calculate_phase_runs(
            phase_sequence
        )
    )

    transition_probability_dict = (
        build_transition_probabilities(
            transitions
        )
    )

    (
        transition_count_matrix,
        transition_probability_matrix,
    ) = create_transition_tables(
        transitions,
        transition_probability_dict,
    )

    generated_pattern = (
        generate_best_phase_pattern(
            transition_probability_dict,
            phase_runs,
            selected_pattern,
        )
    )

    requests_with_phases = (
        assign_request_phases(
            selected_app_df,
            selected_pattern,
        )
    )

    iat_pools = build_phase_iat_pools(
        requests_with_phases
    )

    replay = generate_request_arrivals(
        generated_pattern,
        iat_pools,
        selected_pattern,
    )

    save_outputs(
        selected_app=selected_app,
        ranking=ranking,
        selected_pattern=selected_pattern,
        thresholds=thresholds,
        transition_counts=(
            transition_count_matrix
        ),
        transition_probabilities=(
            transition_probability_matrix
        ),
        generated_pattern=generated_pattern,
        replay=replay,
        requests_with_phases=(
            requests_with_phases
        ),
    )

    print_summary(
        selected_app,
        generated_pattern,
        replay,
    )

    plot_selected_application(
        selected_app,
        selected_pattern,
    )

    plot_transition_matrix(
        transition_probability_matrix
    )

    plot_generated_workload(
        generated_pattern,
        replay,
    )

    print("\nOutput directory:")
    print(output_dir)

    print("\nReplay file:")
    print(
        output_dir
        / "azure_trace_derived_15min_replay.csv"
    )


if __name__ == "__main__":
    main()