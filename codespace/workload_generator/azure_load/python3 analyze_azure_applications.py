from __future__ import annotations

from collections import Counter
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

AZURE_TRACE_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

OUTPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "top_azure_application_candidates"
)

TIMESTAMP_UNIT = "seconds"
DURATION_UNIT = "seconds"

# Short bins reveal real bursts.
RATE_BIN_SECONDS = 5.0

MIN_APP_REQUESTS = 500
MIN_ACTIVE_BINS = 20

TOP_K = 10

# Phase thresholds based on each application's positive-rate bins.
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


# ============================================================
# TRACE READING
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


def read_azure_trace() -> pd.DataFrame:
    trace_path = Path(AZURE_TRACE_FILE)

    if not trace_path.exists():
        raise FileNotFoundError(
            f"Azure trace not found:\n{trace_path}"
        )

    print("Reading Azure trace...")
    print(trace_path)

    df = pd.read_csv(trace_path)

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
            f"Missing columns: {missing}\n"
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
        ["app", "start_time_sec"]
    ).reset_index(drop=True)

    print(f"Valid requests: {len(df):,}")
    print(f"Applications: {df['app'].nunique():,}")
    print(f"Functions: {df['func'].nunique():,}")

    return df


# ============================================================
# RATE BINNING
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
        return phases, {
            "low_upper_rps": np.nan,
            "medium_upper_rps": np.nan,
            "high_upper_rps": np.nan,
        }

    low_upper = float(
        np.percentile(
            positive_rates,
            LOW_PERCENTILE,
        )
    )

    medium_upper = float(
        np.percentile(
            positive_rates,
            MEDIUM_PERCENTILE,
        )
    )

    high_upper = float(
        np.percentile(
            positive_rates,
            HIGH_PERCENTILE,
        )
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

    return phases, {
        "low_upper_rps": low_upper,
        "medium_upper_rps": medium_upper,
        "high_upper_rps": high_upper,
    }


def create_rate_bins(
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

    trace_duration = max(
        last_time - first_time,
        RATE_BIN_SECONDS,
    )

    edges = np.arange(
        0.0,
        trace_duration + RATE_BIN_SECONDS,
        RATE_BIN_SECONDS,
    )

    if len(edges) < 2:
        edges = np.array(
            [0.0, RATE_BIN_SECONDS]
        )

    request_counts, _ = np.histogram(
        app_df["relative_time_sec"],
        bins=edges,
    )

    arrival_rates = (
        request_counts
        / RATE_BIN_SECONDS
    )

    phases, thresholds = classify_rates(
        arrival_rates
    )

    bins = pd.DataFrame(
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

    bins["time_hours"] = (
        bins["bin_start_sec"]
        / 3600.0
    )

    return bins, thresholds


# ============================================================
# TRANSITIONS AND DIVERSITY
# ============================================================

def collapse_phases(
    phases: list[str],
) -> list[str]:
    if not phases:
        return []

    collapsed = [phases[0]]

    for phase in phases[1:]:
        if phase != collapsed[-1]:
            collapsed.append(phase)

    return collapsed


def count_transitions(
    phases: list[str],
) -> Counter:
    collapsed = collapse_phases(
        phases
    )

    transitions = Counter()

    for current_phase, next_phase in zip(
        collapsed[:-1],
        collapsed[1:],
    ):
        transitions[
            (current_phase, next_phase)
        ] += 1

    return transitions


def normalized_entropy(
    values: list[int],
) -> float:
    array = np.asarray(
        values,
        dtype=float,
    )

    total = float(
        array.sum()
    )

    if total <= 0:
        return 0.0

    probabilities = (
        array[array > 0]
        / total
    )

    entropy = float(
        -np.sum(
            probabilities
            * np.log(probabilities)
        )
    )

    maximum_entropy = math.log(
        len(array)
    )

    if maximum_entropy <= 0:
        return 0.0

    return entropy / maximum_entropy


# ============================================================
# APPLICATION METRICS
# ============================================================

def analyze_application(
    app_name: str,
    app_df: pd.DataFrame,
) -> dict | None:
    if len(app_df) < MIN_APP_REQUESTS:
        return None

    rate_bins, thresholds = create_rate_bins(
        app_df
    )

    phase_counts = {
        phase: int(
            (
                rate_bins["phase"]
                == phase
            ).sum()
        )
        for phase in PHASES
    }

    active_bins = (
        len(rate_bins)
        - phase_counts["Idle"]
    )

    if active_bins < MIN_ACTIVE_BINS:
        return None

    positive_rates = rate_bins.loc[
        rate_bins["arrival_rate_rps"] > 0,
        "arrival_rate_rps",
    ]

    if positive_rates.empty:
        return None

    start_min = float(
        app_df["start_time_sec"].min()
    )

    start_max = float(
        app_df["start_time_sec"].max()
    )

    duration_hours = (
        start_max - start_min
    ) / 3600.0

    iats = (
        app_df.sort_values(
            "start_time_sec"
        )["start_time_sec"]
        .diff()
        .dropna()
    )

    iats = iats[
        np.isfinite(iats)
        & (iats >= 0)
    ]

    phase_entropy = normalized_entropy(
        [
            phase_counts[phase]
            for phase in PHASES
        ]
    )

    transitions = count_transitions(
        rate_bins["phase"]
        .astype(str)
        .tolist()
    )

    idle_fraction = (
        phase_counts["Idle"]
        / len(rate_bins)
    )

    return {
        "app": str(app_name),
        "request_count": int(len(app_df)),
        "duration_hours": duration_hours,
        "total_bins": int(len(rate_bins)),
        "active_bins": int(active_bins),
        "idle_bins": phase_counts["Idle"],
        "low_bins": phase_counts["Low"],
        "medium_bins": phase_counts["Medium"],
        "high_bins": phase_counts["High"],
        "spike_bins": phase_counts["Spike"],
        "idle_fraction": idle_fraction,
        "phase_entropy": phase_entropy,
        "unique_transition_count": len(transitions),
        "total_transition_count": int(
            sum(transitions.values())
        ),
        "mean_rate_all_bins_rps": float(
            rate_bins["arrival_rate_rps"].mean()
        ),
        "mean_active_rate_rps": float(
            positive_rates.mean()
        ),
        "median_active_rate_rps": float(
            positive_rates.median()
        ),
        "p90_active_rate_rps": float(
            np.percentile(
                positive_rates,
                90,
            )
        ),
        "p95_active_rate_rps": float(
            np.percentile(
                positive_rates,
                95,
            )
        ),
        "p99_active_rate_rps": float(
            np.percentile(
                positive_rates,
                99,
            )
        ),
        "maximum_rate_rps": float(
            positive_rates.max()
        ),
        "median_iat_seconds": (
            float(iats.median())
            if not iats.empty
            else np.nan
        ),
        "p05_iat_seconds": (
            float(
                np.percentile(
                    iats,
                    5,
                )
            )
            if not iats.empty
            else np.nan
        ),
        "p95_iat_seconds": (
            float(
                np.percentile(
                    iats,
                    95,
                )
            )
            if not iats.empty
            else np.nan
        ),
        **thresholds,
    }


# ============================================================
# RANKING
# ============================================================

def percentile_rank(
    series: pd.Series,
) -> pd.Series:
    if series.nunique() <= 1:
        return pd.Series(
            0.5,
            index=series.index,
        )

    return series.rank(
        pct=True,
        method="average",
    )


def analyze_all_applications(
    trace_df: pd.DataFrame,
) -> pd.DataFrame:
    results = []

    app_groups = list(
        trace_df.groupby("app")
    )

    print("\nAnalyzing all applications...")

    for index, (
        app_name,
        app_df,
    ) in enumerate(
        app_groups,
        start=1,
    ):
        print(
            f"[{index}/{len(app_groups)}] "
            f"Application {app_name}"
        )

        result = analyze_application(
            str(app_name),
            app_df.copy(),
        )

        if result is not None:
            results.append(result)

    if not results:
        raise RuntimeError(
            "No applications passed the minimum requirements."
        )

    summary = pd.DataFrame(
        results
    )

    # Rank rate magnitude.
    summary["p95_rate_rank"] = percentile_rank(
        summary["p95_active_rate_rps"]
    )

    summary["p99_rate_rank"] = percentile_rank(
        summary["p99_active_rate_rps"]
    )

    summary["maximum_rate_rank"] = percentile_rank(
        summary["maximum_rate_rps"]
    )

    # Rank duration.
    summary["duration_rank"] = percentile_rank(
        summary["duration_hours"]
    )

    # Rank transitions and volume.
    summary["transition_rank"] = percentile_rank(
        summary["unique_transition_count"]
    )

    summary["request_count_rank"] = percentile_rank(
        summary["request_count"]
    )

    # Prefer some idle time, but not an application that is
    # almost completely idle.
    summary["idle_suitability"] = (
        1.0
        - np.abs(
            summary["idle_fraction"]
            - 0.20
        )
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # Combined ranking:
    # 45% real workload intensity
    # 20% duration
    # 15% phase diversity
    # 10% transition diversity
    # 5% request count
    # 5% useful idle behavior
    summary["selection_score"] = (
        0.20 * summary["p95_rate_rank"]
        + 0.15 * summary["p99_rate_rank"]
        + 0.10 * summary["maximum_rate_rank"]
        + 0.20 * summary["duration_rank"]
        + 0.15 * summary["phase_entropy"]
        + 0.10 * summary["transition_rank"]
        + 0.05 * summary["request_count_rank"]
        + 0.05 * summary["idle_suitability"]
    )

    summary = summary.sort_values(
        [
            "selection_score",
            "p95_active_rate_rps",
            "duration_hours",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)

    summary["rank"] = (
        np.arange(len(summary))
        + 1
    )

    return summary


# ============================================================
# PHASE STATISTICS
# ============================================================

def calculate_phase_statistics(
    rate_bins: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for phase in PHASES:
        phase_bins = rate_bins[
            rate_bins["phase"] == phase
        ]

        rates = phase_bins[
            "arrival_rate_rps"
        ]

        if rates.empty:
            rows.append(
                {
                    "phase": phase,
                    "bin_count": 0,
                    "request_count": 0,
                    "mean_rate_rps": np.nan,
                    "median_rate_rps": np.nan,
                    "p95_rate_rps": np.nan,
                    "maximum_rate_rps": np.nan,
                }
            )

            continue

        rows.append(
            {
                "phase": phase,
                "bin_count": int(
                    len(phase_bins)
                ),
                "request_count": int(
                    phase_bins[
                        "request_count"
                    ].sum()
                ),
                "mean_rate_rps": float(
                    rates.mean()
                ),
                "median_rate_rps": float(
                    rates.median()
                ),
                "p95_rate_rps": float(
                    np.percentile(
                        rates,
                        95,
                    )
                ),
                "maximum_rate_rps": float(
                    rates.max()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOTTING
# ============================================================

def safe_filename(
    value: str,
) -> str:
    return "".join(
        character
        if character.isalnum()
        or character in "-_"
        else "_"
        for character in value
    )


def plot_candidate_workload(
    rank: int,
    app_name: str,
    rate_bins: pd.DataFrame,
    metrics: pd.Series,
) -> None:
    output_dir = (
        Path(OUTPUT_DIR)
        / "candidate_plots"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, ax = plt.subplots(
        figsize=(13, 5)
    )

    ax.plot(
        rate_bins["time_hours"],
        rate_bins["arrival_rate_rps"],
        linewidth=0.8,
    )

    title = (
        f"Rank {rank}: Azure application {app_name}\n"
        f"Duration={metrics['duration_hours']:.2f} h, "
        f"P95={metrics['p95_active_rate_rps']:.2f} req/s, "
        f"P99={metrics['p99_active_rate_rps']:.2f} req/s, "
        f"Max={metrics['maximum_rate_rps']:.2f} req/s"
    )

    ax.set_title(
        title
    )

    ax.set_xlabel(
        "Time from beginning of application trace (hours)"
    )

    ax.set_ylabel(
        "Arrival rate (requests/second)"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    filename = (
        f"rank_{rank:02d}_"
        f"{safe_filename(app_name)}"
    )

    fig.savefig(
        output_dir
        / f"{filename}.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / f"{filename}.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_top_candidate_comparison(
    top_candidates: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    plot_df = (
        top_candidates
        .sort_values(
            "selection_score",
            ascending=True,
        )
        .copy()
    )

    labels = [
        f"Rank {int(row['rank'])}: {row['app']}"
        for _, row in plot_df.iterrows()
    ]

    positions = np.arange(
        len(plot_df)
    )

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    ax.barh(
        positions,
        plot_df[
            "selection_score"
        ],
    )

    ax.set_yticks(
        positions
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "Selection score"
    )

    ax.set_ylabel(
        "Azure application"
    )

    ax.set_title(
        "Top Azure application candidates"
    )

    ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "top_candidate_selection_scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "top_candidate_selection_scores.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


def plot_rate_duration_comparison(
    summary: pd.DataFrame,
    top_candidates: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.scatter(
        summary["duration_hours"],
        summary["p95_active_rate_rps"],
        s=30,
        alpha=0.5,
    )

    for _, row in top_candidates.iterrows():
        ax.annotate(
            f"R{int(row['rank'])}",
            (
                row["duration_hours"],
                row["p95_active_rate_rps"],
            ),
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.set_xlabel(
        "Application trace duration (hours)"
    )

    ax.set_ylabel(
        "P95 active arrival rate (requests/second)"
    )

    ax.set_title(
        "Azure application duration versus P95 arrival rate"
    )

    ax.grid(
        True,
        alpha=0.3,
    )

    fig.tight_layout()

    fig.savefig(
        output_dir
        / "application_duration_vs_p95_rate.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        output_dir
        / "application_duration_vs_p95_rate.pdf",
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)


# ============================================================
# PROCESS TOP CANDIDATES
# ============================================================

def process_top_candidates(
    trace_df: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    phase_output_dir = (
        output_dir
        / "candidate_phase_statistics"
    )

    bin_output_dir = (
        output_dir
        / "candidate_rate_bins"
    )

    phase_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    bin_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    top_candidates = summary.head(
        TOP_K
    ).copy()

    for _, metrics in top_candidates.iterrows():
        rank = int(
            metrics["rank"]
        )

        app_name = str(
            metrics["app"]
        )

        app_df = trace_df[
            trace_df["app"].astype(str)
            == app_name
        ].copy()

        rate_bins, thresholds = (
            create_rate_bins(
                app_df
            )
        )

        phase_statistics = (
            calculate_phase_statistics(
                rate_bins
            )
        )

        rate_bins.to_csv(
            bin_output_dir
            / (
                f"rank_{rank:02d}_"
                f"{safe_filename(app_name)}"
                "_rate_bins.csv"
            ),
            index=False,
        )

        phase_statistics.to_csv(
            phase_output_dir
            / (
                f"rank_{rank:02d}_"
                f"{safe_filename(app_name)}"
                "_phase_statistics.csv"
            ),
            index=False,
        )

        threshold_df = pd.DataFrame(
            [
                {
                    "rank": rank,
                    "app": app_name,
                    "rate_bin_seconds": (
                        RATE_BIN_SECONDS
                    ),
                    **thresholds,
                }
            ]
        )

        threshold_df.to_csv(
            phase_output_dir
            / (
                f"rank_{rank:02d}_"
                f"{safe_filename(app_name)}"
                "_thresholds.csv"
            ),
            index=False,
        )

        plot_candidate_workload(
            rank=rank,
            app_name=app_name,
            rate_bins=rate_bins,
            metrics=metrics,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    output_dir = Path(
        OUTPUT_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    trace_df = read_azure_trace()

    summary = analyze_all_applications(
        trace_df
    )

    summary.to_csv(
        output_dir
        / "all_azure_application_ranking.csv",
        index=False,
    )

    top_candidates = summary.head(
        TOP_K
    ).copy()

    top_candidates.to_csv(
        output_dir
        / "top_10_azure_application_candidates.csv",
        index=False,
    )

    print("\nTop Azure application candidates")
    print("--------------------------------")

    print(
        top_candidates[
            [
                "rank",
                "app",
                "request_count",
                "duration_hours",
                "mean_active_rate_rps",
                "p95_active_rate_rps",
                "p99_active_rate_rps",
                "maximum_rate_rps",
                "idle_fraction",
                "unique_transition_count",
                "phase_entropy",
                "selection_score",
            ]
        ].to_string(index=False)
    )

    process_top_candidates(
        trace_df,
        summary,
    )

    plot_top_candidate_comparison(
        top_candidates
    )

    plot_rate_duration_comparison(
        summary,
        top_candidates,
    )

    print("\nAll outputs saved to:")
    print(output_dir)

    print("\nImportant files:")
    print(
        output_dir
        / "all_azure_application_ranking.csv"
    )

    print(
        output_dir
        / "top_10_azure_application_candidates.csv"
    )

    print(
        output_dir
        / "candidate_plots"
    )


if __name__ == "__main__":
    main()