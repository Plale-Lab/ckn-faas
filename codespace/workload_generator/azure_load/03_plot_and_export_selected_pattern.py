from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# Change only this section
# ============================================================

INPUT_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/azure_load/"
    "azure_15min_selection"
)

OUTPUT_DIR = INPUT_DIR

SELECTED_REQUESTS_FILE = (
    "selected_15min_requests.csv"
)

SELECTED_PATTERN_FILE = (
    "selected_15min_pattern.csv"
)

# Deadline attached to each workload request.
DEFAULT_DEADLINE_MS = 2000

# Optional image-list file used by your generator.
#
# Leave as None when this script only needs to save timing.
IMAGE_LIST_FILE = None

# ============================================================


PHASE_NUMBER = {
    "Idle": 0,
    "Low": 1,
    "Medium": 2,
    "High": 3,
}


def load_selected_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    input_dir = Path(INPUT_DIR)

    requests_path = (
        input_dir / SELECTED_REQUESTS_FILE
    )

    pattern_path = (
        input_dir / SELECTED_PATTERN_FILE
    )

    if not requests_path.exists():
        raise FileNotFoundError(
            f"Selected requests file was not found:\n"
            f"{requests_path}\n\n"
            "Run 02_select_best_15min_pattern.py first."
        )

    if not pattern_path.exists():
        raise FileNotFoundError(
            f"Selected pattern file was not found:\n"
            f"{pattern_path}\n\n"
            "Run 02_select_best_15min_pattern.py first."
        )

    requests = pd.read_csv(
        requests_path
    )

    pattern = pd.read_csv(
        pattern_path
    )

    required_request_columns = [
        "request_index",
        "relative_arrival_sec",
        "iat_seconds",
    ]

    for column in required_request_columns:
        if column not in requests.columns:
            raise ValueError(
                f"Requests CSV is missing: {column}"
            )

    required_pattern_columns = [
        "bin_start_sec",
        "arrival_rate_rps",
        "phase",
    ]

    for column in required_pattern_columns:
        if column not in pattern.columns:
            raise ValueError(
                f"Pattern CSV is missing: {column}"
            )

    return requests, pattern


def load_image_list() -> list[str]:
    if IMAGE_LIST_FILE is None:
        return []

    image_path = Path(IMAGE_LIST_FILE)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image-list file was not found:\n"
            f"{image_path}"
        )

    with image_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        images = [
            line.strip()
            for line in file
            if line.strip()
        ]

    return images


def create_replay_csv(
    requests: pd.DataFrame,
) -> pd.DataFrame:
    replay = pd.DataFrame()

    replay["request_index"] = (
        requests["request_index"]
        .astype(int)
    )

    replay["iat_seconds"] = pd.to_numeric(
        requests["iat_seconds"],
        errors="coerce",
    )

    replay["scheduled_arrival_sec"] = (
        pd.to_numeric(
            requests["relative_arrival_sec"],
            errors="coerce",
        )
    )

    replay["deadline_ms"] = (
        DEFAULT_DEADLINE_MS
    )

    if "app" in requests.columns:
        replay["azure_app"] = (
            requests["app"]
        )

    if "func" in requests.columns:
        replay["azure_func"] = (
            requests["func"]
        )

    images = load_image_list()

    if images:
        replay["image_path"] = [
            images[index % len(images)]
            for index in range(len(replay))
        ]

    replay = replay.dropna(
        subset=[
            "iat_seconds",
            "scheduled_arrival_sec",
        ]
    ).copy()

    replay["iat_seconds"] = (
        replay["iat_seconds"]
        .clip(lower=0)
    )

    return replay


def plot_selected_pattern(
    requests: pd.DataFrame,
    pattern: pd.DataFrame,
) -> None:
    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    pattern = pattern.copy()

    pattern["time_minute"] = (
        pattern["bin_start_sec"] / 60.0
    )

    if "phase_number" not in pattern.columns:
        pattern["phase_number"] = (
            pattern["phase"].map(
                PHASE_NUMBER
            )
        )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(11, 7),
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
    # Panel 1: Arrival-rate pattern
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
        "Selected 15-minute Azure workload pattern"
    )

    rate_ax.set_ylabel(
        "Arrival rate\n(req/s)"
    )

    rate_ax.grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Panel 2: Classified phases
    # --------------------------------------------------------

    phase_ax.step(
        pattern["time_minute"],
        pattern["phase_number"],
        where="post",
        linewidth=1.8,
    )

    phase_ax.set_yticks(
        list(PHASE_NUMBER.values())
    )

    phase_ax.set_yticklabels(
        list(PHASE_NUMBER.keys())
    )

    phase_ax.set_ylabel(
        "Workload\nphase"
    )

    phase_ax.grid(
        True,
        axis="x",
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Panel 3: Exact Azure IATs
    # --------------------------------------------------------

    iat_ax.plot(
        requests["relative_arrival_sec"] / 60.0,
        requests["iat_seconds"],
        linewidth=0.9,
    )

    iat_ax.set_xlabel(
        "Time within workload (minutes)"
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
        / "selected_15min_workload.png"
    )

    pdf_path = (
        output_dir
        / "selected_15min_workload.pdf"
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


def print_summary(
    requests: pd.DataFrame,
    pattern: pd.DataFrame,
) -> None:
    duration_sec = (
        requests["relative_arrival_sec"].max()
    )

    mean_rate = (
        len(requests) / 900.0
    )

    print("\nSelected 15-minute workload summary")
    print("-----------------------------------")
    print(f"Requests: {len(requests)}")

    print(
        f"Last request arrival: "
        f"{duration_sec:.2f} seconds"
    )

    print(
        f"Mean arrival rate: "
        f"{mean_rate:.4f} req/s"
    )

    print(
        f"Peak binned arrival rate: "
        f"{pattern['arrival_rate_rps'].max():.4f} req/s"
    )

    print(
        f"Median IAT: "
        f"{requests['iat_seconds'].median():.6f} seconds"
    )

    print(
        f"P95 IAT: "
        f"{np.percentile(requests['iat_seconds'], 95):.6f} seconds"
    )

    print(
        f"Maximum IAT: "
        f"{requests['iat_seconds'].max():.6f} seconds"
    )

    print("\nPhase composition:")

    phase_counts = (
        pattern["phase"]
        .value_counts()
    )

    for phase in [
        "Idle",
        "Low",
        "Medium",
        "High",
    ]:
        count = int(
            phase_counts.get(
                phase,
                0,
            )
        )

        percentage = (
            count / len(pattern) * 100.0
        )

        print(
            f"{phase}: "
            f"{count} bins "
            f"({percentage:.2f}%)"
        )


def main() -> None:
    requests, pattern = (
        load_selected_data()
    )

    replay = create_replay_csv(
        requests
    )

    output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    replay_path = (
        output_dir
        / "azure_15min_replay.csv"
    )

    replay.to_csv(
        replay_path,
        index=False,
    )

    print_summary(
        requests,
        pattern,
    )

    plot_selected_pattern(
        requests,
        pattern,
    )

    print("\nWorkload-generator replay CSV:")
    print(replay_path)

    print("\nReplay CSV columns:")
    print(list(replay.columns))


if __name__ == "__main__":
    main()