from pathlib import Path

import numpy as np
import pandas as pd


TRACE_FILE = Path(
    "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
)

SELECTED_APPS = {
    "bursty": (
        "6e42a4e8ea1df408e059ba37d0e04255"
        "fd8385235b310265a57091132d760a65"
    ),
    "idle_heavy": (
        "4b22cc3e1988174edd67d91fd7a82a997"
        "7c1719601065e5740d54074a6910170"
    ),
    "periodic": (
        "f274d71de386ccc77e4ca74766dbc4854"
        "61c3053059d47266463c45ec92001b3"
    ),
}

# Start with one target mean IAT.
TARGET_MEAN_IAT_MS = 500.0

# Number of requests to export for each workload.
REQUESTS_PER_WORKLOAD = 1000


def extract_iats(
    df: pd.DataFrame,
    app_id: str,
) -> np.ndarray:
    app_df = df[df["app"] == app_id].copy()

    app_df["arrival_time_seconds"] = (
        app_df["end_timestamp"]
        - app_df["duration"]
    )

    arrivals = (
        app_df["arrival_time_seconds"]
        .sort_values()
        .to_numpy(dtype=float)
    )

    if len(arrivals) < 2:
        raise ValueError(
            f"Not enough invocations for application {app_id}"
        )

    iats = np.diff(arrivals)

    return iats[
        np.isfinite(iats)
        & (iats >= 0)
    ]


def scale_iats(
    iats_seconds: np.ndarray,
    target_mean_iat_ms: float,
) -> np.ndarray:
    original_mean_seconds = float(
        np.mean(iats_seconds)
    )

    if original_mean_seconds <= 0:
        raise ValueError(
            "Original mean IAT must be positive."
        )

    target_mean_seconds = (
        target_mean_iat_ms / 1000.0
    )

    scale_factor = (
        target_mean_seconds
        / original_mean_seconds
    )

    scaled = iats_seconds * scale_factor

    # Prevent exact zero sleeps.
    return np.maximum(scaled, 0.001)


if not TRACE_FILE.exists():
    raise FileNotFoundError(
        f"Trace not found: {TRACE_FILE}"
    )

print("Loading trace...")

df = pd.read_csv(TRACE_FILE)

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
)

for workload_name, app_id in SELECTED_APPS.items():
    original_iats = extract_iats(
        df=df,
        app_id=app_id,
    )

    if len(original_iats) < REQUESTS_PER_WORKLOAD:
        print(
            f"{workload_name}: only "
            f"{len(original_iats)} IAT values available."
        )
        selected_iats = original_iats
    else:
        selected_iats = original_iats[
            :REQUESTS_PER_WORKLOAD
        ]

    scaled_iats = scale_iats(
        selected_iats,
        TARGET_MEAN_IAT_MS,
    )

    output_df = pd.DataFrame(
        {
            "request_id": np.arange(
                1,
                len(selected_iats) + 1,
            ),
            "original_iat_seconds": selected_iats,
            "scaled_iat_ms": scaled_iats * 1000.0,
        }
    )

    output_file = Path(
        f"azure_{workload_name}_"
        f"{int(TARGET_MEAN_IAT_MS)}ms.csv"
    )

    output_df.to_csv(
        output_file,
        index=False,
    )

    print()
    print(f"Workload: {workload_name}")
    print(f"Application: {app_id}")
    print(
        f"Original mean IAT: "
        f"{selected_iats.mean():.3f} seconds"
    )
    print(
        f"Scaled mean IAT: "
        f"{scaled_iats.mean() * 1000:.3f} ms"
    )
    print(f"Saved: {output_file}")