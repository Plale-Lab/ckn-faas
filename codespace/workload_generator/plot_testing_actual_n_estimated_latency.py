import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/tem_est3.csv"

NETWORK_DELAY_S = 0.100

OUTPUT_PNG = "estimated_vs_actual_latency.png"


# ============================================================
# Helpers
# ============================================================

def parse_list(value):
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

    except Exception:
        pass

    return []


def safe_float(value, default=np.nan):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return default


def normalize_policy(policy):
    policy = str(policy).strip().lower()

    if policy in {
        "sequential",
        "classic",
        "seq",
    }:
        return "sequential"

    if policy in {
        "spec_b",
        "cloud_speculation",
        "cloud speculation",
        "cloud_spec",
    }:
        return "spec_b"

    if policy in {
        "po2",
        "edge_speculation",
        "edge speculation",
        "edge_spec",
    }:
        return "po2"

    return policy


def get_theta(row):
    for column in [
        "selected_theta",
        "static_theta",
        "threshold",
    ]:

        if column not in row.index:
            continue

        value = safe_float(row[column])

        if np.isfinite(value) and value >= 0:
            return value

    return np.nan


def get_model_wait(row, model):
    column = f"{model}_wait"

    if column not in row.index:
        return np.nan

    value = safe_float(row[column])

    # -1 means unavailable in your log
    if not np.isfinite(value) or value < 0:
        return np.nan

    return value


def get_model_probability(row, model):
    column = f"{model}_prob"

    if column not in row.index:
        return np.nan

    value = safe_float(row[column])

    # Your unavailable probability is -1
    if (
        not np.isfinite(value)
        or value < 0
        or value > 1
    ):
        return np.nan

    return value


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"Loaded requests: {len(df)}")


# ============================================================
# Normalize useful columns
# ============================================================

df["policy_norm"] = df[
    "effective_cascade_mode"
].apply(normalize_policy)

df["models_parsed"] = df[
    "selected_models"
].apply(parse_list)

df["theta_value"] = df.apply(
    get_theta,
    axis=1,
)


# ============================================================
# Build configuration key
#
# Example:
#
# spec_b | resnet34,convnext_tiny | 0.55
# ============================================================

def make_config_key(row):

    models = row["models_parsed"]
    theta = row["theta_value"]

    if not models or not np.isfinite(theta):
        return None

    return (
        row["policy_norm"],
        tuple(models),
        round(float(theta), 6),
    )


df["config_key"] = df.apply(
    make_config_key,
    axis=1,
)


# ============================================================
# Calculate h(a, theta) for each configuration
#
# h = probability that edge confidence < theta
# ============================================================

config_h = {}


for config_key, group in df.groupby(
    "config_key",
    dropna=True,
):

    if config_key is None:
        continue

    policy, models, theta = config_key

    models = list(models)

    if len(models) == 0:
        continue

    # --------------------------------------------------------
    # Sequential / Cloud speculation
    #
    # First model = edge model a
    # --------------------------------------------------------

    if policy in {
        "sequential",
        "spec_b",
    }:

        edge_model = models[0]

        prob_column = (
            f"{edge_model}_prob"
        )

        if prob_column not in group.columns:
            continue

        probabilities = pd.to_numeric(
            group[prob_column],
            errors="coerce",
        )

        # Remove unavailable values such as -1
        probabilities = probabilities[
            (probabilities >= 0.0)
            &
            (probabilities <= 1.0)
        ]

        if len(probabilities) == 0:

            print(
                f"No valid probabilities for "
                f"{config_key}"
            )

            continue

        # ----------------------------------------------------
        # h(a,theta) = P(confidence < theta)
        # ----------------------------------------------------

        h = float(
            (
                probabilities
                <
                float(theta)
            ).mean()
        )

        config_h[
            config_key
        ] = h

        print(
            f"[H] "
            f"policy={policy} "
            f"models={models} "
            f"theta={theta:.3f} "
            f"samples={len(probabilities)} "
            f"h={h:.4f}"
        )

    # --------------------------------------------------------
    # Edge speculation
    #
    # models:
    #   a1, a2, b
    #
    # For the parallel edge stage, the request continues if
    # neither edge result satisfies the threshold.
    # --------------------------------------------------------

    elif policy == "po2":

        if len(models) < 3:
            continue

        a1 = models[0]
        a2 = models[1]

        p1_column = f"{a1}_prob"
        p2_column = f"{a2}_prob"

        if (
            p1_column not in group.columns
            or
            p2_column not in group.columns
        ):
            continue

        p1 = pd.to_numeric(
            group[p1_column],
            errors="coerce",
        )

        p2 = pd.to_numeric(
            group[p2_column],
            errors="coerce",
        )

        valid_mask = (
            (p1 >= 0.0)
            & (p1 <= 1.0)
            & (p2 >= 0.0)
            & (p2 <= 1.0)
        )

        p1 = p1[valid_mask]
        p2 = p2[valid_mask]

        if len(p1) == 0:

            print(
                f"No valid edge probabilities "
                f"for {config_key}"
            )

            continue

        # ----------------------------------------------------
        # h_ES:
        # cloud required when BOTH edge models fail threshold
        # ----------------------------------------------------

        h_es = float(
            (
                (p1 < float(theta))
                &
                (p2 < float(theta))
            ).mean()
        )

        config_h[
            config_key
        ] = h_es

        print(
            f"[H_ES] "
            f"models={models} "
            f"theta={theta:.3f} "
            f"samples={len(p1)} "
            f"h_ES={h_es:.4f}"
        )


# ============================================================
# Display calculated h values
# ============================================================

print()
print("===================================")
print("Calculated deferral probabilities")
print("===================================")

for key, value in config_h.items():

    print(
        f"{key} -> h={value:.4f}"
    )


# ============================================================
# Sequential latency
# ============================================================

def estimate_sequential(
    row,
    models,
    theta,
    h,
):

    if len(models) < 2:
        return np.nan

    a = models[0]
    b = models[1]

    T_a = get_model_wait(
        row,
        a,
    )

    T_b = get_model_wait(
        row,
        b,
    )

    if not (
        np.isfinite(T_a)
        and
        np.isfinite(T_b)
    ):
        return np.nan

    # --------------------------------------------------------
    # TSeq(a,b,theta)
    #
    # = T(a) + h(a,theta)(T(b)+d)
    # --------------------------------------------------------

    return (
        T_a
        +
        h
        *
        (
            T_b
            +
            NETWORK_DELAY_S
        )
    )


# ============================================================
# Edge speculation latency
# ============================================================

def estimate_edge_speculation(
    row,
    models,
    theta,
    h_es,
):

    if len(models) < 3:
        return np.nan

    a1 = models[0]
    a2 = models[1]
    b = models[-1]

    T_a1 = get_model_wait(
        row,
        a1,
    )

    T_a2 = get_model_wait(
        row,
        a2,
    )

    T_b = get_model_wait(
        row,
        b,
    )

    if not all(
        np.isfinite(x)
        for x in [
            T_a1,
            T_a2,
            T_b,
        ]
    ):
        return np.nan

    # --------------------------------------------------------
    # Parallel edge stage
    #
    # Currently using max() because both are executed
    # concurrently and the aggregation stage waits for them.
    # --------------------------------------------------------

    T_A = max(
        T_a1,
        T_a2,
    )

    # --------------------------------------------------------
    # TES =
    #
    # TA + hES(Tb+d)
    # --------------------------------------------------------

    return (
        T_A
        +
        h_es
        *
        (
            T_b
            +
            NETWORK_DELAY_S
        )
    )


# ============================================================
# Cloud speculation latency
# ============================================================

def estimate_cloud_speculation(
    row,
    models,
    theta,
    h,
):

    if len(models) < 2:
        return np.nan

    a = models[0]
    b = models[1]

    T_a = get_model_wait(
        row,
        a,
    )

    T_b = get_model_wait(
        row,
        b,
    )

    if not (
        np.isfinite(T_a)
        and
        np.isfinite(T_b)
    ):
        return np.nan

    cloud_time = (
        T_b
        +
        NETWORK_DELAY_S
    )

    # --------------------------------------------------------
    # Case 1
    #
    # Tb + d <= Ta
    # --------------------------------------------------------

    if cloud_time <= T_a:

        return cloud_time

    # --------------------------------------------------------
    # Case 2
    #
    # (1-h)Ta + h(Tb+d)
    # --------------------------------------------------------

    return (
        (1.0 - h)
        *
        T_a
        +
        h
        *
        cloud_time
    )


# ============================================================
# Calculate latency for one request
# ============================================================

def calculate_estimated_latency(row):

    config_key = row["config_key"]

    if config_key is None:
        return np.nan

    if config_key not in config_h:
        return np.nan

    h = config_h[
        config_key
    ]

    policy = row[
        "policy_norm"
    ]

    models = row[
        "models_parsed"
    ]

    theta = row[
        "theta_value"
    ]

    if policy == "sequential":

        return estimate_sequential(
            row,
            models,
            theta,
            h,
        )

    if policy == "po2":

        return estimate_edge_speculation(
            row,
            models,
            theta,
            h,
        )

    if policy == "spec_b":

        return estimate_cloud_speculation(
            row,
            models,
            theta,
            h,
        )

    return np.nan


# ============================================================
# Calculate estimated latency
# ============================================================

df[
    "calculated_est_latency_s"
] = df.apply(
    calculate_estimated_latency,
    axis=1,
)


# ============================================================
# Actual latency = RespTime
# ============================================================

df[
    "actual_latency_s"
] = pd.to_numeric(
    df["RespTime"],
    errors="coerce",
)


# ============================================================
# Convert seconds -> milliseconds
# ============================================================

df[
    "estimated_latency_ms"
] = (
    df[
        "calculated_est_latency_s"
    ]
    *
    1000.0
)

df[
    "actual_latency_ms"
] = (
    df[
        "actual_latency_s"
    ]
    *
    1000.0
)


# ============================================================
# Valid comparisons
# ============================================================

valid = df[
    np.isfinite(
        df["estimated_latency_ms"]
    )
    &
    np.isfinite(
        df["actual_latency_ms"]
    )
].copy()


print()
print(
    f"Valid estimator comparisons: "
    f"{len(valid)} / {len(df)}"
)


# ============================================================
# Diagnostics if still zero
# ============================================================

if len(valid) == 0:

    print()
    print("No valid rows.")
    print()

    print(
        "Policies:",
        df[
            "policy_norm"
        ].value_counts(
            dropna=False
        )
    )

    print()

    print(
        "Configurations with calculated h:",
        len(config_h),
    )

    print()

    print(
        df[
            [
                "ID",
                "effective_cascade_mode",
                "selected_models",
                "selected_theta",
                "config_key",
            ]
        ].head(10)
    )

    raise RuntimeError(
        "No valid estimator comparisons."
    )


# ============================================================
# Errors
# ============================================================

valid[
    "absolute_error_ms"
] = np.abs(
    valid[
        "estimated_latency_ms"
    ]
    -
    valid[
        "actual_latency_ms"
    ]
)


mae = valid[
    "absolute_error_ms"
].mean()


print(
    f"MAE: {mae:.3f} ms"
)


# ============================================================
# Plot
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 4.5)
)


x = np.arange(
    len(valid)
)


ax.plot(
    x,
    valid[
        "actual_latency_ms"
    ],
    label="Actual Latency",
    linewidth=1.5,
)


ax.plot(
    x,
    valid[
        "estimated_latency_ms"
    ],
    label="Estimated Latency",
    linewidth=1.5,
    linestyle="--",
)


ax.set_xlabel(
    "Request"
)

ax.set_ylabel(
    "Latency (ms)"
)

ax.set_title(
    "Estimated vs. Actual Request Latency"
)

ax.legend()

ax.grid(
    True,
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    OUTPUT_PNG,
    dpi=300,
    bbox_inches="tight",
)


plt.show()