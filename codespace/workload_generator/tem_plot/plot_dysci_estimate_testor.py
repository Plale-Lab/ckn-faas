import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/"
    "clone main/ckn-faas/codespace/"
    "workload_generator/data/tem_delete7.csv"
)

MODEL = "resnet34"

# Use 1 for raw per-request error.
# Use 5 only if you want a smoother visualization.
ROLLING_WINDOW = 1


# ============================================================
# Load CSV
# ============================================================

df = pd.read_csv(CSV_PATH)

actual_col = f"{MODEL}_actual"
dysci_col = f"{MODEL}_dysci_est"
iluvatar_col = f"{MODEL}_iluvatar_est"


# ============================================================
# Check required columns
# ============================================================

required_columns = [
    actual_col,
    dysci_col,
    iluvatar_col,
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in CSV."
        )


# ============================================================
# Convert to numeric
# ============================================================

for col in required_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce",
    )


# ============================================================
# Keep only valid completed requests
#
# We require all three values so DYSCI and Iluvatar are
# evaluated on exactly the same requests.
# ============================================================

plot_df = df[
    (df[actual_col] >= 0)
    & (df[dysci_col] >= 0)
    & (df[iluvatar_col] >= 0)
].copy()

plot_df = plot_df.reset_index(drop=True)

if plot_df.empty:
    raise ValueError(
        f"No valid completed requests found for {MODEL}."
    )

plot_df["Request"] = np.arange(
    1,
    len(plot_df) + 1,
)


# ============================================================
# Convert seconds -> milliseconds
# ============================================================

plot_df["Actual_ms"] = (
    plot_df[actual_col] * 1000.0
)

plot_df["DYSCI_Estimated_ms"] = (
    plot_df[dysci_col] * 1000.0
)

plot_df["Iluvatar_Estimated_ms"] = (
    plot_df[iluvatar_col] * 1000.0
)


# ============================================================
# Calculate signed errors
#
# Positive:
#     estimator predicted higher than actual
#
# Negative:
#     estimator predicted lower than actual
# ============================================================

plot_df["DYSCI_Error_ms"] = (
    plot_df["DYSCI_Estimated_ms"]
    - plot_df["Actual_ms"]
)

plot_df["Iluvatar_Error_ms"] = (
    plot_df["Iluvatar_Estimated_ms"]
    - plot_df["Actual_ms"]
)


# ============================================================
# Absolute error
# ============================================================

plot_df["DYSCI_Absolute_Error_ms"] = (
    plot_df["DYSCI_Error_ms"].abs()
)

plot_df["Iluvatar_Absolute_Error_ms"] = (
    plot_df["Iluvatar_Error_ms"].abs()
)


# ============================================================
# Accuracy metrics
# ============================================================

dysci_mae = (
    plot_df["DYSCI_Absolute_Error_ms"].mean()
)

iluvatar_mae = (
    plot_df["Iluvatar_Absolute_Error_ms"].mean()
)


dysci_rmse = np.sqrt(
    np.mean(
        plot_df["DYSCI_Error_ms"] ** 2
    )
)

iluvatar_rmse = np.sqrt(
    np.mean(
        plot_df["Iluvatar_Error_ms"] ** 2
    )
)


# ============================================================
# MAPE
#
# Protect against zero actual latency.
# ============================================================

nonzero_actual = (
    plot_df["Actual_ms"] > 0
)

dysci_mape = (
    (
        plot_df.loc[
            nonzero_actual,
            "DYSCI_Absolute_Error_ms",
        ]
        /
        plot_df.loc[
            nonzero_actual,
            "Actual_ms",
        ]
    ).mean()
    * 100.0
)

iluvatar_mape = (
    (
        plot_df.loc[
            nonzero_actual,
            "Iluvatar_Absolute_Error_ms",
        ]
        /
        plot_df.loc[
            nonzero_actual,
            "Actual_ms",
        ]
    ).mean()
    * 100.0
)


# ============================================================
# Print metrics
# ============================================================

print()
print("==============================================")
print(f"Latency Estimator Evaluation — {MODEL}")
print("==============================================")

print(
    f"Number of requests : {len(plot_df)}"
)

print()

print("DYSCI")
print(
    f"  MAE  : {dysci_mae:.2f} ms"
)
print(
    f"  RMSE : {dysci_rmse:.2f} ms"
)
print(
    f"  MAPE : {dysci_mape:.2f}%"
)

print()

print("Ilúvatar")
print(
    f"  MAE  : {iluvatar_mae:.2f} ms"
)
print(
    f"  RMSE : {iluvatar_rmse:.2f} ms"
)
print(
    f"  MAPE : {iluvatar_mape:.2f}%"
)

print("==============================================")
print()


# ============================================================
# Prepare plot values
# ============================================================

if ROLLING_WINDOW > 1:

    plot_df["DYSCI_Error_Plot"] = (
        plot_df["DYSCI_Absolute_Error_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    plot_df["Iluvatar_Error_Plot"] = (
        plot_df["Iluvatar_Absolute_Error_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

else:

    plot_df["DYSCI_Error_Plot"] = (
        plot_df["DYSCI_Absolute_Error_ms"]
    )

    plot_df["Iluvatar_Error_Plot"] = (
        plot_df["Iluvatar_Absolute_Error_ms"]
    )


# ============================================================
# Plot absolute estimation error
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    plot_df["Request"],
    plot_df["DYSCI_Error_Plot"],
    label="DYSCI",
    linewidth=2.0,
)

plt.plot(
    plot_df["Request"],
    plot_df["Iluvatar_Error_Plot"],
    label="Ilúvatar",
    linewidth=2.0,
    linestyle="--",
)


# ============================================================
# Labels
# ============================================================

plt.xlabel(
    "Request",
    fontsize=12,
)

plt.ylabel(
    "Absolute Estimation Error (ms)",
    fontsize=12,
)

plt.title(
    f"Latency Estimation Error — {MODEL}",
    fontsize=14,
)

plt.grid(
    True,
    alpha=0.3,
)

plt.legend(
    fontsize=11,
)

plt.tight_layout()


# ============================================================
# Save
# ============================================================

OUTPUT_FILE = (
    f"{MODEL}_latency_estimation_error.png"
)

plt.savefig(
    OUTPUT_FILE,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(
    f"Figure saved to: {OUTPUT_FILE}"
)