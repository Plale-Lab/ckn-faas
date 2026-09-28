import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = (
    "/Users/agamage/Desktop/D2I/Codes Original/"
    "clone main/ckn-faas/codespace/"
    "workload_generator/data/tem_delete18.csv"
)

# Model to analyze
MODEL = "resnet34"

# Optional rolling average.
# Set to 1 if you want the raw values without smoothing.
ROLLING_WINDOW = 5


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(CSV_PATH)

print(f"Total requests in CSV: {len(df)}")


# ============================================================
# Column names
# ============================================================

actual_col = f"{MODEL}_actual"
dysci_col = f"{MODEL}_dysci_est"
iluvatar_col = f"{MODEL}_iluvatar_est"


required_columns = [
    actual_col,
    dysci_col,
    iluvatar_col,
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' was not found in the CSV."
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
# Keep only requests where the model actually completed
# ============================================================

plot_df = df[
    (df[actual_col] >= 0)
    & (df[dysci_col] >= 0)
    & (df[iluvatar_col] >= 0)
].copy()


# Reset index so x-axis becomes:
# 0, 1, 2, 3, ...
plot_df = plot_df.reset_index(drop=True)

plot_df["Request"] = range(
    1,
    len(plot_df) + 1,
)


print(
    f"Valid completed {MODEL} requests: "
    f"{len(plot_df)}"
)


if plot_df.empty:
    raise ValueError(
        f"No valid completed requests found for {MODEL}."
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
# Optional smoothing
# ============================================================

if ROLLING_WINDOW > 1:

    plot_df["Actual_plot"] = (
        plot_df["Actual_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    plot_df["DYSCI_plot"] = (
        plot_df["DYSCI_Estimated_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

    plot_df["Iluvatar_plot"] = (
        plot_df["Iluvatar_Estimated_ms"]
        .rolling(
            window=ROLLING_WINDOW,
            min_periods=1,
        )
        .mean()
    )

else:

    plot_df["Actual_plot"] = (
        plot_df["Actual_ms"]
    )

    plot_df["DYSCI_plot"] = (
        plot_df["DYSCI_Estimated_ms"]
    )

    plot_df["Iluvatar_plot"] = (
        plot_df["Iluvatar_Estimated_ms"]
    )


# ============================================================
# Plot
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    plot_df["Request"],
    plot_df["Actual_plot"],
    label="Actual Latency",
    linewidth=2.2,
)

plt.plot(
    plot_df["Request"],
    plot_df["DYSCI_plot"],
    label="DYSCI Estimated Latency",
    linewidth=2.0,
    linestyle="--",
)

plt.plot(
    plot_df["Request"],
    plot_df["Iluvatar_plot"],
    label="Ilúvatar Estimated Latency",
    linewidth=2.0,
    linestyle=":",
)


# ============================================================
# Labels
# ============================================================

plt.xlabel(
    "Request",
    fontsize=12,
)

plt.ylabel(
    "Latency (ms)",
    fontsize=12,
)

plt.title(
    f"Actual vs Estimated Latency — {MODEL}",
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
# Save figure
# ============================================================

OUTPUT_FILE = (
    f"{MODEL}_actual_vs_estimated_latency.png"
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