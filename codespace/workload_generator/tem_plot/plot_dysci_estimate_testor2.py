import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Configuration
# ============================================================

MODEL = "resnet34"

CSV_FILES = {
    "Run 1": (
        "/Users/agamage/Desktop/D2I/Codes Original/"
        "clone main/ckn-faas/codespace/"
        "workload_generator/data/tem_delete5.csv"
    ),

    "Run 2": (
        "/Users/agamage/Desktop/D2I/Codes Original/"
        "clone main/ckn-faas/codespace/"
        "workload_generator/data/tem_delete6.csv"
    ),

    "Run 3": (
        "/Users/agamage/Desktop/D2I/Codes Original/"
        "clone main/ckn-faas/codespace/"
        "workload_generator/data/tem_delete7.csv"
    ),
}


# ============================================================
# Calculate MAE for each run
# ============================================================

results = []

for run_name, csv_path in CSV_FILES.items():

    df = pd.read_csv(csv_path)

    actual_col = f"{MODEL}_actual"
    dysci_col = f"{MODEL}_dysci_est"
    iluvatar_col = f"{MODEL}_iluvatar_est"

    for col in [
        actual_col,
        dysci_col,
        iluvatar_col,
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    # Use exactly the same completed requests
    # for both estimators.
    df = df[
        (df[actual_col] >= 0)
        & (df[dysci_col] >= 0)
        & (df[iluvatar_col] >= 0)
    ].copy()

    # Convert seconds -> milliseconds
    actual_ms = df[actual_col] * 1000.0
    dysci_ms = df[dysci_col] * 1000.0
    iluvatar_ms = df[iluvatar_col] * 1000.0

    # Absolute error
    dysci_error = (
        dysci_ms - actual_ms
    ).abs()

    iluvatar_error = (
        iluvatar_ms - actual_ms
    ).abs()

    dysci_mae = dysci_error.mean()
    iluvatar_mae = iluvatar_error.mean()

    results.append(
        {
            "Run": run_name,
            "DYSCI": dysci_mae,
            "Ilúvatar": iluvatar_mae,
        }
    )


results_df = pd.DataFrame(results)

print(results_df)


# ============================================================
# Plot
# ============================================================

x = np.arange(
    len(results_df)
)

width = 0.35

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    x - width / 2,
    results_df["DYSCI"],
    width,
    label="DYSCI",
)

plt.bar(
    x + width / 2,
    results_df["Ilúvatar"],
    width,
    label="Ilúvatar",
)

plt.xticks(
    x,
    results_df["Run"],
)

plt.xlabel("Experiment Run")
plt.ylabel("Mean Absolute Error (ms)")

plt.title(
    f"Latency Estimation Accuracy — {MODEL}"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.3,
)

plt.tight_layout()

plt.savefig(
    f"{MODEL}_mae_comparison.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()