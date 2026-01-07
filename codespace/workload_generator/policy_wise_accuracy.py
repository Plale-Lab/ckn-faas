"""
Generate summary + data for bar and box plots
"""

import pandas as pd
from pathlib import Path

# Path to your logger output
LOG_PATH = Path(
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/policy_wise_accuracy.csv"
)

# Output processed files
OUT_SUMMARY = LOG_PATH.with_name("ensemble_size3_policy_accuracy.csv")
OUT_BOXDATA = LOG_PATH.with_name("ensemble_size3_policy_boxdata.csv")

def build_summary():
    df = pd.read_csv(LOG_PATH)

    # Filter to ensemble size = 3 + successful requests only
    df3 = df[(df["ModelSize"] == 3) & (df["Success"] == True)]

    if df3.empty:
        print("No valid rows for ModelSize=3 + Success=True")
        return

    # Summary: average accuracy per policy
    summary = (
        df3.groupby("combiner_policy")["Accuracy"]
        .agg(avg_accuracy="mean", num_requests="size")
        .reset_index()
        .sort_values("avg_accuracy", ascending=False)
    )
    summary["ensemble_size"] = 3
    summary.to_csv(OUT_SUMMARY, index=False)

    # Boxplot data: save Accuracy per policy
    box_rows = []
    for pol, group in df3.groupby("combiner_policy"):
        for acc in group["Accuracy"].tolist():
            box_rows.append({"policy": pol, "accuracy": acc})

    box_df = pd.DataFrame(box_rows)
    box_df.to_csv(OUT_BOXDATA, index=False)

    print("Saved summary →", OUT_SUMMARY)
    print("Saved boxplot data →", OUT_BOXDATA)
    print(summary)


if __name__ == "__main__":
    build_summary()
