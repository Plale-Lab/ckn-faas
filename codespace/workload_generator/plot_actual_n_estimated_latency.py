#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Path to your CSV ---
CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_M2.csv"

# ---- If your file is headerless, define the model order here (must match how you write CSV) ----
MODEL_LIST = [
    "mobilenet_v3_small",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "vit_b_16",
]

# In your row example, the first 5 columns are metadata, then 6 est values, then 6 act values
META_COLS = ["ID", "Deadline", "IAR", "RespTime", "RunTime"]
NUM_META = len(META_COLS)

def _load_csv_smart(path: str) -> pd.DataFrame:
    """
    Try to load with headers. If we can't find est_*/act_* columns,
    treat the CSV as headerless with fixed layout and synthesize headers.
    """
    # Try with header row
    df_try = pd.read_csv(path)
    has_named_pairs = any(c.startswith("est_") for c in df_try.columns) and any(
        c.startswith("act_") for c in df_try.columns
    )
    if has_named_pairs:
        return df_try

    # Fallback: headerless layout -> assign names
    df = pd.read_csv(path, header=None)

    # Sanity: total columns should be NUM_META + 2 * len(MODEL_LIST)
    expected_cols = NUM_META + 2 * len(MODEL_LIST)
    if df.shape[1] != expected_cols:
        raise ValueError(
            f"Headerless CSV has {df.shape[1]} columns; expected {expected_cols} "
            f"(= {NUM_META} meta + 2 * {len(MODEL_LIST)} models). "
            f"Update MODEL_LIST or META_COLS to match your file."
        )

    # Build column names
    columns = []
    columns.extend(META_COLS)
    columns.extend([f"est_{m}" for m in MODEL_LIST])
    columns.extend([f"act_{m}" for m in MODEL_LIST])
    df.columns = columns
    return df

def main():
    df = _load_csv_smart(CSV_PATH)

    # --- Discover model columns dynamically ---
    est_cols = [c for c in df.columns if c.startswith("est_")]
    act_cols = [c for c in df.columns if c.startswith("act_")]

    models_est = {c.replace("est_", "") for c in est_cols}
    models_act = {c.replace("act_", "") for c in act_cols}
    models = sorted(models_est & models_act)

    if not models:
        raise ValueError("No matching est_*/act_* column pairs found in CSV (even after headerless parsing).")

    # --- Long-format dataframe: one row per (request, model) ---
    records = []
    for m in models:
        e = pd.to_numeric(df[f"est_{m}"], errors="coerce")
        a = pd.to_numeric(df[f"act_{m}"], errors="coerce")
        for est, act in zip(e, a):
            if pd.isna(est) and pd.isna(act):
                continue
            records.append({"model": m, "estimated": est, "actual": act})

    long_df = pd.DataFrame(records).dropna(subset=["estimated", "actual"])

    # =========================
    # A) SCATTER (show only)
    # =========================
    plt.figure(figsize=(7, 6))
    for m in models:
        sub = long_df[long_df["model"] == m]
        if sub.empty:
            continue
        plt.scatter(sub["estimated"], sub["actual"], label=m, alpha=0.7, s=28)

    # Parity line
    xy_max = float(np.nanmax([long_df["estimated"].max(), long_df["actual"].max()]))
    plt.plot([0, xy_max], [0, xy_max], linestyle="--", linewidth=1)

    plt.xlabel("Estimated latency (s)")
    plt.ylabel("Actual latency (s)")
    plt.title("Estimated vs Actual Latency (per request, all models)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

    # =========================================
    # B) MODEL-WISE TIME SERIES (show only)
    # =========================================
    # Build a request index to represent time order
    request_idx = np.arange(len(df))

    for m in models:
        est_series = pd.to_numeric(df[f"est_{m}"], errors="coerce")
        act_series = pd.to_numeric(df[f"act_{m}"], errors="coerce")

        # Skip empty columns safely
        if est_series.isna().all() and act_series.isna().all():
            continue

        plt.figure(figsize=(8, 3.6))
        plt.plot(request_idx, est_series, marker="o", linewidth=1.0, markersize=3, label="Estimated")
        plt.plot(request_idx, act_series, marker="s", linewidth=1.0, markersize=3, label="Actual")

        plt.title(f"Latency over Requests — {m}")
        plt.xlabel("Request index (time order)")
        plt.ylabel("Latency (s)")
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
