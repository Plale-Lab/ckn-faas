# #!/usr/bin/env python3
# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # --- Path to your CSV ---
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_seq_M3.csv"
#
# # ---- If your file is headerless, define the model order here (must match how you write CSV) ----
# MODEL_LIST = [
#     "mobilenet_v3_small",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "vit_b_16",
# ]
#
# # In your row example, the first 5 columns are metadata, then 6 est values, then 6 act values
# META_COLS = ["ID", "Deadline", "IAR", "RespTime", "RunTime"]
# NUM_META = len(META_COLS)
#
# def _load_csv_smart(path: str) -> pd.DataFrame:
#     """
#     Try to load with headers. If we can't find est_*/act_* columns,
#     treat the CSV as headerless with fixed layout and synthesize headers.
#     """
#     # Try with header row
#     df_try = pd.read_csv(path)
#     has_named_pairs = any(c.startswith("est_") for c in df_try.columns) and any(
#         c.startswith("act_") for c in df_try.columns
#     )
#     if has_named_pairs:
#         return df_try
#
#     # Fallback: headerless layout -> assign names
#     df = pd.read_csv(path, header=None)
#
#     # Sanity: total columns should be NUM_META + 2 * len(MODEL_LIST)
#     expected_cols = NUM_META + 2 * len(MODEL_LIST)
#     if df.shape[1] != expected_cols:
#         raise ValueError(
#             f"Headerless CSV has {df.shape[1]} columns; expected {expected_cols} "
#             f"(= {NUM_META} meta + 2 * {len(MODEL_LIST)} models). "
#             f"Update MODEL_LIST or META_COLS to match your file."
#         )
#
#     # Build column names
#     columns = []
#     columns.extend(META_COLS)
#     columns.extend([f"est_{m}" for m in MODEL_LIST])
#     columns.extend([f"act_{m}" for m in MODEL_LIST])
#     df.columns = columns
#     return df
#
# def main():
#     df = _load_csv_smart(CSV_PATH)
#
#     # --- Discover model columns dynamically ---
#     est_cols = [c for c in df.columns if c.startswith("est_")]
#     act_cols = [c for c in df.columns if c.startswith("act_")]
#
#     models_est = {c.replace("est_", "") for c in est_cols}
#     models_act = {c.replace("act_", "") for c in act_cols}
#     models = sorted(models_est & models_act)
#
#     if not models:
#         raise ValueError("No matching est_*/act_* column pairs found in CSV (even after headerless parsing).")
#
#     # --- Long-format dataframe: one row per (request, model) ---
#     records = []
#     for m in models:
#         e = pd.to_numeric(df[f"est_{m}"], errors="coerce")
#         a = pd.to_numeric(df[f"act_{m}"], errors="coerce")
#         for est, act in zip(e, a):
#             if pd.isna(est) and pd.isna(act):
#                 continue
#             records.append({"model": m, "estimated": est, "actual": act})
#
#     long_df = pd.DataFrame(records).dropna(subset=["estimated", "actual"])
#
#     # =========================
#     # A) SCATTER (show only)
#     # =========================
#     plt.figure(figsize=(7, 6))
#     for m in models:
#         sub = long_df[long_df["model"] == m]
#         if sub.empty:
#             continue
#         plt.scatter(sub["estimated"], sub["actual"], label=m, alpha=0.7, s=28)
#
#     # Parity line
#     xy_max = float(np.nanmax([long_df["estimated"].max(), long_df["actual"].max()]))
#     plt.plot([0, xy_max], [0, xy_max], linestyle="--", linewidth=1)
#
#     plt.xlabel("Estimated latency (s)")
#     plt.ylabel("Actual latency (s)")
#     plt.title("Estimated vs Actual Latency (per request, all models)")
#     plt.legend(fontsize=8, ncol=2)
#     plt.tight_layout()
#     plt.show()
#
#     # =========================================
#     # B) MODEL-WISE TIME SERIES (show only)
#     # =========================================
#     # Build a request index to represent time order
#     request_idx = np.arange(len(df))
#
#     for m in models:
#         est_series = pd.to_numeric(df[f"est_{m}"], errors="coerce")
#         act_series = pd.to_numeric(df[f"act_{m}"], errors="coerce")
#
#         # Skip empty columns safely
#         if est_series.isna().all() and act_series.isna().all():
#             continue
#
#         plt.figure(figsize=(8, 3.6))
#         plt.plot(request_idx, est_series, marker="o", linewidth=1.0, markersize=3, label="Estimated")
#         plt.plot(request_idx, act_series, marker="s", linewidth=1.0, markersize=3, label="Actual")
#
#         plt.title(f"Latency over Requests — {m}")
#         plt.xlabel("Request index (time order)")
#         plt.ylabel("Latency (s)")
#         plt.legend()
#         plt.tight_layout()
#         plt.show()
#
# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_power2_M3.csv"

# Adjust this list if your log schema changes
COLUMNS = [
    "req_id",                 # 0
    "deadline",               # 1
    "iar",                    # 2
    "response_time_s",        # 3
    "current_time_sec",       # 4
    "selected_models",        # 5
    "executed_models",        # 6
    "label",                  # 7
    "accuracy",               # 8
    "combiner",               # 9
    "e2e_ms",                 # 10
    "success",                # 11
    "selected_folder",        # 12
    "ensemble_size",          # 13
    "threshold",              # 14
    "threshold_stage",        # 15
    "parallel_first_n",       # 16
    "stop_reason",            # 17
    "qps_est",                # 18
    "main_policy",            # 19
    "alpha",                  # 20
    "alpha_mode",             # 21

    # model labels
    "mobilenet_label",        # 22
    "resnet18_label",         # 23
    "resnet34_label",         # 24
    "resnet50_label",         # 25
    "resnet101_label",        # 26
    "vit_b_16_label",         # 27

    # model probs
    "mobilenet_prob",         # 28
    "resnet18_prob",          # 29
    "resnet34_prob",          # 30
    "resnet50_prob",          # 31
    "resnet101_prob",         # 32
    "vit_b_16_prob",          # 33

    # model waits
    "mobilenet_wait",         # 34
    "resnet18_wait",          # 35
    "resnet34_wait",          # 36
    "resnet50_wait",          # 37
    "resnet101_wait",         # 38
    "vit_b_16_wait",          # 39

    # final selected estimated latency
    "selected_est_latency_s", # 40
]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)

    if df.shape[1] != len(COLUMNS):
        raise ValueError(
            f"CSV has {df.shape[1]} columns, but expected {len(COLUMNS)}.\n"
            f"Please check whether the logging format changed."
        )

    df.columns = COLUMNS
    return df


def main():
    df = load_csv(CSV_PATH)

    # Convert numeric columns
    df["response_time_s"] = pd.to_numeric(df["response_time_s"], errors="coerce")
    df["e2e_ms"] = pd.to_numeric(df["e2e_ms"], errors="coerce")
    df["selected_est_latency_s"] = pd.to_numeric(df["selected_est_latency_s"], errors="coerce")

    # Choose actual latency source:
    # Option 1: use response_time_s directly
    df["actual_latency_s"] = df["response_time_s"]

    # Option 2: if you prefer e2e_ms, uncomment this instead
    # df["actual_latency_s"] = df["e2e_ms"] / 1000.0

    # Drop rows with missing values
    plot_df = df.dropna(subset=["actual_latency_s", "selected_est_latency_s"]).copy()

    if plot_df.empty:
        raise ValueError("No valid rows found for actual vs estimated latency plotting.")

    request_idx = np.arange(len(plot_df))

    # =========================================
    # 1) Time-series plot
    # =========================================
    plt.figure(figsize=(10, 4.5))
    plt.plot(
        request_idx,
        plot_df["actual_latency_s"],
        marker="o",
        linewidth=1.2,
        markersize=3,
        label="Actual latency"
    )
    plt.plot(
        request_idx,
        plot_df["selected_est_latency_s"],
        marker="s",
        linewidth=1.2,
        markersize=3,
        label="Estimated latency"
    )

    plt.title("Actual vs Estimated Latency over Requests")
    plt.xlabel("Request index")
    plt.ylabel("Latency (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # =========================================
    # 2) Scatter plot
    # =========================================
    plt.figure(figsize=(6.5, 6))
    plt.scatter(
        plot_df["selected_est_latency_s"],
        plot_df["actual_latency_s"],
        alpha=0.7,
        s=28
    )

    xy_max = float(
        np.nanmax([
            plot_df["selected_est_latency_s"].max(),
            plot_df["actual_latency_s"].max()
        ])
    )
    plt.plot([0, xy_max], [0, xy_max], linestyle="--", linewidth=1)

    plt.xlabel("Estimated latency (s)")
    plt.ylabel("Actual latency (s)")
    plt.title("Estimated vs Actual Latency")
    plt.tight_layout()
    plt.show()

    # =========================================
    # 3) Error plot
    # =========================================
    plot_df["latency_error_s"] = plot_df["actual_latency_s"] - plot_df["selected_est_latency_s"]

    plt.figure(figsize=(10, 4))
    plt.plot(
        request_idx,
        plot_df["latency_error_s"],
        marker="o",
        linewidth=1.0,
        markersize=3
    )
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.title("Latency Estimation Error over Requests")
    plt.xlabel("Request index")
    plt.ylabel("Actual - Estimated (s)")
    plt.tight_layout()
    plt.show()

    # =========================================
    # 4) Summary
    # =========================================
    mae = np.mean(np.abs(plot_df["latency_error_s"]))
    rmse = np.sqrt(np.mean(plot_df["latency_error_s"] ** 2))

    print(f"Rows used: {len(plot_df)}")
    print(f"MAE  (s): {mae:.6f}")
    print(f"RMSE (s): {rmse:.6f}")


if __name__ == "__main__":
    main()