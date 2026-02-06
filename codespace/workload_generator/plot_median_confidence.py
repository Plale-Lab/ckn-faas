import ast
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 🔧 CONFIGURATION
# ==========================
CSV_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_confidence_logs_2_3_S_probD.csv")
OUT_MODEL_WISE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/model_wise_confidence.csv")
OUT_ENSEMBLE_WISE = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/ensemble_size_confidence.csv")
OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/confidence_bar.png")

MODEL_LIST: List[str] = [
    "mobilenet_v3_small", "resnet18", "resnet34",
    "resnet50", "resnet101", "vit_b_16"
]

def model_label_col(m: str) -> str:
    return f"{m}_label"

def model_prob_col(m: str) -> str:
    return f"{m}_prob"

def model_wait_col(m: str) -> str:
    return f"{m}_wait"

BASE_COLS = [
    "ID","Deadline","IAR","RespTime","RunTime","selected_models",
    "label","Accuracy","combiner_policy","e2e_time_ms","Success",
    "selected_folder","ModelSize"
]
PER_MODEL_LABEL_COLS = [model_label_col(m) for m in MODEL_LIST]
PER_MODEL_PROB_COLS  = [model_prob_col(m) for m in MODEL_LIST]
PER_MODEL_WAIT_COLS  = [model_wait_col(m) for m in MODEL_LIST]
LOG_HEADER = BASE_COLS + PER_MODEL_LABEL_COLS + PER_MODEL_PROB_COLS + PER_MODEL_WAIT_COLS + ["main_policy", "alpha"]

# ==========================
# helpers
# ==========================
def parse_selected_models(val):
    """Parse selected_models like "['resnet101']" into list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []

def safe_int_series(s: pd.Series, default: int = 0) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").fillna(default).astype(int)

def plot_like_yours(model_wise: pd.DataFrame, ensemble_wise: pd.DataFrame, out_plot: Path):
    """
    Bar plot in your previous style, but now using MEDIAN confidence.
    """
    # Only show size 2 and 3 in the first two bars
    ens_map = {int(r["ModelSize"]): float(r["median_confidence"]) for _, r in ensemble_wise.iterrows()}
    set2 = ens_map.get(2, float("nan"))
    set3 = ens_map.get(3, float("nan"))

    model_names = model_wise["single_model"].astype(str).tolist()
    model_vals  = model_wise["median_confidence"].astype(float).tolist()

    labels = ["Model set size 2", "Model set size 3"] + model_names
    vals   = [set2, set3] + model_vals

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 17,
        "xtick.labelsize": 16,
        "ytick.labelsize": 15,
    })

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, vals)

    # Keep your colors
    for idx, bar in enumerate(bars):
        if idx == 0:
            bar.set_color("#2E8B57")
            bar.set_linewidth(2)
        elif idx == 1:
            bar.set_color("#2F7D32")
            bar.set_linewidth(2)
        else:
            bar.set_color("steelblue")

    # Add value labels
    for bar, val in zip(bars, vals):
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=16
            )

    ax.set_ylabel("Median Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.margins(x=0.02)
    fig.subplots_adjust(left=0.095, right=0.995, top=0.98, bottom=0.28)

    fig.savefig(out_plot, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved to {out_plot}")

def main():
    # Read headerless CSV
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        names=LOG_HEADER,
        engine="python",
        na_values=["", "nan", "NaN", "-1", "-1.0"]
    )

    df["ModelSize"] = safe_int_series(df["ModelSize"], default=0)
    df["selected_models_list"] = df["selected_models"].apply(parse_selected_models)

    # Row-level confidence column (some logs store confidence in Accuracy field)
    row_conf_col = "Confidence" if "Confidence" in df.columns else "Accuracy"
    df[row_conf_col] = pd.to_numeric(df[row_conf_col], errors="coerce")

    # -------------------------
    # (A) Single-model confidence (ModelSize==1), by model
    # Prefer per-model prob column; fallback to row_conf_col
    # -------------------------
    df_m1 = df[df["ModelSize"] == 1].copy()
    df_m1["single_model"] = df_m1["selected_models_list"].apply(
        lambda L: L[0] if isinstance(L, list) and len(L) == 1 else pd.NA
    )
    df_m1 = df_m1.dropna(subset=["single_model"])

    def pick_single_conf(row):
        m = row["single_model"]
        col = model_prob_col(m)
        if col in df_m1.columns:
            v = row.get(col, np.nan)
            try:
                v = float(v)
            except Exception:
                v = np.nan
            if np.isfinite(v) and v >= 0:
                return v
        return row.get(row_conf_col, np.nan)

    df_m1["confidence_used"] = df_m1.apply(pick_single_conf, axis=1)

    # MEDIAN confidence per model
    model_wise = df_m1.groupby("single_model").agg(
        total=("confidence_used", "count"),
        median_confidence=("confidence_used", "median")
    ).reset_index()

    # Keep your ordering
    model_wise = model_wise.sort_values("single_model", ascending=True).reset_index(drop=True)
    model_wise.to_csv(OUT_MODEL_WISE, index=False)

    # -------------------------
    # (B) Ensemble confidence (ModelSize>1), by ModelSize only
    # Use row-level confidence column
    # -------------------------
    df_ens = df[df["ModelSize"] > 1].copy()

    ensemble_wise = df_ens.groupby("ModelSize").agg(
        total=(row_conf_col, "count"),
        median_confidence=(row_conf_col, "median")
    ).reset_index().sort_values("ModelSize")

    ensemble_wise.to_csv(OUT_ENSEMBLE_WISE, index=False)

    print("\n=== Single-model median confidence (ModelSize==1) ===")
    print(model_wise.to_string(index=False))

    print("\n=== Ensemble-size median confidence (ModelSize>1) ===")
    print(ensemble_wise.to_string(index=False))

    # Plot in your exact style (now median)
    plot_like_yours(model_wise, ensemble_wise, OUT_PLOT)

if __name__ == "__main__":
    main()
