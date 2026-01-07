# #!/usr/bin/env python3
# import json
# import re
# from pathlib import Path
# from typing import Dict, Set, List
# import pandas as pd
#
# # ==========================
# # 🔧 CONFIGURATION
# # ==========================
# CSV_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/ensemble_and _model_wise_accuracy.csv")
# LABELS_PATH = Path("Labels.json")  # or absolute path
# OUT_ROW_ANNOTATED = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_rows_M2.csv")
# OUT_SUMMARY = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
#
# # Fixed model list in the order you log them
# MODEL_LIST: List[str] = [
#     "vit_b_16", "resnet101", "mobilenet_v3_small", "resnet50", "resnet34", "resnet18"
# ]
#
# def model_label_col(m: str) -> str: return f"{m}_label"
# def model_prob_col(m: str) -> str:  return f"{m}_prob"
# def model_wait_col(m: str) -> str:  return f"{m}_wait"
#
# # Build header EXACTLY matching your row order (no selected_image)
# BASE_COLS = [
#     "ID","Deadline","IAR","RespTime","RunTime","selected_models",
#     "label","Accuracy","combiner_policy","e2e_time_ms","Success",
#     "selected_folder"
# ]
# PER_MODEL_LABEL_COLS = [model_label_col(m) for m in MODEL_LIST]
# PER_MODEL_PROB_COLS  = [model_prob_col(m) for m in MODEL_LIST]
# PER_MODEL_WAIT_COLS  = [model_wait_col(m) for m in MODEL_LIST]
# LOG_HEADER = BASE_COLS + PER_MODEL_LABEL_COLS + PER_MODEL_PROB_COLS + PER_MODEL_WAIT_COLS
# # ==========================
#
#
# # Keep for display if needed (not used in strict match)
# def normalize_label(s: str) -> str:
#     s = (s or "").strip().lower()
#     s = s.replace("_"," ").replace("-"," ")
#     s = re.sub(r"\s+"," ", s)
#     return s
#
# def tokenize_alpha(s: str):
#     """Return lowercase alphabetic tokens (drop digits/punct)."""
#     return re.findall(r"[a-z]+", (s or "").lower())
#
# def build_synset_map(label_json_path: Path) -> Dict[str, Set[str]]:
#     with open(label_json_path, "r") as f:
#         raw = json.load(f)
#     m: Dict[str, Set[str]] = {}
#     for wnid, names in raw.items():
#         # split synonyms by comma, normalize, but we will tokenize per comparison
#         parts = [normalize_label(x) for x in str(names).split(",")]
#         parts = [p for p in (x.strip() for x in parts) if p]
#         m[wnid] = set(parts)
#     return m
#
# def row_is_correct(wnid_map: Dict[str, Set[str]], wnid: str, pred_raw: str) -> bool:
#     """
#     Token-based matching:
#       - exact token match, OR
#       - predicted tokens ⊆ synonym tokens (e.g., 'nautilus' vs 'chambered nautilus')
#     Digits/junk like '1ww' are ignored as tokens (filtered out), which prevents false positives.
#     """
#     if not wnid or not pred_raw:
#         return False
#
#     pred_tokens = tokenize_alpha(pred_raw)
#     if not pred_tokens:
#         return False
#
#     syns = wnid_map.get(wnid, set())
#     if not syns:
#         return False
#
#     for syn in syns:
#         syn_tokens = tokenize_alpha(syn)
#
#         # exact token match
#         if pred_tokens == syn_tokens:
#             return True
#
#         # subset: allow shorter correct names (e.g., "nautilus") to match longer synonyms
#         if set(pred_tokens).issubset(set(syn_tokens)) and len(pred_tokens) >= 1:
#             return True
#
#     return False
#
#
# def main():
#     if not CSV_PATH.exists():
#         raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
#     if not LABELS_PATH.exists():
#         raise FileNotFoundError(f"Labels.json not found: {LABELS_PATH}")
#
#     wnid_map = build_synset_map(LABELS_PATH)
#
#     # Your file has NO header → read with our explicit header
#     df = pd.read_csv(CSV_PATH, header=None, names=LOG_HEADER)
#
#     # ---- Ensemble correctness (using final 'label') ----
#     df["ensemble_correct"] = [
#         row_is_correct(
#             wnid_map,
#             str(wnid) if pd.notna(wnid) else "",
#             str(pred) if pd.notna(pred) else "",
#         )
#         for wnid, pred in zip(df["selected_folder"], df["label"])
#     ]
#
#     # ---- Per-model correctness (for each *_label column present) ----
#     for m in MODEL_LIST:
#         col = model_label_col(m)
#         if col in df.columns:
#             df[f"{m}_correct"] = [
#                 row_is_correct(
#                     wnid_map,
#                     str(wnid) if pd.notna(wnid) else "",
#                     str(pred) if pd.notna(pred) else "",
#                 )
#                 for wnid, pred in zip(df["selected_folder"], df[col])
#             ]
#         else:
#             df[f"{m}_correct"] = pd.NA
#
#     # ---- Accuracies (as average accuracy) ----
#     evaluable_ens = df[df["selected_folder"].notna() & df["label"].notna()]
#     ens_total = len(evaluable_ens)
#     ens_acc = float(evaluable_ens["ensemble_correct"].mean()) if ens_total > 0 else 0.0
#
#     per_model_acc_rows = []
#     for m in MODEL_LIST:
#         col_label = model_label_col(m)
#         col_corr  = f"{m}_correct"
#         if col_label in df.columns:
#             evaluable_m = df[df["selected_folder"].notna() & df[col_label].notna()]
#             total_m = len(evaluable_m)
#             acc_m = float(evaluable_m[col_corr].mean()) if total_m > 0 else float("nan")
#             per_model_acc_rows.append({"model": m, "total": total_m, "avg_accuracy": acc_m})
#         else:
#             per_model_acc_rows.append({"model": m, "total": 0, "avg_accuracy": float("nan")})
#
#     summary_rows = [{"model": "ensemble", "total": ens_total, "avg_accuracy": ens_acc}] + per_model_acc_rows
#     summary_df = pd.DataFrame(summary_rows)
#
#     # ---- Save per-row annotated (keep useful cols if present) ----
#     keep_cols = [
#         "ID","Deadline","IAR","RespTime","RunTime","selected_models",
#         "selected_folder","label","Accuracy",
#         "combiner_policy","e2e_time_ms","Success","ensemble_correct",
#     ]
#     for m in MODEL_LIST:
#         if model_label_col(m) in df.columns: keep_cols += [model_label_col(m)]
#         if model_prob_col(m)  in df.columns: keep_cols += [model_prob_col(m)]
#         if model_wait_col(m)  in df.columns: keep_cols += [model_wait_col(m)]
#         if f"{m}_correct"     in df.columns: keep_cols += [f"{m}_correct"]
#
#     keep_cols = [c for c in keep_cols if c in df.columns]
#     df[keep_cols].to_csv(OUT_ROW_ANNOTATED, index=False)
#     summary_df.to_csv(OUT_SUMMARY, index=False)
#
#     # ---- Console summary ----
#     print("=== Evaluation Summary ===")
#     print(f"CSV file:          {CSV_PATH}")
#     print(f"Labels (json):     {LABELS_PATH}")
#     print(f"Ensemble support:  {ens_total}")
#     print(f"Ensemble Avg Acc:  {ens_acc:.4f}\n")
#     print("Per-model Avg Acc:")
#     for r in per_model_acc_rows:
#         acc_str = "nan" if pd.isna(r['avg_accuracy']) else f"{r['avg_accuracy']:.4f}"
#         print(f"  - {r['model']:<20} total={r['total']:<5} acc={acc_str}")
#     print(f"\nPer-row annotated: {OUT_ROW_ANNOTATED}")
#     print(f"Summary for plots: {OUT_SUMMARY}")
#
# if __name__ == "__main__":
#     main()
#



#!/usr/bin/env python3



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import re
import ast
from pathlib import Path
from typing import Dict, Set, List

import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# 🔧 CONFIGURATION
# ==========================
CSV_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/ensemble_and _model_wise_accuracy_M2.csv")
LABELS_PATH = Path("Labels.json")
OUT_ROW_ANNOTATED = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_rows_M2.csv")
OUT_SUMMARY = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_bar_M2.png")

# MUST match your logger’s per-model column naming
MODEL_LIST: List[str] = [
    "mobilenet_v3_small", "resnet18", "resnet34",
    "resnet50", "resnet101", "vit_b_16"
]

def model_label_col(m: str) -> str: return f"{m}_label"
def model_prob_col(m: str)  -> str: return f"{m}_prob"
def model_wait_col(m: str)  -> str: return f"{m}_wait"

# CSV header (file is headerless)
BASE_COLS = [
    "ID","Deadline","IAR","RespTime","RunTime","selected_models",
    "label","Accuracy","combiner_policy","e2e_time_ms","Success",
    "selected_folder","ModelSize"
]
PER_MODEL_LABEL_COLS = [model_label_col(m) for m in MODEL_LIST]
PER_MODEL_PROB_COLS  = [model_prob_col(m) for m in MODEL_LIST]
PER_MODEL_WAIT_COLS  = [model_wait_col(m) for m in MODEL_LIST]
LOG_HEADER = BASE_COLS + PER_MODEL_LABEL_COLS + PER_MODEL_PROB_COLS + PER_MODEL_WAIT_COLS + ["main_policy"]

# ==========================
# 🔍 helpers
# ==========================
def normalize_label(s: str) -> str:
    s = (s or "").strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = re.sub(r"\s+", " ", s)
    return s

def tokenize_alpha(s: str):
    return re.findall(r"[a-z]+", (s or "").lower())

def build_synset_map(label_json_path: Path) -> Dict[str, Set[str]]:
    """Labels.json maps wnid -> 'name1, name2, ...' or list; normalize to tokens set."""
    with open(label_json_path, "r") as f:
        raw = json.load(f)
    m: Dict[str, Set[str]] = {}
    for wnid, names in raw.items():
        parts = [normalize_label(x) for x in str(names).split(",")]
        parts = [p for p in (x.strip() for x in parts) if p]
        m[wnid] = set(parts)
    return m

def row_is_correct(wnid_map: Dict[str, Set[str]], wnid: str, pred_raw: str) -> bool:
    if not wnid or not pred_raw:
        return False
    pred_tokens = tokenize_alpha(pred_raw)
    if not pred_tokens:
        return False
    syns = wnid_map.get(wnid, set())
    if not syns:
        return False
    for syn in syns:
        syn_tokens = tokenize_alpha(syn)
        # exact token match or prediction tokens fully included in a synonym’s tokens
        if pred_tokens == syn_tokens or set(pred_tokens).issubset(set(syn_tokens)):
            return True
    return False

def parse_selected_models(val):
    """Safely parse the selected_models string like "['vit_b_16','resnet101']" to a list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            out = ast.literal_eval(val)
            return out if isinstance(out, list) else []
        except Exception:
            return []
    return []

def clean_pred_series(series: pd.Series) -> pd.Series:
    """
    Normalize placeholders to NA but keep true NaNs as NaN.
    DO NOT cast the whole column to str (which turns NaN into 'nan').
    """
    return series.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "-1": pd.NA, "-1.0": pd.NA, -1: pd.NA, -1.0: pd.NA})

# ==========================
# 🧮 main
# ==========================
def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels.json not found: {LABELS_PATH}")

    wnid_map = build_synset_map(LABELS_PATH)

    # Read CSV (headerless); treat standard placeholders as NA
    df = pd.read_csv(
        CSV_PATH,
        header=None,
        names=LOG_HEADER,
        engine="python",
        na_values=["", "nan", "NaN", "-1", "-1.0"]
    )
    df.columns = [c.strip() for c in df.columns]

    # Parse selected models column to lists
    df["selected_models_list"] = df["selected_models"].apply(parse_selected_models)

    # ---- Ensemble correctness (final selected label) ----
    df["ensemble_correct"] = [
        row_is_correct(
            wnid_map,
            str(wnid) if pd.notna(wnid) else "",
            str(pred) if pd.notna(pred) else "",
        )
        for wnid, pred in zip(df.get("selected_folder"), df.get("label"))
    ]

    # ---- Per-model correctness (ONLY on rows where the model was selected & has a label) ----
    per_model_acc_rows = []
    for m in MODEL_LIST:
        col_label = model_label_col(m)
        if col_label not in df.columns:
            per_model_acc_rows.append({"model": m, "total": 0, "avg_accuracy": float("nan")})
            continue

        preds = clean_pred_series(df[col_label])
        # Only evaluate on rows where:
        #   - the model m was selected for that request
        #   - we have a ground-truth (selected_folder)
        #   - we have a non-missing prediction for this model
        mask_selected = df["selected_models_list"].apply(lambda L: m in L)
        mask_eval = df["selected_folder"].notna() & mask_selected & preds.notna()

        total_m = int(mask_eval.sum())
        if total_m == 0:
            per_model_acc_rows.append({"model": m, "total": 0, "avg_accuracy": float("nan")})
            continue

        # Compute correctness on evaluable subset
        wnids = df.loc[mask_eval, "selected_folder"]
        model_preds = preds[mask_eval]
        correct_flags = [
            row_is_correct(
                wnid_map,
                str(wnid) if pd.notna(wnid) else "",
                str(pred) if pd.notna(pred) else "",
            )
            for wnid, pred in zip(wnids, model_preds)
        ]
        acc_m = float(pd.Series(correct_flags, dtype="float").mean()) if total_m > 0 else float("nan")
        per_model_acc_rows.append({"model": m, "total": total_m, "avg_accuracy": acc_m})

    # ---- Ensemble accuracy summary ----
    evaluable_ens = df[df["selected_folder"].notna() & df["label"].notna()]
    ens_total = int(len(evaluable_ens))
    ens_acc = float(evaluable_ens["ensemble_correct"].mean()) if ens_total > 0 else 0.0

    # ---- Build ensemble bar label with size/policy ----
    # ModelSize might be float/NaN; take the max observed as a simple descriptor
    max_size = pd.to_numeric(df.get("ModelSize", pd.Series(dtype=float)), errors="coerce").max()
    max_size = int(max_size) if pd.notna(max_size) else 0
    pol = str(df.get("main_policy", pd.Series(["mode_s"])).iloc[0]).strip().lower()

    if max_size <= 1:
        ensemble_label = "Fastest model"
    elif pol == "randomized":
        ensemble_label = f"Ensemble size {max_size}"
    elif pol in {"greedy", "mode_s", "mode-s", "modes"}:
        ensemble_label = f"Ensemble size {max_size}"
    else:
        ensemble_label = f"Ensemble size {max_size}"

    # ---- Summary table (first row = ensemble) ----
    summary_rows = [{"model": ensemble_label, "total": ens_total, "avg_accuracy": ens_acc}] + per_model_acc_rows
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_SUMMARY, index=False)

    # ---- Per-row annotated export (keep useful columns if present) ----
    keep_cols = [
        "ID","Deadline","IAR","RespTime","RunTime","selected_models",
        "selected_folder","label","Accuracy","combiner_policy","e2e_time_ms","Success",
        "ModelSize","main_policy","ensemble_correct",
    ]
    for m in MODEL_LIST:
        if model_label_col(m) in df.columns: keep_cols.append(model_label_col(m))
        if model_prob_col(m)  in df.columns: keep_cols.append(model_prob_col(m))
        if model_wait_col(m)  in df.columns: keep_cols.append(model_wait_col(m))
        corr_col = f"{m}_correct"
        # Recompute/store per-row correctness column for convenience in the annotated CSV
        if model_label_col(m) in df.columns:
            preds_all = clean_pred_series(df[model_label_col(m)])
            mask_sel = df["selected_models_list"].apply(lambda L, _m=m: _m in L)
            df[corr_col] = pd.NA
            idx = df.index[ df["selected_folder"].notna() & mask_sel & preds_all.notna() ]
            df.loc[idx, corr_col] = [
                row_is_correct(
                    wnid_map,
                    str(wn) if pd.notna(wn) else "",
                    str(pr) if pd.notna(pr) else "",
                )
                for wn, pr in zip(df.loc[idx, "selected_folder"], preds_all.loc[idx])
            ]
            keep_cols.append(corr_col)

    keep_cols = [c for c in keep_cols if c in df.columns]
    df[keep_cols].to_csv(OUT_ROW_ANNOTATED, index=False)

    # ---- Plot: ensemble in green, others blue ----
    colors = ["#2ecc71"] + ["#3498db"] * (len(summary_df := summary_df) - 1)
    plt.figure(figsize=(9, 4.6))
    bars = plt.bar(summary_df["model"], summary_df["avg_accuracy"], color=colors)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Average Accuracy")
    plt.title("Ensemble (with size) and Model-wise Accuracy")
    for b in bars:
        h = b.get_height()
        if pd.notna(h):
            plt.text(b.get_x() + b.get_width()/2, h + 0.01, f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    plt.ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(OUT_PLOT, dpi=200)
    plt.show()

    # ---- Console summary ----
    print("=== Evaluation Summary ===")
    print(f"CSV file:          {CSV_PATH}")
    print(f"Labels (json):     {LABELS_PATH}")
    print(f"Ensemble support:  {ens_total}")
    print(f"Ensemble Avg Acc:  {ens_acc:.4f}\n")
    print("Per-model Avg Acc:")
    for r in per_model_acc_rows:
        acc_str = "nan" if pd.isna(r['avg_accuracy']) else f"{r['avg_accuracy']:.4f}"
        print(f"  - {r['model']:<20} total={r['total']:<5} acc={acc_str}")
    print(f"\nPer-row annotated: {OUT_ROW_ANNOTATED}")
    print(f"Summary for plots: {OUT_SUMMARY}")
    print(f"Bar chart saved:   {OUT_PLOT}")

if __name__ == "__main__":
    main()



