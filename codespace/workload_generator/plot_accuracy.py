# #!/usr/bin/env python3
# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # ==========================
# # 🔧 CONFIGURATION
# # ==========================
# SUMMARY_CSV = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
# OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_plot.png")
# # ==========================
#
# def main():
#     if not SUMMARY_CSV.exists():
#         raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")
#
#     df = pd.read_csv(SUMMARY_CSV)
#
#     # Ensure sorted order: ensemble first, then models
#     if "ensemble" in df["model"].values:
#         df = pd.concat([df[df["model"]=="ensemble"], df[df["model"]!="ensemble"]])
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(df["model"], df["avg_accuracy"], color="skyblue")
#
#     # Highlight ensemble bar in orange
#     for bar, model in zip(bars, df["model"]):
#         if model == "ensemble":
#             bar.set_color("limegreen")
#
#     # Add accuracy values on top of bars
#     for bar, acc in zip(bars, df["avg_accuracy"]):
#         plt.text(
#             bar.get_x() + bar.get_width()/2, bar.get_height(),
#             f"{acc:.2f}", ha="center", va="bottom", fontsize=9
#         )
#
#     plt.title("Model-wise vs Ensemble Average Accuracy", fontsize=14, weight="bold")
#     plt.ylabel("Average Accuracy")
#     plt.xticks(rotation=30, ha="right")
#     plt.ylim(0, 1.05)  # accuracy is between 0 and 1
#
#     plt.tight_layout()
#     plt.savefig(OUT_PLOT, dpi=300)
#     plt.show()
#
#     print(f"✅ Plot saved to {OUT_PLOT}")
#
# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import textwrap



# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ==========================
# # 🔧 CONFIGURATION
# # ==========================
# SUMMARY_CSV = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
# OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_plot.png")
# # ==========================
#
# def main():
#     if not SUMMARY_CSV.exists():
#         raise FileNotFoundError(f"Summary CSV not found: {SUMMARY_CSV}")
#
#     df = pd.read_csv(SUMMARY_CSV)
#
#     # Ensure ensemble first
#     if df["model"].str.contains("ensemble", case=False).any():
#         df = pd.concat([
#             df[df["model"].str.contains("ensemble", case=False)],
#             df[~df["model"].str.contains("ensemble", case=False)]
#         ])
#
#     # Mark ensemble BEFORE renaming
#     df["is_ensemble"] = df["model"].str.contains("Ensemble size", case=False)
#
#     # Rename
#     df["model"] = df["model"].replace({"Ensemble size 3": "Model set size 3"})
#     df["model"] = df["model"].replace({"Ensemble size 2": "Model set size 2"})
#
#     plt.figure(figsize=(10, 6))
#     bars = plt.bar(df["model"], df["avg_accuracy"], color="steelblue")
#
#     # Highlight ensemble bar
#     for bar, is_ensemble in zip(bars, df["is_ensemble"]):
#         if is_ensemble:
#             bar.set_color("forestgreen")
#             bar.set_linewidth(2)
#
#     # Add accuracy values on top with 3 decimal places
#     for bar, acc in zip(bars, df["avg_accuracy"]):
#         plt.text(
#             bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
#             f"{acc:.3f}", ha="center", va="bottom", fontsize=9
#         )
#
#     plt.title("Model-wise vs Model Set Average Accuracy", fontsize=14, weight="bold")
#     plt.ylabel("Average Accuracy")
#     plt.xticks(rotation=30, ha="right")
#     plt.ylim(0, 1.05)
#
#     plt.tight_layout()
#     plt.savefig(OUT_PLOT, dpi=300)
#     plt.show()
#
#     print(f"✅ Plot saved to {OUT_PLOT}")
#
# if __name__ == "__main__":
#     main()


# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ==========================
# # 🔧 CONFIGURATION
# # ==========================
# SUMMARY_CSV_M2 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
# SUMMARY_CSV_M3 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M3.csv")  # update if different
# OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_table1_plus_sets.png")
# # ==========================
#
# def find_value(df: pd.DataFrame, patterns: list[str]) -> float:
#     """Find avg_accuracy row by matching any pattern in model column."""
#     for p in patterns:
#         row = df[df["model"].astype(str).str.contains(p, case=False, na=False)]
#         if not row.empty:
#             return float(row["avg_accuracy"].iloc[0])
#     raise ValueError(f"Could not find any of these in df['model']: {patterns}")
#
# def main():
#     if not SUMMARY_CSV_M2.exists():
#         raise FileNotFoundError(f"Missing: {SUMMARY_CSV_M2}")
#     if not SUMMARY_CSV_M3.exists():
#         raise FileNotFoundError(f"Missing: {SUMMARY_CSV_M3}")
#
#     df2 = pd.read_csv(SUMMARY_CSV_M2)
#     df3 = pd.read_csv(SUMMARY_CSV_M3)
#
#     # --- get model-set accuracies from CSVs (these are usually 0-1 already) ---
#     set2_acc = find_value(df2, ["Ensemble size 2", "Model set size 2"])
#     set3_acc = find_value(df3, ["Ensemble size 3", "Model set size 3"])
#
#     # --- Table 1 values (Accuracy %) -> convert to 0-1 ---
#     table1_models = [
#         ("MobileNetV3-Small", 70.4),
#         ("ResNet-18",         71.2),
#         ("ResNet-34",         77.6),
#         ("ResNet-50",         79.8),
#         ("ResNet-101",        82.2),
#         ("ViT-B/16",          75.1),
#     ]
#     model_names = [m for m, _ in table1_models]
#     model_accs  = [a/100.0 for _, a in table1_models]
#
#     # --- Combine for plot (model sets first, then models) ---
#     labels = ["Model set size 2", "Model set size 3"] + model_names
#     accs   = [set2_acc, set3_acc] + model_accs
#     is_set = [True, True] + [False]*len(model_names)
#
#     plt.figure(figsize=(11, 5.5))
#     bars = plt.bar(labels, accs)
#
#     # Highlight model-set bars
#     for bar, flag in zip(bars, is_set):
#         if flag:
#             bar.set_color("forestgreen")
#             bar.set_linewidth(2)
#         else:
#             bar.set_color("steelblue")
#
#     # Value labels
#     for bar, val in zip(bars, accs):
#         plt.text(
#             bar.get_x() + bar.get_width()/2,
#             bar.get_height() + 0.01,
#             f"{val:.3f}",
#             ha="center", va="bottom", fontsize=9
#         )
#
#     # If you want to save space in paper, you can remove the title and put it in caption
#     # plt.title("Model-wise Accuracy (Table 1) and Model Set Accuracy (Size 2 vs 3)", fontsize=13, weight="bold")
#
#     plt.ylabel("Average Accuracy")
#     plt.xticks(rotation=25, ha="right")
#     plt.ylim(0, 1.05)
#     plt.tight_layout()
#     plt.savefig(OUT_PLOT, dpi=300)
#     plt.show()
#
#     print(f"✅ Plot saved to {OUT_PLOT}")
#
# if __name__ == "__main__":
#     main()



# from pathlib import Path
# import pandas as pd
# import matplotlib.pyplot as plt
#
# # ==========================
# # 🔧 CONFIGURATION
# # ==========================
# SUMMARY_CSV_M2 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
# SUMMARY_CSV_M3 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M3.csv")
# OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_table1_plus_sets.png")
# # ==========================
#
# def find_value(df: pd.DataFrame, patterns: list[str]) -> float:
#     """Find avg_accuracy row by matching any pattern in model column."""
#     for p in patterns:
#         row = df[df["model"].astype(str).str.contains(p, case=False, na=False)]
#         if not row.empty:
#             return float(row["avg_accuracy"].iloc[0])
#     raise ValueError(f"Could not find any of these in df['model']: {patterns}")
#
# def main():
#     if not SUMMARY_CSV_M2.exists():
#         raise FileNotFoundError(f"Missing: {SUMMARY_CSV_M2}")
#     if not SUMMARY_CSV_M3.exists():
#         raise FileNotFoundError(f"Missing: {SUMMARY_CSV_M3}")
#
#     df2 = pd.read_csv(SUMMARY_CSV_M2)
#     df3 = pd.read_csv(SUMMARY_CSV_M3)
#
#     # --- get model-set accuracies from CSVs ---
#     set2_acc = find_value(df2, ["Ensemble size 2", "Model set size 2"])
#     set3_acc = find_value(df3, ["Ensemble size 3", "Model set size 3"])
#
#     # --- Table 1 values (Accuracy %) -> convert to 0-1 ---
#     table1_models = [
#         ("MobileNetV3-Small", 70.4),
#         ("ResNet-18",         71.2),
#         ("ResNet-34",         77.6),
#         ("ResNet-50",         79.8),
#         ("ResNet-101",        82.2),
#         ("ViT-B/16",          75.1),
#     ]
#     model_names = [m for m, _ in table1_models]
#     model_accs  = [a/100.0 for _, a in table1_models]
#
#     # --- Combine for plot (model sets first, then models) ---
#     labels = ["Model set size 2", "Model set size 3"] + model_names
#     accs   = [set2_acc, set3_acc] + model_accs
#
#     # ---- Keep SAME plot size, but increase font sizes (like previous fix) ----
#     plt.rcParams.update({
#         "font.size": 14,        # base
#         "axes.labelsize": 16,
#         "xtick.labelsize": 14,
#         "ytick.labelsize": 14,
#     })
#
#     plt.figure(figsize=(11, 5.5))  # keep your current size (don’t change)
#     bars = plt.bar(labels, accs)
#
#     # Colors:
#     # - Model set size 2: light green
#     # - Model set size 3: darker green
#     # - Single models: steelblue
#     for idx, bar in enumerate(bars):
#         if idx == 0:          # Model set size 2
#             bar.set_color("#2E8B57")
#             bar.set_linewidth(2)
#         elif idx == 1:        # Model set size 3
#             bar.set_color("#2F7D32")
#             bar.set_linewidth(2)
#         else:                 # single models (Table 1)
#             bar.set_color("steelblue")
#
#     # Bigger value labels
#     for bar, val in zip(bars, accs):
#         plt.text(
#             bar.get_x() + bar.get_width()/2,
#             bar.get_height() + 0.012,
#             f"{val:.3f}",
#             ha="center", va="bottom", fontsize=14
#         )
#
#     plt.ylabel("Average Accuracy")
#     plt.xticks(rotation=25, ha="right")
#     plt.ylim(0, 1.05)
#
#     # Give a bit more bottom room for rotated labels (helps readability)
#     plt.subplots_adjust(bottom=0.25)
#
#     plt.savefig(OUT_PLOT, dpi=400, bbox_inches="tight")
#     plt.show()
#
#     print(f"✅ Plot saved to {OUT_PLOT}")
#
# if __name__ == "__main__":
#     main()



from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SUMMARY_CSV_M2 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M2.csv")
SUMMARY_CSV_M3 = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/evaluation_summary_M3.csv")
OUT_PLOT = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/accuracy_table1_plus_sets.png")

def find_value(df: pd.DataFrame, patterns: list[str]) -> float:
    for p in patterns:
        row = df[df["model"].astype(str).str.contains(p, case=False, na=False)]
        if not row.empty:
            return float(row["avg_accuracy"].iloc[0])
    raise ValueError(f"Could not find any of these in df['model']: {patterns}")

def main():
    df2 = pd.read_csv(SUMMARY_CSV_M2)
    df3 = pd.read_csv(SUMMARY_CSV_M3)

    set2_acc = find_value(df2, ["Ensemble size 2", "Model set size 2"])
    set3_acc = find_value(df3, ["Ensemble size 3", "Model set size 3"])

    table1_models = [
        ("MobileNetV3-Small", 70.4),
        ("ResNet-18",         71.2),
        ("ResNet-34",         77.6),
        ("ResNet-50",         79.8),
        ("ResNet-101",        82.2),
        ("ViT-B/16",          75.1),
    ]
    model_names = [m for m, _ in table1_models]
    model_accs  = [a/100.0 for _, a in table1_models]

    labels = ["Model set size 2", "Model set size 3"] + model_names
    accs   = [set2_acc, set3_acc] + model_accs

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 17,
        "xtick.labelsize": 16,
        "ytick.labelsize": 15,
    })

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, accs)

    for idx, bar in enumerate(bars):
        if idx == 0:
            bar.set_color("#2E8B57")
            bar.set_linewidth(2)
        elif idx == 1:
            bar.set_color("#2F7D32")
            bar.set_linewidth(2)
        else:
            bar.set_color("steelblue")

    for bar, val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.012,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=16
        )

    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_xticklabels(labels, rotation=25, ha="right")

    # ✅ FIX: avoid clipping first label ("Model set size 2")
    # Keep plot tight but add a tiny x margin so left label isn't cut
    ax.margins(x=0.02)

    # Keep your tight layout, but give a bit more left padding
    fig.subplots_adjust(left=0.095, right=0.995, top=0.98, bottom=0.28)

    fig.savefig(OUT_PLOT, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"✅ Plot saved to {OUT_PLOT}")

if __name__ == "__main__":
    main()



