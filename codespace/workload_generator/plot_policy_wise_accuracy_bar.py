# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# SUMMARY_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/ensemble_size3_policy_accuracy.csv")
#
# def plot_bar():
#     df = pd.read_csv(SUMMARY_PATH)
#     df = df.sort_values("avg_accuracy", ascending=False)
#
#     plt.figure(figsize=(8, 4))
#     x = range(len(df))
#     plt.bar(x, df["avg_accuracy"], color="#bb8fce")
#
#     plt.xticks(x, df["combiner_policy"], rotation=30, ha="right")
#     plt.ylabel("Average Accuracy")
#     # plt.title("Final Accuracy by Policy (Model Set Size = 3)")
#     plt.ylim(0, 1.05)
#
#     for i, acc in enumerate(df["avg_accuracy"]):
#         plt.text(i, acc + 0.01, f"{acc:.3f}", ha="center", fontsize=9)
#
#     plt.tight_layout()
#     out = SUMMARY_PATH.with_name("ensemble_size3_policy_accuracy_bar.png")
#     plt.savefig(out, dpi=200)
#     plt.show()
#
# if __name__ == "__main__":
#     plot_bar()



# import pandas as pd
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# SUMMARY_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/ensemble_size3_policy_accuracy.csv")
#
# def plot_bar():
#     df = pd.read_csv(SUMMARY_PATH)
#     df = df.sort_values("avg_accuracy", ascending=False)
#
#     # Keep SAME image size, but increase fonts
#     plt.rcParams.update({
#         "font.size": 14,        # base font
#         "axes.labelsize": 16,
#         "xtick.labelsize": 14,
#         "ytick.labelsize": 14,
#     })
#
#     plt.figure(figsize=(8, 4))  # SAME size as before
#     x = range(len(df))
#     plt.bar(x, df["avg_accuracy"], color="#bb8fce")
#
#     plt.xticks(x, df["combiner_policy"], rotation=25, ha="right")
#     plt.ylabel("Average Accuracy")
#     plt.ylim(0, 1.05)
#
#     # Bigger value labels
#     for i, acc in enumerate(df["avg_accuracy"]):
#         plt.text(i, acc + 0.015, f"{acc:.3f}", ha="center", fontsize=14)
#
#     # Give more room for bigger x-axis labels (without changing figure size)
#     plt.subplots_adjust(bottom=0.30)
#
#     out = SUMMARY_PATH.with_name("ensemble_size3_policy_accuracy_bar.png")
#     plt.savefig(out, dpi=400, bbox_inches="tight")
#     plt.show()
#
#     print(f"Saved: {out}")
#
# if __name__ == "__main__":
#     plot_bar()




import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SUMMARY_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/ensemble_size3_policy_accuracy.csv")

def plot_bar():
    df = pd.read_csv(SUMMARY_PATH)
    df = df.sort_values("avg_accuracy", ascending=False).reset_index(drop=True)

    # Keep SAME image size, but increase fonts
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
    })

    fig, ax = plt.subplots(figsize=(8, 4))  # SAME size

    x = list(range(len(df)))
    bars = ax.bar(x, df["avg_accuracy"], color="#bb8fce")

    ax.set_xticks(x)
    ax.set_xticklabels(df["combiner_policy"], rotation=25, ha="right")
    ax.set_ylabel("Average Accuracy")
    ax.set_ylim(0, 1.05)

    for i, acc in enumerate(df["avg_accuracy"]):
        ax.text(i, acc + 0.015, f"{acc:.3f}", ha="center", fontsize=14)

    # ✅ Fix clipped first label (e.g., "weighted_majority")
    # Keep plot tight but add a tiny x margin + slightly more left padding
    n = len(df)
    ax.set_xlim(-0.62, n - 0.45)   # small extra room on both sides
    ax.margins(x=0.02)

    # Tighten white space, but ensure left label is visible
    fig.subplots_adjust(left=0.11, right=0.995, top=0.98, bottom=0.30)

    out = SUMMARY_PATH.with_name("ensemble_size3_policy_accuracy_bar.png")
    fig.savefig(out, dpi=400, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")

if __name__ == "__main__":
    plot_bar()
