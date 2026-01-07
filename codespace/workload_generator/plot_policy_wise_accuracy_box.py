import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BOXDATA_PATH = Path("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/ensemble_size3_policy_boxdata.csv")

def plot_box():
    df = pd.read_csv(BOXDATA_PATH)

    policies = df["policy"].unique().tolist()
    data = [df[df["policy"] == p]["accuracy"].tolist() for p in policies]

    plt.figure(figsize=(10, 4.8))
    bp = plt.boxplot(
        data,
        labels=policies,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6
    )

    # Color styling
    colors = ["#2ecc71" if i==0 else "#3498db" for i in range(len(policies))]
    for box, c in zip(bp["boxes"], colors):
        box.set_facecolor(c)
        box.set_edgecolor("black")
        box.set_linewidth(1)

    # Style medians/means
    for elem in ["medians", "means"]:
        for line in bp[elem]:
            line.set_color("black")
            line.set_linewidth(1.2)

    plt.ylabel("Accuracy (Final Probability)")
    plt.title("Accuracy Distribution by Policy (Ensemble Size = 3)")
    plt.xticks(rotation=25)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    out = BOXDATA_PATH.with_name("ensemble_size3_policy_boxplot.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.show()

if __name__ == "__main__":
    plot_box()
