import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_parallel_newset_results.csv"
OUTPUT_DIR = "seq_cascade_graphs"

TARGET_ACCURACY = 0.90
MAX_RELATIVE_COST = 0.80

EDGE_MODELS = [
    "mobilenetv3_small_100",
    "mobilenetv2_100",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
]

MODEL_NAME_MAP = {
    "mobilenetv3_small_100": "mobilenet_v3_small",
    "mobilenetv2_100": "mobilenet_v2",
    "shufflenet_v2_x1_0": "shufflenet_v2_x1_0",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "efficientnet_b3": "efficientnet_b3",
    "convnext_tiny": "convnext_tiny",
    "vit_base_patch16_224": "vit_b_16",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Normalize model names
df["a_raw"] = df["a"].astype(str).str.strip()
df["b_raw"] = df["b"].astype(str).str.strip()

df["a"] = df["a_raw"].map(MODEL_NAME_MAP).fillna(df["a_raw"])
df["b"] = df["b_raw"].map(MODEL_NAME_MAP).fillna(df["b_raw"])

numeric_cols = [
    "theta",
    "cascade_accuracy",
    "a_accept_rate",
    "relative_compute_cost",
    "expected_latency_cost_ms",
    "relative_latency_cost",
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_cols)

df["cascade_pair"] = df["a"] + " → " + df["b"]

# -----------------------------
# Feasible region condition
# -----------------------------
df["feasible"] = (
    (df["cascade_accuracy"] >= TARGET_ACCURACY)
    & (df["relative_compute_cost"] <= MAX_RELATIVE_COST)
)

feasible_df = df[df["feasible"]].copy()

best = None
if not feasible_df.empty:
    best = feasible_df.sort_values("expected_latency_cost_ms").iloc[0]

    print("\nBest feasible configuration:")
    print("a:", best["a"])
    print("b:", best["b"])
    print("theta:", best["theta"])
    print("cascade_accuracy:", best["cascade_accuracy"])
    print("a_accept_rate:", best["a_accept_rate"])
    print("relative_compute_cost:", best["relative_compute_cost"])
    print("expected_latency_cost_ms:", best["expected_latency_cost_ms"])
else:
    print("\nNo feasible configuration found.")


# -----------------------------
# Graph 1: Cascade Accuracy Heatmap
# -----------------------------
acc_pivot = df.pivot_table(
    index="cascade_pair",
    columns="theta",
    values="cascade_accuracy",
    aggfunc="mean",
)

plt.figure(figsize=(14, max(6, len(acc_pivot) * 0.35)))
plt.imshow(acc_pivot, aspect="auto")
plt.colorbar(label="Cascade Accuracy")

plt.xticks(
    range(len(acc_pivot.columns)),
    [f"{x:.2f}" for x in acc_pivot.columns],
    rotation=45,
    ha="right",
)
plt.yticks(
    range(len(acc_pivot.index)),
    acc_pivot.index,
)

plt.xlabel("Threshold θ")
plt.ylabel("Cascade Pair")
plt.title("Cascade Accuracy Across Thresholds")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph1_cascade_accuracy_heatmap.png", dpi=300)
plt.show()


# -----------------------------
# Graph 2: Edge Hit Rate Line Plot
# -----------------------------
edge_df = df[df["a"].isin([
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
])].copy()

# Average across b models for each first-stage edge model
edge_summary = (
    edge_df
    .groupby(["a", "theta"], as_index=False)["a_accept_rate"]
    .mean()
)

plt.figure(figsize=(9, 6))

for model_name, group in edge_summary.groupby("a"):
    group = group.sort_values("theta")
    plt.plot(
        group["theta"],
        group["a_accept_rate"],
        marker="o",
        linewidth=2,
        label=model_name,
    )

plt.xlabel("Threshold θ")
plt.ylabel("Edge Hit Rate / Accept Rate")
plt.title("Edge Hit Rate vs Threshold for First-Stage Edge Models")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph2_edge_hit_rate_line.png", dpi=300)
plt.show()


# -----------------------------
# Graph 3: Feasible Region
# -----------------------------
plt.figure(figsize=(9, 6))

infeasible = df[~df["feasible"]]
feasible = df[df["feasible"]]

plt.scatter(
    infeasible["relative_compute_cost"],
    infeasible["cascade_accuracy"],
    label="Infeasible",
    alpha=0.45,
    s=35,
)

plt.scatter(
    feasible["relative_compute_cost"],
    feasible["cascade_accuracy"],
    label="Feasible",
    alpha=0.85,
    s=50,
)

plt.axhline(
    TARGET_ACCURACY,
    linestyle="--",
    label=f"Target accuracy = {TARGET_ACCURACY}",
)

plt.axvline(
    MAX_RELATIVE_COST,
    linestyle="--",
    label=f"Max cost = {MAX_RELATIVE_COST}",
)

if best is not None:
    plt.scatter(
        best["relative_compute_cost"],
        best["cascade_accuracy"],
        marker="*",
        s=300,
        label="Selected optimum",
        edgecolors="black",
    )

    label = (
        f"{best['a']} → {best['b']}\n"
        f"θ={best['theta']:.2f}"
    )

    plt.annotate(
        label,
        xy=(best["relative_compute_cost"], best["cascade_accuracy"]),
        xytext=(12, 12),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
    )

plt.xlabel("Relative Compute Cost")
plt.ylabel("Cascade Accuracy")
plt.title("Feasible Region: Accuracy vs Relative Compute Cost")
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/graph3_feasible_region.png", dpi=300)
plt.show()


# -----------------------------
# Save outputs
# -----------------------------
df.to_csv(f"{OUTPUT_DIR}/processed_sequential_cascade_profile.csv", index=False)
feasible_df.to_csv(f"{OUTPUT_DIR}/feasible_sequential_cascades.csv", index=False)

print(f"\nSaved graphs and CSV outputs to: {OUTPUT_DIR}")