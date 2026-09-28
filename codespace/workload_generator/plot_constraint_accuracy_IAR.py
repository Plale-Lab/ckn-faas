import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/constrain_solver_latency_n_IAR_2.csv"
LABEL_JSON_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/Labels.json"

OUT_DIR = "constraint_solver_plots"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

with open(LABEL_JSON_PATH, "r") as f:
    label_map = json.load(f)

numeric_cols = [
    "IAR",
    "RespTime",
    "constraint_solver_max_cost",
    "constraint_solver_target_accuracy",
    "constraint_solver_cost",
    "constraint_solver_accuracy",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["IAR", "RespTime"])

# RespTime is seconds, so convert to ms
df["latency_ms"] = df["RespTime"] * 1000.0

# IAR is ms, so arrival rate = 1000 / IAR
df["arrival_rate_rps"] = 1000.0 / df["IAR"]

# -----------------------------
# Measured accuracy using Labels.json
# -----------------------------
def normalize_text(s):
    return str(s).lower().replace("_", " ").replace("-", " ").strip()


def is_correct(row):
    pred = normalize_text(row.get("label", ""))
    wnid = str(row.get("selected_folder", "")).strip()

    gt_names = label_map.get(wnid, "")
    gt_parts = [normalize_text(x) for x in str(gt_names).split(",")]

    return any(pred == gt or pred in gt or gt in pred for gt in gt_parts)


df["measured_accuracy"] = df.apply(is_correct, axis=1).astype(float)

# -----------------------------
# Create group + label
# -----------------------------
def make_plot_group(row):
    solver_success = str(row.get("constraint_solver_success", "")).lower()

    if solver_success == "true":
        return "Constraint Solver"

    return "Baseline"


def make_plot_label(row):
    solver_success = str(row.get("constraint_solver_success", "")).lower()

    if solver_success == "true":
        c = row.get("constraint_solver_max_cost")
        a = row.get("constraint_solver_target_accuracy")
        return f"C={c:.2f}, A={a:.2f}"

    baseline = str(row.get("baseline_mode", "")).strip()
    mode = str(row.get("effective_cascade_mode", "")).strip()

    if baseline and baseline.lower() != "none":
        return baseline

    return mode


df["plot_group"] = df.apply(make_plot_group, axis=1)
df["plot_label"] = df.apply(make_plot_label, axis=1)

# -----------------------------
# Aggregate
# -----------------------------
summary = (
    df.groupby(["arrival_rate_rps", "IAR", "plot_group", "plot_label"], as_index=False)
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        median_latency_ms=("latency_ms", "median"),
        p95_latency_ms=("latency_ms", lambda x: x.quantile(0.95)),
        measured_accuracy=("measured_accuracy", "mean"),
        mean_selected_cost=("constraint_solver_cost", "mean"),
        mean_profile_accuracy=("constraint_solver_accuracy", "mean"),
        count=("latency_ms", "count"),
    )
)

summary.to_csv(f"{OUT_DIR}/summary_by_arrival_rate.csv", index=False)

# -----------------------------
# Plot helper with grouped legend
# -----------------------------
def plot_metric(
    x_col,
    y_col,
    x_label,
    y_label,
    title,
    filename,
    ylim=None,
):
    plt.figure(figsize=(11, 6))

    ax = plt.gca()
    handles = []
    labels = []

    for group_name in ["Constraint Solver", "Baseline"]:
        group_df = summary[summary["plot_group"] == group_name]

        if group_df.empty:
            continue

        # Add group heading in legend
        handles.append(Line2D([], [], color="none"))
        labels.append(f"{group_name}:")

        for plot_label, g in group_df.groupby("plot_label"):
            g = g.sort_values(x_col)

            line, = ax.plot(
                g[x_col],
                g[y_col],
                marker="o",
                linewidth=2,
                label=plot_label,
            )

            handles.append(line)
            labels.append(f"  {plot_label}")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    if ylim:
        ax.set_ylim(*ylim)

    ax.grid(True, alpha=0.4)
    ax.legend(handles, labels, loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/{filename}", dpi=300)
    plt.show()


# -----------------------------
# Correct graph: Latency vs Arrival Rate
# -----------------------------
plot_metric(
    x_col="arrival_rate_rps",
    y_col="mean_latency_ms",
    x_label="Arrival Rate (requests/second)",
    y_label="Mean Latency (ms)",
    title="Mean Latency vs Arrival Rate",
    filename="mean_latency_vs_arrival_rate.png",
)

# -----------------------------
# Optional: Latency vs Inter-arrival Time
# This uses the SAME latency value, only x-axis changes
# -----------------------------
plot_metric(
    x_col="IAR",
    y_col="mean_latency_ms",
    x_label="Inter-arrival Time (ms)",
    y_label="Mean Latency (ms)",
    title="Mean Latency vs Inter-arrival Time",
    filename="mean_latency_vs_interarrival_time.png",
)

# -----------------------------
# P95 latency vs arrival rate
# -----------------------------
plot_metric(
    x_col="arrival_rate_rps",
    y_col="p95_latency_ms",
    x_label="Arrival Rate (requests/second)",
    y_label="P95 Latency (ms)",
    title="P95 Latency vs Arrival Rate",
    filename="p95_latency_vs_arrival_rate.png",
)

# -----------------------------
# Measured accuracy vs arrival rate
# -----------------------------
plot_metric(
    x_col="arrival_rate_rps",
    y_col="measured_accuracy",
    x_label="Arrival Rate (requests/second)",
    y_label="Measured Accuracy",
    title="Measured Accuracy vs Arrival Rate",
    filename="measured_accuracy_vs_arrival_rate.png",
    ylim=(0, 1.05),
)

# -----------------------------
# Selected cost vs arrival rate
# -----------------------------
solver_summary = summary[summary["plot_group"] == "Constraint Solver"]

if not solver_summary.empty:
    plt.figure(figsize=(11, 6))
    ax = plt.gca()

    handles = [Line2D([], [], color="none")]
    labels = ["Constraint Solver:"]

    for plot_label, g in solver_summary.groupby("plot_label"):
        g = g.sort_values("arrival_rate_rps")

        line, = ax.plot(
            g["arrival_rate_rps"],
            g["mean_selected_cost"],
            marker="o",
            linewidth=2,
            label=plot_label,
        )

        handles.append(line)
        labels.append(f"  {plot_label}")

    ax.set_xlabel("Arrival Rate (requests/second)")
    ax.set_ylabel("Mean Selected Relative Cost")
    ax.set_title("Selected Cost vs Arrival Rate")
    ax.grid(True, alpha=0.4)
    ax.legend(handles, labels, loc="best")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/selected_cost_vs_arrival_rate.png", dpi=300)
    plt.show()

print(f"Saved plots and summary CSV to: {OUT_DIR}")