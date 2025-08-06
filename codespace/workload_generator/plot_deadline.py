import ast

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


column_names = [
    "ID", "Deadline", "IAR", "RespTime", "RunTime", "Model", "Accuracy", "Latency", "State", "Success", "selected_models", "cost_function_execution_time_ms", "status",
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait","run_time"
]

# Load the CSV
df = pd.read_csv("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/deadline_results_I1_D30.csv", header=None, names=column_names)

# Convert string booleans to actual booleans if needed
df["Success"] = df["Success"].astype(bool)

# 1. Response Time vs Deadline
# sns.scatterplot(data=df, x="Deadline", y="RespTime")
# plt.title("Response Time vs Deadline")
# plt.xlabel("Deadline (s)")
# plt.ylabel("Response Time (s)")
# plt.grid(True)
# plt.show()

# 1. Response Time vs IAR
# sns.scatterplot(data=df, x="IAR", y="RespTime")
# plt.title("Response Time vs IAR")
# plt.xlabel("IAR (s)")
# plt.ylabel("Response Time (s)")
# plt.grid(True)
# plt.show()

df["RespTime_sec"] = df["RespTime"]
# df["RespTime_ms"] = df["RespTime"] * 1000

# Group by Deadline and calculate mean response time
grouped = df.groupby("Deadline")["RespTime_sec"].mean()

# Plot
plt.figure(figsize=(8, 5))
grouped.plot(marker='o', linestyle='-')
plt.title("Average Response Time vs Deadline")
plt.xlabel("Deadline (ms)")
plt.ylabel("Average Response Time (s)")
plt.grid(True)
plt.tight_layout()
plt.show()

df = df[df["Model"].notna()]

# Count how many times each model was selected for each deadline
model_counts = df.groupby(["Deadline", "Model"]).size().unstack().fillna(0)

# Plot
model_counts.plot(kind="bar", stacked=True, figsize=(10,6))
plt.title("Model Selection vs Deadline")
plt.xlabel("Deadline (ms)")
plt.ylabel("Number of Requests")
plt.legend(title="Model")
plt.grid(True)
plt.tight_layout()
plt.show()



# Group by deadline, then get avg wait for each model
model_waits = df.groupby("Deadline")[[
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]].mean()

model_waits.plot(marker='o', figsize=(10,6))
plt.title("Estimated Wait Time vs Deadline (per Model)")
plt.xlabel("Deadline (ms)")
plt.ylabel("Estimated Wait Time (us)")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.show()


# Convert run_time to seconds (if not already)
# df["run_time"] = pd.to_numeric(df["run_time"]) / 1000.0

# Plot wait time over time for each model
# === Option 1: Sort by Response Time ===
df_sorted = df.sort_values(by="RespTime", ascending=True).reset_index(drop=True)
fixed_deadline = df_sorted["Deadline"].iloc[0]
fixed_iar = df_sorted["IAR"].iloc[0]

# === Wait time columns ===
wait_cols = [
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]

# === Plot: Estimated Wait Time vs. Response Time (per Model) ===
plt.figure(figsize=(12, 6))
for col in wait_cols:
    plt.plot(df_sorted["RespTime"], df_sorted[col], label=col.replace("_wait", ""), linewidth=1.5)

plt.title("Total Estimated Time vs Runtime (per Model)")
plt.xlabel("Runtime (s)")
plt.ylabel("Total Estimated Time (s)")  # adjust units as needed

plt.grid(True)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
# Add deadline + IAR info to upper-left corner
# text_str = f"Deadline: {fixed_deadline} ms\nIAR: {fixed_iar} ms"
# plt.text(
#     1.06,  0.6, text_str,
#     transform=plt.gca().transAxes,
#     fontsize=10,
#     verticalalignment='top',
#     horizontalalignment='left',
#     bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
# )

plt.tight_layout()
plt.show()



df["MS_size"] = df["selected_models"].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)

ms_size_avg = df.groupby(["IAR", "Deadline"])["MS_size"].mean()

# Plot
ms_size_avg.plot(kind="bar", figsize=(10, 5))
plt.title("Average Model Set Size per Workload")
plt.ylabel("Average Model Set Size")
plt.xlabel("Workload (IAR, Deadline)")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

def safe_model_count(x):
    try:
        # Handle nulls, int -1, str "-1", etc.
        if pd.isnull(x) or x == -1 or str(x).strip() in ("", "-1", "nan"):
            return 0
        parsed = ast.literal_eval(str(x))
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0


df["MS_size"] = df["selected_models"].apply(safe_model_count)

print(df[["selected_models", "MS_size"]])

# Plot MS_size vs Response Time for each Deadline/IAR
grouped = df.groupby(["Deadline", "IAR"])
for (deadline, iar), group in grouped:
    group_sorted = group.sort_values("RunTime")

    plt.figure(figsize=(8, 5))
    plt.plot(group_sorted["RunTime"], group_sorted["MS_size"], marker='o', linestyle='-')

    # Set title and axis labels
    plt.title(f"Model Set Size vs Run Time\nDeadline: {deadline} ms, IAR: {iar} ms")
    plt.xlabel("Run Time (s)")
    plt.ylabel("Model Set Size")

    # Force y-axis to show integer ticks (e.g., 0 to max size)
    max_y = group_sorted["MS_size"].max()
    plt.yticks(np.arange(0, max_y + 1))  # Shows 0, 1, 2, ..., max

    plt.grid(True)
    plt.tight_layout()
    plt.show()


    # // new request

# Sort DataFrame by runtime
df_sorted = df.sort_values(by="RunTime", ascending=True).reset_index(drop=True)

# Get fixed parameters
fixed_deadline = df_sorted["Deadline"].iloc[0]
fixed_iar = df_sorted["IAR"].iloc[0]

# Wait time columns for each model
wait_cols = [
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]

# Filter skipped requests
skipped_df = df_sorted[df_sorted["State"] == "SKIPPED"]

# Get fixed deadline (assumed constant per experiment)
fixed_deadline_ms = df_sorted["Deadline"].iloc[0]
deadline_sec = fixed_deadline_ms / 1000.0

plt.figure(figsize=(14, 6))

# Plot model wait times
for col in wait_cols:
    model_name = col.replace("_wait", "")
    plt.plot(df_sorted["RunTime"], df_sorted[col], label=model_name, linewidth=1.8)

# Draw short vertical dashed lines for dropped requests up to deadline
for drop_time in skipped_df["RunTime"]:
    plt.plot(
        [drop_time, drop_time],
        [0, deadline_sec],
        color='tomato',
        linestyle='--',
        linewidth=1,
        dashes=(6, 6),
        # label="Dropped Request (SKIPPED)"
    )

plt.plot([], [], color='tomato', linestyle='--', linewidth=1, dashes=(6, 6), label="Dropped Request (SKIPPED)")

# Add horizontal line for deadline
plt.axhline(
    y=deadline_sec,
    color='black',
    linestyle='--',
    linewidth=1.5,
    label=f"Deadline = {deadline_sec:.0f}s"
)

# Final plot formatting
plt.title(f"Estimated Wait Time per Model vs Runtime with Dropped Requests (Deadline: {fixed_deadline:.0f} ms, IAR: {fixed_iar:.0f} ms)")
plt.xlabel("Runtime (s)")
plt.ylabel("Estimated Wait Time (s)")
plt.grid(True)
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()


# // req 2
# Define helper safely
def safe_model_count(x):
    try:
        if pd.isnull(x) or x == -1 or str(x).strip() in ("", "-1", "nan"):
            return 0
        parsed = ast.literal_eval(str(x))
        return len(parsed) if isinstance(parsed, list) else 0
    except Exception:
        return 0

# Compute model set size
df["MS_size"] = df["selected_models"].apply(safe_model_count)

# Group by Deadline and IAR
grouped = df.groupby(["Deadline", "IAR"])

# Plot for each group
for (deadline, iar), group in grouped:
    group_sorted = group.sort_values("RunTime")

    plt.figure(figsize=(10, 6))  # Wider figure for better readability
    plt.plot(
        group_sorted["RunTime"],
        group_sorted["MS_size"],
        marker='o',
        linestyle='--',  # <-- dotted line here
        linewidth=2,
        markersize=6,
        color="#228B22",
        label="Model Set Size"
    )

    # Titles and labels
    plt.title(f"Model Set Size vs Run Time\nDeadline: {deadline} ms, IAR: {iar} ms", fontsize=14)
    plt.xlabel("Run Time (s)", fontsize=12)
    plt.ylabel("Model Set Size", fontsize=12)

    # Integer ticks on y-axis
    max_y = group_sorted["MS_size"].max()
    plt.yticks(np.arange(0, max_y + 2), fontsize=10)
    plt.xticks(fontsize=10)

    # Grid and legend
    plt.grid(True, which='major', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.tight_layout()

    plt.show()


