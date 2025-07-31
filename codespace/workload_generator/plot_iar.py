import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data (if no header in CSV)
column_names = [
    "ID", "Deadline", "IAR", "RespTime","RunTime", "Model", "Accuracy", "Latency", "State", "Success",
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]
df = pd.read_csv("/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/iar_results_2.csv", header=None, names=column_names)

# Clean up (optional: ensure correct types)
df["IAR"] = df["IAR"].astype(float)
df["Success"] = df["Success"].astype(bool)

# Group by IAR value and calculate averages
df_grouped = df.groupby("IAR").agg(
    AvgRespTime=("RespTime", "mean"),
    AvgLatency=("Latency", "mean"),
    AvgAccuracy=("Accuracy", "mean"),
    SuccessRate=("Success", "mean")  # % of True values
).reset_index()

# 1. IAR vs Avg Response Time
sns.lineplot(data=df_grouped, x="IAR", y="AvgRespTime", marker="o")
plt.title("IAR vs Average Response Time")
plt.xlabel("IAR (ms)")
plt.ylabel("Avg Response Time (s)")
plt.grid(True)
plt.show()

df = df[df["Model"].notna()]

model_counts = df.groupby(["IAR", "Model"]).size().unstack().fillna(0)

# Plot
model_counts.plot(kind="bar", stacked=True, figsize=(10,6), colormap="tab20")
plt.title("Model Selection vs IAR (Inter-Arrival Rate)")
plt.xlabel("IAR (ms)")
plt.ylabel("Number of Requests")
plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.grid(True)
plt.tight_layout()
plt.show()

# Group by deadline, then get avg wait for each model
model_waits = df.groupby("IAR")[[
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]].mean()

model_waits.plot(marker='o', figsize=(10,6))
plt.title("Estimated Wait Time vs IAR (per Model)")
plt.xlabel("IAR (ms)")
plt.ylabel("Estimated Wait Time (us)")
plt.grid(True)
plt.legend(title="Model")
plt.tight_layout()
plt.show()

