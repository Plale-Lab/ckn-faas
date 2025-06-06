import os
import matplotlib.pyplot as plt
import numpy as np

model_names = ["mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]
accuracy_distributions = []

for model in model_names:
    faas_file = f"{model}_faas.log"
    switch_file = f"{model}_switch.log"

    faas_third_values = []
    faas_second_values = []
    switch_values = []

    if os.path.exists(faas_file):
        with open(faas_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    faas_second_values.append(int(parts[1]))
                    faas_third_values.append(float(parts[2]))
    else:
        continue

    if faas_second_values:
        base_time = min(faas_second_values)
        faas_second_values = [t - base_time for t in faas_second_values]

    if os.path.exists(switch_file):
        with open(switch_file, "r") as f:
            for line in f:
                val = line.strip()
                if val:
                    switch_values.append(float(val))
    else:
        continue

    n = len(faas_third_values)
    if n == 0 or len(switch_values) == 0:
        continue

    step = len(switch_values) / n
    downsampled_switch_values = [switch_values[int(i * step)] for i in range(n)]

    if model == "mobilenet_v3_small":
        downsampled_switch_values = [x - 40 for x in downsampled_switch_values]
        faas_third_values = [x - 40 for x in faas_third_values]
        downsampled_switch_values = downsampled_switch_values[:300]
        faas_third_values = faas_third_values[:300]

    x = np.array(downsampled_switch_values)
    y = np.array(faas_third_values)
    a_init, b_init = np.polyfit(x, y, 1)
    y_pred = a_init * x + b_init
    residuals = y - y_pred

    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inliers = (residuals >= lower_bound) & (residuals <= upper_bound)

    x_inliers = x[inliers]
    y_inliers = y[inliers]
    a, b = np.polyfit(x_inliers, y_inliers, 1)
    fitted_switch_values = [a * val + b for val in downsampled_switch_values]

    # Append five zeros for alignment
    fitted_switch_values.extend([0.0] * 5)
    faas_third_values.extend([0.0] * 5)

    # Compute relative acc (percentage)
    accuracies = []
    for actual, predicted in zip(faas_third_values, fitted_switch_values):
        if actual != 0:
            acc = (1-(abs(actual - predicted) / actual)) * 100
            accuracies.append(acc)

    accuracy_distributions.append(accuracies)

# === Final Box Plot ===
model_names_s = ["mobilenet", "resnet18", "resnet34", "resnet50", "resnet101", "vit"]
plt.figure(figsize=(14, 8))
plt.boxplot(
    accuracy_distributions,
    labels=model_names_s,
    patch_artist=True,
    showfliers=False,
    boxprops=dict(facecolor='white', color='darkgreen', linewidth=4),
    medianprops=dict(color='red', linewidth=4),
    whiskerprops=dict(color='darkgreen', linewidth=4),
    capprops=dict(color='darkgreen', linewidth=4)
)
plt.ylabel("Estimation Accuracy (%)", fontsize=32)
plt.xlabel("Model", fontsize=32)
# plt.title("Queue Time Estimation Accuracy per Model", fontsize=18)
plt.xticks(rotation=45, fontsize=28)
plt.yticks(np.arange(70, 105, 5), fontsize=28)  # <-- this controls tick density
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("accuracy_boxplot.png")