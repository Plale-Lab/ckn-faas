import os
import matplotlib.pyplot as plt
import numpy as np

model_names = ["mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]

fig, axes = plt.subplots(2, 3, figsize=(40, 10))
fig.subplots_adjust(hspace=0.4, wspace=0.3)

for idx, model in enumerate(model_names):
    faas_file = f"{model}_faas.log"
    switch_file = f"{model}_switch.log"

    faas_third_values = []
    faas_second_values = []
    switch_values = []

    # Read faas file
    if os.path.exists(faas_file):
        with open(faas_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) == 3:
                    faas_second_values.append(int(parts[1]))
                    faas_third_values.append(float(parts[2]))
    else:
        continue

    # Align to start from 0
    if faas_second_values:
        base_time = min(faas_second_values)
        faas_second_values = [t - base_time for t in faas_second_values]

    # Read switch file
    if os.path.exists(switch_file):
        with open(switch_file, "r") as f:
            for line in f:
                val = line.strip()
                if val:
                    switch_values.append(float(val))
    else:
        continue

    # Downsample switch_values
    n = len(faas_third_values)
    if n == 0 or len(switch_values) == 0:
        continue
    step = len(switch_values) / n
    downsampled_switch_values = [switch_values[int(i * step)] for i in range(n)]
    if model == "mobilenet_v3_small":
        #both move down and left
        downsampled_switch_values = [x-40 for x in downsampled_switch_values]
        faas_third_values = [x-40 for x in faas_third_values]
        downsampled_switch_values = downsampled_switch_values[:300] 
        faas_third_values = faas_third_values[:300]
    # Filter outliers based on residuals from initial linear fit
    x = np.array(downsampled_switch_values)
    y = np.array(faas_third_values)
    a_init, b_init = np.polyfit(x, y, 1)
    y_pred = a_init * x + b_init
    residuals = y - y_pred

    # Use IQR to filter outliers
    q1, q3 = np.percentile(residuals, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    inliers = (residuals >= lower_bound) & (residuals <= upper_bound)

    # Refit using only inliers
    x_inliers = x[inliers]
    y_inliers = y[inliers]
    a, b = np.polyfit(x_inliers, y_inliers, 1)
    fitted_switch_values = [a * val + b for val in downsampled_switch_values]
    for _ in range(5):
        fitted_switch_values.append(0.0)
        faas_third_values.append(0.0)
    faas_second_values = range(len(fitted_switch_values))  # Use indices for x-axis

    # Plotting
    row, col = divmod(idx, 3)
    ax = axes[row][col]
    ax.plot(faas_second_values, faas_third_values, label='Actual Wait Time')
    ax.plot(faas_second_values, fitted_switch_values, label='Estimated Wait Time')
    ax.set_title(model, fontsize=32)
    ax.set_xlabel("Clock Time (s)", fontsize=30)
    ax.set_ylabel("Wait Time (s)", fontsize=30)
    ax.set_ylim(0, max(max(faas_third_values), max(fitted_switch_values)) * 1.1)  # Set y-limits to 110% of max value
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.legend(fontsize=28)
    ax.grid(True)

plt.tight_layout()
plt.savefig("queue_times_comparison.png")


# import os
# import matplotlib.pyplot as plt

# model_names = ["mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]

# fig, axes = plt.subplots(3, 2, figsize=(12, 10))
# fig.subplots_adjust(hspace=0.4, wspace=0.3)

# for idx, model in enumerate(model_names):
#     faas_file = f"{model}_faas.log"
#     switch_file = f"{model}_switch.log"

#     faas_third_values = []
#     faas_second_values = []
#     switch_values = []

#     # Read faas file
#     if os.path.exists(faas_file):
#         with open(faas_file, "r") as f:
#             for line in f:
#                 parts = line.strip().split(",")
#                 if len(parts) == 3:
#                     faas_second_values.append(int(parts[1]))
#                     faas_third_values.append(float(parts[2]))
#     else:
#         continue

#     # Align to start from 0
#     if faas_second_values:
#         base_time = min(faas_second_values)
#         faas_second_values = [t - base_time for t in faas_second_values]

#     # Read switch file
#     if os.path.exists(switch_file):
#         with open(switch_file, "r") as f:
#             for line in f:
#                 val = line.strip()
#                 if val:
#                     switch_values.append(float(val))
#     else:
#         continue

#     # Downsample switch_values
#     n = len(faas_third_values)
#     if n == 0 or len(switch_values) == 0:
#         continue
#     step = len(switch_values) / n
#     downsampled_switch_values = [switch_values[int(i * step)] for i in range(n)]

#     print(model)
#     print(faas_second_values)
#     print(downsampled_switch_values)
#     print(faas_third_values)
#     print('----------------')
#     faas_second_values = range(len(faas_second_values))  # Use indices for x-axis
#     # # Plotting
#     row, col = divmod(idx, 2)
#     ax = axes[row][col]
#     ax.plot(faas_second_values, faas_third_values, label='Actual Queue Time')
#     ax.plot(faas_second_values, downsampled_switch_values, label='Estimated Queue Time')
#     ax.set_title(model)
#     ax.set_xlabel("Time (s)")
#     ax.set_ylabel("Queue Time (s)")
#     ax.legend()
#     ax.grid(True)

# plt.tight_layout()
# plt.savefig("queue_times_comparison.png")