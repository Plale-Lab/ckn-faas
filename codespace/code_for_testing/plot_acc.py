import json
import matplotlib.pyplot as plt
import numpy as np

# Custom order for plotting
custom_order = [
    "densenet201",
    "mobilenet_v3_small",
    "googlenet",
    "shufflenet_v2_x0_5",  # assuming shufflenet maps to this key
    "resnet152",
    "resnext50_32x4d",
]

# Load model probabilities
with open("model_probabilities.json", "r") as f:
    data = json.load(f)

model_names = []
means = []
stds = []

for model in custom_order:
    if model not in data or not data[model]:
        continue
    probs = np.array(data[model])
    model_names.append(model)
    means.append(np.mean(probs))
    stds.append(np.std(probs))

means = [m * 100 for m in means]
stds = [s * 100 for s in stds]

# Plot
import matplotlib.pyplot as plt
import numpy as np

x = np.arange(len(model_names))
plt.figure(figsize=(10, 10))
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})
# plt.bar(x, means, yerr=stds, capsize=5)
plt.bar(
    x,
    means,
    yerr=stds,
    capsize=5,
    edgecolor='blue',
    linewidth=2,
    facecolor='none'  # transparent bars
)
plt.xticks(x, model_names, rotation=45, ha="right")
plt.ylabel("Average Probability")
plt.ylim(0, 100)
plt.yticks(np.arange(0, 110, 10))  # 0 to 100 with step 10
plt.title("Model Prediction Accuracy (Mean ± Std)")
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("model_accuracy_plot.png")
plt.show()