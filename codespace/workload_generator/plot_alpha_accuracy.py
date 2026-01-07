import csv
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/alpha_wise_accuracy.csv"

rows = []

with open(csv_path, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)                 # skip header row
    acc_idx = header.index("Accuracy")    # position of Accuracy column

    for r in reader:
        if not r:
            continue
        try:
            acc = float(r[acc_idx])
            alpha = float(r[-1])          # last value in the row
            rows.append((alpha, acc))
        except ValueError:
            # skip bad rows if any
            continue

# Create DataFrame with just alpha and accuracy
df = pd.DataFrame(rows, columns=["alpha", "Accuracy"])

# Group by alpha and compute mean accuracy
grouped = df.groupby("alpha")["Accuracy"].mean().reset_index()
grouped = grouped.sort_values("alpha")

print("\nAlpha vs mean accuracy:")
print(grouped)

# Plot
plt.figure(figsize=(8, 5))
plt.plot(grouped["alpha"], grouped["Accuracy"], marker="o")
plt.xlabel("Alpha")
plt.ylabel("Average Accuracy")
plt.title("Alpha vs Average Accuracy")
plt.grid(True)
plt.tight_layout()
plt.show()
