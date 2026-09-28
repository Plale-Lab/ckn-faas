import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------

TRACE_FILE = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

# Largest application from your dataset
APP_ID = "a594f92f84072b4cd031fe5283d1781a6e98f430696dec0a8e3b02eadb5fc0b8"

# --------------------------------------------------
# Read trace
# --------------------------------------------------

print("Loading trace...")

df = pd.read_csv(TRACE_FILE)

print("Total requests:", len(df))

# --------------------------------------------------
# Select one application
# --------------------------------------------------

app_df = df[df["app"] == APP_ID].copy()

print("Requests in selected application:", len(app_df))

# --------------------------------------------------
# Compute arrival timestamps
# --------------------------------------------------

app_df["arrival_time"] = (
    app_df["end_timestamp"] - app_df["duration"]
)

app_df = app_df.sort_values("arrival_time")

# --------------------------------------------------
# Compute inter-arrival times (IAT)
# --------------------------------------------------

app_df["iat"] = app_df["arrival_time"].diff()

print("\nInter-arrival statistics")

print(app_df["iat"].describe())

# --------------------------------------------------
# Requests per second
# --------------------------------------------------

app_df["second"] = app_df["arrival_time"].astype(int)

requests_per_second = (
    app_df.groupby("second")
          .size()
)

# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(12,5))

plt.plot(
    requests_per_second.index,
    requests_per_second.values,
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Requests / second")
plt.title("Azure Application Workload")

plt.grid(True)

plt.tight_layout()

plt.show()