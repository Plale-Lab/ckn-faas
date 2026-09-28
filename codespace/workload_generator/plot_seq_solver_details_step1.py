# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_solver_details_step1.csv"
# OUT_DIR = "seq_solver_analysis"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# # Create numeric request index from REQ-00000
# df["req_index"] = (
#     df["ID"].astype(str)
#     .str.extract(r"(\d+)")
#     .astype(float)
# )
#
# numeric_cols = [
#     "selected_theta",
#     "seq_solver_latency_ms",
#     "seq_solver_feasible_count",
#     "seq_solver_cost",
#     "seq_solver_accuracy",
#     "seq_solver_expected_latency_ms",
#     "e2e_time_ms",
# ]
#
# for col in numeric_cols:
#     df[col] = pd.to_numeric(df[col], errors="coerce")
#
# df = df.dropna(subset=["req_index"])
#
# print(df[[
#     "ID",
#     "selected_theta",
#     "seq_solver_latency_ms",
#     "seq_solver_feasible_count",
#     "seq_solver_cost",
#     "seq_solver_accuracy",
#     "seq_solver_expected_latency_ms",
#     "e2e_time_ms",
# ]].head())
#
# # 1. Selected theta
# plt.figure(figsize=(10, 5))
# plt.plot(df["req_index"], df["selected_theta"], marker="o")
# plt.xlabel("Request")
# plt.ylabel("Selected θ")
# plt.title("Selected Threshold θ Over Requests")
# plt.grid(True, alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/selected_theta_over_requests.png", dpi=300)
# plt.show()
#
# # 2. Solver latency
# plt.figure(figsize=(10, 5))
# plt.plot(df["req_index"], df["seq_solver_latency_ms"], marker="o")
# plt.xlabel("Request")
# plt.ylabel("Solver Latency (ms)")
# plt.title("Constraint Solver Latency Over Requests")
# plt.grid(True, alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/solver_latency_over_requests.png", dpi=300)
# plt.show()
#
# # 3. Expected vs actual latency
# plt.figure(figsize=(10, 5))
# plt.plot(df["req_index"], df["seq_solver_expected_latency_ms"], marker="o", label="CSV Expected Latency")
# plt.plot(df["req_index"], df["e2e_time_ms"], marker="o", label="Actual E2E Latency")
# plt.xlabel("Request")
# plt.ylabel("Latency (ms)")
# plt.title("Expected Solver Latency vs Actual Runtime Latency")
# plt.legend()
# plt.grid(True, alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/expected_vs_actual_latency.png", dpi=300)
# plt.show()
#
# # 4. Feasible count
# plt.figure(figsize=(10, 5))
# plt.plot(df["req_index"], df["seq_solver_feasible_count"], marker="o")
# plt.xlabel("Request")
# plt.ylabel("Feasible Configurations")
# plt.title("Feasible Configurations Over Requests")
# plt.grid(True, alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/feasible_count_over_requests.png", dpi=300)
# plt.show()
#
# # 5. Cost vs accuracy
# plt.figure(figsize=(7, 6))
# plt.scatter(df["seq_solver_cost"], df["seq_solver_accuracy"])
# plt.xlabel("Selected Relative Compute Cost")
# plt.ylabel("Selected Cascade Accuracy")
# plt.title("Selected Solver Configuration: Cost vs Accuracy")
# plt.grid(True, alpha=0.4)
# plt.tight_layout()
# plt.savefig(f"{OUT_DIR}/selected_cost_vs_accuracy.png", dpi=300)
# plt.show()
#
# print(f"Saved plots to {OUT_DIR}")


import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_solver_details_step1.csv"
OUT_DIR = "constraint_solver_analysis"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

df["req_index"] = (
    df["ID"].astype(str)
    .str.extract(r"(\d+)")
    .astype(float)
)

def pick_col(new_col, old_col=None):
    if new_col in df.columns:
        return new_col
    if old_col and old_col in df.columns:
        return old_col
    return None

COL_SOLVER_LAT = pick_col("constraint_solver_latency_ms", "seq_solver_latency_ms")
COL_FEASIBLE = pick_col("constraint_solver_feasible_count", "seq_solver_feasible_count")
COL_COST = pick_col("constraint_solver_cost", "seq_solver_cost")
COL_ACC = pick_col("constraint_solver_accuracy", "seq_solver_accuracy")
COL_EXPECTED_LAT = pick_col("constraint_solver_expected_latency_ms", "seq_solver_expected_latency_ms")
COL_OBJECTIVE_LAT = pick_col("constraint_solver_objective_latency_ms", "seq_solver_expected_latency_ms")
COL_MAX_COST = pick_col("constraint_solver_max_cost")
COL_TARGET_ACC = pick_col("constraint_solver_target_accuracy")

numeric_cols = [
    "selected_theta",
    "e2e_time_ms",
    "RespTime",
    COL_SOLVER_LAT,
    COL_FEASIBLE,
    COL_COST,
    COL_ACC,
    COL_EXPECTED_LAT,
    COL_OBJECTIVE_LAT,
    COL_MAX_COST,
    COL_TARGET_ACC,
]

for col in numeric_cols:
    if col and col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["req_index"])

if "RespTime" in df.columns:
    df["runtime_latency_ms"] = pd.to_numeric(df["RespTime"], errors="coerce") * 1000.0
else:
    df["runtime_latency_ms"] = pd.to_numeric(df["e2e_time_ms"], errors="coerce")

available_print_cols = [
    c for c in [
        "ID",
        "selected_theta",
        COL_MAX_COST,
        COL_TARGET_ACC,
        COL_COST,
        COL_ACC,
        COL_OBJECTIVE_LAT,
        "runtime_latency_ms",
    ]
    if c and c in df.columns
]

print(df[available_print_cols].head())

# 1. Selected theta
plt.figure(figsize=(10, 5))
plt.plot(df["req_index"], df["selected_theta"], marker="o")
plt.xlabel("Request")
plt.ylabel("Selected θ")
plt.title("Selected Threshold θ Over Requests")
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/selected_theta_over_requests.png", dpi=300)
plt.show()

# 2. Solver latency
if COL_SOLVER_LAT:
    plt.figure(figsize=(10, 5))
    plt.plot(df["req_index"], df[COL_SOLVER_LAT], marker="o")
    plt.xlabel("Request")
    plt.ylabel("Solver Latency (ms)")
    plt.title("Constraint Solver Latency Over Requests")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/solver_latency_over_requests.png", dpi=300)
    plt.show()

# 3. Expected/objective vs runtime latency
if COL_OBJECTIVE_LAT:
    plt.figure(figsize=(10, 5))
    plt.plot(df["req_index"], df[COL_OBJECTIVE_LAT], marker="o", label="Solver Objective Latency")
    plt.plot(df["req_index"], df["runtime_latency_ms"], marker="o", label="Actual Runtime Latency")
    plt.xlabel("Request")
    plt.ylabel("Latency (ms)")
    plt.title("Solver Objective Latency vs Actual Runtime Latency")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/objective_vs_actual_latency.png", dpi=300)
    plt.show()

# 4. Feasible count
if COL_FEASIBLE:
    plt.figure(figsize=(10, 5))
    plt.plot(df["req_index"], df[COL_FEASIBLE], marker="o")
    plt.xlabel("Request")
    plt.ylabel("Feasible Configurations")
    plt.title("Feasible Configurations Over Requests")
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/feasible_count_over_requests.png", dpi=300)
    plt.show()

# 5. Cost vs accuracy
if COL_COST and COL_ACC:
    plt.figure(figsize=(7, 6))
    plt.scatter(df[COL_COST], df[COL_ACC], label="Selected configurations")

    if COL_TARGET_ACC and df[COL_TARGET_ACC].notna().any():
        target_acc = df[COL_TARGET_ACC].dropna().iloc[0]
        plt.axhline(target_acc, linestyle="--", label=f"Target Accuracy = {target_acc:.2f}")

    if COL_MAX_COST and df[COL_MAX_COST].notna().any():
        max_cost = df[COL_MAX_COST].dropna().iloc[0]
        plt.axvline(max_cost, linestyle="--", label=f"Max Cost = {max_cost:.2f}")

    plt.xlabel("Selected Relative Compute Cost")
    plt.ylabel("Selected Cascade Accuracy")
    plt.title("Selected Configuration: Cost vs Accuracy")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/selected_cost_vs_accuracy.png", dpi=300)
    plt.show()

print(f"Saved plots to {OUT_DIR}")
