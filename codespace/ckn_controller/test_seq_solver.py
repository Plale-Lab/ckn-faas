from sequential_cascade_constraint_solver import SequentialCascadeConstraintSolver

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_parallel_newset_results.csv"

solver = SequentialCascadeConstraintSolver(CSV_PATH)

result = solver.solve(
    max_cost=0.55,
    target_accuracy=0.82,
)

if result.success:
    print("Selected feasible cascade:")
else:
    print("No feasible cascade found.")

print("Reason:", result.reason)
print("Feasible count:", result.feasible_count)
print("Total candidates:", result.total_count)
print("Solver latency ms:", result.solver_latency_ms)
print("Selected config:")
print(result.selected_config)