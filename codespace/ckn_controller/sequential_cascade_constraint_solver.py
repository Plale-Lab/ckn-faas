import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import pandas as pd


@dataclass
class SolverResult:
    success: bool
    reason: str
    selected_config: Optional[Dict[str, Any]]
    feasible_count: int
    total_count: int
    solver_latency_ms: float


class SequentialCascadeConstraintSolver:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = self._load_csv(csv_path)

    def _load_csv(self, csv_path: str) -> pd.DataFrame:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()

        required_cols = [
            "a",
            "b",
            "theta",
            "cascade_accuracy",
            "a_accept_rate",
            "relative_compute_cost",
            "expected_latency_cost_ms",
        ]

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        numeric_cols = [
            "theta",
            "cascade_accuracy",
            "a_accept_rate",
            "relative_compute_cost",
            "expected_latency_cost_ms",
        ]

        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=numeric_cols)

        df["error"] = 1.0 - df["cascade_accuracy"]
        df["cascade_pair"] = df["a"].astype(str) + " -> " + df["b"].astype(str)

        return df

    def solve(
        self,
        max_cost: float,
        max_error: Optional[float] = None,
        target_accuracy: Optional[float] = None,
        first_model: Optional[str] = None,
        second_model: Optional[str] = None,
    ) -> SolverResult:
        """
        Solve:
            minimize expected_latency_cost_ms
            subject to:
                relative_compute_cost <= max_cost
                error <= max_error
        """

        start = time.perf_counter()

        if max_error is None and target_accuracy is None:
            raise ValueError("Provide either max_error or target_accuracy.")

        if target_accuracy is None:
            target_accuracy = 1.0 - float(max_error)

        candidates = self.df.copy()

        if first_model is not None:
            candidates = candidates[candidates["a"] == first_model]

        if second_model is not None:
            candidates = candidates[candidates["b"] == second_model]

        feasible = candidates[
            (candidates["relative_compute_cost"] <= max_cost)
            & (candidates["cascade_accuracy"] >= target_accuracy)
        ].copy()

        solver_latency_ms = (time.perf_counter() - start) * 1000.0

        if feasible.empty:
            return SolverResult(
                success=False,
                reason="No feasible sequential cascade found.",
                selected_config=None,
                feasible_count=0,
                total_count=len(candidates),
                solver_latency_ms=solver_latency_ms,
            )

        best = feasible.sort_values(
            by=["expected_latency_cost_ms", "relative_compute_cost"],
            ascending=[True, True],
        ).iloc[0]

        selected_config = {
            "a": best["a"],
            "b": best["b"],
            "theta": float(best["theta"]),
            "cascade_accuracy": float(best["cascade_accuracy"]),
            "error": float(1.0 - best["cascade_accuracy"]),
            "a_accept_rate": float(best["a_accept_rate"]),
            "relative_compute_cost": float(best["relative_compute_cost"]),
            "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
            "cascade_pair": best["cascade_pair"],
        }

        solver_latency_ms = (time.perf_counter() - start) * 1000.0

        return SolverResult(
            success=True,
            reason="Feasible sequential cascade found.",
            selected_config=selected_config,
            feasible_count=len(feasible),
            total_count=len(candidates),
            solver_latency_ms=solver_latency_ms,
        )

    def solve_least_violation(
        self,
        max_cost: float,
        target_accuracy: float,
    ) -> SolverResult:
        """
        Fallback if no feasible point exists.
        Select the row with the smallest constraint violation.
        """

        start = time.perf_counter()

        df = self.df.copy()

        df["cost_violation"] = (df["relative_compute_cost"] - max_cost).clip(lower=0)
        df["accuracy_violation"] = (target_accuracy - df["cascade_accuracy"]).clip(lower=0)

        df["total_violation"] = df["cost_violation"] + df["accuracy_violation"]

        best = df.sort_values(
            by=["total_violation", "expected_latency_cost_ms"],
            ascending=[True, True],
        ).iloc[0]

        selected_config = {
            "a": best["a"],
            "b": best["b"],
            "theta": float(best["theta"]),
            "cascade_accuracy": float(best["cascade_accuracy"]),
            "error": float(1.0 - best["cascade_accuracy"]),
            "a_accept_rate": float(best["a_accept_rate"]),
            "relative_compute_cost": float(best["relative_compute_cost"]),
            "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
            "cascade_pair": best["cascade_pair"],
            "cost_violation": float(best["cost_violation"]),
            "accuracy_violation": float(best["accuracy_violation"]),
            "total_violation": float(best["total_violation"]),
        }

        solver_latency_ms = (time.perf_counter() - start) * 1000.0

        return SolverResult(
            success=False,
            reason="No feasible point found. Returning least-violation fallback.",
            selected_config=selected_config,
            feasible_count=0,
            total_count=len(df),
            solver_latency_ms=solver_latency_ms,
        )