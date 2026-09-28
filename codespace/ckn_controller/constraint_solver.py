# import os
# import time
# from dataclasses import dataclass
# from typing import Any, Dict, List, Optional
#
# import pandas as pd
#
#
# @dataclass
# class ConstraintSolverResult:
#     success: bool
#     policy: str
#     reason: str
#     selected_config: Optional[Dict[str, Any]]
#     feasible_count: int
#     total_count: int
#     solver_latency_ms: float
#
#
# class ConstraintSolver:
#     def __init__(
#         self,
#         seq_csv: str,
#         power2_csv: Optional[str] = None,
#         full_parallel_csv: Optional[str] = None,
#     ):
#         self.seq_csv = seq_csv
#         self.power2_csv = power2_csv
#         self.full_parallel_csv = full_parallel_csv
#
#         self.seq_df = self._load_seq_csv(seq_csv)
#         self.power2_df = self._safe_load_csv(power2_csv)
#         self.full_df = self._safe_load_csv(full_parallel_csv)
#
#     def _safe_load_csv(self, path: Optional[str]) -> Optional[pd.DataFrame]:
#         if not path or not os.path.exists(path):
#             return None
#
#         df = pd.read_csv(path)
#         df.columns = df.columns.str.strip()
#         return df
#
#     def _load_seq_csv(self, path: str) -> pd.DataFrame:
#         df = pd.read_csv(path)
#         df.columns = df.columns.str.strip()
#
#         required_cols = [
#             "a",
#             "b",
#             "theta",
#             "cascade_accuracy",
#             "a_accept_rate",
#             "relative_compute_cost",
#             "expected_latency_cost_ms",
#         ]
#
#         missing = [c for c in required_cols if c not in df.columns]
#         if missing:
#             raise ValueError(f"Sequential CSV missing columns: {missing}")
#
#         numeric_cols = [
#             "theta",
#             "cascade_accuracy",
#             "a_accept_rate",
#             "relative_compute_cost",
#             "expected_latency_cost_ms",
#         ]
#
#         for col in numeric_cols:
#             df[col] = pd.to_numeric(df[col], errors="coerce")
#
#         df = df.dropna(subset=numeric_cols)
#         df["error"] = 1.0 - df["cascade_accuracy"]
#         df["cascade_pair"] = df["a"].astype(str) + " -> " + df["b"].astype(str)
#
#         return df
#
#     def solve(
#         self,
#         max_cost: float,
#         target_accuracy: float,
#         policy_order: List[str],
#         time_mode: str,
#         wait_results: Dict[str, float],
#         model_profiles: Dict[str, Dict[str, Any]],
#         estimate_latency_fn=None,
#         network_latency: float = 0.0,
#         parallel_first_n: int = 2,
#     ) -> ConstraintSolverResult:
#         start = time.perf_counter()
#
#         for policy in policy_order:
#             policy = str(policy).lower()
#
#             if policy == "sequential":
#                 result = self._solve_sequential(
#                     max_cost=max_cost,
#                     target_accuracy=target_accuracy,
#                     time_mode=time_mode,
#                     wait_results=wait_results,
#                     model_profiles=model_profiles,
#                     estimate_latency_fn=estimate_latency_fn,
#                     network_latency=network_latency,
#                     parallel_first_n=parallel_first_n,
#                     start_time=start,
#                 )
#
#                 if result.success:
#                     return result
#
#             elif policy == "power2":
#                 result = self._solve_power2_placeholder(start)
#                 if result.success:
#                     return result
#
#             elif policy == "full_parallel":
#                 result = self._solve_full_parallel_placeholder(start)
#                 if result.success:
#                     return result
#
#         solver_latency_ms = (time.perf_counter() - start) * 1000.0
#
#         return ConstraintSolverResult(
#             success=False,
#             policy="fallback",
#             reason="No feasible configuration found for any policy.",
#             selected_config=None,
#             feasible_count=0,
#             total_count=0,
#             solver_latency_ms=solver_latency_ms,
#         )
#
#     def _solve_sequential(
#         self,
#         max_cost: float,
#         target_accuracy: float,
#         time_mode: str,
#         wait_results: Dict[str, float],
#         model_profiles: Dict[str, Dict[str, Any]],
#         estimate_latency_fn,
#         network_latency: float,
#         parallel_first_n: int,
#         start_time: float,
#     ) -> ConstraintSolverResult:
#         df = self.seq_df.copy()
#
#         feasible = df[
#             (df["relative_compute_cost"] <= max_cost)
#             & (df["cascade_accuracy"] >= target_accuracy)
#         ].copy()
#
#         if feasible.empty:
#             solver_latency_ms = (time.perf_counter() - start_time) * 1000.0
#
#             return ConstraintSolverResult(
#                 success=False,
#                 policy="sequential",
#                 reason="No feasible sequential cascade found.",
#                 selected_config=None,
#                 feasible_count=0,
#                 total_count=len(df),
#                 solver_latency_ms=solver_latency_ms,
#             )
#
#         time_mode = str(time_mode).lower()
#
#         if time_mode == "offline":
#             feasible["solver_latency_ms_objective"] = feasible[
#                 "expected_latency_cost_ms"
#             ]
#
#         elif time_mode == "live_et":
#             feasible["solver_latency_ms_objective"] = feasible.apply(
#                 lambda row: self._estimate_seq_live_et_ms(
#                     row=row,
#                     wait_results=wait_results,
#                     model_profiles=model_profiles,
#                     estimate_latency_fn=estimate_latency_fn,
#                     network_latency=network_latency,
#                     parallel_first_n=parallel_first_n,
#                 ),
#                 axis=1,
#             )
#
#         else:
#             raise ValueError(
#                 f"Unknown CONSTRAINT_SOLVER_TIME_MODE={time_mode}. "
#                 f"Use 'offline' or 'live_et'."
#             )
#
#         best = feasible.sort_values(
#             by=["solver_latency_ms_objective", "relative_compute_cost"],
#             ascending=[True, True],
#         ).iloc[0]
#
#         selected_config = {
#             "policy": "sequential",
#             "a": best["a"],
#             "b": best["b"],
#             "models": [best["a"], best["b"]],
#             "theta": float(best["theta"]),
#             "cascade_accuracy": float(best["cascade_accuracy"]),
#             "accuracy": float(best["cascade_accuracy"]),
#             "error": float(1.0 - best["cascade_accuracy"]),
#             "a_accept_rate": float(best["a_accept_rate"]),
#             "relative_compute_cost": float(best["relative_compute_cost"]),
#             "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
#             "solver_latency_ms_objective": float(best["solver_latency_ms_objective"]),
#             "time_mode": time_mode,
#             "cascade_pair": best["cascade_pair"],
#         }
#
#         solver_latency_ms = (time.perf_counter() - start_time) * 1000.0
#
#         return ConstraintSolverResult(
#             success=True,
#             policy="sequential",
#             reason=f"Feasible sequential cascade found using {time_mode}.",
#             selected_config=selected_config,
#             feasible_count=len(feasible),
#             total_count=len(df),
#             solver_latency_ms=solver_latency_ms,
#         )
#
#     def _estimate_seq_live_et_ms(
#         self,
#         row,
#         wait_results: Dict[str, float],
#         model_profiles: Dict[str, Dict[str, Any]],
#         estimate_latency_fn,
#         network_latency: float,
#         parallel_first_n: int,
#     ) -> float:
#         a = row["a"]
#         b = row["b"]
#
#         if estimate_latency_fn is not None:
#             try:
#                 latency_s = estimate_latency_fn(
#                     cascade_mode="sequential",
#                     selected_models=[a, b],
#                     wait_results=wait_results,
#                     network_latency=network_latency,
#                     parallel_first_n=parallel_first_n,
#                 )
#                 return float(latency_s) * 1000.0
#             except Exception:
#                 pass
#
#         # fallback direct E[T] for two-stage sequential cascade
#         a_accept_rate = float(row["a_accept_rate"])
#
#         wait_a = float(wait_results.get(a, 0.0))
#         wait_b = float(wait_results.get(b, 0.0))
#
#         exec_a = float(model_profiles[a]["latency"])
#         exec_b = float(model_profiles[b]["latency"])
#
#         et_s = (wait_a + exec_a) + (1.0 - a_accept_rate) * (
#             wait_b + exec_b + network_latency
#         )
#
#         return et_s * 1000.0
#
#     def _solve_power2_placeholder(self, start_time: float) -> ConstraintSolverResult:
#         solver_latency_ms = (time.perf_counter() - start_time) * 1000.0
#
#         return ConstraintSolverResult(
#             success=False,
#             policy="power2",
#             reason="Power2 constraint solver is not implemented yet.",
#             selected_config=None,
#             feasible_count=0,
#             total_count=0,
#             solver_latency_ms=solver_latency_ms,
#         )
#
#     def _solve_full_parallel_placeholder(self, start_time: float) -> ConstraintSolverResult:
#         solver_latency_ms = (time.perf_counter() - start_time) * 1000.0
#
#         return ConstraintSolverResult(
#             success=False,
#             policy="full_parallel",
#             reason="Full parallel constraint solver is not implemented yet.",
#             selected_config=None,
#             feasible_count=0,
#             total_count=0,
#             solver_latency_ms=solver_latency_ms,
#         )


import os
import ast
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ConstraintSolverResult:
    success: bool
    policy: str
    reason: str
    selected_config: Optional[Dict[str, Any]]
    feasible_count: int
    total_count: int
    solver_latency_ms: float




class ConstraintSolver:
    MODEL_NAME_MAP = {
        "mobilenetv3_small_100": "mobilenet_v3_small",
        "mobilenetv2_100": "mobilenet_v2",
        "vit_base_patch16_224": "vit_b_16",
    }
    def __init__(
        self,
        seq_csv: str,
        power2_csv: Optional[str] = None,
        full_parallel_csv: Optional[str] = None,
    ):
        self.seq_df = self._load_seq_csv(seq_csv)
        self.power2_df = self._safe_load_csv(power2_csv)
        self.full_df = self._safe_load_csv(full_parallel_csv)

    def _safe_load_csv(self, path: Optional[str]) -> Optional[pd.DataFrame]:
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df

    def _load_seq_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        df["a"] = df["a"].astype(str).str.strip().replace(self.MODEL_NAME_MAP)
        df["b"] = df["b"].astype(str).str.strip().replace(self.MODEL_NAME_MAP)

        required = [
            "a", "b", "theta", "cascade_accuracy",
            "a_accept_rate", "relative_compute_cost",
            "expected_latency_cost_ms",
        ]

        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Sequential CSV missing columns: {missing}")

        for col in [
            "theta", "cascade_accuracy", "a_accept_rate",
            "relative_compute_cost", "expected_latency_cost_ms",
        ]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=[
            "theta", "cascade_accuracy", "a_accept_rate",
            "relative_compute_cost", "expected_latency_cost_ms",
        ])

        df["error"] = 1.0 - df["cascade_accuracy"]
        df["cascade_pair"] = df["a"].astype(str) + " -> " + df["b"].astype(str)
        return df

    def solve(
        self,
        max_cost: float,
        target_accuracy: float,
        policy_order: List[str],
        time_mode: str,
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],

        network_latency: float = 0.0,
        parallel_first_n: int = 2,

        base_request_overhead: float = 0.04,
        sequential_stage_overhead: float = 0.01,
        po2_first_batch_overhead: float = 0.005,
        po2_tail_overhead: float = 0.01,
        full_parallel_agg_overhead: float = 0.005,

        et_scale_seq: float = 1.5,
        et_offset_seq: float = 0.02,
        et_scale_p2: float = 1.4,
        et_offset_p2: float = 0.02,
    ) -> ConstraintSolverResult:
        start = time.perf_counter()

        for policy in policy_order:
            policy = str(policy).lower()

            if policy == "sequential":
                result = self._solve_sequential(
                    max_cost=max_cost,
                    target_accuracy=target_accuracy,
                    time_mode=time_mode,
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    network_latency=network_latency,
                    base_request_overhead=base_request_overhead,
                    sequential_stage_overhead=sequential_stage_overhead,
                    et_scale_seq=et_scale_seq,
                    et_offset_seq=et_offset_seq,
                    start_time=start,
                )
            elif policy == "power2":
                result = self._solve_power2(
                    max_cost=max_cost,
                    target_accuracy=target_accuracy,
                    time_mode=time_mode,
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    network_latency=network_latency,
                    parallel_first_n=parallel_first_n,
                    base_request_overhead=base_request_overhead,
                    po2_first_batch_overhead=po2_first_batch_overhead,
                    po2_tail_overhead=po2_tail_overhead,
                    sequential_stage_overhead=sequential_stage_overhead,
                    et_scale_p2=et_scale_p2,
                    et_offset_p2=et_offset_p2,
                    start_time=start,
                )
            elif policy == "full_parallel":
                result = self._solve_full_parallel(
                    max_cost=max_cost,
                    target_accuracy=target_accuracy,
                    time_mode=time_mode,
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    base_request_overhead=base_request_overhead,
                    full_parallel_agg_overhead=full_parallel_agg_overhead,
                    start_time=start,
                )
            else:
                continue

            if result.success:
                return result

        return ConstraintSolverResult(
            success=False,
            policy="fallback",
            reason="No feasible configuration found for any policy.",
            selected_config=None,
            feasible_count=0,
            total_count=0,
            solver_latency_ms=(time.perf_counter() - start) * 1000.0,
        )

    # ------------------------------------------------------------
    # Common helpers
    # ------------------------------------------------------------
    # def _model_time_s(self, model: str, wait_results: Dict[str, float], model_profiles: Dict[str, Dict[str, Any]]) -> float:
    #     wait_s = float(wait_results.get(model, 0.0))
    #     exec_s = float(model_profiles[model]["latency"])
    #     return wait_s + exec_s
    def _model_time_s(
            self,
            model: str,
            wait_results: Dict[str, float],
            model_profiles: Dict[str, Dict[str, Any]],
    ) -> float:
        wait_s = float(wait_results.get(model, 0.0))
        exec_s = float(model_profiles[model]["latency"])

        WAIT_WEIGHT = 12.0

        return (WAIT_WEIGHT * wait_s) + exec_s

    def _parse_models(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(x) for x in value]

        if isinstance(value, str):
            value = value.strip()

            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed]
            except Exception:
                pass

            return [x.strip() for x in value.split(",") if x.strip()]

        return []

    # ------------------------------------------------------------
    # Sequential E[T]
    # ------------------------------------------------------------
    def _seq_et_s(
        self,
        models: List[str],
        exit_probs: Dict[str, float],
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        base_request_overhead: float,
        sequential_stage_overhead: float,
        et_scale_seq: float,
        et_offset_seq: float,
    ) -> float:
        raw_et = 0.0
        continue_prob = 1.0

        for i, model in enumerate(models):
            t_model = self._model_time_s(model, wait_results, model_profiles)

            if i == 0:
                raw_et += continue_prob * t_model
            else:
                raw_et += continue_prob * (sequential_stage_overhead + t_model)

            p_exit = max(0.0, min(1.0, float(exit_probs.get(model, 0.0))))
            continue_prob *= (1.0 - p_exit)

        return base_request_overhead + et_scale_seq * raw_et + et_offset_seq

    # ------------------------------------------------------------
    # Power2 E[T]
    # ------------------------------------------------------------
    def _power2_et_s(
        self,
        models: List[str],
        exit_probs: Dict[str, float],
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        parallel_first_n: int,
        base_request_overhead: float,
        po2_first_batch_overhead: float,
        po2_tail_overhead: float,
        sequential_stage_overhead: float,
        et_scale_p2: float,
        et_offset_p2: float,
    ) -> float:
        first_n = max(1, min(int(parallel_first_n), len(models)))
        first_batch = models[:first_n]
        tail = models[first_n:]

        if len(first_batch) < 2:
            return self._seq_et_s(
                models=models,
                exit_probs=exit_probs,
                wait_results=wait_results,
                model_profiles=model_profiles,
                base_request_overhead=base_request_overhead,
                sequential_stage_overhead=sequential_stage_overhead,
                et_scale_seq=et_scale_p2,
                et_offset_seq=et_offset_p2,
            )

        m1, m2 = first_batch[0], first_batch[1]

        t1 = self._model_time_s(m1, wait_results, model_profiles)
        t2 = self._model_time_s(m2, wait_results, model_profiles)

        p1 = max(0.0, min(1.0, float(exit_probs.get(m1, 0.0))))
        p2 = max(0.0, min(1.0, float(exit_probs.get(m2, 0.0))))

        p_exit_first_batch = 1.0 - ((1.0 - p1) * (1.0 - p2))

        first_batch_latency = (
            p1 * min(t1, t2)
            + (1.0 - p1) * max(t1, t2)
        )

        if tail:
            tail_latency = self._seq_et_s(
                models=tail,
                exit_probs=exit_probs,
                wait_results=wait_results,
                model_profiles=model_profiles,
                base_request_overhead=0.0,
                sequential_stage_overhead=sequential_stage_overhead,
                et_scale_seq=1.0,
                et_offset_seq=0.0,
            )
        else:
            tail_latency = 0.0

        raw_et = (
            po2_first_batch_overhead
            + first_batch_latency
            + (1.0 - p_exit_first_batch) * (po2_tail_overhead + tail_latency)
        )

        return base_request_overhead + et_scale_p2 * raw_et + et_offset_p2

    # ------------------------------------------------------------
    # Full parallel E[T]
    # ------------------------------------------------------------
    def _full_parallel_et_s(
        self,
        models: List[str],
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        base_request_overhead: float,
        full_parallel_agg_overhead: float,
    ) -> float:
        if not models:
            return float("inf")

        max_model_time = max(
            self._model_time_s(m, wait_results, model_profiles)
            for m in models
        )

        return base_request_overhead + max_model_time + full_parallel_agg_overhead

    # ------------------------------------------------------------
    # Sequential solver
    # ------------------------------------------------------------
    def _solve_sequential(
        self,
        max_cost: float,
        target_accuracy: float,
        time_mode: str,
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        network_latency: float,
        base_request_overhead: float,
        sequential_stage_overhead: float,
        et_scale_seq: float,
        et_offset_seq: float,
        start_time: float,
    ) -> ConstraintSolverResult:
        df = self.seq_df.copy()

        feasible = df[
            (df["relative_compute_cost"] <= max_cost)
            & (df["cascade_accuracy"] >= target_accuracy)
        ].copy()

        if feasible.empty:
            return ConstraintSolverResult(
                success=False,
                policy="sequential",
                reason="No feasible sequential cascade found.",
                selected_config=None,
                feasible_count=0,
                total_count=len(df),
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        time_mode = str(time_mode).lower()

        if time_mode == "offline":
            feasible["solver_latency_ms_objective"] = feasible["expected_latency_cost_ms"]

        elif time_mode == "live_et":
            feasible["solver_latency_ms_objective"] = feasible.apply(
                lambda row: self._seq_et_s(
                    models=[row["a"], row["b"]],
                    exit_probs={
                        row["a"]: float(row["a_accept_rate"]),
                        row["b"]: 1.0,
                    },
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    base_request_overhead=base_request_overhead,
                    sequential_stage_overhead=sequential_stage_overhead,
                    et_scale_seq=et_scale_seq,
                    et_offset_seq=et_offset_seq,
                ) * 1000.0,
                axis=1,
            )
        else:
            raise ValueError("time_mode must be 'offline' or 'live_et'")

        best = feasible.sort_values(
            by=["solver_latency_ms_objective", "relative_compute_cost"],
            ascending=[True, True],
        ).iloc[0]

        selected_config = {
            "policy": "sequential",
            "models": [best["a"], best["b"]],
            "a": best["a"],
            "b": best["b"],
            "theta": float(best["theta"]),
            "accuracy": float(best["cascade_accuracy"]),
            "cascade_accuracy": float(best["cascade_accuracy"]),
            "error": float(1.0 - best["cascade_accuracy"]),
            "a_accept_rate": float(best["a_accept_rate"]),
            "relative_compute_cost": float(best["relative_compute_cost"]),
            "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
            "solver_latency_ms_objective": float(best["solver_latency_ms_objective"]),
            "time_mode": time_mode,
            "cascade_pair": best["cascade_pair"],
        }

        return ConstraintSolverResult(
            success=True,
            policy="sequential",
            reason=f"Feasible sequential cascade found using {time_mode}.",
            selected_config=selected_config,
            feasible_count=len(feasible),
            total_count=len(df),
            solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    # ------------------------------------------------------------
    # Power2 solver
    # Expected CSV columns:
    # models, theta, power2_accuracy, relative_compute_cost,
    # expected_latency_cost_ms
    # Optional: exit_probs
    # ------------------------------------------------------------
    def _solve_power2(
        self,
        max_cost: float,
        target_accuracy: float,
        time_mode: str,
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        network_latency: float,
        parallel_first_n: int,
        base_request_overhead: float,
        po2_first_batch_overhead: float,
        po2_tail_overhead: float,
        sequential_stage_overhead: float,
        et_scale_p2: float,
        et_offset_p2: float,
        start_time: float,
    ) -> ConstraintSolverResult:
        if self.power2_df is None:
            return ConstraintSolverResult(
                success=False,
                policy="power2",
                reason="Power2 CSV not found.",
                selected_config=None,
                feasible_count=0,
                total_count=0,
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        df = self.power2_df.copy()

        required = ["models", "theta", "power2_accuracy", "relative_compute_cost", "expected_latency_cost_ms"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return ConstraintSolverResult(
                success=False,
                policy="power2",
                reason=f"Power2 CSV missing columns: {missing}",
                selected_config=None,
                feasible_count=0,
                total_count=len(df),
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        for col in ["theta", "power2_accuracy", "relative_compute_cost", "expected_latency_cost_ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["theta", "power2_accuracy", "relative_compute_cost", "expected_latency_cost_ms"])

        feasible = df[
            (df["relative_compute_cost"] <= max_cost)
            & (df["power2_accuracy"] >= target_accuracy)
        ].copy()

        if feasible.empty:
            return ConstraintSolverResult(
                success=False,
                policy="power2",
                reason="No feasible Power2 configuration found.",
                selected_config=None,
                feasible_count=0,
                total_count=len(df),
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        time_mode = str(time_mode).lower()

        if time_mode == "offline":
            feasible["solver_latency_ms_objective"] = feasible["expected_latency_cost_ms"]

        elif time_mode == "live_et":
            feasible["solver_latency_ms_objective"] = feasible.apply(
                lambda row: self._power2_et_s(
                    models=self._parse_models(row["models"]),
                    exit_probs=self._parse_exit_probs(row),
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    parallel_first_n=parallel_first_n,
                    base_request_overhead=base_request_overhead,
                    po2_first_batch_overhead=po2_first_batch_overhead,
                    po2_tail_overhead=po2_tail_overhead,
                    sequential_stage_overhead=sequential_stage_overhead,
                    et_scale_p2=et_scale_p2,
                    et_offset_p2=et_offset_p2,
                ) * 1000.0,
                axis=1,
            )
        else:
            raise ValueError("time_mode must be 'offline' or 'live_et'")

        best = feasible.sort_values(
            by=["solver_latency_ms_objective", "relative_compute_cost"],
            ascending=[True, True],
        ).iloc[0]

        models = self._parse_models(best["models"])

        selected_config = {
            "policy": "power2",
            "models": models,
            "theta": float(best["theta"]),
            "accuracy": float(best["power2_accuracy"]),
            "error": float(1.0 - best["power2_accuracy"]),
            "relative_compute_cost": float(best["relative_compute_cost"]),
            "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
            "solver_latency_ms_objective": float(best["solver_latency_ms_objective"]),
            "time_mode": time_mode,
        }

        return ConstraintSolverResult(
            success=True,
            policy="power2",
            reason=f"Feasible Power2 configuration found using {time_mode}.",
            selected_config=selected_config,
            feasible_count=len(feasible),
            total_count=len(df),
            solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )

    def _parse_exit_probs(self, row) -> Dict[str, float]:
        if "exit_probs" in row and pd.notna(row["exit_probs"]):
            try:
                parsed = ast.literal_eval(str(row["exit_probs"]))
                if isinstance(parsed, dict):
                    return {str(k): float(v) for k, v in parsed.items()}
            except Exception:
                pass

        models = self._parse_models(row.get("models", ""))

        # fallback approximation
        probs = {}
        for m in models:
            probs[m] = 0.5
        return probs

    # ------------------------------------------------------------
    # Full parallel solver
    # Expected CSV columns:
    # models, ensemble_accuracy, relative_compute_cost,
    # expected_latency_cost_ms
    # ------------------------------------------------------------
    def _solve_full_parallel(
        self,
        max_cost: float,
        target_accuracy: float,
        time_mode: str,
        wait_results: Dict[str, float],
        model_profiles: Dict[str, Dict[str, Any]],
        base_request_overhead: float,
        full_parallel_agg_overhead: float,
        start_time: float,
    ) -> ConstraintSolverResult:
        if self.full_df is None:
            return ConstraintSolverResult(
                success=False,
                policy="full_parallel",
                reason="Full parallel CSV not found.",
                selected_config=None,
                feasible_count=0,
                total_count=0,
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        df = self.full_df.copy()

        required = ["models", "ensemble_accuracy", "relative_compute_cost", "expected_latency_cost_ms"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            return ConstraintSolverResult(
                success=False,
                policy="full_parallel",
                reason=f"Full parallel CSV missing columns: {missing}",
                selected_config=None,
                feasible_count=0,
                total_count=len(df),
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        for col in ["ensemble_accuracy", "relative_compute_cost", "expected_latency_cost_ms"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["ensemble_accuracy", "relative_compute_cost", "expected_latency_cost_ms"])

        feasible = df[
            (df["relative_compute_cost"] <= max_cost)
            & (df["ensemble_accuracy"] >= target_accuracy)
        ].copy()

        if feasible.empty:
            return ConstraintSolverResult(
                success=False,
                policy="full_parallel",
                reason="No feasible full-parallel configuration found.",
                selected_config=None,
                feasible_count=0,
                total_count=len(df),
                solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
            )

        time_mode = str(time_mode).lower()

        if time_mode == "offline":
            feasible["solver_latency_ms_objective"] = feasible["expected_latency_cost_ms"]

        elif time_mode == "live_et":
            feasible["solver_latency_ms_objective"] = feasible.apply(
                lambda row: self._full_parallel_et_s(
                    models=self._parse_models(row["models"]),
                    wait_results=wait_results,
                    model_profiles=model_profiles,
                    base_request_overhead=base_request_overhead,
                    full_parallel_agg_overhead=full_parallel_agg_overhead,
                ) * 1000.0,
                axis=1,
            )
        else:
            raise ValueError("time_mode must be 'offline' or 'live_et'")

        best = feasible.sort_values(
            by=["solver_latency_ms_objective", "relative_compute_cost"],
            ascending=[True, True],
        ).iloc[0]

        models = self._parse_models(best["models"])

        selected_config = {
            "policy": "full_parallel",
            "models": models,
            "theta": None,
            "accuracy": float(best["ensemble_accuracy"]),
            "error": float(1.0 - best["ensemble_accuracy"]),
            "relative_compute_cost": float(best["relative_compute_cost"]),
            "expected_latency_cost_ms": float(best["expected_latency_cost_ms"]),
            "solver_latency_ms_objective": float(best["solver_latency_ms_objective"]),
            "time_mode": time_mode,
        }

        return ConstraintSolverResult(
            success=True,
            policy="full_parallel",
            reason=f"Feasible full-parallel configuration found using {time_mode}.",
            selected_config=selected_config,
            feasible_count=len(feasible),
            total_count=len(df),
            solver_latency_ms=(time.perf_counter() - start_time) * 1000.0,
        )