import os, csv
from rich.table import Table
from rich.console import Console
from ckn_controller.ckn_config import M_TOTAL


console = Console()

# Build the CSV header once from M_TOTAL
def _make_log_header():
    base = [
        "req_id", "deadline_ms", "iar_ms",
        "response_time_s", "current_time_s"
    ]
    est_cols = [f"est_{m}" for m in M_TOTAL]
    act_cols = [f"act_{m}" for m in M_TOTAL]
    return base + est_cols + act_cols

def _collect_latency_lists(response):
    """
    Returns (est_list, act_list) aligned to M_TOTAL.
    Missing values become None.
    """
    est = response.get("estimated_latency", {}) or {}
    act = response.get("actual_latency", {}) or {}
    est_list = [est.get(m, None) for m in M_TOTAL]
    act_list = [act.get(m, None) for m in M_TOTAL]
    return est_list, act_list

def _fmt_num(x, nd=3):
    if isinstance(x, (int, float)) and x == float("inf"):
        return "inf"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}"
    return str(x)

def log_result(mode, req_id, deadline, iar, response_time, current_time_sec, response):
    os.makedirs("data", exist_ok=True)
    if mode == "vary_deadline":
        log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_M2.csv"
    else:
        log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/actual_n_estimated_latency_M2.csv"

    write_header = not os.path.exists(log_path)

    # ---- Pull per-model latencies from response ----
    est_list, act_list = _collect_latency_lists(response)

    # ---- CSV write ----
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_make_log_header())

        row = [
            req_id,
            deadline,
            iar,
            response_time,
            current_time_sec,
            *est_list,
            *act_list,
        ]
        writer.writerow(row)

    # ---- Pretty one-row table ----
    table = Table(show_header=True, header_style="bold magenta")
    for col in _make_log_header():
        table.add_column(col)

    base_vals = [
        str(req_id),
        str(deadline),
        str(iar),
        _fmt_num(response_time, 3),
        _fmt_num(current_time_sec, 3),
    ]
    est_vals = [_fmt_num(v, 3) for v in est_list]
    act_vals = [_fmt_num(v, 3) for v in act_list]

    table.add_row(*base_vals, *est_vals, *act_vals)
    console.print(table)
