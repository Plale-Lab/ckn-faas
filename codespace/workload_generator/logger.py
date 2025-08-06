# import csv
# import os
# from rich.console import Console
# from rich.table import Table
#
# console = Console()
#
# LOG_HEADER = ["ID", "Deadline", "IAR", "RespTime", "Model", "Accuracy", "Latency", "State", "Success"]
#
#
# def log_result(req_id, deadline, iar, response_time, response):
#     os.makedirs("data", exist_ok=True)
#     log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/iar_results.csv"
#     write_header = not os.path.exists(log_path)
#     with open(log_path, "a", newline="") as f:
#         writer = csv.writer(f)
#         if write_header:
#             writer.writerow(LOG_HEADER)
#         writer.writerow([
#             req_id, deadline, iar, response_time,
#             response.get("model"), response.get("accuracy"),
#             response.get("latency"), response.get("container_state"),
#             response.get("success")
#         ])
#
#     # Rich real-time dashboard logging
#     table = Table(show_header=True, header_style="bold magenta")
#     for col in LOG_HEADER:
#         table.add_column(col)
#     table.add_row(
#         str(req_id), str(deadline), str(iar), f"{response_time:.3f}",
#         str(response.get("model")),
#         f"{response.get('accuracy', 0.0):.4f}",
#         f"{response.get('latency', -1):.3f}",
#         str(response.get("container_state")),
#         str(response.get("success"))
#     )
#     console.print(table)

import csv
import os
from rich.console import Console
from rich.table import Table

console = Console()

# Add per-model wait columns to the header
LOG_HEADER = [
    "ID", "Deadline", "IAR", "RespTime","RunTime", "Model", "Accuracy", "Latency", "State", "Success", "selected_models", "cost_function_execution_time_ms", "status"
    "mobilenet_v3_small_wait", "resnet18_wait", "resnet34_wait",
    "resnet50_wait", "resnet101_wait", "vit_b_16_wait"
]

def log_result(mode, req_id, deadline, iar, response_time,current_time_sec, response):
    os.makedirs("data", exist_ok=True)
    if mode == "vary_deadline":
        log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/deadline_results_I1_D60.csv"
    else:
        log_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/iar_results_3.csv"

    write_header = not os.path.exists(log_path)

    # Extract wait times from response (dict of model: wait_time)
    waits = response.get("wait_times", {})

    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(LOG_HEADER)
        writer.writerow([
            req_id, deadline, iar, response_time,current_time_sec,
            response.get("model",-1), response.get("accuracy"),
            response.get("latency"), response.get("container_state"),
            response.get("success"),
            response.get("selected_models", []),
            response.get("cost_function_execution_time_ms",-1),
            response.get("status"),
            waits.get("mobilenet_v3_small", -1),
            waits.get("resnet18", -1),
            waits.get("resnet34", -1),
            waits.get("resnet50", -1),
            waits.get("resnet101", -1),
            waits.get("vit_b_16", -1)
        ])

    # Rich real-time terminal output
    table = Table(show_header=True, header_style="bold magenta")

    for col in LOG_HEADER:
        table.add_column(col)

    table.add_row(
        str(req_id), str(deadline), str(iar), f"{response_time:.3f}", f"{current_time_sec:.3f}",
        str(response.get("model",-1)),
        f"{response.get('accuracy', 0.0):.4f}",
        f"{response.get('latency', -1):.3f}",
        str(response.get("container_state")),
        str(response.get("success")),
        str(response.get("selected_models", [])),
        str(response.get("cost_function_execution_time_ms")),
        str(response.get("status")),
        f"{waits.get('mobilenet_v3_small', -1):.2f}",
        f"{waits.get('resnet18', -1):.2f}",
        f"{waits.get('resnet34', -1):.2f}",
        f"{waits.get('resnet50', -1):.2f}",
        f"{waits.get('resnet101', -1):.2f}",
        f"{waits.get('vit_b_16', -1):.2f}"
    )

    console.print(table)
