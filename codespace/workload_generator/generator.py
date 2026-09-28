# import asyncio, time
#
# import numpy as np
#
# import asyncio
# import time
# from config import workload_profiles, deadline_profiles_ms, iar_profiles_ms
# from client import send_request
# from utils import generate_id
#
# async def generate_workload(profile_name, duration_sec, mode="vary_deadline"):
#     """
#     mode = "vary_deadline" → change deadline, keep IAR fixed
#     mode = "vary_iar"      → change IAR, keep deadline fixed
#     """
#
#     # iar = iar_profiles["short"] / 1000.0
#
#     default_deadline_ms = 2000
#     default_air_ms = 1000
#
#     req_index = 0
#     tasks = []
#
#     if mode == "vary_deadline":
#         iar_sec = default_air_ms / 1000.0
#         deadlines = deadline_profiles_ms[profile_name]
#
#         for deadline_ms in deadlines:
#             start_time = time.time()
#             end_time = time.time() + duration_sec
#             while time.time() < end_time:
#                 req_id = generate_id(req_index)
#                 current_time_sec = time.time() - start_time
#                 task = asyncio.create_task(send_request(req_id, deadline_ms, default_air_ms, mode,current_time_sec))
#                 tasks.append(task)
#                 req_index += 1
#                 sleep_sec = np.random.exponential(scale=iar_sec)
#                 await asyncio.sleep(sleep_sec)
#                 # await asyncio.sleep(iar_sec)
#
#     elif mode == "vary_iar":
#         iars = iar_profiles_ms[profile_name]
#         for iar_ms in iars:
#             iar_sec = iar_ms / 1000.0
#             start_time = time.time()
#             end_time = time.time() + duration_sec
#             while time.time() < end_time:
#                 req_id = generate_id(req_index)
#                 current_time_sec = time.time() - start_time
#                 task = asyncio.create_task(send_request(req_id, default_deadline_ms, iar_ms, mode,current_time_sec))
#                 tasks.append(task)
#                 req_index += 1
#                 sleep_sec = np.random.exponential(scale=iar_sec)
#                 await asyncio.sleep(sleep_sec)
#                 # await asyncio.sleep(iar_sec)
#
#     elif mode == "dynamic_iar":
#         deadline_ms = default_deadline_ms
#
#         dynamic_trace = [
#             # phase_name, IAT_ms, ratio_of_total_duration
#             # ("idle_low", 2000, 0.20),
#             # ("medium", 1000, 0.20),
#             # ("high", 500, 0.25),
#             # ("burst", 100, 0.15),
#             # ("recovery", 2000, 0.20),
#
#             # ("low", 2000, 0.20),
#             # ("high", 250, 0.18),
#             # ("low", 2000, 0.17),
#             # ("burst", 100, 0.20),
#             # ("recovery", 2000, 0.25),
#
#             ("low", 2000, 0.20),
#             ("medium", 1000, 0.18),
#             ("low", 2000, 0.17),
#             ("high", 200, 0.20),
#             ("recovery", 2000, 0.25),
#
#             # ("low", 2000, 0.15),
#             # ("medium", 1000, 0.15),
#             # ("high", 500, 0.15),
#             # # ("very_high", 2000, 0.15),
#             # ("low", 2000, 0.15),
#             # ("burst", 100, 0.20),
#             # ("recovery", 2000, 0.20),
#         ]
#
#         # check total ratio
#         total_ratio = sum(r for _, _, r in dynamic_trace)
#         if abs(total_ratio - 1.0) > 1e-6:
#             raise ValueError(f"Dynamic trace ratios must sum to 1.0, got {total_ratio}")
#
#         global_start = time.time()
#
#         for phase_name, iar_ms, ratio in dynamic_trace:
#             phase_duration_sec = duration_sec * ratio
#             iar_sec = iar_ms / 1000.0
#
#             phase_start = time.time()
#             phase_end = phase_start + phase_duration_sec
#
#             print(
#                 f"[DYNAMIC_IAT] phase={phase_name} "
#                 f"IAT={iar_ms}ms duration={phase_duration_sec:.2f}s"
#             )
#
#             while time.time() < phase_end:
#                 req_id = generate_id(req_index)
#                 current_time_sec = time.time() - global_start
#
#                 task = asyncio.create_task(
#                     send_request(
#                         req_id,
#                         deadline_ms,
#                         iar_ms,
#                         mode,
#                         current_time_sec,
#                     )
#                 )
#                 tasks.append(task)
#
#                 req_index += 1
#
#                 sleep_sec = np.random.exponential(scale=iar_sec)
#                 await asyncio.sleep(sleep_sec)
#
#     await asyncio.gather(*tasks)


import asyncio
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import workload_profiles, deadline_profiles_ms, iar_profiles_ms
from client import send_request
from utils import generate_id


async def generate_workload(
    profile_name,
    duration_sec,
    mode="vary_deadline",
    trace_path="/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/azure_load/generated_trace/chosen-ecdf-5scal.csv",
    speedup=1.0,
):
    """
    Modes:
        vary_deadline  -> change deadline, keep IAR fixed
        vary_iar       -> change IAR, keep deadline fixed
        dynamic_iar    -> use manually defined workload phases
        azure_trace    -> replay requests using invoke_time_ms from CSV

    speedup:
        1.0 -> original trace
        2.0 -> replay twice as fast
        5.0 -> replay five times as fast
    """

    default_deadline_ms = 2000
    default_iar_ms = 100

    req_index = 0
    tasks = []

    if speedup <= 0:
        raise ValueError("speedup must be greater than 0")

    if mode == "vary_deadline":
        iar_sec = default_iar_ms / 1000.0
        deadlines = deadline_profiles_ms[profile_name]

        for deadline_ms in deadlines:
            start_time = time.monotonic()
            end_time = time.monotonic() + duration_sec

            while time.monotonic() < end_time:
                req_id = generate_id(req_index)
                current_time_sec = time.monotonic() - start_time

                task = asyncio.create_task(
                    send_request(
                        req_id,
                        deadline_ms,
                        default_iar_ms,
                        mode,
                        current_time_sec,
                    )
                )
                tasks.append(task)
                req_index += 1

                sleep_sec = np.random.exponential(scale=iar_sec)
                await asyncio.sleep(sleep_sec)

    elif mode == "vary_iar":
        iars = iar_profiles_ms[profile_name]

        for iar_ms in iars:
            iar_sec = iar_ms / 1000.0
            start_time = time.monotonic()
            end_time = time.monotonic() + duration_sec

            while time.monotonic() < end_time:
                req_id = generate_id(req_index)
                current_time_sec = time.monotonic() - start_time

                task = asyncio.create_task(
                    send_request(
                        req_id,
                        default_deadline_ms,
                        iar_ms,
                        mode,
                        current_time_sec,
                    )
                )
                tasks.append(task)
                req_index += 1

                sleep_sec = np.random.exponential(scale=iar_sec)
                await asyncio.sleep(sleep_sec)

    elif mode == "dynamic_iar":
        deadline_ms = default_deadline_ms

        dynamic_trace = [
            ("low", 2000, 0.20),
            ("medium", 1000, 0.18),
            ("low", 2000, 0.17),
            ("high", 200, 0.20),
            ("recovery", 2000, 0.25),
        ]

        total_ratio = sum(ratio for _, _, ratio in dynamic_trace)

        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                f"Dynamic trace ratios must sum to 1.0, got {total_ratio}"
            )

        global_start = time.monotonic()

        for phase_name, iar_ms, ratio in dynamic_trace:
            phase_duration_sec = duration_sec * ratio
            iar_sec = iar_ms / 1000.0

            phase_start = time.monotonic()
            phase_end = phase_start + phase_duration_sec

            print(
                f"[DYNAMIC_IAT] phase={phase_name} "
                f"IAT={iar_ms}ms "
                f"duration={phase_duration_sec:.2f}s"
            )

            while time.monotonic() < phase_end:
                req_id = generate_id(req_index)
                current_time_sec = time.monotonic() - global_start

                task = asyncio.create_task(
                    send_request(
                        req_id,
                        deadline_ms,
                        iar_ms,
                        mode,
                        current_time_sec,
                    )
                )
                tasks.append(task)
                req_index += 1

                sleep_sec = np.random.exponential(scale=iar_sec)
                await asyncio.sleep(sleep_sec)

    elif mode == "azure_trace":
        trace_file = Path(trace_path)

        if not trace_file.exists():
            raise FileNotFoundError(
                f"Azure trace file was not found: {trace_file}"
            )

        trace_df = pd.read_csv(trace_file)

        required_columns = {"func_name", "invoke_time_ms"}
        missing_columns = required_columns - set(trace_df.columns)

        if missing_columns:
            raise ValueError(
                f"Trace is missing required columns: {missing_columns}"
            )

        trace_df = (
            trace_df
            .dropna(subset=["invoke_time_ms"])
            .sort_values("invoke_time_ms")
            .reset_index(drop=True)
        )

        if trace_df.empty:
            raise ValueError("The Azure trace contains no requests")

        # Start the trace at time zero even when its first timestamp is nonzero.
        first_invoke_ms = float(trace_df.iloc[0]["invoke_time_ms"])
        trace_df["relative_time_ms"] = (
            trace_df["invoke_time_ms"] - first_invoke_ms
        )

        # Compute the original IAT between consecutive merged requests.
        trace_df["original_iar_ms"] = (
            trace_df["relative_time_ms"]
            .diff()
            .fillna(0.0)
            .clip(lower=0.0)
        )

        experiment_start = time.monotonic()

        print(
            f"[AZURE_TRACE] requests={len(trace_df)} "
            f"speedup={speedup}x "
            f"trace={trace_file}"
        )

        for _, row in trace_df.iterrows():
            original_relative_ms = float(row["relative_time_ms"])

            # Compress timestamps when speedup is greater than 1.
            scheduled_time_sec = (
                original_relative_ms / speedup
            ) / 1000.0

            # Optional: stop after duration_sec.
            if (
                duration_sec is not None
                and scheduled_time_sec > duration_sec
            ):
                break

            target_send_time = experiment_start + scheduled_time_sec
            remaining_sec = target_send_time - time.monotonic()

            if remaining_sec > 0:
                await asyncio.sleep(remaining_sec)

            req_id = generate_id(req_index)
            current_time_sec = time.monotonic() - experiment_start

            # IAT after applying the speedup.
            original_iar_ms = float(row["original_iar_ms"])
            scaled_iar_ms = original_iar_ms / speedup

            azure_func_name = str(row["func_name"])

            print(
                f"[AZURE_REQUEST] "
                f"request={req_index} "
                f"azure_function={azure_func_name} "
                f"scheduled={scheduled_time_sec:.3f}s "
                f"actual={current_time_sec:.3f}s "
                f"IAT={scaled_iar_ms:.2f}ms"
            )

            task = asyncio.create_task(
                send_request(
                    req_id,
                    default_deadline_ms,
                    scaled_iar_ms,
                    mode,
                    current_time_sec,
                )
            )

            tasks.append(task)
            req_index += 1

    else:
        raise ValueError(f"Unknown workload mode: {mode}")

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=False)

    print(f"Completed workload. Requests sent: {req_index}")