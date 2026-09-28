from __future__ import annotations

import asyncio
import time

import numpy as np

from config import (
    deadline_profiles_ms,
    iar_profiles_ms,
)
from client import send_request
from utils import generate_id


async def generate_workload(
    profile_name: str,
    duration_sec: float,
    mode: str = "vary_deadline",
) -> None:
    """
    mode = "vary_deadline":
        Change deadline and keep IAT fixed.

    mode = "vary_iar":
        Change IAT and keep deadline fixed.

    mode = "dynamic_iar":
        Use a multi-phase dynamic workload.
    """

    default_deadline_ms = 1000
    default_iar_ms = 100

    req_index = 0
    tasks: list[asyncio.Task] = []

    global_start = time.time()

    if mode == "vary_deadline":
        if profile_name not in deadline_profiles_ms:
            raise ValueError(
                f"Unknown profile: {profile_name}"
            )

        iar_sec = default_iar_ms / 1000.0
        deadlines = deadline_profiles_ms[profile_name]

        for deadline_ms in deadlines:
            phase_end = time.time() + duration_sec

            print(
                f"\n[VARY_DEADLINE] "
                f"deadline={deadline_ms}ms "
                f"IAT={default_iar_ms}ms "
                f"duration={duration_sec:.2f}s"
            )

            while time.time() < phase_end:
                req_id = generate_id(req_index)

                current_time_sec = (
                    time.time() - global_start
                )

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

                sleep_sec = np.random.exponential(
                    scale=iar_sec
                )

                await asyncio.sleep(sleep_sec)

    elif mode == "vary_iar":
        if profile_name not in iar_profiles_ms:
            raise ValueError(
                f"Unknown profile: {profile_name}"
            )

        iars = iar_profiles_ms[profile_name]

        for iar_ms in iars:
            iar_sec = iar_ms / 1000.0
            phase_end = time.time() + duration_sec

            print(
                f"\n[VARY_IAR] "
                f"IAT={iar_ms}ms "
                f"deadline={default_deadline_ms}ms "
                f"duration={duration_sec:.2f}s"
            )

            while time.time() < phase_end:
                req_id = generate_id(req_index)

                current_time_sec = (
                    time.time() - global_start
                )

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

                sleep_sec = np.random.exponential(
                    scale=iar_sec
                )

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

        total_ratio = sum(
            ratio
            for _, _, ratio in dynamic_trace
        )

        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(
                "Dynamic trace ratios must sum to 1.0. "
                f"Current sum={total_ratio}"
            )

        for phase_name, iar_ms, ratio in dynamic_trace:
            phase_duration_sec = duration_sec * ratio
            iar_sec = iar_ms / 1000.0
            phase_end = time.time() + phase_duration_sec

            print(
                f"\n[DYNAMIC_IAR] "
                f"phase={phase_name} "
                f"IAT={iar_ms}ms "
                f"duration={phase_duration_sec:.2f}s"
            )

            while time.time() < phase_end:
                req_id = generate_id(req_index)

                current_time_sec = (
                    time.time() - global_start
                )

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

                sleep_sec = np.random.exponential(
                    scale=iar_sec
                )

                await asyncio.sleep(sleep_sec)

    else:
        raise ValueError(
            f"Unsupported workload mode: {mode}"
        )

    print(
        f"\nWaiting for {len(tasks)} requests to finish..."
    )

    if tasks:
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        failures = sum(
            isinstance(result, Exception)
            for result in results
        )

        if failures:
            print(
                f"Workload task failures: {failures}"
            )

    print("\nWorkload completed.")
    print(f"Total submitted requests: {req_index}")


if __name__ == "__main__":
    # Change these values and right-click this file.
    PROFILE_NAME = "short"
    DURATION_SEC = 5
    MODE = "vary_deadline"

    asyncio.run(
        generate_workload(
            PROFILE_NAME,
            duration_sec=DURATION_SEC,
            mode=MODE,
        )
    )