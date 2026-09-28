# #!/usr/bin/env python3
#
# """
# replay_azure_workload.py
#
# Replays requests using the original IATs extracted from an Azure Functions
# production trace.
#
# Important behavior
# ------------------
# 1. Reads azure_original_iats.csv.
# 2. Uses the original iat_seconds values without scaling by default.
# 3. Schedules requests according to cumulative Azure arrival times.
# 4. Sends requests asynchronously so a slow inference request does not delay
#    later arrivals.
# 5. Saves request-level replay results to a CSV file.
#
# Expected HTTP request format
# ----------------------------
# POST /invoke
#
# {
#     "image_b64": "<base64 encoded image>",
#     "deadline": 2000,
#     "request_id": "AZURE-000001"
# }
#
# You can modify build_request_payload() if your server expects different keys.
# """
#
# from __future__ import annotations
#
# import base64
# import json
# import sys
# import threading
# import time
# from concurrent.futures import Future, ThreadPoolExecutor, as_completed
# from pathlib import Path
# from typing import Any
#
# import pandas as pd
# import requests
#
#
# # ============================================================
# # USER SETTINGS
# # Change this section for your experiment
# # ============================================================
#
# SCRIPT_DIRECTORY = Path(__file__).resolve().parent
#
# # Extracted Azure workload
# TRACE_FILE = SCRIPT_DIRECTORY / "azure_original_iats.csv"
#
# # Folder containing test images.
# # Subdirectories are searched automatically.
# IMAGE_DIRECTORY = Path(
#     "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/selected_images"
# )
#
# # DYSCI or inference-server endpoint
# SERVER_URL = "http://127.0.0.1:8080/invoke"
#
# # Deadline sent to the inference server, in milliseconds
# DEADLINE_MS = 2000
#
# # First request in the CSV to replay.
# # Zero means start with the first row.
# START_REQUEST_INDEX = 0
#
# # Use a small number first.
# #
# # Examples:
# #   100   -> quick test
# #   1000  -> larger experiment
# #   None  -> replay every row remaining after START_REQUEST_INDEX
# MAX_REQUESTS = 100
#
# # Number of HTTP requests that may execute simultaneously.
# #
# # This should be high enough that slow requests do not block later Azure
# # arrivals from being submitted.
# MAX_CONCURRENT_REQUESTS = 64
#
# # HTTP timeout for one request
# HTTP_TIMEOUT_SECONDS = 120
#
# # Original replay speed.
# #
# # Keep this at 1.0 for the reviewer experiment.
# #
# #   1.0 -> original Azure timing
# #   2.0 -> twice as fast
# #   0.5 -> twice as slow
# #
# # Changing this means the workload is no longer replayed at its original speed.
# REPLAY_SPEED = 1.0
#
# # Optional safety limit for long idle gaps.
# #
# # Keep this as None for an exact original replay.
# #
# # Example:
# #   MAX_SINGLE_IAT_SECONDS = 30.0
# #
# # This would reduce every gap larger than 30 seconds to 30 seconds, but that
# # would no longer be an exact reproduction of the Azure trace.
# MAX_SINGLE_IAT_SECONDS: float | None = None
#
# # Print one line for every completed request
# PRINT_EACH_RESULT = True
#
# # Stop sending new requests after this many consecutive submission errors.
# # Set to None to disable this safety condition.
# MAX_CONSECUTIVE_ERRORS: int | None = 20
#
# # Output file
# OUTPUT_FILE = SCRIPT_DIRECTORY / "azure_replay_results.csv"
#
# # Supported image extensions
# IMAGE_EXTENSIONS = {
#     ".jpg",
#     ".jpeg",
#     ".png",
#     ".bmp",
#     ".webp",
# }
#
#
# # ============================================================
# # SHARED SESSION MANAGEMENT
# # ============================================================
#
# _thread_local = threading.local()
#
#
# def get_http_session() -> requests.Session:
#     """
#     Return one requests.Session per worker thread.
#
#     Reusing connections reduces repeated TCP setup overhead.
#     """
#
#     if not hasattr(_thread_local, "session"):
#         session = requests.Session()
#
#         session.headers.update(
#             {
#                 "Content-Type": "application/json",
#                 "Accept": "application/json",
#             }
#         )
#
#         _thread_local.session = session
#
#     return _thread_local.session
#
#
# # ============================================================
# # INPUT VALIDATION
# # ============================================================
#
# def validate_settings() -> None:
#     """
#     Validate user-editable settings before beginning the replay.
#     """
#
#     if not TRACE_FILE.exists():
#         raise FileNotFoundError(
#             f"\nAzure trace CSV was not found:\n{TRACE_FILE}\n\n"
#             "Run extract_azure_iats.py first."
#         )
#
#     if not IMAGE_DIRECTORY.exists():
#         raise FileNotFoundError(
#             f"\nImage directory was not found:\n{IMAGE_DIRECTORY}\n\n"
#             "Update IMAGE_DIRECTORY at the top of this script."
#         )
#
#     if REPLAY_SPEED <= 0:
#         raise ValueError("REPLAY_SPEED must be greater than zero.")
#
#     if MAX_CONCURRENT_REQUESTS < 1:
#         raise ValueError(
#             "MAX_CONCURRENT_REQUESTS must be at least 1."
#         )
#
#     if START_REQUEST_INDEX < 0:
#         raise ValueError(
#             "START_REQUEST_INDEX cannot be negative."
#         )
#
#     if MAX_REQUESTS is not None and MAX_REQUESTS < 1:
#         raise ValueError(
#             "MAX_REQUESTS must be at least 1 or None."
#         )
#
#     if (
#         MAX_SINGLE_IAT_SECONDS is not None
#         and MAX_SINGLE_IAT_SECONDS < 0
#     ):
#         raise ValueError(
#             "MAX_SINGLE_IAT_SECONDS must be nonnegative or None."
#         )
#
#
# def load_trace() -> pd.DataFrame:
#     """
#     Load and validate the extracted Azure IAT trace.
#     """
#
#     print("=" * 76)
#     print("Loading Azure workload")
#     print("=" * 76)
#     print(f"Trace file: {TRACE_FILE}")
#
#     trace_df = pd.read_csv(TRACE_FILE)
#
#     required_columns = {
#         "request_index",
#         "iat_seconds",
#     }
#
#     missing_columns = required_columns.difference(
#         trace_df.columns
#     )
#
#     if missing_columns:
#         raise ValueError(
#             f"Missing required columns: {sorted(missing_columns)}\n"
#             f"Available columns: {list(trace_df.columns)}"
#         )
#
#     trace_df["request_index"] = pd.to_numeric(
#         trace_df["request_index"],
#         errors="coerce",
#     )
#
#     trace_df["iat_seconds"] = pd.to_numeric(
#         trace_df["iat_seconds"],
#         errors="coerce",
#     )
#
#     trace_df = trace_df.dropna(
#         subset=[
#             "request_index",
#             "iat_seconds",
#         ]
#     ).copy()
#
#     trace_df = trace_df.loc[
#         trace_df["iat_seconds"] >= 0
#     ].copy()
#
#     trace_df = trace_df.sort_values(
#         "request_index"
#     ).reset_index(drop=True)
#
#     if START_REQUEST_INDEX >= len(trace_df):
#         raise ValueError(
#             f"START_REQUEST_INDEX={START_REQUEST_INDEX} is outside "
#             f"the trace, which contains {len(trace_df):,} rows."
#         )
#
#     trace_df = trace_df.iloc[
#         START_REQUEST_INDEX:
#     ].copy()
#
#     if MAX_REQUESTS is not None:
#         trace_df = trace_df.head(
#             MAX_REQUESTS
#         ).copy()
#
#     trace_df = trace_df.reset_index(drop=True)
#
#     if trace_df.empty:
#         raise ValueError(
#             "No requests remain after applying the trace limits."
#         )
#
#     # The first selected request should be submitted immediately.
#     trace_df.loc[0, "iat_seconds"] = 0.0
#
#     if MAX_SINGLE_IAT_SECONDS is not None:
#         original_iats = trace_df["iat_seconds"].copy()
#
#         trace_df["iat_seconds"] = trace_df[
#             "iat_seconds"
#         ].clip(
#             upper=MAX_SINGLE_IAT_SECONDS
#         )
#
#         changed_count = int(
#             (
#                 trace_df["iat_seconds"]
#                 != original_iats
#             ).sum()
#         )
#
#         print(
#             f"Warning: capped {changed_count:,} IAT values at "
#             f"{MAX_SINGLE_IAT_SECONDS:g} seconds."
#         )
#         print(
#             "This is not an exact original-timing replay."
#         )
#
#     trace_df["scheduled_offset_seconds"] = (
#         trace_df["iat_seconds"].cumsum()
#         / REPLAY_SPEED
#     )
#
#     print(f"Requests selected: {len(trace_df):,}")
#     print(
#         "Scheduled replay duration: "
#         f"{trace_df['scheduled_offset_seconds'].iloc[-1]:.3f} seconds"
#     )
#
#     if REPLAY_SPEED == 1.0 and MAX_SINGLE_IAT_SECONDS is None:
#         print("Replay mode: original Azure timing")
#     else:
#         print(
#             "Replay mode: modified timing "
#             f"(speed={REPLAY_SPEED:g})"
#         )
#
#     return trace_df
#
#
# def discover_images() -> list[Path]:
#     """
#     Find all supported image files recursively.
#     """
#
#     image_paths = sorted(
#         path
#         for path in IMAGE_DIRECTORY.rglob("*")
#         if path.is_file()
#         and path.suffix.lower() in IMAGE_EXTENSIONS
#     )
#
#     if not image_paths:
#         raise FileNotFoundError(
#             f"No supported images were found under:\n"
#             f"{IMAGE_DIRECTORY}"
#         )
#
#     print(f"Images found: {len(image_paths):,}")
#
#     return image_paths
#
#
# # ============================================================
# # REQUEST CREATION
# # ============================================================
#
# def encode_image(image_path: Path) -> str:
#     """
#     Read an image and return its Base64 representation.
#     """
#
#     with image_path.open("rb") as image_file:
#         return base64.b64encode(
#             image_file.read()
#         ).decode("utf-8")
#
#
# def build_request_payload(
#     request_id: str,
#     image_b64: str,
#     image_path: Path,
#     trace_row: pd.Series,
# ) -> dict[str, Any]:
#     """
#     Create the JSON request sent to the inference server.
#
#     Modify this function if your /invoke endpoint expects another format.
#     """
#
#     return {
#         "request_id": request_id,
#         "image_b64": image_b64,
#         "deadline": DEADLINE_MS,
#
#         # These fields are optional. Remove them if your server rejects
#         # unknown JSON keys.
#         "source": "azure_trace",
#         "azure_request_index": int(
#             trace_row["request_index"]
#         ),
#         "image_name": image_path.name,
#     }
#
#
# # ============================================================
# # HTTP REQUEST EXECUTION
# # ============================================================
#
# def send_one_request(
#     request_number: int,
#     trace_row: pd.Series,
#     image_path: Path,
#     image_b64: str,
#     replay_start_monotonic: float,
#     scheduled_submit_monotonic: float,
# ) -> dict[str, Any]:
#     """
#     Send one inference request and collect timing information.
#     """
#
#     request_id = f"AZURE-{request_number:06d}"
#
#     actual_submit_monotonic = time.monotonic()
#     actual_submit_wall_time = time.time()
#
#     scheduling_delay_ms = (
#         actual_submit_monotonic
#         - scheduled_submit_monotonic
#     ) * 1000.0
#
#     payload = build_request_payload(
#         request_id=request_id,
#         image_b64=image_b64,
#         image_path=image_path,
#         trace_row=trace_row,
#     )
#
#     request_start = time.perf_counter()
#
#     status_code: int | None = None
#     response_json: Any = None
#     response_text = ""
#     error_message = ""
#     success = False
#
#     try:
#         session = get_http_session()
#
#         response = session.post(
#             SERVER_URL,
#             json=payload,
#             timeout=HTTP_TIMEOUT_SECONDS,
#         )
#
#         status_code = response.status_code
#         response_text = response.text
#
#         try:
#             response_json = response.json()
#         except ValueError:
#             response_json = None
#
#         success = response.ok
#
#         if not response.ok:
#             error_message = (
#                 f"HTTP {response.status_code}: "
#                 f"{response.text[:500]}"
#             )
#
#     except requests.RequestException as error:
#         error_message = str(error)
#
#     except Exception as error:
#         error_message = (
#             f"{type(error).__name__}: {error}"
#         )
#
#     completion_monotonic = time.monotonic()
#
#     client_latency_ms = (
#         time.perf_counter() - request_start
#     ) * 1000.0
#
#     actual_submit_offset_seconds = (
#         actual_submit_monotonic
#         - replay_start_monotonic
#     )
#
#     completion_offset_seconds = (
#         completion_monotonic
#         - replay_start_monotonic
#     )
#
#     response_json_string = ""
#
#     if response_json is not None:
#         try:
#             response_json_string = json.dumps(
#                 response_json,
#                 ensure_ascii=False,
#             )
#         except TypeError:
#             response_json_string = str(
#                 response_json
#             )
#
#     return {
#         "request_id": request_id,
#         "azure_request_index": int(
#             trace_row["request_index"]
#         ),
#         "original_iat_seconds": float(
#             trace_row["iat_seconds"]
#         ),
#         "scheduled_submit_offset_seconds": float(
#             trace_row["scheduled_offset_seconds"]
#         ),
#         "actual_submit_offset_seconds": (
#             actual_submit_offset_seconds
#         ),
#         "scheduling_delay_ms": scheduling_delay_ms,
#         "completion_offset_seconds": (
#             completion_offset_seconds
#         ),
#         "submit_wall_time_epoch": actual_submit_wall_time,
#         "image_path": str(image_path),
#         "deadline_ms": DEADLINE_MS,
#         "http_status": status_code,
#         "client_latency_ms": client_latency_ms,
#         "success": success,
#         "error": error_message,
#         "response_json": response_json_string,
#         "response_text": (
#             response_text[:2000]
#             if response_json is None
#             else ""
#         ),
#     }
#
#
# # ============================================================
# # PRELOAD IMAGES
# # ============================================================
#
# def preload_images(
#     image_paths: list[Path],
# ) -> dict[Path, str]:
#     """
#     Encode images before starting the replay.
#
#     This prevents disk reading and Base64 conversion from affecting request
#     arrival timing.
#     """
#
#     print()
#     print("=" * 76)
#     print("Preloading images")
#     print("=" * 76)
#
#     encoded_images: dict[Path, str] = {}
#
#     for image_number, image_path in enumerate(
#         image_paths,
#         start=1,
#     ):
#         encoded_images[image_path] = encode_image(
#             image_path
#         )
#
#         if (
#             image_number % 100 == 0
#             or image_number == len(image_paths)
#         ):
#             print(
#                 f"Encoded {image_number:,}/"
#                 f"{len(image_paths):,} images"
#             )
#
#     return encoded_images
#
#
# # ============================================================
# # SCHEDULER
# # ============================================================
#
# def sleep_until(
#     target_monotonic_time: float,
# ) -> None:
#     """
#     Wait until an absolute monotonic time.
#
#     Longer waits use sleep. The final short portion is checked more closely
#     to reduce scheduler drift.
#     """
#
#     while True:
#         remaining = (
#             target_monotonic_time
#             - time.monotonic()
#         )
#
#         if remaining <= 0:
#             return
#
#         if remaining > 0.01:
#             time.sleep(
#                 remaining - 0.005
#             )
#         else:
#             time.sleep(
#                 min(remaining, 0.001)
#             )
#
#
# def replay_workload(
#     trace_df: pd.DataFrame,
#     image_paths: list[Path],
#     encoded_images: dict[Path, str],
# ) -> pd.DataFrame:
#     """
#     Replay the Azure workload asynchronously.
#     """
#
#     print()
#     print("=" * 76)
#     print("Starting Azure workload replay")
#     print("=" * 76)
#     print(f"Server URL: {SERVER_URL}")
#     print(f"Deadline: {DEADLINE_MS} ms")
#     print(
#         f"Maximum concurrent requests: "
#         f"{MAX_CONCURRENT_REQUESTS}"
#     )
#     print()
#
#     futures: list[Future[dict[str, Any]]] = []
#
#     replay_start_monotonic = time.monotonic()
#     replay_start_wall_time = time.time()
#
#     submission_error_count = 0
#
#     with ThreadPoolExecutor(
#         max_workers=MAX_CONCURRENT_REQUESTS,
#         thread_name_prefix="azure-replay",
#     ) as executor:
#
#         for row_number, trace_row in trace_df.iterrows():
#             request_number = row_number + 1
#
#             scheduled_offset = float(
#                 trace_row["scheduled_offset_seconds"]
#             )
#
#             scheduled_submit_monotonic = (
#                 replay_start_monotonic
#                 + scheduled_offset
#             )
#
#             sleep_until(
#                 scheduled_submit_monotonic
#             )
#
#             image_path = image_paths[
#                 row_number % len(image_paths)
#             ]
#
#             try:
#                 future = executor.submit(
#                     send_one_request,
#                     request_number,
#                     trace_row.copy(),
#                     image_path,
#                     encoded_images[image_path],
#                     replay_start_monotonic,
#                     scheduled_submit_monotonic,
#                 )
#
#                 futures.append(future)
#                 submission_error_count = 0
#
#             except Exception as error:
#                 submission_error_count += 1
#
#                 print(
#                     f"Could not submit request "
#                     f"{request_number}: {error}"
#                 )
#
#                 if (
#                     MAX_CONSECUTIVE_ERRORS is not None
#                     and submission_error_count
#                     >= MAX_CONSECUTIVE_ERRORS
#                 ):
#                     print(
#                         "Stopping because the maximum number "
#                         "of consecutive submission errors was reached."
#                     )
#                     break
#
#             if (
#                 request_number % 100 == 0
#                 or request_number == len(trace_df)
#             ):
#                 elapsed = (
#                     time.monotonic()
#                     - replay_start_monotonic
#                 )
#
#                 print(
#                     f"Submitted {request_number:,}/"
#                     f"{len(trace_df):,} requests "
#                     f"after {elapsed:.2f} seconds"
#                 )
#
#         print()
#         print("All selected requests have been submitted.")
#         print("Waiting for outstanding responses...")
#
#         results: list[dict[str, Any]] = []
#
#         completed_count = 0
#
#         for future in as_completed(futures):
#             completed_count += 1
#
#             try:
#                 result = future.result()
#
#             except Exception as error:
#                 result = {
#                     "request_id": "UNKNOWN",
#                     "success": False,
#                     "error": (
#                         f"Unhandled worker error: "
#                         f"{type(error).__name__}: {error}"
#                     ),
#                 }
#
#             results.append(result)
#
#             if PRINT_EACH_RESULT:
#                 status_text = (
#                     "SUCCESS"
#                     if result.get("success")
#                     else "FAILED"
#                 )
#
#                 latency = result.get(
#                     "client_latency_ms"
#                 )
#
#                 latency_text = (
#                     f"{latency:.2f} ms"
#                     if isinstance(
#                         latency,
#                         (int, float),
#                     )
#                     else "unknown latency"
#                 )
#
#                 print(
#                     f"[{completed_count:,}/"
#                     f"{len(futures):,}] "
#                     f"{result.get('request_id')} "
#                     f"{status_text}, {latency_text}"
#                 )
#
#     replay_end_wall_time = time.time()
#
#     print()
#     print(
#         "Replay wall-clock duration: "
#         f"{replay_end_wall_time - replay_start_wall_time:.3f} seconds"
#     )
#
#     results_df = pd.DataFrame(
#         results
#     )
#
#     if not results_df.empty:
#         sort_columns = [
#             column
#             for column in [
#                 "azure_request_index",
#                 "request_id",
#             ]
#             if column in results_df.columns
#         ]
#
#         if sort_columns:
#             results_df = results_df.sort_values(
#                 sort_columns
#             ).reset_index(drop=True)
#
#     return results_df
#
#
# # ============================================================
# # OUTPUT AND SUMMARY
# # ============================================================
#
# def save_results(
#     results_df: pd.DataFrame,
# ) -> None:
#     """
#     Save request-level replay results.
#     """
#
#     OUTPUT_FILE.parent.mkdir(
#         parents=True,
#         exist_ok=True,
#     )
#
#     results_df.to_csv(
#         OUTPUT_FILE,
#         index=False,
#     )
#
#     print(f"Results saved to: {OUTPUT_FILE}")
#
#
# def print_results_summary(
#     results_df: pd.DataFrame,
# ) -> None:
#     """
#     Print replay accuracy and latency statistics.
#     """
#
#     print()
#     print("=" * 76)
#     print("Replay summary")
#     print("=" * 76)
#
#     if results_df.empty:
#         print("No completed request results were collected.")
#         return
#
#     total_requests = len(results_df)
#
#     if "success" in results_df.columns:
#         successful_requests = int(
#             results_df["success"].fillna(False).sum()
#         )
#     else:
#         successful_requests = 0
#
#     failed_requests = (
#         total_requests - successful_requests
#     )
#
#     print(f"Completed requests: {total_requests:,}")
#     print(f"Successful requests: {successful_requests:,}")
#     print(f"Failed requests: {failed_requests:,}")
#
#     success_rate = (
#         successful_requests
#         / total_requests
#         * 100.0
#     )
#
#     print(f"Success rate: {success_rate:.2f}%")
#
#     if "client_latency_ms" in results_df.columns:
#         latency = pd.to_numeric(
#             results_df["client_latency_ms"],
#             errors="coerce",
#         ).dropna()
#
#         if not latency.empty:
#             print()
#             print("Client-observed latency")
#             print("-" * 76)
#             print(f"Mean:   {latency.mean():.3f} ms")
#             print(f"Median: {latency.median():.3f} ms")
#             print(
#                 f"P90:    "
#                 f"{latency.quantile(0.90):.3f} ms"
#             )
#             print(
#                 f"P95:    "
#                 f"{latency.quantile(0.95):.3f} ms"
#             )
#             print(
#                 f"P99:    "
#                 f"{latency.quantile(0.99):.3f} ms"
#             )
#             print(f"Max:    {latency.max():.3f} ms")
#
#     if "scheduling_delay_ms" in results_df.columns:
#         scheduling_delay = pd.to_numeric(
#             results_df["scheduling_delay_ms"],
#             errors="coerce",
#         ).dropna()
#
#         if not scheduling_delay.empty:
#             print()
#             print("Replay scheduling delay")
#             print("-" * 76)
#             print(
#                 f"Mean: "
#                 f"{scheduling_delay.mean():.3f} ms"
#             )
#             print(
#                 f"P95:  "
#                 f"{scheduling_delay.quantile(0.95):.3f} ms"
#             )
#             print(
#                 f"P99:  "
#                 f"{scheduling_delay.quantile(0.99):.3f} ms"
#             )
#             print(
#                 f"Max:  "
#                 f"{scheduling_delay.max():.3f} ms"
#             )
#
#
# # ============================================================
# # MAIN
# # ============================================================
#
# def main() -> None:
#     validate_settings()
#
#     trace_df = load_trace()
#
#     image_paths = discover_images()
#
#     # Preload only the images that can be used by this replay.
#     images_needed = min(
#         len(image_paths),
#         len(trace_df),
#     )
#
#     selected_image_paths = image_paths[
#         :images_needed
#     ]
#
#     encoded_images = preload_images(
#         selected_image_paths
#     )
#
#     print()
#     print("=" * 76)
#     print("Replay configuration")
#     print("=" * 76)
#     print(f"Requests: {len(trace_df):,}")
#     print(f"Images used: {len(selected_image_paths):,}")
#     print(f"Replay speed: {REPLAY_SPEED:g}x")
#     print(
#         "Maximum single IAT: "
#         f"{MAX_SINGLE_IAT_SECONDS}"
#     )
#
#     if (
#         REPLAY_SPEED == 1.0
#         and MAX_SINGLE_IAT_SECONDS is None
#     ):
#         print(
#             "Timing status: exact original Azure IATs"
#         )
#     else:
#         print(
#             "Timing status: modified Azure IATs"
#         )
#
#     results_df = replay_workload(
#         trace_df=trace_df,
#         image_paths=selected_image_paths,
#         encoded_images=encoded_images,
#     )
#
#     save_results(
#         results_df
#     )
#
#     print_results_summary(
#         results_df
#     )
#
#     print()
#     print("=" * 76)
#     print("Completed")
#     print("=" * 76)
#
#
# if __name__ == "__main__":
#     try:
#         main()
#
#     except KeyboardInterrupt:
#         print()
#         print("Replay stopped by the user.")
#         sys.exit(130)
#
#     except Exception as error:
#         print()
#         print("=" * 76)
#         print("ERROR")
#         print("=" * 76)
#         print(
#             f"{type(error).__name__}: {error}"
#         )
#         print()
#         print(
#             "Check TRACE_FILE, IMAGE_DIRECTORY, SERVER_URL, "
#             "and the JSON format expected by your /invoke endpoint."
#         )
#
#         input("\nPress Enter to close...")
#
#         sys.exit(1)
#
#     input("\nPress Enter to close...")



#!/usr/bin/env python3

"""
replay_azure_workload.py

Replay the original request inter-arrival times extracted from the
Azure Functions production trace.

Expected input CSV:
    azure_original_iats.csv

Expected columns:
    request_index
    start_time_seconds
    iat_seconds

The Azure IAT values are replayed directly by default.
No exponential distribution is used.
"""

import asyncio
import time
from pathlib import Path

import pandas as pd


# ============================================================
# IMPORT YOUR EXISTING FUNCTIONS
# ============================================================

# Replace "workload_generator" with the name of your current Python file.
#
# Example:
# If your current file is:
#     generate_workload.py
#
# Change this to:
#     from generate_workload import generate_id, send_request

from workload_generator.generator import generate_id, send_request


# ============================================================
# USER SETTINGS
# ============================================================

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

AZURE_IAT_FILE = SCRIPT_DIRECTORY / "azure_original_iats.csv"

# Fixed deadline assigned to every Azure-trace request.
DEFAULT_DEADLINE_MS = 2000

# Number of requests to replay.
#
# Examples:
#   1000  = replay the first 1,000 requests
#   5000  = replay the first 5,000 requests
#   None  = replay all requests in the CSV
MAX_REQUESTS = 1000

# Starting row in the extracted Azure trace.
#
# 0 means begin from the first request.
START_ROW = 0

# Replay speed:
#
#   1.0 = original Azure timing
#   2.0 = twice as fast
#   5.0 = five times as fast
#
# For the paper's real-trace evaluation, use:
#   REPLAY_SPEED = 1.0
REPLAY_SPEED = 1.0

# Optional gap limit.
#
# None means preserve every original Azure idle gap.
#
# Setting this to a number modifies the original trace. Therefore,
# keep it as None for the primary production-trace experiment.
#
# During code testing, you could temporarily use:
#   MAX_SLEEP_SECONDS = 5.0
MAX_SLEEP_SECONDS = 3.0

# Workload mode saved in your result CSV.
WORKLOAD_MODE = "azure_trace"

# Print one line for every request.
PRINT_EVERY_REQUEST = True

# Print progress after this number of requests.
PROGRESS_INTERVAL = 100

# Wait for all outstanding requests after the final arrival.
WAIT_FOR_ALL_REQUESTS = True


# ============================================================
# LOAD AZURE IATS
# ============================================================

def load_azure_trace() -> pd.DataFrame:
    """
    Load and validate the extracted Azure IAT CSV.
    """

    if not AZURE_IAT_FILE.exists():
        raise FileNotFoundError(
            f"\nAzure IAT file was not found:\n"
            f"{AZURE_IAT_FILE}\n\n"
            "Run extract_azure_iats.py first."
        )

    print("=" * 80)
    print("Loading Azure production trace")
    print("=" * 80)
    print(f"Input file: {AZURE_IAT_FILE}")

    df = pd.read_csv(AZURE_IAT_FILE)

    required_columns = {
        "request_index",
        "start_time_seconds",
        "iat_seconds",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}\n"
            f"Available columns: {list(df.columns)}"
        )

    for column in required_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    before_cleaning = len(df)

    df = df.dropna(
        subset=[
            "request_index",
            "start_time_seconds",
            "iat_seconds",
        ]
    ).copy()

    invalid_rows = before_cleaning - len(df)

    if invalid_rows > 0:
        print(f"Removed invalid rows: {invalid_rows:,}")

    negative_iats = int(
        (df["iat_seconds"] < 0).sum()
    )

    if negative_iats > 0:
        print(f"Removed negative-IAT rows: {negative_iats:,}")

        df = df.loc[
            df["iat_seconds"] >= 0
        ].copy()

    df = df.sort_values(
        by="request_index"
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid Azure requests were found in the input CSV."
        )

    if START_ROW < 0:
        raise ValueError(
            "START_ROW cannot be negative."
        )

    if START_ROW >= len(df):
        raise ValueError(
            f"START_ROW={START_ROW} is outside the trace. "
            f"The trace contains {len(df):,} rows."
        )

    df = df.iloc[START_ROW:].copy()

    if MAX_REQUESTS is not None:
        if MAX_REQUESTS <= 0:
            raise ValueError(
                "MAX_REQUESTS must be greater than zero or None."
            )

        df = df.head(MAX_REQUESTS).copy()

    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No requests remain after applying START_ROW and "
            "MAX_REQUESTS."
        )

    return df


# ============================================================
# TRACE SUMMARY
# ============================================================

def print_trace_summary(df: pd.DataFrame) -> None:
    """
    Print information about the selected replay segment.
    """

    positive_iats = df.loc[
        df["iat_seconds"] > 0,
        "iat_seconds",
    ]

    total_original_wait = float(
        df["iat_seconds"].sum()
    )

    estimated_replay_wait = (
        total_original_wait / REPLAY_SPEED
    )

    if MAX_SLEEP_SECONDS is not None:
        estimated_replay_wait = float(
            (
                df["iat_seconds"] / REPLAY_SPEED
            ).clip(
                upper=MAX_SLEEP_SECONDS
            ).sum()
        )

    print()
    print("=" * 80)
    print("Azure replay configuration")
    print("=" * 80)

    print(f"Requests selected:       {len(df):,}")
    print(f"Starting trace row:      {START_ROW}")
    print(f"Deadline:                {DEFAULT_DEADLINE_MS} ms")
    print(f"Replay speed:            {REPLAY_SPEED:g}x")
    print(f"Original total IAT:      {total_original_wait:.3f} seconds")
    print(f"Expected replay waiting: {estimated_replay_wait:.3f} seconds")

    if MAX_SLEEP_SECONDS is None:
        print("Maximum sleep:           None")
        print("Trace modification:      No IAT gap capping")
    else:
        print(
            f"Maximum sleep:           "
            f"{MAX_SLEEP_SECONDS:.3f} seconds"
        )
        print("Trace modification:      Long gaps are capped")

    if not positive_iats.empty:
        print()
        print("Selected IAT statistics")
        print("-" * 80)
        print(
            f"Mean:   {positive_iats.mean():.6f} seconds"
        )
        print(
            f"Median: {positive_iats.median():.6f} seconds"
        )
        print(
            f"P90:    {positive_iats.quantile(0.90):.6f} seconds"
        )
        print(
            f"P95:    {positive_iats.quantile(0.95):.6f} seconds"
        )
        print(
            f"P99:    {positive_iats.quantile(0.99):.6f} seconds"
        )
        print(
            f"Maximum:{positive_iats.max():.6f} seconds"
        )


# ============================================================
# REPLAY TIMING
# ============================================================

def prepare_replay_delays(df: pd.DataFrame) -> list[float]:
    """
    Convert the original Azure IATs to replay delays.

    With REPLAY_SPEED=1.0, the delays are unchanged.

    With REPLAY_SPEED=2.0, each delay is divided by two.
    """

    if REPLAY_SPEED <= 0:
        raise ValueError(
            "REPLAY_SPEED must be greater than zero."
        )

    delays = (
        df["iat_seconds"].astype(float) / REPLAY_SPEED
    )

    if MAX_SLEEP_SECONDS is not None:
        if MAX_SLEEP_SECONDS < 0:
            raise ValueError(
                "MAX_SLEEP_SECONDS cannot be negative."
            )

        delays = delays.clip(
            upper=MAX_SLEEP_SECONDS
        )

    return delays.tolist()


# ============================================================
# AZURE WORKLOAD REPLAY
# ============================================================

async def replay_azure_workload() -> None:
    """
    Replay requests using the Azure production-trace IAT sequence.

    Requests are created asynchronously. Therefore, a slow response does
    not prevent subsequent requests from arriving at their intended times.
    """

    df = load_azure_trace()

    print_trace_summary(df)

    replay_delays = prepare_replay_delays(df)

    tasks = []

    replay_start_wall = time.time()
    replay_start_monotonic = time.perf_counter()

    # This stores the intended cumulative arrival time.
    target_elapsed_seconds = 0.0

    print()
    print("=" * 80)
    print("Starting Azure trace replay")
    print("=" * 80)

    for local_index, row in df.iterrows():

        original_iat_seconds = float(
            row["iat_seconds"]
        )

        replay_delay_seconds = float(
            replay_delays[local_index]
        )

        # The first extracted row normally has IAT=0 and is sent
        # immediately. Every later request waits for its trace IAT.
        target_elapsed_seconds += replay_delay_seconds

        current_elapsed = (
            time.perf_counter() - replay_start_monotonic
        )

        remaining_sleep = (
            target_elapsed_seconds - current_elapsed
        )

        if remaining_sleep > 0:
            await asyncio.sleep(
                remaining_sleep
            )

        actual_elapsed_seconds = (
            time.perf_counter() - replay_start_monotonic
        )

        req_id = generate_id(local_index)

        # Your existing send_request() writes this value to its IAR/IAT
        # column. It expects milliseconds in your current generator.
        original_iat_ms = (
            original_iat_seconds * 1000.0
        )

        task = asyncio.create_task(
            send_request(
                req_id,
                DEFAULT_DEADLINE_MS,
                original_iat_ms,
                WORKLOAD_MODE,
                actual_elapsed_seconds,
            )
        )

        tasks.append(task)

        if PRINT_EVERY_REQUEST:
            scheduling_lag_ms = (
                actual_elapsed_seconds
                - target_elapsed_seconds
            ) * 1000.0

            print(
                f"[AZURE_TRACE] "
                f"request={local_index + 1}/{len(df)} "
                f"id={req_id} "
                f"original_iat={original_iat_seconds:.6f}s "
                f"replay_delay={replay_delay_seconds:.6f}s "
                f"arrival_time={actual_elapsed_seconds:.6f}s "
                f"scheduling_lag={scheduling_lag_ms:.3f}ms"
            )

        elif (
            (local_index + 1) % PROGRESS_INTERVAL == 0
            or local_index + 1 == len(df)
        ):
            print(
                f"[AZURE_TRACE] Submitted "
                f"{local_index + 1:,}/{len(df):,} requests; "
                f"elapsed={actual_elapsed_seconds:.2f}s"
            )

    submission_end = time.perf_counter()

    total_submission_time = (
        submission_end - replay_start_monotonic
    )

    print()
    print("=" * 80)
    print("All Azure arrivals have been submitted")
    print("=" * 80)
    print(f"Submitted requests: {len(tasks):,}")
    print(
        f"Submission duration: "
        f"{total_submission_time:.3f} seconds"
    )

    if WAIT_FOR_ALL_REQUESTS and tasks:
        print("Waiting for all request responses...")

        # return_exceptions=True prevents one failed request from
        # cancelling the complete experiment.
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        failed_tasks = sum(
            isinstance(result, Exception)
            for result in results
        )

        successful_tasks = (
            len(results) - failed_tasks
        )

        total_experiment_time = (
            time.time() - replay_start_wall
        )

        print()
        print("=" * 80)
        print("Azure replay completed")
        print("=" * 80)
        print(f"Completed tasks: {len(results):,}")
        print(f"Successful tasks: {successful_tasks:,}")
        print(f"Task exceptions:  {failed_tasks:,}")
        print(
            f"Total experiment time: "
            f"{total_experiment_time:.3f} seconds"
        )

        if failed_tasks > 0:
            print()
            print("Task exceptions:")

            for result in results:
                if isinstance(result, Exception):
                    print(f"  - {result}")

    else:
        print(
            "WAIT_FOR_ALL_REQUESTS=False; the program is not "
            "waiting for request responses."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    try:
        asyncio.run(
            replay_azure_workload()
        )

    except KeyboardInterrupt:
        print()
        print("Azure replay stopped by the user.")

    except Exception as error:
        print()
        print("=" * 80)
        print("AZURE REPLAY ERROR")
        print("=" * 80)
        print(error)
        raise