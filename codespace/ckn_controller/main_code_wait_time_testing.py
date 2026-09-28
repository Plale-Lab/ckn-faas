import base64
import csv
import json
import multiprocessing
import os
import time
import uuid

from collections import defaultdict
from concurrent.futures import (
    Future,
    ThreadPoolExecutor,
)
from threading import Lock
from typing import Dict, List

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller import ckn_config as config

from ckn_controller.wait_time_iluvatar_debug import (
    IluvatarWaitEstimator,
)


# ============================================================
# Image directory
# ============================================================

IMAGE_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/"
    "clone main/ckn-faas/ckn_data/images/tem"
)


# ============================================================
# Workload settings
# ============================================================

IAT_MS = 100

REQUESTS_PER_MODEL = 150


# ============================================================
# Models that receive ACTUAL inference workload
# ============================================================

MODELS_TO_EXECUTE: List[str] = [

    # "mobilenet_v3_small",
    # "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    # "resnet34",
    # "resnet50",
    # "resnet101",
    # "efficientnet_b3",
    # "convnext_tiny",
    # "vit_b_16",
]


# ============================================================
# Models whose wait state is recorded for EVERY request
# ============================================================

MODELS_TO_MONITOR: List[str] = [

    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
]


# ============================================================
# Thread settings
# ============================================================

MAX_WORKERS_PER_MODEL = max(
    32,
    REQUESTS_PER_MODEL,
)

ESTIMATOR_MAX_WORKERS = 128


# ============================================================
# ONE output CSV
# ============================================================

# CSV_OUTPUT_FILE = (
#     f"iluvatar_latency_wait_snapshot_"
#     f"iat_{IAT_MS}ms_"
#     f"requests_{REQUESTS_PER_MODEL}.csv"
# )


CSV_OUTPUT_FILE = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/"
    # "/Users/agamage/Desktop/D2I/Codes Original/Fass Edge Cascades/"
    "2models_iluvatar_latency_wait_snapshot_iat_100ms_requests_100.csv"
)

# ============================================================
# Multi-node configuration
# ============================================================

USE_MULTI_ILUVATAR_INSTANCES = bool(
    getattr(
        config,
        "USE_MULTI_ILUVATAR_INSTANCES",
        False,
    )
)

ILUVATAR_NODES = getattr(
    config,
    "ILUVATAR_NODES",
    {},
)


# ============================================================
# Legacy configuration
# ============================================================

USE_TWO_ILUVATAR_INSTANCES = bool(
    getattr(
        config,
        "USE_TWO_ILUVATAR_INSTANCES",
        False,
    )
)

SERVER_ADDRESS = getattr(
    config,
    "SERVER_ADDRESS",
    None,
)

SMALL_MODEL_SERVER_ADDRESS = getattr(
    config,
    "SMALL_MODEL_SERVER_ADDRESS",
    SERVER_ADDRESS,
)

LARGE_MODEL_SERVER_ADDRESS = getattr(
    config,
    "LARGE_MODEL_SERVER_ADDRESS",
    SERVER_ADDRESS,
)

SMALL_INSTANCE_MODELS = list(
    getattr(
        config,
        "SMALL_INSTANCE_MODELS",
        [],
    )
)

LARGE_INSTANCE_MODELS = list(
    getattr(
        config,
        "LARGE_INSTANCE_MODELS",
        [],
    )
)


# ============================================================
# gRPC caches
# ============================================================

_worker_map: Dict[
    str,
    object,
] = {}

_channel_map: Dict[
    str,
    grpc.Channel,
] = {}

_estimator_map: Dict[
    str,
    IluvatarWaitEstimator,
] = {}

_cache_lock = Lock()


# ============================================================
# Global estimator executor
# ============================================================

_estimator_executor = (
    ThreadPoolExecutor(
        max_workers=(
            ESTIMATOR_MAX_WORKERS
        )
    )
)


# ============================================================
# Get cached worker + estimator
# ============================================================

def get_rpc_clients(
    server_address: str,
):

    with _cache_lock:

        if (
            server_address
            not in _worker_map
        ):

            channel = (
                grpc.insecure_channel(
                    server_address
                )
            )

            worker = (
                pb2_grpc.IluvatarWorkerStub(
                    channel
                )
            )

            estimator = (
                IluvatarWaitEstimator(
                    server_address=(
                        server_address
                    ),
                    channel=channel,
                )
            )

            _channel_map[
                server_address
            ] = channel

            _worker_map[
                server_address
            ] = worker

            _estimator_map[
                server_address
            ] = estimator

    return (
        _worker_map[
            server_address
        ],
        _estimator_map[
            server_address
        ],
    )


def get_worker_for_address(
    server_address: str,
):

    worker, _ = (
        get_rpc_clients(
            server_address
        )
    )

    return worker


def get_estimator_for_address(
    server_address: str,
):

    _, estimator = (
        get_rpc_clients(
            server_address
        )
    )

    return estimator


def close_all_channels():

    with _cache_lock:

        for channel in (
            _channel_map.values()
        ):

            try:
                channel.close()

            except Exception:
                pass

        _worker_map.clear()

        _estimator_map.clear()

        _channel_map.clear()


# ============================================================
# Validate multi-node config
# ============================================================

def validate_multi_node_config():

    if not isinstance(
        ILUVATAR_NODES,
        dict,
    ):

        raise ValueError(
            "ILUVATAR_NODES must be a dictionary."
        )

    if not ILUVATAR_NODES:

        raise ValueError(
            "USE_MULTI_ILUVATAR_INSTANCES=True "
            "but ILUVATAR_NODES is empty."
        )

    for (
        node_id,
        node,
    ) in ILUVATAR_NODES.items():

        if not isinstance(
            node,
            dict,
        ):

            raise ValueError(
                f"Node '{node_id}' must be a dictionary."
            )

        if not node.get(
            "address"
        ):

            raise ValueError(
                f"Node '{node_id}' has no address."
            )

        models = node.get(
            "models",
            [],
        )

        if not isinstance(
            models,
            list,
        ):

            raise ValueError(
                f"Node '{node_id}' models must be a list."
            )


# ============================================================
# Build legacy nodes
# ============================================================

def build_legacy_nodes():

    if not USE_TWO_ILUVATAR_INSTANCES:

        return {

            "single_instance": {

                "type":
                    "single",

                "address":
                    SERVER_ADDRESS,

                "models":
                    list(
                        set(
                            MODELS_TO_EXECUTE
                            + MODELS_TO_MONITOR
                        )
                    ),

                "enabled":
                    True,
            }
        }

    return {

        "edge_1": {

            "type":
                "edge",

            "address":
                SMALL_MODEL_SERVER_ADDRESS,

            "models":
                list(
                    SMALL_INSTANCE_MODELS
                ),

            "enabled":
                True,
        },

        "cloud_1": {

            "type":
                "cloud",

            "address":
                LARGE_MODEL_SERVER_ADDRESS,

            "models":
                list(
                    LARGE_INSTANCE_MODELS
                ),

            "enabled":
                True,
        },
    }


# ============================================================
# Active nodes
# ============================================================

def get_active_nodes():

    if USE_MULTI_ILUVATAR_INSTANCES:

        validate_multi_node_config()

        return ILUVATAR_NODES

    return build_legacy_nodes()


# ============================================================
# Actual execution targets
# ============================================================

def build_execution_targets():

    active_nodes = (
        get_active_nodes()
    )

    targets = []

    allowed_models = set(
        MODELS_TO_EXECUTE
    )

    for (
        node_id,
        node_config,
    ) in active_nodes.items():

        if not bool(
            node_config.get(
                "enabled",
                True,
            )
        ):
            continue

        node_models = list(
            node_config.get(
                "models",
                [],
            )
        )

        for model_name in node_models:

            if (
                model_name
                not in allowed_models
            ):
                continue

            targets.append(
                {

                    "node_id":
                        node_id,

                    "node_type":
                        str(
                            node_config.get(
                                "type",
                                "unknown",
                            )
                        ),

                    "server_address":
                        str(
                            node_config[
                                "address"
                            ]
                        ),

                    "model_name":
                        model_name,
                }
            )

    return targets


# ============================================================
# Wait-time monitoring targets
# ============================================================

def build_monitoring_targets():

    active_nodes = (
        get_active_nodes()
    )

    targets = []

    allowed_models = set(
        MODELS_TO_MONITOR
    )

    for (
        node_id,
        node_config,
    ) in active_nodes.items():

        if not bool(
            node_config.get(
                "enabled",
                True,
            )
        ):
            continue

        node_models = list(
            node_config.get(
                "models",
                [],
            )
        )

        for model_name in node_models:

            if (
                model_name
                not in allowed_models
            ):
                continue

            targets.append(
                {

                    "node_id":
                        node_id,

                    "node_type":
                        str(
                            node_config.get(
                                "type",
                                "unknown",
                            )
                        ),

                    "server_address":
                        str(
                            node_config[
                                "address"
                            ]
                        ),

                    "model_name":
                        model_name,
                }
            )

    return targets


# ============================================================
# Images
# ============================================================

def read_image_as_base64(
    image_path: str,
) -> str:

    with open(
        image_path,
        "rb",
    ) as image_file:

        return base64.b64encode(
            image_file.read()
        ).decode(
            "utf-8"
        )


def get_image_paths():

    if not os.path.isdir(
        IMAGE_DIR
    ):

        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{IMAGE_DIR}"
        )

    image_paths = []

    for filename in sorted(
        os.listdir(
            IMAGE_DIR
        )
    ):

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
            )
        ):
            continue

        image_paths.append(
            os.path.join(
                IMAGE_DIR,
                filename,
            )
        )

    return image_paths


# ============================================================
# Query ONE model wait state
# ============================================================

def query_one_wait_time(
    request_id: str,
    request_number: int,
    executed_node_id: str,
    executed_model: str,
    request_arrival_ms: float,
    monitoring_target: dict,
) -> dict:

    monitor_node_id = (
        monitoring_target[
            "node_id"
        ]
    )

    monitor_node_type = (
        monitoring_target[
            "node_type"
        ]
    )

    monitor_address = (
        monitoring_target[
            "server_address"
        ]
    )

    monitored_model = (
        monitoring_target[
            "model_name"
        ]
    )

    estimator = (
        get_estimator_for_address(
            monitor_address
        )
    )

    snapshot_start = (
        time.perf_counter()
    )

    details = (
        estimator.query(
            monitored_model,
    compute_type="gpu"
        )
    )

    snapshot_end = (
        time.perf_counter()
    )

    details.update(
        {

            "request_id":
                request_id,

            "request_number":
                request_number,

            "executed_node_id":
                executed_node_id,

            "executed_model":
                executed_model,

            "monitored_node_id":
                monitor_node_id,

            "monitored_node_type":
                monitor_node_type,

            "monitored_server_address":
                monitor_address,

            "monitored_model":
                monitored_model,

            "request_arrival_ms":
                request_arrival_ms,

            "snapshot_total_ms":
                (
                    snapshot_end
                    - snapshot_start
                )
                * 1000.0,
        }
    )

    return details


# ============================================================
# Launch ALL wait queries in parallel
# ============================================================

def launch_all_wait_queries(
    request_id: str,
    request_number: int,
    executed_node_id: str,
    executed_model: str,
    request_arrival_ms: float,
    monitoring_targets: List[dict],
) -> List[Future]:

    futures = []

    for target in (
        monitoring_targets
    ):

        future = (
            _estimator_executor.submit(

                query_one_wait_time,

                request_id,
                request_number,

                executed_node_id,
                executed_model,

                request_arrival_ms,

                target,
            )
        )

        futures.append(
            future
        )

    return futures


# ============================================================
# Send ONE actual inference request
# ============================================================

def send_request(
    request_id: str,
    request_number: int,
    node_id: str,
    node_type: str,
    server_address: str,
    model_name: str,
    image_path: str,
    request_arrival_ms: float,
    monitoring_targets: List[dict],
) -> dict:

    wait_futures = []

    try:

        worker = (
            get_worker_for_address(
                server_address
            )
        )

        # ----------------------------------------------------
        # Prepare inference input
        # ----------------------------------------------------

        image_b64 = (
            read_image_as_base64(
                image_path
            )
        )

        request = pb2.InvokeRequest(

            function_name=(
                model_name
            ),

            function_version="1",

            json_args=json.dumps(
                {

                    "model_name":
                        model_name,

                    "image_data":
                        image_b64,
                }
            ),

            transaction_id=(
                request_id
            ),
        )

        # ====================================================
        # Launch estimator queries for ALL models.
        #
        # Important:
        # We do NOT wait for them here.
        # ====================================================

        estimator_launch_start = (
            time.perf_counter()
        )

        wait_futures = (
            launch_all_wait_queries(

                request_id=(
                    request_id
                ),

                request_number=(
                    request_number
                ),

                executed_node_id=(
                    node_id
                ),

                executed_model=(
                    model_name
                ),

                request_arrival_ms=(
                    request_arrival_ms
                ),

                monitoring_targets=(
                    monitoring_targets
                ),
            )
        )

        estimator_launch_end = (
            time.perf_counter()
        )

        estimator_launch_overhead_ms = (
            estimator_launch_end
            - estimator_launch_start
        ) * 1000.0

        # ====================================================
        # Send actual inference immediately.
        # ====================================================

        invoke_start = (
            time.perf_counter()
        )

        response = (
            worker.invoke(
                request
            )
        )

        invoke_end = (
            time.perf_counter()
        )

        wall_latency_ms = (
            invoke_end
            - invoke_start
        ) * 1000.0

        rpc_latency_ms = (
            float(
                response.duration_us
            )
            / 1000.0
        )

        # ====================================================
        # Now collect estimator responses.
        # ====================================================

        wait_snapshot = []

        for future in (
            wait_futures
        ):

            try:

                wait_result = (
                    future.result()
                )

                wait_snapshot.append(
                    wait_result
                )

            except Exception as error:

                print(
                    f"[WAIT_QUERY_FAIL] "
                    f"request={request_number} "
                    f"error={error}"
                )

        # ----------------------------------------------------
        # Parse inference result
        # ----------------------------------------------------

        result_json = (
            json.loads(
                response.json_result
            )
        )

        if (
            "body"
            not in result_json
        ):

            raise ValueError(
                f"Unexpected response: "
                f"{result_json}"
            )

        body = (
            result_json[
                "body"
            ]
        )

        if (
            "Probability"
            not in body
        ):

            raise ValueError(
                f"Inference failed: "
                f"{result_json}"
            )

        probability = float(
            body[
                "Probability"
            ]
        )

        label = body.get(
            "Prediction Class",
            "UNKNOWN",
        )

        return {

            "request_id":
                request_id,

            "request_number":
                request_number,

            "node_id":
                node_id,

            "node_type":
                node_type,

            "server_address":
                server_address,

            "model_name":
                model_name,

            "image_name":
                os.path.basename(
                    image_path
                ),

            "request_arrival_ms":
                request_arrival_ms,

            "estimator_launch_overhead_ms":
                estimator_launch_overhead_ms,

            "wall_latency_ms":
                wall_latency_ms,

            "rpc_latency_ms":
                rpc_latency_ms,

            "probability":
                probability,

            "label":
                label,

            "success":
                bool(
                    response.success
                ),

            "container_state":
                pb2.ContainerState.Name(
                    response.container_state
                ),

            "wait_snapshot":
                wait_snapshot,

            "error":
                None,
        }

    except grpc.RpcError as error:

        try:

            error_message = (
                error.details()
            )

        except Exception:

            error_message = str(
                error
            )

        return {

            "request_id":
                request_id,

            "request_number":
                request_number,

            "node_id":
                node_id,

            "node_type":
                node_type,

            "server_address":
                server_address,

            "model_name":
                model_name,

            "image_name":
                os.path.basename(
                    image_path
                ),

            "request_arrival_ms":
                request_arrival_ms,

            "estimator_launch_overhead_ms":
                None,

            "wall_latency_ms":
                None,

            "rpc_latency_ms":
                None,

            "probability":
                None,

            "label":
                None,

            "success":
                False,

            "container_state":
                None,

            "wait_snapshot":
                [],

            "error":
                error_message,
        }

    except Exception as error:

        return {

            "request_id":
                request_id,

            "request_number":
                request_number,

            "node_id":
                node_id,

            "node_type":
                node_type,

            "server_address":
                server_address,

            "model_name":
                model_name,

            "image_name":
                os.path.basename(
                    image_path
                ),

            "request_arrival_ms":
                request_arrival_ms,

            "estimator_launch_overhead_ms":
                None,

            "wall_latency_ms":
                None,

            "rpc_latency_ms":
                None,

            "probability":
                None,

            "label":
                None,

            "success":
                False,

            "container_state":
                None,

            "wait_snapshot":
                [],

            "error":
                str(
                    error
                ),
        }


# ============================================================
# Fixed-IAT workload for ONE execution target
# ============================================================

def execute_model_at_fixed_iat(
    target: dict,
    image_paths: List[str],
    monitoring_targets: List[dict],
) -> List[dict]:

    node_id = (
        target[
            "node_id"
        ]
    )

    node_type = (
        target[
            "node_type"
        ]
    )

    server_address = (
        target[
            "server_address"
        ]
    )

    model_name = (
        target[
            "model_name"
        ]
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"EXECUTING MODEL: "
        f"{model_name}"
    )

    print(
        f"Node: "
        f"{node_id}"
    )

    print(
        f"Server: "
        f"{server_address}"
    )

    print(
        f"IAT: "
        f"{IAT_MS} ms"
    )

    print(
        f"Requests: "
        f"{REQUESTS_PER_MODEL}"
    )

    print(
        f"Wait-time deployments monitored/request: "
        f"{len(monitoring_targets)}"
    )

    print(
        "=" * 80
    )

    results = []

    iat_sec = (
        IAT_MS
        / 1000.0
    )

    with ThreadPoolExecutor(
        max_workers=(
            MAX_WORKERS_PER_MODEL
        )
    ) as executor:

        futures = []

        workload_start = (
            time.perf_counter()
        )

        for request_index in range(
            REQUESTS_PER_MODEL
        ):

            # ------------------------------------------------
            # Absolute fixed-IAT submission time
            # ------------------------------------------------

            target_submit_time = (
                workload_start
                + (
                    request_index
                    * iat_sec
                )
            )

            sleep_time = (
                target_submit_time
                - time.perf_counter()
            )

            if sleep_time > 0:

                time.sleep(
                    sleep_time
                )

            actual_submit = (
                time.perf_counter()
            )

            relative_submit_ms = (
                actual_submit
                - workload_start
            ) * 1000.0

            request_number = (
                request_index
                + 1
            )

            request_id = str(
                uuid.uuid4()
            )

            image_path = (
                image_paths[
                    request_index
                    % len(
                        image_paths
                    )
                ]
            )

            print(
                f"[SUBMIT] "
                f"model={model_name} "
                f"request="
                f"{request_number}/"
                f"{REQUESTS_PER_MODEL} "
                f"target="
                f"{request_index * IAT_MS:.2f}ms "
                f"actual="
                f"{relative_submit_ms:.2f}ms"
            )

            future = executor.submit(

                send_request,

                request_id,
                request_number,

                node_id,
                node_type,
                server_address,

                model_name,
                image_path,

                relative_submit_ms,

                monitoring_targets,
            )

            futures.append(
                future
            )

        # ----------------------------------------------------
        # Collect results
        # ----------------------------------------------------

        for future in futures:

            result = (
                future.result()
            )

            results.append(
                result
            )

            if (
                result.get(
                    "error"
                )
                is not None
            ):

                print(
                    f"[EXECUTE_FAIL] "
                    f"model={model_name} "
                    f"request="
                    f"{result['request_number']} "
                    f"error="
                    f"{result['error']}"
                )

                continue

            print(
                f"[EXECUTE_OK] "
                f"model={model_name} "
                f"request="
                f"{result['request_number']} "
                f"e2e="
                f"{result['wall_latency_ms']:.2f}ms "
                f"rpc="
                f"{result['rpc_latency_ms']:.2f}ms "
                f"wait_samples="
                f"{len(result['wait_snapshot'])}"
            )

    # ========================================================
    # Model summary
    # ========================================================

    successful = [

        result

        for result in results

        if (
            result.get(
                "error"
            ) is None
            and result.get(
                "wall_latency_ms"
            ) is not None
        )
    ]

    latencies = [

        float(
            result[
                "wall_latency_ms"
            ]
        )

        for result
        in successful
    ]

    print(
        "\n"
        + "-" * 80
    )

    if latencies:

        print(
            f"[MODEL COMPLETE] "
            f"{model_name}"
        )

        print(
            f"Successful requests: "
            f"{len(successful)}/"
            f"{REQUESTS_PER_MODEL}"
        )

        print(
            f"Average E2E latency: "
            f"{sum(latencies) / len(latencies):.2f} ms"
        )

        print(
            f"Minimum E2E latency: "
            f"{min(latencies):.2f} ms"
        )

        print(
            f"Maximum E2E latency: "
            f"{max(latencies):.2f} ms"
        )

    else:

        print(
            f"No successful requests "
            f"for {model_name}"
        )

    print(
        "-" * 80
    )

    return results


# ============================================================
# Average helper
# ============================================================

def average(
    values,
):

    values = [

        float(value)

        for value in values

        if value is not None
    ]

    if not values:
        return None

    return (
        sum(values)
        / len(values)
    )


# ============================================================
# Safe model/node prefix for CSV columns
# ============================================================

def make_wait_prefix(
    node_id: str,
    model_name: str,
) -> str:

    value = (
        f"{node_id}__{model_name}"
    )

    for old_value in (
        "-",
        ".",
        ":",
        "/",
        " ",
    ):

        value = value.replace(
            old_value,
            "_",
        )

    return value


# ============================================================
# ONE CSV
#
# ONE ROW = ONE ACTUAL REQUEST
# ============================================================

def save_combined_csv(
    file_path: str,
    all_results: List[dict],
    monitoring_targets: List[dict],
):

    # ========================================================
    # Normal actual request fields
    # ========================================================

    base_columns = [

        "request_id",
        "request_number",

        "node_id",
        "node_type",
        "server_address",

        "model_name",
        "image_name",

        "request_arrival_ms",

        "estimator_launch_overhead_ms",

        "wall_latency_ms",
        "rpc_latency_ms",

        "probability",
        "label",

        "success",
        "container_state",

        "error",
    ]


    # ========================================================
    # Wait-state columns for every monitored deployment
    # ========================================================

    wait_columns = []

    seen_prefixes = set()

    for target in (
        monitoring_targets
    ):

        prefix = make_wait_prefix(

            node_id=(
                target[
                    "node_id"
                ]
            ),

            model_name=(
                target[
                    "model_name"
                ]
            ),
        )

        if prefix in seen_prefixes:
            continue

        seen_prefixes.add(
            prefix
        )

        wait_columns.extend(
            [

                f"{prefix}__cpu_wait_sec",
                f"{prefix}__cpu_wait_ms",

                f"{prefix}__gpu_wait_sec",
                f"{prefix}__gpu_wait_ms",

                f"{prefix}__selected_wait_sec",
                f"{prefix}__selected_wait_ms",

                f"{prefix}__selected_compute",

                f"{prefix}__cpu_waiting",
                f"{prefix}__cpu_running",

                f"{prefix}__gpu_waiting",
                f"{prefix}__gpu_running",

                f"{prefix}__estimator_query_ms",

                f"{prefix}__snapshot_total_ms",

                f"{prefix}__estimator_error",
            ]
        )

    columns = (
        base_columns
        + wait_columns
    )

    # ========================================================
    # Write output
    # ========================================================

    with open(
        file_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = (
            csv.DictWriter(
                csv_file,
                fieldnames=(
                    columns
                ),
            )
        )

        writer.writeheader()

        for result in (
            all_results
        ):

            # =================================================
            # ONE result = ONE actual request
            # =================================================

            row = {

                "request_id":
                    result.get(
                        "request_id"
                    ),

                "request_number":
                    result.get(
                        "request_number"
                    ),

                "node_id":
                    result.get(
                        "node_id"
                    ),

                "node_type":
                    result.get(
                        "node_type"
                    ),

                "server_address":
                    result.get(
                        "server_address"
                    ),

                "model_name":
                    result.get(
                        "model_name"
                    ),

                "image_name":
                    result.get(
                        "image_name"
                    ),

                "request_arrival_ms":
                    result.get(
                        "request_arrival_ms"
                    ),

                "estimator_launch_overhead_ms":
                    result.get(
                        "estimator_launch_overhead_ms"
                    ),

                "wall_latency_ms":
                    result.get(
                        "wall_latency_ms"
                    ),

                "rpc_latency_ms":
                    result.get(
                        "rpc_latency_ms"
                    ),

                "probability":
                    result.get(
                        "probability"
                    ),

                "label":
                    result.get(
                        "label"
                    ),

                "success":
                    result.get(
                        "success"
                    ),

                "container_state":
                    result.get(
                        "container_state"
                    ),

                "error":
                    result.get(
                        "error"
                    ),
            }


            # =================================================
            # Put every monitored model into SAME row
            # =================================================

            snapshots = (
                result.get(
                    "wait_snapshot",
                    [],
                )
            )

            for snapshot in (
                snapshots
            ):

                monitored_node = (
                    snapshot.get(
                        "monitored_node_id"
                    )
                )

                monitored_model = (
                    snapshot.get(
                        "monitored_model"
                    )
                )

                if (
                    monitored_node is None
                    or monitored_model is None
                ):
                    continue

                prefix = make_wait_prefix(

                    node_id=(
                        monitored_node
                    ),

                    model_name=(
                        monitored_model
                    ),
                )

                cpu_wait_sec = (
                    snapshot.get(
                        "cpu_estimated_wait_time_sec"
                    )
                )

                gpu_wait_sec = (
                    snapshot.get(
                        "gpu_estimated_wait_time_sec"
                    )
                )

                selected_wait_sec = (
                    snapshot.get(
                        "selected_estimated_wait_time_sec"
                    )
                )

                cpu_wait_ms = (
                    float(
                        cpu_wait_sec
                    )
                    * 1000.0

                    if cpu_wait_sec
                    is not None

                    else None
                )

                gpu_wait_ms = (
                    float(
                        gpu_wait_sec
                    )
                    * 1000.0

                    if gpu_wait_sec
                    is not None

                    else None
                )

                selected_wait_ms = (
                    float(
                        selected_wait_sec
                    )
                    * 1000.0

                    if selected_wait_sec
                    is not None

                    else None
                )

                row[
                    f"{prefix}__cpu_wait_sec"
                ] = cpu_wait_sec

                row[
                    f"{prefix}__cpu_wait_ms"
                ] = cpu_wait_ms


                row[
                    f"{prefix}__gpu_wait_sec"
                ] = gpu_wait_sec

                row[
                    f"{prefix}__gpu_wait_ms"
                ] = gpu_wait_ms


                row[
                    f"{prefix}__selected_wait_sec"
                ] = selected_wait_sec

                row[
                    f"{prefix}__selected_wait_ms"
                ] = selected_wait_ms


                row[
                    f"{prefix}__selected_compute"
                ] = snapshot.get(
                    "selected_compute"
                )


                row[
                    f"{prefix}__cpu_waiting"
                ] = snapshot.get(
                    "cpu_waiting_functions"
                )

                row[
                    f"{prefix}__cpu_running"
                ] = snapshot.get(
                    "cpu_running_functions"
                )


                row[
                    f"{prefix}__gpu_waiting"
                ] = snapshot.get(
                    "gpu_waiting_functions"
                )

                row[
                    f"{prefix}__gpu_running"
                ] = snapshot.get(
                    "gpu_running_functions"
                )


                row[
                    f"{prefix}__estimator_query_ms"
                ] = snapshot.get(
                    "estimator_query_ms"
                )


                row[
                    f"{prefix}__snapshot_total_ms"
                ] = snapshot.get(
                    "snapshot_total_ms"
                )


                row[
                    f"{prefix}__estimator_error"
                ] = snapshot.get(
                    "estimator_error"
                )

            writer.writerow(
                row
            )


# ============================================================
# Inference summary
# ============================================================

def print_inference_summary(
    all_results: List[dict],
):

    grouped = defaultdict(
        list
    )

    for result in (
        all_results
    ):

        if (
            result.get(
                "error"
            )
            is not None
        ):
            continue

        key = (

            result.get(
                "node_id"
            ),

            result.get(
                "model_name"
            ),
        )

        grouped[
            key
        ].append(
            result
        )

    print(
        "\n"
        + "=" * 85
    )

    print(
        "FINAL INFERENCE LATENCY SUMMARY"
    )

    print(
        "=" * 85
    )

    for (
        node_id,
        model_name,
    ), results in sorted(
        grouped.items()
    ):

        avg_e2e = average(
            [
                x.get(
                    "wall_latency_ms"
                )
                for x in results
            ]
        )

        avg_rpc = average(
            [
                x.get(
                    "rpc_latency_ms"
                )
                for x in results
            ]
        )

        if avg_e2e is None:
            continue

        print(
            f"{str(node_id):<12} "
            f"{str(model_name):<24} "
            f"requests="
            f"{len(results):<5} "
            f"avg_e2e="
            f"{avg_e2e:>9.2f} ms "
            f"avg_rpc="
            f"{avg_rpc:>9.2f} ms"
        )

    print(
        "=" * 85
    )


# ============================================================
# Wait-time summary
# ============================================================

def print_wait_time_summary(
    all_results: List[dict],
):

    grouped = defaultdict(
        list
    )

    for result in (
        all_results
    ):

        for snapshot in (
            result.get(
                "wait_snapshot",
                [],
            )
        ):

            key = (

                snapshot.get(
                    "monitored_node_id"
                ),

                snapshot.get(
                    "monitored_model"
                ),
            )

            grouped[
                key
            ].append(
                snapshot
            )

    print(
        "\n"
        + "=" * 100
    )

    print(
        "AVERAGE ILUVATAR WAIT-TIME SUMMARY"
    )

    print(
        "=" * 100
    )

    for (
        node_id,
        model_name,
    ), snapshots in sorted(
        grouped.items()
    ):

        cpu_wait_sec = average(
            [
                x.get(
                    "cpu_estimated_wait_time_sec"
                )
                for x in snapshots
            ]
        )

        gpu_wait_sec = average(
            [
                x.get(
                    "gpu_estimated_wait_time_sec"
                )
                for x in snapshots
            ]
        )

        cpu_waiting = average(
            [
                x.get(
                    "cpu_waiting_functions"
                )
                for x in snapshots
            ]
        )

        cpu_running = average(
            [
                x.get(
                    "cpu_running_functions"
                )
                for x in snapshots
            ]
        )

        gpu_waiting = average(
            [
                x.get(
                    "gpu_waiting_functions"
                )
                for x in snapshots
            ]
        )

        gpu_running = average(
            [
                x.get(
                    "gpu_running_functions"
                )
                for x in snapshots
            ]
        )

        cpu_ms = (
            cpu_wait_sec * 1000.0
            if cpu_wait_sec is not None
            else None
        )

        gpu_ms = (
            gpu_wait_sec * 1000.0
            if gpu_wait_sec is not None
            else None
        )

        cpu_text = (
            f"{cpu_ms:.2f}"
            if cpu_ms is not None
            else "N/A"
        )

        gpu_text = (
            f"{gpu_ms:.2f}"
            if gpu_ms is not None
            else "N/A"
        )

        print(
            f"{str(node_id):<12} "
            f"{str(model_name):<24} "
            f"cpu_wait={cpu_text:>9} ms "
            f"gpu_wait={gpu_text:>9} ms "
            f"cpu_waiting={str(cpu_waiting):<8} "
            f"cpu_running={str(cpu_running):<8} "
            f"gpu_waiting={str(gpu_waiting):<8} "
            f"gpu_running={str(gpu_running):<8}"
        )

    print(
        "=" * 100
    )


# ============================================================
# Main
# ============================================================

def main():

    execution_targets = (
        build_execution_targets()
    )

    monitoring_targets = (
        build_monitoring_targets()
    )

    image_paths = (
        get_image_paths()
    )

    if not execution_targets:

        raise ValueError(
            "No execution targets found."
        )

    if not monitoring_targets:

        raise ValueError(
            "No monitoring targets found."
        )

    if not image_paths:

        raise ValueError(
            f"No supported images found in "
            f"{IMAGE_DIR}"
        )


    # ========================================================
    # Show setup
    # ========================================================

    print(
        "\n=== ACTUAL INFERENCE TARGETS ==="
    )

    for target in (
        execution_targets
    ):

        print(
            f"{target['model_name']:<24} "
            f"-> "
            f"{target['node_id']:<12} "
            f"{target['server_address']}"
        )


    print(
        "\n=== WAIT-TIME MONITORING TARGETS ==="
    )

    for target in (
        monitoring_targets
    ):

        print(
            f"{target['model_name']:<24} "
            f"-> "
            f"{target['node_id']:<12} "
            f"{target['server_address']}"
        )


    print(
        "\n=== WORKLOAD ==="
    )

    print(
        f"IAT: "
        f"{IAT_MS} ms"
    )

    print(
        f"Rate: "
        f"{1000.0 / IAT_MS:.2f} req/s"
    )

    print(
        f"Requests/model: "
        f"{REQUESTS_PER_MODEL}"
    )

    print(
        f"Execution targets: "
        f"{len(execution_targets)}"
    )

    print(
        f"Wait observations/request: "
        f"{len(monitoring_targets)}"
    )


    all_results = []

    experiment_start = (
        time.perf_counter()
    )

    try:

        # ----------------------------------------------------
        # Same structure as your old experiment:
        #
        # model 1 workload
        # then model 2 workload
        # then model 3 workload
        # ...
        # ----------------------------------------------------

        for (
            target_index,
            target,
        ) in enumerate(
            execution_targets,
            start=1,
        ):

            print(
                f"\nStarting target "
                f"{target_index}/"
                f"{len(execution_targets)}"
            )

            model_results = (
                execute_model_at_fixed_iat(

                    target=(
                        target
                    ),

                    image_paths=(
                        image_paths
                    ),

                    monitoring_targets=(
                        monitoring_targets
                    ),
                )
            )

            all_results.extend(
                model_results
            )

    finally:

        _estimator_executor.shutdown(
            wait=True
        )

        close_all_channels()


    experiment_end = (
        time.perf_counter()
    )


    # ========================================================
    # ONE CSV
    # ========================================================

    save_combined_csv(

        file_path=(
            CSV_OUTPUT_FILE
        ),

        all_results=(
            all_results
        ),

        monitoring_targets=(
            monitoring_targets
        ),
    )


    # ========================================================
    # Terminal summaries
    # ========================================================

    print_inference_summary(
        all_results
    )

    print_wait_time_summary(
        all_results
    )


    print(
        f"\nTotal experiment time: "
        f"{experiment_end - experiment_start:.2f} seconds"
    )

    print(
        "\nCSV file:"
    )

    print(
        os.path.abspath(
            CSV_OUTPUT_FILE
        )
    )


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()