import base64
import json
import multiprocessing
import os
import time
import uuid

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller import ckn_config as config


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = (
    "/Users/agamage/Desktop/D2I/Codes Original/"
    "clone main/ckn-faas/ckn_data/images/tem"
)


IAT_MS = 200
REQUESTS_PER_MODEL = 100


MAX_WORKERS_PER_MODEL = REQUESTS_PER_MODEL



MODELS_TO_EXECUTE: List[str] = [
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
# Shared worker cache
# ============================================================

_worker_map: Dict[str, object] = {}
_channel_map: Dict[str, grpc.Channel] = {}
_worker_lock = Lock()


def get_worker_for_address(
    server_address: str,
):
    """
    Reuse one synchronous gRPC channel and worker stub
    for each server address.
    """

    with _worker_lock:
        if server_address not in _worker_map:
            channel = grpc.insecure_channel(
                server_address
            )

            worker = pb2_grpc.IluvatarWorkerStub(
                channel
            )

            _channel_map[server_address] = channel
            _worker_map[server_address] = worker

    return _worker_map[server_address]


def close_all_channels() -> None:
    """
    Close all cached gRPC channels.
    """

    with _worker_lock:
        for channel in _channel_map.values():
            try:
                channel.close()
            except Exception:
                pass

        _worker_map.clear()
        _channel_map.clear()


# ============================================================
# Configuration helpers
# ============================================================

def validate_multi_node_config() -> None:
    """
    Validate the N-edge/N-cloud configuration.
    """

    if not isinstance(ILUVATAR_NODES, dict):
        raise ValueError(
            "ILUVATAR_NODES must be a dictionary."
        )

    if not ILUVATAR_NODES:
        raise ValueError(
            "USE_MULTI_ILUVATAR_INSTANCES=True, "
            "but ILUVATAR_NODES is empty."
        )

    for node_id, node in ILUVATAR_NODES.items():

        if not isinstance(node, dict):
            raise ValueError(
                f"Configuration for node '{node_id}' "
                f"must be a dictionary."
            )

        if not node.get("address"):
            raise ValueError(
                f"Node '{node_id}' does not have an address."
            )

        models = node.get(
            "models",
            [],
        )

        if not isinstance(models, list):
            raise ValueError(
                f"Node '{node_id}' field 'models' "
                f"must be a list."
            )

        node_type = str(
            node.get(
                "type",
                "",
            )
        ).lower()

        if node_type not in (
            "edge",
            "cloud",
        ):
            raise ValueError(
                f"Node '{node_id}' has invalid type "
                f"'{node_type}'. Expected 'edge' or 'cloud'."
            )


def build_legacy_nodes() -> Dict[str, dict]:
    """
    Convert the old single-instance or two-instance
    configuration into the general node format.
    """

    if not USE_TWO_ILUVATAR_INSTANCES:

        return {
            "single_instance": {
                "type": "single",
                "address": SERVER_ADDRESS,
                "models": list(
                    MODELS_TO_EXECUTE
                ),
                "enabled": True,
            }
        }

    return {

        "edge_1": {
            "type": "edge",
            "address": SMALL_MODEL_SERVER_ADDRESS,
            "models": [
                model_name
                for model_name in MODELS_TO_EXECUTE
                if model_name
                in SMALL_INSTANCE_MODELS
            ],
            "enabled": True,
        },

        "cloud_1": {
            "type": "cloud",
            "address": LARGE_MODEL_SERVER_ADDRESS,
            "models": [
                model_name
                for model_name in MODELS_TO_EXECUTE
                if model_name
                in LARGE_INSTANCE_MODELS
            ],
            "enabled": True,
        },
    }


def get_active_nodes() -> Dict[str, dict]:
    """
    Return the node registry used by this execution.

    Multi-node mode:
        Uses ILUVATAR_NODES.

    Legacy mode:
        Builds a compatible node registry
        from the old settings.
    """

    if USE_MULTI_ILUVATAR_INSTANCES:
        validate_multi_node_config()

        return ILUVATAR_NODES

    return build_legacy_nodes()


def get_models_for_node(
    node_config: dict,
) -> List[str]:
    """
    Return the configured models that should
    be executed on one node.
    """

    configured_models = list(
        node_config.get(
            "models",
            [],
        )
    )

    if not MODELS_TO_EXECUTE:
        return configured_models

    allowed_models = set(
        MODELS_TO_EXECUTE
    )

    return [
        model_name
        for model_name in configured_models
        if model_name in allowed_models
    ]


def build_deployment_targets() -> List[dict]:
    """
    Create one target for every model-node deployment.

    Example:

        resnet18 on edge_1
        resnet18 on edge_2
        resnet101 on cloud_1
    """

    active_nodes = get_active_nodes()

    targets: List[dict] = []

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

        address = str(
            node_config["address"]
        )

        node_type = str(
            node_config.get(
                "type",
                "unknown",
            )
        )

        models = get_models_for_node(
            node_config
        )

        for model_name in models:

            targets.append(
                {
                    "node_id": node_id,
                    "node_type": node_type,
                    "server_address": address,
                    "model_name": model_name,
                }
            )

    return targets


# ============================================================
# Image helpers
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


def get_image_paths() -> List[str]:
    """
    Return supported images from IMAGE_DIR.
    """

    if not os.path.isdir(
        IMAGE_DIR
    ):
        raise FileNotFoundError(
            f"Image directory does not exist: "
            f"{IMAGE_DIR}"
        )

    image_paths: List[str] = []

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
# Inference request
# ============================================================

def send_request(
    node_id: str,
    node_type: str,
    server_address: str,
    model_name: str,
    image_path: str,
) -> dict:
    """
    Execute one model on one specific Iluvatar node.
    """

    try:

        worker = get_worker_for_address(
            server_address
        )

        image_b64 = (
            read_image_as_base64(
                image_path
            )
        )

        request = pb2.InvokeRequest(
            function_name=model_name,
            function_version="1",
            json_args=json.dumps(
                {
                    "model_name": model_name,
                    "image_data": image_b64,
                }
            ),
            transaction_id=str(
                uuid.uuid4()
            ),
        )

        # ----------------------------------------------------
        # Start E2E latency measurement immediately before
        # sending the RPC.
        # ----------------------------------------------------

        start_time = (
            time.perf_counter()
        )

        response = worker.invoke(
            request
        )

        end_time = (
            time.perf_counter()
        )

        # ----------------------------------------------------
        # Client-observed E2E latency
        # ----------------------------------------------------

        wall_latency_ms = (
            end_time
            - start_time
        ) * 1000.0

        # ----------------------------------------------------
        # Iluvatar-reported RPC duration
        # ----------------------------------------------------

        rpc_latency_ms = (
            float(
                response.duration_us
            )
            / 1000.0
        )

        result_json = json.loads(
            response.json_result
        )

        if "body" not in result_json:

            raise ValueError(
                f"Unexpected response: "
                f"{result_json}"
            )

        body = result_json[
            "body"
        ]

        if "Probability" not in body:

            raise ValueError(
                f"Inference failed: "
                f"{result_json}"
            )

        probability = float(
            body["Probability"]
        )

        label = body.get(
            "Prediction Class",
            "UNKNOWN",
        )

        return {
            "node_id": node_id,
            "node_type": node_type,
            "server_address": server_address,
            "model_name": model_name,
            "image_path": image_path,
            "image_name": os.path.basename(
                image_path
            ),
            "probability": probability,
            "label": label,
            "wall_latency_ms": (
                wall_latency_ms
            ),
            "rpc_latency_ms": (
                rpc_latency_ms
            ),
            "success": bool(
                response.success
            ),
            "container_state": (
                pb2.ContainerState.Name(
                    response.container_state
                )
            ),
            "error": None,
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
            "node_id": node_id,
            "node_type": node_type,
            "server_address": server_address,
            "model_name": model_name,
            "image_path": image_path,
            "image_name": os.path.basename(
                image_path
            ),
            "probability": None,
            "label": None,
            "wall_latency_ms": None,
            "rpc_latency_ms": None,
            "success": False,
            "container_state": None,
            "error": error_message,
        }

    except Exception as error:

        return {
            "node_id": node_id,
            "node_type": node_type,
            "server_address": server_address,
            "model_name": model_name,
            "image_path": image_path,
            "image_name": os.path.basename(
                image_path
            ),
            "probability": None,
            "label": None,
            "wall_latency_ms": None,
            "rpc_latency_ms": None,
            "success": False,
            "container_state": None,
            "error": str(
                error
            ),
        }


# ============================================================
# Fixed-IAT workload for ONE model
# ============================================================

def execute_model_at_fixed_iat(
    target: dict,
    image_paths: List[str],
) -> List[dict]:
    """
    Execute the workload for ONE model.

    Example for IAT_MS = 200 and REQUESTS_PER_MODEL = 5:

        Request 1 ->   0 ms
        Request 2 -> 200 ms
        Request 3 -> 400 ms
        Request 4 -> 600 ms
        Request 5 -> 800 ms

    Requests may overlap if inference latency is greater
    than the configured IAT.

    After all requests finish, the function calculates
    the average latency for this model.

    Only after this function returns will main() start
    profiling the next model.
    """

    node_id = target[
        "node_id"
    ]

    node_type = target[
        "node_type"
    ]

    server_address = target[
        "server_address"
    ]

    model_name = target[
        "model_name"
    ]

    print(
        "\n"
        + "=" * 75
    )

    print(
        f"Profiling model: "
        f"{model_name}"
    )

    print(
        f"Node: "
        f"{node_id}"
    )

    print(
        f"Node type: "
        f"{node_type}"
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
        "=" * 75
    )

    results: List[dict] = []

    iat_sec = (
        IAT_MS
        / 1000.0
    )

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS_PER_MODEL
    ) as executor:

        futures = []

        # ----------------------------------------------------
        # Workload clock starts here.
        # ----------------------------------------------------

        workload_start = (
            time.perf_counter()
        )

        for request_index in range(
            REQUESTS_PER_MODEL
        ):

            # ------------------------------------------------
            # Absolute target submission time.
            #
            # This is better than simply:
            #
            #     time.sleep(0.2)
            #
            # because repeated sleep calls can accumulate
            # scheduling drift.
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

            # ------------------------------------------------
            # Cycle through available images.
            #
            # If there are 5+ images, requests 1-5 use
            # different images.
            #
            # If there are fewer than 5 images, the images
            # are reused.
            # ------------------------------------------------

            image_path = (
                image_paths[
                    request_index
                    % len(
                        image_paths
                    )
                ]
            )

            actual_submit_time = (
                time.perf_counter()
            )

            relative_submit_ms = (
                actual_submit_time
                - workload_start
            ) * 1000.0

            print(
                f"[SUBMIT] "
                f"model={model_name} "
                f"request="
                f"{request_index + 1}/"
                f"{REQUESTS_PER_MODEL} "
                f"target="
                f"{request_index * IAT_MS}ms "
                f"actual="
                f"{relative_submit_ms:.2f}ms "
                f"image="
                f"{os.path.basename(image_path)}"
            )

            future = executor.submit(
                send_request,
                node_id,
                node_type,
                server_address,
                model_name,
                image_path,
            )

            futures.append(
                {
                    "request_number": (
                        request_index
                        + 1
                    ),
                    "submit_time_ms": (
                        relative_submit_ms
                    ),
                    "future": future,
                }
            )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # At this point all five requests have been submitted.
        #
        # Now wait for EVERY request belonging to THIS model
        # before continuing.
        # ----------------------------------------------------

        for future_info in futures:

            request_number = (
                future_info[
                    "request_number"
                ]
            )

            submit_time_ms = (
                future_info[
                    "submit_time_ms"
                ]
            )

            future = (
                future_info[
                    "future"
                ]
            )

            result = (
                future.result()
            )

            result[
                "request_number"
            ] = request_number

            result[
                "submit_time_ms"
            ] = submit_time_ms

            results.append(
                result
            )

            if result["error"]:

                print(
                    f"[EXECUTE_FAIL] "
                    f"node={node_id} "
                    f"model={model_name} "
                    f"request="
                    f"{request_number} "
                    f"error="
                    f"{result['error']}"
                )

                continue

            print(
                f"[EXECUTE_OK] "
                f"node={node_id} "
                f"model={model_name} "
                f"request="
                f"{request_number} "
                f"image="
                f"{result['image_name']} "
                f"label="
                f"{result['label']} "
                f"prob="
                f"{result['probability']:.4f} "
                f"wall_latency="
                f"{result['wall_latency_ms']:.2f}ms "
                f"rpc_latency="
                f"{result['rpc_latency_ms']:.2f}ms "
                f"state="
                f"{result['container_state']}"
            )

    # ========================================================
    # Calculate statistics for THIS model
    # ========================================================

    successful_results = [
        result
        for result in results
        if (
            result["error"] is None
            and result[
                "wall_latency_ms"
            ] is not None
        )
    ]

    wall_latencies = [
        float(
            result[
                "wall_latency_ms"
            ]
        )
        for result
        in successful_results
    ]

    rpc_latencies = [
        float(
            result[
                "rpc_latency_ms"
            ]
        )
        for result
        in successful_results
        if result[
            "rpc_latency_ms"
        ] is not None
    ]

    print(
        "\n"
        + "-" * 75
    )

    if wall_latencies:

        average_wall_latency = (
            sum(
                wall_latencies
            )
            / len(
                wall_latencies
            )
        )

        min_wall_latency = min(
            wall_latencies
        )

        max_wall_latency = max(
            wall_latencies
        )

        if rpc_latencies:

            average_rpc_latency = (
                sum(
                    rpc_latencies
                )
                / len(
                    rpc_latencies
                )
            )

        else:

            average_rpc_latency = None

        print(
            f"[MODEL_COMPLETE] "
            f"model={model_name}"
        )

        print(
            f"Successful requests: "
            f"{len(successful_results)}/"
            f"{REQUESTS_PER_MODEL}"
        )

        print(
            f"Average E2E latency: "
            f"{average_wall_latency:.2f} ms"
        )

        print(
            f"Minimum E2E latency: "
            f"{min_wall_latency:.2f} ms"
        )

        print(
            f"Maximum E2E latency: "
            f"{max_wall_latency:.2f} ms"
        )

        if (
            average_rpc_latency
            is not None
        ):

            print(
                f"Average RPC latency: "
                f"{average_rpc_latency:.2f} ms"
            )

    else:

        print(
            f"[MODEL_COMPLETE] "
            f"model={model_name} "
            f"No successful requests."
        )

    print(
        "-" * 75
    )

    return results


# ============================================================
# Output helpers
# ============================================================

def deployment_key(
    node_id: str,
    model_name: str,
) -> str:
    """
    Unique key for a model deployment.
    """

    return (
        f"{node_id}:"
        f"{model_name}"
    )


def build_summary(
    all_results: List[dict],
) -> Dict[str, dict]:
    """
    Build average values for every node-model deployment.
    """

    grouped_results: Dict[
        str,
        List[dict],
    ] = defaultdict(
        list
    )

    for result in all_results:

        if result.get(
            "error"
        ):
            continue

        key = deployment_key(
            node_id=result[
                "node_id"
            ],
            model_name=result[
                "model_name"
            ],
        )

        grouped_results[
            key
        ].append(
            result
        )

    summary: Dict[
        str,
        dict,
    ] = {}

    for (
        key,
        results,
    ) in grouped_results.items():

        wall_latencies = [
            float(
                result[
                    "wall_latency_ms"
                ]
            )
            for result in results
            if result[
                "wall_latency_ms"
            ] is not None
        ]

        rpc_latencies = [
            float(
                result[
                    "rpc_latency_ms"
                ]
            )
            for result in results
            if result[
                "rpc_latency_ms"
            ] is not None
        ]

        probabilities = [
            float(
                result[
                    "probability"
                ]
            )
            for result in results
            if result[
                "probability"
            ] is not None
        ]

        first_result = (
            results[0]
        )

        summary[key] = {

            "node_id": (
                first_result[
                    "node_id"
                ]
            ),

            "node_type": (
                first_result[
                    "node_type"
                ]
            ),

            "server_address": (
                first_result[
                    "server_address"
                ]
            ),

            "model_name": (
                first_result[
                    "model_name"
                ]
            ),

            "request_count": len(
                results
            ),

            "average_wall_latency_ms": (
                sum(
                    wall_latencies
                )
                / len(
                    wall_latencies
                )
                if wall_latencies
                else None
            ),

            "average_rpc_latency_ms": (
                sum(
                    rpc_latencies
                )
                / len(
                    rpc_latencies
                )
                if rpc_latencies
                else None
            ),

            "average_probability": (
                sum(
                    probabilities
                )
                / len(
                    probabilities
                )
                if probabilities
                else None
            ),
        }

    return summary


def save_json(
    file_path: str,
    data,
) -> None:

    with open(
        file_path,
        "w",
        encoding="utf-8",
    ) as output_file:

        json.dump(
            data,
            output_file,
            indent=2,
            allow_nan=True,
        )


# ============================================================
# Main execution
# ============================================================

def main() -> None:

    deployment_targets = (
        build_deployment_targets()
    )

    image_paths = (
        get_image_paths()
    )

    if not deployment_targets:

        raise ValueError(
            "No enabled model-node "
            "deployments were found."
        )

    if not image_paths:

        raise ValueError(
            f"No supported images found "
            f"in {IMAGE_DIR}"
        )

    # ========================================================
    # Print deployment configuration
    # ========================================================

    print(
        "=== Deployment Summary ==="
    )

    for target in deployment_targets:

        print(
            f"{target['model_name']:<22} "
            f"-> "
            f"node="
            f"{target['node_id']:<12} "
            f"type="
            f"{target['node_type']:<6} "
            f"server="
            f"{target['server_address']}"
        )

    print(
        "\n=== Workload Configuration ==="
    )

    print(
        f"Images available: "
        f"{len(image_paths)}"
    )

    print(
        f"Deployments: "
        f"{len(deployment_targets)}"
    )

    print(
        f"Requests per model: "
        f"{REQUESTS_PER_MODEL}"
    )

    print(
        f"IAT: "
        f"{IAT_MS} ms"
    )

    print(
        f"Approximate request rate: "
        f"{1000.0 / IAT_MS:.2f} req/s"
    )

    print(
        f"Total requests: "
        f"{len(deployment_targets) * REQUESTS_PER_MODEL}"
    )

    # ========================================================
    # Execute workload
    # ========================================================

    all_results: List[
        dict
    ] = []

    experiment_start_time = (
        time.perf_counter()
    )

    try:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # deployment_targets are processed sequentially.
        #
        # That means:
        #
        # Model A:
        #     5 requests @ 200-ms IAT
        #     wait until all complete
        #     calculate latency
        #
        # Model B:
        #     5 requests @ 200-ms IAT
        #     wait until all complete
        #     calculate latency
        #
        # etc.
        # ----------------------------------------------------

        for (
            target_index,
            target,
        ) in enumerate(
            deployment_targets,
            start=1,
        ):

            print(
                f"\nStarting deployment "
                f"{target_index}/"
                f"{len(deployment_targets)}"
            )

            model_results = (
                execute_model_at_fixed_iat(
                    target=target,
                    image_paths=image_paths,
                )
            )

            all_results.extend(
                model_results
            )

    finally:

        close_all_channels()

    experiment_end_time = (
        time.perf_counter()
    )

    total_time_sec = (
        experiment_end_time
        - experiment_start_time
    )

    # ========================================================
    # Build final summary
    # ========================================================

    summary = build_summary(
        all_results
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "FINAL LATENCY SUMMARY"
    )

    print(
        "=" * 75
    )

    print(
        f"IAT: "
        f"{IAT_MS} ms"
    )

    print(
        f"Request rate: "
        f"{1000.0 / IAT_MS:.2f} req/s"
    )

    print(
        f"Requests per model: "
        f"{REQUESTS_PER_MODEL}"
    )

    print(
        f"Total experiment time: "
        f"{total_time_sec:.2f} seconds"
    )

    print()

    for key in sorted(
        summary
    ):

        item = summary[
            key
        ]

        avg_wall = item[
            "average_wall_latency_ms"
        ]

        avg_rpc = item[
            "average_rpc_latency_ms"
        ]

        request_count = item[
            "request_count"
        ]

        if avg_wall is None:
            continue

        print(
            f"{item['node_id']:<12} "
            f"{item['model_name']:<22} "
            f"e2e_latency="
            f"{avg_wall:>8.2f} ms "
            # f"requests="
            # f"{request_count}"
            f"    rpc_latency="
            f"{avg_rpc:.2f} ms"
        )

        # Uncomment if you also want RPC latency:

        # if avg_rpc is not None:
        #     print(
        #         f"    rpc_latency="
        #         f"{avg_rpc:.2f} ms"
        #     )

    print(
        "=" * 75
    )

    # ========================================================
    # Optional JSON output
    # ========================================================

    # save_json(
    #     "model_execution_results.json",
    #     all_results,
    # )
    #
    # save_json(
    #     "model_node_latency_summary.json",
    #     summary,
    # )
    #
    # print(
    #     "\nResults saved to:"
    # )
    #
    # print(
    #     "  model_execution_results.json"
    # )
    #
    # print(
    #     "  model_node_latency_summary.json"
    # )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()