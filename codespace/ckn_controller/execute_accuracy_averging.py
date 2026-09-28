# import grpc
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
# import json
# import os
# import uuid
# import base64
# import time
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from concurrent.futures import ProcessPoolExecutor, as_completed
#
# # Configuration
# image_dir = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/tem"
# # model_list = ["shufflenet_v2_x0_5", "mobilenet_v3_small", "googlenet",
# #               "resnext50_32x4d", "densenet201", "resnet152"]
#
# model_list = ["mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"]
#
# # Output: {model_name: [prob1, prob2, ...]}
# probability_results = defaultdict(list)
# latency_results = defaultdict(list)
#
# # Connect to gRPC
# channel = grpc.insecure_channel("149.165.152.13:8079")
# worker = pb2_grpc.IluvatarWorkerStub(channel)
#
# def read_image_as_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")
#
# def send_request(model_name, image_path):
#     try:
#         image_b64 = read_image_as_base64(image_path)
#         request = pb2.InvokeRequest(
#             function_name=model_name,
#             function_version="1",
#             json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#             transaction_id=str(uuid.uuid4()),
#         )
#         start = time.perf_counter()
#         response = worker.invoke(request)
#         end = time.perf_counter()  # ✅ end timing
#
#         latency_ms = (end - start) * 1000  # milliseconds
#         result_json = json.loads(response.json_result)
#         prob = result_json["body"]["Probability"]
#         return (model_name, prob, image_path,latency_ms, None)
#     except Exception as e:
#         return (model_name, None, image_path,None, str(e))
#
# # Gather all image/model tasks
# tasks = []
# for filename in sorted(os.listdir(image_dir)):
#     if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#         continue
#     image_path = os.path.join(image_dir, filename)
#     for model_name in model_list:
#         tasks.append((model_name, image_path))
#
# # Run in parallel
# start = time.perf_counter()
# # with ThreadPoolExecutor(max_workers=1) as executor:
# with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
#     futures = [executor.submit(send_request, m, i) for m, i in tasks]
#
#     for future in as_completed(futures):
#         model_name, prob, image_path, latency, error = future.result()
#         if error:
#             print(f"❌ {model_name} on {os.path.basename(image_path)}: {error}")
#         else:
#             probability_results[model_name].append(prob)
#             latency_results[model_name].append(latency)
#             print(f"✅ {model_name} on {os.path.basename(image_path)}: prob={prob:.4f}, latency={latency:.2f} ms")
#             print(f"✅ {model_name} on {os.path.basename(image_path)}: {prob:.4f}")
#
# end = time.perf_counter()
# print(f"\n⏱️ Total time: {(end - start):.2f} seconds")
#
# # === Summary Stats ===
# print("\n📊 Average Latency per Model:")
# for model, latencies in latency_results.items():
#     avg_latency = sum(latencies) / len(latencies)
#     print(f"{model:<20}: {avg_latency:.2f} ms")
#
# # output_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/results/model_latencies.json"
# output_path = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/results/model_latencies.json"
# with open("model_latencies.json", "w") as f:
#     json.dump(latency_results, f, indent=2)
#
# # Save results
# with open("model_probabilities.json", "w") as f:
#     json.dump(probability_results, f, indent=2)
# print(probability_results)
# print("✅ Results saved to model_probabilities.json")


#########
#last working code start
######

# import grpc
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
# import json
# import os
# import uuid
# import base64
# import time
# import multiprocessing
# from collections import defaultdict
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from concurrent.futures import ThreadPoolExecutor, as_completed
#
# # Configuration
# image_dir = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/tem"
# model_list = ["mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]
#
# channel = grpc.insecure_channel("149.165.175.242:8079")
# worker = pb2_grpc.IluvatarWorkerStub(channel)
#
# def read_image_as_base64(path):
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")
#
# def send_request(model_name, image_path):
#     try:
#         # Re-initialize gRPC connection in each process
#
#         image_b64 = read_image_as_base64(image_path)
#         request = pb2.InvokeRequest(
#             function_name=model_name,
#             function_version="1",
#             json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
#             transaction_id=str(uuid.uuid4()),
#         )
#
#         start = time.perf_counter()
#         response = worker.invoke(request)
#         end = time.perf_counter()
#
#         latency_ms = (end - start) * 1000
#         result_json = json.loads(response.json_result)
#         prob = result_json["body"]["Probability"]
#         return (model_name, prob, image_path, latency_ms, None)
#     except Exception as e:
#         return (model_name, None, image_path, None, str(e))
#
# def main():
#     probability_results = defaultdict(list)
#     latency_results = defaultdict(list)
#
#     # Prepare tasks: (model_name, image_path) tuples
#     tasks = []
#     for filename in sorted(os.listdir(image_dir)):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#         image_path = os.path.join(image_dir, filename)
#         for model_name in model_list:
#             tasks.append((model_name, image_path))
#
#     start = time.perf_counter()
#
#     # Run parallel inference using multiple cores
#     with ThreadPoolExecutor(max_workers=1) as executor:
#     # with ProcessPoolExecutor(max_workers=6) as executor:
#         futures = [executor.submit(send_request, m, i) for m, i in tasks]
#         for future in as_completed(futures):
#             model_name, prob, image_path, latency, error = future.result()
#             if error:
#                 print(f"❌ {model_name} on {os.path.basename(image_path)}: {error}")
#             else:
#                 probability_results[model_name].append(prob)
#                 latency_results[model_name].append(latency)
#                 print(f"✅ {model_name} on {os.path.basename(image_path)}: prob={prob:.4f}, latency={latency:.2f} ms")
#
#     end = time.perf_counter()
#     print(f"\n⏱️ Total time: {(end - start):.2f} seconds")
#
#     # === Summary Stats ===
#     print("\n📊 Average Latency per Model:")
#     for model, latencies in latency_results.items():
#         avg_latency = sum(latencies) / len(latencies)
#         print(f"{model:<20}: {avg_latency:.2f} ms")
#
#     # Save results
#     with open("model_latencies.json", "w") as f:
#         json.dump(latency_results, f, indent=2)
#     with open("model_probabilities.json", "w") as f:
#         json.dump(probability_results, f, indent=2)
#     print("✅ Results saved to model_probabilities.json")
#
# # Required for multiprocessing on macOS/Windows
# if __name__ == "__main__":
#     multiprocessing.freeze_support()  # Optional on macOS; safe to include
#     main()



#########
#end working code start
######

#
# import grpc
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
# import json
# import os
# import uuid
# import base64
# import time
# import multiprocessing
# from collections import defaultdict
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from threading import Lock
#
# from ckn_controller.ckn_config import (
#     USE_TWO_ILUVATAR_INSTANCES,
#     SERVER_ADDRESS,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
# # -----------------------------
# # Configuration
# # -----------------------------
# image_dir = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/tem"
# model_list = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]
#
# # -----------------------------
# # Routing helpers
# # -----------------------------
# def get_server_address_for_model(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return SERVER_ADDRESS
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return SMALL_MODEL_SERVER_ADDRESS
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return LARGE_MODEL_SERVER_ADDRESS
#
#     raise ValueError(f"Model {model_name} is not assigned to any Iluvatar instance.")
#
#
# # -----------------------------
# # Shared worker cache
# # -----------------------------
# _worker_map = {}
# _channel_map = {}
# _worker_lock = Lock()
#
#
# def get_worker_for_model(model_name: str):
#     """
#     Reuse one gRPC stub per server address.
#     """
#     address = get_server_address_for_model(model_name)
#
#     with _worker_lock:
#         if address not in _worker_map:
#             channel = grpc.insecure_channel(address)
#             worker = pb2_grpc.IluvatarWorkerStub(channel)
#             _channel_map[address] = channel
#             _worker_map[address] = worker
#
#     return _worker_map[address], address
#
#
# # -----------------------------
# # Utilities
# # -----------------------------
# def read_image_as_base64(path: str) -> str:
#     with open(path, "rb") as f:
#         return base64.b64encode(f.read()).decode("utf-8")
#
#
# def send_request(model_name: str, image_path: str):
#     try:
#         worker, address = get_worker_for_model(model_name)
#
#         image_b64 = read_image_as_base64(image_path)
#         request = pb2.InvokeRequest(
#             function_name=model_name,
#             function_version="1",
#             json_args=json.dumps({
#                 "model_name": model_name,
#                 "image_data": image_b64
#             }),
#             transaction_id=str(uuid.uuid4()),
#         )
#
#         start = time.perf_counter()
#         response = worker.invoke(request)
#         end = time.perf_counter()
#
#         latency_ms = (end - start) * 1000.0
#         result_json = json.loads(response.json_result)
#         if "body" not in result_json:
#             raise ValueError(f"Unexpected response: {result_json}")
#
#         body = result_json["body"]
#
#         if "Probability" not in body:
#             raise ValueError(f"Inference failed: {result_json}")
#
#         prob = float(body["Probability"])
#         label = body.get("Prediction Class", "UNKNOWN")
#         # prob = float(result_json["body"]["Probability"])
#         # label = result_json["body"].get("Prediction Class", "UNKNOWN")
#
#         return {
#             "model_name": model_name,
#             "image_path": image_path,
#             "probability": prob,
#             "label": label,
#             "latency_ms": latency_ms,
#             "server_address": address,
#             "error": None,
#         }
#
#     except Exception as e:
#         return {
#             "model_name": model_name,
#             "image_path": image_path,
#             "probability": None,
#             "label": None,
#             "latency_ms": None,
#             "server_address": None,
#             "error": str(e),
#         }
#
#
# def main():
#     probability_results = defaultdict(list)
#     latency_results = defaultdict(list)
#     route_results = defaultdict(list)
#
#     tasks = []
#     for filename in sorted(os.listdir(image_dir)):
#         if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
#             continue
#
#         image_path = os.path.join(image_dir, filename)
#         for model_name in model_list:
#             tasks.append((model_name, image_path))
#
#     print("=== Routing Summary ===")
#     for model_name in model_list:
#         addr = get_server_address_for_model(model_name)
#         print(f"{model_name:<20} -> {addr}")
#
#     start = time.perf_counter()
#
#     # Adjust max_workers as needed
#     with ThreadPoolExecutor(max_workers=1) as executor:
#         futures = [executor.submit(send_request, m, i) for m, i in tasks]
#
#         for future in as_completed(futures):
#             result = future.result()
#
#             model_name = result["model_name"]
#             image_path = result["image_path"]
#             prob = result["probability"]
#             label = result["label"]
#             latency = result["latency_ms"]
#             address = result["server_address"]
#             error = result["error"]
#
#             if error:
#                 print(f"❌ {model_name} on {os.path.basename(image_path)}: {error}")
#             else:
#                 probability_results[model_name].append(prob)
#                 latency_results[model_name].append(latency)
#                 route_results[model_name].append(address)
#
#                 print(
#                     f"✅ {model_name} on {os.path.basename(image_path)}: "
#                     f"label={label}, prob={prob:.4f}, latency={latency:.2f} ms, server={address}"
#                 )
#
#     end = time.perf_counter()
#     print(f"\n⏱️ Total time: {(end - start):.2f} seconds")
#
#     print("\n📊 Average Latency per Model:")
#     for model, latencies in latency_results.items():
#         avg_latency = sum(latencies) / len(latencies)
#         print(f"{model:<20}: {avg_latency:.2f} ms")
#
#     print("\n📍 Route per Model:")
#     for model, addresses in route_results.items():
#         unique_addresses = sorted(set(addresses))
#         print(f"{model:<20}: {unique_addresses}")
#
#     with open("model_latencies.json", "w") as f:
#         json.dump(latency_results, f, indent=2)
#
#     with open("model_probabilities.json", "w") as f:
#         json.dump(probability_results, f, indent=2)
#
#     with open("model_routes.json", "w") as f:
#         json.dump(route_results, f, indent=2)
#
#     print("✅ Results saved to model_latencies.json, model_probabilities.json, model_routes.json")
#
#
# if __name__ == "__main__":
#     multiprocessing.freeze_support()
#     main()





import base64
import json
import multiprocessing
import os
import time
import uuid

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Tuple

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

MAX_WORKERS = 1

# When empty, the script executes every model configured
# under ILUVATAR_NODES.
#
# Add model names here to profile only selected models.
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

        models = node.get("models", [])

        if not isinstance(models, list):
            raise ValueError(
                f"Node '{node_id}' field 'models' "
                f"must be a list."
            )

        node_type = str(
            node.get("type", "")
        ).lower()

        if node_type not in ("edge", "cloud"):
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
                "models": list(MODELS_TO_EXECUTE),
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
                if model_name in SMALL_INSTANCE_MODELS
            ],
            "enabled": True,
        },

        "cloud_1": {
            "type": "cloud",
            "address": LARGE_MODEL_SERVER_ADDRESS,
            "models": [
                model_name
                for model_name in MODELS_TO_EXECUTE
                if model_name in LARGE_INSTANCE_MODELS
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
        Builds a compatible node registry from the old settings.
    """

    if USE_MULTI_ILUVATAR_INSTANCES:
        validate_multi_node_config()
        return ILUVATAR_NODES

    return build_legacy_nodes()


def get_models_for_node(
    node_config: dict,
) -> List[str]:
    """
    Return the configured models that should be executed
    on one node.
    """

    configured_models = list(
        node_config.get("models", [])
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

    for node_id, node_config in active_nodes.items():
        if not bool(
            node_config.get("enabled", True)
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
    with open(image_path, "rb") as image_file:
        return base64.b64encode(
            image_file.read()
        ).decode("utf-8")


def get_image_paths() -> List[str]:
    """
    Return supported images from IMAGE_DIR.
    """

    if not os.path.isdir(IMAGE_DIR):
        raise FileNotFoundError(
            f"Image directory does not exist: {IMAGE_DIR}"
        )

    image_paths: List[str] = []

    for filename in sorted(
        os.listdir(IMAGE_DIR)
    ):
        if not filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
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

        image_b64 = read_image_as_base64(
            image_path
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

        start_time = time.perf_counter()

        response = worker.invoke(
            request
        )

        end_time = time.perf_counter()

        wall_latency_ms = (
            end_time - start_time
        ) * 1000.0

        rpc_latency_ms = (
            float(response.duration_us)
            / 1000.0
        )

        result_json = json.loads(
            response.json_result
        )

        if "body" not in result_json:
            raise ValueError(
                f"Unexpected response: {result_json}"
            )

        body = result_json["body"]

        if "Probability" not in body:
            raise ValueError(
                f"Inference failed: {result_json}"
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
            "wall_latency_ms": wall_latency_ms,
            "rpc_latency_ms": rpc_latency_ms,
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
            error_message = error.details()
        except Exception:
            error_message = str(error)

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
            "error": str(error),
        }


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

    return f"{node_id}:{model_name}"


def build_summary(
    all_results: List[dict],
) -> Dict[str, dict]:
    """
    Build average values for every node-model deployment.
    """

    grouped_results: Dict[
        str,
        List[dict],
    ] = defaultdict(list)

    for result in all_results:
        if result.get("error"):
            continue

        key = deployment_key(
            node_id=result["node_id"],
            model_name=result["model_name"],
        )

        grouped_results[key].append(
            result
        )

    summary: Dict[str, dict] = {}

    for key, results in grouped_results.items():
        wall_latencies = [
            float(result["wall_latency_ms"])
            for result in results
            if result["wall_latency_ms"] is not None
        ]

        rpc_latencies = [
            float(result["rpc_latency_ms"])
            for result in results
            if result["rpc_latency_ms"] is not None
        ]

        probabilities = [
            float(result["probability"])
            for result in results
            if result["probability"] is not None
        ]

        first_result = results[0]

        summary[key] = {
            "node_id": first_result["node_id"],
            "node_type": first_result["node_type"],
            "server_address": (
                first_result["server_address"]
            ),
            "model_name": (
                first_result["model_name"]
            ),
            "request_count": len(results),
            "average_wall_latency_ms": (
                sum(wall_latencies)
                / len(wall_latencies)
                if wall_latencies
                else None
            ),
            "average_rpc_latency_ms": (
                sum(rpc_latencies)
                / len(rpc_latencies)
                if rpc_latencies
                else None
            ),
            "average_probability": (
                sum(probabilities)
                / len(probabilities)
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

    image_paths = get_image_paths()

    if not deployment_targets:
        raise ValueError(
            "No enabled model-node deployments were found."
        )

    if not image_paths:
        raise ValueError(
            f"No supported images found in {IMAGE_DIR}"
        )

    print("=== Deployment Summary ===")

    for target in deployment_targets:
        print(
            f"{target['model_name']:<22} "
            f"-> node={target['node_id']:<12} "
            f"type={target['node_type']:<6} "
            f"server={target['server_address']}"
        )

    tasks: List[Tuple[str, str, str, str, str]] = []

    for image_path in image_paths:
        for target in deployment_targets:
            tasks.append(
                (
                    target["node_id"],
                    target["node_type"],
                    target["server_address"],
                    target["model_name"],
                    image_path,
                )
            )

    print(
        f"\nImages: {len(image_paths)}"
    )

    print(
        f"Deployments: {len(deployment_targets)}"
    )

    print(
        f"Total requests: {len(tasks)}"
    )

    all_results: List[dict] = []

    start_time = time.perf_counter()

    try:
        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:

            futures = [
                executor.submit(
                    send_request,
                    node_id,
                    node_type,
                    server_address,
                    model_name,
                    image_path,
                )
                for (
                    node_id,
                    node_type,
                    server_address,
                    model_name,
                    image_path,
                ) in tasks
            ]

            for future in as_completed(
                futures
            ):
                result = future.result()
                all_results.append(result)

                model_name = result["model_name"]
                node_id = result["node_id"]
                image_name = result["image_name"]
                error = result["error"]

                if error:
                    print(
                        f"[EXECUTE_FAIL] "
                        f"node={node_id} "
                        f"model={model_name} "
                        f"image={image_name} "
                        f"error={error}"
                    )

                    continue

                print(
                    f"[EXECUTE_OK] "
                    f"node={node_id} "
                    f"model={model_name} "
                    f"image={image_name} "
                    f"label={result['label']} "
                    f"prob={result['probability']:.4f} "
                    f"wall_latency="
                    f"{result['wall_latency_ms']:.2f}ms "
                    f"rpc_latency="
                    f"{result['rpc_latency_ms']:.2f}ms "
                    f"state={result['container_state']} "
                    f"server={result['server_address']}"
                )

    finally:
        close_all_channels()

    end_time = time.perf_counter()

    total_time_sec = (
        end_time - start_time
    )

    summary = build_summary(
        all_results
    )

    print(
        f"\nTotal time: "
        f"{total_time_sec:.2f} seconds"
    )

    print(
        "\n=== Average Latency per Node and Model ==="
    )

    for key in sorted(summary):
        item = summary[key]

        avg_wall = item[
            "average_wall_latency_ms"
        ]

        avg_rpc = item[
            "average_rpc_latency_ms"
        ]

        print(
            f"{item['node_id']:<12} "
            f"{item['model_name']:<22} "
            f"e2e_latency={avg_wall:.2f}ms "
            # f"rpc={avg_rpc:.2f}ms "
            # f"requests={item['request_count']}"
        )

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


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()



