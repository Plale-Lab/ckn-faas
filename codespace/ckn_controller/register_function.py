# import uuid
# import grpc
#
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     USE_TWO_ILUVATAR_INSTANCES,
#     SMALL_MODEL_SERVER_ADDRESS,
#     LARGE_MODEL_SERVER_ADDRESS,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
# DOCKER_HUB = "docker.io"
# DOCKER_REPO = "plalelab"
# IMAGE_TAG = "latest"
# SERVER_TYPE = "http"
#
# REGISTER_MODELS = list(SMALL_INSTANCE_MODELS) + list(LARGE_INSTANCE_MODELS)
#
#
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
#     raise ValueError(f"Model {model_name} is not assigned to small or large instance")
#
#
# def get_instance_type(model_name: str) -> str:
#     if not USE_TWO_ILUVATAR_INSTANCES:
#         return "single"
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return "small/cpu"
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return "large/gpu"
#
#     return "unknown"
#
#
# def get_image_name(model_name: str) -> str:
#     if USE_TWO_ILUVATAR_INSTANCES and model_name in LARGE_INSTANCE_MODELS:
#         return (
#             f"{DOCKER_HUB}/{DOCKER_REPO}/"
#             f"ckn_faas_{model_name}-iluvatar-gpu-{SERVER_TYPE}:{IMAGE_TAG}"
#         )
#
#     return (
#         f"{DOCKER_HUB}/{DOCKER_REPO}/"
#         f"ckn_faas_{model_name}-iluvatar-action-{SERVER_TYPE}:{IMAGE_TAG}"
#     )
#
#
# def get_register_resources(model_name: str):
#     if USE_TWO_ILUVATAR_INSTANCES and model_name in LARGE_INSTANCE_MODELS:
#         return {
#             "memory": 1024,
#             "cpus": 1,
#             "compute": 2,
#             "isolate": 2,
#         }
#
#     return {
#         "memory": 512,
#         "cpus": 3,
#         "compute": 1,
#         "isolate": 1,
#     }
#
#
# def build_register_request(model_name: str) -> pb2.RegisterRequest:
#     resources = get_register_resources(model_name)
#
#     return pb2.RegisterRequest(
#         function_name=model_name,
#         function_version="1",
#         image_name=get_image_name(model_name),
#         memory=resources["memory"],
#         cpus=resources["cpus"],
#         parallel_invokes=1,
#         transaction_id=str(uuid.uuid4()),
#         language=pb2.LanguageRuntime.PYTHON3,
#         compute=resources["compute"],
#         isolate=resources["isolate"],
#         container_server=0,
#     )
#
#
# def get_worker_map():
#     addresses = {get_server_address_for_model(m) for m in REGISTER_MODELS}
#     worker_map = {}
#
#     for addr in addresses:
#         channel = grpc.insecure_channel(addr)
#         worker_map[addr] = pb2_grpc.IluvatarWorkerStub(channel)
#
#     return worker_map
#
#
# def is_already_registered_error(text: str) -> bool:
#     text = str(text).lower()
#     keywords = [
#         "already exists",
#         "already registered",
#         "function exists",
#         "duplicate",
#     ]
#     return any(k in text for k in keywords)
#
#
# def main():
#     print("===== Registration Started =====")
#     print(f"USE_TWO_ILUVATAR_INSTANCES = {USE_TWO_ILUVATAR_INSTANCES}")
#     print(f"SMALL_MODEL_SERVER_ADDRESS = {SMALL_MODEL_SERVER_ADDRESS}")
#     print(f"LARGE_MODEL_SERVER_ADDRESS = {LARGE_MODEL_SERVER_ADDRESS}")
#     print(f"REGISTER_MODELS = {REGISTER_MODELS}")
#     print("================================")
#
#     worker_map = get_worker_map()
#
#     ok_count = 0
#     exists_count = 0
#     fail_count = 0
#
#     failed_models = []
#
#     for model_name in REGISTER_MODELS:
#         address = get_server_address_for_model(model_name)
#         instance_type = get_instance_type(model_name)
#         image_name = get_image_name(model_name)
#         worker = worker_map[address]
#         request = build_register_request(model_name)
#
#         print(
#             f"\n[REGISTER_START] model={model_name} "
#             f"instance={instance_type} server={address}"
#         )
#         print(
#             f"[REGISTER_CONFIG] image={image_name} "
#             f"memory={request.memory} cpus={request.cpus} "
#             f"compute={request.compute} isolate={request.isolate}"
#         )
#
#         try:
#             response = worker.register(request)
#
#             response_text = str(response)
#             success = bool(getattr(response, "success", False))
#             error_msg = str(getattr(response, "error", ""))
#
#             if success:
#                 ok_count += 1
#                 print(
#                     f"[REGISTER_OK] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"response={response_text}"
#                 )
#
#             elif is_already_registered_error(error_msg) or is_already_registered_error(response_text):
#                 exists_count += 1
#                 print(
#                     f"[REGISTER_EXISTS] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"message={response_text}"
#                 )
#
#             else:
#                 fail_count += 1
#                 failed_models.append((model_name, image_name, response_text))
#                 print(
#                     f"[REGISTER_FAIL] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"error={response_text}"
#                 )
#
#         except grpc.RpcError as e:
#             error_details = str(e.details())
#             error_code = e.code()
#
#             if is_already_registered_error(error_details):
#                 exists_count += 1
#                 print(
#                     f"[REGISTER_EXISTS] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"code={error_code} error={error_details}"
#                 )
#             else:
#                 fail_count += 1
#                 failed_models.append((model_name, image_name, error_details))
#                 print(
#                     f"[REGISTER_FAIL] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"code={error_code} error={error_details}"
#                 )
#
#         except Exception as e:
#             error_text = str(e)
#
#             if is_already_registered_error(error_text):
#                 exists_count += 1
#                 print(
#                     f"[REGISTER_EXISTS] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"error={error_text}"
#                 )
#             else:
#                 fail_count += 1
#                 failed_models.append((model_name, image_name, error_text))
#                 print(
#                     f"[REGISTER_FAIL] model={model_name} "
#                     f"instance={instance_type} server={address} "
#                     f"error={error_text}"
#                 )
#
#     print("\n===== Registration Summary =====")
#     print(f"REGISTER_OK     = {ok_count}")
#     print(f"REGISTER_EXISTS = {exists_count}")
#     print(f"REGISTER_FAIL   = {fail_count}")
#
#     if failed_models:
#         print("\nFailed models:")
#         for model_name, image_name, error in failed_models:
#             print(f"- {model_name}: {image_name}")
#             print(f"  error={error}")
#
#     print("===== Registration Finished =====")
#
#
# if __name__ == "__main__":
#     main()



import uuid
from typing import Dict, Tuple

import grpc

import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    USE_MULTI_ILUVATAR_INSTANCES,
    ILUVATAR_NODES,
)


# ============================================================
# Configuration validation
# ============================================================

def validate_config() -> None:
    """
    Validate the multi-node configuration before registration.
    """

    if not USE_MULTI_ILUVATAR_INSTANCES:
        raise ValueError(
            "USE_MULTI_ILUVATAR_INSTANCES must be True "
            "to use this registration script."
        )

    if not isinstance(ILUVATAR_NODES, dict):
        raise ValueError(
            "ILUVATAR_NODES must be a dictionary."
        )

    if not ILUVATAR_NODES:
        raise ValueError(
            "ILUVATAR_NODES is empty."
        )

    valid_node_types = {
        "edge",
        "cloud",
    }

    for node_id, node_config in ILUVATAR_NODES.items():
        if not isinstance(node_config, dict):
            raise ValueError(
                f"Node '{node_id}' configuration must be a dictionary."
            )

        address = node_config.get("address")

        if not address:
            raise ValueError(
                f"Node '{node_id}' is missing 'address'."
            )

        node_type = str(
            node_config.get("type", "")
        ).lower()

        if node_type not in valid_node_types:
            raise ValueError(
                f"Node '{node_id}' has invalid type '{node_type}'. "
                f"Expected 'edge' or 'cloud'."
            )

        models = node_config.get("models")

        if not isinstance(models, list):
            raise ValueError(
                f"Node '{node_id}' field 'models' must be a list."
            )

        if not models:
            print(
                f"[CONFIG_WARNING] "
                f"node={node_id} has no configured models"
            )

        for model_name in models:
            if not isinstance(model_name, str):
                raise ValueError(
                    f"Node '{node_id}' contains an invalid model name: "
                    f"{model_name}"
                )


# ============================================================
# Resource configuration
# ============================================================

def get_model_resources(
    model_name: str,
    node_config: dict,
) -> Tuple[int, int, int]:
    """
    Read model registration resources from ckn_config.py.

    Priority:

    1. model_resources[model_name]
    2. Node-level defaults
    3. Safe fallback values
    """

    model_resources = node_config.get(
        "model_resources",
        {},
    )

    model_override = model_resources.get(
        model_name,
        {},
    )

    memory_mb = int(
        model_override.get(
            "memory_mb",
            node_config.get(
                "default_memory_mb",
                512,
            ),
        )
    )

    cpus = int(
        model_override.get(
            "cpus",
            node_config.get(
                "default_cpus",
                1,
            ),
        )
    )

    parallel_invokes = int(
        model_override.get(
            "parallel_invokes",
            node_config.get(
                "default_parallel_invokes",
                1,
            ),
        )
    )

    return (
        memory_mb,
        cpus,
        parallel_invokes,
    )


# ============================================================
# Docker image configuration
# ============================================================

def get_image_name(
    model_name: str,
    node_config: dict,
) -> str:
    """
    Read the Docker image from the configuration when provided.

    Supported configuration:

    "model_images": {
        "resnet18": "docker.io/...:latest"
    }

    Otherwise, generate the image name using the default pattern.
    """

    model_images = node_config.get(
        "model_images",
        {},
    )

    configured_image = model_images.get(
        model_name
    )

    if configured_image:
        return str(configured_image)

    image_prefix = str(
        node_config.get(
            "image_prefix",
            "docker.io/plalelab/ckn_faas_",
        )
    )

    image_suffix = str(
        node_config.get(
            "image_suffix",
            "-iluvatar-action-http:latest",
        )
    )

    return (
        f"{image_prefix}"
        f"{model_name}"
        f"{image_suffix}"
    )


# ============================================================
# Registration request
# ============================================================

def build_register_request(
    model_name: str,
    node_config: dict,
) -> pb2.RegisterRequest:
    """
    Build an Iluvatar registration request using only information
    from ckn_config.py.
    """

    (
        memory_mb,
        cpus,
        parallel_invokes,
    ) = get_model_resources(
        model_name=model_name,
        node_config=node_config,
    )

    image_name = get_image_name(
        model_name=model_name,
        node_config=node_config,
    )

    compute = int(
        node_config.get(
            "compute",
            1,
        )
    )

    isolate = int(
        node_config.get(
            "isolate",
            1,
        )
    )

    container_server = int(
        node_config.get(
            "container_server",
            0,
        )
    )

    return pb2.RegisterRequest(
        function_name=model_name,
        function_version="1",
        image_name=image_name,
        memory=memory_mb,
        cpus=cpus,
        parallel_invokes=parallel_invokes,
        transaction_id=str(uuid.uuid4()),
        language=pb2.LanguageRuntime.PYTHON3,
        compute=compute,
        isolate=isolate,
        container_server=container_server,
    )


# ============================================================
# gRPC connection
# ============================================================

def create_worker(
    server_address: str,
):
    """
    Create one synchronous gRPC channel and worker stub.
    """

    channel = grpc.insecure_channel(
        server_address
    )

    worker = pb2_grpc.IluvatarWorkerStub(
        channel
    )

    return channel, worker


# ============================================================
# Register one model
# ============================================================

def register_model(
    worker,
    node_id: str,
    node_type: str,
    server_address: str,
    model_name: str,
    node_config: dict,
) -> bool:
    """
    Register one model on one Iluvatar node.
    """

    request = build_register_request(
        model_name=model_name,
        node_config=node_config,
    )

    print(
        f"[REGISTER_START] "
        f"node={node_id} "
        f"type={node_type} "
        f"model={model_name} "
        f"server={server_address} "
        f"image={request.image_name} "
        f"memory_mb={request.memory} "
        f"cpus={request.cpus} "
        f"parallel_invokes={request.parallel_invokes}"
    )

    try:
        response = worker.register(
            request
        )

        print(
            f"[REGISTER_OK] "
            f"node={node_id} "
            f"type={node_type} "
            f"model={model_name} "
            f"server={server_address} "
            f"response={response}"
        )

        return True

    except grpc.RpcError as error:
        try:
            details = error.details()
        except Exception:
            details = str(error)

        print(
            f"[REGISTER_FAIL] "
            f"node={node_id} "
            f"type={node_type} "
            f"model={model_name} "
            f"server={server_address} "
            f"error={details}"
        )

        return False

    except Exception as error:
        print(
            f"[REGISTER_FAIL] "
            f"node={node_id} "
            f"type={node_type} "
            f"model={model_name} "
            f"server={server_address} "
            f"error={error}"
        )

        return False


# ============================================================
# Register all models on one node
# ============================================================

def register_node(
    node_id: str,
    node_config: dict,
) -> Dict[str, bool]:
    """
    Register all models configured for one node.
    """

    enabled = bool(
        node_config.get(
            "enabled",
            True,
        )
    )

    if not enabled:
        print(
            f"[NODE_SKIP] "
            f"node={node_id} "
            f"reason=disabled"
        )

        return {}

    server_address = str(
        node_config["address"]
    )

    node_type = str(
        node_config["type"]
    ).lower()

    models = list(
        node_config.get(
            "models",
            [],
        )
    )

    if not models:
        print(
            f"[NODE_SKIP] "
            f"node={node_id} "
            f"type={node_type} "
            f"server={server_address} "
            f"reason=no_models"
        )

        return {}

    channel, worker = create_worker(
        server_address=server_address
    )

    results: Dict[str, bool] = {}

    try:
        for model_name in models:
            success = register_model(
                worker=worker,
                node_id=node_id,
                node_type=node_type,
                server_address=server_address,
                model_name=model_name,
                node_config=node_config,
            )

            results[model_name] = success

    finally:
        try:
            channel.close()
        except Exception:
            pass

    return results


# ============================================================
# Summary
# ============================================================

def print_summary(
    registration_results: Dict[
        str,
        Dict[str, bool],
    ],
) -> None:
    total = 0
    successful = 0
    failed = 0

    print(
        "\n=========================================="
    )
    print(
        "          REGISTRATION SUMMARY"
    )
    print(
        "=========================================="
    )

    for node_id, model_results in registration_results.items():
        node_config = ILUVATAR_NODES[node_id]

        print(
            f"\nNode: {node_id}"
        )

        print(
            f"Type: {node_config.get('type')}"
        )

        print(
            f"Address: {node_config.get('address')}"
        )

        if not model_results:
            print(
                "No models registered."
            )
            continue

        for model_name, success in model_results.items():
            total += 1

            if success:
                successful += 1
                status = "OK"
            else:
                failed += 1
                status = "FAILED"

            print(
                f"  {model_name:<25} {status}"
            )

    print(
        "\n------------------------------------------"
    )

    print(
        f"Total registrations: {total}"
    )

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    print(
        "==========================================\n"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    validate_config()

    registration_results: Dict[
        str,
        Dict[str, bool],
    ] = {}

    for node_id, node_config in ILUVATAR_NODES.items():
        registration_results[node_id] = register_node(
            node_id=node_id,
            node_config=node_config,
        )

    print_summary(
        registration_results
    )


if __name__ == "__main__":
    main()