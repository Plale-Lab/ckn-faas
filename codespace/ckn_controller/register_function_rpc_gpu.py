import uuid
import grpc

import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    USE_TWO_ILUVATAR_INSTANCES,
    SMALL_MODEL_SERVER_ADDRESS,
    LARGE_MODEL_SERVER_ADDRESS,
)

model_list = [
    # "mobilenet_v3_small",
    # "resnet18",
    # "resnet34",
    # "resnet50",
    # "resnet101",
    # "vit_b_16",

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

SMALL_INSTANCE_MODELS = {"mobilenet_v3_small", "resnet18", "resnet34", "resnet50"}
LARGE_INSTANCE_MODELS = {"resnet101", "vit_b_16"}

DOCKER_HUB = "docker.io"
DOCKER_REPO = "plalelab"
IMAGE_TAG = "latest"
SERVER_TYPE = "http"   # use "unix" if you want unix server images


def get_server_address_for_model(model_name: str) -> str:
    if not USE_TWO_ILUVATAR_INSTANCES:
        return SERVER_ADDRESS
    if model_name in SMALL_INSTANCE_MODELS:
        return SMALL_MODEL_SERVER_ADDRESS
    return LARGE_MODEL_SERVER_ADDRESS


def get_worker_map():
    addresses = {get_server_address_for_model(m) for m in model_list}
    worker_map = {}

    for addr in addresses:
        channel = grpc.insecure_channel(addr)
        worker_map[addr] = pb2_grpc.IluvatarWorkerStub(channel)

    return worker_map


def get_image_name(model_name: str) -> str:
    return f"{DOCKER_HUB}/{DOCKER_REPO}/ckn_faas_{model_name}-iluvatar-gpu-{SERVER_TYPE}:{IMAGE_TAG}"

# def get_image_name(model_name: str) -> str:
#     return f"{DOCKER_HUB}/{DOCKER_REPO}/cnn_image_classification_gpu-iluvatar-gpu-{SERVER_TYPE}:{IMAGE_TAG}"

# one image
# def get_image_name():
#     return "docker.io/plalelab/cnn_image_classification_gpu-iluvatar-gpu-http:latest"

def build_register_request(model_name: str) -> pb2.RegisterRequest:
    if model_name in {"resnet101", "vit_b_16"}:
        memory_mb = 1024
        cpus = 3
    else:
        memory_mb = 512
        cpus = 3

    return pb2.RegisterRequest(
        function_name=model_name,
        function_version="1",
        image_name=get_image_name(model_name),
        memory=memory_mb,
        cpus=1,
        parallel_invokes=1,
        transaction_id=str(uuid.uuid4()),
        language=pb2.LanguageRuntime.PYTHON3,
        compute=2,
        isolate=2,
        container_server=0,
    )


def main():
    worker_map = get_worker_map()

    for model_name in model_list:
        address = get_server_address_for_model(model_name)
        worker = worker_map[address]

        request = build_register_request(model_name)

        try:
            response = worker.register(request)
            print(
                f"[REGISTER_OK] model={model_name} "
                f"server={address} image={get_image_name(model_name)} response={response}"
            )
        except grpc.RpcError as e:
            print(f"[REGISTER_FAIL] model={model_name} server={address} error={e.details()}")
        except Exception as e:
            print(f"[REGISTER_FAIL] model={model_name} server={address} error={e}")


if __name__ == "__main__":
    main()