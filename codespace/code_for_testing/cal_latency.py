import grpc
import code_for_testing.iluvatar_rpc_pb2 as pb2
import code_for_testing.iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import uuid
import base64
import time
import random
import asyncio
from datetime import datetime
from code_for_testing.ckn_config import (
    SERVER_ADDRESS, M_TOTAL, POLICY, K, C, ALPHA, ETA,
    MODEL_PROFILES, COLD_PENALTY, OMEGA, MAX_MODEL_SIZE
)
import wait_time_iluvatar
import logging
import os

active_models = set()
A = []


def read_image_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()


async def send_request(stub, model_name, image_b64):
    """
    Invoke a single model and return server-reported metrics.
    (We'll add wall-clock e2e time around this in the caller.)
    """
    request = pb2.InvokeRequest(
        function_name=model_name,
        function_version="1",
        json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
        transaction_id=str(uuid.uuid4()),
    )
    try:
        response = await stub.invoke(request)
        result_json = json.loads(response.json_result)
        return {
            "model": model_name,
            "label": result_json["body"]["Prediction Class"],
            "probability": float(result_json["body"]["Probability"]),
            "latency": response.duration_us / 1e6,  # server-reported
            "success": response.success,
            "container_state": pb2.ContainerState.Name(response.container_state),
        }
    except Exception as e:
        logging.error(f"send_request failed for model {model_name}: {e}")
        return {
            "model": model_name,
            "label": -1,
            "probability": 0.0,
            "latency": -1.0,
            "success": False,
            "container_state": "UNKNOWN",
        }


def get_estimated_wait(stub, model_name):
    """Get estimated wait time for a model from Iluvatar."""
    fqdn = f"{model_name}-1"
    request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
    try:
        response = stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else float("inf")
    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float("inf")


def get_wait():
    """Get estimated wait time for all models (seconds)."""
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = pb2_grpc.IluvatarWorkerStub(channel)
    wait_results = {}
    for model in M_TOTAL:
        wait_time = get_estimated_wait(stub, model)
        wait_results[model] = wait_time
    print("Estimated latency (sec):")
    print(json.dumps(wait_results, indent=2))
    return wait_results


def build_model_set(wait_results, deadline_ms):
    """
    Select ALL models (ignore any policy); keep the estimates as-is.
    """
    M_D = list(M_TOTAL)
    total_estimates = {m: wait_results.get(m, float("inf")) for m in M_TOTAL}
    print(f"M_D: {M_D}")
    return M_D, total_estimates


async def main_ensemble_invoke(transaction_id: str, deadline: int) -> dict:
    start_time = time.perf_counter()

    # Step 0: Load random image
    base_folder = "/Users/agamage/Desktop/D2I/Codes Original/Mode-S/archive/train.X1"
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    selected_folder = random.choice(folders)
    folder_path = os.path.join(base_folder, selected_folder)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random_image = random.choice(images)
    image_path = os.path.join(folder_path, random_image)

    image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")
    print(f"[Request {transaction_id}] Using image: {image_path}")

    # Step 1: gRPC connection
    async_channel = grpc.aio.insecure_channel(SERVER_ADDRESS)
    stub = pb2_grpc.IluvatarWorkerStub(async_channel)

    # Step 2: Get estimated latency (seconds) for all models
    # If you prefer direct RPC per run, you can call get_wait().
    # Here we keep your existing helper:
    estimated = wait_time_iluvatar.main()  # must return {model: seconds}
    # Fallback if you want to ensure structure:
    # estimated = get_wait()

    M_D, total_estimates = build_model_set(estimated, deadline)
    print(f"[Request {transaction_id}] Selected Models: {M_D}")

    # Step 3: Invoke all models and measure wall-clock e2e time
    results = []
    actual_latency = {}  # model -> e2e seconds
    for m in M_D:
        t0 = time.perf_counter()
        res = await send_request(stub, m, image_b64)
        t1 = time.perf_counter()
        res["e2e_time_s"] = t1 - t0
        results.append(res)
        actual_latency[m] = res["e2e_time_s"]

    # Optional: log
    for m, e2e in actual_latency.items():
        print(f"Model: {m} | actual e2e: {e2e:.3f}s | estimated: {total_estimates.get(m, float('inf')):.3f}s")

    # Final return — rename keys exactly as requested
    return {
        "estimated_latency": total_estimates,  # { model: seconds }
        "actual_latency": actual_latency       # { model: seconds } (wall-clock e2e)
    }


# Example direct run:
# if __name__ == "__main__":
#     out = asyncio.run(main_ensemble_invoke(transaction_id=str(uuid.uuid4()), deadline=20000))
#     print(json.dumps(out, indent=2))
