import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import os
import uuid
import base64
import time
import random
import asyncio
from datetime import datetime

# Configuration
M_total = ["shufflenet_v2_x0_5", "mobilenet_v3_small", "googlenet",
           "resnext50_32x4d", "densenet201", "resnet152"]
policy = "greedy"
D = 3.5  # seconds
K = 8
c = 2
alpha = 1.0
eta = 0.1
active_models = set()
omega = {m: 1.0 / len(m) for m in M_total}
cold_penalty = {m: random.uniform(0.3, 0.7) for m in M_total}

def read_image_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()

async def send_request(stub, model_name, image_b64):
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
            "accuracy": float(result_json["body"]["Probability"]),
            "latency": response.duration_us / 1e6,
            "success": response.success,
            "container_state": pb2.ContainerState.Name(response.container_state),
        }
    except Exception as e:
        return {"model": model_name, "error": str(e), "accuracy": 0.0, "latency": D + 1}

def get_estimated_wait(stub, model_name):
    fqdn = f"{model_name}-1"
    request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
    try:
        response = stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else float('inf')
    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float('inf')

async def QoED_test():
    # Load one image for all requests
    img_num = random.randint(0, 999)
    category_choice = random.choice(["cat", "dog"])
    image_path = f"/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/{category_choice}.{img_num}.jpg"
    image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")

    async_channel = grpc.aio.insecure_channel("127.0.0.1:8079")
    sync_channel = grpc.insecure_channel("127.0.0.1:8079")
    stub_async = pb2_grpc.IluvatarWorkerStub(async_channel)
    stub_sync = pb2_grpc.IluvatarWorkerStub(sync_channel)

    A = []
    R = list(range(10))

    for r in R:
        x = r
        M_D = []

        if policy == "greedy":
            cost = {}
            for m in M_total:
                penalty = 0 if m in active_models else cold_penalty[m]
                est_wait = get_estimated_wait(stub_sync, m)
                est_latency = 1.0  # Placeholder or profiled latency
                total_est = est_wait + est_latency
                if total_est > D:
                    cost[m] = float('inf')
                    continue
                cost[m] = (total_est + penalty) / D + alpha / omega[m]

            sorted_models = sorted(M_total, key=lambda m: cost[m])
            used_cores = 0
            for m in sorted_models:
                if len(M_D) < K // c and used_cores + c <= K:
                    M_D.append(m)
                    used_cores += c

        elif policy == "randomized":
            candidate_pool = random.sample(M_total, min(len(M_total), K // c))
            valid = []
            for m in candidate_pool:
                est_wait = get_estimated_wait(stub_sync, m)
                est_latency = 1.0
                total = est_wait + est_latency
                if total <= D:
                    valid.append(m)
            M_D = valid if len(valid) * c <= K else []

        if not M_D:
            print(f"[Request {r}] No valid models under deadline.")
            continue

        tasks = [send_request(stub_async, m, image_b64) for m in M_D]
        results = await asyncio.gather(*tasks)
        results = [res for res in results if res.get("success", False)]
        if not results:
            continue

        best = max(results, key=lambda x: x["accuracy"])
        A.append(best["model"])

        for res in results:
            omega[res["model"]] = (1 - eta) * omega[res["model"]] + eta * res["accuracy"]
            active_models.add(res["model"])

        print(f"\n[Request {r}]")
        for res in results:
            print(f"Model: {res['model']} | Accuracy: {res['accuracy']:.4f} | Latency: {res['latency']:.3f}s | State: {res['container_state']}")
        print(f"Best Model: {best['model']}")

    print("\nFinal Predictions:")
    print(A)

if __name__ == "__main__":
    asyncio.run(QoED_test())