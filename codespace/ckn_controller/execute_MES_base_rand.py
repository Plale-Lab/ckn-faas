import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import uuid
import base64
import time
import random
import asyncio
from datetime import datetime

#Code for randomly selecting n models for ensemble

# Configuration
M_total = ["mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"]
policy = "greedy"  # or "randomized"
D = [1000*1000,1900*1000,1900*1000] # tight deadline for each request
R = range(len(D))
K = 8     # total cores
c = 1     # cores per model
alpha = 1.0
eta = 0.1
active_models = set()
cold_penalty = {
    "mobilenet_v3_small": 30,
    "resnet18": 40,
    "resnet34": 50,
    "resnet50": 60,
    "resnet101": 70,
    "vit_b_16": 80,
}

omega = {}
A = []
assert len(R) == len(D)
# Model profiles
model_profiles = {
    "mobilenet_v3_small": {"latency": 14.2121078288183, "accuracy": 0.704413581114262},
    "resnet18": {"latency": 28.992410373874, "accuracy": 0.712173486609011},
    "resnet34": {"latency": 42.8559443559499, "accuracy": 0.775743676409125},
    "resnet50": {"latency": 63.9653322417292, "accuracy": 0.797511352837085},
    "resnet101": {"latency": 105.657157612033, "accuracy": 0.822238205507397},
    "vit_b_16": {"latency": 175.83452499425, "accuracy": 0.751089387536048},
}
omega = {
    "mobilenet_v3_small": 2.5,
    "resnet18": 11.7,
    "resnet34": 21.8,
    "resnet50": 25.6,
    "resnet101": 44.5,
    "vit_b_16": 86.4,
}

import time

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
        print(response)
        result_json = json.loads(response.json_result)
        return {
            "model": model_name,
            "accuracy": float(result_json["body"]["Probability"]),
            "latency": response.duration_us / 1e6,
            "success": response.success,
            "container_state": pb2.ContainerState.Name(response.container_state),
        }
    except Exception as e:
        return {"model": model_name, "error": str(e), "accuracy": 0.0, "latency": -1}

async def get_estimated_wait_async(stub, model_name):
    fqdn = f"{model_name}-1"
    request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
    try:
        response = await stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else float('inf')
    except grpc.aio.AioRpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float('inf')

async def QoED_test():
    start_time = time.perf_counter()
    # Load image once
    img_num = random.randint(0, 999)
    category_choice = random.choice(["cat", "dog"])
    image_path = f"/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/d2iedgeai3/{category_choice}.{img_num}.jpg"
    image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")

    async_channel = grpc.aio.insecure_channel("149.165.155.226:8079")
    stub = pb2_grpc.IluvatarWorkerStub(async_channel)

    A = []
    # R = list(range(10))  # Simulate 10 requests

    for i,r in enumerate(R):
        M_D = []

        # Step 1: Get async wait times for all models
        wait_tasks = {m: asyncio.create_task(get_estimated_wait_async(stub, m)) for m in M_total}
        wait_results = {m: await t for m, t in wait_tasks.items()}

        # Step 2: Select models based on policy
        if policy == "greedy":
            cost = {}
            for m in M_total:
                penalty = 0 if m in active_models or (K >= c) else cold_penalty[m]
                est_wait = wait_results[m]
                est_latency = model_profiles[m]["latency"] # replace with profiling if available
                total_est = est_wait + est_latency
                if total_est > D[i] / 1000:
                    cost[m] = float('inf')
                    continue
                cost[m] = (total_est + penalty) / (D[i] / 1000) + alpha / omega[m]

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
                est_wait = wait_results[m]
                est_latency = model_profiles[m]["latency"]
                if est_wait + est_latency <= D[i]:
                    valid.append(m)
            M_D = valid if len(valid) * c <= K else []

        print(f"[Request {r}] Selected Models: {M_D}")
        if not M_D:
            print(f"[Request {r}] No valid models under deadline.")
            continue

        # Step 3: Invoke models concurrently
        tasks = [send_request(stub, m, image_b64) for m in M_D]
        results = await asyncio.gather(*tasks)
        results = [res for res in results if res.get("success", False)]
        if not results:
            continue

        # Step 4: Select best
        best = max(results, key=lambda x: x["accuracy"])
        A.append(best["model"])

        # Step 5: Feedback update
        for res in results:
            omega[res["model"]] = (1 - eta) * omega[res["model"]] + eta * res["accuracy"]
            active_models.add(res["model"])

        print(f"\n[Request {r}]")
        for res in results:
            print(f"Model: {res['model']} | Accuracy: {res['accuracy']:.4f} | Latency: {res['latency']:.3f}s | State: {res['container_state']}")
        print(f"Best Model: {best['model']}")

    print("\nFinal Predictions:")
    print(A)
    end_time = time.perf_counter()
    print(f"Total time for {len(R)} requests: {(end_time - start_time) * 1000:.2f} ms")

if __name__ == "__main__":
    asyncio.run(QoED_test())