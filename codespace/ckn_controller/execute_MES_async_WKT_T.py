import grpc
import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import uuid
import base64
import time
import random
import asyncio
from datetime import datetime

# Configuration
M_total = ["mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"]
policy = "greedy"  # or "randomized"
K = 12     # total cores
c = 2     # cores per model
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

model_profiles = {
    "mobilenet_v3_small": {"latency": 0.11397, "accuracy": 0.704413581114262},
    "resnet18": {"latency": 0.10618, "accuracy": 0.712173486609011},
    "resnet34": {"latency": 0.16935, "accuracy": 0.775743676409125},
    "resnet50": {"latency": 0.19834 , "accuracy": 0.797511352837085},
    "resnet101": {"latency": 0.33677, "accuracy": 0.822238205507397},
    "vit_b_16": {"latency": 0.48291, "accuracy": 0.751089387536048},
}
omega = {
    "mobilenet_v3_small": 2.5,
    "resnet18": 11.7,
    "resnet34": 21.8,
    "resnet50": 25.6,
    "resnet101": 44.5,
    "vit_b_16": 86.4,
}

def read_image_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()

async def get_estimated_wait_async(stub, model_name):
    fqdn = f"{model_name}-1"
    request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
    try:
        response = await stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else float('inf')
    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float('inf')

async def get_wait_async(stub):
    wait_tasks = {m: asyncio.create_task(get_estimated_wait_async(stub, m)) for m in M_total}
    return {m: await t for m, t in wait_tasks.items()}


def build_model_set(policy, wait_results, deadline_ms):
    """
    Returns selected model list M_D and total_estimates per model.
    deadline_ms is in milliseconds.
    """
    M_D = []
    total_estimates = {}
    if policy == "greedy":
        cost = {}
        for m in M_total:
            penalty = 0 if m in active_models or (K >= c) else cold_penalty.get(m, 0)
            est_wait = wait_results.get(m, float("inf"))
            est_latency = model_profiles.get(m, {}).get("latency", 0)
            total_est = est_wait + est_latency
            total_estimates[m] = total_est
            if total_est > deadline_ms / 1000:
                cost[m] = float("inf")
                continue
            cost[m] = (total_est + penalty) / (deadline_ms / 1000) + alpha / omega.get(m, 1.0)
            # Debug print
            print(f"[select_models] model={m} cost={cost[m]:.4f} est_wait={est_wait:.4f} est_latency={est_latency:.4f}")

        sorted_models = sorted(M_total, key=lambda m: cost.get(m, float("inf")))
        used_cores = 0
        for m in sorted_models:
            if cost.get(m, float("inf")) == float("inf"):
                continue
            if len(M_D) < K // c and used_cores + c <= K:
                M_D.append(m)
                used_cores += c

    elif policy == "randomized":
        candidate_pool = random.sample(M_total, min(len(M_total), K // c))
        valid = []
        for m in candidate_pool:
            est_wait = wait_results.get(m, float("inf"))
            est_latency = model_profiles.get(m, {}).get("latency", 0)
            if est_wait + est_latency <= deadline_ms / 1000:
                valid.append(m)
        M_D = valid if len(valid) * c <= K else []

    return M_D, total_estimates


async def invoke_ensemble(stub, models, image_b64):
    tasks = [send_request(stub, m, image_b64) for m in models]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return [r for r in results if r.get("success", False)]



async def send_request(stub, model_name, image_b64):
    try:
        request = pb2.InvokeRequest(
            function_name=model_name,
            function_version="1",
            json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
            transaction_id=str(uuid.uuid4()),
        )
        start = time.perf_counter()
        response = await stub.invoke(request)
        end = time.perf_counter()

        latency = end - start
        result_json = json.loads(response.json_result)
        prob = result_json["body"].get("Probability", 0.0)
        accuracy = prob  # or some mapping if probability is proxy
        return {
            "model": model_name,
            "success": True,
            "latency": latency,
            "accuracy": accuracy,
            "container_state": result_json.get("container_state", "UNKNOWN"),
            "status": result_json.get("status", "OK"),
            "prediction": result_json.get("body", {}).get("Prediction", None),
        }
    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "latency": -1,
            "accuracy": 0.0,
            "container_state": "ERROR",
            "status": "ERROR",
            "error": str(e),
        }

def aggregate_outputs(results, method="max_accuracy"):
    if not results:
        return None
    if method == "max_accuracy":
        return max(results, key=lambda x: x["accuracy"])
    elif method == "majority":
        # placeholder: majority vote on labels if available
        counts = {}
        for r in results:
            label = r.get("prediction")  # assuming prediction field
            counts[label] = counts.get(label, 0) + 1
        majority_label = max(counts, key=counts.get)
        # return one result with majority label (simplified)
        for r in results:
            if r.get("prediction") == majority_label:
                return r
    # extend for other methods like weighted average, confidence-weighted
    return results[0]



async def QoED_test(transaction_id: str, deadline: int) -> dict:
    start_time = time.perf_counter()

    # Step 0: Load random image
    img_num = random.randint(0, 999)
    category_choice = random.choice(["cat", "dog"])
    image_path = f"/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/d2iedgeai3/{category_choice}.{img_num}.jpg"
    image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")

    # Step 1: Async gRPC connection
    async_channel = grpc.aio.insecure_channel("149.165.168.72:8079")
    stub = pb2_grpc.IluvatarWorkerStub(async_channel)

    # Step 2: Get wait time estimates
    wait_results = await get_wait_async(stub)

    # Step 3: Select models according to policy
    M_D, total_estimates = build_model_set(policy, wait_results, deadline)

    print(f"[Request {transaction_id}] Selected Models: {M_D}")
    if not M_D:
        print(f"[Request {transaction_id}] No valid models under deadline.")
        return {
            "model": -1,
            "success": False,
            "latency": -1,
            "accuracy": 0.0,
            "container_state": "SKIPPED",
            "status": "Skipped",
            "wait_times": total_estimates,
        }

    # Step 4: Invoke ensemble
    results = await invoke_ensemble(stub, M_D, image_b64)
    if not results:
        print(f"[Request {transaction_id}] No successful model responses.")
        return {
            "model": -1,
            "success": False,
            "latency": -1,
            "accuracy": 0.0,
            "container_state": "FAILED",
            "status": "False",
            "wait_times": total_estimates,
        }

    # Step 5: Aggregate outputs
    best = aggregate_outputs(results, method="max_accuracy")

    # Step 6: Feedback updates
    for res in results:
        omega[res["model"]] = (1 - eta) * omega.get(res["model"], 1.0) + eta * res["accuracy"]
        active_models.add(res["model"])

    # Logging
    print(f"\n[Request {transaction_id}]")
    for res in results:
        print(
            f"Model: {res['model']} | Accuracy: {res['accuracy']:.4f} | Latency: {res['latency']:.3f}s | State: {res['container_state']}"
        )
    print(f"Best Model: {best['model']}")

    end_time = time.perf_counter()
    total_time_ms = (end_time - start_time) * 1000
    print(f"Request {transaction_id} total time: {total_time_ms:.2f} ms\n")

    return {
        "model": best["model"],
        "success": best["success"],
        "latency": best["latency"],
        "accuracy": best["accuracy"],
        "container_state": best["container_state"],
        "selected_models": M_D,
        "cost_function_execution_time_ms": total_time_ms,
        "status": best.get("status", ""),
        "wait_times": total_estimates,
    }


# if __name__ == "__main__":
#     asyncio.run(QoED_test())