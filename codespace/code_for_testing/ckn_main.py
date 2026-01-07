import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import code_for_testing.iluvatar_rpc_pb2 as pb2
import code_for_testing.iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import uuid
import base64
import time
import random
import asyncio
from datetime import datetime
from code_for_testing.ckn_config  import SERVER_ADDRESS, M_TOTAL, POLICY, K, C, ALPHA, ETA,MODEL_PROFILES,COLD_PENALTY, OMEGA, MAX_MODEL_SIZE, DEFAULT_WEIGHTS, GAMMA, RHO, WEIGHTS_STATE_PATH, MODEL_SIZES
import wait_time_iluvatar
from code_for_testing.output_combiner import combine_outputs
from code_for_testing.weights_io import load_model_weights, save_model_weights_atomic, diff_weights
import logging
import os
import itertools
from code_for_testing.label_utils import wnid_matches_text_label

MODEL_WEIGHTS = load_model_weights(WEIGHTS_STATE_PATH, DEFAULT_WEIGHTS)
active_models = set()

A = []

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
        print("result_json........", result_json)
        return {
            "model": model_name,
            "label": result_json["body"]["Prediction Class"],
            "probability": float(result_json["body"]["Probability"]),
            "latency": response.duration_us / 1e6,
            "success": response.success,
            "container_state": pb2.ContainerState.Name(response.container_state),
        }
    except Exception as e:
        logging.error(f"send_request failed for model {model_name}: {e}")
        raise


def get_estimated_wait(stub, model_name):
    fqdn = f"{model_name}-1"
    request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
    try:
        response = stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else float('inf')
    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float('inf')

def get_wait():
    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    wait_results = {}

    for model in M_TOTAL:
        wait_time = get_estimated_wait(stub, model)
        wait_results[model] = wait_time

    print(json.dumps(wait_results, indent=2))
    return wait_results

# def build_model_set(wait_results, deadline_ms):
#     """
#     Returns selected model list M_D and total_estimates per model.
#     deadline_ms is in milliseconds.
#     """
#     M_D = []
#     total_estimates = {}
#     if POLICY == "greedy":
#         cost = {}
#         for m in M_TOTAL:
#             print(f"model........111: {m}")
#             penalty = 0 if m in active_models or (K >= C) else COLD_PENALTY[m]
#
#             total_est = wait_results[m]
#             total_estimates[m] = total_est
#             if total_est > deadline_ms / 1000:
#                 cost[m] = float('inf')
#                 continue
#             cost[m] = (total_est + penalty) / (deadline_ms / 1000) + ALPHA / OMEGA[m]
#             print(f"cost........: {cost[m]}")
#
#         sorted_models = sorted(M_TOTAL, key=lambda m: cost[m])
#         used_cores = 0
#         for m in sorted_models:
#             if cost[m] == float('inf'):
#                 continue
#             if len(M_D) < min(K // C,
#                               MAX_MODEL_SIZE) and used_cores + C <= K:  # the maximum number of models that can run in parallel on the available cores.
#                 M_D.append(m)
#                 used_cores += C
#         print(f"M_D: {M_D}")
#
#     elif POLICY == "randomized":
#         # candidate_pool = random.sample(M_TOTAL, min(len(M_TOTAL), K // C, MAX_MODEL_SIZE))
#         # valid = []
#         # for m in candidate_pool:
#         #     est_wait = wait_results[m]
#         #     est_latency = MODEL_PROFILES[m]["latency"]
#         #     if est_wait + est_latency <= deadline_ms / 1000:
#         #         valid.append(m)
#         # M_D = valid if len(valid) * C <= K else []
#         # randomly pick exactly 2 models from M_TOTAL
#         sample_size = min(MAX_MODEL_SIZE, len(M_TOTAL))  # avoid error if <2 models available
#         M_D = random.sample(M_TOTAL, sample_size)
#
#         # include their wait estimates (optional)
#         total_estimates = {m: wait_results.get(m, float("inf")) for m in M_D}
#
#         print(f"[randomized] Randomly selected models: {M_D}")
#
#     return M_D, total_estimates



def build_model_set(wait_results: dict, deadline_ms: int):
    """
    Select a model-set MS using  cost function:

        C_r = (T_e / D) + ALPHA * (MaxMS[size] / sum_bytes_MS)

    where:
      - T_e = estimated service time (sec) for executing x on all models in MS
      - D   = deadline (sec)
      - MaxMS[size] = sum of 'size' largest model sizes (max possible accuracy proxy)
      - sum_bytes_MS = sum of model sizes in the selected set MS
    """
    D_sec = deadline_ms / 1000.0

    # Max number of models we can run in parallel
    max_parallel_models = min(MAX_MODEL_SIZE, K // C, len(M_TOTAL))


    sorted_by_size = sorted(M_TOTAL, key=lambda m: MODEL_SIZES[m], reverse=True)
    MaxMS = {}
    rolling_sum = 0
    for i, m in enumerate(sorted_by_size):
        rolling_sum += MODEL_SIZES[m]
        size = i + 1
        if size <= max_parallel_models:
            MaxMS[size] = rolling_sum

    def estimate_Te(MS):
        """
        Estimate service time T_e(MS): wait + compute for each model, then max
        (since they run logically in parallel).
        """
        times = []
        for m in MS:
            wait_t = wait_results.get(m, float("inf"))
            compute_t = MODEL_PROFILES[m]["latency"]
            times.append(wait_t + compute_t)
        return max(times) if times else float("inf")

    best_MS = None
    best_cost = float("inf")
    best_waits = {}

    if POLICY == "greedy":
        # Consider all sizes 1..max_parallel_models (no hard preference order).
        for size in range(1, max_parallel_models + 1):
            for MS in itertools.combinations(M_TOTAL, size):
                MS = list(MS)

                if size * C > K:
                    continue

                T_e = estimate_Te(MS)
                raw = T_e / D_sec

                if raw >= 1.0:
                    continue

                latency_term = min(max(raw, 0.1), 0.9)

                sum_bytes_MS = sum(MODEL_SIZES[m] for m in MS)
                if sum_bytes_MS <= 0:
                    continue

                max_bytes_for_size = MaxMS[size]

                accuracy_term = ALPHA * (max_bytes_for_size / sum_bytes_MS)
                cost = latency_term + accuracy_term

                if cost < best_cost:
                    best_cost = cost
                    best_MS = MS
                    best_waits = {
                        m: wait_results.get(m, float("inf")) for m in MS
                    }

        # If no ensemble of any size satisfies T_e / D < 1, pick single fastest model
        if best_MS is None:
            fastest = min(M_TOTAL, key=lambda m: MODEL_PROFILES[m]["latency"])
            best_MS = [fastest]
            best_waits = {fastest: wait_results.get(fastest, float("inf"))}

        M_D = best_MS
        total_estimates = best_waits

    elif POLICY == "randomized":
        import random
        sample_size = min(MAX_MODEL_SIZE, len(M_TOTAL))
        M_D = random.sample(M_TOTAL, sample_size)
        total_estimates = {m: wait_results.get(m, float("inf")) for m in M_D}
        best_cost = float("nan")  # not computed in randomized

    else:
        M_D = []
        total_estimates = {}
        best_cost = float("nan")

    print(
        f"[build_model_set] Selected: {M_D}, policy={POLICY}, "
        f"cost={best_cost if POLICY == 'greedy' else 'N/A'}"
    )
    return M_D, total_estimates







async def main_ensemble_invoke(transaction_id: str, deadline: int) -> dict:
    start_time = time.perf_counter()

    # Step 0: Load random image
    # img_num = random.randint(0, 999)
    # category_choice = random.choice(["cat", "dog"])
    # image_path = f"/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/d2iedgeai3/{category_choice}.{img_num}.jpg"
    # image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")
    # Path to your train.X2 folder
    base_folder = "/Users/agamage/Desktop/D2I/Codes Original/Mode-S/archive/train.X1"

    # Pick a random folder (only directories)
    folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]
    selected_folder = random.choice(folders)  # 👈 keep this
    folder_path = os.path.join(base_folder, selected_folder)

    # Pick a random image (only common image extensions)
    images = [f for f in os.listdir(folder_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    random_image = random.choice(images)
    image_path = os.path.join(folder_path, random_image)

    # Convert image to base64
    image_b64 = base64.b64encode(read_image_as_bytes(image_path)).decode("utf-8")
    print(f"[Request {transaction_id}] Using image: {image_path}")

    # Step 1: gRPC connection
    async_channel = grpc.aio.insecure_channel(SERVER_ADDRESS)
    stub = pb2_grpc.IluvatarWorkerStub(async_channel)

    # Step 2: Get wait time estimates
    # wait_tasks = {m: asyncio.create_task(get_estimated_wait_async(stub, m)) for m in M_total}
    # wait_results = {m: await t for m, t in wait_tasks.items()}
    wait_results = wait_time_iluvatar.main()

    M_D, total_estimates = build_model_set(wait_results, deadline)
    skip_status = None

    print(f"[Request {transaction_id}] Selected Models: {M_D}")

    if not M_D:
        # Pick the fastest model by latency
        fastest_model = min(MODEL_PROFILES, key=lambda m: MODEL_PROFILES[m]["latency"])
        M_D = [fastest_model]
        skip_status = "Skipped"
        print(f"[Request {transaction_id}] No valid models found — using fastest model: {fastest_model}")

    # Step 4: Send requests
    send_req_start_time_sec = time.perf_counter()
    results = []
    for m in M_D:
        try:
            res = await send_request(stub, m, image_b64)
            results.append(res)
        except Exception as e:
            logging.warning(f"[Request {transaction_id}] Model {m} failed: {e}")
            # Optionally, append a placeholder result indicating failure
            results.append({
                "model": m,
                "label": -1,
                "probability": 0.0,
                "latency": -1,
                "success": False,
                "container_state": "UNKNOWN"
            })
    # tasks = [asyncio.create_task(send_request(stub, m, image_b64)) for m in M_D]
    # results = await asyncio.gather(*tasks)

    results = [res for res in results if res.get("success", False)]
    if not results:
        print(f"[Request {transaction_id}] No successful model responses.")
        return {  "selected_models": -1, "label": -1, "accuracy": 0.0, "combiner_policy": "FAILED", "e2e_time_ms": -1, "success": False, "wait_times": total_estimates}

    prev_snapshot = dict(MODEL_WEIGHTS)

    # Step 5: Choose best result
    # best = max(results, key=lambda x: x["accuracy"])
    final_result = combine_outputs(
        results,
        policy="weighted_majority",
        historical_acc=OMEGA,
        model_weights=MODEL_WEIGHTS,
        ground_truth=selected_folder,
        gamma=GAMMA,
        update_weights=True,
        label_matcher=wnid_matches_text_label,
        rho=RHO
    )

    save_model_weights_atomic(WEIGHTS_STATE_PATH, MODEL_WEIGHTS)

    # Step 6: Feedback updates
    for res in results:
        OMEGA[res["model"]] = (1 - ETA) * OMEGA[res["model"]] + ETA * res["probability"]
        active_models.add(res["model"])

    # Print all model responses
    # Print all model responses and build per-model summary
    per_model = {}
    for res in results:
        model = res.get("model")
        prob = float(res.get("probability", 0.0))
        label = res.get("label")
        lat = float(res.get("latency", -1))
        state = res.get("container_state", "UNKNOWN")
        succ = bool(res.get("success", False))

        print(f"Model: {model} | Accuracy: {prob:.4f} | label: {label} | Latency: {lat:.3f}s | State: {state}")

        per_model[model] = {
            "label": label,
            "probability": prob,
            "latency_s": lat,
            "success": succ,
            "state": state,
        }

    print("--------------------------------------------------------------------------")
    print(f"Final Result {final_result}")
    print("--------------------------------------------------------------------------")

    end_time = time.perf_counter()
    print(f"Request {transaction_id} total time: {(end_time - start_time) * 1000:.2f} ms\n")

    return {
        "selected_models": M_D,
        "label": final_result["label"],
        "accuracy": final_result["accuracy"],
        "combiner_policy": final_result["combiner_policy"],
        "e2e_time_ms": (end_time - start_time) * 1000,
        "success": skip_status or final_result["success"],
        "selected_folder": selected_folder,
        "wait_times": total_estimates,
        "per_model": per_model,
        "model_Size":MAX_MODEL_SIZE,
        "main_policy": POLICY,
        "alpha":ALPHA

    }



# if __name__ == "__main__":
#     asyncio.run(QoED_test())