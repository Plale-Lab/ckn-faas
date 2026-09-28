import json
import os
import base64
import time
import multiprocessing
import requests

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed


# -----------------------------
# Configuration
# -----------------------------
OPENFAAS_GATEWAY = "http://149.165.152.91:31112"

image_dir = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/ckn_data/images/tem"

model_list = [
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


# -----------------------------
# Model name → OpenFaaS function name
# -----------------------------
MODEL_TO_FUNCTION = {
    "mobilenet_v3_small": "mobilenet-v3-small",
    "mobilenet_v2": "mobilenet-v2",
    "shufflenet_v2_x1_0": "shufflenet-v2-x1-0",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "efficientnet_b3": "efficientnet-b3",
    "convnext_tiny": "convnext-tiny",
    "vit_b_16": "vit-b-16",
}


def get_openfaas_url_for_model(model_name: str) -> str:
    function_name = MODEL_TO_FUNCTION[model_name]
    return f"{OPENFAAS_GATEWAY}/function/{function_name}"


def read_image_as_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_openfaas_response(response: requests.Response):
    result_json = response.json()

    if "body" not in result_json:
        raise ValueError(f"Unexpected response: {result_json}")

    body = result_json["body"]

    if "Probability" not in body:
        raise ValueError(f"Inference failed: {result_json}")

    prob = float(body["Probability"])
    label = body.get("Prediction Class", "UNKNOWN")

    return prob, label, result_json


def send_request(model_name: str, image_path: str):
    try:
        image_b64 = read_image_as_base64(image_path)

        payload = {
            "model_name": model_name,
            "image_data": image_b64,
        }

        url = get_openfaas_url_for_model(model_name)

        start = time.perf_counter()
        response = requests.post(url, json=payload, timeout=120)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000.0

        if response.status_code != 200:
            raise ValueError(
                f"HTTP {response.status_code}: {response.text[:500]}"
            )

        prob, label, result_json = parse_openfaas_response(response)

        return {
            "model_name": model_name,
            "image_path": image_path,
            "probability": prob,
            "label": label,
            "latency_ms": latency_ms,
            "server_address": url,
            "raw_response": result_json,
            "error": None,
        }

    except Exception as e:
        return {
            "model_name": model_name,
            "image_path": image_path,
            "probability": None,
            "label": None,
            "latency_ms": None,
            "server_address": get_openfaas_url_for_model(model_name),
            "raw_response": None,
            "error": str(e),
        }


def main():
    probability_results = defaultdict(list)
    latency_results = defaultdict(list)
    route_results = defaultdict(list)

    tasks = []

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(image_dir, filename)

        for model_name in model_list:
            tasks.append((model_name, image_path))

    print("=== OpenFaaS Routing Summary ===")
    for model_name in model_list:
        print(f"{model_name:<20} -> {get_openfaas_url_for_model(model_name)}")

    start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(send_request, m, i) for m, i in tasks]

        for future in as_completed(futures):
            result = future.result()

            model_name = result["model_name"]
            image_path = result["image_path"]
            prob = result["probability"]
            label = result["label"]
            latency = result["latency_ms"]
            address = result["server_address"]
            error = result["error"]

            if error:
                print(f"FAILED {model_name} on {os.path.basename(image_path)}: {error}")
            else:
                probability_results[model_name].append(prob)
                latency_results[model_name].append(latency)
                route_results[model_name].append(address)

                print(
                    f"OK {model_name} on {os.path.basename(image_path)}: "
                    f"label={label}, prob={prob:.4f}, "
                    f"latency={latency:.2f} ms, endpoint={address}"
                )

    end = time.perf_counter()

    print(f"\nTotal time: {(end - start):.2f} seconds")

    print("\nAverage Latency per Model:")
    for model, latencies in latency_results.items():
        avg_latency = sum(latencies) / len(latencies)
        print(f"{model:<20}: {avg_latency:.2f} ms")

    print("\nRoute per Model:")
    for model, addresses in route_results.items():
        unique_addresses = sorted(set(addresses))
        print(f"{model:<20}: {unique_addresses}")

    with open("openfaas_model_latencies.json", "w") as f:
        json.dump(latency_results, f, indent=2)

    with open("openfaas_model_probabilities.json", "w") as f:
        json.dump(probability_results, f, indent=2)

    with open("openfaas_model_routes.json", "w") as f:
        json.dump(route_results, f, indent=2)

    print("Results saved to openfaas_model_latencies.json, openfaas_model_probabilities.json, openfaas_model_routes.json")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()