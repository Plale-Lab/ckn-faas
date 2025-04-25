import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import os
import uuid
import base64
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
image_dir = "/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3"
model_list = ["shufflenet_v2_x0_5", "mobilenet_v3_small", "googlenet",
              "resnext50_32x4d", "densenet201", "resnet152"]

# Output: {model_name: [prob1, prob2, ...]}
probability_results = defaultdict(list)

# Connect to gRPC
channel = grpc.insecure_channel("127.0.0.1:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)

def read_image_as_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def send_request(model_name, image_path):
    try:
        image_b64 = read_image_as_base64(image_path)
        request = pb2.InvokeRequest(
            function_name=model_name,
            function_version="1",
            json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
            transaction_id=str(uuid.uuid4()),
        )
        response = worker.invoke(request)
        result_json = json.loads(response.json_result)
        prob = result_json["body"]["Probability"]
        return (model_name, prob, image_path, None)
    except Exception as e:
        return (model_name, None, image_path, str(e))

# Gather all image/model tasks
tasks = []
for filename in sorted(os.listdir(image_dir)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue
    image_path = os.path.join(image_dir, filename)
    for model_name in model_list:
        tasks.append((model_name, image_path))

# Run in parallel
start = time.perf_counter()
with ThreadPoolExecutor(max_workers=12) as executor:
    futures = [executor.submit(send_request, m, i) for m, i in tasks]

    for future in as_completed(futures):
        model_name, prob, image_path, error = future.result()
        if error:
            print(f"❌ {model_name} on {os.path.basename(image_path)}: {error}")
        else:
            probability_results[model_name].append(prob)
            print(f"✅ {model_name} on {os.path.basename(image_path)}: {prob:.4f}")

end = time.perf_counter()
print(f"\n⏱️ Total time: {(end - start):.2f} seconds")

# Save results
with open("model_probabilities.json", "w") as f:
    json.dump(probability_results, f, indent=2)
print(probability_results)
print("✅ Results saved to model_probabilities.json")