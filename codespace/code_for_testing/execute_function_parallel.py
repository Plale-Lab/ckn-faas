import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import os
import uuid
import base64
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

def read_image_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()

def send_request(worker, model_name, image_b64):
    request = pb2.InvokeRequest(
        function_name=model_name,
        function_version="1",
        json_args=json.dumps({"model_name": model_name, "image_data": image_b64}),
        transaction_id=str(uuid.uuid4()),
    )

    try:
        response = worker.invoke(request)
        return {
            "model": model_name,
            "success": response.success,
            "result": response.json_result,
            "duration_us": response.duration_us,
            "compute": response.compute,
            "container_state": pb2.ContainerState.Name(response.container_state),
        }
    except Exception as e:
        return {"model": model_name, "error": str(e)}

# Config
model_list_total = ["shufflenet_v2_x0_5", "mobilenet_v3_small", "googlenet",
                    "resnext50_32x4d", "densenet201", "resnet152"]
num_requests = 1  # Total request groups (each group sends 2 models)

# Load image and connect
image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg")
image_b64 = base64.b64encode(image_bytes).decode("utf-8")

channel = grpc.insecure_channel("127.0.0.1:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)

# Prepare and send requests concurrently
start = time.perf_counter()
futures = []
with ThreadPoolExecutor(max_workers=8) as executor:
    for _ in range(num_requests):
        model_list = random.sample(model_list_total, 2)
        for model_name in model_list:
            futures.append(executor.submit(send_request, worker, model_name, image_b64))

# Collect results
for future in as_completed(futures):
    res = future.result()
    print("✅ Invocation response for", res.get("model"))
    if "error" in res:
        print("❌ Error:", res["error"])
    else:
        print("Success:", res["success"])
        print("Result:", res["result"])
        print("Duration (μs):", res["duration_us"])
        print("Compute:", res["compute"])
        print("Container state:", res["container_state"])
        print()

end = time.perf_counter()
print("⏱️ Total time for {} requests: {:.2f} ms".format(len(futures), (end - start) * 1000))