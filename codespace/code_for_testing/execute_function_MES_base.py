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


model_list_total = ["mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"]
async def QoED_test():
    # Config
    # model_list = random.sample(model_list_total, 3) # Randomly select 3 models from the total list
    # model_list = ["resnet101"] # accuracy only
    model_list = model_list_total # full ensemble
    # model_list = ["mobilenet_v3_small"] # Resource only

    num_requests = 256
    img_num = random.randint(0, 999)
    category_choice = random.choice(["cat", "dog"])
    image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/{}.{}.jpg".format(category_choice,img_num))
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    channel = grpc.aio.insecure_channel("149.165.150.17:8079")
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    tasks = []
    start = time.perf_counter()
    for _ in range(num_requests):
        # Optional jitter between requests
        for model_name in model_list:
            #generate random time to sleep between requests
            jitter = random.uniform(0, 0.005) # 0.1 seconds max jitter
            await asyncio.sleep(jitter)
            tasks.append(send_request(stub, model_name, image_b64))

    # Run all requests concurrently
    results = await asyncio.gather(*tasks)
    # 2004598
    # Output results
    acc_list = []
    for res in results:
        print("✅ Invocation response for", res.get("model"))
        if "error" in res:
            print("❌ Error:", res["error"])
        else:
            acc_list.append(json.loads(res["result"])["body"]["Probability"])
            print("Success:", res["success"])
            result_json = json.loads(res["result"])
            print("Result:", result_json["body"]["Probability"])
            print("Duration (μs):", res["duration_us"])
            print("Compute:", res["compute"])
            print("Container state:", res["container_state"])
            print()

    end = time.perf_counter()
    print("Total time for {} requests: {:.2f} ms, average accuracy: {:.2f}".format(len(results), (end - start) * 1000, sum(acc_list) / len(acc_list)))

if __name__ == "__main__":
    asyncio.run(QoED_test())