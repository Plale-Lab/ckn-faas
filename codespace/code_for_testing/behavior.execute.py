# import grpc
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
# import json
# import os
# import uuid
# import base64
# import time
# import random

# def read_image_as_bytes(path):
#     with open(path, "rb") as f:
#         return f.read()

# # Connect to the worker
# channel = grpc.insecure_channel("127.0.0.1:8079")
# worker = pb2_grpc.IluvatarWorkerStub(channel)

# # # Get your username (same as `whoami`)
# username = os.getlogin()  # or os.environ["USER"]

# Create the InvokeRequest
# request = pb2.InvokeRequest(
#     function_name="Baixi",
#     function_version="1",
#     json_args=json.dumps({"name": username}),
#     transaction_id=str(uuid.uuid4()),  # unique transaction ID
# )
# model_name="mobilenet_v3_small"
# image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg")
# image_b64 = base64.b64encode(image_bytes).decode("utf-8")
# request = pb2.InvokeRequest(
#             function_name=model_name,
#             function_version="1",
#             json_args=json.dumps({"model_name": model_name,'image_data':image_b64}),
#             transaction_id=str(uuid.uuid4()),  # unique transaction ID
#         )

'''
## Sequentially invoke the function
image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg")
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
# model_list_total = ["shufflenet_v2_x0_5","mobilenet_v3_small","googlenet","resnext50_32x4d","densenet201","resnet152"]
model_list_total = ["resnet152"]
for model_name in model_list_total:
    for cpu_core in [1,1,2,4,8]:
        request = pb2.InvokeRequest(
                function_name=model_name,
                function_version=str(cpu_core),
                json_args=json.dumps({"model_name": model_name,'image_data':image_b64}),
                transaction_id=str(uuid.uuid4()),  # unique transaction ID
            )  

        response = worker.invoke(request)

        # Print the result
        print("✅ Invocation response for {} cores:".format(cpu_core))
        print("Success:", response.success)
        print("Result:", response.json_result)
        print("Duration (μs):", response.duration_us)
        print("Compute:", response.compute)
        print("Container state:", pb2.ContainerState.Name(response.container_state))
        print("Response Time: {} ms".format(response.duration_us / 1000))
'''
'''
model_list_total = ["shufflenet_v2_x0_5","mobilenet_v3_small","googlenet","resnext50_32x4d","densenet201","resnet152"]
num_requests = 1
start = time.perf_counter()
for i in range(num_requests):
    model_list = random.sample(model_list_total, 2)
    for model_name in model_list:
        request = pb2.InvokeRequest(
            function_name=model_name,
            function_version="1",
            json_args=json.dumps({"model_name": model_name,'image_data':image_b64}),
            transaction_id=str(uuid.uuid4()),  # unique transaction ID
        )

        # request = pb2.InvokeRequest(
        #     function_name="cnn",
        #     function_version="1",
        #     transaction_id=str(uuid.uuid4()),  # unique transaction ID
        # )

        # request = pb2.InvokeRequest(
        #     function_name="Baixi",
        #     function_version="1",
        #     json_args=json.dumps({"name": username}),
        #     transaction_id=str(uuid.uuid4()),  # unique transaction ID
    # )


    # Invoke the function
    response = worker.invoke(request)

    # Print the result
    print("✅ Invocation response:")
    print("Success:", response.success)
    print("Result:", response.json_result)
    print("Duration (μs):", response.duration_us)
    print("Compute:", response.compute)
    print("Container state:", pb2.ContainerState.Name(response.container_state))
end = time.perf_counter()
print("Total time for {} requests: {} ms".format(num_requests,end-start))
    print(response)
'''


import grpc.aio
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
        transaction_id=str('333'),
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

async def QoED_test():
    # Config
    model_name = "shufflenet_v2_x0_5"
    num_requests = 128
    img_num = random.randint(0, 999)
    category_choice = random.choice(["cat", "dog"])
    image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/{}.{}.jpg".format(category_choice,img_num))
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")

    channel = grpc.aio.insecure_channel("127.0.0.1:8079")
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    tasks = []
    start = time.perf_counter()
    for _ in range(num_requests):
        # Optional jitter between requests
        # await asyncio.sleep(0.5)
        tasks.append(send_request(stub, model_name, image_b64))

    # Run all requests concurrently
    results = await asyncio.gather(*tasks)

    # Output results
    for res in results:
        print("✅ Invocation response for", res.get("model"))
        if "error" in res:
            print("❌ Error:", res["error"])
        else:
            print("Success:", res["success"])
            result_json = json.loads(res["result"])
            print("Result:", result_json["body"]["Probability"])
            print("Duration (μs):", res["duration_us"])
            print("Compute:", res["compute"])
            print("Container state:", res["container_state"])
            print()

    end = time.perf_counter()
    print("Total time for {} requests: {:.2f} ms".format(len(results), (end - start) * 1000))

if __name__ == "__main__":
    asyncio.run(QoED_test())

