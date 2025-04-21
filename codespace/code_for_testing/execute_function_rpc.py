import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import json
import os
import uuid
import base64
import time
import random

def read_image_as_bytes(path):
    with open(path, "rb") as f:
        return f.read()

# Connect to the worker
channel = grpc.insecure_channel("127.0.0.1:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)

# # Get your username (same as `whoami`)
username = os.getlogin()  # or os.environ["USER"]

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

## Sequentially invoke the function
image_bytes = read_image_as_bytes("/home/exouser/ckn-faas/codespace/ckn/jetsons/device/data/images/d2iedgeai3/cat.12.jpg")
image_b64 = base64.b64encode(image_bytes).decode("utf-8")
model_list_total = ["shufflenet_v2_x0_5","mobilenet_v3_small","googlenet","resnext50_32x4d","densenet201","resnet152"]
for model_name in model_list_total:
    request = pb2.InvokeRequest(
            function_name=model_name,
            function_version="1",
            json_args=json.dumps({"model_name": model_name,'image_data':image_b64}),
            transaction_id=str(uuid.uuid4()),  # unique transaction ID
        )  

    response = worker.invoke(request)

    # Print the result
    print("✅ Invocation response:")
    print("Success:", response.success)
    print("Result:", response.json_result)
    print("Duration (μs):", response.duration_us)
    print("Compute:", response.compute)
    print("Container state:", pb2.ContainerState.Name(response.container_state))
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