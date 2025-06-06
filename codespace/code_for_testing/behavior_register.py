import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import uuid
import time

channel = grpc.insecure_channel("149.165.150.17:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)
# "mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"
model_name="mobilenet_v3_small"
for cpu_core in [1]:
    start = time.perf_counter()
    if model_name == "resnet101" or model_name == "vit_b_16":
        request = pb2.RegisterRequest(
            function_name=model_name,
            function_version=str(cpu_core),
            image_name="docker.io/sunbaixi96/ckn_faas_{}-iluvatar-action-http:latest".format(model_name),
            memory=1024,
            cpus=cpu_core,
            parallel_invokes=1,
            transaction_id=str(uuid.uuid4()),
            language=pb2.LanguageRuntime.PYTHON3,
            compute=1,        # or appropriate platform ID
            isolate=1,
            container_server=0
        )
    else:
        request = pb2.RegisterRequest(
                    function_name=model_name,
                    function_version=str(cpu_core),
                    image_name="docker.io/sunbaixi96/ckn_faas_{}-iluvatar-action-http:latest".format(model_name),
                    memory=512,
                    cpus=cpu_core,
                    parallel_invokes=1,
                    transaction_id=str(uuid.uuid4()),
                    language=pb2.LanguageRuntime.PYTHON3,
                    compute=1,        # or appropriate platform ID
                    isolate=1,
                    container_server=0
                )
    end = time.perf_counter()
    response = worker.register(request)
    print(response)
    print(end-start)