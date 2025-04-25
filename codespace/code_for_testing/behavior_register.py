import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import uuid

channel = grpc.insecure_channel("127.0.0.1:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)
model_name="shufflenet_v2_x0_5"
request = pb2.RegisterRequest(
            function_name=model_name,
            function_version="1",
            image_name="docker.io/sunbaixi96/ckn_faas_{}-iluvatar-action-http:latest".format(model_name),
            memory=512,
            cpus=1,
            parallel_invokes=1,
            transaction_id=str(uuid.uuid4()),
            language=pb2.LanguageRuntime.PYTHON3,
            compute=1,        # or appropriate platform ID
            isolate=1,
            container_server=0
        )

response = worker.register(request)
print(response)