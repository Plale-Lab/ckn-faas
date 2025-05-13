import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import uuid

channel = grpc.insecure_channel("127.0.0.1:8079")
worker = pb2_grpc.IluvatarWorkerStub(channel)
request = EstInvokeRequest(transaction_id=transaction_id)
response = worker.register(request)
print(response)