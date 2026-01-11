# import grpc
# import iluvatar_rpc_pb2 as pb2
# import iluvatar_rpc_pb2_grpc as pb2_grpc
# import uuid

# def get_estimated_invoke_time(server_address, fqdns, transaction_id):
#     # Establish a channel with the gRPC server
#     with grpc.insecure_channel(server_address) as channel:
#         stub = pb2_grpc.IluvatarWorkerStub(channel)
        
#         # Create the request message
#         request = pb2.EstInvokeRequest(transaction_id=transaction_id, fqdns=fqdns)
        
#         try:
#             # Make the RPC call
#             response = stub.est_invoke_time(request)
#             return response.est_time
#         except grpc.RpcError as e:
#             print(f"RPC failed: {e.code()} - {e.details()}")
#             return None

# # Example usage
# if __name__ == "__main__":
#     server_address = '127.0.0.1:8079'  # Replace with your server's address
#     # fqdns = ["shufflenet_v2_x0_5", "mobilenet_v3_small", "googlenet",
#     #                 "resnext50_32x4d", "densenet201", "resnet152"]
#     # fqdns = ["resnet152-1"]
#     fqdns = [
#     "shufflenet_v2_x0_5-1",
#     "mobilenet_v3_small-1",
#     "googlenet-1",
#     "resnext50_32x4d-1",
#     "densenet201-1",
#     "resnet152-1"
#     ]
#     transaction_id = str(uuid.uuid4())
    
#     est_times = get_estimated_invoke_time(server_address, fqdns, transaction_id)
#     if est_times is not None:
#         for fqdn, time in zip(fqdns, est_times):
#             print(f"Estimated time for {fqdn}: {time} seconds")

import grpc
import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc
import uuid
import time
from datetime import datetime

def get_estimated_invoke_time(stub, fqdn):
    request = pb2.EstInvokeRequest(
        transaction_id=str(uuid.uuid4()),
        fqdns=[fqdn]
    )

    try:
        response = stub.est_invoke_time(request)
        return response.est_time[0] if response.est_time else None
    except grpc.RpcError as e:
        print(f"[{datetime.now()}] RPC failed: {e.code()} - {e.details()}")
        return None

if __name__ == "__main__":
    server_address = '149.165.150.17:8079'
    # "mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"
    fqdn = "mobilenet_v3_small-1"

    with grpc.insecure_channel(server_address) as channel:
        stub = pb2_grpc.IluvatarWorkerStub(channel)

        last_est_time = None
        # print()
        while True:
            start_time = time.time()
            est_time = get_estimated_invoke_time(stub, fqdn)

            if est_time is not None:
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                service_time = time.time() - start_time
                # print(f"[{now}] Estimated wait time for {fqdn}: {est_time:.4f} sec | Service time: {service_time:.4f} sec")
                # print(f"{est_time:.4f}; {service_time:.4f}")
                print(f"{est_time:.4f}",flush=True)
            else:
                print(f"[{datetime.now()}] Failed to get estimate.")

            time.sleep(0.1)
