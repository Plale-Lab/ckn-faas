# import uuid
# import grpc
# import json
# from datetime import datetime
# from typing import Dict, List, Optional, Any
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import SERVER_ADDRESS, M_TOTAL
#
#
# def get_estimated_wait(stub, model_name: str) -> float:
#     fqdn = f"{model_name}-1"
#     request = pb2.EstInvokeRequest(
#         transaction_id=str(uuid.uuid4()),
#         fqdns=[fqdn]
#     )
#
#     try:
#         response = stub.est_invoke_time(request)
#         return float(response.est_time[0]) if response.est_time else float("inf")
#     except grpc.RpcError as e:
#         print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
#         return float("inf")
#
#
# def get_worker_status(stub) -> Dict[str, Any]:
#     request = pb2.StatusRequest(
#         transaction_id=str(uuid.uuid4())
#     )
#
#     try:
#         # If this gives an error, check your proto service method name.
#         # It may be stub.status(request), stub.get_status(request), or stub.Status(request)
#         response = stub.status(request)
#
#         return {
#             "success": response.success,
#             "queue_len": response.queue_len,
#             "used_mem": response.used_mem,
#             "total_mem": response.total_mem,
#             "cpu_us": response.cpu_us,
#             "cpu_sy": response.cpu_sy,
#             "cpu_id": response.cpu_id,
#             "cpu_wa": response.cpu_wa,
#             "load_avg_1minute": response.load_avg_1minute,
#             "num_system_cores": response.num_system_cores,
#             "num_running_funcs": response.num_running_funcs,
#         }
#
#     except grpc.RpcError as e:
#         print(f"[{datetime.now()}] Failed to get worker status: {e.details()}")
#         return {}
#
#     except Exception as e:
#         print(f"[{datetime.now()}] Unexpected status error: {e}")
#         return {}
#
#
# def main(
#     server_address: Optional[str] = None,
#     model_list: Optional[List[str]] = None,
# ) -> Dict[str, Any]:
#
#     address = server_address or SERVER_ADDRESS
#     models = model_list if model_list is not None else list(M_TOTAL)
#
#     channel = grpc.insecure_channel(address)
#     stub = pb2_grpc.IluvatarWorkerStub(channel)
#
#     status_result = get_worker_status(stub)
#
#     wait_results: Dict[str, float] = {}
#     for model in models:
#         wait_results[model] = get_estimated_wait(stub, model)
#
#     result = {
#         "server_address": address,
#         "worker_status": status_result,
#         "wait_results": wait_results,
#     }
#
#     print(json.dumps(result, indent=2))
#     return result
#
#
# if __name__ == "__main__":
#     main()



import uuid
import grpc
import json
from datetime import datetime
from typing import Dict, List, Optional, Any

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    M_TOTAL,
    USE_TWO_ILUVATAR_INSTANCES,
    SMALL_MODEL_SERVER_ADDRESS,
    LARGE_MODEL_SERVER_ADDRESS,
    SMALL_INSTANCE_MODELS,
    LARGE_INSTANCE_MODELS,
)


def get_estimated_wait(stub, model_name: str) -> float:
    fqdn = f"{model_name}-1"

    request = pb2.EstInvokeRequest(
        transaction_id=str(uuid.uuid4()),
        fqdns=[fqdn],
    )

    try:
        response = stub.est_invoke_time(request)
        return float(response.est_time[0]) if response.est_time else float("inf")

    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
        return float("inf")

    except Exception as e:
        print(f"[{datetime.now()}] Unexpected wait estimate error for {model_name}: {e}")
        return float("inf")


def get_worker_status(stub) -> Dict[str, Any]:
    request = pb2.StatusRequest(
        transaction_id=str(uuid.uuid4())
    )

    try:
        response = stub.status(request)

        return {
            "success": response.success,
            "queue_len": response.queue_len,
            "used_mem": response.used_mem,
            "total_mem": response.total_mem,
            "cpu_us": response.cpu_us,
            "cpu_sy": response.cpu_sy,
            "cpu_id": response.cpu_id,
            "cpu_wa": response.cpu_wa,
            "load_avg_1minute": response.load_avg_1minute,
            "num_system_cores": response.num_system_cores,
            "num_running_funcs": response.num_running_funcs,
        }

    except grpc.RpcError as e:
        print(f"[{datetime.now()}] Failed to get worker status: {e.details()}")
        return {}

    except Exception as e:
        print(f"[{datetime.now()}] Unexpected status error: {e}")
        return {}


def query_instance(
    server_address: str,
    model_list: List[str],
    instance_name: str,
) -> Dict[str, Any]:

    channel = grpc.insecure_channel(server_address)
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    status_result = get_worker_status(stub)

    wait_results: Dict[str, float] = {}

    for model in model_list:
        wait_results[model] = get_estimated_wait(stub, model)

    return {
        # "instance_name": instance_name,
        "server_address": server_address,
        # "models": model_list,
        "worker_status": status_result,
        "wait_results": wait_results,
    }


def main(
    model_list: Optional[List[str]] = None,
) -> Dict[str, Any]:

    # -----------------------------
    # Dual-instance mode
    # -----------------------------
    if USE_TWO_ILUVATAR_INSTANCES:
        requested_models = model_list if model_list is not None else list(M_TOTAL)

        small_models = [
            m for m in requested_models
            if m in SMALL_INSTANCE_MODELS
        ]

        large_models = [
            m for m in requested_models
            if m in LARGE_INSTANCE_MODELS
        ]

        unknown_models = [
            m for m in requested_models
            if m not in SMALL_INSTANCE_MODELS and m not in LARGE_INSTANCE_MODELS
        ]

        result = {
            "mode": "dual_instance",
            "small_instance": query_instance(
                server_address=SMALL_MODEL_SERVER_ADDRESS,
                model_list=small_models,
                instance_name="small_instance",
            ),
            "large_instance": query_instance(
                server_address=LARGE_MODEL_SERVER_ADDRESS,
                model_list=large_models,
                instance_name="large_instance",
            ),
            # "unknown_models": unknown_models,
        }

        print(json.dumps(result, indent=2))
        return result

    # -----------------------------
    # Single-instance mode
    # -----------------------------
    requested_models = model_list if model_list is not None else list(M_TOTAL)

    result = {
        "mode": "single_instance",
        "single_instance": query_instance(
            server_address=SERVER_ADDRESS,
            model_list=requested_models,
            instance_name="single_instance",
        ),
    }

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    main()