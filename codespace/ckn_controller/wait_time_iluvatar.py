# # wait_time_client.py
# import asyncio
# import uuid
#
# import grpc
# import json
# import sys
# from datetime import datetime
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
#
# M_total = ["mobilenet_v3_small", "resnet18", "resnet34", "resnet50", "resnet101", "vit_b_16"]
#
# def get_estimated_wait(stub, model_name):
#     fqdn = f"{model_name}-1"
#     request = pb2.EstInvokeRequest(transaction_id=str(uuid.uuid4()), fqdns=[fqdn])
#     try:
#         response = stub.est_invoke_time(request)
#         return response.est_time[0] if response.est_time else float('inf')
#     except grpc.RpcError as e:
#         print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
#         return float('inf')
#
# def main():
#     channel = grpc.insecure_channel("149.165.175.242:8079")
#     stub = pb2_grpc.IluvatarWorkerStub(channel)
#
#     wait_results = {}
#     for model in M_total:
#         wait_time = get_estimated_wait(stub, model)
#         wait_results[model] = wait_time
#
#     print(json.dumps(wait_results, indent=2))
#
#     return wait_results
#
# if __name__ == "__main__":
#     main()






# 3 Aug 2026

# import uuid
# import grpc
# import json
# from datetime import datetime
# from typing import Dict, List, Optional
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
#     try:
#         response = stub.est_invoke_time(request)
#         return float(response.est_time[0]) if response.est_time else float("inf")
#     except grpc.RpcError as e:
#         print(f"[{datetime.now()}] Failed to estimate for {model_name}: {e.details()}")
#         return float("inf")
#     except Exception as e:
#         print(f"[{datetime.now()}] Unexpected error for {model_name}: {e}")
#         return float("inf")
#
#
# def main(
#     server_address: Optional[str] = None,
#     model_list: Optional[List[str]] = None,
# ) -> Dict[str, float]:
#     address = server_address or SERVER_ADDRESS
#     models = model_list if model_list is not None else list(M_TOTAL)
#
#     channel = grpc.insecure_channel(address)
#     stub = pb2_grpc.IluvatarWorkerStub(channel)
#
#     wait_results: Dict[str, float] = {}
#     for model in models:
#         wait_results[model] = get_estimated_wait(stub, model)
#
#     print(json.dumps({
#         "server_address": address,
#         "wait_results": wait_results
#     }, indent=2))
#
#     return wait_results
#
#
# if __name__ == "__main__":
#     main()


import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    M_TOTAL,
)


# ============================================================
# Get estimated invocation time for one model
# ============================================================

def get_estimated_wait(
    stub,
    model_name: str,
    server_address: Optional[str] = None,
) -> float:
    """
    Query one Iluvatar instance for the estimated invocation time
    of one model.

    Args:
        stub:
            Iluvatar gRPC stub connected to one server.

        model_name:
            Model/function name, such as "resnet18".

        server_address:
            Used only for logging.

    Returns:
        Estimated invocation time in seconds.

        Returns float("inf") when:
        - the RPC fails,
        - no estimate is returned,
        - or another error occurs.
    """

    fqdn = f"{model_name}-1"

    request = pb2.EstInvokeRequest(
        transaction_id=str(uuid.uuid4()),
        fqdns=[fqdn],
    )

    try:
        response = stub.est_invoke_time(request)

        if not response.est_time:
            print(
                f"[{datetime.now()}] "
                f"No estimate returned "
                f"server={server_address} "
                f"model={model_name}"
            )

            return float("inf")

        estimated_time = float(response.est_time[0])

        print(
            f"[{datetime.now()}] "
            f"Estimate received "
            f"server={server_address} "
            f"model={model_name} "
            f"estimated_time_s={estimated_time:.6f}"
        )

        return estimated_time

    except grpc.RpcError as error:
        try:
            details = error.details()
        except Exception:
            details = str(error)

        print(
            f"[{datetime.now()}] "
            f"Failed to estimate "
            f"server={server_address} "
            f"model={model_name} "
            f"error={details}"
        )

        return float("inf")

    except Exception as error:
        print(
            f"[{datetime.now()}] "
            f"Unexpected estimation error "
            f"server={server_address} "
            f"model={model_name} "
            f"error={error}"
        )

        return float("inf")


# ============================================================
# Query one Iluvatar instance
# ============================================================

def main(
    server_address: Optional[str] = None,
    model_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Query one Iluvatar instance for a list of models.

    This function does not select the best edge or cloud node.

    ckn_main.py calls this function once for every enabled node,
    compares the returned estimates, and selects the best node.

    Args:
        server_address:
            Address of one Iluvatar instance.

            Example:
                "149.165.175.242:8081"

        model_list:
            Models deployed on that Iluvatar instance.

            Example:
                ["resnet18", "resnet34"]

    Returns:
        Dictionary mapping each model to its estimated invocation
        time on this specific server.

        Example:
            {
                "resnet18": 0.025,
                "resnet34": 0.041
            }
    """

    address = server_address or SERVER_ADDRESS

    models = (
        list(model_list)
        if model_list is not None
        else list(M_TOTAL)
    )

    if not address:
        raise ValueError(
            "server_address is required."
        )

    if not models:
        print(
            f"[{datetime.now()}] "
            f"No models provided "
            f"server={address}"
        )

        return {}

    channel = grpc.insecure_channel(address)
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    wait_results: Dict[str, float] = {}

    try:
        for model_name in models:
            wait_results[model_name] = get_estimated_wait(
                stub=stub,
                model_name=model_name,
                server_address=address,
            )

    finally:
        try:
            channel.close()
        except Exception:
            pass

    output = {
        "server_address": address,
        "models": models,
        "wait_results": wait_results,
    }

    print(
        json.dumps(
            output,
            indent=2,
            default=str,
        )
    )

    return wait_results


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":
    main()

