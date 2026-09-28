import json
import math
import uuid
from datetime import datetime
from typing import Any, Dict, List, Tuple

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    USE_MULTI_ILUVATAR_INSTANCES,
    ILUVATAR_NODES,
)


# ============================================================
# Type aliases
# ============================================================

NodeWaitResults = Dict[str, Dict[str, float]]
BestModelWaits = Dict[str, float]
BestModelRoutes = Dict[str, Dict[str, Any]]


# ============================================================
# Validate configuration
# ============================================================

def validate_config() -> None:
    if not USE_MULTI_ILUVATAR_INSTANCES:
        raise ValueError(
            "USE_MULTI_ILUVATAR_INSTANCES must be True."
        )

    if not isinstance(ILUVATAR_NODES, dict):
        raise ValueError(
            "ILUVATAR_NODES must be a dictionary."
        )

    if not ILUVATAR_NODES:
        raise ValueError(
            "ILUVATAR_NODES is empty."
        )

    for node_id, node_config in ILUVATAR_NODES.items():
        if not isinstance(node_config, dict):
            raise ValueError(
                f"Node '{node_id}' configuration must be a dictionary."
            )

        if not node_config.get("address"):
            raise ValueError(
                f"Node '{node_id}' is missing 'address'."
            )

        if not isinstance(node_config.get("models", []), list):
            raise ValueError(
                f"Node '{node_id}' field 'models' must be a list."
            )

        node_type = str(
            node_config.get("type", "")
        ).lower()

        if node_type not in ("edge", "cloud"):
            raise ValueError(
                f"Node '{node_id}' has invalid type '{node_type}'."
            )


# ============================================================
# Query one model on one Iluvatar instance
# ============================================================

def get_estimated_wait(
    stub,
    node_id: str,
    server_address: str,
    model_name: str,
    rpc_timeout_sec: float = 10.0,
) -> float:
    """
    Return Iluvatar's estimated invocation time in seconds.

    Returns infinity if the RPC fails or no estimate is returned.
    """

    request = pb2.EstInvokeRequest(
        transaction_id=str(uuid.uuid4()),
        fqdns=[f"{model_name}-1"],
    )

    try:
        response = stub.est_invoke_time(
            request,
            timeout=rpc_timeout_sec,
        )

        if not response.est_time:
            print(
                f"[WAIT_EMPTY] "
                f"node={node_id} "
                f"server={server_address} "
                f"model={model_name}"
            )
            return float("inf")

        estimated_wait_s = float(
            response.est_time[0]
        )

        print(
            f"[WAIT_OK] "
            f"node={node_id} "
            f"server={server_address} "
            f"model={model_name} "
            f"wait_s={estimated_wait_s:.6f} "
            f"wait_ms={estimated_wait_s * 1000.0:.2f}"
        )

        return estimated_wait_s

    except grpc.RpcError as error:
        try:
            details = error.details()
        except Exception:
            details = str(error)

        print(
            f"[WAIT_FAIL] "
            f"node={node_id} "
            f"server={server_address} "
            f"model={model_name} "
            f"error={details}"
        )

        return float("inf")

    except Exception as error:
        print(
            f"[WAIT_FAIL] "
            f"node={node_id} "
            f"server={server_address} "
            f"model={model_name} "
            f"error={error}"
        )

        return float("inf")


# ============================================================
# Query every configured model on one node
# ============================================================

def get_node_wait_times(
    node_id: str,
    node_config: dict,
) -> Dict[str, float]:
    address = str(
        node_config["address"]
    )

    models = list(
        node_config.get("models", [])
    )

    wait_results: Dict[str, float] = {}

    channel = grpc.insecure_channel(address)
    stub = pb2_grpc.IluvatarWorkerStub(channel)

    print(
        f"\n[NODE_QUERY_START] "
        f"node={node_id} "
        f"type={node_config.get('type')} "
        f"server={address} "
        f"models={models}"
    )

    try:
        for model_name in models:
            wait_results[model_name] = get_estimated_wait(
                stub=stub,
                node_id=node_id,
                server_address=address,
                model_name=model_name,
            )

    finally:
        try:
            channel.close()
        except Exception:
            pass

    return wait_results


# ============================================================
# Query all enabled nodes
# ============================================================

def get_all_node_wait_times() -> NodeWaitResults:
    """
    Return wait times separately for every node.

    Example:

    {
        "edge_1": {
            "resnet18": 0.040,
            "resnet34": 0.071
        },
        "edge_2": {
            "resnet18": 0.038,
            "resnet34": 0.095
        },
        "cloud_1": {
            "convnext_tiny": 0.090
        }
    }
    """

    validate_config()

    all_wait_results: NodeWaitResults = {}

    for node_id, node_config in ILUVATAR_NODES.items():
        if not bool(
            node_config.get("enabled", True)
        ):
            print(
                f"[NODE_SKIP] "
                f"node={node_id} "
                f"reason=disabled"
            )
            continue

        all_wait_results[node_id] = (
            get_node_wait_times(
                node_id=node_id,
                node_config=node_config,
            )
        )

    return all_wait_results


# ============================================================
# Find the best node for every model
# ============================================================

def build_best_model_routes(
    node_wait_results: NodeWaitResults,
) -> Tuple[BestModelWaits, BestModelRoutes]:
    """
    Compare all nodes supporting each model and select the node
    with the minimum finite estimated wait time.
    """

    model_candidates: Dict[
        str,
        List[Dict[str, Any]],
    ] = {}

    for node_id, model_waits in node_wait_results.items():
        node_config = ILUVATAR_NODES[node_id]

        for model_name, wait_s in model_waits.items():
            model_candidates.setdefault(
                model_name,
                [],
            ).append(
                {
                    "model": model_name,
                    "node_id": node_id,
                    "node_type": node_config.get("type"),
                    "server_address": node_config["address"],
                    "estimated_wait_s": float(wait_s),
                    "estimated_wait_ms": (
                        float(wait_s) * 1000.0
                        if math.isfinite(float(wait_s))
                        else float("inf")
                    ),
                }
            )

    best_model_waits: BestModelWaits = {}
    best_model_routes: BestModelRoutes = {}

    for model_name, candidates in model_candidates.items():
        valid_candidates = [
            candidate
            for candidate in candidates
            if math.isfinite(
                candidate["estimated_wait_s"]
            )
        ]

        if not valid_candidates:
            best_model_waits[model_name] = float("inf")

            best_model_routes[model_name] = {
                "model": model_name,
                "available": False,
                "candidates": candidates,
            }

            continue

        best_route = min(
            valid_candidates,
            key=lambda candidate: (
                candidate["estimated_wait_s"],
                candidate["node_id"],
            ),
        )

        best_model_waits[model_name] = float(
            best_route["estimated_wait_s"]
        )

        best_model_routes[model_name] = {
            **best_route,
            "available": True,
            "candidates": candidates,
        }

    return best_model_waits, best_model_routes


# ============================================================
# Build JSON-safe output
# ============================================================

def make_json_safe(value):
    """
    Replace infinity with None so the output JSON is portable.
    """

    if isinstance(value, dict):
        return {
            key: make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, float) and not math.isfinite(value):
        return None

    return value


# ============================================================
# Print summary
# ============================================================

def print_summary(
    node_wait_results: NodeWaitResults,
    best_model_routes: BestModelRoutes,
) -> None:
    print("\n")
    print("=" * 90)
    print("ALL ILUVATAR NODE WAIT TIMES")
    print("=" * 90)

    for node_id, model_waits in node_wait_results.items():
        node_config = ILUVATAR_NODES[node_id]

        print(
            f"\nNode: {node_id} | "
            f"Type: {node_config.get('type')} | "
            f"Address: {node_config.get('address')}"
        )

        for model_name, wait_s in model_waits.items():
            if math.isfinite(wait_s):
                print(
                    f"  {model_name:<24} "
                    f"{wait_s:.6f} s "
                    f"({wait_s * 1000.0:.2f} ms)"
                )
            else:
                print(
                    f"  {model_name:<24} unavailable"
                )

    print("\n")
    print("=" * 90)
    print("BEST NODE FOR EACH MODEL")
    print("=" * 90)

    for model_name in sorted(best_model_routes):
        route = best_model_routes[model_name]

        if not route.get("available", False):
            print(
                f"{model_name:<24} unavailable"
            )
            continue

        print(
            f"{model_name:<24} "
            f"node={route['node_id']:<10} "
            f"type={route['node_type']:<6} "
            f"wait={route['estimated_wait_ms']:.2f} ms "
            f"server={route['server_address']}"
        )


# ============================================================
# Main
# ============================================================

def main() -> dict:
    collection_start = datetime.now()

    node_wait_results = get_all_node_wait_times()

    (
        best_model_waits,
        best_model_routes,
    ) = build_best_model_routes(
        node_wait_results=node_wait_results,
    )

    output = {
        "collected_at": collection_start.isoformat(),
        "node_wait_results": node_wait_results,
        "best_model_waits": best_model_waits,
        "best_model_routes": best_model_routes,
    }

    print_summary(
        node_wait_results=node_wait_results,
        best_model_routes=best_model_routes,
    )

    output_file = "all_iluvatar_wait_times.json"

    with open(
        output_file,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            make_json_safe(output),
            file,
            indent=2,
        )

    print(
        f"\nResults saved to {output_file}"
    )

    return output


if __name__ == "__main__":
    main()