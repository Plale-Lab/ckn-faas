import json
import uuid
import struct
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    M_TOTAL,
)


# ============================================================
# Protobuf helper
# ============================================================

def _read_varint(
    data: bytes,
    index: int,
) -> Tuple[int, int]:
    """
    Read one protobuf varint.

    Returns:
        (value, new_index)
    """

    value = 0
    shift = 0

    while True:
        if index >= len(data):
            raise ValueError(
                "Unexpected end of protobuf data while reading varint."
            )

        byte = data[index]
        index += 1

        value |= (byte & 0x7F) << shift

        if not (byte & 0x80):
            return value, index

        shift += 7

        if shift >= 64:
            raise ValueError(
                "Invalid protobuf varint."
            )


# ============================================================
# Parse one map<string, double> entry
# ============================================================

def _parse_map_entry(
    data: bytes,
) -> Optional[Tuple[str, float]]:
    """
    Parse one map<string, double> protobuf entry.

    Internally protobuf represents:

        map<string, double> est_times = 1;

    as something equivalent to:

        message EstTimesEntry {
            string key = 1;
            double value = 2;
        }

    Important:
        If the double value is omitted from the wire,
        protobuf's default value is 0.0.
    """

    index = 0

    key: Optional[str] = None

    # protobuf default value for double
    value: float = 0.0

    try:
        while index < len(data):

            tag, index = _read_varint(
                data,
                index,
            )

            field_number = tag >> 3
            wire_type = tag & 0x07

            # ------------------------------------------------
            # field 1 = string key
            # ------------------------------------------------
            if field_number == 1 and wire_type == 2:

                length, index = _read_varint(
                    data,
                    index,
                )

                end = index + length

                if end > len(data):
                    return None

                key = data[index:end].decode(
                    "utf-8"
                )

                index = end
                continue

            # ------------------------------------------------
            # field 2 = double value
            # ------------------------------------------------
            if field_number == 2 and wire_type == 1:

                if index + 8 > len(data):
                    return None

                value = struct.unpack(
                    "<d",
                    data[index:index + 8],
                )[0]

                index += 8
                continue

            # ------------------------------------------------
            # Skip unknown varint field
            # ------------------------------------------------
            if wire_type == 0:

                _, index = _read_varint(
                    data,
                    index,
                )
                continue

            # ------------------------------------------------
            # Skip unknown fixed64
            # ------------------------------------------------
            if wire_type == 1:

                if index + 8 > len(data):
                    return None

                index += 8
                continue

            # ------------------------------------------------
            # Skip unknown length-delimited
            # ------------------------------------------------
            if wire_type == 2:

                length, index = _read_varint(
                    data,
                    index,
                )

                end = index + length

                if end > len(data):
                    return None

                index = end
                continue

            # ------------------------------------------------
            # Skip unknown fixed32
            # ------------------------------------------------
            if wire_type == 5:

                if index + 4 > len(data):
                    return None

                index += 4
                continue

            return None

        if key is not None:
            return (
                key,
                float(value),
            )

    except Exception:
        return None

    return None


# ============================================================
# Decode EstInvokeResponse
# ============================================================

def _decode_est_invoke_response(
    raw_data: bytes,
) -> Dict[str, float]:
    """
    Decode:

        message EstInvokeResponse {
            map<string, double> est_times = 1;
        }

    Example server response:

        {
            "cpu_estimated_wait_time_sec": 0.0318,
            "gpu_estimated_wait_time_sec": 0.0,
            "cpu_waiting_functions": 0.0,
            "cpu_running_functions": 0.0,
            "gpu_waiting_functions": 0.0,
            "gpu_running_functions": 0.0
        }
    """

    if not raw_data:
        return {}

    index = 0

    results: Dict[str, float] = {}

    while index < len(raw_data):

        tag, index = _read_varint(
            raw_data,
            index,
        )

        field_number = tag >> 3
        wire_type = tag & 0x07

        # ----------------------------------------------------
        # est_times = field 1
        #
        # map entries are length-delimited
        # ----------------------------------------------------
        if field_number == 1 and wire_type == 2:

            length, index = _read_varint(
                raw_data,
                index,
            )

            end = index + length

            if end > len(raw_data):
                raise ValueError(
                    "Invalid protobuf map-entry length."
                )

            payload = raw_data[
                index:end
            ]

            index = end

            entry = _parse_map_entry(
                payload
            )

            if entry is not None:

                key, value = entry

                results[key] = value

            continue

        # ----------------------------------------------------
        # Skip unknown top-level fields
        # ----------------------------------------------------
        if wire_type == 0:

            _, index = _read_varint(
                raw_data,
                index,
            )
            continue

        if wire_type == 1:

            if index + 8 > len(raw_data):
                raise ValueError(
                    "Invalid fixed64 field."
                )

            index += 8
            continue

        if wire_type == 2:

            length, index = _read_varint(
                raw_data,
                index,
            )

            end = index + length

            if end > len(raw_data):
                raise ValueError(
                    "Invalid length-delimited field."
                )

            index = end
            continue

        if wire_type == 5:

            if index + 4 > len(raw_data):
                raise ValueError(
                    "Invalid fixed32 field."
                )

            index += 4
            continue

        raise ValueError(
            f"Unsupported protobuf wire type: {wire_type}"
        )

    return results


# ============================================================
# Create raw-response RPC
# ============================================================

def _get_raw_est_invoke_rpc(
    channel,
    normal_stub,
):
    """
    Create est_invoke_time RPC that returns raw bytes instead
    of automatically deserializing EstInvokeResponse.
    """

    normal_rpc = (
        normal_stub.est_invoke_time
    )

    method = getattr(
        normal_rpc,
        "_method",
        None,
    )

    if method is None:
        raise RuntimeError(
            "Could not determine est_invoke_time RPC method path."
        )

    if isinstance(
        method,
        bytes,
    ):
        method = method.decode(
            "utf-8"
        )

    return channel.unary_unary(
        method,
        request_serializer=lambda request: (
            request.SerializeToString()
        ),
        response_deserializer=lambda data: data,
    )


# ============================================================
# Select estimated latency from returned metrics
# ============================================================

def _select_estimated_time(
    estimates: Dict[str, float],
) -> Optional[float]:
    """
    Select the usable estimated time.

    Current server returns:

        cpu_estimated_wait_time_sec
        gpu_estimated_wait_time_sec

    Policy:
        1. Use GPU estimate if it is > 0
        2. Otherwise use CPU estimate if it is > 0
        3. Otherwise return None
    """

    cpu_estimate = float(
        estimates.get(
            "cpu_estimated_wait_time_sec",
            0.0,
        )
    )

    gpu_estimate = float(
        estimates.get(
            "gpu_estimated_wait_time_sec",
            0.0,
        )
    )

    if gpu_estimate > 0.0:
        return gpu_estimate

    if cpu_estimate > 0.0:
        return cpu_estimate

    return None


# ============================================================
# Get estimated invocation time for one model
# ============================================================

def get_estimated_wait(
    raw_est_rpc,
    model_name: str,
    server_address: Optional[str] = None,
) -> float:
    """
    Query one Iluvatar instance for one model.

    Returns:
        Estimated invocation time in seconds.

        Returns float("inf") if:
        - RPC fails
        - response cannot be decoded
        - CPU/GPU estimate is unavailable
    """

    fqdn = f"{model_name}-1"

    request = pb2.EstInvokeRequest(
        transaction_id=str(
            uuid.uuid4()
        ),
        fqdns=[
            fqdn
        ],
    )

    try:
        # ----------------------------------------------------
        # Get raw protobuf bytes
        # ----------------------------------------------------
        raw_response = raw_est_rpc(
            request
        )

        print(
            f"[{datetime.now()}] "
            f"Raw response received "
            f"server={server_address} "
            f"model={model_name} "
            f"fqdn={fqdn} "
            f"bytes={len(raw_response)}"
        )

        # ----------------------------------------------------
        # Decode metric map
        # ----------------------------------------------------
        estimates = (
            _decode_est_invoke_response(
                raw_response
            )
        )

        if not estimates:

            print(
                f"[{datetime.now()}] "
                f"No estimator metrics decoded "
                f"server={server_address} "
                f"model={model_name} "
                f"fqdn={fqdn}"
            )

            return float(
                "inf"
            )

        # ----------------------------------------------------
        # Extract individual metrics
        # ----------------------------------------------------
        cpu_estimate = float(
            estimates.get(
                "cpu_estimated_wait_time_sec",
                0.0,
            )
        )

        gpu_estimate = float(
            estimates.get(
                "gpu_estimated_wait_time_sec",
                0.0,
            )
        )

        cpu_waiting = float(
            estimates.get(
                "cpu_waiting_functions",
                0.0,
            )
        )

        cpu_running = float(
            estimates.get(
                "cpu_running_functions",
                0.0,
            )
        )

        gpu_waiting = float(
            estimates.get(
                "gpu_waiting_functions",
                0.0,
            )
        )

        gpu_running = float(
            estimates.get(
                "gpu_running_functions",
                0.0,
            )
        )

        print(
            f"[{datetime.now()}] "
            f"Estimator response "
            f"server={server_address} "
            f"model={model_name} "
            f"fqdn={fqdn} "
            f"cpu_estimate_s={cpu_estimate:.6f} "
            f"gpu_estimate_s={gpu_estimate:.6f} "
            f"cpu_waiting={cpu_waiting:.0f} "
            f"cpu_running={cpu_running:.0f} "
            f"gpu_waiting={gpu_waiting:.0f} "
            f"gpu_running={gpu_running:.0f}"
        )

        # ----------------------------------------------------
        # Select actual latency estimate
        # ----------------------------------------------------
        estimated_time = (
            _select_estimated_time(
                estimates
            )
        )

        if estimated_time is None:

            print(
                f"[{datetime.now()}] "
                f"No usable latency estimate "
                f"server={server_address} "
                f"model={model_name} "
                f"fqdn={fqdn}"
            )

            return float(
                "inf"
            )

        print(
            f"[{datetime.now()}] "
            f"Estimate selected "
            f"server={server_address} "
            f"model={model_name} "
            f"fqdn={fqdn} "
            f"estimated_time_s="
            f"{estimated_time:.6f}"
        )

        return float(
            estimated_time
        )

    except grpc.RpcError as error:

        try:
            details = (
                error.details()
            )

        except Exception:
            details = str(
                error
            )

        print(
            f"[{datetime.now()}] "
            f"Failed to estimate "
            f"server={server_address} "
            f"model={model_name} "
            f"fqdn={fqdn} "
            f"error={details}"
        )

        return float(
            "inf"
        )

    except Exception as error:

        print(
            f"[{datetime.now()}] "
            f"Unexpected estimation error "
            f"server={server_address} "
            f"model={model_name} "
            f"fqdn={fqdn} "
            f"error={error}"
        )

        return float(
            "inf"
        )


# ============================================================
# Query one Iluvatar instance
# ============================================================

def main(
    server_address: Optional[str] = None,
    model_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Query one Iluvatar instance for a list of models.

    This function does not select the best node.

    ckn_main.py can call this function for each edge/cloud node
    and compare the returned estimated times.

    Example return:

        {
            "mobilenet_v3_small": 0.0074,
            "mobilenet_v2": 0.0136,
            "resnet18": 0.0198,
            "resnet34": 0.0318
        }
    """

    address = (
        server_address
        or SERVER_ADDRESS
    )

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

    # --------------------------------------------------------
    # Create gRPC channel
    # --------------------------------------------------------
    channel = grpc.insecure_channel(
        address
    )

    normal_stub = (
        pb2_grpc.IluvatarWorkerStub(
            channel
        )
    )

    raw_est_rpc = (
        _get_raw_est_invoke_rpc(
            channel=channel,
            normal_stub=normal_stub,
        )
    )

    wait_results: Dict[
        str,
        float
    ] = {}

    try:
        for model_name in models:

            wait_results[
                model_name
            ] = get_estimated_wait(
                raw_est_rpc=raw_est_rpc,
                model_name=model_name,
                server_address=address,
            )

    finally:
        try:
            channel.close()

        except Exception:
            pass

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------
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














# # ============================================================
# # First code end- just consider wait time
# # ============================================================







#
# import json
# import uuid
# import struct
# from datetime import datetime
# from typing import Dict, List, Optional, Tuple
#
# import grpc
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
# from ckn_controller.ckn_config import (
#     SERVER_ADDRESS,
#     M_TOTAL,
#     MODEL_PROFILES,
#     ILUVATAR_NODES,
#     SMALL_INSTANCE_MODELS,
#     LARGE_INSTANCE_MODELS,
# )
#
#
# # ============================================================
# # Get model execution/runtime estimate from config
# # ============================================================
#
# def get_model_runtime_s(
#     model_name: str,
# ) -> Optional[float]:
#     """
#     Get expected model execution time from MODEL_PROFILES.
#
#     Example:
#
#         MODEL_PROFILES["resnet34"]["latency"]
#             -> 0.016935 seconds
#
#     Returns:
#         Runtime in seconds.
#
#         None if the model or latency is not available.
#     """
#
#     profile = MODEL_PROFILES.get(
#         model_name
#     )
#
#     if profile is None:
#         return None
#
#     latency = profile.get(
#         "latency"
#     )
#
#     if latency is None:
#         return None
#
#     try:
#         latency = float(latency)
#
#         if latency < 0.0:
#             return None
#
#         return latency
#
#     except (TypeError, ValueError):
#         return None
#
#
# # ============================================================
# # Determine edge/cloud node for a model + server
# # ============================================================
#
# def get_node_info_for_server_model(
#     server_address: str,
#     model_name: str,
# ) -> Optional[Dict[str, str]]:
#     """
#     Find the configured Iluvatar node corresponding to the
#     supplied server address and model.
#
#     This uses BOTH address and model because an edge and cloud
#     node may currently share the same physical server address.
#
#     Example configuration:
#
#         edge_1:
#             address = 129.79.242.28:8079
#             models  = [resnet18, resnet34, ...]
#
#         cloud_1:
#             address = 129.79.242.28:8079
#             models  = [resnet50, convnext_tiny, ...]
#
#     Therefore address alone is NOT enough.
#
#     Returns example:
#
#         {
#             "node_id": "edge_1",
#             "node_type": "edge",
#             "compute_type": "cpu",
#         }
#     """
#
#     for node_id, node in ILUVATAR_NODES.items():
#
#         # ----------------------------------------------------
#         # Ignore disabled nodes
#         # ----------------------------------------------------
#         if not bool(
#             node.get(
#                 "enabled",
#                 True,
#             )
#         ):
#             continue
#
#         node_address = str(
#             node.get(
#                 "address",
#                 "",
#             )
#         )
#
#         # ----------------------------------------------------
#         # Address must match
#         # ----------------------------------------------------
#         if node_address != str(server_address):
#             continue
#
#         deployed_models = list(
#             node.get(
#                 "models",
#                 []
#             )
#         )
#
#         # ----------------------------------------------------
#         # Model must also exist on this node
#         # ----------------------------------------------------
#         if model_name not in deployed_models:
#             continue
#
#         node_type = str(
#             node.get(
#                 "type",
#                 "",
#             )
#         ).lower()
#
#         if node_type == "edge":
#
#             return {
#                 "node_id": str(node_id),
#                 "node_type": "edge",
#                 "compute_type": "cpu",
#             }
#
#         if node_type == "cloud":
#
#             return {
#                 "node_id": str(node_id),
#                 "node_type": "cloud",
#                 "compute_type": "cpu",
#             }
#
#     return None
#
#
# # ============================================================
# # Determine CPU/GPU compute type
# # ============================================================
#
# def get_compute_type_for_server(
#     server_address: str,
#     model_name: str,
# ) -> str:
#     """
#     Determine whether the requested model should use the
#     CPU or GPU estimator.
#
#     Primary method:
#         ILUVATAR_NODES
#
#     Fallback:
#         SMALL_INSTANCE_MODELS -> CPU
#         LARGE_INSTANCE_MODELS -> GPU
#     """
#
#     node_info = get_node_info_for_server_model(
#         server_address=server_address,
#         model_name=model_name,
#     )
#
#     if node_info is not None:
#
#         return node_info[
#             "compute_type"
#         ]
#
#     # --------------------------------------------------------
#     # Backward-compatible fallback
#     # --------------------------------------------------------
#
#     if model_name in SMALL_INSTANCE_MODELS:
#         return "cpu"
#
#     if model_name in LARGE_INSTANCE_MODELS:
#         return "gpu"
#
#     # --------------------------------------------------------
#     # Conservative fallback
#     # --------------------------------------------------------
#
#     print(
#         f"[{datetime.now()}] "
#         f"Could not determine compute type "
#         f"server={server_address} "
#         f"model={model_name}; "
#         f"defaulting to CPU"
#     )
#
#     return "cpu"
#
#
# # ============================================================
# # Protobuf varint helper
# # ============================================================
#
# def _read_varint(
#     data: bytes,
#     index: int,
# ) -> Tuple[int, int]:
#     """
#     Read one protobuf varint.
#
#     Returns:
#         (value, new_index)
#     """
#
#     value = 0
#     shift = 0
#
#     while True:
#
#         if index >= len(data):
#
#             raise ValueError(
#                 "Unexpected end of protobuf data "
#                 "while reading varint."
#             )
#
#         byte = data[index]
#         index += 1
#
#         value |= (
#             byte & 0x7F
#         ) << shift
#
#         if not (
#             byte & 0x80
#         ):
#
#             return (
#                 value,
#                 index,
#             )
#
#         shift += 7
#
#         if shift >= 64:
#
#             raise ValueError(
#                 "Invalid protobuf varint."
#             )
#
#
# # ============================================================
# # Parse map<string, double> entry
# # ============================================================
#
# def _parse_map_entry(
#     data: bytes,
# ) -> Optional[Tuple[str, float]]:
#     """
#     Parse one protobuf:
#
#         map<string, double>
#
#     entry.
#
#     Internally:
#
#         message Entry {
#             string key = 1;
#             double value = 2;
#         }
#
#     Important:
#
#     Protobuf may omit value when value == 0.0.
#
#     Therefore:
#
#         key present
#         value absent
#
#     means:
#
#         value = 0.0
#     """
#
#     index = 0
#
#     key: Optional[str] = None
#
#     # protobuf default double
#     value: float = 0.0
#
#     try:
#
#         while index < len(data):
#
#             tag, index = _read_varint(
#                 data,
#                 index,
#             )
#
#             field_number = (
#                 tag >> 3
#             )
#
#             wire_type = (
#                 tag & 0x07
#             )
#
#             # =================================================
#             # field 1 = string key
#             # =================================================
#
#             if (
#                 field_number == 1
#                 and wire_type == 2
#             ):
#
#                 length, index = _read_varint(
#                     data,
#                     index,
#                 )
#
#                 end = (
#                     index
#                     +
#                     length
#                 )
#
#                 if end > len(data):
#                     return None
#
#                 key = data[
#                     index:end
#                 ].decode(
#                     "utf-8"
#                 )
#
#                 index = end
#
#                 continue
#
#             # =================================================
#             # field 2 = double
#             # =================================================
#
#             if (
#                 field_number == 2
#                 and wire_type == 1
#             ):
#
#                 if (
#                     index + 8
#                     >
#                     len(data)
#                 ):
#                     return None
#
#                 value = struct.unpack(
#                     "<d",
#                     data[
#                         index:index + 8
#                     ],
#                 )[0]
#
#                 index += 8
#
#                 continue
#
#             # =================================================
#             # Unknown varint
#             # =================================================
#
#             if wire_type == 0:
#
#                 _, index = _read_varint(
#                     data,
#                     index,
#                 )
#
#                 continue
#
#             # =================================================
#             # Unknown fixed64
#             # =================================================
#
#             if wire_type == 1:
#
#                 if (
#                     index + 8
#                     >
#                     len(data)
#                 ):
#                     return None
#
#                 index += 8
#
#                 continue
#
#             # =================================================
#             # Unknown length-delimited
#             # =================================================
#
#             if wire_type == 2:
#
#                 length, index = _read_varint(
#                     data,
#                     index,
#                 )
#
#                 end = (
#                     index
#                     +
#                     length
#                 )
#
#                 if end > len(data):
#                     return None
#
#                 index = end
#
#                 continue
#
#             # =================================================
#             # Unknown fixed32
#             # =================================================
#
#             if wire_type == 5:
#
#                 if (
#                     index + 4
#                     >
#                     len(data)
#                 ):
#                     return None
#
#                 index += 4
#
#                 continue
#
#             return None
#
#         if key is not None:
#
#             return (
#                 key,
#                 float(value),
#             )
#
#     except Exception:
#
#         return None
#
#     return None
#
#
# # ============================================================
# # Decode EstInvokeResponse
# # ============================================================
#
# def _decode_est_invoke_response(
#     raw_data: bytes,
# ) -> Dict[str, float]:
#     """
#     Decode:
#
#         message EstInvokeResponse {
#             map<string, double> est_times = 1;
#         }
#
#     Current Iluvatar response contains:
#
#         cpu_estimated_wait_time_sec
#         gpu_estimated_wait_time_sec
#
#         cpu_waiting_functions
#         cpu_running_functions
#
#         gpu_waiting_functions
#         gpu_running_functions
#     """
#
#     if not raw_data:
#         return {}
#
#     index = 0
#
#     results: Dict[
#         str,
#         float
#     ] = {}
#
#     while index < len(raw_data):
#
#         tag, index = _read_varint(
#             raw_data,
#             index,
#         )
#
#         field_number = (
#             tag >> 3
#         )
#
#         wire_type = (
#             tag & 0x07
#         )
#
#         # ====================================================
#         # est_times map = field 1
#         # ====================================================
#
#         if (
#             field_number == 1
#             and wire_type == 2
#         ):
#
#             length, index = _read_varint(
#                 raw_data,
#                 index,
#             )
#
#             end = (
#                 index
#                 +
#                 length
#             )
#
#             if end > len(raw_data):
#
#                 raise ValueError(
#                     "Invalid protobuf "
#                     "map-entry length."
#                 )
#
#             payload = raw_data[
#                 index:end
#             ]
#
#             index = end
#
#             entry = _parse_map_entry(
#                 payload
#             )
#
#             if entry is not None:
#
#                 key, value = entry
#
#                 results[
#                     key
#                 ] = value
#
#             continue
#
#         # ====================================================
#         # Skip unknown top-level varint
#         # ====================================================
#
#         if wire_type == 0:
#
#             _, index = _read_varint(
#                 raw_data,
#                 index,
#             )
#
#             continue
#
#         # ====================================================
#         # Skip unknown fixed64
#         # ====================================================
#
#         if wire_type == 1:
#
#             if (
#                 index + 8
#                 >
#                 len(raw_data)
#             ):
#
#                 raise ValueError(
#                     "Invalid fixed64 field."
#                 )
#
#             index += 8
#
#             continue
#
#         # ====================================================
#         # Skip unknown length-delimited
#         # ====================================================
#
#         if wire_type == 2:
#
#             length, index = _read_varint(
#                 raw_data,
#                 index,
#             )
#
#             end = (
#                 index
#                 +
#                 length
#             )
#
#             if end > len(raw_data):
#
#                 raise ValueError(
#                     "Invalid "
#                     "length-delimited field."
#                 )
#
#             index = end
#
#             continue
#
#         # ====================================================
#         # Skip unknown fixed32
#         # ====================================================
#
#         if wire_type == 5:
#
#             if (
#                 index + 4
#                 >
#                 len(raw_data)
#             ):
#
#                 raise ValueError(
#                     "Invalid fixed32 field."
#                 )
#
#             index += 4
#
#             continue
#
#         raise ValueError(
#             f"Unsupported protobuf "
#             f"wire type: {wire_type}"
#         )
#
#     return results
#
#
# # ============================================================
# # Create raw-response RPC
# # ============================================================
#
# def _get_raw_est_invoke_rpc(
#     channel,
#     normal_stub,
# ):
#     """
#     Use the normal generated request serializer but return
#     response bytes without protobuf response deserialization.
#
#     This is necessary because the server's current response
#     structure does not match the generated Python response
#     definition correctly.
#     """
#
#     normal_rpc = (
#         normal_stub.est_invoke_time
#     )
#
#     method = getattr(
#         normal_rpc,
#         "_method",
#         None,
#     )
#
#     if method is None:
#
#         raise RuntimeError(
#             "Could not determine "
#             "est_invoke_time RPC method path."
#         )
#
#     if isinstance(
#         method,
#         bytes,
#     ):
#
#         method = method.decode(
#             "utf-8"
#         )
#
#     return channel.unary_unary(
#         method,
#         request_serializer=lambda request: (
#             request.SerializeToString()
#         ),
#         response_deserializer=lambda data: data,
#     )
#
#
# # ============================================================
# # Select CPU/GPU queue wait
# # ============================================================
#
# def _select_estimated_wait(
#     estimates: Dict[str, float],
#     model_name: str,
#     server_address: str,
# ) -> Optional[float]:
#     """
#     Select the appropriate queue estimate.
#
#     Edge model:
#         cpu_estimated_wait_time_sec
#
#     Cloud model:
#         gpu_estimated_wait_time_sec
#
#     We DO NOT automatically prefer GPU merely because it is
#     non-zero.
#     """
#
#     compute_type = get_compute_type_for_server(
#         server_address=server_address,
#         model_name=model_name,
#     )
#
#     cpu_wait_s = float(
#         estimates.get(
#             "cpu_estimated_wait_time_sec",
#             0.0,
#         )
#     )
#
#     gpu_wait_s = float(
#         estimates.get(
#             "gpu_estimated_wait_time_sec",
#             0.0,
#         )
#     )
#
#     if compute_type == "cpu":
#
#         if cpu_wait_s > 0.0:
#
#             return cpu_wait_s
#
#         return None
#
#     if compute_type == "gpu":
#
#         if gpu_wait_s > 0.0:
#
#             return gpu_wait_s
#
#         return None
#
#     return None
#
#
# # ============================================================
# # Calculate total model latency
# # ============================================================
#
# def _calculate_total_model_latency(
#     model_name: str,
#     estimated_wait_s: float,
# ) -> Optional[float]:
#     """
#     Calculate estimated model latency:
#
#         T(m) = Q(m) + S(m)
#
#     where:
#
#         Q(m)
#             = current Iluvatar estimated queue/wait time
#
#         S(m)
#             = model execution/service time from MODEL_PROFILES
#
#     cpu_waiting_functions and cpu_running_functions are NOT
#     added here because they are counts, not time measurements.
#     """
#
#     runtime_s = get_model_runtime_s(
#         model_name
#     )
#
#     if runtime_s is None:
#
#         print(
#             f"[{datetime.now()}] "
#             f"No model runtime profile "
#             f"model={model_name}"
#         )
#
#         return None
#
#     estimated_total_s = (
#         float(estimated_wait_s)
#         +
#         float(runtime_s)
#     )
#
#     return estimated_total_s
#
#
# # ============================================================
# # Get estimated latency for one model
# # ============================================================
#
# def get_estimated_wait(
#     raw_est_rpc,
#     model_name: str,
#     server_address: Optional[str] = None,
# ) -> float:
#     """
#     Query one Iluvatar instance for one model.
#
#     Final returned value:
#
#         T(model)
#             =
#         estimated queue/wait time
#             +
#         MODEL_PROFILES[model]["latency"]
#
#     Returns:
#         Total estimated latency in seconds.
#
#     Returns float("inf") if:
#         - RPC fails
#         - estimator cannot be decoded
#         - correct CPU/GPU estimate is unavailable
#         - runtime profile is unavailable
#     """
#
#     if server_address is None:
#
#         raise ValueError(
#             "server_address is required."
#         )
#
#     fqdn = (
#         f"{model_name}-1"
#     )
#
#     request = pb2.EstInvokeRequest(
#         transaction_id=str(
#             uuid.uuid4()
#         ),
#         fqdns=[
#             fqdn
#         ],
#     )
#
#     try:
#
#         # ====================================================
#         # Get raw response
#         # ====================================================
#
#         raw_response = raw_est_rpc(
#             request
#         )
#
#         print(
#             f"[{datetime.now()}] "
#             f"Raw response received "
#             f"server={server_address} "
#             f"model={model_name} "
#             f"fqdn={fqdn} "
#             f"bytes={len(raw_response)}"
#         )
#
#         # ====================================================
#         # Decode Iluvatar metrics
#         # ====================================================
#
#         estimates = (
#             _decode_est_invoke_response(
#                 raw_response
#             )
#         )
#
#         if not estimates:
#
#             print(
#                 f"[{datetime.now()}] "
#                 f"No estimator metrics decoded "
#                 f"server={server_address} "
#                 f"model={model_name} "
#                 f"fqdn={fqdn}"
#             )
#
#             return float(
#                 "inf"
#             )
#
#         # ====================================================
#         # Extract estimator metrics
#         # ====================================================
#
#         cpu_wait_s = float(
#             estimates.get(
#                 "cpu_estimated_wait_time_sec",
#                 0.0,
#             )
#         )
#
#         gpu_wait_s = float(
#             estimates.get(
#                 "gpu_estimated_wait_time_sec",
#                 0.0,
#             )
#         )
#
#         cpu_waiting = float(
#             estimates.get(
#                 "cpu_waiting_functions",
#                 0.0,
#             )
#         )
#
#         cpu_running = float(
#             estimates.get(
#                 "cpu_running_functions",
#                 0.0,
#             )
#         )
#
#         gpu_waiting = float(
#             estimates.get(
#                 "gpu_waiting_functions",
#                 0.0,
#             )
#         )
#
#         gpu_running = float(
#             estimates.get(
#                 "gpu_running_functions",
#                 0.0,
#             )
#         )
#
#         # ====================================================
#         # Determine node / compute type from config
#         # ====================================================
#
#         node_info = get_node_info_for_server_model(
#             server_address=server_address,
#             model_name=model_name,
#         )
#
#         compute_type = get_compute_type_for_server(
#             server_address=server_address,
#             model_name=model_name,
#         )
#
#         node_id = (
#             node_info["node_id"]
#             if node_info is not None
#             else "unknown"
#         )
#
#         node_type = (
#             node_info["node_type"]
#             if node_info is not None
#             else "unknown"
#         )
#
#         # ====================================================
#         # Log raw estimator state
#         # ====================================================
#
#         print(
#             f"[{datetime.now()}] "
#             f"Estimator response "
#             f"server={server_address} "
#             f"node={node_id} "
#             f"node_type={node_type} "
#             f"model={model_name} "
#             f"fqdn={fqdn} "
#             f"compute_type={compute_type} "
#             f"cpu_wait_s={cpu_wait_s:.6f} "
#             f"gpu_wait_s={gpu_wait_s:.6f} "
#             f"cpu_waiting={cpu_waiting:.0f} "
#             f"cpu_running={cpu_running:.0f} "
#             f"gpu_waiting={gpu_waiting:.0f} "
#             f"gpu_running={gpu_running:.0f}"
#         )
#
#         # ====================================================
#         # Select CPU or GPU queue delay
#         # ====================================================
#
#         estimated_wait_s = (
#             _select_estimated_wait(
#                 estimates=estimates,
#                 model_name=model_name,
#                 server_address=server_address,
#             )
#         )
#
#         if estimated_wait_s is None:
#
#             print(
#                 f"[{datetime.now()}] "
#                 f"No usable queue estimate "
#                 f"server={server_address} "
#                 f"node={node_id} "
#                 f"model={model_name} "
#                 f"compute_type={compute_type} "
#                 f"cpu_wait_s={cpu_wait_s:.6f} "
#                 f"gpu_wait_s={gpu_wait_s:.6f}"
#             )
#
#             return float(
#                 "inf"
#             )
#
#         # ====================================================
#         # Get execution time from MODEL_PROFILES
#         # ====================================================
#
#         runtime_s = get_model_runtime_s(
#             model_name
#         )
#
#         if runtime_s is None:
#
#             print(
#                 f"[{datetime.now()}] "
#                 f"No runtime available "
#                 f"model={model_name}"
#             )
#
#             return float(
#                 "inf"
#             )
#
#         # ====================================================
#         # T(model) = queue delay + execution time
#         # ====================================================
#
#         estimated_total_s = (
#             _calculate_total_model_latency(
#                 model_name=model_name,
#                 estimated_wait_s=estimated_wait_s,
#             )
#         )
#
#         if estimated_total_s is None:
#
#             return float(
#                 "inf"
#             )
#
#         # ====================================================
#         # Final estimate
#         # ====================================================
#
#         print(
#             f"[{datetime.now()}] "
#             f"Model latency estimate "
#             f"server={server_address} "
#             f"node={node_id} "
#             f"node_type={node_type} "
#             f"model={model_name} "
#             f"compute_type={compute_type} "
#             f"queue_wait_s={estimated_wait_s:.6f} "
#             f"runtime_s={runtime_s:.6f} "
#             f"estimated_total_s={estimated_total_s:.6f}"
#         )
#
#         return float(
#             estimated_total_s
#         )
#
#     # ========================================================
#     # gRPC errors
#     # ========================================================
#
#     except grpc.RpcError as error:
#
#         try:
#
#             details = (
#                 error.details()
#             )
#
#         except Exception:
#
#             details = str(
#                 error
#             )
#
#         print(
#             f"[{datetime.now()}] "
#             f"Failed to estimate "
#             f"server={server_address} "
#             f"model={model_name} "
#             f"fqdn={fqdn} "
#             f"error={details}"
#         )
#
#         return float(
#             "inf"
#         )
#
#     # ========================================================
#     # Other errors
#     # ========================================================
#
#     except Exception as error:
#
#         print(
#             f"[{datetime.now()}] "
#             f"Unexpected estimation error "
#             f"server={server_address} "
#             f"model={model_name} "
#             f"fqdn={fqdn} "
#             f"error={error}"
#         )
#
#         return float(
#             "inf"
#         )
#
#
# # ============================================================
# # Query one Iluvatar instance
# # ============================================================
#
# def main(
#     server_address: Optional[str] = None,
#     model_list: Optional[List[str]] = None,
# ) -> Dict[str, float]:
#     """
#     Query one Iluvatar instance for a list of models.
#
#     This function does NOT perform node selection.
#
#     ckn_main.py calls this function for each eligible Iluvatar
#     node.
#
#     Returns:
#
#         {
#             model_name: total_estimated_latency_s
#         }
#
#     Example:
#
#         {
#             "resnet18": 0.078,
#             "resnet34": 0.093
#         }
#
#     where:
#
#         total_estimated_latency
#             =
#         Iluvatar queue/wait estimate
#             +
#         MODEL_PROFILES execution latency
#     """
#
#     address = (
#         server_address
#         or SERVER_ADDRESS
#     )
#
#     models = (
#         list(model_list)
#         if model_list is not None
#         else list(M_TOTAL)
#     )
#
#     if not address:
#
#         raise ValueError(
#             "server_address is required."
#         )
#
#     if not models:
#
#         print(
#             f"[{datetime.now()}] "
#             f"No models provided "
#             f"server={address}"
#         )
#
#         return {}
#
#     # ========================================================
#     # Create gRPC channel
#     # ========================================================
#
#     channel = grpc.insecure_channel(
#         address
#     )
#
#     normal_stub = (
#         pb2_grpc.IluvatarWorkerStub(
#             channel
#         )
#     )
#
#     # ========================================================
#     # Raw response RPC
#     # ========================================================
#
#     raw_est_rpc = (
#         _get_raw_est_invoke_rpc(
#             channel=channel,
#             normal_stub=normal_stub,
#         )
#     )
#
#     wait_results: Dict[
#         str,
#         float
#     ] = {}
#
#     try:
#
#         for model_name in models:
#
#             wait_results[
#                 model_name
#             ] = get_estimated_wait(
#                 raw_est_rpc=raw_est_rpc,
#                 model_name=model_name,
#                 server_address=address,
#             )
#
#     finally:
#
#         try:
#
#             channel.close()
#
#         except Exception:
#
#             pass
#
#     # ========================================================
#     # Final output
#     # ========================================================
#
#     output = {
#         "server_address": address,
#         "models": models,
#         "estimated_model_latencies": wait_results,
#     }
#
#     print(
#         json.dumps(
#             output,
#             indent=2,
#             default=str,
#         )
#     )
#
#     return wait_results
#
#
# # ============================================================
# # Standalone test
# # ============================================================
#
# if __name__ == "__main__":
#
#     main()














# # ============================================================
# # second code end- wait time + service time
# # ============================================================





























