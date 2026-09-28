# import struct
# import time
# import uuid
#
# from typing import Dict, Optional, Tuple
#
# import grpc
#
# import ckn_controller.iluvatar_rpc_pb2 as pb2
# import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc
#
#
# # ============================================================
# # Protobuf helpers
# # ============================================================
#
# def _read_varint(
#     data: bytes,
#     index: int,
# ) -> Tuple[int, int]:
#
#     value = 0
#     shift = 0
#
#     while True:
#
#         if index >= len(data):
#             raise ValueError(
#                 "Unexpected end of protobuf data."
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
#             return value, index
#
#         shift += 7
#
#         if shift >= 64:
#             raise ValueError(
#                 "Invalid protobuf varint."
#             )
#
#
# # ============================================================
# # Parse map<string, double>
# # ============================================================
#
# def _parse_map_entry(
#     data: bytes,
# ) -> Optional[
#     Tuple[str, float]
# ]:
#
#     index = 0
#     key = None
#
#     # protobuf default double value
#     value = 0.0
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
#             # ------------------------------------------------
#             # Field 1 = string key
#             # ------------------------------------------------
#
#             if (
#                 field_number == 1
#                 and wire_type == 2
#             ):
#
#                 length, index = (
#                     _read_varint(
#                         data,
#                         index,
#                     )
#                 )
#
#                 end = (
#                     index + length
#                 )
#
#                 if end > len(data):
#                     return None
#
#                 key = (
#                     data[
#                         index:end
#                     ].decode(
#                         "utf-8"
#                     )
#                 )
#
#                 index = end
#
#                 continue
#
#             # ------------------------------------------------
#             # Field 2 = double value
#             # ------------------------------------------------
#
#             if (
#                 field_number == 2
#                 and wire_type == 1
#             ):
#
#                 if (
#                     index + 8
#                     > len(data)
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
#             # ------------------------------------------------
#             # Skip unknown varint
#             # ------------------------------------------------
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
#             # ------------------------------------------------
#             # Skip fixed64
#             # ------------------------------------------------
#
#             if wire_type == 1:
#
#                 if (
#                     index + 8
#                     > len(data)
#                 ):
#                     return None
#
#                 index += 8
#
#                 continue
#
#             # ------------------------------------------------
#             # Skip length-delimited
#             # ------------------------------------------------
#
#             if wire_type == 2:
#
#                 length, index = (
#                     _read_varint(
#                         data,
#                         index,
#                     )
#                 )
#
#                 end = (
#                     index + length
#                 )
#
#                 if end > len(data):
#                     return None
#
#                 index = end
#
#                 continue
#
#             # ------------------------------------------------
#             # Skip fixed32
#             # ------------------------------------------------
#
#             if wire_type == 5:
#
#                 if (
#                     index + 4
#                     > len(data)
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
#
#     if not raw_data:
#         return {}
#
#     index = 0
#     results = {}
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
#         # ----------------------------------------------------
#         # est_times map = field 1
#         # ----------------------------------------------------
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
#                 index + length
#             )
#
#             if end > len(raw_data):
#                 raise ValueError(
#                     "Invalid protobuf map-entry length."
#                 )
#
#             payload = (
#                 raw_data[
#                     index:end
#                 ]
#             )
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
#                 results[key] = value
#
#             continue
#
#         # ----------------------------------------------------
#         # Skip unknown fields
#         # ----------------------------------------------------
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
#         if wire_type == 1:
#
#             if (
#                 index + 8
#                 > len(raw_data)
#             ):
#                 raise ValueError(
#                     "Invalid fixed64 field."
#                 )
#
#             index += 8
#
#             continue
#
#         if wire_type == 2:
#
#             length, index = _read_varint(
#                 raw_data,
#                 index,
#             )
#
#             end = (
#                 index + length
#             )
#
#             if end > len(raw_data):
#                 raise ValueError(
#                     "Invalid length-delimited field."
#                 )
#
#             index = end
#
#             continue
#
#         if wire_type == 5:
#
#             if (
#                 index + 4
#                 > len(raw_data)
#             ):
#                 raise ValueError(
#                     "Invalid fixed32 field."
#                 )
#
#             index += 4
#
#             continue
#
#         raise ValueError(
#             f"Unsupported protobuf wire type: "
#             f"{wire_type}"
#         )
#
#     return results
#
#
# # ============================================================
# # Create raw estimator RPC
# # ============================================================
#
# def _get_raw_est_invoke_rpc(
#     channel,
#     normal_stub,
# ):
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
#             "est_invoke_time RPC path."
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
#
#         method,
#
#         request_serializer=(
#             lambda request:
#             request.SerializeToString()
#         ),
#
#         response_deserializer=(
#             lambda data:
#             data
#         ),
#     )
#
#
# # ============================================================
# # Estimator client
# # ============================================================
#
# class IluvatarWaitEstimator:
#
#     def __init__(
#         self,
#         server_address: str,
#         channel=None,
#     ):
#
#         self.server_address = (
#             server_address
#         )
#
#         self.owns_channel = (
#             channel is None
#         )
#
#         if channel is None:
#
#             self.channel = grpc.insecure_channel(
#                 server_address
#             )
#
#         else:
#
#             self.channel = channel
#
#         self.normal_stub = (
#             pb2_grpc.IluvatarWorkerStub(
#                 self.channel
#             )
#         )
#
#         self.raw_est_rpc = (
#             _get_raw_est_invoke_rpc(
#                 channel=self.channel,
#                 normal_stub=(
#                     self.normal_stub
#                 ),
#             )
#         )
#
#     # ========================================================
#     # Query ONE model
#     # ========================================================
#
#     def query(
#         self,
#         model_name: str,
#     ) -> dict:
#
#         fqdn = (
#             f"{model_name}-1"
#         )
#
#         request = pb2.EstInvokeRequest(
#
#             transaction_id=str(
#                 uuid.uuid4()
#             ),
#
#             fqdns=[
#                 fqdn
#             ],
#         )
#
#         query_start = (
#             time.perf_counter()
#         )
#
#         try:
#
#             raw_response = (
#                 self.raw_est_rpc(
#                     request
#                 )
#             )
#
#             query_end = (
#                 time.perf_counter()
#             )
#
#             estimates = (
#                 _decode_est_invoke_response(
#                     raw_response
#                 )
#             )
#
#             cpu_wait = float(
#                 estimates.get(
#                     "cpu_estimated_wait_time_sec",
#                     0.0,
#                 )
#             )
#
#             gpu_wait = float(
#                 estimates.get(
#                     "gpu_estimated_wait_time_sec",
#                     0.0,
#                 )
#             )
#
#             cpu_waiting = float(
#                 estimates.get(
#                     "cpu_waiting_functions",
#                     0.0,
#                 )
#             )
#
#             cpu_running = float(
#                 estimates.get(
#                     "cpu_running_functions",
#                     0.0,
#                 )
#             )
#
#             gpu_waiting = float(
#                 estimates.get(
#                     "gpu_waiting_functions",
#                     0.0,
#                 )
#             )
#
#             gpu_running = float(
#                 estimates.get(
#                     "gpu_running_functions",
#                     0.0,
#                 )
#             )
#
#             # ------------------------------------------------
#             # Convenience selected value.
#             #
#             # IMPORTANT:
#             # Raw CPU/GPU values are still always saved.
#             # ------------------------------------------------
#
#             if gpu_wait > 0.0:
#
#                 selected_wait = gpu_wait
#                 selected_compute = "gpu"
#
#             elif cpu_wait > 0.0:
#
#                 selected_wait = cpu_wait
#                 selected_compute = "cpu"
#
#             else:
#
#                 selected_wait = 0.0
#                 selected_compute = "none"
#
#             return {
#
#                 "fqdn":
#                     fqdn,
#
#                 "cpu_estimated_wait_time_sec":
#                     cpu_wait,
#
#                 "gpu_estimated_wait_time_sec":
#                     gpu_wait,
#
#                 "selected_estimated_wait_time_sec":
#                     selected_wait,
#
#                 "selected_compute":
#                     selected_compute,
#
#                 "cpu_waiting_functions":
#                     cpu_waiting,
#
#                 "cpu_running_functions":
#                     cpu_running,
#
#                 "gpu_waiting_functions":
#                     gpu_waiting,
#
#                 "gpu_running_functions":
#                     gpu_running,
#
#                 "estimator_query_ms":
#                     (
#                         query_end
#                         - query_start
#                     )
#                     * 1000.0,
#
#                 "estimator_response_bytes":
#                     len(
#                         raw_response
#                     ),
#
#                 "estimator_error":
#                     None,
#             }
#
#         except grpc.RpcError as error:
#
#             query_end = (
#                 time.perf_counter()
#             )
#
#             try:
#
#                 error_message = (
#                     error.details()
#                 )
#
#             except Exception:
#
#                 error_message = str(
#                     error
#                 )
#
#             return {
#
#                 "fqdn":
#                     fqdn,
#
#                 "cpu_estimated_wait_time_sec":
#                     None,
#
#                 "gpu_estimated_wait_time_sec":
#                     None,
#
#                 "selected_estimated_wait_time_sec":
#                     None,
#
#                 "selected_compute":
#                     None,
#
#                 "cpu_waiting_functions":
#                     None,
#
#                 "cpu_running_functions":
#                     None,
#
#                 "gpu_waiting_functions":
#                     None,
#
#                 "gpu_running_functions":
#                     None,
#
#                 "estimator_query_ms":
#                     (
#                         query_end
#                         - query_start
#                     )
#                     * 1000.0,
#
#                 "estimator_response_bytes":
#                     None,
#
#                 "estimator_error":
#                     error_message,
#             }
#
#         except Exception as error:
#
#             query_end = (
#                 time.perf_counter()
#             )
#
#             return {
#
#                 "fqdn":
#                     fqdn,
#
#                 "cpu_estimated_wait_time_sec":
#                     None,
#
#                 "gpu_estimated_wait_time_sec":
#                     None,
#
#                 "selected_estimated_wait_time_sec":
#                     None,
#
#                 "selected_compute":
#                     None,
#
#                 "cpu_waiting_functions":
#                     None,
#
#                 "cpu_running_functions":
#                     None,
#
#                 "gpu_waiting_functions":
#                     None,
#
#                 "gpu_running_functions":
#                     None,
#
#                 "estimator_query_ms":
#                     (
#                         query_end
#                         - query_start
#                     )
#                     * 1000.0,
#
#                 "estimator_response_bytes":
#                     None,
#
#                 "estimator_error":
#                     str(
#                         error
#                     ),
#             }
#
#     def close(
#         self,
#     ):
#
#         if self.owns_channel:
#
#             try:
#                 self.channel.close()
#
#             except Exception:
#                 pass




import struct
import time
import uuid

from typing import Dict, Optional, Tuple

import grpc

import ckn_controller.iluvatar_rpc_pb2 as pb2
import ckn_controller.iluvatar_rpc_pb2_grpc as pb2_grpc


# ============================================================
# Protobuf helpers
# ============================================================

def _read_varint(
    data: bytes,
    index: int,
) -> Tuple[int, int]:

    value = 0
    shift = 0

    while True:

        if index >= len(data):
            raise ValueError(
                "Unexpected end of protobuf data."
            )

        byte = data[index]
        index += 1

        value |= (
            byte & 0x7F
        ) << shift

        if not (
            byte & 0x80
        ):
            return value, index

        shift += 7

        if shift >= 64:
            raise ValueError(
                "Invalid protobuf varint."
            )


# ============================================================
# Parse protobuf map<string, double> entry
#
# Expected map entry:
#
# field 1 -> string key
# field 2 -> double value
# ============================================================

def _parse_map_entry(
    data: bytes,
) -> Optional[
    Tuple[str, float]
]:

    index = 0
    key = None

    # protobuf default double value
    value = 0.0

    try:

        while index < len(data):

            tag, index = _read_varint(
                data,
                index,
            )

            field_number = (
                tag >> 3
            )

            wire_type = (
                tag & 0x07
            )

            # ------------------------------------------------
            # Field 1 = string key
            # ------------------------------------------------

            if (
                field_number == 1
                and wire_type == 2
            ):

                length, index = _read_varint(
                    data,
                    index,
                )

                end = (
                    index + length
                )

                if end > len(data):
                    return None

                key = (
                    data[
                        index:end
                    ].decode(
                        "utf-8"
                    )
                )

                index = end

                continue

            # ------------------------------------------------
            # Field 2 = double value
            # ------------------------------------------------

            if (
                field_number == 2
                and wire_type == 1
            ):

                if (
                    index + 8
                    > len(data)
                ):
                    return None

                value = struct.unpack(
                    "<d",
                    data[
                        index:index + 8
                    ],
                )[0]

                index += 8

                continue

            # ------------------------------------------------
            # Skip unknown varint
            # ------------------------------------------------

            if wire_type == 0:

                _, index = _read_varint(
                    data,
                    index,
                )

                continue

            # ------------------------------------------------
            # Skip fixed64
            # ------------------------------------------------

            if wire_type == 1:

                if (
                    index + 8
                    > len(data)
                ):
                    return None

                index += 8

                continue

            # ------------------------------------------------
            # Skip length-delimited
            # ------------------------------------------------

            if wire_type == 2:

                length, index = _read_varint(
                    data,
                    index,
                )

                end = (
                    index + length
                )

                if end > len(data):
                    return None

                index = end

                continue

            # ------------------------------------------------
            # Skip fixed32
            # ------------------------------------------------

            if wire_type == 5:

                if (
                    index + 4
                    > len(data)
                ):
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
#
# Assumption:
#
# EstInvokeResponse:
#     field 1 -> map<string, double> est_times
#
# ============================================================

def _decode_est_invoke_response(
    raw_data: bytes,
) -> Dict[str, float]:

    if not raw_data:
        return {}

    index = 0

    results: Dict[
        str,
        float,
    ] = {}

    while index < len(raw_data):

        tag, index = _read_varint(
            raw_data,
            index,
        )

        field_number = (
            tag >> 3
        )

        wire_type = (
            tag & 0x07
        )

        # ----------------------------------------------------
        # est_times map = field 1
        # ----------------------------------------------------

        if (
            field_number == 1
            and wire_type == 2
        ):

            length, index = _read_varint(
                raw_data,
                index,
            )

            end = (
                index + length
            )

            if end > len(raw_data):

                raise ValueError(
                    "Invalid protobuf "
                    "map-entry length."
                )

            payload = (
                raw_data[
                    index:end
                ]
            )

            index = end

            entry = _parse_map_entry(
                payload
            )

            if entry is not None:

                key, value = entry

                results[
                    key
                ] = value

            continue

        # ----------------------------------------------------
        # Skip unknown fields
        # ----------------------------------------------------

        if wire_type == 0:

            _, index = _read_varint(
                raw_data,
                index,
            )

            continue

        if wire_type == 1:

            if (
                index + 8
                > len(raw_data)
            ):

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

            end = (
                index + length
            )

            if end > len(raw_data):

                raise ValueError(
                    "Invalid length-delimited field."
                )

            index = end

            continue

        if wire_type == 5:

            if (
                index + 4
                > len(raw_data)
            ):

                raise ValueError(
                    "Invalid fixed32 field."
                )

            index += 4

            continue

        raise ValueError(
            f"Unsupported protobuf "
            f"wire type: {wire_type}"
        )

    return results


# ============================================================
# Safely convert estimator value to float
# ============================================================

def _get_float_value(
    estimates: Dict[str, float],
    key: str,
) -> Optional[float]:

    value = estimates.get(
        key
    )

    if value is None:
        return None

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


# ============================================================
# Create raw estimator RPC
# ============================================================

def _get_raw_est_invoke_rpc(
    channel,
    normal_stub,
):

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
            "Could not determine "
            "est_invoke_time RPC path."
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

        request_serializer=(
            lambda request:
            request.SerializeToString()
        ),

        response_deserializer=(
            lambda data:
            data
        ),
    )


# ============================================================
# Estimator client
# ============================================================

class IluvatarWaitEstimator:

    def __init__(
        self,
        server_address: str,
        channel=None,
        debug: bool = False,
    ):

        self.server_address = (
            server_address
        )

        self.debug = (
            debug
        )

        self.owns_channel = (
            channel is None
        )

        if channel is None:

            self.channel = (
                grpc.insecure_channel(
                    server_address
                )
            )

        else:

            self.channel = (
                channel
            )

        self.normal_stub = (
            pb2_grpc.IluvatarWorkerStub(
                self.channel
            )
        )

        self.raw_est_rpc = (
            _get_raw_est_invoke_rpc(
                channel=self.channel,
                normal_stub=(
                    self.normal_stub
                ),
            )
        )


    # ========================================================
    # Query ONE model
    #
    # compute_type:
    #     "cpu"
    #     "gpu"
    #
    # Example:
    #
    # estimator.query(
    #     "resnet34",
    #     compute_type="cpu",
    # )
    #
    # estimator.query(
    #     "convnext_tiny",
    #     compute_type="gpu",
    # )
    #
    # ========================================================

    def query(
        self,
        model_name: str,
        compute_type: str,
    ) -> dict:

        compute_type = (
            compute_type
            .strip()
            .lower()
        )

        if compute_type not in {
            "cpu",
            "gpu",
        }:

            raise ValueError(
                "compute_type must be "
                "'cpu' or 'gpu'."
            )

        fqdn = (
            f"{model_name}-1"
        )

        request = (
            pb2.EstInvokeRequest(

                transaction_id=str(
                    uuid.uuid4()
                ),

                fqdns=[
                    fqdn
                ],
            )
        )

        query_start = (
            time.perf_counter()
        )

        try:

            # -----------------------------------------------
            # Raw estimator RPC
            # -----------------------------------------------

            raw_response = (
                self.raw_est_rpc(
                    request
                )
            )

            query_end = (
                time.perf_counter()
            )

            # -----------------------------------------------
            # Decode protobuf manually
            # -----------------------------------------------

            estimates = (
                _decode_est_invoke_response(
                    raw_response
                )
            )

            # -----------------------------------------------
            # Optional debug output
            # -----------------------------------------------

            if self.debug:

                print(
                    f"[EST_RAW] "
                    f"server={self.server_address} "
                    f"model={model_name} "
                    f"fqdn={fqdn} "
                    f"bytes={len(raw_response)} "
                    f"decoded={estimates}"
                )

            # -----------------------------------------------
            # Raw estimator values
            # -----------------------------------------------

            cpu_wait = _get_float_value(
                estimates,
                "cpu_estimated_wait_time_sec",
            )

            gpu_wait = _get_float_value(
                estimates,
                "gpu_estimated_wait_time_sec",
            )

            cpu_waiting = _get_float_value(
                estimates,
                "cpu_waiting_functions",
            )

            cpu_running = _get_float_value(
                estimates,
                "cpu_running_functions",
            )

            gpu_waiting = _get_float_value(
                estimates,
                "gpu_waiting_functions",
            )

            gpu_running = _get_float_value(
                estimates,
                "gpu_running_functions",
            )

            # -----------------------------------------------
            # Select wait based on actual compute type.
            #
            # IMPORTANT:
            # wait = 0.0 is valid.
            # It means no estimated queue wait.
            # -----------------------------------------------

            if compute_type == "gpu":

                selected_wait = (
                    gpu_wait
                )

                selected_compute = (
                    "gpu"
                )

            else:

                selected_wait = (
                    cpu_wait
                )

                selected_compute = (
                    "cpu"
                )

            # -----------------------------------------------
            # Detect missing selected wait
            # -----------------------------------------------

            estimator_error = None

            if selected_wait is None:

                estimator_error = (
                    f"Missing "
                    f"{selected_compute} "
                    f"estimated wait value "
                    f"in EstInvokeResponse."
                )

            # -----------------------------------------------
            # Result
            # -----------------------------------------------

            return {

                "fqdn":
                    fqdn,

                "server_address":
                    self.server_address,

                "requested_compute":
                    compute_type,

                "cpu_estimated_wait_time_sec":
                    cpu_wait,

                "gpu_estimated_wait_time_sec":
                    gpu_wait,

                "selected_estimated_wait_time_sec":
                    selected_wait,

                "selected_compute":
                    selected_compute,

                "cpu_waiting_functions":
                    cpu_waiting,

                "cpu_running_functions":
                    cpu_running,

                "gpu_waiting_functions":
                    gpu_waiting,

                "gpu_running_functions":
                    gpu_running,

                "estimator_query_ms":
                    (
                        query_end
                        - query_start
                    )
                    * 1000.0,

                "estimator_response_bytes":
                    len(
                        raw_response
                    ),

                "estimator_error":
                    estimator_error,

                # Useful for debugging
                "estimator_decoded":
                    estimates,
            }


        # ====================================================
        # gRPC error
        # ====================================================

        except grpc.RpcError as error:

            query_end = (
                time.perf_counter()
            )

            try:

                error_message = (
                    error.details()
                )

            except Exception:

                error_message = (
                    str(
                        error
                    )
                )

            return {

                "fqdn":
                    fqdn,

                "server_address":
                    self.server_address,

                "requested_compute":
                    compute_type,

                "cpu_estimated_wait_time_sec":
                    None,

                "gpu_estimated_wait_time_sec":
                    None,

                "selected_estimated_wait_time_sec":
                    None,

                "selected_compute":
                    compute_type,

                "cpu_waiting_functions":
                    None,

                "cpu_running_functions":
                    None,

                "gpu_waiting_functions":
                    None,

                "gpu_running_functions":
                    None,

                "estimator_query_ms":
                    (
                        query_end
                        - query_start
                    )
                    * 1000.0,

                "estimator_response_bytes":
                    None,

                "estimator_error":
                    error_message,

                "estimator_decoded":
                    None,
            }


        # ====================================================
        # Other error
        # ====================================================

        except Exception as error:

            query_end = (
                time.perf_counter()
            )

            return {

                "fqdn":
                    fqdn,

                "server_address":
                    self.server_address,

                "requested_compute":
                    compute_type,

                "cpu_estimated_wait_time_sec":
                    None,

                "gpu_estimated_wait_time_sec":
                    None,

                "selected_estimated_wait_time_sec":
                    None,

                "selected_compute":
                    compute_type,

                "cpu_waiting_functions":
                    None,

                "cpu_running_functions":
                    None,

                "gpu_waiting_functions":
                    None,

                "gpu_running_functions":
                    None,

                "estimator_query_ms":
                    (
                        query_end
                        - query_start
                    )
                    * 1000.0,

                "estimator_response_bytes":
                    None,

                "estimator_error":
                    str(
                        error
                    ),

                "estimator_decoded":
                    None,
            }


    # ========================================================
    # Close channel
    # ========================================================

    def close(
        self,
    ):

        if self.owns_channel:

            try:

                self.channel.close()

            except Exception:

                pass