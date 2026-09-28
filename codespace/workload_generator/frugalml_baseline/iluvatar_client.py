from __future__ import annotations

import json
import uuid
from typing import Any

import grpc

import iluvatar_rpc_pb2 as pb2
import iluvatar_rpc_pb2_grpc as pb2_grpc

from ckn_controller.ckn_config import (
    SERVER_ADDRESS,
    USE_TWO_ILUVATAR_INSTANCES,
    SMALL_MODEL_SERVER_ADDRESS,
    LARGE_MODEL_SERVER_ADDRESS,
    SMALL_INSTANCE_MODELS,
    LARGE_INSTANCE_MODELS,
)

from policy import (
    get_class_decision,
    get_policy_models,
    normalize_label,
)


def get_server_address_for_model(
    model_name: str,
) -> str:
    """
    Return the Iluvatar worker address for one model.
    """

    if not USE_TWO_ILUVATAR_INSTANCES:
        return SERVER_ADDRESS

    if model_name in SMALL_INSTANCE_MODELS:
        return SMALL_MODEL_SERVER_ADDRESS

    if model_name in LARGE_INSTANCE_MODELS:
        return LARGE_MODEL_SERVER_ADDRESS

    raise ValueError(
        f"Model {model_name!r} is not assigned "
        f"to either Iluvatar instance. "
        f"SMALL_INSTANCE_MODELS="
        f"{SMALL_INSTANCE_MODELS}; "
        f"LARGE_INSTANCE_MODELS="
        f"{LARGE_INSTANCE_MODELS}"
    )


def build_stub_map_for_selected_models(
    model_list: list[str],
):
    """
    Build one gRPC channel per unique Iluvatar worker.

    stub_map is indexed by server address, matching your DYSCI code.
    """

    addresses = {
        get_server_address_for_model(
            model
        )
        for model in model_list
    }

    channel_map: dict[
        str,
        Any,
    ] = {}

    stub_map: dict[
        str,
        Any,
    ] = {}

    for address in addresses:
        channel = (
            grpc.aio.insecure_channel(
                address
            )
        )

        stub = (
            pb2_grpc
            .IluvatarWorkerStub(
                channel
            )
        )

        channel_map[
            address
        ] = channel

        stub_map[
            address
        ] = stub

    return (
        channel_map,
        stub_map,
    )


async def close_channels(
    channel_map: dict[
        str,
        Any,
    ],
) -> None:
    for channel in (
        channel_map.values()
    ):
        try:
            await channel.close()
        except Exception:
            pass


def get_stub_for_model(
    model_name: str,
    stub_map: dict[
        str,
        Any,
    ],
):
    address = (
        get_server_address_for_model(
            model_name
        )
    )

    if address not in stub_map:
        raise KeyError(
            f"No Iluvatar stub exists for "
            f"model={model_name}, "
            f"address={address}"
        )

    return stub_map[
        address
    ]


async def invoke_iluvatar_function(
    *,
    stub: Any,
    model_name: str,
    image_b64: str,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    """
    Invoke one model function already deployed in Iluvatar.
    """

    request = pb2.InvokeRequest(
        function_name=model_name,
        function_version="1",
        json_args=json.dumps(
            {
                "model_name": model_name,
                "image_data": image_b64,
            }
        ),
        transaction_id=(
            transaction_id
            or str(uuid.uuid4())
        ),
    )

    response = await stub.invoke(
        request
    )

    result_json = json.loads(
        response.json_result
    )

    body = result_json.get(
        "body",
        {},
    )

    return {
        "model": model_name,
        "label": body.get(
            "Prediction Class"
        ),
        "probability": float(
            body.get(
                "Probability",
                0.0,
            )
        ),
        "latency": (
            float(
                response.duration_us
            )
            / 1_000_000.0
        ),
        "success": bool(
            response.success
        ),
        "container_state": (
            pb2.ContainerState.Name(
                response.container_state
            )
        ),
        "server_address": (
            get_server_address_for_model(
                model_name
            )
        ),
    }


async def execute_frugalml_request(
    *,
    request_id: str,
    image_b64: str,
    policy: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
]:
    """
    Execute one FrugalML request.

    Workflow:
        1. Invoke base model.
        2. Read predicted label and confidence.
        3. Retrieve class-specific threshold and add-on model.
        4. Return base result if confidence is sufficient.
        5. Otherwise invoke add-on model sequentially.

    FrugalML does not perform speculation.
    """

    policy_models = (
        get_policy_models(
            policy
        )
    )

    channel_map, stub_map = (
        build_stub_map_for_selected_models(
            policy_models
        )
    )

    results: list[
        dict[str, Any]
    ] = []

    try:
        base_model = str(
            policy[
                "base_model"
            ]
        )

        base_stub = (
            get_stub_for_model(
                base_model,
                stub_map,
            )
        )

        base_result = (
            await invoke_iluvatar_function(
                stub=base_stub,
                model_name=base_model,
                image_b64=image_b64,
                transaction_id=(
                    f"{request_id}-base"
                ),
            )
        )

        results.append(
            base_result
        )

        (
            threshold,
            addon_model,
            used_class_policy,
        ) = get_class_decision(
            base_result.get(
                "label"
            ),
            policy,
        )

        base_confidence = float(
            base_result.get(
                "probability",
                0.0,
            )
        )

        decision: dict[
            str,
            Any,
        ] = {
            "base_model": (
                base_model
            ),
            "base_label": (
                base_result.get(
                    "label"
                )
            ),
            "normalized_base_label": (
                normalize_label(
                    base_result.get(
                        "label"
                    )
                )
            ),
            "base_confidence": (
                base_confidence
            ),
            "threshold": float(
                threshold
            ),
            "addon_model": (
                addon_model
            ),
            "used_class_policy": bool(
                used_class_policy
            ),
            "deferred": False,
            "decision": None,
        }

        if not base_result.get(
            "success",
            False,
        ):
            decision[
                "decision"
            ] = "base_failed"

            return (
                results,
                decision,
            )

        if (
            base_confidence
            >= threshold
        ):
            decision[
                "decision"
            ] = "accept_base"

            return (
                results,
                decision,
            )

        decision[
            "deferred"
        ] = True

        decision[
            "decision"
        ] = "defer_to_addon"

        addon_stub = (
            get_stub_for_model(
                addon_model,
                stub_map,
            )
        )

        addon_result = (
            await invoke_iluvatar_function(
                stub=addon_stub,
                model_name=addon_model,
                image_b64=image_b64,
                transaction_id=(
                    f"{request_id}-addon"
                ),
            )
        )

        results.append(
            addon_result
        )

        decision[
            "addon_label"
        ] = addon_result.get(
            "label"
        )

        decision[
            "addon_confidence"
        ] = float(
            addon_result.get(
                "probability",
                0.0,
            )
        )

        decision[
            "addon_success"
        ] = bool(
            addon_result.get(
                "success",
                False,
            )
        )

        return (
            results,
            decision,
        )

    finally:
        await close_channels(
            channel_map
        )


def choose_final_result(
    results: list[
        dict[str, Any]
    ],
    decision: dict[
        str,
        Any,
    ],
) -> dict[str, Any]:
    """
    FrugalML returns the base result when accepted.

    If the request was deferred, it returns the add-on result.
    It does not use DYSCI weighted-majority aggregation.
    """

    if not results:
        return {
            "model": None,
            "label": -1,
            "confidence": 0.0,
            "success": False,
            "combiner_policy": (
                "frugalml_failed"
            ),
        }

    base_result = results[0]

    if not decision.get(
        "deferred",
        False,
    ):
        if not base_result.get(
            "success",
            False,
        ):
            return {
                "model": (
                    base_result.get(
                        "model"
                    )
                ),
                "label": -1,
                "confidence": 0.0,
                "success": False,
                "combiner_policy": (
                    "frugalml_base_failed"
                ),
            }

        return {
            "model": (
                base_result.get(
                    "model"
                )
            ),
            "label": (
                base_result.get(
                    "label"
                )
            ),
            "confidence": float(
                base_result.get(
                    "probability",
                    0.0,
                )
            ),
            "success": True,
            "combiner_policy": (
                "frugalml_base"
            ),
        }

    if len(results) >= 2:
        addon_result = results[-1]

        if addon_result.get(
            "success",
            False,
        ):
            return {
                "model": (
                    addon_result.get(
                        "model"
                    )
                ),
                "label": (
                    addon_result.get(
                        "label"
                    )
                ),
                "confidence": float(
                    addon_result.get(
                        "probability",
                        0.0,
                    )
                ),
                "success": True,
                "combiner_policy": (
                    "frugalml_addon"
                ),
            }

    # If the add-on request fails, use the successful base result.
    if base_result.get(
        "success",
        False,
    ):
        return {
            "model": (
                base_result.get(
                    "model"
                )
            ),
            "label": (
                base_result.get(
                    "label"
                )
            ),
            "confidence": float(
                base_result.get(
                    "probability",
                    0.0,
                )
            ),
            "success": True,
            "combiner_policy": (
                "frugalml_base_fallback"
            ),
        }

    return {
        "model": None,
        "label": -1,
        "confidence": 0.0,
        "success": False,
        "combiner_policy": (
            "frugalml_failed"
        ),
    }