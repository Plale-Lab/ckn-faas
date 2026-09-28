from pathlib import Path


# ============================================================
# Dataset and local FrugalML controller
# ============================================================

BASE_FOLDER = (
    "/Users/agamage/Desktop/D2I/Codes Original/"
    "Mode-S/archive/train.X1"
)

REMOTE_CKN_URL = "http://127.0.0.1:9000/invoke"

FRUGALML_HOST = "127.0.0.1"
FRUGALML_PORT = 9000

FRUGALML_POLICY_FILE = Path(__file__).resolve().parent / (
    "frugalml_policy.json"
)


# ============================================================
# Iluvatar instances
# ============================================================

SMALL_SERVER_URL = "http://149.165.175.242:8079"
LARGE_SERVER_URL = "http://149.165.174.213:8070"

MODEL_INVOKE_PATH = "/invoke"

SMALL_MODELS = {
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
}

LARGE_MODELS = {
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
}


def get_server_address_for_model(model_name: str) -> str:
    if model_name in SMALL_MODELS:
        return SMALL_SERVER_URL

    if model_name in LARGE_MODELS:
        return LARGE_SERVER_URL

    raise ValueError(
        f"Model '{model_name}' is not listed in "
        "SMALL_MODELS or LARGE_MODELS."
    )


# ============================================================
# Workload profiles
# ============================================================

workload_profiles = {
    "short": "short",
    "medium": "medium",
    "long": "long",
}

deadline_profiles_ms = {
    "short": [1000],
    "medium": [500, 700, 1000, 2000, 6000],
    "long": [500, 700, 1000, 2000, 4000, 6000],
}

iar_profiles_ms = {
    "short": [2000, 1000, 500, 200],
    "medium": [2000, 1000, 750, 500, 200, 100],
    "long": [2000, 1000, 750, 500, 200, 100, 50, 10],
}