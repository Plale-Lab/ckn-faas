# SERVER_ADDRESS = "149.165.168.72:8079"

# Model and policy settings
M_TOTAL = [
    "mobilenet_v3_small",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "vit_b_16"
]

POLICY = "greedy"  # greedy  or "randomized"
K = 12  # total cores
C = 2   # cores per model
ALPHA = 1
ETA = 0.1

COLD_PENALTY = {
    "mobilenet_v3_small": 30,
    "resnet18": 40,
    "resnet34": 50,
    "resnet50": 60,
    "resnet101": 70,
    "vit_b_16": 80,
}


# MODEL_PROFILES = {
#     "mobilenet_v3_small": {"latency": 0.09597, "accuracy": 0.704413581114262},
#     "resnet18": {"latency": 0.10618, "accuracy": 0.712173486609011},
#     "resnet34": {"latency": 0.16935, "accuracy": 0.775743676409125},
#     "resnet50": {"latency": 0.19834, "accuracy": 0.797511352837085},
#     "resnet101": {"latency": 0.33677, "accuracy": 0.808838205507397},
#     "vit_b_16": {"latency": 0.48291, "accuracy": 0.751089387536048},
# }


MODEL_PROFILES = {
    "mobilenet_v3_small": {"latency": 0.009597, "accuracy": 0.704413581114262},
    "resnet18": {"latency": 0.010618, "accuracy": 0.712173486609011},
    "resnet34": {"latency": 0.016935, "accuracy": 0.775743676409125},
    "resnet50": {"latency": 0.019834, "accuracy": 0.797511352837085},
    "resnet101": {"latency": 0.033677, "accuracy": 0.808838205507397},
    "vit_b_16": {"latency": 0.048291, "accuracy": 0.751089387536048},
}


OMEGA = {
    "mobilenet_v3_small": 0.25,
    "resnet18": 0.117,
    "resnet34": 0.218,
    "resnet50": 0.256,
    "resnet101": 0.445,
    "vit_b_16": 0.864,
}

SERVER_ADDRESS = "149.165.152.35:8079"

MAX_MODEL_SIZE=4

MODEL_WEIGHTS = {}
GAMMA = 0.9
RHO=1.0001

DEFAULT_WEIGHTS = {
    "mobilenet_v3_small": 1.0,
    "resnet18": 1.0,
    "resnet34": 1.0,
    "resnet50": 1.0,
    "resnet101": 1.0,
    "vit_b_16": 1.0,
}
WEIGHTS_STATE_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/ckn_controller/ckn_weights.json"


MODEL_SIZES = {
    "mobilenet_v3_small": 3200000,
    "resnet18":           45000000,
    "resnet34":           83000000,
    "resnet50":           102000000,
    "resnet101":          171000000,
    "vit_b_16":           330000000,
}
