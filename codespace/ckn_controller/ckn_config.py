# SERVER_ADDRESS = "149.165.168.72:8079"

# Model and policy settings
M_TOTAL = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",

]

POLICY = "greedy"  # greedy  or "randomized"
K = 12  # total cores
C = 2   # cores per model
ALPHA = 1
ETA = 0.1

COLD_PENALTY = {
    "mobilenet_v3_small": 30,
    "mobilenet_v2": 30,
    "shufflenet_v2_x1_0": 25,
    "resnet18": 40,
    "resnet34": 50,
    "resnet50": 60,
    "resnet101": 70,
    "efficientnet_b3": 70,
    "convnext_tiny": 75,
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
    # small / CPU-side models
    "shufflenet_v2_x1_0": {"latency": 0.009500, "accuracy": 0.690000},
    "mobilenet_v3_small": {"latency": 0.010597, "accuracy": 0.704413581114262},
    "mobilenet_v2":       {"latency": 0.010000, "accuracy": 0.720000},
    "resnet18":           {"latency": 0.010618, "accuracy": 0.712173486609011},
    "resnet34":           {"latency": 0.016935, "accuracy": 0.775743676409125},

    # large / GPU-side models
    "resnet50":           {"latency": 0.019834, "accuracy": 0.797511352837085},
    "resnet101":          {"latency": 0.033677, "accuracy": 0.808838205507397},
    "efficientnet_b3":    {"latency": 0.035000, "accuracy": 0.820000},
    "convnext_tiny":      {"latency": 0.040000, "accuracy": 0.830000},
    "vit_b_16":           {"latency": 0.048291, "accuracy": 0.751089387536048},
}

OMEGA = {
    "mobilenet_v3_small": 0.25,
    "mobilenet_v2": 0.25,
    "shufflenet_v2_x1_0": 0.25,
    "resnet18": 0.117,
    "resnet34": 0.218,
    "resnet50": 0.256,
    "resnet101": 0.445,
    "efficientnet_b3": 0.45,
    "convnext_tiny": 0.45,
    "vit_b_16": 0.864,
}

# SERVER_ADDRESS = "149.165.175.242:8079"

MAX_MODEL_SIZE=3

MODEL_WEIGHTS = {}
GAMMA = 0.9
RHO=1.0001



DEFAULT_WEIGHTS = {
    "mobilenet_v3_small": 1.0,
    "mobilenet_v2": 1.0,
    "shufflenet_v2_x1_0": 1.0,
    "resnet18": 1.0,
    "resnet34": 1.0,
    "resnet50": 1.0,
    "resnet101": 1.0,
    "efficientnet_b3": 1.0,
    "convnext_tiny": 1.0,
    "vit_b_16": 1.0,
}
WEIGHTS_STATE_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/ckn_controller/ckn_weights.json"


MODEL_SIZES = {
    "shufflenet_v2_x1_0": 8900000,
    "mobilenet_v3_small": 3200000,
    "mobilenet_v2": 14000000,
    "resnet18": 45000000,
    "resnet34": 83000000,
    "resnet50": 102000000,
    "resnet101": 171000000,
    "efficientnet_b3": 48000000,
    "convnext_tiny": 114000000,
    "vit_b_16": 330000000,
}



# --- Cascaded ensemble parameters ---
ENSEMBLE_SIZE = 3       # always select exactly this many models (if feasible)
THRESHOLD = 0.85         # early-exit confidence threshold


THRESHOLD_STAGE = []

# --- Alpha mode ---
ALPHA_MODE = "adaptive"   # "adaptive" or "static"
ALPHA_STATIC = 0.5       # used only if ALPHA_MODE="static"

# --- Adaptive alpha range ---
ALPHA_MIN = 0.1          # high load -> prioritize latency
ALPHA_MAX = 0.9          # low load  -> prioritize accuracy


# --- Load estimation (QPS sliding window) ---
QPS_WINDOW_SEC = 5.0     # how far back to count requests for QPS
QPS_LOW = 1             # below this, treat as low load
QPS_HIGH = 10           # above this, treat as high load


# # --- Wait-time normalization (Ilúvatar congestion) ---
# WAIT_LOW = 0.01          # seconds (10ms): low congestion
# WAIT_HIGH = 0.2          # seconds (200ms): high congestion


# # -----------------------------
# # Iluvatar instance mode
# # -----------------------------
# USE_TWO_ILUVATAR_INSTANCES = True
#
# # Single-instance fallback
# SERVER_ADDRESS = "149.165.175.242:8079"
#
# # Dual-instance addresses
# SMALL_MODEL_SERVER_ADDRESS = "149.165.175.242:8081"
# LARGE_MODEL_SERVER_ADDRESS = "149.165.174.241:8070"
#
# # Manual routing
#
# SMALL_INSTANCE_MODELS = [
#     "mobilenet_v3_small",
#     "mobilenet_v2",
#     "shufflenet_v2_x1_0",
#     "resnet18",
#     "resnet34",
# ]
#
# LARGE_INSTANCE_MODELS = [
#     "resnet50",
#     "resnet101",
#     "efficientnet_b3",
#     "convnext_tiny",
#     "vit_b_16",
# ]


# ============================================================
# Multi-edge and multi-cloud Iluvatar configuration
# ============================================================

# Enable the new N-edge / N-cloud architecture.
USE_MULTI_ILUVATAR_INSTANCES = True

# Disable the old fixed two-instance routing.
# Keep this variable because older parts of the code still import it.
USE_TWO_ILUVATAR_INSTANCES = False


# ============================================================
# Single-instance fallback
# ============================================================

# Used only when both multi-node and two-instance modes are disabled.
SERVER_ADDRESS = "129.79.242.28:8079"


# ============================================================
# Legacy two-instance values
# ============================================================

# These are retained for backward compatibility with files that
# still import these variables.
SMALL_MODEL_SERVER_ADDRESS = "149.165.175.242:8079"
LARGE_MODEL_SERVER_ADDRESS = "149.165.174.241:8070"




ILUVATAR_NODES = {
    "edge_1": {
        "type": "edge",
        "device": "cpu",
        "address": "129.79.242.28:8079",
        "enabled": True,
        # Controller <-> Iluvatar RPC/network baseline in seconds
        "rpc_baseline_s": 0.0,

        "models": [
            "mobilenet_v3_small",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "resnet18",
            "resnet34",
        ],

        "default_memory_mb": 512,
        "default_cpus": 3,
        "default_parallel_invokes": 1,

        "image_prefix": "docker.io/plalelab/ckn_faas_",
        # "image_suffix": "-iluvatar-gpu-http:latest",
        "image_suffix": "-iluvatar-action-http:latest",
    },

    "edge_2": {
        "type": "edge",
        "device": "cpu",
        "address": "149.165.175.242:8081",
        "enabled": False,
        # Controller <-> Iluvatar RPC/network baseline in seconds
        "rpc_baseline_s": 0.0,

        "models": [
            "mobilenet_v3_small",
            "mobilenet_v2",
            "shufflenet_v2_x1_0",
            "resnet18",
            "resnet34",
        ],

        "default_memory_mb": 512,
        "default_cpus": 3,
        "default_parallel_invokes": 1,

        "image_prefix": "docker.io/plalelab/ckn_faas_",
        "image_suffix": "-iluvatar-action-http:latest",
    },

    "cloud_1": {
        "type": "cloud",
        "device": "cpu",
        "address": "129.79.242.28:8079",
        "enabled": True,
        # Controller <-> Iluvatar RPC/network baseline in seconds
        "rpc_baseline_s":  0.05,

        "models": [
            "resnet50",
            "resnet101",
            "efficientnet_b3",
            "convnext_tiny",
            "vit_b_16",
        ],

        "default_memory_mb": 1024,
        "default_cpus": 3,
        "default_parallel_invokes": 1,

        "image_prefix": "docker.io/plalelab/ckn_faas_",
        # "image_suffix": "-iluvatar-gpu-http:latest",
        "image_suffix": "-iluvatar-action-http:latest",

        "model_resources": {
            "vit_b_16": {
                "memory_mb": 2048,
                "cpus": 3,
                "parallel_invokes": 1,
            },
        },
    },

    "cloud_2": {
        "type": "cloud",
        "device": "gpu",
        "address": "149.165.174.241:8071",
        "enabled": False,
        # Controller <-> Iluvatar RPC/network baseline in seconds
        "rpc_baseline_s": 0.05,

        "models": [
            "resnet50",
            "resnet101",
            "efficientnet_b3",
            "convnext_tiny",
            "vit_b_16",
        ],

        "default_memory_mb": 1024,
        "default_cpus": 3,
        "default_parallel_invokes": 1,

        "image_prefix": "docker.io/plalelab/ckn_faas_",
        # "image_suffix": "-iluvatar-action-http:latest",
        "image_suffix": "-iluvatar-gpu-http:latest",

    },
}



SMALL_INSTANCE_MODELS = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
]

LARGE_INSTANCE_MODELS = [
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
]

M_TOTAL = [
    "mobilenet_v3_small",
    "mobilenet_v2",
    "shufflenet_v2_x1_0",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "efficientnet_b3",
    "convnext_tiny",
    "vit_b_16",
]

# -----------------------------
# Instance-aware ensemble size
# -----------------------------
SMALL_ENSEMBLE_SIZE = 2   # CPU / small instance models
LARGE_ENSEMBLE_SIZE = 1   # GPU / large instance models

# -----------------------------
# Baseline mode
# -----------------------------
# options:
# "none"
# "smallest_to_largest"
# "random_to_random"
# "smallest_to_ensemble_3"
# "power2_smallest_to_largest"
# "spec_b_smallest_to_largest"
# "spec_b_random_to_random"
# "full_ensemble_edge"
# "full_ensemble_cloud"
BASELINE_MODE = "none"

# # -----------------------------
# # Cascade execution mode
# # -----------------------------
# # "sequential"
# # "parallel_first"
# # "full_parallel"
# CASCADE_MODE = "full_parallel"

PARALLEL_FIRST_N = 2   # run first N models in parallel at the beginning
PARALLEL_BATCH_SIZE = 2   # keep 1 for sequential after first batch


# -----------------------------
# Cascade mode
# -----------------------------
# options: "mode_s", "sequential", "parallel_first_finish"
CASCADE_MODE = "sequential"

# -----------------------------
# Selection strategy
# Only used for sequential and parallel_first_finish
# options: "greedy", "exhaustive"
# -----------------------------
SELECTION_STRATEGY = "exhaustive"

# -----------------------------
# Deadline fast-path
# If True: if selected set estimated latency >= deadline,
# fallback to fastest model
# -----------------------------
USE_DEADLINE_FAST_PATH = False

# -----------------------------
# Network latency between stages (seconds)
# -----------------------------
NETWORK_LATENCY = 0.05



# BASE_REQUEST_OVERHEAD = 0.1      # fixed overhead per request
# SEQUENTIAL_STAGE_OVERHEAD = 0.30  # extra overhead when continuing to next stage
# PO2_FIRST_BATCH_OVERHEAD = 0.5   # overhead for launching/checking first batch
# PO2_TAIL_OVERHEAD = 0.8          # overhead before continuing to tail
# FULL_PARALLEL_AGG_OVERHEAD = 0.02



# -----------------------------
# E[T] overhead terms
# -----------------------------
BASE_REQUEST_OVERHEAD = 0.04
SEQUENTIAL_STAGE_OVERHEAD = 0.01
PO2_FIRST_BATCH_OVERHEAD = 0.005
PO2_TAIL_OVERHEAD = 0.01
FULL_PARALLEL_AGG_OVERHEAD = 0.005

# -----------------------------
# E[T] calibration terms
# -----------------------------
ET_SCALE_SEQ = 1.0
ET_OFFSET_SEQ = 0.02

ET_SCALE_P2 = 1.4
ET_OFFSET_P2 = 0.02


ENABLE_RISK_STRATEGY_SELECTION = False

RISK_ALPHA_LATENCY = 0.4
RISK_BETA_COST = 0.2
RISK_GAMMA_ERROR = 0.4

HARDWARE_COST_RATE = {
    "cpu": 1.0,
    "gpu": 4.0,
}



# USE_SEQ_CONSTRAINT_SOLVER = True
# SEQ_CASCADE_PROFILE_CSV = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/cascade_parallel_newset_results.csv"
#
# SEQ_MAX_COST = 0.55
# SEQ_TARGET_ACCURACY = 0.85



USE_CONSTRAINT_SOLVER = False

CONSTRAINT_SOLVER_TIME_MODE = "live_et"
# "offline" = use CSV expected latency
# "live_et" = use current E[T] / wait estimates


# C=0.45, A=0.80
# C=0.55, A=0.82
# C=0.70, A=0.85
# C=0.85, A=0.88

# C=0.90, A=0.90 ['mobilenet_v3_small', 'vit_b_16']

# C=0.50, A=0.50 ['mobilenet_v3_small', 'resnet50']
# C=0.50, A=0.70 same ['mobilenet_v3_small', 'resnet50']
# C=0.50, A=0.70 same ['mobilenet_v3_small', 'resnet50']

# C=0.80, A=0.80 ['shufflenet_v2_x1_0', 'vit_b_16'] / ['mobilenet_v3_small', 'vit_b_16']

CONSTRAINT_MAX_COST = 0.80
CONSTRAINT_TARGET_ACCURACY = 0.88

CONSTRAINT_POLICY_ORDER = [
    "sequential",
    # "power2",
    # "full_parallel",
]

CONSTRAINT_SEQ_PROFILE_CSV = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/"
    "codespace/workload_generator/data/cascade_parallel_newset_results.csv"
)

CONSTRAINT_POWER2_PROFILE_CSV = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/"
    "codespace/workload_generator/data/power2_profile.csv"
)

CONSTRAINT_FULL_PARALLEL_PROFILE_CSV = (
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/"
    "codespace/workload_generator/data/full_parallel_profile.csv"
)



# -----------------------------
# Static optimal config mode
# -----------------------------
# True  = use  table values from optimal_configs.py
# False = use baseline / normal MODE-S / constraint solver depending on flags
USE_STATIC_OPTIMAL_CONFIG = True

# Pick one C,E pair from the table
# C = 35, 50, 100
# E = 0.090, 0.100, 0.120, 0.140
STATIC_COST = 50
STATIC_ERROR = 0.120

# options:
# "sequential"
# "po2"
# "spec_b"
STATIC_POLICY = "spec_b"


# -----------------------------
# Manual cloud hit-rate routing
# -----------------------------
USE_MANUAL_CLOUD_HIT_RATE = True
MANUAL_CLOUD_HIT_RATE = 0.70