"""
DYSCI-side latency estimator.

Purpose
-------
Estimate per-model invocation latency from the application side without
depending entirely on Iluvatar's internal wait-time estimator.

DYSCI tracks requests that have been dispatched but whose responses have
not yet been received. Therefore the application-visible outstanding state
naturally includes requests that may still be:

    - travelling to Iluvatar
    - waiting inside Iluvatar
    - executing
    - returning to DYSCI

The estimator uses two application-visible load signals:

    O = number of outstanding requests
    W = total baseline work of outstanding requests

Historical latency is learned for:

    (node, device, model, O_bucket, W_bucket)

The estimator can be switched between:

    ESTIMATOR_SOURCE = "dysci"
    ESTIMATOR_SOURCE = "iluvatar"
"""

import time
import threading

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ckn_config import ILUVATAR_NODES


# ============================================================
# Configuration
# ============================================================

ESTIMATOR_SOURCE = "dysci"

# Query Iluvatar for logging/comparison even when DYSCI
# is the primary estimator.
QUERY_ILUVATAR_FOR_DEBUG = True


# ============================================================
# EWMA configuration
# ============================================================

# Kept as the default/fallback value.
EWMA_ALPHA = 0.40

# Number of observations required before a historical state
# is considered sufficiently sampled.
MIN_SAMPLES = 3


# ============================================================
# Outstanding-count bucket configuration
# ============================================================

MAX_LOAD_BUCKET = 12


# ============================================================
# Outstanding-work bucket configuration
# ============================================================

# 50 ms of baseline outstanding work per bucket.
#
# Examples:
#
#   0 - 49 ms     -> bucket 0
#   50 - 99 ms    -> bucket 1
#   100 - 149 ms  -> bucket 2
#   150 - 199 ms  -> bucket 3
#
WORK_BUCKET_SIZE_S = 0.050

# Collapse extremely high work values into one final bucket.
MAX_WORK_BUCKET = 20


# ============================================================
# Cold-start configuration
# ============================================================

# Conservative increase to model service time when the node
# is loaded but useful history does not yet exist.
COLD_LOAD_MARGIN = 0.10


# ============================================================
# Outstanding request representation
# ============================================================

@dataclass
class OutstandingRequest:

    request_id: str

    node_id: str
    node_type: str
    device: str
    model: str

    # Time immediately before invocation.
    dispatch_time: float

    # Baseline service time from MODEL_PROFILES.
    baseline_service_s: float

    # DYSCI-visible state immediately BEFORE this request
    # was inserted into the outstanding set.
    load_at_dispatch: int

    work_at_dispatch_s: float


# ============================================================
# DYSCI latency estimator
# ============================================================

class DYSCILatencyEstimator:

    def __init__(
        self,
        model_profiles: Dict,
    ):

        self.model_profiles = model_profiles

        self.ewma_alpha = float(
            EWMA_ALPHA
        )

        self.min_samples = int(
            MIN_SAMPLES
        )

        self.max_load_bucket = int(
            MAX_LOAD_BUCKET
        )

        self.work_bucket_size_s = float(
            WORK_BUCKET_SIZE_S
        )

        self.max_work_bucket = int(
            MAX_WORK_BUCKET
        )

        self.cold_load_margin = float(
            COLD_LOAD_MARGIN
        )

        self._lock = threading.RLock()


        # ====================================================
        # Outstanding requests
        # ====================================================
        #
        # request_id -> OutstandingRequest
        #

        self.outstanding: Dict[
            str,
            OutstandingRequest
        ] = {}


        # ====================================================
        # Outstanding request IDs per node/device
        # ====================================================
        #
        # (node_id, device) -> set(request_id)
        #

        self.outstanding_by_node_device = (
            defaultdict(set)
        )


        # ====================================================
        # Historical observed invocation latency
        # ====================================================
        #
        # Key:
        #
        # (
        #     node_id,
        #     device,
        #     model,
        #     load_bucket,
        #     work_bucket,
        # )
        #
        # Value:
        #
        #     EWMA of actual dispatch -> response latency
        #
        # This observed latency already includes:
        #
        #     RPC/network
        #     worker overhead
        #     queueing/contention
        #     model execution
        #     response RPC/network
        #

        self.history_ewma: Dict[
            Tuple[str, str, str, int, int],
            float
        ] = {}

        self.history_count = defaultdict(int)


    # ========================================================
    # Basic helpers
    # ========================================================

    def _baseline_service(
        self,
        model: str,
    ) -> float:
        """
        Return baseline model service time.

        Unit: seconds.
        """

        if model not in self.model_profiles:

            raise KeyError(
                f"Unknown model '{model}' "
                "in MODEL_PROFILES"
            )

        return float(
            self.model_profiles[
                model
            ]["latency"]
        )


    def _node_rpc_baseline(
        self,
        node_id: str,
    ) -> float:
        """
        Return configured controller <-> Iluvatar
        RPC/network baseline for this node.

        Unit: seconds.

        This is used only when learned E2E history
        does not yet exist.
        """

        node_config = ILUVATAR_NODES.get(
            str(node_id),
            {},
        )

        return float(
            node_config.get(
                "rpc_baseline_s",
                0.0,
            )
        )


    # ========================================================
    # Outstanding-count bucket
    # ========================================================

    def _bucket(
        self,
        outstanding_count: int,
    ) -> int:
        """
        Convert outstanding request count into a bounded
        load bucket.
        """

        return min(
            max(
                0,
                int(outstanding_count),
            ),
            self.max_load_bucket,
        )


    # ========================================================
    # Outstanding-work bucket
    # ========================================================

    def _work_bucket(
        self,
        work_s: float,
    ) -> int:
        """
        Convert outstanding baseline work into a bucket.

        Example with 50-ms buckets:

            W = 0.030 s -> bucket 0
            W = 0.080 s -> bucket 1
            W = 0.120 s -> bucket 2
            W = 0.180 s -> bucket 3

        W is a load indicator.

        It is NOT interpreted as exact queueing delay.
        """

        work_s = max(
            0.0,
            float(work_s),
        )

        if work_s <= 0.0:
            return 0

        bucket = int(
            work_s
            / self.work_bucket_size_s
        )

        return min(
            bucket,
            self.max_work_bucket,
        )


    # ========================================================
    # Adaptive EWMA alpha
    # ========================================================

    def _alpha_for_load(
        self,
        load: int,
    ) -> float:
        """
        Select EWMA alpha according to outstanding count.

        Lower-load states are allowed to react more quickly
        because their latency tends to be more stable.

        High-load states retain more historical smoothing.
        """

        if load <= 1:
            return 0.60

        elif load <= 3:
            return 0.50

        else:
            return 0.40


    # ========================================================
    # Historical key
    # ========================================================

    def _history_key(
        self,
        node_id: str,
        device: str,
        model: str,
        load: int,
        work_s: float,
    ) -> Tuple[
        str,
        str,
        str,
        int,
        int,
    ]:
        """
        Build the historical-state key.

        History is now indexed by:

            node
            device
            model
            outstanding request count O
            outstanding work bucket W
        """

        return (
            str(node_id),
            str(device).lower(),
            str(model),
            self._bucket(load),
            self._work_bucket(work_s),
        )


    # ========================================================
    # Current application-visible load
    # ========================================================

    def get_outstanding_count(
        self,
        node_id: str,
        device: str,
    ) -> int:

        key = (
            str(node_id),
            str(device).lower(),
        )

        with self._lock:

            return len(
                self.outstanding_by_node_device[
                    key
                ]
            )


    def get_outstanding_work(
        self,
        node_id: str,
        device: str,
    ) -> float:
        """
        Calculate total baseline work of all unresolved
        requests on this node/device.

        W = sum(S_r)

        where S_r is the baseline model service time.

        IMPORTANT:

        W is NOT exact queueing delay.

        It is an application-visible indicator of how much
        model work is currently outstanding.
        """

        key = (
            str(node_id),
            str(device).lower(),
        )

        with self._lock:

            total_s = 0.0

            for request_id in (
                self.outstanding_by_node_device[
                    key
                ]
            ):

                req = self.outstanding.get(
                    request_id
                )

                if req is not None:

                    total_s += (
                        req.baseline_service_s
                    )

            return total_s


    # ========================================================
    # Request lifecycle: start
    # ========================================================

    def request_started(
        self,
        request_id: str,
        node_id: str,
        node_type: str,
        device: str,
        model: str,
    ) -> dict:
        """
        Call immediately BEFORE stub.invoke().

        The load state recorded here describes the system
        immediately before the new request is inserted.

        Therefore:

            load_at_dispatch = O
            work_at_dispatch = W
        """

        now = time.perf_counter()

        node_id = str(node_id)
        node_type = str(node_type)
        device = str(device).lower()
        model = str(model)

        with self._lock:

            # ------------------------------------------------
            # State BEFORE inserting this request
            # ------------------------------------------------

            load_before = (
                self.get_outstanding_count(
                    node_id,
                    device,
                )
            )

            work_before_s = (
                self.get_outstanding_work(
                    node_id,
                    device,
                )
            )

            baseline_s = (
                self._baseline_service(
                    model
                )
            )


            # ------------------------------------------------
            # Create request record
            # ------------------------------------------------

            req = OutstandingRequest(
                request_id=request_id,
                node_id=node_id,
                node_type=node_type,
                device=device,
                model=model,
                dispatch_time=now,
                baseline_service_s=baseline_s,
                load_at_dispatch=load_before,
                work_at_dispatch_s=work_before_s,
            )


            # ------------------------------------------------
            # Add to outstanding state
            # ------------------------------------------------

            self.outstanding[
                request_id
            ] = req

            self.outstanding_by_node_device[
                (
                    node_id,
                    device,
                )
            ].add(
                request_id
            )


            return {

                "outstanding_before":
                    load_before,

                "outstanding_after":
                    load_before + 1,

                "work_before_s":
                    work_before_s,

                "work_after_s":
                    work_before_s
                    + baseline_s,

                "work_bucket_before":
                    self._work_bucket(
                        work_before_s
                    ),

                "baseline_service_s":
                    baseline_s,
            }


    # ========================================================
    # Request lifecycle: finish
    # ========================================================

    def request_finished(
        self,
        request_id: str,
        learn: bool = True,
    ) -> Optional[dict]:
        """
        Remove request from outstanding state.

        If learn=True:

            measure actual dispatch -> response latency

            update EWMA for the state observed when the
            request was originally dispatched.

        Failed/cancelled requests should use:

            learn=False
        """

        finish_time = time.perf_counter()

        with self._lock:

            req = self.outstanding.pop(
                request_id,
                None,
            )

            if req is None:
                return None


            # ------------------------------------------------
            # Remove from outstanding state
            # ------------------------------------------------

            node_device_key = (
                req.node_id,
                req.device,
            )

            self.outstanding_by_node_device[
                node_device_key
            ].discard(
                request_id
            )


            # ------------------------------------------------
            # Actual observed invocation latency
            #
            # dispatch -> response
            # ------------------------------------------------

            actual_s = max(
                0.0,
                finish_time
                - req.dispatch_time,
            )


            # ------------------------------------------------
            # Historical state corresponding to conditions
            # BEFORE this request was dispatched.
            # ------------------------------------------------

            history_key = (
                self._history_key(
                    req.node_id,
                    req.device,
                    req.model,
                    req.load_at_dispatch,
                    req.work_at_dispatch_s,
                )
            )


            # ------------------------------------------------
            # Learn successful invocation
            # ------------------------------------------------

            if learn:

                old_value = (
                    self.history_ewma.get(
                        history_key
                    )
                )

                if old_value is None:

                    # First observation for this state.
                    new_value = actual_s

                else:

                    alpha = (
                        self._alpha_for_load(
                            req.load_at_dispatch
                        )
                    )

                    # EWMA:
                    #
                    # E_new =
                    #     (1-alpha) E_old
                    #     + alpha T_actual
                    #

                    new_value = (
                        (1.0 - alpha)
                        * old_value
                        + alpha
                        * actual_s
                    )


                self.history_ewma[
                    history_key
                ] = new_value

                self.history_count[
                    history_key
                ] += 1


            return {

                "actual_s":
                    actual_s,

                "load_at_dispatch":
                    req.load_at_dispatch,

                "work_at_dispatch_s":
                    req.work_at_dispatch_s,

                "work_bucket_at_dispatch":
                    self._work_bucket(
                        req.work_at_dispatch_s
                    ),

                "remaining_outstanding":
                    self.get_outstanding_count(
                        req.node_id,
                        req.device,
                    ),

                "remaining_work_s":
                    self.get_outstanding_work(
                        req.node_id,
                        req.device,
                    ),

                "history_ewma_s":
                    self.history_ewma.get(
                        history_key
                    ),

                "history_samples":
                    self.history_count.get(
                        history_key,
                        0,
                    ),
            }


    # ========================================================
    # Exact historical lookup
    # ========================================================

    def _get_exact_history(
        self,
        node_id: str,
        device: str,
        model: str,
        load: int,
        work_s: float,
    ) -> Optional[dict]:

        key = self._history_key(
            node_id,
            device,
            model,
            load,
            work_s,
        )

        value = (
            self.history_ewma.get(
                key
            )
        )

        samples = (
            self.history_count.get(
                key,
                0,
            )
        )


        if (
            value is not None
            and samples >= self.min_samples
        ):

            return {

                "value_s":
                    float(value),

                "samples":
                    int(samples),

                "load":
                    self._bucket(
                        load
                    ),

                "work_bucket":
                    self._work_bucket(
                        work_s
                    ),
            }

        return None


    # ========================================================
    # Nearby historical lookup
    # ========================================================

    def _get_nearby_history(
        self,
        node_id: str,
        device: str,
        model: str,
        load: int,
        work_s: float,
    ) -> Optional[dict]:
        """
        Find the nearest sufficiently sampled historical state.

        Distance considers:

            O = outstanding request count
            W = outstanding work bucket

        Outstanding-count difference is given somewhat more
        importance than work-bucket difference.
        """

        node_id = str(
            node_id
        )

        device = str(
            device
        ).lower()

        model = str(
            model
        )


        target_load = (
            self._bucket(
                load
            )
        )

        target_work = (
            self._work_bucket(
                work_s
            )
        )


        candidates = []


        for key, value in (
            self.history_ewma.items()
        ):

            (
                hist_node,
                hist_device,
                hist_model,
                hist_load,
                hist_work,
            ) = key


            # ------------------------------------------------
            # Node/device/model must match
            # ------------------------------------------------

            if hist_node != node_id:
                continue

            if hist_device != device:
                continue

            if hist_model != model:
                continue


            samples = (
                self.history_count.get(
                    key,
                    0,
                )
            )


            if samples < self.min_samples:
                continue


            # ------------------------------------------------
            # Distance in outstanding request count
            # ------------------------------------------------

            load_distance = abs(
                hist_load
                - target_load
            )


            # ------------------------------------------------
            # Distance in outstanding work
            # ------------------------------------------------

            work_distance = abs(
                hist_work
                - target_work
            )


            # ------------------------------------------------
            # Combined state distance
            #
            # Give request-count difference higher weight.
            # ------------------------------------------------

            distance = (
                2 * load_distance
                + work_distance
            )


            candidates.append(
                (
                    distance,
                    load_distance,
                    work_distance,
                    hist_load,
                    hist_work,
                    float(value),
                    int(samples),
                )
            )


        if not candidates:
            return None


        # ----------------------------------------------------
        # Prefer smallest combined distance.
        #
        # Tie breaking:
        #     smaller load distance
        #     smaller work distance
        # ----------------------------------------------------

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
                item[2],
            )
        )


        (
            distance,
            load_distance,
            work_distance,
            hist_load,
            hist_work,
            value_s,
            samples,
        ) = candidates[0]


        return {

            "value_s":
                value_s,

            "samples":
                samples,

            "load":
                hist_load,

            "work_bucket":
                hist_work,

            "distance":
                distance,

            "load_distance":
                load_distance,

            "work_distance":
                work_distance,
        }


    # ========================================================
    # Prediction
    # ========================================================

    def estimate(
        self,
        node_id: str,
        node_type: str,
        device: str,
        model: str,
    ) -> dict:
        """
        Predict per-model invocation E2E latency.

        Prediction hierarchy:

            1. Idle learned history
            2. Idle profile + RPC baseline
            3. Exact O/W historical state
            4. Nearby O/W historical state
            5. Cold-load fallback
        """

        node_id = str(
            node_id
        )

        node_type = str(
            node_type
        )

        device = str(
            device
        ).lower()

        model = str(
            model
        )


        with self._lock:

            # ------------------------------------------------
            # Current outstanding request count
            # ------------------------------------------------

            outstanding = (
                self.get_outstanding_count(
                    node_id,
                    device,
                )
            )


            # ------------------------------------------------
            # Current outstanding baseline work
            # ------------------------------------------------

            outstanding_work_s = (
                self.get_outstanding_work(
                    node_id,
                    device,
                )
            )


            # ------------------------------------------------
            # Candidate model baseline service time
            # ------------------------------------------------

            baseline_s = (
                self._baseline_service(
                    model
                )
            )


            current_work_bucket = (
                self._work_bucket(
                    outstanding_work_s
                )
            )


            # =================================================
            # Case 1: Idle
            # =================================================

            if outstanding == 0:

                # --------------------------------------------
                # Try learned idle history first.
                #
                # For idle:
                #
                #     O = 0
                #     W = 0
                # --------------------------------------------

                exact = (
                    self._get_exact_history(
                        node_id=node_id,
                        device=device,
                        model=model,
                        load=0,
                        work_s=0.0,
                    )
                )


                if exact is not None:

                    estimated_e2e_s = float(
                        exact["value_s"]
                    )


                    return {

                        "estimated_e2e_s":
                            estimated_e2e_s,

                        "estimated_wait_s":
                            max(
                                0.0,
                                estimated_e2e_s
                                - baseline_s,
                            ),

                        "estimated_service_s":
                            baseline_s,

                        # Learned E2E already contains RPC.
                        "estimated_rpc_s":
                            0.0,

                        "source":
                            "idle_history",

                        "outstanding":
                            0,

                        "outstanding_work_s":
                            0.0,

                        "work_bucket":
                            0,

                        "history_samples":
                            exact["samples"],
                    }


                # --------------------------------------------
                # No learned idle history yet.
                #
                # Initial estimate:
                #
                # baseline service + node RPC baseline
                # --------------------------------------------

                rpc_baseline_s = (
                    self._node_rpc_baseline(
                        node_id
                    )
                )


                estimated_e2e_s = (
                    baseline_s
                    + rpc_baseline_s
                )


                return {

                    "estimated_e2e_s":
                        estimated_e2e_s,

                    "estimated_wait_s":
                        0.0,

                    "estimated_service_s":
                        baseline_s,

                    "estimated_rpc_s":
                        rpc_baseline_s,

                    "source":
                        "idle_profile_with_rpc",

                    "outstanding":
                        0,

                    "outstanding_work_s":
                        0.0,

                    "work_bucket":
                        0,

                    "history_samples":
                        0,
                }


            # =================================================
            # Case 2: Exact O/W historical state
            # =================================================

            exact = (
                self._get_exact_history(
                    node_id=node_id,
                    device=device,
                    model=model,
                    load=outstanding,
                    work_s=outstanding_work_s,
                )
            )


            if exact is not None:

                estimated_e2e_s = float(
                    exact["value_s"]
                )


                return {

                    "estimated_e2e_s":
                        estimated_e2e_s,

                    # Diagnostic only.
                    # Not directly measured queue time.
                    "estimated_wait_s":
                        max(
                            0.0,
                            estimated_e2e_s
                            - baseline_s,
                        ),

                    "estimated_service_s":
                        baseline_s,

                    "estimated_rpc_s":
                        0.0,

                    "source":
                        "exact_history",

                    "outstanding":
                        outstanding,

                    "outstanding_work_s":
                        outstanding_work_s,

                    "work_bucket":
                        current_work_bucket,

                    "history_samples":
                        exact["samples"],
                }


            # =================================================
            # Case 3: Nearby O/W historical state
            # =================================================

            nearby = (
                self._get_nearby_history(
                    node_id=node_id,
                    device=device,
                    model=model,
                    load=outstanding,
                    work_s=outstanding_work_s,
                )
            )


            if nearby is not None:

                estimated_e2e_s = float(
                    nearby["value_s"]
                )


                return {

                    "estimated_e2e_s":
                        estimated_e2e_s,

                    "estimated_wait_s":
                        max(
                            0.0,
                            estimated_e2e_s
                            - baseline_s,
                        ),

                    "estimated_service_s":
                        baseline_s,

                    "estimated_rpc_s":
                        0.0,

                    "source":
                        (
                            "nearby_history_"
                            f"load_{nearby['load']}_"
                            f"work_{nearby['work_bucket']}"
                        ),

                    "outstanding":
                        outstanding,

                    "outstanding_work_s":
                        outstanding_work_s,

                    "work_bucket":
                        current_work_bucket,

                    "history_samples":
                        nearby["samples"],

                    "history_distance":
                        nearby["distance"],

                    "history_load_distance":
                        nearby["load_distance"],

                    "history_work_distance":
                        nearby["work_distance"],
                }


            # =================================================
            # Case 4: Cold-load fallback
            # =================================================
            #
            # No sufficiently sampled exact/nearby history.
            #
            # W is used only as a conservative load signal.
            #
            # It is NOT treated as exact queueing delay.
            # =================================================

            loaded_service_s = (
                baseline_s
                * (
                    1.0
                    + self.cold_load_margin
                )
            )


            # ------------------------------------------------
            # Because requests may execute concurrently,
            # use only part of outstanding work as the
            # bootstrap penalty.
            # ------------------------------------------------

            bootstrap_load_penalty_s = (
                0.5
                * outstanding_work_s
            )


            estimated_e2e_s = (
                loaded_service_s
                + bootstrap_load_penalty_s
            )


            return {

                "estimated_e2e_s":
                    estimated_e2e_s,

                "estimated_wait_s":
                    bootstrap_load_penalty_s,

                "estimated_service_s":
                    loaded_service_s,

                "estimated_rpc_s":
                    0.0,

                "source":
                    "cold_load_fallback",

                "outstanding":
                    outstanding,

                "outstanding_work_s":
                    outstanding_work_s,

                "work_bucket":
                    current_work_bucket,

                "history_samples":
                    0,
            }


    # ========================================================
    # Debug snapshot
    # ========================================================

    def snapshot(
        self,
    ) -> dict:

        with self._lock:

            outstanding = {

                (
                    node_id,
                    device,
                ):
                    len(
                        request_ids
                    )

                for (
                    node_id,
                    device
                ), request_ids

                in (
                    self.outstanding_by_node_device.items()
                )
            }


            history = {

                key: {

                    "ewma_s":
                        self.history_ewma[
                            key
                        ],

                    "samples":
                        self.history_count.get(
                            key,
                            0,
                        ),
                }

                for key
                in self.history_ewma
            }


            return {

                "outstanding":
                    outstanding,

                "history":
                    history,
            }


# ============================================================
# Estimator-source selector
# ============================================================

def choose_estimate(
    dysci_estimate_s: float,
    iluvatar_estimate_s: float,
) -> Tuple[float, str]:
    """
    Select which estimator drives routing.

    ESTIMATOR_SOURCE = "dysci"
        -> use DYSCI

    ESTIMATOR_SOURCE = "iluvatar"
        -> use Iluvatar

    If Iluvatar is selected but returns an invalid value,
    fall back to DYSCI.
    """

    source = str(
        ESTIMATOR_SOURCE
    ).strip().lower()


    dysci_estimate_s = float(
        dysci_estimate_s
    )

    iluvatar_estimate_s = float(
        iluvatar_estimate_s
    )


    # ========================================================
    # Iluvatar selected
    # ========================================================

    if source == "iluvatar":

        if (
            iluvatar_estimate_s
            != float("inf")
            and iluvatar_estimate_s >= 0.0
        ):

            return (
                iluvatar_estimate_s,
                "iluvatar",
            )


        # Iluvatar failed -> use DYSCI.
        return (
            dysci_estimate_s,
            "dysci_fallback",
        )


    # ========================================================
    # Validate configuration
    # ========================================================

    if source != "dysci":

        raise ValueError(
            "ESTIMATOR_SOURCE must be "
            "'dysci' or 'iluvatar', "
            f"got '{ESTIMATOR_SOURCE}'"
        )


    # ========================================================
    # DYSCI selected
    # ========================================================

    return (
        dysci_estimate_s,
        "dysci",
    )