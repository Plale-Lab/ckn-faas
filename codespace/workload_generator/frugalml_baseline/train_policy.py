from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    CANDIDATE_MODELS,
    get_model_cost,
)

from policy import (
    normalize_label,
)


THRESHOLD_GRID = [
    round(
        value,
        2,
    )
    for value in np.arange(
        0.0,
        1.0001,
        0.05,
    )
]


def build_sample_map(
    dataframe: pd.DataFrame,
) -> dict[
    str,
    dict[str, Any],
]:
    samples: dict[
        str,
        dict[str, Any],
    ] = {}

    for row in dataframe.itertuples(
        index=False
    ):
        sample_id = str(
            row.sample_id
        )

        samples.setdefault(
            sample_id,
            {
                "ground_truth": (
                    row.ground_truth
                ),
                "models": {},
            },
        )

        samples[
            sample_id
        ]["models"][
            str(row.model)
        ] = {
            "predicted_label": (
                row.predicted_label
            ),
            "confidence": float(
                row.confidence
            ),
            "correct": int(
                row.correct
            ),
        }

    return samples


def simulate_global_policy(
    *,
    samples: dict[
        str,
        dict[str, Any],
    ],
    base_model: str,
    addon_model: str,
    threshold: float,
) -> dict[str, float]:
    total = 0
    correct = 0
    total_cost = 0.0
    deferred = 0

    for sample in (
        samples.values()
    ):
        model_results = (
            sample["models"]
        )

        if (
            base_model
            not in model_results
            or addon_model
            not in model_results
        ):
            continue

        base_result = (
            model_results[
                base_model
            ]
        )

        total += 1

        total_cost += (
            get_model_cost(
                base_model
            )
        )

        if (
            float(
                base_result[
                    "confidence"
                ]
            )
            >= threshold
        ):
            correct += int(
                base_result[
                    "correct"
                ]
            )
        else:
            deferred += 1

            total_cost += (
                get_model_cost(
                    addon_model
                )
            )

            correct += int(
                model_results[
                    addon_model
                ]["correct"]
            )

    if total == 0:
        return {
            "accuracy": 0.0,
            "average_cost": (
                float("inf")
            ),
            "deferral_rate": 0.0,
            "count": 0,
        }

    return {
        "accuracy": (
            correct / total
        ),
        "average_cost": (
            total_cost / total
        ),
        "deferral_rate": (
            deferred / total
        ),
        "count": total,
    }


def select_global_configuration(
    *,
    samples: dict[
        str,
        dict[str, Any],
    ],
    target_accuracy: float,
    max_average_cost: float,
) -> dict[str, Any]:
    feasible: list[
        dict[str, Any]
    ] = []

    available_models = [
        model
        for model
        in CANDIDATE_MODELS
        if any(
            model in sample["models"]
            for sample
            in samples.values()
        )
    ]

    for base_model in (
        available_models
    ):
        for addon_model in (
            available_models
        ):
            if (
                addon_model
                == base_model
            ):
                continue

            for threshold in (
                THRESHOLD_GRID
            ):
                result = (
                    simulate_global_policy(
                        samples=samples,
                        base_model=(
                            base_model
                        ),
                        addon_model=(
                            addon_model
                        ),
                        threshold=(
                            threshold
                        ),
                    )
                )

                if (
                    result["accuracy"]
                    >= target_accuracy
                    and result[
                        "average_cost"
                    ]
                    <= max_average_cost
                ):
                    feasible.append(
                        {
                            "base_model": (
                                base_model
                            ),
                            "addon_model": (
                                addon_model
                            ),
                            "threshold": (
                                threshold
                            ),
                            **result,
                        }
                    )

    if not feasible:
        raise RuntimeError(
            "No feasible FrugalML "
            "configuration was found. "
            "Check the target accuracy, "
            "cost budget, model costs, "
            "and calibration data."
        )

    feasible.sort(
        key=lambda item: (
            item[
                "average_cost"
            ],
            -item[
                "accuracy"
            ],
            item[
                "deferral_rate"
            ],
        )
    )

    return feasible[0]


def learn_class_policies(
    *,
    samples: dict[
        str,
        dict[str, Any],
    ],
    base_model: str,
    target_accuracy: float,
    max_average_cost: float,
    minimum_class_samples: int,
) -> dict[
    str,
    dict[str, Any],
]:
    predicted_classes = set()

    for sample in (
        samples.values()
    ):
        base_result = (
            sample["models"].get(
                base_model
            )
        )

        if base_result is not None:
            predicted_classes.add(
                normalize_label(
                    base_result[
                        "predicted_label"
                    ]
                )
            )

    available_models = [
        model
        for model
        in CANDIDATE_MODELS
        if any(
            model in sample["models"]
            for sample
            in samples.values()
        )
    ]

    class_policies: dict[
        str,
        dict[str, Any],
    ] = {}

    for predicted_class in sorted(
        predicted_classes
    ):
        class_samples = []

        for sample in (
            samples.values()
        ):
            base_result = (
                sample["models"].get(
                    base_model
                )
            )

            if base_result is None:
                continue

            if (
                normalize_label(
                    base_result[
                        "predicted_label"
                    ]
                )
                == predicted_class
            ):
                class_samples.append(
                    sample
                )

        if (
            len(class_samples)
            < minimum_class_samples
        ):
            continue

        candidates: list[
            dict[str, Any]
        ] = []

        for addon_model in (
            available_models
        ):
            if (
                addon_model
                == base_model
            ):
                continue

            for threshold in (
                THRESHOLD_GRID
            ):
                total = 0
                correct = 0
                total_cost = 0.0
                deferred = 0

                for sample in (
                    class_samples
                ):
                    model_results = (
                        sample[
                            "models"
                        ]
                    )

                    if (
                        addon_model
                        not in model_results
                    ):
                        continue

                    base_result = (
                        model_results[
                            base_model
                        ]
                    )

                    total += 1

                    total_cost += (
                        get_model_cost(
                            base_model
                        )
                    )

                    if (
                        float(
                            base_result[
                                "confidence"
                            ]
                        )
                        >= threshold
                    ):
                        correct += int(
                            base_result[
                                "correct"
                            ]
                        )
                    else:
                        deferred += 1

                        total_cost += (
                            get_model_cost(
                                addon_model
                            )
                        )

                        correct += int(
                            model_results[
                                addon_model
                            ]["correct"]
                        )

                if total == 0:
                    continue

                accuracy = (
                    correct / total
                )

                average_cost = (
                    total_cost / total
                )

                deferral_rate = (
                    deferred / total
                )

                if (
                    accuracy
                    >= target_accuracy
                    and average_cost
                    <= max_average_cost
                ):
                    candidates.append(
                        {
                            "threshold": (
                                threshold
                            ),
                            "addon_model": (
                                addon_model
                            ),
                            "accuracy": (
                                accuracy
                            ),
                            "average_cost": (
                                average_cost
                            ),
                            "deferral_rate": (
                                deferral_rate
                            ),
                            "sample_count": (
                                total
                            ),
                        }
                    )

        if not candidates:
            continue

        candidates.sort(
            key=lambda item: (
                item[
                    "average_cost"
                ],
                -item[
                    "accuracy"
                ],
                item[
                    "deferral_rate"
                ],
            )
        )

        best = candidates[0]

        class_policies[
            predicted_class
        ] = {
            "threshold": float(
                best["threshold"]
            ),
            "addon_model": str(
                best[
                    "addon_model"
                ]
            ),
        }

    return class_policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--error",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--max-average-cost",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--minimum-class-samples",
        type=int,
        default=20,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    dataframe = pd.read_csv(
        args.csv
    )

    required_columns = {
        "sample_id",
        "ground_truth",
        "model",
        "predicted_label",
        "confidence",
        "correct",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Calibration CSV is "
            "missing columns: "
            f"{sorted(missing_columns)}"
        )

    samples = build_sample_map(
        dataframe
    )

    error_constraint = round(
        float(args.error),
        3,
    )

    target_accuracy = (
        1.0 - error_constraint
    )

    global_config = (
        select_global_configuration(
            samples=samples,
            target_accuracy=(
                target_accuracy
            ),
            max_average_cost=float(
                args.max_average_cost
            ),
        )
    )

    class_policies = (
        learn_class_policies(
            samples=samples,
            base_model=(
                global_config[
                    "base_model"
                ]
            ),
            target_accuracy=(
                target_accuracy
            ),
            max_average_cost=float(
                args.max_average_cost
            ),
            minimum_class_samples=int(
                args.minimum_class_samples
            ),
        )
    )

    policy = {
        "error_constraint": (
            error_constraint
        ),
        "target_accuracy": (
            target_accuracy
        ),
        "base_model": (
            global_config[
                "base_model"
            ]
        ),
        "default_addon_model": (
            global_config[
                "addon_model"
            ]
        ),
        "default_threshold": float(
            global_config[
                "threshold"
            ]
        ),
        "class_policies": (
            class_policies
        ),
        "metadata": {
            "policy_type": (
                "standalone_frugalml"
            ),
            "training_samples": (
                len(samples)
            ),
            "maximum_average_cost": float(
                args.max_average_cost
            ),
            "minimum_class_samples": int(
                args.minimum_class_samples
            ),
            "global_accuracy": float(
                global_config[
                    "accuracy"
                ]
            ),
            "global_average_cost": float(
                global_config[
                    "average_cost"
                ]
            ),
            "global_deferral_rate": float(
                global_config[
                    "deferral_rate"
                ]
            ),
            "class_policy_count": (
                len(class_policies)
            ),
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            policy,
            file,
            indent=2,
            sort_keys=True,
        )

    print(
        json.dumps(
            policy,
            indent=2,
            sort_keys=True,
        )
    )

    print(
        "Saved FrugalML policy "
        f"to: {args.output}"
    )


if __name__ == "__main__":
    main()