from collections import Counter, defaultdict

from ckn_controller.label_utils import name_from_wnid


def combine_outputs(results, policy="most_confident", historical_acc=None, model_weights=None, ground_truth=None, gamma=0.9, rho=1, update_weights=True, label_matcher=None):
    """
    Combine outputs from multiple models.

    results:  list of dictionaries returned by send_request
    policy: selection strategy
    historical_acc: mapping of model_name -> historical accuracy (used in 'top1_accuracy' case)
    """

    if policy == "top1_accuracy":
        best_model = max(results, key=lambda r: historical_acc.get(r["model"], 0.0))
        return {
            "label": best_model["label"],
            "accuracy": historical_acc.get(best_model["model"], 0.0),
            "success": True,
            "combiner_policy": policy,
        }

    elif policy == "most_confident":
        best_model = max(results, key=lambda r: r["probability"])
        return {
            "label": best_model["label"],
            "accuracy": best_model["probability"],
            "success": best_model["success"],
            "combiner_policy": policy,
        }

    # elif policy == "majority":
    #     labels = [r["label"] for r in results]
    #     label, count = Counter(labels).most_common(1)[0]
    #     avg_prob = sum(r["probability"] for r in results if r["label"] == label) / count
    #     return {
    #         "label": label,
    #         "accuracy": avg_prob,
    #         "success": True,
    #         "combiner_policy": policy,
    #     }
    elif policy == "majority":
        labels = [r["label"] for r in results]
        counts = Counter(labels)
        max_count = max(counts.values())

        tied_labels = [label for label, count in counts.items() if count == max_count]

        if len(tied_labels) == 1:
            label = tied_labels[0]
        else:

            avg_probs = {
                l: sum(r["probability"] for r in results if r["label"] == l) / counts[l]
                for l in tied_labels
            }
            label = max(avg_probs, key=avg_probs.get)

        avg_prob = sum(r["probability"] for r in results if r["label"] == label) / counts[label]

        return {
            "label": label,
            "accuracy": avg_prob,
            "success": True,
            "combiner_policy": policy,
        }


    elif policy == "weighted_majority":
        weights = defaultdict(float)
        for r in results:
            weights[r["label"]] += r["probability"]

        label = max(weights, key=weights.get)
        total_weight = sum(weights.values())
        accuracy = weights[label] / total_weight if total_weight > 0 else 0.0

        return {
            "label": label,
            "accuracy":  accuracy,
            "success": True,
            "combiner_policy": policy,
        }


    elif policy == "online_weighted":
        if model_weights is None:
            model_weights = {r["model"]: 1.0 for r in results}
        else:
            for r in results:
                model_weights.setdefault(r["model"], 1.0)

        prev_weights = {m: float(w) for m, w in model_weights.items()}

        def _is_correct(r):
            if ground_truth is None:
                return False
            if label_matcher:
                return label_matcher(ground_truth, r["label"])
            return r["label"] == ground_truth

        correct = [r for r in results if _is_correct(r)] if ground_truth is not None else []

        # --- choose best model (prob * weight)
        if correct:
            chosen = max(correct, key=lambda r: r["probability"] * model_weights[r["model"]])
            y_hat_id = ground_truth  # keep WNID for metrics
            y_hat_text = chosen["label"]  # human-readable predicted text
        else:
            chosen = max(results, key=lambda r: r["probability"] * model_weights[r["model"]])
            y_hat_id = None  # unknown WNID here
            y_hat_text = chosen["label"]

        p_hat = float(chosen["probability"])
        chosen_model = chosen["model"]

        # --- update weights (reduce wrong ones)
        if ground_truth is not None and update_weights:
            for r in results:
                m = r["model"]
                if not _is_correct(r):
                    model_weights[m] *= float(gamma)  # penalize wrong
                else:
                    model_weights[m] *= float(rho)  # reward correct

                # Clamp each model’s weight between 0.1 and 1.0
                model_weights[m] = max(0.1, min(1.0, model_weights[m]))

        new_weights = {m: float(w) for m, w in model_weights.items()}

        # --- pretty names for printing
        gt_name = name_from_wnid(ground_truth) if ground_truth is not None else "N/A"

        print("\n====================== ONLINE WEIGHTED UPDATE ======================")
        print(f"Ground Truth (WNID) : {ground_truth}")
        print(f"Ground Truth (Name) : {gt_name}")
        print(f"Predicted (Text)    : {y_hat_text}")
        if y_hat_id is not None:
            print(f"Predicted (WNID)    : {y_hat_id}")
        print(f"Chosen Model        : {chosen_model}")
        print(f"Chosen Probability  : {p_hat:.4f}")

        print("\nPrevious Weights:")
        for m, w in prev_weights.items():
            print(f"  {m:<20}: {w:.3f}")

        print("\nUpdated Weights:")
        for m, w in new_weights.items():
            print(f"  {m:<20}: {w:.3f}")
        print("=====================================================================\n")

        # success: rely on matcher (robust to synonyms)
        success_flag = bool(ground_truth is None or _is_correct(chosen))

        return {
            "label": y_hat_text,  # <--- HUMAN label returned
            "accuracy": p_hat,
            "success": success_flag,
            "combiner_policy": policy,
            # "chosen_model": chosen_model,
        }



    else:
        raise ValueError(f"Unknown policy: {policy}")
