from typing import Dict, List, Tuple


def get_model_accuracy(model_name: str, model_profiles: Dict, omega: Dict) -> float:
    if "accuracy" in model_profiles[model_name]:
        return float(model_profiles[model_name]["accuracy"])
    return float(omega.get(model_name, 0.5))


def get_model_hardware(
    model_name: str,
    small_instance_models: List[str],
    large_instance_models: List[str],
) -> str:
    if model_name in small_instance_models:
        return "cpu"
    if model_name in large_instance_models:
        return "gpu"
    return "cpu"


def get_model_exec_time_s(model_name: str, model_profiles: Dict) -> float:
    return float(model_profiles[model_name]["latency"])


def get_model_cost(
    model_name: str,
    model_profiles: Dict,
    small_instance_models: List[str],
    large_instance_models: List[str],
    hardware_cost_rate: Dict,
) -> float:
    exec_s = get_model_exec_time_s(model_name, model_profiles)
    hardware = get_model_hardware(
        model_name,
        small_instance_models,
        large_instance_models,
    )
    rate = float(hardware_cost_rate.get(hardware, 1.0))
    return exec_s * rate


def estimate_full_cost(models, model_profiles, small_instance_models, large_instance_models, hardware_cost_rate):
    return sum(
        get_model_cost(m, model_profiles, small_instance_models, large_instance_models, hardware_cost_rate)
        for m in models
    )


def estimate_sequential_cost(models, model_profiles, omega, small_instance_models, large_instance_models, hardware_cost_rate):
    total = 0.0
    continue_prob = 1.0

    for m in models:
        total += continue_prob * get_model_cost(
            m, model_profiles, small_instance_models, large_instance_models, hardware_cost_rate
        )

        p_exit = get_model_accuracy(m, model_profiles, omega)
        continue_prob *= (1.0 - p_exit)

    return total


def estimate_power2_cost(models, parallel_first_n, model_profiles, omega, small_instance_models, large_instance_models, hardware_cost_rate):
    first_n = max(1, min(int(parallel_first_n), len(models)))
    first_batch = models[:first_n]
    tail = models[first_n:]

    total = estimate_full_cost(
        first_batch,
        model_profiles,
        small_instance_models,
        large_instance_models,
        hardware_cost_rate,
    )

    fail_prob = 1.0
    for m in first_batch:
        p_exit = get_model_accuracy(m, model_profiles, omega)
        fail_prob *= (1.0 - p_exit)

    continue_prob = fail_prob

    for m in tail:
        total += continue_prob * get_model_cost(
            m, model_profiles, small_instance_models, large_instance_models, hardware_cost_rate
        )

        p_exit = get_model_accuracy(m, model_profiles, omega)
        continue_prob *= (1.0 - p_exit)

    return total


def estimate_full_accuracy(models, model_profiles, omega):
    return max(get_model_accuracy(m, model_profiles, omega) for m in models)


def estimate_sequential_accuracy(models, model_profiles, omega):
    expected_acc = 0.0
    continue_prob = 1.0

    for m in models:
        acc = get_model_accuracy(m, model_profiles, omega)
        p_exit = acc

        expected_acc += continue_prob * p_exit * acc
        continue_prob *= (1.0 - p_exit)

    if models:
        expected_acc += continue_prob * get_model_accuracy(models[-1], model_profiles, omega)

    return expected_acc


def estimate_power2_accuracy(models, parallel_first_n, model_profiles, omega):
    first_n = max(1, min(int(parallel_first_n), len(models)))
    first_batch = models[:first_n]
    tail = models[first_n:]

    batch_acc = max(get_model_accuracy(m, model_profiles, omega) for m in first_batch)

    fail_prob = 1.0
    for m in first_batch:
        p_exit = get_model_accuracy(m, model_profiles, omega)
        fail_prob *= (1.0 - p_exit)

    p_exit_batch = 1.0 - fail_prob

    if not tail:
        return batch_acc

    tail_acc = estimate_sequential_accuracy(tail, model_profiles, omega)

    return (p_exit_batch * batch_acc) + (fail_prob * tail_acc)


def select_min_risk_strategy(
    selected_models: List[str],
    wait_results: Dict[str, float],
    deadline_ms: int,
    network_latency: float,
    parallel_first_n: int,
    model_profiles: Dict,
    omega: Dict,
    small_instance_models: List[str],
    large_instance_models: List[str],
    hardware_cost_rate: Dict,
    estimate_latency_fn,
    alpha_latency: float = 0.4,
    beta_cost: float = 0.2,
    gamma_error: float = 0.4,
) -> Tuple[str, Dict[str, dict]]:

    candidate_modes = [
        "full_parallel",
        "sequential",
        "parallel_first_finish",
    ]

    costs = {
        "full_parallel": estimate_full_cost(
            selected_models,
            model_profiles,
            small_instance_models,
            large_instance_models,
            hardware_cost_rate,
        ),
        "sequential": estimate_sequential_cost(
            selected_models,
            model_profiles,
            omega,
            small_instance_models,
            large_instance_models,
            hardware_cost_rate,
        ),
        "parallel_first_finish": estimate_power2_cost(
            selected_models,
            parallel_first_n,
            model_profiles,
            omega,
            small_instance_models,
            large_instance_models,
            hardware_cost_rate,
        ),
    }

    accuracies = {
        "full_parallel": estimate_full_accuracy(selected_models, model_profiles, omega),
        "sequential": estimate_sequential_accuracy(selected_models, model_profiles, omega),
        "parallel_first_finish": estimate_power2_accuracy(
            selected_models,
            parallel_first_n,
            model_profiles,
            omega,
        ),
    }

    latencies = {}

    for mode in candidate_modes:
        latencies[mode] = estimate_latency_fn(
            cascade_mode=mode,
            selected_models=selected_models,
            wait_results=wait_results,
            network_latency=network_latency,
            parallel_first_n=parallel_first_n,
        )

    max_cost = max(costs.values()) if costs else 1.0
    deadline_s = max(float(deadline_ms) / 1000.0, 1e-9)

    risk_details = {}

    for mode in candidate_modes:
        latency_risk = latencies[mode] / deadline_s
        cost_risk = costs[mode] / max(max_cost, 1e-9)
        error_risk = 1.0 - accuracies[mode]

        risk = (
            alpha_latency * latency_risk
            + beta_cost * cost_risk
            + gamma_error * error_risk
        )

        risk_details[mode] = {
            "risk": risk,
            "latency_s": latencies[mode],
            "latency_risk": latency_risk,
            "cost": costs[mode],
            "cost_risk": cost_risk,
            "accuracy": accuracies[mode],
            "error_risk": error_risk,
        }

    selected_mode = min(risk_details, key=lambda m: risk_details[m]["risk"])

    return selected_mode, risk_details