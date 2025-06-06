import random

# Mock helper functions
def expected_time(model, x):
    # return model_profiles[model]["latency"] * (1 + 0.1 * random.random())
    return model_profiles[model]["latency"]

def get_accuracy(model, x):
    return model_profiles[model]["accuracy"] + random.uniform(-0.02, 0.02)
    # return model_profiles[model]["accuracy"]

# Configuration and initialization
M_total = ["mobilenet_v3_small","resnet18","resnet34","resnet50","resnet101","vit_b_16"]
R = ["req1", "req2", "req3"]
D = [10000,19000,18000] # tight deadline for each request
policy = "greedy"
K = 8     # total cores
c = 1     # cores per model
alpha = 1.0
eta = 0.1
active_models = set()
cold_penalty = {
    "mobilenet_v3_small": 30,
    "resnet18": 40,
    "resnet34": 50,
    "resnet50": 60,
    "resnet101": 70,
    "vit_b_16": 80,
}
omega = {}
A = []
assert len(R) == len(D)
# Model profiles
model_profiles = {
    "mobilenet_v3_small": {"latency": 14.2121078288183, "accuracy": 0.704413581114262},
    "resnet18": {"latency": 28.992410373874, "accuracy": 0.712173486609011},
    "resnet34": {"latency": 42.8559443559499, "accuracy": 0.775743676409125},
    "resnet50": {"latency": 63.9653322417292, "accuracy": 0.797511352837085},
    "resnet101": {"latency": 105.657157612033, "accuracy": 0.822238205507397},
    "vit_b_16": {"latency": 175.83452499425, "accuracy": 0.751089387536048},
}
import time

start_time = time.perf_counter()
# Main algorithm
for i,r in enumerate(R):
    x = r  # placeholder input
    M_D = []

    for m in M_total:
        if m not in omega:
            omega[m] = 1.0 / len(m)  # initial proxy

    cost = {}
    if policy == "greedy":
        for m in M_total:
            latency = expected_time(m, x)
            if latency > D[i]:
                cost[m] = float('inf')
                continue

            penalty = 0 if m in active_models or (K >= c) else cold_penalty[m]
            cost[m] = (latency + penalty) / D[i] + alpha / omega[m]

        sorted_models = sorted(M_total, key=lambda m: cost[m])
        T_est = 0
        used_cores = 0

        for m in sorted_models:
            latency = expected_time(m, x)
            if len(M_D) < K // c and used_cores + c <= K and T_est + latency <= D[i]:
                M_D.append(m)
                T_est = max(T_est, latency)
                used_cores += c

    elif policy == "randomized":
        candidate_pool = random.sample(M_total, min(len(M_total), K // c))
        valid = []
        for m in candidate_pool:
            est_latency = model_profiles[m]["latency"]
            if est_latency <= D[i]:
                valid.append(m)
        M_D = valid if len(valid) * c <= K else []
    # Simulate model predictions and select best
    Y = {m: get_accuracy(m, x) for m in M_D}
    y_best = max(Y, key=Y.get)
    A.append(y_best)

    for m in M_D:
        A_m = Y[m]
        omega[m] = (1 - eta) * omega[m] + eta * A_m
        active_models.add(m)

    # print(f"Request: {r}")
    # print(f"  Selected Models: {M_D}")
    # print(f"  Best Model: {y_best} with estimated accuracy {Y[y_best]:.4f}")
    print(M_D)
end_time = time.perf_counter()
print((end_time - start_time)*1000)
# print("\nFinal Predictions per Request:")
