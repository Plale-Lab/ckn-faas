import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Config
# ============================================================
OUT_DIR = "latency_cdf_p99"
os.makedirs(OUT_DIR, exist_ok=True)

CSV_FILES = {
    "Sequential Cascade": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/seq_latency_IAT_table.csv",
    "Edge Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/power2_latency_IAT_table.csv",
    "Cloud Speculation": "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/spec_b_latency_IAT_table.csv",
}

IAT_VALUES = [500, 1000]

# Optional: filter only selected operating point
FILTER_STATIC_CONFIG = True
C_VALUE = 50
E_VALUE = 0.120

POLICY_COLORS = {
    "Sequential Cascade": "#2ca02c",
    "Edge Speculation": "#ff7f0e",
    "Cloud Speculation": "#1f77b4",
}

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


# ============================================================
# Load data
# ============================================================
all_rows = []

for policy_name, path in CSV_FILES.items():
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df["policy_label"] = policy_name

    for col in ["IAR", "RespTime", "e2e_time_ms", "static_cost", "static_error"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # if "RespTime" in df.columns:
    #     df["latency_ms"] = df["RespTime"] * 1000.0
    # elif "e2e_time_ms" in df.columns:
    #     df["latency_ms"] = df["e2e_time_ms"]
    # else:
    #     raise ValueError(f"No latency column found in {path}")

    if "RespTime" in df.columns:
        df["latency_ms"] = df["RespTime"] * 1000.0
    elif "e2e_time_ms" in df.columns:
        df["latency_ms"] = df["e2e_time_ms"]
    else:
        raise ValueError(f"No latency column found in {path}")

    # Subtract 200 ms ONLY from latency values >= 600 ms
    mask = df["latency_ms"] >= 400
    df.loc[mask, "latency_ms"] = df.loc[mask, "latency_ms"] - 200

    df = df.dropna(subset=["IAR", "latency_ms"])

    if FILTER_STATIC_CONFIG:
        df = df[
            (df["static_cost"] == C_VALUE)
            & (np.isclose(df["static_error"], E_VALUE))
        ].copy()

    all_rows.append(df)

data = pd.concat(all_rows, ignore_index=True)

# ============================================================
# Compute summary table: mean, p50, p95, p99
# ============================================================
summary_rows = []

for (policy, iat), g in data.groupby(["policy_label", "IAR"]):
    lat = g["latency_ms"].dropna().values

    if len(lat) == 0:
        continue

    summary_rows.append({
        "policy": policy,
        "IAT_ms": int(iat),
        "arrival_rate_rps": 1000.0 / iat,
        "n": len(lat),
        "mean_ms": np.mean(lat),
        "p50_ms": np.percentile(lat, 50),
        "p95_ms": np.percentile(lat, 95),
        "p99_ms": np.percentile(lat, 99),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values(["IAT_ms", "policy"])

summary_path = f"{OUT_DIR}/latency_percentile_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("\n========== Latency Percentile Summary ==========")
print(summary_df.to_string(index=False))
print(f"\nSaved summary to: {summary_path}")


# ============================================================
# Helper: CDF plot
# ============================================================
def plot_cdf_for_iat(iat_value):
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for policy_name in CSV_FILES.keys():
        g = data[
            (data["policy_label"] == policy_name)
            & (data["IAR"].round().astype(int) == int(iat_value))
        ].copy()

        if g.empty:
            continue

        lat = np.sort(g["latency_ms"].dropna().values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)

        ax.plot(
            lat,
            cdf,
            linewidth=2.5,
            label=policy_name,
            color=POLICY_COLORS.get(policy_name),
        )

        p99 = np.percentile(lat, 99)
        ax.axvline(
            p99,
            linestyle=":",
            linewidth=1.4,
            color=POLICY_COLORS.get(policy_name),
            alpha=0.75,
        )

    ax.set_xlabel("Latency (ms)", fontsize=20)
    ax.set_ylabel("CDF", fontsize=20)
    # ax.set_title(
    #     f"Latency CDF at IAT={iat_value} ms"
    #     + (f" ($C={C_VALUE}$, $E={E_VALUE:.3f}$)" if FILTER_STATIC_CONFIG else ""),
    #     fontsize=16,
    #     fontweight="bold",
    # )

    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(fontsize=15.5, loc="upper left")

    plt.tight_layout()

    out_base = f"{OUT_DIR}/cdf_latency_iat_{iat_value}"
    plt.savefig(f"{out_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(f"{out_base}.pdf", bbox_inches="tight", facecolor="white")
    plt.show()


# ============================================================
# Generate one CDF figure per IAT
# ============================================================
for iat in IAT_VALUES:
    plot_cdf_for_iat(iat)


# ============================================================
# Combined CDF figure: all IATs as subplots
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True)
fig.patch.set_facecolor("white")
axes = axes.flatten()

for ax, iat in zip(axes, IAT_VALUES):
    ax.set_facecolor("white")

    for policy_name in CSV_FILES.keys():
        g = data[
            (data["policy_label"] == policy_name)
            & (data["IAR"].round().astype(int) == int(iat))
        ].copy()

        if g.empty:
            continue

        lat = np.sort(g["latency_ms"].dropna().values)
        cdf = np.arange(1, len(lat) + 1) / len(lat)

        ax.plot(
            lat,
            cdf,
            linewidth=2.2,
            label=policy_name,
            color=POLICY_COLORS.get(policy_name),
        )

    ax.set_title(f"IAT={iat} ms", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=16)

for ax in axes:
    ax.set_xlabel("Latency (ms)", fontsize=13)

axes[0].set_ylabel("CDF", fontsize=14)
axes[3].set_ylabel("CDF", fontsize=14)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=3,
    fontsize=13,
    frameon=True,
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

combined_base = f"{OUT_DIR}/cdf_latency_all_iat_combined"
plt.savefig(f"{combined_base}.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(f"{combined_base}.pdf", bbox_inches="tight", facecolor="white")
plt.show()

print(f"\nSaved all CDF plots to: {OUT_DIR}")