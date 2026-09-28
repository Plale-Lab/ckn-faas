# import os
# import pandas as pd
# import matplotlib.pyplot as plt
#
# CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/error_with_latency_all_policy.csv"
# OUT_DIR = "error_vs_latency_plots"
# os.makedirs(OUT_DIR, exist_ok=True)
#
# POLICY_LABELS = {
#     "po2": "Edge Speculation",
#     "power2": "Edge Speculation",
#     "parallel_first_finish": "Edge Speculation",
#     "sequential": "Sequential Cascade",
#     "spec_b": "Cloud Speculation",
#     "full_parallel": "Full Parallel",
# }
#
# POLICY_MARKERS = {
#     "Edge Speculation": "o",
#     "Sequential Cascade": "s",
#     "Cloud Speculation": "^",
#     "Full Parallel": "D",
# }
#
# FIG_BG_COLOR = "white"
# AX_BG_COLOR = "white"
#
# plt.rcParams["pdf.fonttype"] = 42
# plt.rcParams["ps.fonttype"] = 42
#
# df = pd.read_csv(CSV_PATH)
# df.columns = df.columns.str.strip()
#
# numeric_cols = [
#     "RespTime",
#     "e2e_time_ms",
#     "static_error",
#     "static_cost",
# ]
#
# for col in numeric_cols:
#     if col in df.columns:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#
# # -----------------------------
# # Latency column
# # -----------------------------
# if "RespTime" in df.columns:
#     df["latency_ms"] = df["RespTime"] * 1000.0
# elif "e2e_time_ms" in df.columns:
#     df["latency_ms"] = df["e2e_time_ms"]
# else:
#     raise ValueError("No latency column found. Expected RespTime or e2e_time_ms.")
#
# # -----------------------------
# # Error column
# # -----------------------------
# if "static_error" not in df.columns:
#     raise ValueError("static_error column is required for Error vs Latency plot.")
#
# df = df.dropna(subset=["latency_ms", "static_error"])
#
# # -----------------------------
# # Policy label
# # -----------------------------
# def get_policy_label(row):
#     policy = str(row.get("static_policy", "")).strip().lower()
#
#     if policy in POLICY_LABELS:
#         return POLICY_LABELS[policy]
#
#     mode = str(row.get("effective_cascade_mode", "")).strip().lower()
#     return POLICY_LABELS.get(mode, mode if mode else "unknown")
#
# df["policy_label"] = df.apply(get_policy_label, axis=1)
#
# # -----------------------------
# # Summary: mean latency per policy/error
# # -----------------------------
# summary = (
#     df.groupby(["policy_label", "static_error"], as_index=False)
#     .agg(
#         mean_latency_ms=("latency_ms", "mean"),
#         std_latency_ms=("latency_ms", "std"),
#         n=("latency_ms", "count"),
#     )
# )
#
# summary["std_latency_ms"] = summary["std_latency_ms"].fillna(0)
#
# # -----------------------------
# # Plot
# # -----------------------------
# fig, ax = plt.subplots(figsize=(9, 6))
# fig.patch.set_facecolor(FIG_BG_COLOR)
# ax.set_facecolor(AX_BG_COLOR)
#
# for policy, g in summary.groupby("policy_label"):
#     g = g.sort_values("static_error")
#
#     marker = POLICY_MARKERS.get(policy, "o")
#
#     ax.plot(
#         g["static_error"],
#         g["mean_latency_ms"],
#         marker=marker,
#         linewidth=2.4,
#         markersize=7,
#         label=policy,
#     )
#
#     # Optional shaded STD band
#     lower = (g["mean_latency_ms"] - g["std_latency_ms"]).clip(lower=0)
#     upper = g["mean_latency_ms"] + g["std_latency_ms"]
#
#     ax.fill_between(
#         g["static_error"],
#         lower,
#         upper,
#         alpha=0.12,
#     )
#
# # -----------------------------
# # Highlight example error points
# # -----------------------------
# for e in sorted(summary["static_error"].unique()):
#     ax.axvline(e, linestyle=":", linewidth=1, alpha=0.35)
#
# ax.set_xlabel("Error Constraint (E)", fontsize=16)
# ax.set_ylabel("Average Latency (ms)", fontsize=16)
# # ax.set_title("Error Constraint vs Mean Latency")
#
# ax.grid(True, alpha=0.3)
# ax.legend(fontsize=12)
# ax.tick_params(axis="both", labelsize=14)
#
# plt.tight_layout()
# # plt.savefig(
# #     f"{OUT_DIR}/error_vs_latency_mean_std.png",
# #     dpi=300,
# #     bbox_inches="tight",
# #     facecolor=fig.get_facecolor(),
# # )
#
# out_base = f"{OUT_DIR}/error_vs_latency_mean_std"
#
# plt.savefig(
#     f"{out_base}.png",
#     dpi=300,
#     bbox_inches="tight",
#     facecolor="white",
# )
#
# plt.savefig(
#     f"{out_base}.pdf",
#     bbox_inches="tight",
#     facecolor="white",
# )
#
# plt.show()
#
# print(f"Saved plot to {OUT_DIR}")



import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_PATH = "/Users/agamage/Desktop/D2I/Codes Original/clone main/ckn-faas/codespace/workload_generator/data/error_with_latency_all_policy.csv"
OUT_DIR = "error_vs_latency_plots"
os.makedirs(OUT_DIR, exist_ok=True)

POLICY_LABELS = {
    "po2": "Edge Speculation",
    "power2": "Edge Speculation",
    "parallel_first_finish": "Edge Speculation",
    "sequential": "Sequential Cascade",
    "spec_b": "Cloud Speculation",
    "full_parallel": "Full Parallel",
}

POLICY_MARKERS = {
    "Edge Speculation": "o",
    "Sequential Cascade": "s",
    "Cloud Speculation": "^",
    "Full Parallel": "D",
}

MODEL_DISPLAY = {
    "mobilenet_v3_small": "MobileNetV3-S",
    "mobilenetv3_small_100": "MobileNetV3-S",
    "mobilenet_v2": "MobileNetV2",
    "mobilenetv2_100": "MobileNetV2",
    "shufflenet_v2_x1_0": "ShuffleNetV2",
    "resnet18": "R18",
    "resnet34": "R34",
    "resnet50": "R50",
    "resnet101": "R101",
    "efficientnet_b3": "Eff-B3",
    "convnext_tiny": "ConvNeXt-T",
    "vit_b_16": "ViT-B/16",
    "vit_base_patch16_224": "ViT-B/16",
}

FIG_BG_COLOR = "white"
AX_BG_COLOR = "white"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def parse_models(x):
    try:
        models = ast.literal_eval(str(x))
        if not isinstance(models, list):
            return ""
        return ", ".join([MODEL_DISPLAY.get(str(m), str(m)) for m in models])
    except Exception:
        return ""


def get_theta(row):
    for col in ["static_theta", "selected_theta", "theta"]:
        if col in row and pd.notna(row[col]):
            try:
                return float(row[col])
            except Exception:
                pass
    return np.nan


def get_policy_label(row):
    policy = str(row.get("static_policy", "")).strip().lower()

    if policy in POLICY_LABELS:
        return POLICY_LABELS[policy]

    mode = str(row.get("effective_cascade_mode", "")).strip().lower()
    return POLICY_LABELS.get(mode, mode if mode else "unknown")


# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

numeric_cols = [
    "RespTime",
    "e2e_time_ms",
    "static_error",
    "static_cost",
    "static_theta",
    "selected_theta",
    "theta",
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Latency column
# -----------------------------
# if "RespTime" in df.columns:
#     df["latency_ms"] = df["RespTime"] * 1000.0
# elif "e2e_time_ms" in df.columns:
#     df["latency_ms"] = df["e2e_time_ms"]
# else:
#     raise ValueError("No latency column found. Expected RespTime or e2e_time_ms.")


if "RespTime" in df.columns:
    df["latency_ms"] = (df["RespTime"] * 1000.0 - 200).clip(lower=0)
elif "e2e_time_ms" in df.columns:
    df["latency_ms"] = (df["e2e_time_ms"] - 200).clip(lower=0)
else:
    raise ValueError("No latency column found. Expected RespTime or e2e_time_ms.")

if "static_error" not in df.columns:
    raise ValueError("static_error column is required for Error vs Latency plot.")

df = df.dropna(subset=["latency_ms", "static_error"])

# -----------------------------
# Policy/model/theta labels
# -----------------------------
df["policy_label"] = df.apply(get_policy_label, axis=1)

if "selected_models" in df.columns:
    df["model_text"] = df["selected_models"].apply(parse_models)
else:
    df["model_text"] = ""

df["theta_value"] = df.apply(get_theta, axis=1)

# -----------------------------
# Summary: mean latency + selected model/theta
# -----------------------------
summary = (
    df.groupby(["policy_label", "static_error"], as_index=False)
    .agg(
        mean_latency_ms=("latency_ms", "mean"),
        std_latency_ms=("latency_ms", "std"),
        n=("latency_ms", "count"),
        model_text=("model_text", "first"),
        theta_value=("theta_value", "first"),
    )
)

summary["std_latency_ms"] = summary["std_latency_ms"].fillna(0)

# Save summary for checking values
summary.to_csv(f"{OUT_DIR}/error_vs_latency_summary_with_models.csv", index=False)

# -----------------------------
# Plot
# -----------------------------
# fig, ax = plt.subplots(figsize=(9.8, 6.4))
fig, ax = plt.subplots(figsize=(11.5, 7.2))
fig.patch.set_facecolor(FIG_BG_COLOR)
ax.set_facecolor(AX_BG_COLOR)

for policy, g in summary.groupby("policy_label"):
    g = g.sort_values("static_error")

    marker = POLICY_MARKERS.get(policy, "o")

    ax.plot(
        g["static_error"],
        g["mean_latency_ms"],
        marker=marker,
        linewidth=2.4,
        markersize=7,
        label=policy,
    )

    lower = (g["mean_latency_ms"] - g["std_latency_ms"]).clip(lower=0)
    upper = g["mean_latency_ms"] + g["std_latency_ms"]

    ax.fill_between(
        g["static_error"],
        lower,
        upper,
        alpha=0.12,
    )

# -----------------------------
# Vertical guide lines
# -----------------------------
for e in sorted(summary["static_error"].unique()):
    ax.axvline(e, linestyle=":", linewidth=1, alpha=0.35)

# -----------------------------
# Annotate selected models and theta
# Adjust dx/dy if labels overlap
# -----------------------------
# ANNOTATE_POINTS = {
#     ("Sequential Cascade", 0.09): (-60, 42),
#     ("Sequential Cascade", 0.10): (-80, 5),
#     ("Sequential Cascade", 0.12): (-70, -52),
#     ("Sequential Cascade", 0.14): (-70, -48),
#
#     ("Edge Speculation", 0.09): (25, 45),
#     ("Edge Speculation", 0.10): (25, 10),
#     ("Edge Speculation", 0.12): (25, -55),
#     ("Edge Speculation", 0.14): (-120, -70),
#
#     ("Cloud Speculation", 0.12): (25, 40),
#     ("Cloud Speculation", 0.14): (25, -45),
# }

ANNOTATE_POINTS = {
    # E = 0.09
    ("Cloud Speculation", 0.09): (-35, 40),
    ("Sequential Cascade", 0.09): (65, 55),
    ("Edge Speculation", 0.09): (55, 50),

    # E = 0.10
    ("Cloud Speculation", 0.10): (25, -40),
    ("Sequential Cascade", 0.10): (100, 40),
    ("Edge Speculation", 0.10): (-140, -85),

    # E = 0.12
    ("Cloud Speculation", 0.12): (-95, -40),
    ("Sequential Cascade", 0.12): (-90, 95),
    ("Edge Speculation", 0.12): (40, 10),

    # E = 0.14
    ("Cloud Speculation", 0.14): (-130, -25),
    ("Sequential Cascade", 0.14): (-120, 90),
    ("Edge Speculation", 0.14): (-255, -65),
}

for _, row in summary.iterrows():
    policy = row["policy_label"]
    e = round(float(row["static_error"]), 2)

    key = (policy, e)

    if key not in ANNOTATE_POINTS:
        continue

    model_text = row["model_text"]
    theta = row["theta_value"]

    if model_text == "" or pd.isna(theta):
        continue

    # label = f"{model_text}\n$\\theta$={theta:.2f}"

    label = f"({model_text}, {theta:.2f})"

    dx, dy = ANNOTATE_POINTS[key]

    ax.annotate(
        label,
        xy=(row["static_error"], row["mean_latency_ms"]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=16,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.9,
            alpha=0.75,
        ),
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor="gray",
            alpha=0.88,
        ),
    )
ax.set_xlim(0.087, 0.146)
# ax.set_ylim(220, 1700)
ax.set_ylim(20, 1500)
plt.tight_layout(pad=1.5)
ax.set_xlabel("Error Constraint (E)", fontsize=24)
ax.set_ylabel("Average Latency (ms)", fontsize=24)

ax.grid(True, alpha=0.3)
ax.legend(fontsize=17, loc="upper right")
ax.tick_params(axis="both", labelsize=20)

plt.tight_layout()

out_base = f"{OUT_DIR}/error_vs_latency_mean_std_annotated"

plt.savefig(
    f"{out_base}.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.savefig(
    f"{out_base}.pdf",
    bbox_inches="tight",
    facecolor="white",
)

plt.show()

print(f"Saved annotated plot to {OUT_DIR}")
print(f"Saved summary to {OUT_DIR}/error_vs_latency_summary_with_models.csv")