from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = Path(
    "/Users/agamage/Desktop/D2I/Codes Original/clone main/"
    "ckn-faas/codespace/workload_generator/data/"
    "seq_latency_IAT_table_corrected_labels.csv"
)

OUTPUT_CSV = INPUT_CSV.with_name(
    f"{INPUT_CSV.stem}_corrected_labels.csv"
)

AUDIT_CSV = INPUT_CSV.with_name(
    f"{INPUT_CSV.stem}_label_correction_audit.csv"
)


# ------------------------------------------------------------
# Choose which experiment group to correct
# ------------------------------------------------------------

# Use one of:
#   "static_error"
#   "baseline"

SELECTION_MODE = "baseline"


# Used only when SELECTION_MODE = "static_error"
SELECTED_ERROR = 0.090


# Used only when SELECTION_MODE = "baseline"
SELECTED_BASELINE = "frugalml"


# Maximum number of rows that may be corrected in one run.
MAX_CHANGES = 23


# Floating-point comparison tolerance for E values.
ERROR_TOLERANCE = 1e-6


# ============================================================
# SUPPORTED MODELS
# ============================================================

MODEL_NAMES = [
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


# ============================================================
# HELPERS
# ============================================================

def normalize_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe.columns = (
        dataframe.columns
        .str.strip()
        .str.lower()
    )
    return dataframe


def convert_to_bool(value):
    if pd.isna(value):
        return pd.NA

    normalized = str(value).strip().lower()

    if normalized in {"true", "1", "yes"}:
        return True

    if normalized in {"false", "0", "no"}:
        return False

    return pd.NA


def normalize_label(value) -> str:
    if pd.isna(value):
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def parse_executed_models(value) -> list[str]:
    """
    Convert values such as:
        "['resnet34']"
        "['resnet34', 'vit_b_16']"
    into Python lists.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [
            str(model).strip().lower()
            for model in value
        ]

    text = str(value).strip()

    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return [
                str(model).strip().lower()
                for model in parsed
            ]
    except (ValueError, SyntaxError):
        pass

    # Fallback for malformed list strings.
    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    return [
        model.strip().lower()
        for model in text.split(",")
        if model.strip()
    ]


def determine_final_model(row: pd.Series) -> str | None:
    """
    Determine which model produced the final response.

    For a sequential cascade, the final executed model is normally
    the last entry in executed_models.
    """
    executed_models = parse_executed_models(
        row.get("executed_models")
    )

    if not executed_models:
        return None

    final_model = executed_models[-1]

    if final_model not in MODEL_NAMES:
        return None

    return final_model


def get_final_model_label(
    row: pd.Series,
    final_model: str,
):
    label_column = f"{final_model}_label"

    if label_column not in row.index:
        return pd.NA

    value = row[label_column]

    if pd.isna(value):
        return pd.NA

    value = str(value).strip()

    if not value:
        return pd.NA

    return value


# ============================================================
# LOAD CSV
# ============================================================

df = pd.read_csv(INPUT_CSV)
df = normalize_column_names(df)

required_columns = {
    "label",
    "executed_models",
    "static_config_enabled",
    "static_error",
    "baseline_mode",
}

missing_columns = sorted(
    required_columns - set(df.columns)
)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# Make sure all model-label columns are available.
missing_model_columns = [
    f"{model}_label"
    for model in MODEL_NAMES
    if f"{model}_label" not in df.columns
]

if missing_model_columns:
    print(
        "Warning: Some model label columns are missing:"
    )

    for column in missing_model_columns:
        print(f"  {column}")


# ============================================================
# CLEAN FILTERING COLUMNS
# ============================================================

df["is_static"] = df[
    "static_config_enabled"
].apply(convert_to_bool)

df["static_error"] = pd.to_numeric(
    df["static_error"],
    errors="coerce",
)

df["baseline_mode"] = (
    df["baseline_mode"]
    .fillna("none")
    .astype(str)
    .str.strip()
    .str.lower()
)


# Preserve the original main label.
if "original_label" not in df.columns:
    df["original_label"] = df["label"]


# ============================================================
# SELECT TARGET EXPERIMENT
# ============================================================

if SELECTION_MODE == "static_error":

    target_mask = (
        (df["is_static"] == True)
        & (
            (df["static_error"] - SELECTED_ERROR)
            .abs()
            <= ERROR_TOLERANCE
        )
    )

    selection_description = (
        f"DYSCI E={SELECTED_ERROR:.3f}"
    )

elif SELECTION_MODE == "baseline":

    selected_baseline_normalized = (
        SELECTED_BASELINE
        .strip()
        .lower()
    )

    target_mask = (
        (df["is_static"] == False)
        & (
            df["baseline_mode"]
            == selected_baseline_normalized
        )
    )

    selection_description = (
        f"baseline={selected_baseline_normalized}"
    )

else:
    raise ValueError(
        "SELECTION_MODE must be either "
        "'static_error' or 'baseline'."
    )


# ============================================================
# FIND LOGGING INCONSISTENCIES
# ============================================================

candidate_records = []

for index, row in df[target_mask].iterrows():

    final_model = determine_final_model(row)

    if final_model is None:
        continue

    expected_label = get_final_model_label(
        row=row,
        final_model=final_model,
    )

    if pd.isna(expected_label):
        continue

    current_label = row["label"]

    # Only correct actual mismatches.
    if (
        normalize_label(current_label)
        == normalize_label(expected_label)
    ):
        continue

    candidate_records.append({
        "row_index": index,
        "id": row.get("id", ""),
        "selection": selection_description,
        "executed_models": row.get(
            "executed_models",
            "",
        ),
        "final_model": final_model,
        "old_label": current_label,
        "new_label": expected_label,
        "stop_reason": row.get(
            "stop_reason",
            "",
        ),
    })


candidate_df = pd.DataFrame(candidate_records)

print(
    f"\nSelected experiment: "
    f"{selection_description}"
)

print(
    f"Rows in selected group: "
    f"{int(target_mask.sum())}"
)

print(
    f"Label inconsistencies found: "
    f"{len(candidate_df)}"
)


# ============================================================
# APPLY LIMITED NUMBER OF CORRECTIONS
# ============================================================

if candidate_df.empty:

    print(
        "\nNo main-label/model-label inconsistencies "
        "were found."
    )

    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print(
        f"\nUnchanged copy saved to:\n"
        f"{OUTPUT_CSV}"
    )

else:

    corrections_to_apply = candidate_df.head(
        MAX_CHANGES
    ).copy()

    for _, correction in (
        corrections_to_apply.iterrows()
    ):
        row_index = int(
            correction["row_index"]
        )

        df.at[
            row_index,
            "label",
        ] = correction["new_label"]

    corrections_to_apply[
        "correction_reason"
    ] = (
        "Main label did not match the label "
        "from the final executed model"
    )

    corrections_to_apply[
        "selection_mode"
    ] = SELECTION_MODE

    corrections_to_apply[
        "maximum_changes_requested"
    ] = MAX_CHANGES

    # Save corrected full CSV.
    df.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    # Save exact rows changed.
    corrections_to_apply.to_csv(
        AUDIT_CSV,
        index=False,
    )

    print(
        f"\nCorrections applied: "
        f"{len(corrections_to_apply)}"
    )

    print(
        f"Maximum permitted changes: "
        f"{MAX_CHANGES}"
    )

    print(
        "\nCorrection summary:"
    )

    print(
        corrections_to_apply[
            [
                "id",
                "final_model",
                "old_label",
                "new_label",
                "stop_reason",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nCorrected CSV saved to:\n"
        f"{OUTPUT_CSV}"
    )

    print(
        f"\nAudit file saved to:\n"
        f"{AUDIT_CSV}"
    )