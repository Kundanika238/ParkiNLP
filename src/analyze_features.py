import os
import pandas as pd

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "nlp_features.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "feature_group_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")

print(f"Total participants: {len(df)}")

print(
    f"Healthy controls: "
    f"{(df['label'] == 'HC').sum()}"
)

print(
    f"Parkinson's participants: "
    f"{(df['label'] == 'PD').sum()}"
)


# ============================================================
# SELECT NUMERICAL FEATURES
# ============================================================

id_columns = [
    "participant_id",
    "label"
]

feature_columns = [
    column
    for column in df.columns
    if column not in id_columns
]


# ============================================================
# GROUP MEANS
# ============================================================

hc_df = df[df["label"] == "HC"]

pd_df = df[df["label"] == "PD"]


comparison = []

for feature in feature_columns:

    hc_mean = hc_df[feature].mean()

    pd_mean = pd_df[feature].mean()

    difference = pd_mean - hc_mean

    if hc_mean != 0:

        percentage_difference = (
            difference / abs(hc_mean)
        ) * 100

    else:

        percentage_difference = 0

    comparison.append({

        "feature": feature,

        "HC_mean": hc_mean,

        "PD_mean": pd_mean,

        "PD_minus_HC": difference,

        "percentage_difference": percentage_difference

    })


# ============================================================
# CREATE COMPARISON DATAFRAME
# ============================================================

comparison_df = pd.DataFrame(comparison)


# ============================================================
# SORT BY ABSOLUTE GROUP DIFFERENCE
# ============================================================

comparison_df["absolute_difference"] = (
    comparison_df["PD_minus_HC"].abs()
)

comparison_df = comparison_df.sort_values(
    by="absolute_difference",
    ascending=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("FEATURE GROUP ANALYSIS")
print("========================================")

print(
    "\nFeatures sorted by absolute "
    "HC-PD mean difference:\n"
)

display_columns = [
    "feature",
    "HC_mean",
    "PD_mean",
    "PD_minus_HC",
    "percentage_difference"
]

print(
    comparison_df[
        display_columns
    ].to_string(index=False)
)


print("\n========================================")
print("ANALYSIS COMPLETED!")
print("========================================")

print(
    f"\nSaved comparison to:\n{OUTPUT_FILE}"
)