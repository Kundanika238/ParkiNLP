import os
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

ERROR_FILE = os.path.join(
    BASE_DIR,
    "results",
    "error_analysis.csv"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "nlp_features.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "error_plots"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

error_df = pd.read_csv(
    ERROR_FILE
)

features_df = pd.read_csv(
    FEATURE_FILE
)

print(
    "Error analysis dataset loaded successfully!"
)

print(
    f"Total participants: "
    f"{len(error_df)}"
)


# ============================================================
# MERGE ERROR INFORMATION WITH NLP FEATURES
# ============================================================

df = error_df.merge(
    features_df,
    on="participant_id",
    how="inner"
)


print(
    f"Merged dataset participants: "
    f"{len(df)}"
)


# ============================================================
# CHECK MERGE
# ============================================================

if len(df) != len(error_df):

    raise ValueError(
        "Some participants could not be matched "
        "between error analysis and NLP feature data."
    )


# ============================================================
# CREATE CORRECT / INCORRECT GROUPS
# ============================================================

correct_group = df[
    df["prediction_correct"] == True
]

incorrect_group = df[
    df["prediction_correct"] == False
]


print(
    f"Correct predictions: "
    f"{len(correct_group)}"
)

print(
    f"Incorrect predictions: "
    f"{len(incorrect_group)}"
)


# ============================================================
# FEATURES TO VISUALIZE
# ============================================================

features = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "average_sentence_length",
    "average_word_length"
]


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing NLP features: "
        +
        ", ".join(missing_features)
    )


# ============================================================
# CALCULATE GROUP MEANS
# ============================================================

correct_means = [
    correct_group[feature].mean()
    for feature in features
]

incorrect_means = [
    incorrect_group[feature].mean()
    for feature in features
]


# ============================================================
# CORRECT VS INCORRECT FEATURE COMPARISON
# ============================================================

x = range(
    len(features)
)

width = 0.35


fig, ax = plt.subplots(
    figsize=(12, 7)
)


ax.bar(
    [
        i - width / 2
        for i in x
    ],
    correct_means,
    width,
    label="Correct"
)


ax.bar(
    [
        i + width / 2
        for i in x
    ],
    incorrect_means,
    width,
    label="Incorrect"
)


ax.set_xlabel(
    "NLP Feature"
)

ax.set_ylabel(
    "Mean Value"
)

ax.set_title(
    "Correct vs Incorrect Prediction Feature Comparison"
)

ax.set_xticks(
    list(x)
)

ax.set_xticklabels(
    features,
    rotation=45,
    ha="right"
)

ax.legend()

plt.tight_layout()


feature_plot = os.path.join(
    OUTPUT_DIR,
    "correct_vs_incorrect_features.png"
)


plt.savefig(
    feature_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved feature comparison plot:\n"
    f"{feature_plot}"
)


# ============================================================
# PD PROBABILITY DISTRIBUTION
# ============================================================

correct_pd = correct_group[
    "PD_probability"
]

incorrect_pd = incorrect_group[
    "PD_probability"
]


fig, ax = plt.subplots(
    figsize=(9, 6)
)


ax.hist(
    correct_pd,
    bins=10,
    alpha=0.7,
    label="Correct predictions"
)


ax.hist(
    incorrect_pd,
    bins=10,
    alpha=0.7,
    label="Incorrect predictions"
)


ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Decision threshold"
)


ax.set_xlabel(
    "Predicted Probability of PD"
)

ax.set_ylabel(
    "Number of Participants"
)

ax.set_title(
    "PD Probability Distribution: Correct vs Incorrect Predictions"
)

ax.legend()

plt.tight_layout()


probability_plot = os.path.join(
    OUTPUT_DIR,
    "pd_probability_distribution.png"
)


plt.savefig(
    probability_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved probability distribution plot:\n"
    f"{probability_plot}"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("ERROR VISUALIZATION COMPLETED!")
print("========================================")

print(
    f"\nPlots saved in:\n"
    f"{OUTPUT_DIR}"
)