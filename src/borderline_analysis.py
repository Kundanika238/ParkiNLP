import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROFILE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "participant_profiles.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "borderline_group_comparison.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    PROFILE_FILE
)

print(
    "Participant profile dataset loaded successfully!"
)

print(
    f"Total participants: {len(df)}"
)


# ============================================================
# CREATE BORDERLINE GROUP
# ============================================================

df["group"] = "Non-borderline"

df.loc[
    df["distance_from_boundary"] <= 0.10,
    "group"
] = "Borderline"


# ============================================================
# GROUP COUNTS
# ============================================================

borderline = df[
    df["group"] == "Borderline"
]

non_borderline = df[
    df["group"] == "Non-borderline"
]


print("\n========================================")
print("BORDERLINE GROUP ANALYSIS")
print("========================================")


print(
    f"\nBorderline participants: "
    f"{len(borderline)}"
)

print(
    f"Non-borderline participants: "
    f"{len(non_borderline)}"
)


# ============================================================
# ACCURACY BY GROUP
# ============================================================

borderline_accuracy = (
    borderline["prediction_correct"]
    .mean()
)

non_borderline_accuracy = (
    non_borderline["prediction_correct"]
    .mean()
)


print("\n----------------------------------------")
print("PREDICTION ACCURACY")
print("----------------------------------------")


print(
    f"Borderline accuracy: "
    f"{borderline_accuracy:.4f}"
)

print(
    f"Non-borderline accuracy: "
    f"{non_borderline_accuracy:.4f}"
)


# ============================================================
# FEATURES TO COMPARE
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
# GROUP COMPARISON
# ============================================================

comparison_rows = []


for feature in features:

    borderline_mean = (
        borderline[feature]
        .mean()
    )

    non_borderline_mean = (
        non_borderline[feature]
        .mean()
    )

    difference = (
        borderline_mean
        -
        non_borderline_mean
    )

    comparison_rows.append({

        "feature": feature,

        "borderline_mean":
            borderline_mean,

        "non_borderline_mean":
            non_borderline_mean,

        "borderline_minus_non_borderline":
            difference

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# DISPLAY COMPARISON
# ============================================================

print("\n----------------------------------------")
print("NLP FEATURE COMPARISON")
print("----------------------------------------")


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# CORRECTNESS WITHIN BORDERLINE GROUP
# ============================================================

print("\n----------------------------------------")
print("BORDERLINE PREDICTION OUTCOMES")
print("----------------------------------------")


borderline_correct = (
    borderline[
        "prediction_correct"
    ].sum()
)

borderline_incorrect = (
    len(borderline)
    -
    borderline_correct
)


print(
    f"Correct: "
    f"{borderline_correct}"
)

print(
    f"Incorrect: "
    f"{borderline_incorrect}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

comparison_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("BORDERLINE ANALYSIS COMPLETED!")
print("========================================")


print(
    f"\nResults saved to:"
)

print(
    OUTPUT_FILE
)