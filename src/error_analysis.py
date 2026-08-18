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

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "final_predictions.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "error_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

features_df = pd.read_csv(
    FEATURE_FILE
)

predictions_df = pd.read_csv(
    PREDICTION_FILE
)


print("NLP feature dataset loaded successfully!")
print(
    f"Total participants: "
    f"{len(features_df)}"
)


# ============================================================
# MERGE FEATURES WITH PREDICTIONS
# ============================================================

df = features_df.merge(
    predictions_df[
        [
            "participant_id",
            "actual_label",
            "predicted_label",
            "PD_probability"
        ]
    ],
    on="participant_id",
    how="inner"
)


# ============================================================
# IDENTIFY CORRECT / INCORRECT PREDICTIONS
# ============================================================

df["prediction_correct"] = (
    df["actual_label"]
    ==
    df["predicted_label"]
)


df["error_type"] = "Correct"


df.loc[
    (
        df["actual_label"] == "HC"
    )
    &
    (
        df["predicted_label"] == "PD"
    ),
    "error_type"
] = "False Positive"


df.loc[
    (
        df["actual_label"] == "PD"
    )
    &
    (
        df["predicted_label"] == "HC"
    ),
    "error_type"
] = "False Negative"


# ============================================================
# DISPLAY OVERALL RESULTS
# ============================================================

total = len(df)

correct = df[
    "prediction_correct"
].sum()

incorrect = total - correct


print("\n========================================")
print("ERROR ANALYSIS")
print("========================================")


print(
    f"\nTotal participants: "
    f"{total}"
)

print(
    f"Correct predictions: "
    f"{correct}"
)

print(
    f"Incorrect predictions: "
    f"{incorrect}"
)


# ============================================================
# DISPLAY ERRORS
# ============================================================

errors = df[
    ~df["prediction_correct"]
].copy()


print("\n----------------------------------------")
print("MISCLASSIFIED PARTICIPANTS")
print("----------------------------------------")


if len(errors) == 0:

    print(
        "\nNo misclassified participants."
    )

else:

    display_columns = [
        "participant_id",
        "actual_label",
        "predicted_label",
        "PD_probability",
        "error_type"
    ]

    print(
        errors[
            display_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# ERROR COUNTS
# ============================================================

print("\n----------------------------------------")
print("ERROR COUNTS")
print("----------------------------------------")


false_positive_count = len(
    df[
        df["error_type"]
        ==
        "False Positive"
    ]
)

false_negative_count = len(
    df[
        df["error_type"]
        ==
        "False Negative"
    ]
)


print(
    f"False Positives (HC → PD): "
    f"{false_positive_count}"
)

print(
    f"False Negatives (PD → HC): "
    f"{false_negative_count}"
)


# ============================================================
# COMPARE CORRECT VS INCORRECT GROUPS
# ============================================================

analysis_features = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "average_sentence_length",
    "average_word_length"
]


print("\n----------------------------------------")
print("CORRECT vs INCORRECT PREDICTIONS")
print("----------------------------------------")


correct_group = df[
    df["prediction_correct"]
]

incorrect_group = df[
    ~df["prediction_correct"]
]


comparison_rows = []


for feature in analysis_features:

    correct_mean = (
        correct_group[feature]
        .mean()
    )

    incorrect_mean = (
        incorrect_group[feature]
        .mean()
    )

    difference = (
        incorrect_mean
        -
        correct_mean
    )

    comparison_rows.append({

        "feature": feature,

        "correct_mean": correct_mean,

        "incorrect_mean": incorrect_mean,

        "incorrect_minus_correct":
            difference

    })


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    comparison_df.to_string(
        index=False
    )
)


# ============================================================
# ADD FEATURE COMPARISON TO OUTPUT
# ============================================================

error_output_columns = [
    "participant_id",
    "actual_label",
    "predicted_label",
    "PD_probability",
    "prediction_correct",
    "error_type"
]


error_output = df[
    error_output_columns
].copy()


# ============================================================
# SAVE PARTICIPANT ERROR ANALYSIS
# ============================================================

error_output.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE FEATURE COMPARISON
# ============================================================

FEATURE_COMPARISON_FILE = os.path.join(
    BASE_DIR,
    "results",
    "error_feature_comparison.csv"
)


comparison_df.to_csv(
    FEATURE_COMPARISON_FILE,
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n========================================")
print("ERROR ANALYSIS COMPLETED!")
print("========================================")


print(
    f"\nParticipant-level analysis saved to:"
)

print(
    OUTPUT_FILE
)


print(
    f"\nFeature comparison saved to:"
)

print(
    FEATURE_COMPARISON_FILE
)