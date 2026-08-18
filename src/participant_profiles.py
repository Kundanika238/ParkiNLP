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
    "participant_profiles.csv"
)

BORDERLINE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "borderline_participants.csv"
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
# MERGE DATA
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


if len(df) != len(features_df):

    raise ValueError(
        "Participant matching failed."
    )


# ============================================================
# PREDICTION STATUS
# ============================================================

df["prediction_correct"] = (
    df["actual_label"]
    ==
    df["predicted_label"]
)


df["prediction_status"] = "Correct"

df.loc[
    ~df["prediction_correct"],
    "prediction_status"
] = "Incorrect"


# ============================================================
# DISTANCE FROM DECISION BOUNDARY
# ============================================================

df["distance_from_boundary"] = (
    abs(
        df["PD_probability"] - 0.5
    )
)


# ============================================================
# UNCERTAINTY CATEGORY
# ============================================================

def classify_uncertainty(probability):

    distance = abs(
        probability - 0.5
    )

    if distance <= 0.10:
        return "Very uncertain"

    elif distance <= 0.20:
        return "Uncertain"

    elif distance <= 0.30:
        return "Moderately confident"

    else:
        return "Confident"


df["uncertainty_category"] = (
    df["PD_probability"]
    .apply(
        classify_uncertainty
    )
)


# ============================================================
# SELECT IMPORTANT FEATURES
# ============================================================

profile_features = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "average_sentence_length",
    "average_word_length"
]


# ============================================================
# CREATE PARTICIPANT PROFILE
# ============================================================

profile_columns = [
    "participant_id",
    "actual_label",
    "predicted_label",
    "PD_probability",
    "prediction_correct",
    "prediction_status",
    "distance_from_boundary",
    "uncertainty_category"
] + profile_features


profiles = df[
    profile_columns
].copy()


# ============================================================
# SORT BY MODEL UNCERTAINTY
# ============================================================

profiles = profiles.sort_values(
    by="distance_from_boundary"
)


# ============================================================
# DISPLAY MOST UNCERTAIN PARTICIPANTS
# ============================================================

print("\n========================================")
print("MOST UNCERTAIN PARTICIPANTS")
print("========================================")


display_columns = [
    "participant_id",
    "actual_label",
    "predicted_label",
    "PD_probability",
    "distance_from_boundary",
    "uncertainty_category",
    "prediction_status"
]


print(
    profiles[
        display_columns
    ]
    .head(10)
    .to_string(
        index=False
    )
)


# ============================================================
# IDENTIFY BORDERLINE PARTICIPANTS
# ============================================================

borderline = profiles[
    profiles[
        "distance_from_boundary"
    ] <= 0.10
].copy()


print("\n========================================")
print("BORDERLINE PARTICIPANTS")
print("========================================")


print(
    f"\nNumber of borderline participants: "
    f"{len(borderline)}"
)


if len(borderline) > 0:

    print(
        borderline[
            display_columns
        ].to_string(
            index=False
        )
    )

else:

    print(
        "No participants found within "
        "0.10 of the decision boundary."
    )


# ============================================================
# SAVE ALL PROFILES
# ============================================================

profiles.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SAVE BORDERLINE PARTICIPANTS
# ============================================================

borderline.to_csv(
    BORDERLINE_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("PARTICIPANT PROFILE ANALYSIS COMPLETED!")
print("========================================")


print(
    f"\nParticipant profiles saved to:"
)

print(
    OUTPUT_FILE
)


print(
    f"\nBorderline participants saved to:"
)

print(
    BORDERLINE_FILE
)