import os
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


FEATURE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "nlp_features.csv"
)


MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "parkinlp_model.joblib"
)


# ============================================================
# FINAL FEATURES
# ============================================================

SELECTED_FEATURES = [

    "filler_count",

    "short_sentence_ratio",

    "type_token_ratio",

    "sentence_count",

    "filler_rate"

]


# ============================================================
# NEW SPEECH SAMPLE
# ============================================================

NEW_SAMPLE = {

    "filler_count": 0,

    "short_sentence_ratio": 0.0,

    "type_token_ratio": 0.726027397260274,

    "sentence_count": 5,

    "filler_rate": 0.0

}


# ============================================================
# LOAD TRAINING DATA
# ============================================================

df = pd.read_csv(
    FEATURE_FILE
)


print("========================================")
print("PARKINLP MODEL DIAGNOSTIC")
print("========================================")


print(
    f"\nTraining participants: {len(df)}"
)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)


print(
    "Deployment model loaded successfully!"
)


# ============================================================
# TRAINING DATA DISTRIBUTION
# ============================================================

print("\n========================================")
print("TRAINING FEATURE DISTRIBUTION")
print("========================================")


distribution_rows = []


for feature in SELECTED_FEATURES:

    values = df[feature].dropna()

    mean = values.mean()

    std = values.std()

    minimum = values.min()

    maximum = values.max()

    new_value = NEW_SAMPLE[feature]


    if std > 0:

        z_score = (
            new_value - mean
        ) / std

    else:

        z_score = 0


    distribution_rows.append({

        "feature":
            feature,

        "training_mean":
            mean,

        "training_std":
            std,

        "training_min":
            minimum,

        "training_max":
            maximum,

        "new_value":
            new_value,

        "z_score":
            z_score

    })


distribution_df = pd.DataFrame(
    distribution_rows
)


print(
    distribution_df.to_string(
        index=False
    )
)


# ============================================================
# CHECK WHETHER FEATURES ARE WITHIN TRAINING RANGE
# ============================================================

print("\n========================================")
print("TRAINING RANGE CHECK")
print("========================================")


for feature in SELECTED_FEATURES:

    minimum = df[
        feature
    ].min()

    maximum = df[
        feature
    ].max()

    value = NEW_SAMPLE[
        feature
    ]


    if minimum <= value <= maximum:

        status = "INSIDE training range"

    else:

        status = "OUTSIDE training range"


    print(
        f"{feature}: "
        f"{value} → {status}"
    )


# ============================================================
# NEW SAMPLE DATAFRAME
# ============================================================

new_df = pd.DataFrame(
    [NEW_SAMPLE],
    columns=SELECTED_FEATURES
)


# ============================================================
# MODEL PREDICTION
# ============================================================

prediction = model.predict(
    new_df
)[0]


probabilities = model.predict_proba(
    new_df
)[0]


class_order = list(
    model.classes_
)


pd_index = class_order.index(
    "PD"
)


hc_index = class_order.index(
    "HC"
)


pd_probability = float(
    probabilities[pd_index]
)


hc_probability = float(
    probabilities[hc_index]
)


# ============================================================
# DISPLAY MODEL PREDICTION
# ============================================================

print("\n========================================")
print("MODEL PREDICTION")
print("========================================")


print(
    f"\nPrediction: {prediction}"
)


print(
    f"HC probability: "
    f"{hc_probability:.4f}"
)


print(
    f"PD probability: "
    f"{pd_probability:.4f}"
)


# ============================================================
# LOGISTIC REGRESSION COEFFICIENTS
# ============================================================

print("\n========================================")
print("MODEL COEFFICIENT ANALYSIS")
print("========================================")


classifier = model.named_steps[
    "classifier"
]


coefficients = classifier.coef_[0]


coefficient_rows = []


for feature, coefficient in zip(
    SELECTED_FEATURES,
    coefficients
):

    coefficient_rows.append({

        "feature":
            feature,

        "coefficient":
            coefficient,

        "direction":
            "PD" if coefficient > 0
            else "HC"

    })


coefficient_df = pd.DataFrame(
    coefficient_rows
)


print(
    coefficient_df.to_string(
        index=False
    )
)


# ============================================================
# FEATURE CONTRIBUTION APPROXIMATION
# ============================================================

print("\n========================================")
print("FEATURE CONTRIBUTION")
print("========================================")


scaler = model.named_steps[
    "scaler"
]


scaled_values = scaler.transform(
    new_df
)[0]


contribution_rows = []


for feature, scaled_value, coefficient in zip(
    SELECTED_FEATURES,
    scaled_values,
    coefficients
):

    contribution = (
        scaled_value * coefficient
    )


    contribution_rows.append({

        "feature":
            feature,

        "scaled_value":
            scaled_value,

        "coefficient":
            coefficient,

        "contribution":
            contribution

    })


contribution_df = pd.DataFrame(
    contribution_rows
)


print(
    contribution_df.to_string(
        index=False
    )
)


# ============================================================
# OVERALL Z-SCORE CHECK
# ============================================================

absolute_z_scores = np.abs(
    distribution_df["z_score"]
)


maximum_z = absolute_z_scores.max()


print("\n========================================")
print("OVERALL DISTRIBUTION CHECK")
print("========================================")


print(
    f"\nMaximum absolute z-score: "
    f"{maximum_z:.4f}"
)


if maximum_z <= 2:

    print(
        "Overall status: "
        "Within approximately 2 standard deviations "
        "of the training distribution."
    )

elif maximum_z <= 3:

    print(
        "Overall status: "
        "Somewhat unusual compared with "
        "the training distribution."
    )

else:

    print(
        "Overall status: "
        "Potentially unusual compared with "
        "the training distribution."
    )


print("\n========================================")
print("MODEL DIAGNOSTIC COMPLETED!")
print("========================================")