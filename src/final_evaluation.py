import os
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


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

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

PREDICTION_FILE = os.path.join(
    RESULTS_DIR,
    "final_predictions.csv"
)

METRICS_FILE = os.path.join(
    RESULTS_DIR,
    "final_metrics.csv"
)

CONFUSION_FILE = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# SELECT PRE-SPECIFIED FEATURES
# ============================================================

selected_features = [
    "filler_count",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "filler_rate"
]


print("\nFinal feature set:")

for feature in selected_features:
    print(f"- {feature}")


# ============================================================
# CHECK FEATURES
# ============================================================

missing_features = [
    feature
    for feature in selected_features
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing features: "
        + ", ".join(missing_features)
    )


# ============================================================
# PREPARE DATA
# ============================================================

X = df[selected_features]

y = df["label"]


print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# FINAL LEAKAGE-SAFE PIPELINE
# ============================================================

model = Pipeline([

    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    )

])


# ============================================================
# STRATIFIED CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# CROSS-VALIDATED PREDICTIONS
# ============================================================

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict"
)


probabilities = cross_val_predict(
    model,
    X,
    y,
    cv=cv,
    method="predict_proba"
)


# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)

precision = precision_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y,
    predictions,
    average="macro",
    zero_division=0
)


# ============================================================
# ROC-AUC
# ============================================================

class_order = list(
    model.fit(X, y).classes_
)

pd_index = class_order.index("PD")

pd_probability = probabilities[:, pd_index]

y_binary = (
    y == "PD"
).astype(int)


roc_auc = roc_auc_score(
    y_binary,
    pd_probability
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "HC",
    "PD"
]

cm = confusion_matrix(
    y,
    predictions,
    labels=labels
)


# ============================================================
# SAVE PARTICIPANT-LEVEL PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({

    "participant_id":
        df["participant_id"],

    "actual_label":
        y,

    "predicted_label":
        predictions,

    "PD_probability":
        pd_probability

})


prediction_df.to_csv(
    PREDICTION_FILE,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame({

    "metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1-score",
        "ROC-AUC"
    ],

    "value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]

})


metrics_df.to_csv(
    METRICS_FILE,
    index=False
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_HC",
        "Actual_PD"
    ],
    columns=[
        "Predicted_HC",
        "Predicted_PD"
    ]
)

cm_df.to_csv(
    CONFUSION_FILE
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("FINAL CROSS-VALIDATED EVALUATION")
print("========================================")

print(
    f"\nAccuracy: "
    f"{accuracy:.4f}"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1-score: "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC: "
    f"{roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    cm_df.to_string()
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y,
        predictions,
        labels=labels,
        zero_division=0
    )
)


# ============================================================
# SAVE LOCATIONS
# ============================================================

print("\n========================================")
print("FILES SAVED")
print("========================================")

print(
    f"\nPredictions:\n"
    f"{PREDICTION_FILE}"
)

print(
    f"\nMetrics:\n"
    f"{METRICS_FILE}"
)

print(
    f"\nConfusion matrix:\n"
    f"{CONFUSION_FILE}"
)


print("\n========================================")
print("FINAL EVALUATION COMPLETED!")
print("========================================")