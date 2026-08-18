import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate


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


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# SELECT FEATURES
# ============================================================

selected_features = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio"
]


X = df[selected_features]

y = df["label"]


# ============================================================
# DISPLAY DATA INFORMATION
# ============================================================

print("\nSelected features:")

for feature in selected_features:
    print(f"- {feature}")


print("\nClass distribution:")

print(y.value_counts())


# ============================================================
# CREATE MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline([
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
# CROSS-VALIDATION
# ============================================================

scoring = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]


results = cross_validate(
    model,
    X,
    y,
    cv=cv,
    scoring=scoring
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("BASELINE CLASSIFICATION RESULTS")
print("========================================")


print(
    f"\nAccuracy: "
    f"{results['test_accuracy'].mean():.4f}"
)


print(
    f"Precision: "
    f"{results['test_precision_macro'].mean():.4f}"
)


print(
    f"Recall: "
    f"{results['test_recall_macro'].mean():.4f}"
)


print(
    f"F1-score: "
    f"{results['test_f1_macro'].mean():.4f}"
)


print("\nFold-wise accuracy:")

for i, score in enumerate(
    results["test_accuracy"],
    start=1
):
    print(
        f"Fold {i}: {score:.4f}"
    )


print("\n========================================")
print("CROSS-VALIDATION COMPLETED!")
print("========================================")