import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
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
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# PREPARE DATA
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

X = df[feature_columns]

y = df["label"]


print(
    f"Total available NLP features: "
    f"{len(feature_columns)}"
)

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# CREATE LEAKAGE-SAFE PIPELINE
# ============================================================

model = Pipeline([
    
    (
        "scaler",
        StandardScaler()
    ),

    (
        "feature_selection",
        SelectKBest(
            score_func=f_classif,
            k=5
        )
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
# EVALUATION
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
print("LEAKAGE-SAFE CLASSIFICATION")
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


# ============================================================
# FOLD-WISE RESULTS
# ============================================================

print("\nFold-wise accuracy:")

for i, score in enumerate(
    results["test_accuracy"],
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.4f}"
    )


print("\n========================================")
print("LEAKAGE-SAFE EVALUATION COMPLETED!")
print("========================================")