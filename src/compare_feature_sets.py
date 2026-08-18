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
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# DEFINE FEATURES
# ============================================================

id_columns = [
    "participant_id",
    "label"
]

all_features = [
    column
    for column in df.columns
    if column not in id_columns
]


selected_features = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio"
]


X_selected = df[selected_features]

X_all = df[all_features]

y = df["label"]


# ============================================================
# DISPLAY INFORMATION
# ============================================================

print("\nDataset information:")

print(f"Total NLP features: {len(all_features)}")

print(
    f"Baseline features: "
    f"{len(selected_features)}"
)

print("\nBaseline features:")

for feature in selected_features:
    print(f"- {feature}")


# ============================================================
# CREATE MODEL
# ============================================================

def create_model():

    return Pipeline([
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
# CROSS-VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


scoring = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro"
]


# ============================================================
# FUNCTION: EVALUATE FEATURE SET
# ============================================================

def evaluate_feature_set(X):

    model = create_model()

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    return {

        "accuracy": results[
            "test_accuracy"
        ].mean(),

        "precision": results[
            "test_precision_macro"
        ].mean(),

        "recall": results[
            "test_recall_macro"
        ].mean(),

        "f1": results[
            "test_f1_macro"
        ].mean(),

        "fold_accuracy": results[
            "test_accuracy"
        ]

    }


# ============================================================
# EVALUATE BASELINE
# ============================================================

print("\n========================================")
print("MODEL A: SELECTED FEATURES")
print("========================================")

baseline_results = evaluate_feature_set(
    X_selected
)


print(
    f"\nAccuracy: "
    f"{baseline_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{baseline_results['precision']:.4f}"
)

print(
    f"Recall: "
    f"{baseline_results['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{baseline_results['f1']:.4f}"
)


# ============================================================
# EVALUATE ALL FEATURES
# ============================================================

print("\n========================================")
print("MODEL B: ALL NLP FEATURES")
print("========================================")

all_results = evaluate_feature_set(
    X_all
)


print(
    f"\nAccuracy: "
    f"{all_results['accuracy']:.4f}"
)

print(
    f"Precision: "
    f"{all_results['precision']:.4f}"
)

print(
    f"Recall: "
    f"{all_results['recall']:.4f}"
)

print(
    f"F1-score: "
    f"{all_results['f1']:.4f}"
)


# ============================================================
# COMPARE RESULTS
# ============================================================

print("\n========================================")
print("FEATURE SET COMPARISON")
print("========================================")

accuracy_difference = (
    all_results["accuracy"]
    - baseline_results["accuracy"]
)

f1_difference = (
    all_results["f1"]
    - baseline_results["f1"]
)


print(
    f"\nAccuracy change: "
    f"{accuracy_difference:+.4f}"
)

print(
    f"F1-score change: "
    f"{f1_difference:+.4f}"
)


# ============================================================
# FOLD-WISE COMPARISON
# ============================================================

print("\nFold-wise accuracy comparison:")

for i in range(5):

    baseline_score = (
        baseline_results[
            "fold_accuracy"
        ][i]
    )

    all_score = (
        all_results[
            "fold_accuracy"
        ][i]
    )

    print(
        f"Fold {i + 1}: "
        f"Selected = {baseline_score:.4f}, "
        f"All = {all_score:.4f}"
    )


print("\n========================================")
print("FEATURE SET COMPARISON COMPLETED!")
print("========================================")