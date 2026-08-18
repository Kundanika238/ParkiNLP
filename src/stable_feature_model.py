import os
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate


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

STABILITY_FILE = os.path.join(
    BASE_DIR,
    "results",
    "feature_selection_stability.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

stability_df = pd.read_csv(
    STABILITY_FILE
)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# SELECT STABLE FEATURES
# ============================================================

stable_features = stability_df[
    stability_df["selection_rate"] >= 0.50
]["feature"].tolist()


print("\nStable features selected:")

for feature in stable_features:
    print(f"- {feature}")


print(
    f"\nTotal stable features: "
    f"{len(stable_features)}"
)


# ============================================================
# PREPARE DATA
# ============================================================

X = df[stable_features]

y = df["label"]


print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# CREATE MODEL
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
# REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
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
# CALCULATE RESULTS
# ============================================================

accuracy_mean = results[
    "test_accuracy"
].mean()

accuracy_std = results[
    "test_accuracy"
].std()

precision_mean = results[
    "test_precision_macro"
].mean()

precision_std = results[
    "test_precision_macro"
].std()

recall_mean = results[
    "test_recall_macro"
].mean()

recall_std = results[
    "test_recall_macro"
].std()

f1_mean = results[
    "test_f1_macro"
].mean()

f1_std = results[
    "test_f1_macro"
].std()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("STABLE FEATURE MODEL")
print("========================================")


print(
    f"\nAccuracy: "
    f"{accuracy_mean:.4f} "
    f"+/- {accuracy_std:.4f}"
)

print(
    f"Precision: "
    f"{precision_mean:.4f} "
    f"+/- {precision_std:.4f}"
)

print(
    f"Recall: "
    f"{recall_mean:.4f} "
    f"+/- {recall_std:.4f}"
)

print(
    f"F1-score: "
    f"{f1_mean:.4f} "
    f"+/- {f1_std:.4f}"
)


# ============================================================
# NUMBER OF EVALUATIONS
# ============================================================

total_evaluations = len(
    results["test_accuracy"]
)

print(
    f"\nTotal model evaluations: "
    f"{total_evaluations}"
)


print("\n========================================")
print("STABLE FEATURE MODEL COMPLETED!")
print("========================================")