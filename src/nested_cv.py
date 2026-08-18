import os
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
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
    f"Total NLP features: "
    f"{len(feature_columns)}"
)

print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# OUTER CROSS-VALIDATION
# ============================================================

outer_cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# STORE OUTER RESULTS
# ============================================================

outer_accuracy = []
outer_f1 = []
outer_precision = []
outer_recall = []


# ============================================================
# NESTED CROSS-VALIDATION
# ============================================================

print("\n========================================")
print("STARTING NESTED CROSS-VALIDATION")
print("========================================")


for fold_number, (train_index, test_index) in enumerate(
    outer_cv.split(X, y),
    start=1
):

    print(
        f"\nProcessing outer fold "
        f"{fold_number}/5..."
    )


    # --------------------------------------------------------
    # Split outer training and test data
    # --------------------------------------------------------

    X_train = X.iloc[train_index]

    X_test = X.iloc[test_index]

    y_train = y.iloc[train_index]

    y_test = y.iloc[test_index]


    # --------------------------------------------------------
    # Inner cross-validation
    # --------------------------------------------------------

    inner_cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=42
    )


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Inner CV model selection
    # --------------------------------------------------------

    inner_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=inner_cv,
        scoring="accuracy"
    )


    print(
        f"Inner CV accuracy: "
        f"{inner_scores.mean():.4f}"
    )


    # --------------------------------------------------------
    # Train on complete outer training set
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Evaluate on completely unseen outer test set
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Calculate metrics
    # --------------------------------------------------------

    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score
    )


    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )


    outer_accuracy.append(
        accuracy
    )

    outer_precision.append(
        precision
    )

    outer_recall.append(
        recall
    )

    outer_f1.append(
        f1
    )


    print(
        f"Outer accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"Outer precision: "
        f"{precision:.4f}"
    )

    print(
        f"Outer recall: "
        f"{recall:.4f}"
    )

    print(
        f"Outer F1-score: "
        f"{f1:.4f}"
    )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n========================================")
print("NESTED CROSS-VALIDATION RESULTS")
print("========================================")


print(
    f"\nAccuracy: "
    f"{np.mean(outer_accuracy):.4f} "
    f"+/- {np.std(outer_accuracy):.4f}"
)

print(
    f"Precision: "
    f"{np.mean(outer_precision):.4f} "
    f"+/- {np.std(outer_precision):.4f}"
)

print(
    f"Recall: "
    f"{np.mean(outer_recall):.4f} "
    f"+/- {np.std(outer_recall):.4f}"
)

print(
    f"F1-score: "
    f"{np.mean(outer_f1):.4f} "
    f"+/- {np.std(outer_f1):.4f}"
)


# ============================================================
# FOLD-WISE RESULTS
# ============================================================

print("\nFold-wise outer accuracy:")

for i, score in enumerate(
    outer_accuracy,
    start=1
):

    print(
        f"Fold {i}: "
        f"{score:.4f}"
    )


print("\n========================================")
print("NESTED CV COMPLETED!")
print("========================================")