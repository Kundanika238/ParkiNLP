import os
import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import RepeatedStratifiedKFold


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "feature_selection_stability.csv"
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


# ============================================================
# REPEATED STRATIFIED CROSS-VALIDATION
# ============================================================

cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)


# ============================================================
# COUNT FEATURE SELECTION
# ============================================================

selection_counts = {
    feature: 0
    for feature in feature_columns
}


total_folds = 0


for train_indices, test_indices in cv.split(X, y):

    X_train = X.iloc[train_indices]

    y_train = y.iloc[train_indices]

    # --------------------------------------------------------
    # Remove constant features inside the training fold
    # --------------------------------------------------------

    variable_features = [
        column
        for column in feature_columns
        if X_train[column].nunique() > 1
    ]

    X_train_variable = X_train[
        variable_features
    ]

    # --------------------------------------------------------
    # Select top 5 features
    # --------------------------------------------------------

    selector = SelectKBest(
        score_func=f_classif,
        k=min(5, len(variable_features))
    )

    selector.fit(
        X_train_variable,
        y_train
    )

    selected_mask = selector.get_support()

    selected_features = X_train_variable.columns[
        selected_mask
    ]

    # --------------------------------------------------------
    # Update selection counts
    # --------------------------------------------------------

    for feature in selected_features:

        selection_counts[
            feature
        ] += 1

    total_folds += 1


# ============================================================
# CREATE STABILITY TABLE
# ============================================================

stability_results = []


for feature in feature_columns:

    count = selection_counts[feature]

    selection_rate = (
        count / total_folds
    )

    stability_results.append({

        "feature": feature,

        "selection_count": count,

        "total_folds": total_folds,

        "selection_rate": selection_rate

    })


stability_df = pd.DataFrame(
    stability_results
)


# ============================================================
# SORT BY STABILITY
# ============================================================

stability_df = stability_df.sort_values(
    by=[
        "selection_rate",
        "selection_count"
    ],
    ascending=False
)


# ============================================================
# SAVE RESULTS
# ============================================================

stability_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("FEATURE SELECTION STABILITY")
print("========================================")

print(
    f"\nTotal evaluation folds: "
    f"{total_folds}"
)

print("\nFeatures ranked by selection stability:\n")

print(
    stability_df.to_string(
        index=False
    )
)


print("\n========================================")
print("FEATURE STABILITY ANALYSIS COMPLETED!")
print("========================================")

print(
    f"\nResults saved to:\n"
    f"{OUTPUT_FILE}"
)