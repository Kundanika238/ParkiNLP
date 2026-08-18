import os
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "parkinlp_model.joblib"
)


# ============================================================
# CREATE MODEL DIRECTORY
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# LOAD NLP FEATURE DATASET
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")
print(f"Total participants: {len(df)}")


# ============================================================
# FINAL RESEARCH FEATURE SET
# ============================================================

selected_features = [
    "filler_count",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "filler_rate"
]


print("\nDeployment feature set:")

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
# PREPARE TRAINING DATA
# ============================================================

X = df[selected_features]

y = df["label"]


print("\nClass distribution:")
print(y.value_counts())


# ============================================================
# CREATE FINAL DEPLOYMENT PIPELINE
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
# TRAIN MODEL ON COMPLETE RESEARCH DATASET
# ============================================================

print("\nTraining deployment model...")

model.fit(
    X,
    y
)

print("Deployment model trained successfully!")


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_FILE
)


# ============================================================
# VERIFY SAVED MODEL
# ============================================================

print("\nTesting saved model...")

loaded_model = joblib.load(
    MODEL_FILE
)

test_predictions = loaded_model.predict(X)

test_probabilities = loaded_model.predict_proba(X)


print(
    f"Test predictions generated: "
    f"{len(test_predictions)}"
)

print(
    f"Probability matrix shape: "
    f"{test_probabilities.shape}"
)


# ============================================================
# DISPLAY MODEL INFORMATION
# ============================================================

print("\n========================================")
print("DEPLOYMENT MODEL CREATED")
print("========================================")

print(
    "\nModel pipeline:"
)

print(
    "1. Median imputation"
)

print(
    "2. Standard scaling"
)

print(
    "3. Logistic regression"
)

print(
    "\nFeatures used:"
)

for feature in selected_features:
    print(
        f"- {feature}"
    )


print(
    "\nModel saved to:"
)

print(
    MODEL_FILE
)

print("\n========================================")
print("DEPLOYMENT MODEL READY!")
print("========================================")