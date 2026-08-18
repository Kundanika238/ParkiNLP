import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PREDICTION_FILE = os.path.join(
    BASE_DIR,
    "results",
    "final_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    "final_plots"
)

CONFUSION_PLOT = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

ROC_PLOT = os.path.join(
    OUTPUT_DIR,
    "roc_curve.png"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTION_FILE
)

print(
    "Final prediction dataset loaded successfully!"
)

print(
    f"Total participants: {len(df)}"
)


# ============================================================
# PREPARE LABELS
# ============================================================

actual = df["actual_label"]

predicted = df["predicted_label"]

pd_probability = df["PD_probability"]


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = [
    "HC",
    "PD"
]

cm = confusion_matrix(
    actual,
    predicted,
    labels=labels
)


# ============================================================
# CREATE CONFUSION MATRIX FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

image = ax.imshow(
    cm
)

ax.set_xticks(
    np.arange(len(labels))
)

ax.set_yticks(
    np.arange(len(labels))
)

ax.set_xticklabels(
    labels
)

ax.set_yticklabels(
    labels
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_title(
    "Confusion Matrix - NLP Parkinson's Classification"
)


# ============================================================
# ADD VALUES TO CELLS
# ============================================================

for i in range(
    len(labels)
):

    for j in range(
        len(labels)
    ):

        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=14
        )


plt.tight_layout()

plt.savefig(
    CONFUSION_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved confusion matrix:\n"
    f"{CONFUSION_PLOT}"
)


# ============================================================
# ROC CURVE
# ============================================================

actual_binary = (
    actual == "PD"
).astype(int)


roc_auc = roc_auc_score(
    actual_binary,
    pd_probability
)


false_positive_rate, true_positive_rate, thresholds = (
    roc_curve(
        actual_binary,
        pd_probability
    )
)


# ============================================================
# CREATE ROC FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

ax.plot(
    false_positive_rate,
    true_positive_rate,
    linewidth=2,
    label=f"ROC curve (AUC = {roc_auc:.4f})"
)

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1
)

ax.set_xlabel(
    "False Positive Rate"
)

ax.set_ylabel(
    "True Positive Rate"
)

ax.set_title(
    "ROC Curve - NLP Parkinson's Classification"
)

ax.legend(
    loc="lower right"
)

ax.set_xlim(
    0,
    1
)

ax.set_ylim(
    0,
    1.05
)

plt.tight_layout()

plt.savefig(
    ROC_PLOT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    f"Saved ROC curve:\n"
    f"{ROC_PLOT}"
)


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\n========================================")
print("FINAL VISUALIZATION COMPLETED!")
print("========================================")

print(
    f"\nROC-AUC: "
    f"{roc_auc:.4f}"
)

print(
    f"\nPlots saved in:\n"
    f"{OUTPUT_DIR}"
)