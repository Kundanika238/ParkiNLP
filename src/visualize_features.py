import os
import pandas as pd
import matplotlib.pyplot as plt


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
    "results",
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("Feature dataset loaded successfully!")

print(f"Total participants: {len(df)}")


# ============================================================
# FEATURES TO VISUALIZE
# ============================================================

features_to_plot = [
    "filler_count",
    "filler_rate",
    "short_sentence_ratio",
    "type_token_ratio",
    "average_sentence_length",
    "immediate_repetition_rate"
]


# ============================================================
# CREATE BOXPLOTS
# ============================================================

for feature in features_to_plot:

    hc_values = df[
        df["label"] == "HC"
    ][feature]

    pd_values = df[
        df["label"] == "PD"
    ][feature]

    plt.figure(figsize=(7, 5))

    plt.boxplot(
        [
            hc_values,
            pd_values
        ],
        tick_labels=[
            "Healthy Control",
            "Parkinson's"
        ]
    )

    plt.title(
        f"HC vs PD: {feature}"
    )

    plt.ylabel(feature)

    plt.tight_layout()

    output_file = os.path.join(
        OUTPUT_DIR,
        f"{feature}.png"
    )

    plt.savefig(
        output_file,
        dpi=300
    )

    plt.close()

    print(
        f"Saved plot: {output_file}"
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("VISUALIZATION COMPLETED!")
print("========================================")

print(
    f"\nPlots saved in:\n{OUTPUT_DIR}"
)