import os
import pandas as pd
from scipy.stats import mannwhitneyu

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

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "results",
    "statistical_analysis.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(FEATURE_FILE)

print("NLP feature dataset loaded successfully!")

print(f"Total participants: {len(df)}")

print(
    f"Healthy controls: "
    f"{(df['label'] == 'HC').sum()}"
)

print(
    f"Parkinson's participants: "
    f"{(df['label'] == 'PD').sum()}"
)


# ============================================================
# SEPARATE GROUPS
# ============================================================

hc_df = df[df["label"] == "HC"]

pd_df = df[df["label"] == "PD"]


# ============================================================
# FEATURE LIST
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


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

results = []


for feature in feature_columns:

    hc_values = hc_df[feature].dropna()

    pd_values = pd_df[feature].dropna()

    # --------------------------------------------------------
    # Mann-Whitney U test
    # --------------------------------------------------------

    statistic, p_value = mannwhitneyu(
        hc_values,
        pd_values,
        alternative="two-sided"
    )

    # --------------------------------------------------------
    # Medians
    # --------------------------------------------------------

    hc_median = hc_values.median()

    pd_median = pd_values.median()

    median_difference = pd_median - hc_median

    # --------------------------------------------------------
    # Rank-biserial effect size
    # --------------------------------------------------------

    n_hc = len(hc_values)

    n_pd = len(pd_values)

    rank_biserial = (
        (2 * statistic)
        / (n_hc * n_pd)
    ) - 1

    # --------------------------------------------------------
    # Interpretation
    # --------------------------------------------------------

    if p_value < 0.05:

        significance = "Potentially significant"

    else:

        significance = "Not significant"

    results.append({

        "feature": feature,

        "HC_median": hc_median,

        "PD_median": pd_median,

        "median_difference": median_difference,

        "U_statistic": statistic,

        "p_value": p_value,

        "rank_biserial_effect": rank_biserial,

        "significance": significance

    })


# ============================================================
# CREATE RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================

results_df["p_value_bonferroni"] = (
    results_df["p_value"]
    * len(results_df)
)

results_df["p_value_bonferroni"] = (
    results_df["p_value_bonferroni"]
    .clip(upper=1.0)
)


# ============================================================
# SORT BY P-VALUE
# ============================================================

results_df = results_df.sort_values(
    by="p_value"
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("STATISTICAL ANALYSIS")
print("========================================")

print(
    "\nFeatures sorted by p-value:\n"
)

display_columns = [
    "feature",
    "HC_median",
    "PD_median",
    "median_difference",
    "p_value",
    "p_value_bonferroni",
    "rank_biserial_effect",
    "significance"
]

print(
    results_df[
        display_columns
    ].to_string(index=False)
)


print("\n========================================")
print("STATISTICAL ANALYSIS COMPLETED!")
print("========================================")

print(
    f"\nResults saved to:\n{OUTPUT_FILE}"
)