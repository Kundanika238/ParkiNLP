from reliability_service import assess_reliability


# ============================================================
# TEST 1 — CURRENT ATYPICAL SAMPLE
# ============================================================

transcript = (
    "Today I am working on my project and I am "
    "trying to understand how this speech analysis "
    "system works. I recorded my voice using the "
    "microphone and then the system converted my "
    "recording into text. After that, the text was "
    "analyzed using several language features. The "
    "system looks at sentence structure, word usage, "
    "filler words, and other characteristics of "
    "speech. I am testing the complete ParkiNLP "
    "pipeline before connecting everything to the "
    "website."
)


features = {

    "filler_count": 0,

    "short_sentence_ratio": 0.0,

    "type_token_ratio": 0.726027397260274,

    "sentence_count": 5,

    "filler_rate": 0.0

}


# ============================================================
# RUN TEST
# ============================================================

result = assess_reliability(
    transcript,
    features
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("========================================")
print("PARKINLP RELIABILITY TEST")
print("========================================")


print("\nReliability status:")
print(
    result["status"]
)


print("\nReliable:")
print(
    result["reliable"]
)


print("\nWord count:")
print(
    result["word_count"]
)


print("\nSentence count:")
print(
    result["sentence_count"]
)


print("\nMessage:")
print(
    result["message"]
)


print("\nIssues:")

if result["issues"]:

    for issue in result["issues"]:

        print(
            f"- {issue}"
        )

else:

    print(
        "None"
    )


print("\nCaution features:")

if result["caution_features"]:

    for feature in result[
        "caution_features"
    ]:

        print(
            f"- {feature}"
        )

else:

    print(
        "None"
    )


print("\nUnusual features:")

if result["unusual_features"]:

    for feature in result[
        "unusual_features"
    ]:

        print(
            f"- {feature}"
        )

else:

    print(
        "None"
    )


print("\n========================================")
print("FEATURE STATUS")
print("========================================")


for feature, check in result[
    "feature_checks"
].items():

    print(
        f"\n{feature}"
    )

    print(
        f"Value: "
        f"{check['value']}"
    )

    print(
        f"Z-score: "
        f"{check['z_score']:.4f}"
    )

    print(
        f"Training range: "
        f"{check['training_min']:.4f} "
        f"to "
        f"{check['training_max']:.4f}"
    )

    print(
        f"Status: "
        f"{check['status']}"
    )


print("\n========================================")
print("RELIABILITY TEST COMPLETED!")
print("========================================")