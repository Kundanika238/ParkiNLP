from src.prediction_service import predict_from_transcript


# ============================================================
# TEST TRANSCRIPT
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


print("========================================")
print("PARKINLP PREDICTION SERVICE TEST")
print("========================================")


print("\nInput transcript:")
print(transcript)


# ============================================================
# RUN PREDICTION
# ============================================================

result = predict_from_transcript(
    transcript
)


# ============================================================
# DISPLAY PREDICTION
# ============================================================

print("\nPrediction:")
print(
    result["prediction"]
)


print("\nHC probability:")
print(
    f"{result['hc_probability']:.4f}"
)


print("\nPD probability:")
print(
    f"{result['pd_probability']:.4f}"
)


# ============================================================
# DISPLAY STABLE FEATURES
# ============================================================

print("\nStable NLP features:")

for feature, value in result["features"].items():

    print(
        f"{feature}: {value}"
    )


print("\n========================================")
print("PREDICTION SERVICE TEST COMPLETED!")
print("========================================")