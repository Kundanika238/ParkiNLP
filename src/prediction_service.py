import os
import joblib
import pandas as pd

from src.extract_text_features import extract_features


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_FILE = os.path.join(
    BASE_DIR,
    "models",
    "parkinlp_model.joblib"
)


# ============================================================
# FINAL STABLE FEATURES
# ============================================================

SELECTED_FEATURES = [

    "filler_count",

    "short_sentence_ratio",

    "type_token_ratio",

    "sentence_count",

    "filler_rate"

]


# ============================================================
# LOAD DEPLOYMENT MODEL
# ============================================================

print(
    "Loading ParkiNLP deployment model..."
)


model = joblib.load(
    MODEL_FILE
)


print(
    "ParkiNLP deployment model loaded successfully!"
)


# ============================================================
# EXTRACT FINAL FEATURES
# ============================================================

def extract_stable_features(
    transcript
):
    """
    Extract the complete NLP feature set
    and return only the five stable
    features used by the final model.
    """

    all_features = extract_features(
        transcript
    )


    stable_features = {

        feature:
            all_features[feature]

        for feature in SELECTED_FEATURES

    }


    return stable_features


# ============================================================
# PREDICT FROM TRANSCRIPT
# ============================================================

def predict_from_transcript(
    transcript
):
    """
    Generate an exploratory HC/PD prediction
    from a transcript.
    """

    transcript = transcript.strip()


    if not transcript:

        raise ValueError(
            "Transcript is empty."
        )


    # --------------------------------------------------------
    # Extract NLP features
    # --------------------------------------------------------

    stable_features = (
        extract_stable_features(
            transcript
        )
    )


    # --------------------------------------------------------
    # Convert features into DataFrame
    # --------------------------------------------------------

    feature_df = pd.DataFrame(
        [stable_features],
        columns=SELECTED_FEATURES
    )


    # --------------------------------------------------------
    # Generate prediction
    # --------------------------------------------------------

    prediction = model.predict(
        feature_df
    )[0]


    # --------------------------------------------------------
    # Generate probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        feature_df
    )[0]


    class_order = list(
        model.classes_
    )


    pd_index = class_order.index(
        "PD"
    )


    hc_index = class_order.index(
        "HC"
    )


    pd_probability = float(
        probabilities[pd_index]
    )


    hc_probability = float(
        probabilities[hc_index]
    )


    # --------------------------------------------------------
    # Return complete result
    # --------------------------------------------------------

    return {

        "prediction":
            str(prediction),

        "pd_probability":
            pd_probability,

        "hc_probability":
            hc_probability,

        "features":
            stable_features

    }