import os
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "results",
    "nlp_features.csv"
)


# ============================================================
# FEATURES USED BY FINAL MODEL
# ============================================================

SELECTED_FEATURES = [
    "filler_count",
    "short_sentence_ratio",
    "type_token_ratio",
    "sentence_count",
    "filler_rate"
]


# ============================================================
# LOAD TRAINING FEATURE DISTRIBUTION
# ============================================================

training_df = pd.read_csv(
    FEATURE_FILE
)


# ============================================================
# RELIABILITY SETTINGS
# ============================================================

MIN_WORD_COUNT = 40

MIN_SENTENCE_COUNT = 5


# ------------------------------------------------------------
# Z-SCORE THRESHOLDS
# ------------------------------------------------------------

RELIABLE_Z_SCORE = 2.0

CAUTION_Z_SCORE = 3.0

LOW_RELIABILITY_Z_SCORE = 3.0


# ============================================================
# FEATURE DISTRIBUTION
# ============================================================

FEATURE_STATISTICS = {}


for feature in SELECTED_FEATURES:

    values = training_df[
        feature
    ].dropna()

    FEATURE_STATISTICS[
        feature
    ] = {

        "mean":
            float(values.mean()),

        "std":
            float(values.std()),

        "min":
            float(values.min()),

        "max":
            float(values.max())

    }


# ============================================================
# CHECK SINGLE FEATURE
# ============================================================

def check_feature(
    feature,
    value
):

    statistics = FEATURE_STATISTICS[
        feature
    ]

    mean = statistics["mean"]

    std = statistics["std"]

    minimum = statistics["min"]

    maximum = statistics["max"]


    # --------------------------------------------------------
    # Z-SCORE
    # --------------------------------------------------------

    if std > 0:

        z_score = (
            value - mean
        ) / std

    else:

        z_score = 0.0


    absolute_z_score = abs(
        z_score
    )


    # --------------------------------------------------------
    # RANGE
    # --------------------------------------------------------

    inside_range = (
        minimum
        <= value
        <= maximum
    )


    # --------------------------------------------------------
    # FEATURE STATUS
    # --------------------------------------------------------

    if absolute_z_score <= RELIABLE_Z_SCORE:

        status = "NORMAL"

    elif absolute_z_score <= CAUTION_Z_SCORE:

        status = "CAUTION"

    else:

        status = "UNUSUAL"


    return {

        "value":
            float(value),

        "training_min":
            minimum,

        "training_max":
            maximum,

        "z_score":
            float(z_score),

        "inside_training_range":
            inside_range,

        "status":
            status

    }


# ============================================================
# CHECK TRANSCRIPT RELIABILITY
# ============================================================

def assess_reliability(
    transcript,
    features
):
    """
    Assess whether a transcript contains enough
    information and whether its NLP feature profile
    resembles the training distribution.
    """

    transcript = transcript.strip()


    # ========================================================
    # BASIC TEXT STATISTICS
    # ========================================================

    words = transcript.split()

    word_count = len(words)

    sentence_count = int(
        features.get(
            "sentence_count",
            0
        )
    )


    # ========================================================
    # ISSUE COLLECTION
    # ========================================================

    issues = []

    caution_features = []

    unusual_features = []


    # ========================================================
    # WORD COUNT
    # ========================================================

    if word_count < MIN_WORD_COUNT:

        issues.append(
            f"Speech sample is too short. "
            f"At least {MIN_WORD_COUNT} words "
            f"are recommended."
        )


    # ========================================================
    # SENTENCE COUNT
    # ========================================================

    if sentence_count < MIN_SENTENCE_COUNT:

        issues.append(
            f"Speech sample contains only "
            f"{sentence_count} sentence(s). "
            f"Please provide a longer continuous "
            f"speech sample."
        )


    # ========================================================
    # FEATURE CHECKS
    # ========================================================

    feature_checks = {}


    for feature in SELECTED_FEATURES:

        value = features.get(
            feature
        )


        if value is None:

            continue


        check = check_feature(
            feature,
            float(value)
        )


        feature_checks[
            feature
        ] = check


        # ----------------------------------------------------
        # CAUTION
        # ----------------------------------------------------

        if check["status"] == "CAUTION":

            caution_features.append(
                feature
            )


        # ----------------------------------------------------
        # UNUSUAL
        # ----------------------------------------------------

        if check["status"] == "UNUSUAL":

            unusual_features.append(
                feature
            )


    # ========================================================
    # DETERMINE FINAL RELIABILITY LEVEL
    # ========================================================

    # --------------------------------------------------------
    # LOW RELIABILITY
    #
    # Hard failures:
    # - Too few words
    # - Too few sentences
    #
    # Unusual NLP features DO NOT automatically
    # block the prediction.
    # --------------------------------------------------------

    if (
        word_count < MIN_WORD_COUNT
        or sentence_count < MIN_SENTENCE_COUNT
    ):

        reliability_level = (
            "LOW_RELIABILITY"
        )

        reliable = False


    # --------------------------------------------------------
    # CAUTION
    #
    # The speech is long enough, but one or more
    # linguistic features differ from the training
    # distribution.
    # --------------------------------------------------------

    elif (
        len(unusual_features) >= 1
        or len(caution_features) >= 1
    ):

        reliability_level = (
            "CAUTION"
        )

        reliable = True


    # --------------------------------------------------------
    # NORMAL
    # --------------------------------------------------------

    else:

        reliability_level = (
            "RELIABLE"
        )

        reliable = True


    # ========================================================
    # USER MESSAGE
    # ========================================================

    if reliability_level == "RELIABLE":

        message = (
            "The speech sample contains sufficient "
            "information and its NLP feature profile "
            "is reasonably consistent with the "
            "training distribution."
        )


    elif reliability_level == "CAUTION":

        message = (
            "The speech sample contains sufficient "
            "information for analysis, but some "
            "linguistic features differ from the "
            "typical training distribution. "
            "Interpret the model output with caution."
        )


    else:

        message = (
            "The speech sample should not be treated "
            "as a reliable model input yet. "
            "Please provide a longer and more natural "
            "continuous speech sample."
        )


    # ========================================================
    # RETURN RESULT
    # ========================================================

    return {

        "reliable":
            reliable,

        "status":
            reliability_level,

        "message":
            message,

        "word_count":
            word_count,

        "sentence_count":
            sentence_count,

        "issues":
            issues,

        "caution_features":
            caution_features,

        "unusual_features":
            unusual_features,

        "feature_checks":
            feature_checks

    }