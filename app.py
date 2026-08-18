from flask import Flask, render_template, request, jsonify

import os
from datetime import datetime

import joblib
import pandas as pd

from src.whisper_service import process_audio
from src.prediction_service import (
    extract_stable_features,
    predict_from_transcript
)
from src.reliability_service import assess_reliability


# ============================================================
# CREATE FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# UPLOAD FOLDER
# ============================================================

UPLOAD_FOLDER = os.path.join(
    app.root_path,
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)


# ============================================================
# DEPLOYMENT MODEL
# ============================================================

MODEL_FILE = os.path.join(
    app.root_path,
    "models",
    "parkinlp_model.joblib"
)

model = joblib.load(
    MODEL_FILE
)

print(
    "ParkiNLP deployment model loaded successfully!"
)


# ============================================================
# HOME
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# ANALYZE PAGE
# ============================================================

@app.route("/analyze")
def analyze():

    return render_template(
        "analyze.html"
    )


# ============================================================
# AUDIO UPLOAD
# ============================================================

@app.route(
    "/upload-audio",
    methods=["POST"]
)
def upload_audio():

    if "audio" not in request.files:

        return jsonify({

            "success": False,

            "message":
                "No audio file received."

        }), 400


    audio_file = request.files[
        "audio"
    ]


    if audio_file.filename == "":

        return jsonify({

            "success": False,

            "message":
                "No audio file selected."

        }), 400


    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


    filename = (
        f"recording_{timestamp}.webm"
    )


    filepath = os.path.join(
        UPLOAD_FOLDER,
        filename
    )


    audio_file.save(
        filepath
    )


    return jsonify({

        "success": True,

        "message":
            "Audio uploaded successfully.",

        "filename":
            filename

    })


# ============================================================
# COMPLETE AUDIO ANALYSIS
# ============================================================

@app.route(
    "/analyze-audio",
    methods=["POST"]
)
def analyze_audio():

    try:

        # ====================================================
        # CHECK AUDIO
        # ====================================================

        if "audio" not in request.files:

            return jsonify({

                "success": False,

                "message":
                    "No audio file received."

            }), 400


        audio_file = request.files[
            "audio"
        ]


        if audio_file.filename == "":

            return jsonify({

                "success": False,

                "message":
                    "No audio file selected."

            }), 400


        # ====================================================
        # SAVE WEBM
        # ====================================================

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )


        filename = (
            f"recording_{timestamp}.webm"
        )


        filepath = os.path.join(
            UPLOAD_FOLDER,
            filename
        )


        audio_file.save(
            filepath
        )


        print(
            "\n========================================"
        )

        print(
            "NEW PARKINLP AUDIO ANALYSIS"
        )

        print(
            "========================================"
        )


        print(
            f"\nAudio saved:\n{filepath}"
        )


        # ====================================================
        # WHISPER + FFMPEG
        # ====================================================

        print(
            "\nRunning Whisper pipeline..."
        )


        whisper_result = process_audio(
            filepath
        )


        transcript = whisper_result[
            "transcript"
        ]


        wav_path = whisper_result[
            "wav_path"
        ]


        print(
            f"\nTranscript:\n{transcript}"
        )


        # ====================================================
        # EMPTY TRANSCRIPT CHECK
        # ====================================================

        if not transcript.strip():

            return jsonify({

                "success": False,

                "stage":
                    "transcription",

                "message":
                    "Whisper could not detect "
                    "usable speech in the recording."

            }), 400


        # ====================================================
        # NLP FEATURE EXTRACTION
        # ====================================================

        print(
            "\nExtracting NLP features..."
        )


        stable_features = (
            extract_stable_features(
                transcript
            )
        )


        print(
            "\nStable NLP features:"
        )


        for feature, value in (
            stable_features.items()
        ):

            print(
                f"{feature}: {value}"
            )


        # ====================================================
        # RELIABILITY ANALYSIS
        # ====================================================

        print(
            "\nRunning reliability analysis..."
        )


        reliability = assess_reliability(
            transcript,
            stable_features
        )


        print(
            "\nReliability status: "
            f"{reliability['status']}"
        )


        # ====================================================
        # LOW RELIABILITY
        # ====================================================

        if reliability[
            "status"
        ] == "LOW_RELIABILITY":

            return jsonify({

                "success": True,

                "analysis_complete":
                    True,

                "prediction_available":
                    False,

                "reliability":
                    reliability,

                "transcript":
                    transcript,

                "features":
                    stable_features,

                "audio":
                    {
                        "webm_filename":
                            filename,

                        "wav_path":
                            wav_path
                    },

                "message":
                    "The speech sample requires "
                    "a better-quality recording "
                    "before model prediction."

            })


        # ====================================================
        # MODEL PREDICTION
        # ====================================================

        print(
            "\nRunning ParkiNLP model..."
        )


        prediction_result = (
            predict_from_transcript(
                transcript
            )
        )


        print(
            "\nPrediction: "
            + prediction_result[
                "prediction"
            ]
        )


        print(
            "PD probability: "
            + f"{prediction_result['pd_probability']:.4f}"
        )


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return jsonify({

            "success": True,

            "analysis_complete":
                True,

            "prediction_available":
                True,

            "prediction":
                prediction_result[
                    "prediction"
                ],

            "hc_probability":
                round(
                    prediction_result[
                        "hc_probability"
                    ],
                    4
                ),

            "pd_probability":
                round(
                    prediction_result[
                        "pd_probability"
                    ],
                    4
                ),

            "reliability":
                reliability,

            "transcript":
                transcript,

            "features":
                prediction_result[
                    "features"
                ],

            "audio":
                {
                    "webm_filename":
                        filename,

                    "wav_path":
                        wav_path
                },

            "message":
                "Speech analysis completed. "
                "This is an exploratory research-model "
                "output and is not a medical diagnosis."

        })


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        print(
            "\nANALYSIS ERROR:"
        )

        print(
            str(error)
        )


        return jsonify({

            "success": False,

            "message":
                "An error occurred while "
                "processing the speech.",

            "error":
                str(error)

        }), 500


# ============================================================
# MODEL TEST
# ============================================================

@app.route("/test-model")
def test_model():

    test_features = pd.DataFrame([{

        "filler_count": 8,

        "short_sentence_ratio": 0.45,

        "type_token_ratio": 0.50,

        "sentence_count": 30,

        "filler_rate": 0.03

    }])


    prediction = model.predict(
        test_features
    )[0]


    probabilities = model.predict_proba(
        test_features
    )[0]


    class_order = list(
        model.classes_
    )


    pd_index = class_order.index(
        "PD"
    )


    pd_probability = probabilities[
        pd_index
    ]


    return jsonify({

        "success": True,

        "prediction":
            prediction,

        "pd_probability":
            round(
                float(
                    pd_probability
                ),
                4
            )

    })


# ============================================================
# RESEARCH PAGE
# ============================================================

@app.route("/research")
def research():

    return render_template(
        "research.html"
    )


# ============================================================
# ABOUT PAGE
# ============================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=4000
    )