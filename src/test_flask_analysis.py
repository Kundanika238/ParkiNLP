import os
import requests


# ============================================================
# FLASK SERVER
# ============================================================

SERVER_URL = (
    "http://127.0.0.1:4000/analyze-audio"
)


# ============================================================
# TEST AUDIO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FIND LATEST AUDIO RECORDING
# ============================================================

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "uploads"
)


webm_files = [

    os.path.join(
        UPLOAD_FOLDER,
        filename
    )

    for filename in os.listdir(
        UPLOAD_FOLDER
    )

    if filename.lower().endswith(
        ".webm"
    )

]


if not webm_files:

    raise FileNotFoundError(
        "No .webm recording found in uploads folder."
    )


AUDIO_FILE = max(
    webm_files,
    key=os.path.getmtime
)


# ============================================================
# CHECK AUDIO
# ============================================================

if not os.path.exists(
    AUDIO_FILE
):

    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_FILE}"
    )


print(
    "Test audio found:"
)

print(
    AUDIO_FILE
)


# ============================================================
# SEND AUDIO TO FLASK
# ============================================================

print(
    "\nSending audio to Flask..."
)


with open(
    AUDIO_FILE,
    "rb"
) as audio:

    response = requests.post(

        SERVER_URL,

        files={
            "audio": (
                "test_recording.webm",
                audio,
                "audio/webm"
            )
        }

    )


# ============================================================
# DISPLAY RESPONSE
# ============================================================

print(
    "\n========================================"
)

print(
    "FLASK ANALYSIS RESPONSE"
)

print(
    "========================================"
)


print(
    "\nHTTP status:"
)

print(
    response.status_code
)


print(
    "\nJSON response:"
)

try:

    print(
        response.json()
    )

except Exception:

    print(
        response.text
    )


print(
    "\n========================================"
)

print(
    "FLASK ANALYSIS TEST COMPLETED!"
)

print(
    "========================================"
)