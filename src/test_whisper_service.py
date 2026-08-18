import os

from whisper_service import process_audio


# ============================================================
# TEST AUDIO
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ============================================================
# FIND LATEST WEBSITE RECORDING
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
    if filename.lower().endswith(".webm")
]

if not webm_files:
    raise FileNotFoundError(
        "No .webm recording found in uploads folder."
    )

audio_file = max(
    webm_files,
    key=os.path.getmtime
)


# ============================================================
# CHECK AUDIO EXISTS
# ============================================================

# ============================================================
# FIND LATEST WEBSITE RECORDING
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
    if filename.lower().endswith(".webm")
]

if not webm_files:

    raise FileNotFoundError(
        "No .webm recording found in uploads folder."
    )

AUDIO_FILE = max(
    webm_files,
    key=os.path.getmtime
)

print("Website recording found!")
print(f"Audio file: {AUDIO_FILE}")

if not os.path.exists(AUDIO_FILE):

    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_FILE}"
    )


print("Website recording found!")
print(f"Audio file: {AUDIO_FILE}")


# ============================================================
# PROCESS AUDIO
# ============================================================

result = process_audio(
    AUDIO_FILE
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print("\n========================================")
print("WHISPER SERVICE TEST")
print("========================================")

print("\nConverted WAV:")
print(result["wav_path"])

print("\nTranscript:")
print(result["transcript"])

print("\n========================================")
print("WHISPER SERVICE TEST COMPLETED!")
print("========================================")