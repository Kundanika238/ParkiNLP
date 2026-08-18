import os
import subprocess

import whisper


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEMP_AUDIO_DIR = os.path.join(
    BASE_DIR,
    "uploads",
    "processed"
)

os.makedirs(
    TEMP_AUDIO_DIR,
    exist_ok=True
)


# ============================================================
# LOAD WHISPER MODEL ONCE
# ============================================================

print("Loading Whisper model...")

whisper_model = whisper.load_model(
    "base"
)

print("Whisper model loaded successfully!")


# ============================================================
# CONVERT WEBM → WAV
# ============================================================

def convert_to_wav(
    input_path,
    output_path
):
    """
    Convert browser-recorded WebM audio
    into a 16 kHz mono WAV file.
    """

    command = [

        "ffmpeg",

        "-y",

        "-i",
        input_path,

        "-ar",
        "16000",

        "-ac",
        "1",

        output_path

    ]


    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )


    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg conversion failed:\n"
            + result.stderr
        )


    if not os.path.exists(
        output_path
    ):

        raise FileNotFoundError(
            "Converted WAV file was not created."
        )


    return output_path


# ============================================================
# TRANSCRIBE AUDIO
# ============================================================

def transcribe_audio(
    audio_path
):
    """
    Transcribe a WAV audio file using
    Whisper with English output.
    """

    result = whisper_model.transcribe(
        audio_path,
        language="en"
    )


    transcript = result.get(
        "text",
        ""
    ).strip()


    return transcript


# ============================================================
# COMPLETE AUDIO PIPELINE
# ============================================================

def process_audio(
    input_audio_path
):
    """
    Convert uploaded browser audio
    and transcribe it using Whisper.
    """

    filename = os.path.basename(
        input_audio_path
    )

    filename_without_extension = os.path.splitext(
        filename
    )[0]


    wav_filename = (
        filename_without_extension
        + ".wav"
    )


    wav_path = os.path.join(
        TEMP_AUDIO_DIR,
        wav_filename
    )


    print(
        "\nConverting audio..."
    )


    convert_to_wav(
        input_audio_path,
        wav_path
    )


    print(
        "Audio conversion completed."
    )


    print(
        "Transcribing with Whisper..."
    )


    transcript = transcribe_audio(
        wav_path
    )


    print(
        "Whisper transcription completed."
    )


    return {
        "wav_path": wav_path,
        "transcript": transcript
    }