from pathlib import Path
import csv
import whisper

# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent
METADATA_FILE = PROJECT_DIR / "dataset" / "dataset_metadata.csv"
TRANSCRIPT_DIR = PROJECT_DIR / "transcripts"

# Create transcript folder if it doesn't exist
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading Whisper model...")
model = whisper.load_model("base")

print("Whisper model loaded successfully!")
print()

# Read metadata
with open(METADATA_FILE, "r", encoding="utf-8") as file:
    reader = csv.DictReader(file)
    records = list(reader)

print(f"Total recordings found: {len(records)}")
print()

# Process each recording
for index, record in enumerate(records, start=1):

    audio_path = PROJECT_DIR / record["file_path"]
    output_file = TRANSCRIPT_DIR / f"{record['participant_id']}.txt"

    print(f"[{index}/{len(records)}] Processing: {record['file_name']}")

    # Skip if transcript already exists
    if output_file.exists():
        print("Transcript already exists. Skipping...")
        print()
        continue

    result = model.transcribe(
        str(audio_path),
        language="en"
    )

    transcript = result["text"].strip()

    # Save transcript
    with open(output_file, "w", encoding="utf-8") as text_file:
        text_file.write(transcript)

    print("Transcript saved.")
    print()

print("====================================")
print("ALL TRANSCRIPTIONS COMPLETED!")
print("====================================")
print(f"Transcripts saved in: {TRANSCRIPT_DIR}")