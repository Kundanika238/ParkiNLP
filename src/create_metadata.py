from pathlib import Path
import csv

# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
AUDIO_DIR = PROJECT_DIR / "audio"
OUTPUT_FILE = PROJECT_DIR / "dataset" / "dataset_metadata.csv"

# Store dataset information
records = []

# Look inside HC and PD folders
for label in ["HC", "PD"]:
    label_folder = AUDIO_DIR / label

    for audio_file in sorted(label_folder.glob("*.wav")):
        participant_id = audio_file.stem.split("_")[0]

        records.append({
            "file_name": audio_file.name,
            "participant_id": participant_id,
            "label": label,
            "task": "SpontaneousDialogue",
            "file_path": str(audio_file.relative_to(PROJECT_DIR))
        })

# Write CSV file
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as file:
    fieldnames = [
        "file_name",
        "participant_id",
        "label",
        "task",
        "file_path"
    ]

    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)

print("Metadata file created successfully!")
print(f"Total recordings: {len(records)}")
print(f"Healthy controls: {sum(1 for r in records if r['label'] == 'HC')}")
print(f"Parkinson's recordings: {sum(1 for r in records if r['label'] == 'PD')}")
print(f"Saved to: {OUTPUT_FILE}")