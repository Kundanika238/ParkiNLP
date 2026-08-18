import whisper

print("Loading Whisper model...")

model = whisper.load_model("base")

print("Model loaded!")
print("Transcribing audio...")

result = model.transcribe(
    "audio/PD/ID02_pd_2_0_0.wav",
    language="en"
)

print("\nTRANSCRIPT:")
print(result["text"])