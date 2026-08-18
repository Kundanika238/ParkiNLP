import spacy

print("Loading English spaCy model...")

nlp = spacy.load("en_core_web_sm")

print("Model loaded successfully!")

# Read one transcript
with open(
    "transcripts/ID02.txt",
    "r",
    encoding="utf-8"
) as file:

    text = file.read()

doc = nlp(text)

print("\nFirst 30 words with POS tags:\n")

count = 0

for token in doc:

    if token.is_alpha:

        print(
            f"{token.text:15} "
            f"{token.pos_:10} "
            f"{token.tag_}"
        )

        count += 1

        if count >= 30:
            break

print("\nspaCy test completed successfully!")