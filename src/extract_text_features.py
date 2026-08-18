import os
import re
import pandas as pd
import spacy
from collections import Counter

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRANSCRIPTS_DIR = os.path.join(BASE_DIR, "transcripts")
METADATA_FILE = os.path.join(BASE_DIR, "dataset", "dataset_metadata.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "nlp_features.csv")


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD METADATA
# ============================================================

metadata = None


# ============================================================
# FUNCTION: CLEAN TEXT
# ============================================================

def clean_text(text):
    """
    Basic text cleaning.

    Converts text to lowercase and keeps alphabetic words.
    """

    text = text.lower()

    words = re.findall(r"\b[a-zA-Z]+\b", text)

    return words


# ============================================================
# FUNCTION: EXTRACT FEATURES
# ============================================================

def extract_features(text):

    words = clean_text(text)

    # --------------------------------------------------------
    # POS analysis
    # --------------------------------------------------------

    doc = nlp(text)

    pos_counts = Counter(
        token.pos_
        for token in doc
        if token.is_alpha
    )

    pos_word_count = sum(pos_counts.values())

    if pos_word_count > 0:

        noun_ratio = (
            pos_counts["NOUN"] / pos_word_count
        )

        verb_ratio = (
            pos_counts["VERB"] / pos_word_count
        )

        adjective_ratio = (
            pos_counts["ADJ"] / pos_word_count
        )

        adverb_ratio = (
            pos_counts["ADV"] / pos_word_count
        )

        pronoun_ratio = (
            pos_counts["PRON"] / pos_word_count
        )

        content_word_ratio = (
            pos_counts["NOUN"]
            + pos_counts["VERB"]
            + pos_counts["ADJ"]
            + pos_counts["ADV"]
        ) / pos_word_count

    else:

        noun_ratio = 0
        verb_ratio = 0
        adjective_ratio = 0
        adverb_ratio = 0
        pronoun_ratio = 0
        content_word_ratio = 0

    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    word_count = len(words)

    unique_words = set(words)

    unique_word_count = len(unique_words)

    if word_count > 0:
        type_token_ratio = unique_word_count / word_count
    else:
        type_token_ratio = 0

    # --------------------------------------------------------
    # Word length
    # --------------------------------------------------------

    if word_count > 0:
        average_word_length = sum(len(word) for word in words) / word_count
    else:
        average_word_length = 0

    # --------------------------------------------------------
    # Immediate word repetition
    # --------------------------------------------------------

    immediate_repetition_count = 0

    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            immediate_repetition_count += 1

    if word_count > 0:
            immediate_repetition_rate = (
                immediate_repetition_count / word_count
            )
    else:
        immediate_repetition_rate = 0

    # --------------------------------------------------------
    # Immediate repeated two-word phrases
    # --------------------------------------------------------

    immediate_repeated_bigram_count = 0

    for i in range(3, len(words)):
        previous_bigram = (words[i - 3], words[i - 2])
        current_bigram = (words[i - 1], words[i])

        if previous_bigram == current_bigram:
            immediate_repeated_bigram_count += 1

    if word_count > 0:
        immediate_repeated_bigram_rate = (
            immediate_repeated_bigram_count / word_count
        )
    else:
        immediate_repeated_bigram_rate = 0

    # --------------------------------------------------------
    # Sentence statistics
    # --------------------------------------------------------

    sentences = re.split(r"[.!?]+", text)

    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    sentence_count = len(sentences)

    sentence_lengths = []

    for sentence in sentences:

        sentence_words = clean_text(sentence)

        if len(sentence_words) > 0:
            sentence_lengths.append(len(sentence_words))

    if sentence_lengths:

        average_sentence_length = (
            sum(sentence_lengths) / len(sentence_lengths)
        )

        if len(sentence_lengths) > 1:

            mean_length = average_sentence_length

            variance = sum(
                (x - mean_length) ** 2
                for x in sentence_lengths
            ) / len(sentence_lengths)

            sentence_length_std = variance ** 0.5

        else:
            sentence_length_std = 0

    else:

        average_sentence_length = 0
        sentence_length_std = 0

    # --------------------------------------------------------
    # Short sentence ratio
    # --------------------------------------------------------

    if sentence_count > 0:

        short_sentences = sum(
            1
            for length in sentence_lengths
            if length <= 5
        )

        short_sentence_ratio = (
            short_sentences / sentence_count
        )

    else:

        short_sentence_ratio = 0

    # --------------------------------------------------------
    # Common hesitation / conversational words
    # --------------------------------------------------------

    word_counts = Counter(words)

    hesitation_words = {
        "um",
        "uh",
        "er",
        "erm",
        "hmm"
    }

    filler_words = {
        "okay",
        "yeah",
        "well"
    }

    hesitation_count = sum(
        word_counts[word]
        for word in hesitation_words
    )

    filler_count = sum(
        word_counts[word]
        for word in filler_words
    )

    if word_count > 0:

        hesitation_rate = hesitation_count / word_count

        filler_rate = filler_count / word_count

    else:

        hesitation_rate = 0
        filler_rate = 0

    # --------------------------------------------------------
    # Return all features
    # --------------------------------------------------------

    return {

        "noun_ratio": noun_ratio,

        "verb_ratio": verb_ratio,

        "adjective_ratio": adjective_ratio,

        "adverb_ratio": adverb_ratio,

        "pronoun_ratio": pronoun_ratio,

        "content_word_ratio": content_word_ratio,

        "word_count": word_count,

        "unique_word_count": unique_word_count,

        "type_token_ratio": type_token_ratio,

        "average_word_length": average_word_length,

        "immediate_repetition_count": immediate_repetition_count,

        "immediate_repetition_rate": immediate_repetition_rate,

        "immediate_repeated_bigram_count": immediate_repeated_bigram_count,

        "immediate_repeated_bigram_rate": immediate_repeated_bigram_rate,

        "sentence_count": sentence_count,

        "average_sentence_length": average_sentence_length,

        "sentence_length_std": sentence_length_std,

        "short_sentence_ratio": short_sentence_ratio,

        "hesitation_count": hesitation_count,

        "hesitation_rate": hesitation_rate,

        "filler_count": filler_count,

        "filler_rate": filler_rate
    }


# ============================================================
# PROCESS ALL TRANSCRIPTS
# ============================================================

def process_all_transcripts():

    global metadata

    # --------------------------------------------------------
    # LOAD METADATA
    # --------------------------------------------------------

    metadata = pd.read_csv(
        METADATA_FILE
    )

    print("Metadata loaded successfully!")

    print(
        f"Total participants: {len(metadata)}"
    )


    # --------------------------------------------------------
    # INITIALIZE RESULTS
    # --------------------------------------------------------

    results = []

    print(
        "\nStarting NLP feature extraction...\n"
    )


    # --------------------------------------------------------
    # PROCESS PARTICIPANTS
    # --------------------------------------------------------

    for index, row in metadata.iterrows():

        participant_id = str(
            row["participant_id"]
        )

        label = row["label"]


        transcript_file = os.path.join(
            TRANSCRIPTS_DIR,
            participant_id + ".txt"
        )


        print(
            f"[{index + 1}/{len(metadata)}] "
            f"Processing {participant_id}..."
        )


        # ----------------------------------------------------
        # CHECK TRANSCRIPT
        # ----------------------------------------------------

        if not os.path.exists(
            transcript_file
        ):

            print(
                f"WARNING: Transcript not found: "
                f"{transcript_file}"
            )

            continue


        # ----------------------------------------------------
        # READ TRANSCRIPT
        # ----------------------------------------------------

        with open(
            transcript_file,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()


        # ----------------------------------------------------
        # EXTRACT NLP FEATURES
        # ----------------------------------------------------

        features = extract_features(
            text
        )


        # ----------------------------------------------------
        # ADD PARTICIPANT INFORMATION
        # ----------------------------------------------------

        features[
            "participant_id"
        ] = participant_id

        features[
            "label"
        ] = label


        results.append(
            features
        )


    # ========================================================
    # CREATE DATAFRAME
    # ========================================================

    features_df = pd.DataFrame(
        results
    )


    # ========================================================
    # REORDER COLUMNS
    # ========================================================

    first_columns = [
        "participant_id",
        "label"
    ]


    remaining_columns = [

        column

        for column in features_df.columns

        if column not in first_columns

    ]


    features_df = features_df[
        first_columns
        + remaining_columns
    ]


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "NLP FEATURE EXTRACTION COMPLETED!"
    )

    print(
        "========================================"
    )


    print(
        f"\nParticipants processed: "
        f"{len(features_df)}"
    )


    print(
        f"Features generated: "
        f"{len(features_df.columns) - 2}"
    )


    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )


    print(
        "\nFirst five rows:\n"
    )


    print(
        features_df.head().to_string(
            index=False
        )
    )


    return features_df


# ============================================================
# RUN ONLY WHEN FILE IS EXECUTED DIRECTLY
# ============================================================

if __name__ == "__main__":

    process_all_transcripts()