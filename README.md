 # ParkiNLP

ParkiNLP is a web-based speech analysis application that converts spoken audio into text using Whisper, extracts linguistic features using NLP, evaluates speech-sample reliability, and produces an interpretable machine-learning classification result.

## Key Features

- Browser-based microphone recording
- 20–60 second recording workflow
- Automatic audio processing and conversion
- Whisper-based speech-to-text transcription
- NLP-based linguistic feature extraction
- Speech-sample reliability assessment
- Reliability-aware classification handling
- HC and PD classification probabilities
- Interactive results dashboard
- Transcript and linguistic-feature visualization
- Pipeline status visualization
- Analyze Another Recording workflow
- Browser microphone and analysis error handling
- Responsive web interface

## What Makes ParkiNLP Different

ParkiNLP is designed as a complete speech-to-analysis workflow rather than a standalone machine-learning model.

The system first converts speech into text, extracts a compact set of interpretable linguistic features, evaluates whether the resulting linguistic profile is suitable for model interpretation, and then presents the classification output through an interactive web interface.

This reliability-aware layer allows the application to distinguish between:

- Reliable speech samples
- Cautionary speech samples
- Low-reliability speech samples

Instead of blindly displaying a classification result for every input, the system can provide a caution state or withhold model interpretation when the speech sample is not sufficiently suitable.

## How It Works

```text
Browser Microphone
        ↓
Audio Recording
        ↓
WebM → WAV Conversion
        ↓
Whisper Transcription
        ↓
NLP Feature Extraction
        ↓
Speech-Sample Reliability Assessment
        ↓
Machine-Learning Classification
        ↓
Interactive Results Dashboard