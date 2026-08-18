/* ============================================================
   PARKINLP — MAIN JAVASCRIPT
   Professional Browser Microphone Recording
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    const recordButton =
        document.getElementById("recordButton");

    const startRecordingButton =
        document.getElementById("startRecordingButton");

    const stopRecordingButton =
        document.getElementById("stopRecordingButton");

    const analyzeRecordingButton =
        document.getElementById("analyzeRecordingButton");

    const recorderTitle =
        document.getElementById("recorderTitle");

    const recorderDescription =
        document.getElementById("recorderDescription");

    const recordingTimer =
        document.getElementById("recordingTimer");

    const recordingStatus =
        document.getElementById("recordingStatus");

    const recordingProgressBar =
        document.getElementById("recordingProgressBar");

    const audioPreview =
        document.getElementById("audioPreview");


    /* ========================================================
       CHECK ANALYZE PAGE
       ======================================================== */

    if (!recordButton) {
        return;
    }


    /* ========================================================
       RECORDING SETTINGS
       ======================================================== */

    const MIN_RECORDING_SECONDS = 20;

    const MAX_RECORDING_SECONDS = 60;


    /* ========================================================
       RECORDING VARIABLES
       ======================================================== */

    let mediaRecorder = null;

    let audioChunks = [];

    let recordingStream = null;

    let timerInterval = null;

    let elapsedSeconds = 0;

    let recordedAudioBlob = null;


    /* ========================================================
       FORMAT TIMER
       ======================================================== */

    function formatTime(seconds) {

        const minutes =
            Math.floor(seconds / 60)
                .toString()
                .padStart(2, "0");


        const remainingSeconds =
            (seconds % 60)
                .toString()
                .padStart(2, "0");


        return `${minutes}:${remainingSeconds}`;
    }


    /* ========================================================
       UPDATE TIMER
       ======================================================== */

    function updateTimer() {

        recordingTimer.textContent =
            formatTime(elapsedSeconds);


        const progress =
            (
                elapsedSeconds /
                MAX_RECORDING_SECONDS
            ) * 100;


        recordingProgressBar.style.width =
            `${Math.min(progress, 100)}%`;
    }


    /* ========================================================
       RESET RECORDING UI
       ======================================================== */

    function resetRecordingUI() {

        recordButton.classList.remove(
            "recording"
        );


        startRecordingButton.classList.remove(
            "hidden"
        );


        stopRecordingButton.classList.add(
            "hidden"
        );


        analyzeRecordingButton.classList.add(
            "hidden"
        );


        analyzeRecordingButton.disabled =
            false;


        recorderTitle.textContent =
            "Ready to analyze?";


        recorderDescription.textContent =
            "Speak naturally for at least 20 seconds. " +
            "A continuous 30–60 second sample is recommended " +
            "for more reliable analysis.";


        recordingTimer.textContent =
            "00:00";


        recordingStatus.textContent =
            "Microphone ready";


        recordingProgressBar.style.width =
            "0%";


        if (audioPreview) {

            audioPreview.pause();

            audioPreview.removeAttribute(
                "src"
            );

            audioPreview.load();

            audioPreview.style.display =
                "none";
        }


        elapsedSeconds = 0;

        recordedAudioBlob = null;
    }


    /* ========================================================
       START RECORDING
       ======================================================== */

    async function startRecording() {

        try {

            /* ------------------------------------------------
               Request microphone permission
               ------------------------------------------------ */

            recordingStream =
                await navigator.mediaDevices.getUserMedia({
                    audio: true
                });


            /* ------------------------------------------------
               Check MediaRecorder support
               ------------------------------------------------ */

            if (!window.MediaRecorder) {

                throw new Error(
                    "MediaRecorder is not supported by this browser."
                );
            }


            /* ------------------------------------------------
               Create MediaRecorder
               ------------------------------------------------ */

            let recorderOptions = {};


            if (
                MediaRecorder.isTypeSupported(
                    "audio/webm;codecs=opus"
                )
            ) {

                recorderOptions.mimeType =
                    "audio/webm;codecs=opus";

            }


            mediaRecorder =
                new MediaRecorder(
                    recordingStream,
                    recorderOptions
                );


            console.log(
                "MediaRecorder MIME type:",
                mediaRecorder.mimeType
            );


            /* ------------------------------------------------
               Reset recording state
               ------------------------------------------------ */

            audioChunks = [];

            elapsedSeconds = 0;

            recordedAudioBlob = null;

            updateTimer();


            /* ------------------------------------------------
               Store audio chunks
               ------------------------------------------------ */

            mediaRecorder.ondataavailable =
                (event) => {

                    console.log(
                        "Audio chunk:",
                        event.data.size,
                        "bytes"
                    );


                    if (
                        event.data &&
                        event.data.size > 0
                    ) {

                        audioChunks.push(
                            event.data
                        );
                    }
                };


            /* ------------------------------------------------
               Recording started
               ------------------------------------------------ */

            mediaRecorder.onstart =
                () => {

                    console.log(
                        "Recording started."
                    );


                    recordButton.classList.add(
                        "recording"
                    );


                    startRecordingButton.classList.add(
                        "hidden"
                    );


                    stopRecordingButton.classList.remove(
                        "hidden"
                    );


                    analyzeRecordingButton.classList.add(
                        "hidden"
                    );


                    recorderTitle.textContent =
                        "Recording your speech...";


                    recorderDescription.textContent =
                        "Speak naturally and continuously. " +
                        "Try to reach at least 20 seconds; " +
                        "30–60 seconds is recommended.";


                    recordingStatus.textContent =
                        "Recording • speak naturally";


                    timerInterval =
                        setInterval(
                            () => {

                                elapsedSeconds++;

                                updateTimer();


                                if (
                                    elapsedSeconds >=
                                    MAX_RECORDING_SECONDS
                                ) {

                                    console.log(
                                        "Maximum recording duration reached."
                                    );

                                    recorderDescription.textContent =
                                        "The 60-second recording limit was reached. " +
                                        "Your recording is ready for review.";


                                    stopRecording();
                                }

                            },
                            1000
                        );
                };


            /* ------------------------------------------------
               Recording stopped
               ------------------------------------------------ */

            mediaRecorder.onstop =
                () => {

                    console.log(
                        "Recording stopped."
                    );


                    clearInterval(
                        timerInterval
                    );


                    if (recordingStream) {

                        recordingStream
                            .getTracks()
                            .forEach(
                                track =>
                                    track.stop()
                            );

                    }


                    /* ----------------------------------------
                       Create final audio blob
                       ---------------------------------------- */

                    recordedAudioBlob =
                        new Blob(
                            audioChunks,
                            {
                                type:
                                    mediaRecorder.mimeType ||
                                    "audio/webm"
                            }
                        );


                    console.log(
                        "Final recording size:",
                        recordedAudioBlob.size,
                        "bytes"
                    );


                    console.log(
                        "Recorded duration:",
                        elapsedSeconds,
                        "seconds"
                    );


                    /* ----------------------------------------
                       Create preview
                       ---------------------------------------- */

                    const audioURL =
                        URL.createObjectURL(
                            recordedAudioBlob
                        );


                    if (audioPreview) {

                        audioPreview.src =
                            audioURL;

                        audioPreview.style.display =
                            "block";
                    }


                    /* ----------------------------------------
                       Check minimum duration
                       ---------------------------------------- */

                    if (
                        elapsedSeconds <
                        MIN_RECORDING_SECONDS
                    ) {

                        recorderTitle.textContent =
                            "Recording is too short";


                        recorderDescription.textContent =
                            `Your recording was only ` +
                            `${elapsedSeconds} seconds. ` +
                            `Please record for at least ` +
                            `${MIN_RECORDING_SECONDS} seconds.`;


                        recordingStatus.textContent =
                            "Recording too short • minimum 20 seconds";


                        stopRecordingButton.classList.add(
                            "hidden"
                        );


                        analyzeRecordingButton.classList.add(
                            "hidden"
                        );


                        startRecordingButton.classList.remove(
                            "hidden"
                        );


                        return;
                    }


                    /* ----------------------------------------
                       Valid recording
                       ---------------------------------------- */

                    recorderTitle.textContent =
                        "Recording complete";


                    recorderDescription.textContent =
                        "Your recording is ready. " +
                        "Review the audio before starting the " +
                        "ParkiNLP exploratory analysis.";


                    recordingStatus.textContent =
                        "Recording ready • review before analysis";


                    stopRecordingButton.classList.add(
                        "hidden"
                    );


                    analyzeRecordingButton.classList.remove(
                        "hidden"
                    );


                    analyzeRecordingButton.disabled =
                        false;
                };


            /* ------------------------------------------------
               Recording error
               ------------------------------------------------ */

            mediaRecorder.onerror =
                (event) => {

                    console.error(
                        "MediaRecorder error:",
                        event.error
                    );


                    clearInterval(
                        timerInterval
                    );


                    recorderTitle.textContent =
                        "Recording error";


                    recorderDescription.textContent =
                        "The browser could not complete the recording. " +
                        "Please check microphone access and try again.";


                    recordingStatus.textContent =
                        "Recording error";


                    resetRecordingUI();
                };


            /* ------------------------------------------------
               Start recorder
               ------------------------------------------------ */

            mediaRecorder.start();


        }

        catch (error) {

            console.error(
                "Microphone access error:",
                error
            );


            recorderTitle.textContent =
                "Microphone access unavailable";


            recorderDescription.textContent =
                "Microphone access is required for recording. " +
                "Please allow microphone permission in your browser " +
                "and then try again.";


            recordingStatus.textContent =
                "Microphone access required";


            resetRecordingUI();
        }
    }


    /* ========================================================
       STOP RECORDING
       ======================================================== */

    function stopRecording() {

        if (
            mediaRecorder &&
            mediaRecorder.state === "recording"
        ) {

            console.log(
                "Stopping recording..."
            );


            mediaRecorder.stop();
        }
    }


    /* ========================================================
       START BUTTON
       ======================================================== */

    startRecordingButton.addEventListener(
        "click",
        startRecording
    );


    /* ========================================================
       MICROPHONE BUTTON
       ======================================================== */

    recordButton.addEventListener(
        "click",
        () => {

            if (
                !mediaRecorder ||
                mediaRecorder.state === "inactive"
            ) {

                startRecording();

            }

            else if (
                mediaRecorder.state === "recording"
            ) {

                stopRecording();
            }

        }
    );


    /* ========================================================
       STOP BUTTON
       ======================================================== */

    stopRecordingButton.addEventListener(
        "click",
        stopRecording
    );


    /* ========================================================
       ANALYZE BUTTON
       ======================================================== */

    analyzeRecordingButton.addEventListener(
        "click",
        async () => {

            if (!recordedAudioBlob) {

                recorderTitle.textContent =
                    "No recording available";


                recordingStatus.textContent =
                    "Please record your speech first.";


                return;
            }


            /* ------------------------------------------------
               Final duration safety check
               ------------------------------------------------ */

            if (
                elapsedSeconds <
                MIN_RECORDING_SECONDS
            ) {

                recorderTitle.textContent =
                    "Recording is too short";


                recorderDescription.textContent =
                    "Please record for at least " +
                    `${MIN_RECORDING_SECONDS} seconds.`;


                return;
            }


            /* ------------------------------------------------
               Update interface
               ------------------------------------------------ */

            recorderTitle.textContent =
                "Analyzing your speech...";


            recorderDescription.textContent =
                "Whisper is transcribing your recording, " +
                "while NLP features, reliability, and the " +
                "classification model are being processed.";


            recordingStatus.textContent =
                "Analyzing • Whisper + NLP + ML";


            analyzeRecordingButton.disabled =
                true;


            startRecordingButton.classList.add(
                "hidden"
            );

            analyzeRecordingButton.textContent =
                "Analyzing...";


            /* ------------------------------------------------
               Create form data
               ------------------------------------------------ */

            const formData =
                new FormData();


            formData.append(
                "audio",
                recordedAudioBlob,
                "recording.webm"
            );


            try {

                /* --------------------------------------------
                   Send to COMPLETE ParkiNLP pipeline
                   -------------------------------------------- */

                const response =
                    await fetch(
                        "/analyze-audio",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                const result =
                    await response.json();


                console.log(
                    "ParkiNLP analysis result:",
                    result
                );


                if (
                    !response.ok ||
                    !result.success
                ) {

                    throw new Error(
                        result.message ||
                        "Speech analysis failed."
                    );
                }


                /* --------------------------------------------
                   Low reliability
                   -------------------------------------------- */

                if (
                    result.prediction_available === false
                ) {

                    recorderTitle.textContent =
                        "More speech is needed";

                    recorderDescription.textContent =
                        "The recording was processed successfully, " +
                        "but its linguistic profile was not reliable " +
                        "enough for the exploratory model. " +
                        "Please provide a longer and more natural " +
                        "continuous speech sample.";

                    recordingStatus.textContent =
                        "Low reliability";

                    console.warn(
                        "Prediction blocked:",
                        result.reliability
                    );

                    analyzeRecordingButton.textContent =
                        "Analyze Speech →";

                    analyzeRecordingButton.disabled =
                        false;

                    return;
                }


                /* --------------------------------------------
                   Successful model analysis
                   -------------------------------------------- */

                recorderTitle.textContent =
                    "Analysis complete";


                recorderDescription.textContent =
                    "Your speech was processed successfully " +
                    "through Whisper transcription, NLP feature " +
                    "analysis, reliability assessment, and the " +
                    "classification model.";


                recordingStatus.textContent =
                    "Analysis complete";


                console.log(
                    "Prediction:",
                    result.prediction
                );


                console.log(
                    "HC probability:",
                    result.hc_probability
                );


                console.log(
                    "PD probability:",
                    result.pd_probability
                );


                console.log(
                    "Reliability:",
                    result.reliability
                );


                /*
                 * Temporary display.
                 *
                 * We will replace this with the
                 * professional results dashboard
                 * in the next step.
                 */

                displayAnalysisResults(result);

                analyzeRecordingButton.textContent =
                    "Analyze Speech →";

                analyzeRecordingButton.disabled =
                    false;

            }

            catch (error) {

                console.error(
                    "Analysis error:",
                    error
                );


                analyzeRecordingButton.textContent =
                    "Analyze Speech →";
    
                recorderTitle.textContent =
                    "Analysis failed";


                recorderDescription.textContent =
                    error.message ||
                    "The speech analysis could not be completed. " +
                    "Please try the recording again.";


                recordingStatus.textContent =
                    "Analysis failed";


                analyzeRecordingButton.disabled =
                    false;
            }

        }
    );


    /* ========================================================
       INITIAL STATE
       ======================================================== */

    resetRecordingUI();

    // ============================================================
    // DISPLAY PARKINLP ANALYSIS RESULTS
    // ============================================================

    function displayAnalysisResults(result) {

        // ========================================================
        // RESULTS SECTION
        // ========================================================

        const resultsSection =
            document.getElementById("resultsSection");

        if (!resultsSection) {

            console.error(
                "Results section not found."
            );

            return;
        }


        // ========================================================
        // PREDICTION
        // ========================================================

        const predictionValue =
            document.getElementById(
                "predictionValue"
            );

        const predictionSubtitle =
            document.getElementById(
                "predictionSubtitle"
            );


        if (
            result.prediction_available === true
        ) {

            predictionValue.textContent =
                result.prediction || "—";

             if (
                result.reliability &&
                result.reliability.status === "CAUTION"
            ) {

                predictionSubtitle.textContent =
                    "Model output " +
                    "— interpret with caution";

            } else {

                predictionSubtitle.textContent =
                    "Model output";
            }
        } else {

            predictionValue.textContent =
                "Unavailable";

            predictionSubtitle.textContent =
                "Prediction was not generated";
        }


        // ========================================================
        // HC PROBABILITY
        // ========================================================

        const hcProbability =
            document.getElementById(
                "hcProbability"
            );


        if (
            result.hc_probability !== undefined
        ) {

            hcProbability.textContent =
                (
                    Number(
                        result.hc_probability
                    ) * 100
                ).toFixed(2) + "%";

        } else {

            hcProbability.textContent =
                "—";
        }


        // ========================================================
        // PD PROBABILITY
        // ========================================================

        const pdProbability =
            document.getElementById(
                "pdProbability"
            );

        const pdPercent =
            result.pd_probability !== undefined
                ? Number(
                    result.pd_probability
                ) * 100
                : null;


        if (pdPercent !== null) {

            pdProbability.textContent =
                pdPercent.toFixed(2) + "%";

        } else {

            pdProbability.textContent =
                "—";
        }


        // ========================================================
        // RELIABILITY
        // ========================================================

        const reliability =
            result.reliability || {};


        const reliabilityStatus =
            document.getElementById(
                "reliabilityStatus"
            );


        const reliabilityMessage =
            document.getElementById(
                "reliabilityMessage"
            );


        const reliabilityBadge =
            document.getElementById(
                "reliabilityBadge"
            );


        const reliabilityLevel =
            reliability.status || "UNKNOWN";

        reliabilityStatus.textContent =
            reliabilityLevel;

        reliabilityBadge.textContent =
            reliabilityLevel;
        
        reliabilityBadge.classList.remove(
            "reliable",
            "caution",
            "low-reliability"
        );

        if (reliabilityLevel === "RELIABLE") {

            reliabilityBadge.classList.add(
                "reliable"
            );

        }
        else if (reliabilityLevel === "CAUTION") {

            reliabilityBadge.classList.add(
                "caution"
            );

        }
        else if (
            reliabilityLevel === "LOW_RELIABILITY"
        ) {

            reliabilityBadge.classList.add(
                "low-reliability"
            );

        }

        if (reliabilityLevel === "CAUTION") {

            reliabilityMessage.textContent =
                "This speech sample can be analyzed, " +
                "but some linguistic characteristics " +
                "differ from the typical training " +
                "distribution. Interpret the model output " +
                "with caution. This is a model result, " +
                "not a medical diagnosis.";

        }
        else if (
            reliabilityLevel === "LOW_RELIABILITY"
        ) {

            reliabilityMessage.textContent =
                "This speech sample does not contain " +
                "sufficiently reliable information for " +
                "model interpretation. Please provide " +
                "a longer and more natural continuous " +
                "speech sample.";

        }
        else if (
            reliabilityLevel === "RELIABLE"
        ) {

            reliabilityMessage.textContent =
                "The speech sample contains sufficient " +
                "information and its linguistic feature " +
                "profile is reasonably consistent with " +
                "the training distribution. This is a " +
                "model result, not a medical diagnosis.";

        }
        else {

            reliabilityMessage.textContent =
                reliability.message ||
                "No reliability information available.";

        }


        // ========================================================
        // TRANSCRIPT
        // ========================================================

        const transcriptText =
            document.getElementById(
                "transcriptText"
            );


        transcriptText.textContent =
            result.transcript ||
            "No transcript available.";


        // ========================================================
        // NLP FEATURES
        // ========================================================

        const features =
            result.features || {};


        const fillerCount =
            document.getElementById(
                "featureFillerCount"
            );


        const fillerRate =
            document.getElementById(
                "featureFillerRate"
            );


        const sentenceCount =
            document.getElementById(
                "featureSentenceCount"
            );


        const shortSentenceRatio =
            document.getElementById(
                "featureShortSentenceRatio"
            );


        const typeTokenRatio =
            document.getElementById(
                "featureTypeTokenRatio"
            );


        fillerCount.textContent =
            features.filler_count !== undefined
                ? features.filler_count
                : "—";


        fillerRate.textContent =
            features.filler_rate !== undefined
                ? Number(
                    features.filler_rate
                ).toFixed(4)
                : "—";


        sentenceCount.textContent =
            features.sentence_count !== undefined
                ? features.sentence_count
                : "—";


        shortSentenceRatio.textContent =
            features.short_sentence_ratio !== undefined
                ? Number(
                    features.short_sentence_ratio
                ).toFixed(4)
                : "—";


        typeTokenRatio.textContent =
            features.type_token_ratio !== undefined
                ? Number(
                    features.type_token_ratio
                ).toFixed(4)
                : "—";


        // ========================================================
        // RELIABILITY STATISTICS
        // ========================================================

        const wordCount =
            document.getElementById(
                "wordCount"
            );


        const reliabilitySentenceCount =
            document.getElementById(
                "sentenceCount"
            );


        const unusualFeatureCount =
            document.getElementById(
                "unusualFeatureCount"
            );


        wordCount.textContent =
            reliability.word_count !== undefined
                ? reliability.word_count
                : "—";


        reliabilitySentenceCount.textContent =
            reliability.sentence_count !== undefined
                ? reliability.sentence_count
                : "—";


        const unusualFeatures =
            reliability.unusual_features || [];


        unusualFeatureCount.textContent =
            unusualFeatures.length;


        // ========================================================
        // UNUSUAL FEATURE TAGS
        // ========================================================

        const reliabilityFeatures =
            document.getElementById(
                "reliabilityFeatures"
            );


        reliabilityFeatures.innerHTML = "";


        if (
            unusualFeatures.length > 0
        ) {

            unusualFeatures.forEach(
                function(feature) {

                    const tag =
                        document.createElement(
                            "div"
                        );


                    tag.className =
                        "reliability-feature-tag";


                    tag.textContent =
                        feature;


                    reliabilityFeatures.appendChild(
                        tag
                    );

                }
            );

        } else {

            const tag =
                document.createElement(
                    "div"
                );


            tag.className =
                "reliability-feature-tag";


            tag.textContent =
                "No unusual features detected";


            reliabilityFeatures.appendChild(
                tag
            );
        }


        // ========================================================
        // PIPELINE MODEL STATUS
        // ========================================================

        const pipelineModelStep =
            document.getElementById(
                "pipelineModelStep"
            );
        
        const pipelineModelIcon =
            document.getElementById(
                "pipelineModelIcon"
            );


        if (
            pipelineModelStep
        ) {

            if (
                result.prediction_available === true
            ) {

                pipelineModelStep.classList.add(
                    "complete"
                );

                if (pipelineModelIcon) {

                    pipelineModelIcon.textContent =
                        "✓";

                }

            } else {

                pipelineModelStep.classList.remove(
                    "complete"
                );

                if (pipelineModelIcon) {

                    pipelineModelIcon.textContent =
                        "—";

                }

            }
        }


        // ========================================================
        // SHOW RESULTS
        // ========================================================

        resultsSection.classList.remove(
            "hidden"
        );


        // ========================================================
        // SCROLL TO RESULTS
        // ========================================================

        setTimeout(
            function() {

                resultsSection.scrollIntoView({
                    behavior: "smooth",
                    block: "start"
                });

            },
            100
        );
    }

    // ============================================================
    // ANALYZE ANOTHER RECORDING
    // ============================================================

    const analyzeAnotherButton =
        Array.from(
            document.querySelectorAll("button")
        ).find(
            button =>
                button.textContent.trim() ===
                "Analyze Another Recording"
        );

    if (analyzeAnotherButton) {

        analyzeAnotherButton.addEventListener(
            "click",
            () => {

                // Reset recorder
                resetRecordingUI();

                analyzeRecordingButton.textContent =
                    "Analyze Speech →";

                // Hide previous results
                const resultsSection =
                    document.getElementById(
                        "resultsSection"
                    );

                if (resultsSection) {

                    resultsSection.classList.add(
                        "hidden"
                    );
                }

                // Return to top of recorder
                window.scrollTo({
                    top: 0,
                    behavior: "smooth"
                });

            }
        );
    }   

});