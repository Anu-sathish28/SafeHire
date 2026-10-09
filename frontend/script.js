/* ============================================================
   SAFEHIRE FRONTEND
   ============================================================ */


/* ============================================================
   SPLASH SCREEN
   ============================================================ */

setTimeout(function () {

    const splashScreen =
        document.getElementById("splash-screen");

    const mainApp =
        document.getElementById("main-app");


    if (splashScreen) {
        splashScreen.style.display = "none";
    }


    if (mainApp) {
        mainApp.style.display = "block";
    }

}, 3000);


/* ============================================================
   GET ELEMENTS
   ============================================================ */

const jobInput =
    document.getElementById("job-input");

const sendButton =
    document.getElementById("send-btn");

const chatMessages =
    document.getElementById("chat-messages");


const uploadButton =
    document.getElementById("upload-btn");

const uploadMenu =
    document.getElementById("upload-menu");


const cameraOption =
    document.getElementById("camera-option");

const photoOption =
    document.getElementById("photo-option");

const fileOption =
    document.getElementById("file-option");

const urlOption =
    document.getElementById("url-option");

const cameraModal =
    document.getElementById("camera-modal");

const cameraVideo =
    document.getElementById("camera-video");

const cameraCanvas =
    document.getElementById("camera-canvas");

const captureButton =
    document.getElementById("capture-btn");

const closeCameraButton =
    document.getElementById("close-camera-btn");

let cameraStream = null;

const photoInput =
    document.getElementById("photo-input");

const fileInput =
    document.getElementById("file-input");


/* ============================================================
   URL MODAL
   ============================================================ */

const urlModal =
    document.getElementById("url-modal");

const urlInput =
    document.getElementById("url-input");

const addUrlButton =
    document.getElementById("add-url-btn");

const closeUrlButton =
    document.getElementById("close-url-btn");


/* ============================================================
   BACKEND API
   ============================================================ */

const API_URL =
    "https://safehire.onrender.com/predict";


/* ============================================================
   ADD USER MESSAGE
   ============================================================ */

function addMessage(text, className) {

    const messageElement =
        document.createElement("div");


    messageElement.classList.add(
        className
    );


    messageElement.textContent =
        text;


    chatMessages.appendChild(
        messageElement
    );


    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


/* ============================================================
   CREATE SAFEHIRE RESULT
   ============================================================ */

function createSafeHireResult(result) {

    const resultContainer =
        document.createElement("div");


    resultContainer.classList.add(
        "bot-message",
        "safehire-result"
    );


    /* ========================================================
       TITLE
       ======================================================== */

    const title =
        document.createElement("div");


    title.textContent =
        "SAFEHIRE RESULT";


    title.style.fontWeight =
        "700";

    title.style.marginBottom =
        "10px";


    resultContainer.appendChild(
        title
    );


    /* ========================================================
       FINAL SAFEHIRE RESULT
       ======================================================== */

    const prediction =
        document.createElement("div");


    const finalResult =
        result.safehire_result ||
        "Unknown";


    prediction.textContent =
        finalResult;


    prediction.style.fontWeight =
        "700";

    prediction.style.fontSize =
        "18px";

    prediction.style.marginBottom =
        "5px";


    resultContainer.appendChild(
        prediction
    );


    /* ========================================================
       SAFEHIRE CONFIDENCE
       ======================================================== */

    const confidence =
        document.createElement("div");


    const safehireConfidence =
        result.confidence !== undefined
            ? result.confidence
            : "N/A";


    confidence.textContent =
        `SafeHire Confidence: ${safehireConfidence}%`;


    confidence.style.marginBottom =
        "15px";


    resultContainer.appendChild(
        confidence
    );


    /* ========================================================
       POSITIVE INDICATORS
       ======================================================== */

    addSection(
        resultContainer,
        "Positive Indicators",
        result.positive_indicators,
        "✓ "
    );


    /* ========================================================
       RED FLAGS
       ======================================================== */

    addSection(
        resultContainer,
        "Red Flags",
        result.red_flags,
        "✗ "
    );


    /* ========================================================
       CAUTION INDICATORS
       ======================================================== */

    addSection(
        resultContainer,
        "Caution Indicators",
        result.caution_indicators,
        "⚠ "
    );


    /* ========================================================
       REASONS
       ======================================================== */

    /*
     * app.py returns reasons based on the detected
     * red flags and caution indicators.
     *
     * We remove exact duplicates so the same sentence
     * is not displayed twice.
     */

    const displayedIndicators = [

        ...(result.red_flags || []),

        ...(result.caution_indicators || [])

    ];


    const uniqueReasons = [];


    (result.reasons || []).forEach(
        function (reason) {

            if (
                !displayedIndicators.includes(
                    reason
                )
                &&
                !uniqueReasons.includes(
                    reason
                )
            ) {

                uniqueReasons.push(
                    reason
                );

            }

        }
    );


    if (uniqueReasons.length > 0) {

        addSection(
            resultContainer,
            "Reasons",
            uniqueReasons,
            "• "
        );

    }


    /* ========================================================
       RECOMMENDATION
       ======================================================== */

    const recommendationTitle =
        document.createElement("div");


    recommendationTitle.textContent =
        "Recommendation";


    recommendationTitle.style.fontWeight =
        "700";


    recommendationTitle.style.marginTop =
        "15px";


    recommendationTitle.style.marginBottom =
        "5px";


    resultContainer.appendChild(
        recommendationTitle
    );


    const recommendation =
        document.createElement("div");


    recommendation.textContent =
        result.recommendation ||
        "No recommendation available.";


    resultContainer.appendChild(
        recommendation
    );


    return resultContainer;
}


/* ============================================================
   ADD RESULT SECTION
   ============================================================ */

function addSection(
    container,
    heading,
    items,
    symbol
) {

    const headingElement =
        document.createElement("div");


    headingElement.textContent =
        heading;


    headingElement.style.fontWeight =
        "700";


    headingElement.style.marginTop =
        "10px";


    headingElement.style.marginBottom =
        "5px";


    container.appendChild(
        headingElement
    );


    if (
        Array.isArray(items) &&
        items.length > 0
    ) {

        items.forEach(
            function (item) {

                const line =
                    document.createElement("div");


                line.textContent =
                    symbol + item;


                line.style.marginBottom =
                    "5px";


                container.appendChild(
                    line
                );

            }
        );

    } else {

        const none =
            document.createElement("div");


        none.textContent =
            "- None identified";


        none.style.marginBottom =
            "5px";


        container.appendChild(
            none
        );

    }
}


/* ============================================================
   ANALYZE JOB / INTERNSHIP
   ============================================================ */

async function analyzeJob(payload) {

    const loadingMessage =
        document.createElement("div");


    loadingMessage.classList.add(
        "bot-message"
    );


    loadingMessage.textContent =
        "SafeHire is analyzing the job or internship details...";


    chatMessages.appendChild(
        loadingMessage
    );


    chatMessages.scrollTop =
        chatMessages.scrollHeight;


    try {

        const response =
            await fetch(
                API_URL,
                {
                    method: "POST",
                    body: payload
                }
            );


        let result;


        try {

            result =
                await response.json();

        } catch (jsonError) {

            throw new Error(
                "The backend returned an invalid response."
            );

        }


        if (!response.ok) {

            throw new Error(
                result.error ||
                "Unable to analyze this submission."
            );

        }


        loadingMessage.remove();


        const resultMessage =
            createSafeHireResult(
                result
            );


        chatMessages.appendChild(
            resultMessage
        );


    } catch (error) {

        loadingMessage.textContent =
            "Analysis failed: " +
            error.message;

    }


    chatMessages.scrollTop =
        chatMessages.scrollHeight;
}


/* ============================================================
   SEND TEXT TO BACKEND
   ============================================================ */

function sendTextMessage() {

    const userMessage =
        jobInput.value.trim();


    if (userMessage === "") {

        return;

    }


    /* --------------------------------------------------------
       DISPLAY USER MESSAGE
       -------------------------------------------------------- */

    addMessage(
        userMessage,
        "user-message"
    );


    /* --------------------------------------------------------
       CLEAR INPUT
       -------------------------------------------------------- */

    jobInput.value = "";


    /* --------------------------------------------------------
       CREATE FORM DATA
       -------------------------------------------------------- */

    const formData =
        new FormData();


    formData.append(
        "job_text",
        userMessage
    );


    /* --------------------------------------------------------
       SEND TO FLASK
       -------------------------------------------------------- */

    analyzeJob(
        formData
    );
}


/* ============================================================
   ARROW SEND BUTTON
   ============================================================ */

sendButton.addEventListener(
    "click",
    function () {

        sendTextMessage();

    }
);


/* ============================================================
   ENTER KEY
   ============================================================ */

/*
 * Enter:
 *     Send message
 *
 * Shift + Enter:
 *     Create a new line
 */

jobInput.addEventListener(
    "keydown",
    function (event) {

        if (
            event.key === "Enter" &&
            !event.shiftKey
        ) {

            event.preventDefault();

            sendTextMessage();

        }

    }
);


/* ============================================================
   OPEN UPLOAD MENU
   ============================================================ */

uploadButton.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();


        uploadMenu.classList.toggle(
            "show"
        );

    }
);


/* ============================================================
   CLOSE UPLOAD MENU
   ============================================================ */

document.addEventListener(
    "click",
    function (event) {

        if (
            !uploadMenu.contains(
                event.target
            )
            &&
            !uploadButton.contains(
                event.target
            )
        ) {

            uploadMenu.classList.remove(
                "show"
            );

        }

    }
);


/* ============================================================
   CAMERA
   ============================================================ */

cameraOption.addEventListener(
    "click",
    async function () {

        uploadMenu.classList.remove(
            "show"
        );

        try {

            cameraStream =
                await navigator.mediaDevices.getUserMedia({
                    video: {
                        facingMode: {
                            ideal: "environment"
                        }
                    },
                    audio: false
                });

            cameraVideo.srcObject =
                cameraStream;

            cameraModal.classList.add(
                "show"
            );

        } catch (error) {

            console.error(
                "Camera error:",
                error
            );

            alert(
                "Unable to access the camera. Please allow camera permission and try again."
            );

        }

    }
);


/* ============================================================
   CAPTURE CAMERA PHOTO
   ============================================================ */

captureButton.addEventListener(
    "click",
    function () {

        if (!cameraStream) {

            return;

        }


        const videoWidth =
            cameraVideo.videoWidth;

        const videoHeight =
            cameraVideo.videoHeight;


        if (
            videoWidth === 0 ||
            videoHeight === 0
        ) {

            alert(
                "Camera is not ready yet. Please wait a moment and try again."
            );

            return;

        }


        cameraCanvas.width =
            videoWidth;

        cameraCanvas.height =
            videoHeight;


        const context =
            cameraCanvas.getContext(
                "2d"
            );


        context.drawImage(
            cameraVideo,
            0,
            0,
            videoWidth,
            videoHeight
        );


        cameraCanvas.toBlob(
            function (blob) {

                if (!blob) {

                    alert(
                        "Unable to capture the image."
                    );

                    return;

                }


                const capturedFile =
                    new File(
                        [blob],
                        "camera_capture.jpg",
                        {
                            type: "image/jpeg"
                        }
                    );


                addMessage(
                    "Captured image: camera_capture.jpg",
                    "user-message"
                );


                const formData =
                    new FormData();


                formData.append(
                    "file",
                    capturedFile
                );


                closeCamera();


                analyzeJob(
                    formData
                );

            },
            "image/jpeg",
            0.92
        );

    }
);


/* ============================================================
   CLOSE CAMERA
   ============================================================ */

closeCameraButton.addEventListener(
    "click",
    function () {

        closeCamera();

    }
);


/* ============================================================
   STOP CAMERA
   ============================================================ */

function closeCamera() {

    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                function (track) {

                    track.stop();

                }
            );

        cameraStream = null;

    }


    if (cameraVideo) {

        cameraVideo.srcObject =
            null;

    }


    if (cameraModal) {

        cameraModal.classList.remove(
            "show"
        );

    }

}

/* ============================================================
   PHOTO
   ============================================================ */

photoOption.addEventListener(
    "click",
    function () {

        uploadMenu.classList.remove(
            "show"
        );


        photoInput.click();

    }
);


photoInput.addEventListener(
    "change",
    function () {

        const selectedFile =
            photoInput.files[0];


        if (!selectedFile) {

            return;

        }


        addMessage(
            "Uploaded photo: " +
            selectedFile.name,
            "user-message"
        );


        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        analyzeJob(
            formData
        );


        /* Reset input */

        photoInput.value = "";

    }
);


/* ============================================================
   FILE
   ============================================================ */

fileOption.addEventListener(
    "click",
    function () {

        uploadMenu.classList.remove(
            "show"
        );


        fileInput.click();

    }
);


fileInput.addEventListener(
    "change",
    function () {

        const selectedFile =
            fileInput.files[0];


        if (!selectedFile) {

            return;

        }


        addMessage(
            "Uploaded file: " +
            selectedFile.name,
            "user-message"
        );


        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        analyzeJob(
            formData
        );


        /* Reset input */

        fileInput.value = "";

    }
);


/* ============================================================
   URL OPTION
   ============================================================ */

urlOption.addEventListener(
    "click",
    function (event) {

        event.stopPropagation();


        uploadMenu.classList.remove(
            "show"
        );


        urlModal.classList.add(
            "show"
        );


        urlInput.focus();

    }
);


/* ============================================================
   ADD URL
   ============================================================ */

addUrlButton.addEventListener(
    "click",
    function () {

        const url =
            urlInput.value.trim();


        if (url === "") {

            return;

        }


        addMessage(
            "Added URL: " + url,
            "user-message"
        );


        const formData =
            new FormData();


        formData.append(
            "url",
            url
        );


        analyzeJob(
            formData
        );


        urlInput.value = "";


        urlModal.classList.remove(
            "show"
        );

    }
);


/* ============================================================
   CLOSE URL MODAL
   ============================================================ */

closeUrlButton.addEventListener(
    "click",
    function () {

        urlInput.value = "";


        urlModal.classList.remove(
            "show"
        );

    }
);

/* ============================================================
   MICROPHONE / VOICE INPUT
   ============================================================ */

const microphoneButton =
    document.getElementById("voice-btn");


if (microphoneButton) {

    const SpeechRecognition =
        window.SpeechRecognition ||
        window.webkitSpeechRecognition;


    if (!SpeechRecognition) {

        microphoneButton.addEventListener(
            "click",
            function () {

                alert(
                    "Voice input is not supported in this browser."
                );

            }
        );

    } else {

        const recognition =
            new SpeechRecognition();


        recognition.lang =
            "en-IN";


        recognition.continuous =
            false;


        recognition.interimResults =
            false;


        recognition.onstart =
            function () {

                microphoneButton.classList.add(
                    "recording"
                );

            };


        recognition.onresult =
            function (event) {

                const transcript =
                    event.results[0][0].transcript;


                jobInput.value =
                    jobInput.value
                    ? jobInput.value + " " + transcript
                    : transcript;

            };


        recognition.onerror =
            function (event) {

                console.log(
                    "Microphone error:",
                    event.error
                );

            };


        recognition.onend =
            function () {

                microphoneButton.classList.remove(
                    "recording"
                );

            };


        microphoneButton.addEventListener(
            "click",
            function () {

                recognition.start();

            }
        );

    }

}


if ("serviceWorker" in navigator) {
    window.addEventListener("load", () => {
        navigator.serviceWorker
            .register("./service-worker.js")
            .then(() => {
                console.log("SafeHire service worker registered.");
            })
            .catch((error) => {
                console.error("Service worker registration failed:", error);
            });
    });
}
