const predictBtn = document.getElementById("predictBtn");
const textInput = document.getElementById("textInput");
const sentimentSpan = document.getElementById("sentiment");
const emojiSpan = document.getElementById("emoji");
const confidenceFill = document.getElementById("confidenceFill");
const resultCard = document.getElementById("resultCard");
const confidenceText = document.getElementById("confidenceText");
const wordCountText = document.getElementById("wordCountText");
const themeToggle = document.getElementById("themeToggle");
const themeIcon = document.getElementById("themeIcon");
const uploadBtn = document.getElementById("uploadBtn");
const voiceBtn = document.getElementById("voiceBtn");
const speakBtn = document.getElementById("speakBtn");
const clearBtn = document.getElementById("clearBtn");
const copyResultBtn = document.getElementById("copyResultBtn");
const downloadHistoryBtn = document.getElementById("downloadHistoryBtn");
const clearHistoryBtn = document.getElementById("clearHistoryBtn");
const statusMessage = document.getElementById("statusMessage");
const inputStats = document.getElementById("inputStats");
const historyList = document.getElementById("historyList");
const totalCount = document.getElementById("totalCount");
const positiveCount = document.getElementById("positiveCount");
const negativeCount = document.getElementById("negativeCount");
const neutralCount = document.getElementById("neutralCount");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
let recognition = null;
let isListening = false;
let latestResult = null;
let history = JSON.parse(localStorage.getItem("sentimentHistory") || "[]");

const fileInput = document.createElement("input");
fileInput.type = "file";
fileInput.accept = ".txt,text/plain";
fileInput.hidden = true;
document.body.appendChild(fileInput);

function getWords(text) {
    return text.trim() ? text.trim().split(/\s+/).length : 0;
}

function setStatus(message, type = "info") {
    statusMessage.textContent = message;
    statusMessage.dataset.type = type;
}

function updateInputStats() {
    const words = getWords(textInput.value);
    const chars = textInput.value.length;
    inputStats.textContent = `${words} words | ${chars} chars`;
}

function getEmoji(sentiment) {
    const label = sentiment.toLowerCase();
    if (label.includes("positive")) return ":)";
    if (label.includes("negative")) return ":(";
    return ":|";
}

function getSentimentClass(sentiment) {
    const label = sentiment.toLowerCase();
    if (label.includes("positive")) return "sentiment-positive";
    if (label.includes("negative")) return "sentiment-negative";
    return "sentiment-neutral";
}

function saveHistory() {
    localStorage.setItem("sentimentHistory", JSON.stringify(history.slice(0, 20)));
}

function renderHistory() {
    const totals = history.reduce((acc, item) => {
        acc.total += 1;
        acc[item.sentiment.toLowerCase()] = (acc[item.sentiment.toLowerCase()] || 0) + 1;
        return acc;
    }, { total: 0, positive: 0, negative: 0, neutral: 0 });

    totalCount.textContent = totals.total;
    positiveCount.textContent = totals.positive || 0;
    negativeCount.textContent = totals.negative || 0;
    neutralCount.textContent = totals.neutral || 0;

    if (!history.length) {
        historyList.innerHTML = '<p class="empty-state">No reviews analyzed yet.</p>';
        return;
    }

    historyList.innerHTML = history.map((item, index) => `
        <article class="history-item">
            <div>
                <strong class="${getSentimentClass(item.sentiment)}">${item.sentiment}</strong>
                <span>${item.confidence.toFixed(1)}%</span>
            </div>
            <p>${escapeHtml(item.text.slice(0, 120))}${item.text.length > 120 ? "..." : ""}</p>
            <button type="button" data-history-index="${index}">Reuse</button>
        </article>
    `).join("");
}

function escapeHtml(value) {
    return value
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function updateResult(data, originalText) {
    latestResult = {
        text: originalText,
        sentiment: data.sentiment,
        confidence: Number(data.confidence),
        reviewLength: data.review_length || getWords(originalText),
        createdAt: new Date().toISOString()
    };

    sentimentSpan.textContent = latestResult.sentiment;
    sentimentSpan.className = getSentimentClass(latestResult.sentiment);
    emojiSpan.textContent = getEmoji(latestResult.sentiment);
    confidenceFill.style.width = `${Math.min(latestResult.confidence, 100)}%`;
    confidenceFill.className = getSentimentClass(latestResult.sentiment);
    confidenceText.textContent = `Confidence: ${latestResult.confidence.toFixed(2)}%`;
    wordCountText.textContent = `Words: ${latestResult.reviewLength}`;
    resultCard.style.display = "grid";

    history = [latestResult, ...history.filter(item => item.text !== originalText)].slice(0, 20);
    saveHistory();
    renderHistory();
}

async function analyzeText({ speak = false } = {}) {
    const text = textInput.value.trim();
    if (!text) {
        setStatus("Enter, upload, or speak text before analysis.", "error");
        textInput.focus();
        return;
    }

    predictBtn.disabled = true;
    setStatus("Analyzing sentiment...", "loading");
    sentimentSpan.textContent = "Analyzing";
    emojiSpan.textContent = "";
    confidenceFill.style.width = "0%";

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ review: text })
        });
        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || "Prediction failed.");
        }

        updateResult(data, text);
        setStatus(`${data.sentiment} sentiment detected with ${Number(data.confidence).toFixed(1)}% confidence.`, "success");

        if (speak) {
            speakLatestResult();
        }
    } catch (error) {
        console.error(error);
        setStatus(error.message || "Something went wrong. Check the backend server.", "error");
        sentimentSpan.textContent = "Error";
    } finally {
        predictBtn.disabled = false;
    }
}

function setupVoiceRecognition() {
    if (!SpeechRecognition) {
        voiceBtn.disabled = true;
        voiceBtn.textContent = "Voice unavailable";
        setStatus("Voice input is not supported in this browser. Chrome or Edge works best.", "error");
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
        isListening = true;
        voiceBtn.textContent = "Listening...";
        voiceBtn.setAttribute("aria-pressed", "true");
        setStatus("Listening. Speak your review now.", "loading");
    };

    recognition.onresult = (event) => {
        const transcript = Array.from(event.results)
            .map(result => result[0].transcript)
            .join(" ");
        textInput.value = transcript.trim();
        updateInputStats();
    };

    recognition.onerror = (event) => {
        setStatus(`Voice input error: ${event.error}.`, "error");
    };

    recognition.onend = () => {
        isListening = false;
        voiceBtn.textContent = "Start voice";
        voiceBtn.setAttribute("aria-pressed", "false");
        if (textInput.value.trim()) {
            analyzeText({ speak: true });
        } else {
            setStatus("No speech was captured. Try again closer to the microphone.", "error");
        }
    };
}

function toggleVoiceInput() {
    if (!recognition) return;
    if (isListening) {
        recognition.stop();
        return;
    }
    recognition.start();
}

function speakLatestResult() {
    if (!latestResult) {
        setStatus("Analyze a review before using read result.", "error");
        return;
    }

    if (!window.speechSynthesis) {
        setStatus("Text-to-speech is not supported in this browser.", "error");
        return;
    }

    window.speechSynthesis.cancel();
    const message = `The sentiment is ${latestResult.sentiment}, with ${latestResult.confidence.toFixed(1)} percent confidence.`;
    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
    setStatus("Reading the result aloud.", "success");
}

function downloadHistory() {
    if (!history.length) {
        setStatus("There is no history to export yet.", "error");
        return;
    }

    const rows = [
        ["created_at", "sentiment", "confidence", "word_count", "text"],
        ...history.map(item => [
            item.createdAt,
            item.sentiment,
            item.confidence.toFixed(2),
            item.reviewLength,
            `"${item.text.replace(/"/g, '""')}"`
        ])
    ];
    const blob = new Blob([rows.map(row => row.join(",")).join("\n")], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "sentiment-history.csv";
    link.click();
    URL.revokeObjectURL(url);
    setStatus("History exported as CSV.", "success");
}

themeToggle.addEventListener("click", () => {
    const isDark = document.body.classList.toggle("dark-mode");
    document.body.classList.toggle("light-mode", !isDark);
    themeIcon.textContent = isDark ? "D" : "L";
});

predictBtn.addEventListener("click", () => analyzeText());
voiceBtn.addEventListener("click", toggleVoiceInput);
speakBtn.addEventListener("click", speakLatestResult);

clearBtn.addEventListener("click", () => {
    textInput.value = "";
    updateInputStats();
    setStatus("Input cleared.", "info");
    textInput.focus();
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
        textInput.value = event.target.result;
        updateInputStats();
        setStatus(`${file.name} loaded. Ready to analyze.`, "success");
    };
    reader.onerror = () => setStatus("Could not read that file.", "error");
    reader.readAsText(file);
});

copyResultBtn.addEventListener("click", async () => {
    if (!latestResult) {
        setStatus("Analyze a review before copying.", "error");
        return;
    }

    const text = `${latestResult.sentiment} sentiment (${latestResult.confidence.toFixed(2)}% confidence)`;
    await navigator.clipboard.writeText(text);
    setStatus("Result copied to clipboard.", "success");
});

downloadHistoryBtn.addEventListener("click", downloadHistory);

clearHistoryBtn.addEventListener("click", () => {
    history = [];
    saveHistory();
    renderHistory();
    setStatus("History cleared.", "info");
});

historyList.addEventListener("click", (event) => {
    const button = event.target.closest("[data-history-index]");
    if (!button) return;

    const item = history[Number(button.dataset.historyIndex)];
    textInput.value = item.text;
    updateInputStats();
    updateResult({
        sentiment: item.sentiment,
        confidence: item.confidence,
        review_length: item.reviewLength
    }, item.text);
    setStatus("Previous review loaded.", "success");
});

textInput.addEventListener("input", updateInputStats);

setupVoiceRecognition();
updateInputStats();
renderHistory();
