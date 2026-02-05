const statusEl = document.getElementById("status");
const promptInput = document.getElementById("prompt");
const bufferTargetInput = document.getElementById("bufferTarget");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const skipBtn = document.getElementById("skipBtn");
const playPauseBtn = document.getElementById("playPauseBtn");
const audioEl = document.getElementById("audio");
const trackTitleEl = document.getElementById("trackTitle");
const trackMetaEl = document.getElementById("trackMeta");
const lyricsEl = document.getElementById("lyrics");
const bufferCountEl = document.getElementById("bufferCount");
const bufferTargetDisplayEl = document.getElementById("bufferTargetDisplay");
const promptDisplayEl = document.getElementById("promptDisplay");
const bufferingIndicator = document.getElementById("bufferingIndicator");

let pollingInterval = null;
let waitingForTrack = false;

function setStatus(text, isActive = false) {
  statusEl.textContent = text;
  statusEl.classList.toggle("active", isActive);
}

async function fetchStatus() {
  const response = await fetch("/api/status");
  return response.json();
}

async function updateStatus() {
  const status = await fetchStatus();
  bufferCountEl.textContent = status.buffered_tracks;
  bufferTargetDisplayEl.textContent = status.buffer_target;
  promptDisplayEl.textContent = status.prompt || "None";
  bufferingIndicator.style.opacity = status.buffered_tracks === 0 && status.running ? 1 : 0;
  if (!status.running) {
    setStatus("Idle", false);
  } else if (status.buffered_tracks === 0) {
    setStatus("Buffering", true);
  } else {
    setStatus("Ready", true);
  }
}

async function startRadio() {
  const prompt = promptInput.value.trim();
  if (!prompt) {
    setStatus("Add a prompt first", false);
    return;
  }
  const bufferTarget = parseInt(bufferTargetInput.value, 10) || 2;
  await fetch("/api/start", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, buffer_target: bufferTarget }),
  });
  await updateStatus();
  await ensureTrack();
  startPolling();
}

async function stopRadio() {
  await fetch("/api/stop", { method: "POST" });
  audioEl.pause();
  audioEl.src = "";
  trackTitleEl.textContent = "No track loaded";
  trackMetaEl.textContent = "Awaiting prompt...";
  lyricsEl.textContent = "---";
  waitingForTrack = false;
  updatePlayPauseLabel();
  await updateStatus();
}

async function ensureTrack() {
  if (waitingForTrack) return;
  waitingForTrack = true;
  const response = await fetch("/api/next");
  if (!response.ok) {
    waitingForTrack = false;
    return;
  }
  const payload = await response.json();
  if (payload.status === "buffering") {
    waitingForTrack = false;
    return;
  }
  const track = payload.track;
  audioEl.src = track.audio_url;
  trackTitleEl.textContent = track.title;
  trackMetaEl.textContent = `${track.mood} • ${track.bpm} BPM • ${track.duration_seconds}s`;
  lyricsEl.textContent = track.lyrics;
  waitingForTrack = false;
  audioEl.play();
  updatePlayPauseLabel();
}

function updatePlayPauseLabel() {
  if (audioEl.paused) {
    playPauseBtn.textContent = "Play";
  } else {
    playPauseBtn.textContent = "Pause";
  }
}

function startPolling() {
  if (pollingInterval) return;
  pollingInterval = setInterval(async () => {
    await updateStatus();
    if (!audioEl.src || audioEl.ended) {
      await ensureTrack();
    }
  }, 1500);
}

startBtn.addEventListener("click", startRadio);
stopBtn.addEventListener("click", stopRadio);
skipBtn.addEventListener("click", async () => {
  audioEl.pause();
  audioEl.currentTime = 0;
  await ensureTrack();
});

playPauseBtn.addEventListener("click", () => {
  if (!audioEl.src) {
    ensureTrack();
    return;
  }
  if (audioEl.paused) {
    audioEl.play();
  } else {
    audioEl.pause();
  }
  updatePlayPauseLabel();
});

audioEl.addEventListener("ended", () => {
  ensureTrack();
});

audioEl.addEventListener("play", updatePlayPauseLabel);
audioEl.addEventListener("pause", updatePlayPauseLabel);

updateStatus();
