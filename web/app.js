// DOM Elements
const els = {
  status: document.getElementById("status"),
  connectionStatus: document.getElementById("connectionStatus"),
  prompt: document.getElementById("prompt"),
  bufferTarget: document.getElementById("bufferTarget"),
  bufferCount: document.getElementById("bufferCount"),
  bufferTargetDisplay: document.getElementById("bufferTargetDisplay"),
  modeLyrics: document.getElementById("modeLyrics"),
  modeInstrumental: document.getElementById("modeInstrumental"),
  vocalLanguage: document.getElementById("vocalLanguage"),
  startBtn: document.getElementById("startBtn"),
  stopBtn: document.getElementById("stopBtn"),
  playPauseBtn: document.getElementById("playPauseBtn"),
  skipBtn: document.getElementById("skipBtn"),
  audio: document.getElementById("audio"),
  trackTitle: document.getElementById("trackTitle"),
  trackMeta: document.getElementById("trackMeta"),
  lyrics: document.getElementById("lyrics"),
  canvas: document.getElementById("visualizer"),
  progressBar: document.getElementById("progressBar"),
  playIcon: document.getElementById("playIcon"),
  pauseIcon: document.getElementById("pauseIcon"),
  logContainer: document.getElementById("logContainer"),
  clearLogsBtn: document.getElementById("clearLogsBtn"),
};

// State
let state = {
  waitingForTrack: false,
  isPlaying: false,
  visualizerActive: false,
  audioContext: null,
  analyser: null,
  source: null,
  socket: null,
  generationMode: "lyrics",
  vocalLanguage: "unknown",
};

// --- Update UI ---

function setStatus(text, type = "offline") {
  els.status.textContent = text;
  els.status.className = "status-badge"; // reset
  if (type === "online") els.status.classList.add("online");

  els.connectionStatus.textContent = text;
}

function updatePlaybackControls(isPlaying) {
  state.isPlaying = isPlaying;
  els.playIcon.style.display = isPlaying ? "none" : "block";
  els.pauseIcon.style.display = isPlaying ? "block" : "none";

  els.playPauseBtn.disabled = false;
  els.skipBtn.disabled = false;
}

function setGenerationMode(mode) {
  state.generationMode = mode === "instrumental" ? "instrumental" : "lyrics";
  const isLyrics = state.generationMode === "lyrics";
  els.modeLyrics.classList.toggle("active", isLyrics);
  els.modeInstrumental.classList.toggle("active", !isLyrics);
}

function setVocalLanguage(lang) {
  state.vocalLanguage = lang || "unknown";
  if (els.vocalLanguage) {
    els.vocalLanguage.value = state.vocalLanguage;
  }
}

function updateTrackInfo(track) {
  if (!track) {
    els.trackTitle.textContent = "Waiting for signal...";
    els.trackMeta.textContent = "---";
    els.lyrics.textContent = "---";
    els.progressBar.style.width = "0%";
    return;
  }

  els.trackTitle.textContent = track.title || "Untitled Signal";
  els.trackMeta.textContent = `${track.mood || 'Unknown'} • ${track.bpm} BPM • ${track.duration_seconds}s`;
  els.lyrics.textContent = track.lyrics || "(Instrumental)";
}

function appendLog(message) {
  const line = document.createElement("div");
  line.className = "log-entry";
  line.textContent = message;
  els.logContainer.appendChild(line);
  // Auto scroll
  els.logContainer.scrollTop = els.logContainer.scrollHeight;
}

// --- WebSocket & API Interaction ---

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  state.socket = new WebSocket(wsUrl);

  state.socket.onopen = () => {
    appendLog(">> System Link Established [SECURE]");
    // We don't need to manually fetch status, backend sends one on connect.
  };

  state.socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "status") {
        updateUIStatus(msg.data);
      } else if (msg.type === "log") {
        appendLog(msg.data);
      } else if (msg.type === "track_ready") {
        // Check if we need a track
        if ((!els.audio.src || els.audio.ended || els.audio.paused) && !state.isPlaying) {
          // Only auto-play if we are expecting stream
          if (els.startBtn.textContent === "Stream Active") {
            ensureTrack();
          }
        }
      }
    } catch (e) {
      console.error("WS Parse Error", e);
    }
  };

  state.socket.onclose = () => {
    appendLog(">> System Link Lost. Retrying connection...");
    setStatus("DISCONNECTED", "offline");
    setTimeout(connectWebSocket, 3000);
  };

  state.socket.onerror = (err) => {
    console.warn("WebSocket Error", err);
  };
}


function updateUIStatus(data) {
  if (data) {
    els.bufferCount.textContent = data.buffered_tracks;
    els.bufferTargetDisplay.textContent = data.buffer_target;

    if (data.generation_mode) {
      setGenerationMode(data.generation_mode);
    }
    if (data.vocal_language) {
      setVocalLanguage(data.vocal_language);
    }

    const disableModeToggle = Boolean(data.running);
    els.modeLyrics.disabled = disableModeToggle;
    els.modeInstrumental.disabled = disableModeToggle;
    if (els.vocalLanguage) {
      els.vocalLanguage.disabled = disableModeToggle;
    }

    if (!data.running) {
      els.startBtn.disabled = false;
      if (els.startBtn.textContent !== "Initialize Stream") {
        els.startBtn.textContent = "Initialize Stream";
        els.startBtn.classList.replace("secondary", "primary");
      }
      setStatus("IDLE", "offline");
    } else {
      els.startBtn.disabled = true;
      els.startBtn.textContent = "Stream Active";
      els.startBtn.classList.replace("primary", "secondary");

      if (data.buffered_tracks === 0 && !state.isPlaying && !els.audio.src) {
        setStatus("BUFFERING", "offline");
      } else {
        setStatus("ONLINE", "online");
      }
    }

    // Auto-fetch if stream is active and we have no track
    if (data.running && (!els.audio.src || els.audio.ended) && !state.waitingForTrack && data.buffered_tracks > 0) {
      ensureTrack();
    }
  }
}

async function startRadio() {
  const prompt = els.prompt.value.trim();
  if (!prompt) {
    alert("Please enter a Vibe Prompt first.");
    return;
  }

  // initialize Audio Context on user gesture
  initAudioContext();
  const bufferTarget = parseInt(els.bufferTarget.value, 10) || 2;

  els.startBtn.disabled = true;
  els.startBtn.textContent = "Initializing...";
  appendLog(`>> Initializing stream: "${prompt}"...`);

  try {
    await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        buffer_target: bufferTarget,
        generation_mode: state.generationMode,
        vocal_language: state.vocalLanguage,
      }),
    });
    // WS will handle status updates from here
  } catch (e) {
    console.error(e);
    els.startBtn.disabled = false;
    els.startBtn.textContent = "Initialize Stream";
    alert("Failed to start radio.");
  }
}

async function stopRadio() {
  await fetch("/api/stop", { method: "POST" });

  els.audio.pause();
  els.audio.src = "";
  updateTrackInfo(null);
  updatePlaybackControls(false);
  appendLog(">> Stream Terminated.");
}

async function ensureTrack() {
  if (state.waitingForTrack) return;
  state.waitingForTrack = true;

  try {
    const res = await fetch("/api/next");
    if (!res.ok) {
      // If 400 or empty, just ignore
      state.waitingForTrack = false;
      return;
    }
    const data = await res.json();

    if (data.status === "buffering") {
      state.waitingForTrack = false;
      return;
    }

    const track = data.track;
    if (track) {
      appendLog(`>> Loading Signal: ${track.title} [${track.bpm} BPM]`);
      playTrack(track);
    }

  } catch (e) {
    // console.log("No track available yet");
  } finally {
    state.waitingForTrack = false;
  }
}

function playTrack(track) {
  // Set source
  els.audio.src = track.audio_url;
  updateTrackInfo(track);

  // Attempt play
  const p = els.audio.play();
  if (p !== undefined) {
    p.then(() => {
      updatePlaybackControls(true);
    }).catch(e => {
      console.warn("Autoplay blocked or failed", e);
      appendLog(`>> Autoplay Blocked: ${e.message}. Click Play manually.`);
      updatePlaybackControls(false);
    });
  }
}

// --- Audio & Visualizer ---

function initAudioContext() {
  if (state.audioContext) return;

  const AudioContext = window.AudioContext || window.webkitAudioContext;
  state.audioContext = new AudioContext();
  state.analyser = state.audioContext.createAnalyser();
  state.analyser.fftSize = 256;

  state.source = state.audioContext.createMediaElementSource(els.audio);
  state.source.connect(state.analyser);
  state.analyser.connect(state.audioContext.destination);

  state.visualizerActive = true;
  drawVisualizer();
}

function drawVisualizer() {
  if (!state.visualizerActive) return;
  requestAnimationFrame(drawVisualizer);

  const bufferLength = state.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  state.analyser.getByteFrequencyData(dataArray);

  const ctx = els.canvas.getContext("2d");
  const width = els.canvas.width;
  const height = els.canvas.height;

  // Clear
  ctx.fillStyle = 'rgba(0, 0, 0, 0.2)'; // trail effect
  ctx.fillRect(0, 0, width, height);

  const barWidth = (width / bufferLength) * 2.5;
  let barHeight;
  let x = 0;

  for (let i = 0; i < bufferLength; i++) {
    barHeight = dataArray[i];

    const hue = 240 + (i / bufferLength) * 60; // Blue -> Purple
    const lightness = 50 + (barHeight / 255) * 40;
    ctx.fillStyle = `hsl(${hue}, 80%, ${lightness}%)`;

    ctx.fillRect(x, height - barHeight, barWidth, barHeight);

    x += barWidth + 1;
  }

  // Progress Bar update (piggyback on visualizer loop)
  if (els.audio.duration) {
    const pct = (els.audio.currentTime / els.audio.duration) * 100;
    els.progressBar.style.width = `${pct}%`;
  }
}


// --- Event Listeners ---

els.startBtn.addEventListener("click", startRadio);
els.stopBtn.addEventListener("click", stopRadio);
els.modeLyrics.addEventListener("click", () => {
  if (els.modeLyrics.disabled) return;
  setGenerationMode("lyrics");
});
els.modeInstrumental.addEventListener("click", () => {
  if (els.modeInstrumental.disabled) return;
  setGenerationMode("instrumental");
});
els.vocalLanguage.addEventListener("change", (event) => {
  if (els.vocalLanguage.disabled) return;
  setVocalLanguage(event.target.value);
});

els.skipBtn.addEventListener("click", () => {
  els.audio.pause();
  els.audio.currentTime = 0;
  appendLog(">> Skipping Signal...");
  ensureTrack();
});

els.playPauseBtn.addEventListener("click", () => {
  if (els.audio.paused) {
    // Resume context if needed
    if (state.audioContext && state.audioContext.state === 'suspended') {
      state.audioContext.resume();
    }
    els.audio.play();
    updatePlaybackControls(true);
  } else {
    els.audio.pause();
    updatePlaybackControls(false);
  }
});

els.clearLogsBtn.addEventListener("click", () => {
  els.logContainer.innerHTML = '';
});

// Audio Element Events
els.audio.addEventListener("ended", () => {
  appendLog(">> Signal Ended.");
  updatePlaybackControls(false);
  ensureTrack(); // Immediately fetch next
});

// Initialize canvas size
function resizeCanvas() {
  els.canvas.width = els.canvas.offsetWidth;
  els.canvas.height = els.canvas.offsetHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas(); // init

// Connect socket
connectWebSocket();

// Populate language picker
const languageOptions = [
  "unknown",
  "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
  "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
  "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
  "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
  "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
];

if (els.vocalLanguage) {
  els.vocalLanguage.innerHTML = "";
  for (const lang of languageOptions) {
    const opt = document.createElement("option");
    opt.value = lang;
    opt.textContent = lang === "unknown" ? "Auto (any language)" : lang;
    els.vocalLanguage.appendChild(opt);
  }
  setVocalLanguage(state.vocalLanguage);
}
