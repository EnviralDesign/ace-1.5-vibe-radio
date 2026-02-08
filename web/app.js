const MAX_LOG_LINES = 200;

const els = {
  starscape: document.getElementById("starscape"),
  status: document.getElementById("status"),
  statusDot: document.getElementById("statusDot"),
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
  stopBtnInline: document.getElementById("stopBtnInline"),
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
  pauseLogsBtn: document.getElementById("pauseLogsBtn"),
  presetSelect: document.getElementById("presetSelect"),
  ditModelSelect: document.getElementById("ditModelSelect"),
  lmModelSelect: document.getElementById("lmModelSelect"),
  lmBackendSelect: document.getElementById("lmBackendSelect"),
  offloadCpuToggle: document.getElementById("offloadCpuToggle"),
  offloadDitCpuToggle: document.getElementById("offloadDitCpuToggle"),
  inferenceStepsInput: document.getElementById("inferenceStepsInput"),
  guidanceScaleInput: document.getElementById("guidanceScaleInput"),
  durationInput: document.getElementById("durationInput"),
  thinkingToggle: document.getElementById("thinkingToggle"),
  reuseSampleToggle: document.getElementById("reuseSampleToggle"),
  applyConfigBtn: document.getElementById("applyConfigBtn"),
  devConfigState: document.getElementById("devConfigState"),
};

let state = {
  waitingForTrack: false,
  radioRunning: false,
  nextTrackPollTimer: null,
  nextTrackPollMs: 1200,
  isPlaying: false,
  visualizerActive: false,
  audioContext: null,
  analyser: null,
  source: null,
  socket: null,
  generationMode: "lyrics",
  vocalLanguage: "unknown",
  logAutoscroll: true,
  runtimeConfig: null,
  runtimeOptions: null,
  stars: [],
  starCtx: null,
  starFrame: null,
};

function appendLog(message, type = "") {
  const line = document.createElement("div");
  line.className = `log-entry ${type}`.trim();
  line.textContent = message;
  els.logContainer.appendChild(line);

  while (els.logContainer.children.length > MAX_LOG_LINES) {
    els.logContainer.removeChild(els.logContainer.firstChild);
  }

  if (state.logAutoscroll) {
    els.logContainer.scrollTop = els.logContainer.scrollHeight;
  }
}

function setStatus(text, kind = "offline") {
  els.status.textContent = text;
  els.connectionStatus.textContent = text;
  els.statusDot.classList.remove("live");
  if (kind === "online") {
    els.statusDot.classList.add("live");
  }
}

function setDevState(text) {
  els.devConfigState.textContent = text;
}

function setAppMode(isStreaming) {
  document.body.classList.toggle("is-streaming", Boolean(isStreaming));
}

function clearTrackPollTimer() {
  if (state.nextTrackPollTimer) {
    clearTimeout(state.nextTrackPollTimer);
    state.nextTrackPollTimer = null;
  }
}

function scheduleEnsureTrack(delayMs = 0) {
  if (!state.radioRunning || state.waitingForTrack || state.nextTrackPollTimer) return;
  state.nextTrackPollTimer = setTimeout(() => {
    state.nextTrackPollTimer = null;
    ensureTrack();
  }, Math.max(0, delayMs));
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
  els.vocalLanguage.value = state.vocalLanguage;
}

function updateTrackInfo(track) {
  if (!track) {
    els.trackTitle.textContent = "Waiting for transmission...";
    els.trackMeta.textContent = "---";
    els.lyrics.textContent = "---";
    els.progressBar.style.width = "0%";
    return;
  }

  els.trackTitle.textContent = track.title || "Untitled Signal";
  els.trackMeta.textContent = `${track.mood || "Unknown"} • ${track.bpm} BPM • ${Math.round(track.duration_seconds)}s`;
  els.lyrics.textContent = track.lyrics || "(Instrumental)";
}

function fillSelect(selectEl, values, labels = null) {
  selectEl.innerHTML = "";
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = labels && labels[value] ? labels[value] : value;
    selectEl.appendChild(opt);
  }
}

function applyRuntimePayload(payload) {
  if (!payload || !payload.config) return;

  state.runtimeConfig = payload.config;
  state.runtimeOptions = payload.options || state.runtimeOptions;

  const options = payload.options || {};
  if (options.presets) {
    const labels = {};
    const presetValues = [];
    for (const item of options.presets) {
      labels[item.id] = item.label;
      presetValues.push(item.id);
    }
    fillSelect(els.presetSelect, presetValues, labels);
  }
  if (options.dit_models) fillSelect(els.ditModelSelect, options.dit_models);
  if (options.lm_models) fillSelect(els.lmModelSelect, options.lm_models);
  if (options.lm_backends) fillSelect(els.lmBackendSelect, options.lm_backends);

  const cfg = payload.config;
  if (cfg.preset) els.presetSelect.value = cfg.preset;
  if (cfg.model_name) els.ditModelSelect.value = cfg.model_name;
  if (cfg.lm_model_path) els.lmModelSelect.value = cfg.lm_model_path;
  if (cfg.lm_backend) els.lmBackendSelect.value = cfg.lm_backend;
  els.offloadCpuToggle.checked = Boolean(cfg.offload_to_cpu);
  els.offloadDitCpuToggle.checked = Boolean(cfg.offload_dit_to_cpu);
  els.inferenceStepsInput.value = cfg.inference_steps ?? 8;
  els.guidanceScaleInput.value = cfg.guidance_scale ?? 7.0;
  els.durationInput.value = cfg.duration_seconds ?? 60;
  els.thinkingToggle.checked = Boolean(cfg.thinking);
  els.reuseSampleToggle.checked = Boolean(cfg.reuse_lm_sample);

  setDevState(payload.models_initialized ? "Engine ready" : "Engine reloading...");
}

async function fetchRuntimeConfig() {
  try {
    const res = await fetch("/api/dev/config");
    if (!res.ok) return;
    const payload = await res.json();
    applyRuntimePayload(payload);
  } catch (e) {
    appendLog(`Config fetch failed: ${e.message}`);
  }
}

async function applyRuntimeConfig() {
  const body = {
    preset: els.presetSelect.value,
    model_name: els.ditModelSelect.value,
    lm_model_path: els.lmModelSelect.value,
    lm_backend: els.lmBackendSelect.value,
    offload_to_cpu: els.offloadCpuToggle.checked,
    offload_dit_to_cpu: els.offloadDitCpuToggle.checked,
    inference_steps: Number(els.inferenceStepsInput.value),
    guidance_scale: Number(els.guidanceScaleInput.value),
    duration_seconds: Number(els.durationInput.value),
    thinking: els.thinkingToggle.checked,
    reuse_lm_sample: els.reuseSampleToggle.checked,
    restart_engine: true,
  };

  els.applyConfigBtn.disabled = true;
  setDevState("Applying config...");

  try {
    const res = await fetch("/api/dev/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || "Failed to apply runtime config");
    }

    applyRuntimePayload(data);
    appendLog("Developer config applied. Engine restarting...", "system");
  } catch (e) {
    setDevState("Apply failed");
    appendLog(`Config error: ${e.message}`);
  } finally {
    els.applyConfigBtn.disabled = false;
  }
}

function connectWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws`;

  state.socket = new WebSocket(wsUrl);

  state.socket.onopen = () => {
    appendLog("System link established", "system");
  };

  state.socket.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === "status") {
        updateUIStatus(msg.data);
      } else if (msg.type === "log") {
        appendLog(msg.data);
      } else if (msg.type === "track_ready") {
        if (state.radioRunning && (!els.audio.src || els.audio.ended || els.audio.paused || !state.isPlaying)) {
          scheduleEnsureTrack(0);
        }
      } else if (msg.type === "runtime_config") {
        applyRuntimePayload(msg.data);
      }
    } catch (e) {
      console.error("WS parse error", e);
    }
  };

  state.socket.onclose = () => {
    appendLog("System link lost. Retrying...");
    setStatus("DISCONNECTED", "offline");
    setTimeout(connectWebSocket, 3000);
  };
}

function updateUIStatus(data) {
  if (!data) return;
  state.radioRunning = Boolean(data.running);

  els.bufferCount.textContent = data.buffered_tracks;
  els.bufferTargetDisplay.textContent = data.buffer_target;

  if (data.generation_mode) setGenerationMode(data.generation_mode);
  if (data.vocal_language) setVocalLanguage(data.vocal_language);

  const disableModeToggle = Boolean(data.running);
  els.modeLyrics.disabled = disableModeToggle;
  els.modeInstrumental.disabled = disableModeToggle;
  els.vocalLanguage.disabled = disableModeToggle;

  if (!data.running) {
    els.startBtn.disabled = false;
    els.startBtn.textContent = "Stream Live";
    setStatus(data.is_loading_models ? "LOADING" : "IDLE", "offline");
    setAppMode(false);
  } else {
    els.startBtn.disabled = true;
    els.startBtn.textContent = "Stream Live";
    if (data.is_loading_models || data.buffered_tracks === 0) {
      setStatus("BUFFERING", "offline");
    } else {
      setStatus("ONLINE", "online");
    }
    setAppMode(true);
  }

  setDevState(data.is_loading_models ? "Engine reloading..." : "Engine ready");

  if (data.running && (!els.audio.src || els.audio.ended) && !state.waitingForTrack && data.buffered_tracks > 0) {
    ensureTrack();
  }
}

async function startRadio() {
  const prompt = els.prompt.value.trim();
  if (!prompt) {
    alert("Enter a transmission prompt first.");
    return;
  }

  initAudioContext();
  const bufferTarget = parseInt(els.bufferTarget.value, 10) || 2;

  els.startBtn.disabled = true;
  els.startBtn.textContent = "Starting...";
  appendLog(`Starting stream: "${prompt}"`, "system");
  setAppMode(true);

  try {
    const res = await fetch("/api/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        buffer_target: bufferTarget,
        generation_mode: state.generationMode,
        vocal_language: state.vocalLanguage,
      }),
    });
    if (!res.ok) {
      throw new Error(`Start failed (${res.status})`);
    }
    state.radioRunning = true;
  } catch (e) {
    els.startBtn.disabled = false;
    els.startBtn.textContent = "Stream Live";
    setAppMode(false);
    appendLog(`Start failed: ${e.message}`);
  }
}

async function stopRadio() {
  await fetch("/api/stop", { method: "POST" });
  state.radioRunning = false;
  clearTrackPollTimer();
  els.audio.pause();
  els.audio.src = "";
  updateTrackInfo(null);
  updatePlaybackControls(false);
  setAppMode(false);
  appendLog("Stream stopped", "system");
}

async function ensureTrack() {
  if (state.waitingForTrack || !state.radioRunning) return;
  state.waitingForTrack = true;

  try {
    const res = await fetch("/api/next");
    if (!res.ok) return;
    const data = await res.json();

    if (data.status === "buffering") {
      state.nextTrackPollMs = Math.min(5000, Math.round(state.nextTrackPollMs * 1.4));
      scheduleEnsureTrack(state.nextTrackPollMs);
      return;
    }

    const track = data.track;
    if (track) {
      state.nextTrackPollMs = 1200;
      appendLog(`Loading: ${track.title} (${track.bpm} BPM)`);
      playTrack(track);
    }
  } finally {
    state.waitingForTrack = false;
  }
}

function playTrack(track) {
  els.audio.src = track.audio_url;
  updateTrackInfo(track);

  const playPromise = els.audio.play();
  if (playPromise !== undefined) {
    playPromise.then(() => updatePlaybackControls(true)).catch((e) => {
      appendLog(`Autoplay blocked: ${e.message}`);
      updatePlaybackControls(false);
    });
  }
}

function resetStar(width, height) {
  const side = Math.floor(Math.random() * 4);
  const depth = 0.12 + Math.random() * 0.9;
  const speed = 0.2 + Math.random() * 1.2;
  const twinkle = 0.2 + Math.random() * 0.8;
  const hue = 185 + Math.random() * 40;
  const size = 0.5 + Math.random() * 1.8;
  const drift = (Math.random() - 0.5) * 0.3;
  let x = Math.random() * width;
  let y = Math.random() * height;

  if (side === 0) {
    x = Math.random() * width;
    y = -8;
  } else if (side === 1) {
    x = width + 8;
    y = Math.random() * height;
  } else if (side === 2) {
    x = Math.random() * width;
    y = height + 8;
  } else {
    x = -8;
    y = Math.random() * height;
  }

  return { x, y, depth, speed, twinkle, hue, size, drift };
}

function initStarfield() {
  if (!els.starscape) return;
  const ctx = els.starscape.getContext("2d");
  if (!ctx) return;
  state.starCtx = ctx;
  resizeStarfield();
  if (!state.starFrame) {
    animateStarfield();
  }
}

function resizeStarfield() {
  if (!els.starscape || !state.starCtx) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = window.innerWidth;
  const height = window.innerHeight;
  els.starscape.width = Math.floor(width * dpr);
  els.starscape.height = Math.floor(height * dpr);
  state.starCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const starCount = Math.max(100, Math.min(280, Math.floor((width * height) / 9000)));
  state.stars = Array.from({ length: starCount }, () => resetStar(width, height));
}

function getAudioEnergy() {
  if (!state.analyser) return 0;
  const bins = state.analyser.frequencyBinCount;
  if (!bins) return 0;
  const data = new Uint8Array(bins);
  state.analyser.getByteFrequencyData(data);

  let sum = 0;
  const sampleStep = Math.max(1, Math.floor(bins / 48));
  for (let i = 0; i < bins; i += sampleStep) {
    sum += data[i];
  }
  const avg = sum / Math.ceil(bins / sampleStep);
  return Math.min(1, avg / 255);
}

function animateStarfield() {
  const ctx = state.starCtx;
  if (!ctx || !els.starscape) return;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const cx = width * 0.5;
  const cy = height * 0.5;
  const energy = getAudioEnergy();
  const baseVelocity = state.isPlaying ? 0.95 : 0.45;
  const velocity = baseVelocity + energy * 1.6;

  ctx.clearRect(0, 0, width, height);

  for (const star of state.stars) {
    const dx = (star.x - cx) * (0.00028 * velocity * (1.2 - star.depth));
    const dy = (star.y - cy) * (0.00028 * velocity * (1.2 - star.depth));
    star.x += dx + star.drift * velocity;
    star.y += dy + velocity * star.speed * (0.4 + (1 - star.depth));

    if (star.x < -20 || star.x > width + 20 || star.y < -20 || star.y > height + 20) {
      Object.assign(star, resetStar(width, height));
      continue;
    }

    const flicker = 0.4 + 0.6 * Math.abs(Math.sin((performance.now() * 0.0012) * star.twinkle));
    const alpha = 0.16 + (1 - star.depth) * 0.62 * flicker;
    const r = star.size * (1.1 - star.depth);
    ctx.fillStyle = `hsla(${star.hue}, 90%, 78%, ${alpha})`;
    ctx.beginPath();
    ctx.arc(star.x, star.y, r, 0, Math.PI * 2);
    ctx.fill();

    if (velocity > 1.15 && star.depth < 0.5) {
      ctx.strokeStyle = `hsla(${star.hue}, 100%, 80%, ${alpha * 0.55})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(star.x, star.y);
      ctx.lineTo(star.x - dx * 14, star.y - dy * 14);
      ctx.stroke();
    }
  }

  state.starFrame = requestAnimationFrame(animateStarfield);
}

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

  const ctx = els.canvas.getContext("2d");
  const width = els.canvas.width;
  const height = els.canvas.height;

  ctx.fillStyle = "rgba(2, 8, 14, 0.22)";
  ctx.fillRect(0, 0, width, height);

  if (!state.analyser) return;

  const bufferLength = state.analyser.frequencyBinCount;
  const dataArray = new Uint8Array(bufferLength);
  state.analyser.getByteFrequencyData(dataArray);

  const barWidth = (width / bufferLength) * 2.1;
  let x = 0;
  for (let i = 0; i < bufferLength; i += 1) {
    const barHeight = dataArray[i] * 1.1;
    const hue = 175 + (i / bufferLength) * 45;
    const lightness = 38 + (barHeight / 255) * 30;
    ctx.fillStyle = `hsl(${hue}, 85%, ${lightness}%)`;
    ctx.fillRect(x, height - barHeight, barWidth, barHeight);
    x += barWidth + 1;
  }

  if (els.audio.duration) {
    const pct = (els.audio.currentTime / els.audio.duration) * 100;
    els.progressBar.style.width = `${pct}%`;
  }
}

function resizeCanvas() {
  els.canvas.width = els.canvas.offsetWidth;
  els.canvas.height = els.canvas.offsetHeight;
}

els.startBtn.addEventListener("click", startRadio);
els.stopBtn.addEventListener("click", stopRadio);
els.stopBtnInline.addEventListener("click", stopRadio);
els.modeLyrics.addEventListener("click", () => {
  if (!els.modeLyrics.disabled) setGenerationMode("lyrics");
});
els.modeInstrumental.addEventListener("click", () => {
  if (!els.modeInstrumental.disabled) setGenerationMode("instrumental");
});
els.vocalLanguage.addEventListener("change", (event) => {
  if (!els.vocalLanguage.disabled) setVocalLanguage(event.target.value);
});

els.skipBtn.addEventListener("click", () => {
  els.audio.pause();
  els.audio.currentTime = 0;
  appendLog("Skipping current track", "system");
  scheduleEnsureTrack(0);
});

els.playPauseBtn.addEventListener("click", () => {
  if (els.audio.paused) {
    if (state.audioContext && state.audioContext.state === "suspended") {
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
  els.logContainer.innerHTML = "";
});

els.pauseLogsBtn.addEventListener("click", () => {
  state.logAutoscroll = !state.logAutoscroll;
  els.pauseLogsBtn.textContent = state.logAutoscroll ? "Pause Scroll" : "Resume Scroll";
});

els.applyConfigBtn.addEventListener("click", applyRuntimeConfig);

els.audio.addEventListener("ended", () => {
  appendLog("Track ended");
  updatePlaybackControls(false);
  scheduleEnsureTrack(0);
});

window.addEventListener("resize", resizeCanvas);
window.addEventListener("resize", resizeStarfield);
resizeCanvas();
initStarfield();
connectWebSocket();
setAppMode(false);

const languageOptions = [
  "unknown",
  "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
  "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
  "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
  "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
  "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
];

fillSelect(
  els.vocalLanguage,
  languageOptions,
  { unknown: "Auto (any language)" },
);
setVocalLanguage("unknown");

fetchRuntimeConfig();
