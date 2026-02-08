import asyncio
import os
import uuid
import logging
import random
import time
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, List, Dict, Any, Tuple
from collections import deque
from contextlib import asynccontextmanager
import queue
import urllib.request
import urllib.error

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Set environment variables before importing torch/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prefer upstream ACE-Step from submodule when available.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_ACE_DIR = PROJECT_ROOT / "vendor" / "ace-step"
if (UPSTREAM_ACE_DIR / "acestep").exists():
    upstream_path = str(UPSTREAM_ACE_DIR)
    if upstream_path not in sys.path:
        sys.path.insert(0, upstream_path)

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig, create_sample
from acestep.constants import VALID_LANGUAGES
from acestep.model_downloader import ensure_lm_model
from acestep.gpu_config import set_global_gpu_config, GPUConfig

DIT_MODEL_OPTIONS = [
    "acestep-v15-turbo",
    "acestep-v15-turbo-shift1",
    "acestep-v15-turbo-shift3",
    "acestep-v15-turbo-continuous",
    "acestep-v15-sft",
    "acestep-v15-base",
]

LM_MODEL_OPTIONS = [
    "acestep-5Hz-lm-0.6B",
    "acestep-5Hz-lm-1.7B",
    "acestep-5Hz-lm-4B",
]

LM_BACKEND_OPTIONS = ["pt", "vllm"]

PRESET_OPTIONS: Dict[str, Dict[str, Any]] = {
    "balanced": {
        "label": "Balanced",
        "model_name": "acestep-v15-turbo",
        "lm_model_path": "acestep-5Hz-lm-1.7B",
        "inference_steps": 8,
        "guidance_scale": 7.0,
        "duration_seconds": 120.0,
        "thinking": True,
    },
    "quality": {
        "label": "Quality",
        "model_name": "acestep-v15-sft",
        "lm_model_path": "acestep-5Hz-lm-4B",
        "inference_steps": 24,
        "guidance_scale": 7.0,
        "duration_seconds": 120.0,
        "thinking": True,
    },
    "fast": {
        "label": "Fast",
        "model_name": "acestep-v15-turbo",
        "lm_model_path": "acestep-5Hz-lm-0.6B",
        "inference_steps": 8,
        "guidance_scale": 6.5,
        "duration_seconds": 90.0,
        "thinking": True,
    },
}

# Load .env if present (best-effort, no override)
def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception:
        # Best-effort only
        pass


_load_env_file()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ace-radio")
ui_logger = logging.getLogger("ace-radio-ui")

AUDIO_DIR = Path(os.environ.get("ACE_RADIO_AUDIO_DIR", "./tmp"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Global handlers
dit_handler: Optional[AceStepHandler] = None
llm_handler: Optional[LLMHandler] = None
models_initialized = False


def _is_instrumental(lyrics: str) -> bool:
    """Determine if the track should be instrumental based on lyrics text."""
    if not lyrics:
        return True
    lyrics_clean = lyrics.strip().lower()
    if not lyrics_clean:
        return True
    return lyrics_clean in ("[inst]", "[instrumental]")


def _prompt_suggests_instrumental(prompt: str) -> bool:
    if not prompt:
        return False
    p = prompt.lower()
    cues = [
        "instrumental",
        "no vocals",
        "no vocal",
        "without vocals",
        "without vocal",
        "no singing",
        "without singing",
    ]
    return any(cue in p for cue in cues)

# WebSocket & Logging Integration
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
ui_log_queue = queue.Queue()

class QueueLogHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            ui_log_queue.put(msg)
        except Exception:
            self.handleError(record)

# Attach custom handler to UI logger only (avoid flooding frontend with all logs)
queue_handler = QueueLogHandler()
queue_handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
ui_logger.setLevel(logging.INFO)
ui_logger.propagate = False
ui_logger.addHandler(queue_handler)

async def log_broadcaster():
    """Reads logs from queue and broadcasts via WebSocket."""
    while True:
        while not ui_log_queue.empty():
            try:
                msg = ui_log_queue.get_nowait()
                await manager.broadcast({"type": "log", "data": msg})
            except queue.Empty:
                break
            except Exception as e:
                print(f"Log broadcast error: {e}")
        await asyncio.sleep(0.1)


def ui_log(message: str) -> None:
    """Send a polished log line to the frontend."""
    try:
        ui_logger.info(message)
    except Exception:
        pass


def _fmt_time(value: float) -> str:
    return f"{value:.2f}s"


def _format_time_costs(time_costs: Dict[str, Any]) -> List[str]:
    if not time_costs:
        return []
    lines: List[str] = []

    lm_phase1 = float(time_costs.get("lm_phase1_time", 0.0) or 0.0)
    lm_phase2 = float(time_costs.get("lm_phase2_time", 0.0) or 0.0)
    lm_total = float(time_costs.get("lm_total_time", 0.0) or 0.0)
    if lm_total > 0:
        parts = []
        if lm_phase1 > 0:
            parts.append(f"metas {_fmt_time(lm_phase1)}")
        if lm_phase2 > 0:
            parts.append(f"codes {_fmt_time(lm_phase2)}")
        parts.append(f"total {_fmt_time(lm_total)}")
        lines.append("LM done • " + " • ".join(parts))

    dit_encoder = float(time_costs.get("dit_encoder_time_cost", 0.0) or 0.0)
    dit_model = float(time_costs.get("dit_model_time_cost", 0.0) or 0.0)
    dit_decode = float(time_costs.get("dit_vae_decode_time_cost", 0.0) or 0.0)
    dit_offload = float(time_costs.get("dit_offload_time_cost", 0.0) or 0.0)
    dit_total = float(time_costs.get("dit_total_time_cost", 0.0) or 0.0)
    if dit_total > 0:
        parts = []
        if dit_encoder > 0:
            parts.append(f"encode {_fmt_time(dit_encoder)}")
        if dit_model > 0:
            parts.append(f"denoise {_fmt_time(dit_model)}")
        if dit_decode > 0:
            parts.append(f"decode {_fmt_time(dit_decode)}")
        if dit_offload > 0:
            parts.append(f"offload {_fmt_time(dit_offload)}")
        parts.append(f"total {_fmt_time(dit_total)}")
        lines.append("DiT done • " + " • ".join(parts))

    pipeline_total = float(time_costs.get("pipeline_total_time", 0.0) or 0.0)
    if pipeline_total > 0:
        lines.append(f"Pipeline total • {_fmt_time(pipeline_total)}")

    return lines


def _get_external_llm_config() -> Optional[Dict[str, str]]:
    api_key = os.environ.get("ACE_RADIO_EXT_LLM_API_KEY") or os.environ.get("GROK_API_KEY")
    base_url = os.environ.get("ACE_RADIO_EXT_LLM_BASE_URL") or os.environ.get("GROK_API_BASE_URL")
    model_id = os.environ.get("ACE_RADIO_EXT_LLM_MODEL") or os.environ.get("GROK_API_MODEL")
    if not api_key or not base_url or not model_id:
        return None
    return {
        "api_key": api_key.strip(),
        "base_url": base_url.strip(),
        "model": model_id.strip(),
    }


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # Remove optional json label
        cleaned = cleaned.replace("json", "", 1).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except Exception:
        return None


def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(float(value))
        return v
    except Exception:
        return None


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _external_llm_create_sample(
    prompt: str,
    instrumental: bool,
    vocal_language: Optional[str],
    config: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    base_url = config["base_url"].rstrip("/")
    url = f"{base_url}/chat/completions"
    language_hint = vocal_language if vocal_language else "unknown"
    system_msg = (
        "You are a music metadata generator. "
        "Return ONLY valid JSON with keys: caption, lyrics, bpm, keyscale, timesignature, language. "
        "Use language codes like en, fr, es, ja, etc. Use 'unknown' if unsure. "
        "bpm must be an integer 30-300 or null. "
        "timesignature must be one of 2,3,4,6 or null. "
        "keyscale must be like 'C Major' or 'A minor' or empty string. "
        "If instrumental is true, set lyrics to \"[Instrumental]\"."
    )
    user_msg = (
        f"Prompt: {prompt}\n"
        f"Instrumental: {instrumental}\n"
        f"Preferred language: {language_hint}\n"
    )
    payload = {
        "model": config["model"],
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.7,
        "top_p": 0.95,
        "max_tokens": 900,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config['api_key']}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="ignore")
        except Exception:
            raw = str(e)
        logger.warning(f"External LLM HTTP error: {raw}")
        return None
    except Exception as e:
        logger.warning(f"External LLM request failed: {e}")
        return None

    try:
        data = json.loads(raw)
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception:
        content = raw

    parsed = _extract_json_block(content)
    if not parsed:
        logger.warning("External LLM returned non-JSON output")
        return None

    caption = _safe_str(parsed.get("caption"))
    lyrics = _safe_str(parsed.get("lyrics"))
    bpm = _safe_int(parsed.get("bpm"))
    keyscale = _safe_str(parsed.get("keyscale"))
    timesignature = _safe_int(parsed.get("timesignature"))
    language = _safe_str(parsed.get("language"))

    return {
        "caption": caption,
        "lyrics": lyrics,
        "bpm": bpm,
        "keyscale": keyscale,
        "timesignature": timesignature,
        "language": language,
    }

@dataclass
class Track:
    track_id: str
    title: str
    prompt: str
    lyrics: str
    duration_seconds: float
    created_at: datetime
    audio_path: Path
    bpm: int
    mood: str

@dataclass
class RadioState:
    prompt: str = ""
    buffer_target: int = 2
    running: bool = False
    generation_mode: str = "lyrics"  # "lyrics" or "instrumental"
    vocal_language: str = "unknown"
    tracks: Deque[Track] = field(default_factory=deque)
    generation_task: Optional[asyncio.Task] = None
    initialization_task: Optional[asyncio.Task] = None
    runtime_config: Dict[str, Any] = field(default_factory=dict)

state = RadioState()
state_lock = asyncio.Lock()

class StartRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    buffer_target: int = Field(2, ge=1, le=6)
    generation_mode: str = Field("lyrics")
    vocal_language: str = Field("unknown")

class StatusResponse(BaseModel):
    running: bool
    prompt: str
    buffer_target: int
    buffered_tracks: int
    now: str
    is_loading_models: bool
    generation_mode: str
    vocal_language: str

class TrackResponse(BaseModel):
    status: str
    track: Optional[dict] = None


class RuntimeConfigRequest(BaseModel):
    preset: Optional[str] = None
    model_name: Optional[str] = None
    lm_model_path: Optional[str] = None
    lm_backend: Optional[str] = None
    offload_to_cpu: Optional[bool] = None
    offload_dit_to_cpu: Optional[bool] = None
    inference_steps: Optional[int] = Field(default=None, ge=1, le=100)
    guidance_scale: Optional[float] = Field(default=None, ge=0.0, le=30.0)
    duration_seconds: Optional[float] = Field(default=None, ge=10.0, le=600.0)
    thinking: Optional[bool] = None
    restart_engine: bool = True


def _normalize_runtime_config(config: Dict[str, Any]) -> Dict[str, Any]:
    normalized = dict(config)

    preset = str(normalized.get("preset") or "balanced").strip().lower()
    if preset not in PRESET_OPTIONS:
        preset = "balanced"
    normalized["preset"] = preset

    model_name = str(normalized.get("model_name") or "acestep-v15-turbo").strip()
    if not model_name:
        model_name = "acestep-v15-turbo"
    normalized["model_name"] = model_name

    lm_model_path = str(normalized.get("lm_model_path") or "acestep-5Hz-lm-1.7B").strip()
    if not lm_model_path:
        lm_model_path = "acestep-5Hz-lm-1.7B"
    normalized["lm_model_path"] = lm_model_path

    lm_backend = str(normalized.get("lm_backend") or "pt").strip().lower()
    if lm_backend not in LM_BACKEND_OPTIONS:
        lm_backend = "pt"
    normalized["lm_backend"] = lm_backend

    normalized["offload_to_cpu"] = bool(normalized.get("offload_to_cpu", False))
    normalized["offload_dit_to_cpu"] = bool(normalized.get("offload_dit_to_cpu", False))

    try:
        inference_steps = int(normalized.get("inference_steps", 8))
    except Exception:
        inference_steps = 8
    normalized["inference_steps"] = max(1, min(100, inference_steps))

    try:
        guidance_scale = float(normalized.get("guidance_scale", 7.0))
    except Exception:
        guidance_scale = 7.0
    normalized["guidance_scale"] = max(0.0, min(30.0, guidance_scale))

    try:
        duration_seconds = float(normalized.get("duration_seconds", 120.0))
    except Exception:
        duration_seconds = 120.0
    normalized["duration_seconds"] = max(10.0, min(600.0, duration_seconds))

    normalized["thinking"] = bool(normalized.get("thinking", True))
    return normalized


def _build_runtime_config_from_env() -> Dict[str, Any]:
    env_model = os.environ.get("ACESTEP_MODEL_NAME", "").strip()
    env_lm = os.environ.get("ACESTEP_LM_MODEL_PATH", "").strip()
    env_backend = os.environ.get("ACE_RADIO_LM_BACKEND", os.environ.get("ACESTEP_LM_BACKEND", "pt")).strip().lower()

    config = dict(PRESET_OPTIONS["balanced"])
    config["preset"] = "balanced"
    if env_model:
        config["model_name"] = env_model
    if env_lm:
        config["lm_model_path"] = env_lm
    if env_backend:
        config["lm_backend"] = env_backend
    config["offload_to_cpu"] = str(os.environ.get("ACE_RADIO_OFFLOAD_TO_CPU", "0")).lower() in {"1", "true", "yes"}
    config["offload_dit_to_cpu"] = str(os.environ.get("ACE_RADIO_OFFLOAD_DIT_TO_CPU", "0")).lower() in {"1", "true", "yes"}
    config["inference_steps"] = int(os.environ.get("ACE_RADIO_INFERENCE_STEPS", config["inference_steps"]))
    config["guidance_scale"] = float(os.environ.get("ACE_RADIO_GUIDANCE_SCALE", config["guidance_scale"]))
    config["duration_seconds"] = float(os.environ.get("ACE_RADIO_DURATION_SECONDS", config["duration_seconds"]))
    config["thinking"] = str(os.environ.get("ACE_RADIO_THINKING", "1")).lower() not in {"0", "false", "no"}
    return _normalize_runtime_config(config)


def _runtime_config_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "config": dict(config),
        "options": {
            "presets": [
                {"id": preset_id, "label": preset_data["label"]}
                for preset_id, preset_data in PRESET_OPTIONS.items()
            ],
            "dit_models": DIT_MODEL_OPTIONS,
            "lm_models": LM_MODEL_OPTIONS,
            "lm_backends": LM_BACKEND_OPTIONS,
        },
        "models_initialized": models_initialized,
    }

async def broadcast_status():
    """Helper to send status update to all clients."""
    st = await get_status()
    await manager.broadcast({
        "type": "status",
        "data": st.dict()
    })

async def initialize_models():
    """Initialize ACE-Step models in the background."""
    global dit_handler, llm_handler, models_initialized
    
    logger.info("Initializing ACE-Step models...")
    
    # Run in thread to avoid blocking loop
    async with state_lock:
        runtime_cfg = dict(state.runtime_config)

    def _init_sync(config: Dict[str, Any]):
        d_handler = AceStepHandler()
        l_handler = LLMHandler()
        
        project_root = os.getcwd()
        model_name = config["model_name"]
        
        # Initialize DiT
        msg, success = d_handler.initialize_service(
            project_root=project_root,
            config_path=model_name,
            device="auto",
            use_flash_attention=False,
            compile_model=False,
            offload_to_cpu=config["offload_to_cpu"],
            offload_dit_to_cpu=config["offload_dit_to_cpu"],
        )
        logger.info(f"DiT Init: {msg}")
        if not success:
            raise RuntimeError(f"Failed to initialize DiT: {msg}")
            
        # Initialize LLM (5Hz)
        logger.info("Initializing LLM...")
        lm_model_path = config["lm_model_path"]
        prefer_source = os.environ.get("ACESTEP_DOWNLOAD_SOURCE")
        if prefer_source:
            prefer_source = prefer_source.strip().lower()
        checkpoint_dir = os.path.join(project_root, "checkpoints")
        # Auto-download LM model if not present
        try:
            ensure_lm_model(
                model_name=lm_model_path or None,
                checkpoints_dir=Path(checkpoint_dir),
                prefer_source=prefer_source or None,
            )
        except Exception as e:
            logger.warning(f"LM auto-download failed: {e}")
        lm_backend = config["lm_backend"]
        if lm_backend not in {"pt", "vllm"}:
            logger.warning(f"Unknown LM backend '{lm_backend}', falling back to 'pt'")
            lm_backend = "pt"
        if lm_backend == "vllm" and torch.cuda.is_available():
            try:
                major, minor = torch.cuda.get_device_capability(0)
                if major < 8:
                    logger.warning(f"vLLM requires Ampere+ GPUs. Detected capability {major}.{minor}; falling back to PyTorch.")
                    ui_log(f"vLLM disabled • requires Ampere+ (found {major}.{minor})")
                    lm_backend = "pt"
            except Exception as e:
                logger.warning(f"Failed to check GPU capability: {e}. Falling back to PyTorch.")
                lm_backend = "pt"
        lm_msg, lm_success = l_handler.initialize(
            checkpoint_dir=checkpoint_dir,
            lm_model_path=lm_model_path or None,
            backend=lm_backend,
            offload_to_cpu=config["offload_to_cpu"],
        )
        logger.info(f"LLM Init: {lm_msg}")
        if not lm_success:
             logger.warning(f"LLM initialization failed: {lm_msg}. Proceeding without LLM constrained decoding/thinking.")
        
        return d_handler, l_handler

    try:
        dit_handler, llm_handler = await asyncio.to_thread(_init_sync, runtime_cfg)
        models_initialized = True
        logger.info("Models initialized successfully.")
        logger.info("\n" + "="*40 + "\n🚀  Server is ready\n" + "="*40)
        ui_log("Models ready • stream can start")
        await manager.broadcast({"type": "runtime_config", "data": _runtime_config_payload(runtime_cfg)})
        await broadcast_status()
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        ui_log(f"Model init failed • {e}")
        await manager.broadcast({"type": "runtime_config", "data": _runtime_config_payload(runtime_cfg)})


async def reload_models(reason: str = "Config updated") -> None:
    global models_initialized, dit_handler, llm_handler

    models_initialized = False
    dit_handler = None
    llm_handler = None
    ui_log(f"Reloading models • {reason}")

    async with state_lock:
        state.tracks.clear()
        if state.initialization_task and not state.initialization_task.done():
            state.initialization_task.cancel()
        state.initialization_task = asyncio.create_task(initialize_models())

    await broadcast_status()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start initialization task
    async with state_lock:
        state.runtime_config = _build_runtime_config_from_env()
    state.initialization_task = asyncio.create_task(initialize_models())
    log_task = asyncio.create_task(log_broadcaster())
    yield
    # Cleanup if needed
    if state.initialization_task:
        state.initialization_task.cancel()
    if state.generation_task:
        state.generation_task.cancel()
    log_task.cancel()

app = FastAPI(title="Ace 1.5 Vibe Radio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web"), name="static")


async def compose_track(prompt: str, generation_mode: str, vocal_language: str) -> Track:
    if not models_initialized or not dit_handler:
        raise RuntimeError("Models are not initialized yet.")

    track_id = str(uuid.uuid4())
    track_start = time.perf_counter()
    mode = (generation_mode or "lyrics").strip().lower()
    if mode not in ("lyrics", "instrumental"):
        mode = "lyrics"

    ui_log(f"Generation started • \"{prompt}\" • mode={mode}")

    # Optionally create a full sample (caption + lyrics + metadata) via LLM
    sample_used = False
    sample_caption = prompt
    sample_lyrics = ""
    sample_bpm: Optional[int] = None
    sample_keyscale = ""
    sample_timesignature = ""
    sample_language = "unknown"
    selected_language = (vocal_language or "unknown").strip().lower()
    if selected_language not in VALID_LANGUAGES:
        selected_language = "unknown"
    user_language = None if selected_language in ("unknown", "auto") else selected_language

    ext_cfg = _get_external_llm_config()
    if ext_cfg:
        try:
            sample_start = time.perf_counter()
            ext_sample = _external_llm_create_sample(
                prompt=prompt,
                instrumental=True if mode == "instrumental" else False,
                vocal_language=user_language,
                config=ext_cfg,
            )
            if ext_sample:
                sample_used = True
                sample_caption = ext_sample.get("caption") or prompt
                sample_lyrics = ext_sample.get("lyrics") or ""
                sample_bpm = ext_sample.get("bpm")
                sample_keyscale = ext_sample.get("keyscale") or ""
                ts = ext_sample.get("timesignature")
                sample_timesignature = str(ts) if ts else ""
                sample_language = ext_sample.get("language") or "unknown"
                sample_elapsed = time.perf_counter() - sample_start
                ui_log(f"External LM sample ready • {_fmt_time(sample_elapsed)}")
            else:
                logger.warning("External LM sample failed; falling back to local LM")
        except Exception as e:
            logger.warning(f"External LM sample error: {e}")

    if not sample_used and llm_handler and llm_handler.llm_initialized:
        try:
            sample_start = time.perf_counter()
            sample = create_sample(
                llm_handler=llm_handler,
                query=prompt,
                instrumental=True if mode == "instrumental" else False,
                vocal_language=user_language,
            )
            if sample.success:
                sample_used = True
                sample_caption = sample.caption or prompt
                sample_lyrics = sample.lyrics or ""
                sample_bpm = sample.bpm
                sample_keyscale = sample.keyscale or ""
                sample_timesignature = sample.timesignature or ""
                sample_language = sample.language or "unknown"
                sample_elapsed = time.perf_counter() - sample_start
                ui_log(f"LM sample ready • {_fmt_time(sample_elapsed)}")
            else:
                logger.warning(f"create_sample failed: {sample.error or sample.status_message}")
        except Exception as e:
            logger.warning(f"create_sample error: {e}")

    # Configure generation
    if mode == "instrumental" and not sample_lyrics:
        sample_lyrics = "[Instrumental]"

    effective_language = user_language or sample_language or "unknown"
    async with state_lock:
        runtime_cfg = dict(state.runtime_config)

    params = GenerationParams(
        task_type="text2music",
        caption=sample_caption,
        lyrics=sample_lyrics,
        instrumental=_is_instrumental(sample_lyrics),
        vocal_language=effective_language,
        bpm=sample_bpm,
        keyscale=sample_keyscale,
        timesignature=sample_timesignature,
        thinking=runtime_cfg["thinking"],
        duration=runtime_cfg["duration_seconds"],
        inference_steps=runtime_cfg["inference_steps"],
        guidance_scale=runtime_cfg["guidance_scale"],
    )
    
    config = GenerationConfig(
        batch_size=1,
        audio_format="wav" # web usage
    )
    
    logger.info(f"Generating track for prompt: {prompt}")
    
    # Run generation in thread
    def _run_gen():
        result = generate_music(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            params=params,
            config=config,
            save_dir=str(AUDIO_DIR)
        )
        return result

    result = await asyncio.to_thread(_run_gen)
    
    if not result.success or not result.audios:
        ui_log(f"Generation failed • {result.error or result.status_message}")
        raise RuntimeError(f"Generation failed: {result.error or result.status_message}")

    audio_data = result.audios[0]
    
    # RENAME FILE TO MATCH TRACK ID for simpler serving
    original_path = Path(audio_data["path"])
    # Ensure we use the correct extension
    ext = original_path.suffix
    new_path = AUDIO_DIR / f"{track_id}{ext}"
    try:
        if original_path.exists():
            original_path.rename(new_path)
            logger.info(f"Renamed audio to {new_path}")
        else:
            logger.warning(f"Original audio path not found: {original_path}")
            # fall back if somehow the path is weird, but usually rename works
            new_path = original_path 
    except Exception as e:
        logger.error(f"Failed to rename audio file: {e}")
        new_path = original_path

    audio_path = new_path
    params_used = audio_data.get("params", {})
    extra_outputs = result.extra_outputs if isinstance(result.extra_outputs, dict) else {}
    extra = extra_outputs.get("lm_metadata") or {}
    if not isinstance(extra, dict):
        extra = {}
    time_costs = extra_outputs.get("time_costs", {}) if extra_outputs else {}
    for line in _format_time_costs(time_costs):
        ui_log(line)
    end_to_end = time.perf_counter() - track_start
    ui_log(f"End-to-end • {_fmt_time(end_to_end)}")
    
    # Extract metadata
    title = f"{prompt.title()}"
    mood = sample_keyscale or extra.get("keyscale", "Unknown")
    if sample_bpm is not None:
        bpm = int(sample_bpm)
    else:
        bpm = int(extra.get("bpm", 120) or 120)
    lyrics = sample_lyrics if sample_used else extra.get("lyrics", "")
    if not lyrics and not sample_used and extra.get("caption"):
        lyrics = extra.get("caption")
         
    track = Track(
        track_id=track_id,
        title=title,
        prompt=prompt,
        lyrics=lyrics,
        duration_seconds=params.duration,
        created_at=datetime.utcnow(),
        audio_path=audio_path,
        bpm=bpm,
        mood=mood,
    )
    ui_log(f"Track ready • {track.title} • {track.bpm} BPM • {int(track.duration_seconds)}s")
    return track

async def generator_loop() -> None:
    while True:
        async with state_lock:
            if not state.running:
                return
            target = state.buffer_target
            prompt = state.prompt
            generation_mode = state.generation_mode
            vocal_language = state.vocal_language
            current_tracks = len(state.tracks)
        
        needs_track = current_tracks < target
        
        if not needs_track or not models_initialized:
            await asyncio.sleep(1.0)
            continue
            
        try:
            track = await compose_track(prompt, generation_mode, vocal_language)
            status_snapshot: Optional[StatusResponse] = None
            track_ready_id: Optional[str] = None
            async with state_lock:
                if not state.running:
                    return
                if len(state.tracks) < state.buffer_target:
                    state.tracks.append(track)
                    track_ready_id = track.track_id
                    status_snapshot = StatusResponse(
                        running=state.running,
                        prompt=state.prompt,
                        buffer_target=state.buffer_target,
                        buffered_tracks=len(state.tracks),
                        now=datetime.utcnow().isoformat() + "Z",
                        is_loading_models=not models_initialized,
                        generation_mode=state.generation_mode,
                        vocal_language=state.vocal_language,
                    )
            if status_snapshot:
                await manager.broadcast({"type": "status", "data": status_snapshot.dict()})
            if track_ready_id:
                await manager.broadcast({"type": "track_ready", "track_id": track_ready_id})
        except Exception as e:
            logger.error(f"Error generating track: {e}")
            ui_log(f"Generation error • {e}")
            await asyncio.sleep(5.0) 

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial status
        st = await get_status()
        await websocket.send_json({"type": "status", "data": st.dict()})
        async with state_lock:
            config_snapshot = dict(state.runtime_config)
        await websocket.send_json({"type": "runtime_config", "data": _runtime_config_payload(config_snapshot)})
        
        while True:
            # Keep alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def index() -> FileResponse:
    return FileResponse("web/index.html")

@app.post("/api/start", response_model=StatusResponse)
async def start_radio(request: StartRequest) -> StatusResponse:
    if not models_initialized:
        # Just allow it, loop waits
        pass

    async with state_lock:
        state.prompt = request.prompt.strip()
        state.buffer_target = request.buffer_target
        mode = (request.generation_mode or "lyrics").strip().lower()
        if mode not in ("lyrics", "instrumental"):
            mode = "lyrics"
        state.generation_mode = mode
        lang = (request.vocal_language or "unknown").strip().lower()
        if lang == "auto":
            lang = "unknown"
        if lang not in VALID_LANGUAGES:
            lang = "unknown"
        state.vocal_language = lang
        state.running = True
        state.tracks.clear()
        if state.generation_task and not state.generation_task.done():
            state.generation_task.cancel()
        state.generation_task = asyncio.create_task(generator_loop())

    ui_log(f"Stream started • mode={state.generation_mode} • lang={state.vocal_language} • buffer={state.buffer_target}")
    await broadcast_status()
    return await get_status()

@app.post("/api/stop", response_model=StatusResponse)
async def stop_radio() -> StatusResponse:
    async with state_lock:
        state.running = False
        state.prompt = ""
        state.tracks.clear()
        if state.generation_task and not state.generation_task.done():
            state.generation_task.cancel()

    ui_log("Stream stopped")
    await broadcast_status()
    return await get_status()


@app.get("/api/dev/config")
async def get_runtime_config() -> dict:
    async with state_lock:
        config_snapshot = dict(state.runtime_config)
    return _runtime_config_payload(config_snapshot)


@app.post("/api/dev/config")
async def update_runtime_config(request: RuntimeConfigRequest) -> dict:
    async with state_lock:
        current = dict(state.runtime_config)

        if request.preset:
            preset = request.preset.strip().lower()
            if preset not in PRESET_OPTIONS:
                raise HTTPException(status_code=400, detail=f"Unknown preset '{request.preset}'")
            preset_values = dict(PRESET_OPTIONS[preset])
            preset_values["preset"] = preset
            current.update(preset_values)

        for key in [
            "model_name",
            "lm_model_path",
            "lm_backend",
            "offload_to_cpu",
            "offload_dit_to_cpu",
            "inference_steps",
            "guidance_scale",
            "duration_seconds",
            "thinking",
        ]:
            value = getattr(request, key)
            if value is not None:
                current[key] = value

        state.runtime_config = _normalize_runtime_config(current)
        config_snapshot = dict(state.runtime_config)

    await manager.broadcast({"type": "runtime_config", "data": _runtime_config_payload(config_snapshot)})

    if request.restart_engine:
        await reload_models(reason="Developer config changed")

    return {
        "ok": True,
        "restarted": bool(request.restart_engine),
        **_runtime_config_payload(config_snapshot),
    }

@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    async with state_lock:
        return StatusResponse(
            running=state.running,
            prompt=state.prompt,
            buffer_target=state.buffer_target,
            buffered_tracks=len(state.tracks),
            now=datetime.utcnow().isoformat() + "Z",
            is_loading_models=not models_initialized,
            generation_mode=state.generation_mode,
            vocal_language=state.vocal_language,
        )

@app.get("/api/next", response_model=TrackResponse)
async def next_track() -> TrackResponse:
    async with state_lock:
        if not state.running:
            raise HTTPException(status_code=400, detail="Radio is not running")
        if not state.tracks:
            return TrackResponse(status="buffering")
        track = state.tracks.popleft()
    
    await broadcast_status()
    
    return TrackResponse(
        status="ready",
        track={
            "track_id": track.track_id,
            "title": track.title,
            "prompt": track.prompt,
            "lyrics": track.lyrics,
            "duration_seconds": track.duration_seconds,
            "bpm": track.bpm,
            "mood": track.mood,
            "audio_url": f"/api/audio/{track.track_id}",
            "created_at": track.created_at.isoformat() + "Z",
        },
    )

@app.get("/api/audio/{track_id}")
async def audio(track_id: str) -> FileResponse:
    # Look for the file by trying specific extensions since request might not have it
    audio_path = AUDIO_DIR / f"{track_id}.wav"
    if not audio_path.exists():
         # try extensions
         for ext in [".wav", ".flac", ".mp3"]:
             p = AUDIO_DIR / f"{track_id}{ext}"
             if p.exists():
                 audio_path = p
                 break
    
    if not audio_path.exists():
        logger.error(f"Media 404: {track_id} not found in {AUDIO_DIR}")
        raise HTTPException(status_code=404, detail="Track audio not found")
        
    return FileResponse(str(audio_path), media_type=f"audio/{audio_path.suffix.strip('.')}")

@app.get("/api/history")
async def history_stub() -> dict:
    return {"items": []}

@app.get("/api/style-samples")
async def style_samples_stub() -> dict:
    return {"enabled": False, "message": "Style sample upload coming soon."}
