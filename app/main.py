import asyncio
import os
import uuid
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional, List, Dict, Any
from collections import deque
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Set environment variables before importing torch/transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import generate_music, GenerationParams, GenerationConfig
from acestep.gpu_config import set_global_gpu_config, GPUConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ace-radio")

AUDIO_DIR = Path(os.environ.get("ACE_RADIO_AUDIO_DIR", "./tmp"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Global handlers
dit_handler: Optional[AceStepHandler] = None
llm_handler: Optional[LLMHandler] = None
models_initialized = False

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
    tracks: Deque[Track] = field(default_factory=deque)
    generation_task: Optional[asyncio.Task] = None
    initialization_task: Optional[asyncio.Task] = None

state = RadioState()
state_lock = asyncio.Lock()

class StartRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    buffer_target: int = Field(2, ge=1, le=6)

class StatusResponse(BaseModel):
    running: bool
    prompt: str
    buffer_target: int
    buffered_tracks: int
    now: str
    is_loading_models: bool

class TrackResponse(BaseModel):
    status: str
    track: Optional[dict] = None

async def initialize_models():
    """Initialize ACE-Step models in the background."""
    global dit_handler, llm_handler, models_initialized
    
    logger.info("Initializing ACE-Step models...")
    
    # Run in thread to avoid blocking loop
    def _init_sync():
        d_handler = AceStepHandler()
        l_handler = LLMHandler()
        
        project_root = os.getcwd()
        # You might want to let the user configure these via env vars
        model_name = os.environ.get("ACESTEP_MODEL_NAME", "acestep-v15-turbo")
        
        # Initialize DiT
        msg, success = d_handler.initialize_service(
            project_root=project_root,
            config_path=model_name,
            device="auto",
            use_flash_attention=True,
            compile_model=False
        )
        logger.info(f"DiT Init: {msg}")
        if not success:
            raise RuntimeError(f"Failed to initialize DiT: {msg}")
            
        # Initialize LLM (5Hz)
        # Using default 'acestep-5Hz-lm-1.7B' inside initialize() if None passed
        # Try to use 'vllm' backend if available, else 'pt'
        backend = "vllm" if torch.version.cuda and os.name != 'nt' else "pt" # Simple check, Windows usually simpler with pt for now unless configured
        # Note: acestep code attempts vllm first then falls back in _initialize_5hz_lm_vllm if it fails.
        # But let's stick to 'pt' for Windows user unless they have nano-vllm working perfectly.
        # User is on Windows. 'nano-vllm' might be tricky. Let's try 'pt' as safe default or let handler decide.
        # handler initialize() default backend is 'vllm'.
        
        lm_msg, lm_success = l_handler.initialize(
            checkpoint_dir=os.path.join(project_root, "checkpoints"),
            lm_model_path=None, # uses default
            backend="pt" # forcing PT for stability on Windows for now
        )
        logger.info(f"LLM Init: {lm_msg}")
        if not lm_success:
             logger.warning(f"LLM initialization failed: {lm_msg}. Proceeding without LLM constrained decoding/thinking.")
        
        return d_handler, l_handler

    try:
        dit_handler, llm_handler = await asyncio.to_thread(_init_sync)
        models_initialized = True
        logger.info("Models initialized successfully.")
        logger.info("\n" + "="*40 + "\n🚀  Server is ready at: http://localhost:8000\n" + "="*40)
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        # We don't stop the app, but generation will fail if tried.

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start initialization task
    state.initialization_task = asyncio.create_task(initialize_models())
    yield
    # Cleanup if needed
    if state.initialization_task:
        state.initialization_task.cancel()
    if state.generation_task:
        state.generation_task.cancel()

app = FastAPI(title="Ace 1.5 Vibe Radio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="web"), name="static")

async def compose_track(prompt: str) -> Track:
    if not models_initialized or not dit_handler:
        raise RuntimeError("Models are not initialized yet.")

    track_id = str(uuid.uuid4())
    # Configure generation
    # We'll use the prompt as caption
    # thinking=True allows LLM to expand on the prompt and generat metadata
    params = GenerationParams(
        task_type="text2music",
        caption=prompt,
        thinking=True,
        duration=30.0, # 30 seconds default
        inference_steps=8, # Turbo model is fast
        guidance_scale=7.0,
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
        raise RuntimeError(f"Generation failed: {result.error or result.status_message}")

    audio_data = result.audios[0]
    audio_path = Path(audio_data["path"])
    params_used = audio_data.get("params", {})
    extra = result.extra_outputs.get("lm_metadata", {})
    
    # Extract metadata
    # If LLM generated metadata, use it
    title = f"{prompt.title()}"
    mood = extra.get("keyscale", "Unknown") 
    bpm = int(extra.get("bpm", 120) or 120)
    lyrics = extra.get("lyrics", "")
    
    if not lyrics and extra.get("caption"):
         lyrics = extra.get("caption") # Use caption as lyrics fallback or description
         
    return Track(
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

async def generator_loop() -> None:
    while True:
        async with state_lock:
            if not state.running:
                return
            needs_track = len(state.tracks) < state.buffer_target
            prompt = state.prompt
        
        if not needs_track or not models_initialized:
            await asyncio.sleep(1.0)
            continue
            
        try:
            track = await compose_track(prompt)
            async with state_lock:
                if not state.running:
                    return
                if len(state.tracks) < state.buffer_target:
                    state.tracks.append(track)
        except Exception as e:
            logger.error(f"Error generating track: {e}")
            await asyncio.sleep(5.0) # Wait a bit before retry on error

@app.get("/")
async def index() -> FileResponse:
    return FileResponse("web/index.html")

@app.post("/api/start", response_model=StatusResponse)
async def start_radio(request: StartRequest) -> StatusResponse:
    if not models_initialized:
        # Check if initialization failed?
        # For now just warn or allow but loop will wait
        pass

    async with state_lock:
        state.prompt = request.prompt.strip()
        state.buffer_target = request.buffer_target
        state.running = True
        state.tracks.clear()
        if state.generation_task and not state.generation_task.done():
            state.generation_task.cancel()
        state.generation_task = asyncio.create_task(generator_loop())
    return await get_status()

@app.post("/api/stop", response_model=StatusResponse)
async def stop_radio() -> StatusResponse:
    async with state_lock:
        state.running = False
        state.prompt = ""
        state.tracks.clear()
        if state.generation_task and not state.generation_task.done():
            state.generation_task.cancel()
    return await get_status()

@app.get("/api/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    async with state_lock:
        return StatusResponse(
            running=state.running,
            prompt=state.prompt,
            buffer_target=state.buffer_target,
            buffered_tracks=len(state.tracks),
            now=datetime.utcnow().isoformat() + "Z",
            is_loading_models=not models_initialized
        )

@app.get("/api/next", response_model=TrackResponse)
async def next_track() -> TrackResponse:
    async with state_lock:
        if not state.running:
            raise HTTPException(status_code=400, detail="Radio is not running")
        if not state.tracks:
            return TrackResponse(status="buffering")
        track = state.tracks.popleft()
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
    audio_path = AUDIO_DIR / f"{track_id}.wav"
    # Search for any extension result if not .wav exact
    if not audio_path.exists():
         # try other extensions
         for ext in [".wav", ".flac", ".mp3"]:
             p = AUDIO_DIR / f"{track_id}{ext}"
             if p.exists():
                 audio_path = p
                 break
    
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Track audio not found")
    return FileResponse(str(audio_path), media_type=f"audio/{audio_path.suffix.strip('.')}")

@app.get("/api/history")
async def history_stub() -> dict:
    return {"items": []}

@app.get("/api/style-samples")
async def style_samples_stub() -> dict:
    return {"enabled": False, "message": "Style sample upload coming soon."}
