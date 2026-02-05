import asyncio
import math
import os
import uuid
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from collections import deque

AUDIO_DIR = Path(os.environ.get("ACE_RADIO_AUDIO_DIR", "./tmp"))
AUDIO_DIR.mkdir(parents=True, exist_ok=True)


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


class TrackResponse(BaseModel):
    status: str
    track: Optional[dict] = None


app = FastAPI(title="Ace 1.5 Vibe Radio")
app.mount("/static", StaticFiles(directory="web"), name="static")


def generate_dummy_audio(track_id: str, duration_seconds: float) -> Path:
    sample_rate = 44100
    frequency = 220.0 + (hash(track_id) % 220)
    samples = int(sample_rate * duration_seconds)
    audio_path = AUDIO_DIR / f"{track_id}.wav"

    with wave.open(str(audio_path), "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(samples):
            value = int(32767 * 0.2 * math.sin(2 * math.pi * frequency * (i / sample_rate)))
            wav_file.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
    return audio_path


def compose_track(prompt: str) -> Track:
    track_id = str(uuid.uuid4())
    duration_seconds = 12.0
    mood = "Dreamwave" if "dream" in prompt.lower() else "Night Drive"
    bpm = 96 + (hash(track_id) % 24)
    title = f"{prompt.title()} — {mood}"
    lyrics = (
        f"{prompt.title()} lights are humming tonight\n"
        "Shadows dance in neon lines\n"
        "Stay with the pulse, stay in time\n"
        "Let the bassline carry us home"
    )
    audio_path = generate_dummy_audio(track_id, duration_seconds)
    return Track(
        track_id=track_id,
        title=title,
        prompt=prompt,
        lyrics=lyrics,
        duration_seconds=duration_seconds,
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
        if not needs_track:
            await asyncio.sleep(0.5)
            continue
        await asyncio.sleep(1.5)
        track = compose_track(prompt)
        async with state_lock:
            if not state.running:
                return
            if len(state.tracks) < state.buffer_target:
                state.tracks.append(track)


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("web/index.html")


@app.post("/api/start", response_model=StatusResponse)
async def start_radio(request: StartRequest) -> StatusResponse:
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
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Track audio not found")
    return FileResponse(str(audio_path), media_type="audio/wav")


@app.get("/api/history")
async def history_stub() -> dict:
    return {"items": []}


@app.get("/api/style-samples")
async def style_samples_stub() -> dict:
    return {"enabled": False, "message": "Style sample upload coming soon."}
