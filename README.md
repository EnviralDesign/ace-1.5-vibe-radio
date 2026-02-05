# Ace 1.5 Vibe Radio (MVP)

A single-page FastAPI + vanilla JS prototype that buffers AI-generated tracks ahead of playback. The backend currently generates short placeholder audio clips so the streaming/buffering UX can be tested without the ACE model wired in yet.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Notes

- Buffer target is configurable from the UI (1-6 tracks).
- `/api/history` and `/api/style-samples` are stubbed for future prompt history and audio-style upload support.
