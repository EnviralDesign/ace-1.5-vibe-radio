# ACE 1.5 Vibe Radio

A generative music radio station powered by ACE-Step 1.5. This application generates infinite, non-repeating music streams based on text prompts using a locally running DiT (Diffusion Transformer) and 5Hz Language Model.

## Features

- **Text-to-Music**: Generate high-fidelity audio from text descriptions.
- **Infinite Stream**: Continuously queues new tracks to keep the radio playing.
- **ACE-Step 1.5 Integration**: Uses the latest open-weights music generation models.
- **Metadata Generation**: Automatically hallucinates track titles, lyrics, and moods using a 1.7B LM.

## Prerequisites

- **NVIDIA GPU**: Approximately 12GB+ VRAM recommended for the default models.
- **Python**: 3.11+
- **uv**: Project and dependency management.

## Installation

1.  Clone the repository.
2.  Install dependencies:

    ```bash
    uv sync
    ```

## Usage

Start the server:

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 6109
```

*Note: The first run will automatically download the necessary model weights (approx. 10GB+), which may take some time.*

## Configuration

- **Audio Output**: Generated files are stored in `./tmp` by default. You can change this by setting the `ACE_RADIO_AUDIO_DIR` environment variable.
- **Model Storage**: Models are downloaded to `./checkpoints`.

## License

MIT
