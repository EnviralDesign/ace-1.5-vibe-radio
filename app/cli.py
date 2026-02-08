import argparse
import subprocess
import sys
from pathlib import Path

import uvicorn

from acestep.model_downloader import (
    download_main_model,
    ensure_dit_model,
    ensure_lm_model,
    get_checkpoints_dir,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = PROJECT_ROOT / "vendor" / "ace-step"
UPSTREAM_URL = "https://github.com/ace-step/ACE-Step-1.5.git"

# Keep in sync with app/main.py runtime options.
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


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"+ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def serve_main() -> None:
    parser = argparse.ArgumentParser(description="Run ACE 1.5 Vibe Radio API server.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind uvicorn to.")
    parser.add_argument("--port", type=int, default=6109, help="Port to bind uvicorn to.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode (for local dev only).",
    )
    args = parser.parse_args()

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def update_upstream_main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize or update upstream ACE-Step checkout at vendor/ace-step."
    )
    parser.parse_args()

    UPSTREAM_DIR.parent.mkdir(parents=True, exist_ok=True)

    if UPSTREAM_DIR.exists():
        _run(["git", "-C", str(UPSTREAM_DIR), "fetch", "--tags", "--prune"])
        _run(["git", "-C", str(UPSTREAM_DIR), "pull", "--ff-only"])
        print("Upstream ACE-Step updated.")
        return

    # Try submodule init first when metadata exists.
    try:
        _run(["git", "submodule", "update", "--init", "--recursive", "vendor/ace-step"], cwd=PROJECT_ROOT)
    except subprocess.CalledProcessError:
        pass

    if UPSTREAM_DIR.exists():
        print("Upstream ACE-Step initialized via submodule.")
        return

    _run(["git", "clone", UPSTREAM_URL, str(UPSTREAM_DIR)], cwd=PROJECT_ROOT)
    print("Upstream ACE-Step cloned.")


def download_all_models_main() -> None:
    parser = argparse.ArgumentParser(
        description="Download all ACE-Step models used by this project (main + all DiT + all LM)."
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Optional checkpoints directory override (defaults to ./checkpoints).",
    )
    parser.add_argument(
        "--prefer-source",
        choices=["huggingface", "modelscope"],
        default=None,
        help="Preferred download source (auto-detect if omitted).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model folders already exist.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional HuggingFace token (or set HF_TOKEN).",
    )
    args = parser.parse_args()

    checkpoints_dir = get_checkpoints_dir(args.dir) if args.dir else get_checkpoints_dir()
    token = args.token or None
    prefer_source = args.prefer_source or None

    print(f"Checkpoints directory: {checkpoints_dir}")
    print("Downloading main package (includes vae, text encoder, turbo DiT, LM 1.7B)...")
    ok, msg = download_main_model(
        checkpoints_dir=checkpoints_dir,
        force=args.force,
        token=token,
        prefer_source=prefer_source,
    )
    print(msg)
    failures: list[str] = []
    if not ok:
        failures.append(f"main: {msg}")

    print("\nEnsuring all DiT variants...")
    for model_name in DIT_MODEL_OPTIONS:
        ok, msg = ensure_dit_model(
            model_name=model_name,
            checkpoints_dir=checkpoints_dir,
            token=token,
            prefer_source=prefer_source,
        )
        print(f"- {model_name}: {msg}")
        if not ok:
            failures.append(f"{model_name}: {msg}")

    print("\nEnsuring all LM variants...")
    for model_name in LM_MODEL_OPTIONS:
        ok, msg = ensure_lm_model(
            model_name=model_name,
            checkpoints_dir=checkpoints_dir,
            token=token,
            prefer_source=prefer_source,
        )
        print(f"- {model_name}: {msg}")
        if not ok:
            failures.append(f"{model_name}: {msg}")

    if failures:
        print("\nCompleted with failures:")
        for item in failures:
            print(f"  - {item}")
        raise SystemExit(1)

    print("\nAll requested models are available.")


def _main() -> int:
    # Guard for direct execution. Entry points should call the specific mains.
    print("Use `uv run ace-radio`, `uv run ace-radio-upstream`, or `uv run ace-radio-download-all-models`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
