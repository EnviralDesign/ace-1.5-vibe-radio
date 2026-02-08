import argparse
import subprocess
import sys
from pathlib import Path

import uvicorn


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_DIR = PROJECT_ROOT / "vendor" / "ace-step"
UPSTREAM_URL = "https://github.com/ace-step/ACE-Step-1.5.git"


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


def _main() -> int:
    # Guard for direct execution. Entry points should call the specific mains.
    print("Use `uv run ace-radio` or `uv run ace-radio-upstream`.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
