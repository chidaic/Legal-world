"""Launch the LEGALWORLD backend for local development.

Usage:
    python start.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent
BACKEND_DIR = ROOT / "backend"
ROOT_ENV_PATH = ROOT / ".env"
DEFAULT_BACKEND_HOST = "127.0.0.1"
DEFAULT_BACKEND_PORT = "8000"

process: subprocess.Popen | None = None


def read_root_env() -> dict[str, str]:
    if not ROOT_ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in ROOT_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            values[key] = value.strip().strip('"').strip("'")
    return values


def build_backend_env() -> dict[str, str]:
    env = {**read_root_env(), **os.environ}
    env["PYTHONUNBUFFERED"] = "1"
    return env


def cleanup(*_: object) -> None:
    global process
    if process and process.poll() is None:
        process.terminate()
    sys.exit(0)


def main() -> None:
    global process

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, cleanup)

    env = build_backend_env()
    host = env.get("BACKEND_HOST", DEFAULT_BACKEND_HOST)
    port = env.get("BACKEND_PORT", DEFAULT_BACKEND_PORT)

    print("=" * 56)
    print("  LEGALWORLD Core Backend")
    print("=" * 56)
    print("Starting FastAPI backend...")

    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "ws_server:app",
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(BACKEND_DIR),
        env=env,
    )

    print()
    print("=" * 56)
    print(f"  Backend API: http://{host}:{port}")
    print(f"  Status:      http://{host}:{port}/api/status")
    print(f"  WebSocket:   ws://{host}:{port}/ws")
    print("=" * 56)
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            ret = process.poll()
            if ret is not None:
                print(f"\nBackend exited with code {ret}.")
                sys.exit(ret)
            time.sleep(0.5)
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    main()
