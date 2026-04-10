"""Minimal Streamlit smoke test for CI and Hugging Face-style startup validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def _wait_for_http(url: str, timeout_seconds: int = 20) -> bool:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    body = response.read(2048)
                    return b"Syndrome-Net" in body or len(body) > 0
        except (urllib.error.URLError, TimeoutError) as exc:  # pragma: no cover - transport noise
            last_error = str(exc)
            time.sleep(0.5)
    if last_error:
        raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_error}")
    raise RuntimeError(f"Timed out waiting for {url}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a smoke boot check for the Streamlit app.")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--timeout", type=int, default=25)
    parser.add_argument("--timeout-wait", type=float, default=0.1)
    args = parser.parse_args()

    app_path = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
    ]

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(args.timeout_wait)
        ok = _wait_for_http(f"http://127.0.0.1:{args.port}", timeout_seconds=args.timeout)
        if not ok:
            raise RuntimeError("Smoke check did not receive valid streamlit response.")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()

