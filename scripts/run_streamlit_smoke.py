"""Minimal Streamlit smoke test for CI and Hugging Face-style startup validation."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.error
import urllib.request
import tempfile
from pathlib import Path


def _wait_for_http(url: str, timeout_seconds: int = 20, marker: bytes = b"window.prerenderReady") -> bool:
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    body = response.read(2048)
                    if marker in body:
                        return True
                    last_error = "HTTP 200 received without startup marker in response body"
                    time.sleep(0.5)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:  # pragma: no cover - transport noise
            last_error = str(exc)
            time.sleep(0.5)
    if last_error:
        raise RuntimeError(f"Timed out waiting for {url}. Last error: {last_error}")
    raise RuntimeError(f"Timed out waiting for {url}.")


def _read_startup_log(log_path: Path) -> str:
    if not log_path.exists():
        return "Startup log not available."
    try:
        return log_path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - environment edge case
        return f"Unable to read startup log: {exc}"


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

    process: subprocess.Popen[str] | None = None
    log_path: Path | None = None
    startup_log = "Startup log not captured."
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".streamlit-smoke.log", delete=False) as handle:
            process = subprocess.Popen(
                cmd,
                stdout=handle,
                stderr=handle,
                text=True,
            )
            log_path = Path(handle.name)

        time.sleep(args.timeout_wait)
        ok = _wait_for_http(f"http://127.0.0.1:{args.port}", timeout_seconds=args.timeout)
        if not ok:
            raise RuntimeError("Smoke check did not receive expected streamlit marker.")
    except RuntimeError as exc:
        if log_path is not None:
            startup_log = _read_startup_log(log_path)
        raise RuntimeError(f"Streamlit smoke check failed.\n{startup_log}") from exc
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            finally:
                process = None

        if log_path is not None:
            log_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

