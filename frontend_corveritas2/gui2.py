#!/usr/bin/env python3
"""Start the local CorVeritas API and React development server.

Install the dependencies described in README.md first. This launcher does not
install software, train models, open the browser, or save patient data.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


APP_ROOT = Path(__file__).resolve().parent


def main() -> int:
    environment_python = APP_ROOT / ".venv" / "bin" / "python"
    python = str(environment_python) if environment_python.is_file() else sys.executable
    npm = shutil.which("npm")
    if not npm:
        print("Node.js/npm was not found. Install Node.js 22, then run npm ci in frontend/.", file=sys.stderr)
        return 1
    if not (APP_ROOT / "frontend" / "node_modules").is_dir():
        print("Frontend dependencies are missing. Run: cd frontend_corveritas2/frontend && npm ci", file=sys.stderr)
        return 1
    environment = os.environ.copy()
    cache = APP_ROOT / ".cache"
    environment["KERAS_HOME"] = str(cache / "keras")
    environment["XDG_CACHE_HOME"] = str(cache)
    environment["MPLCONFIGDIR"] = str(cache / "matplotlib")
    environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    check = subprocess.run(
        [python, "-c", "import fastapi, uvicorn, pandas, numpy, sklearn, joblib, tensorflow"],
        cwd=APP_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if check.returncode:
        print("Backend dependencies could not be imported. Follow the virtual environment instructions in README.md.", file=sys.stderr)
        print(check.stderr.strip(), file=sys.stderr)
        return 1

    children: list[subprocess.Popen] = []
    interrupted = False

    def stop(signum=None, frame=None):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    commands = [
        ([python, "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "127.0.0.1", "--port", "8000"], APP_ROOT),
        ([npm, "run", "dev", "--", "--host", "127.0.0.1"], APP_ROOT / "frontend"),
    ]
    print("Starting CorVeritas. Frontend: http://127.0.0.1:5173 | API docs: http://127.0.0.1:8000/docs", flush=True)
    print("Press Ctrl+C once to stop both processes. Model loading may take several seconds.", flush=True)
    try:
        for command, directory in commands:
            children.append(subprocess.Popen(command, cwd=directory, start_new_session=True, env=environment))
        while not interrupted:
            failed = next((child for child in children if child.poll() is not None), None)
            if failed is not None:
                print(f"A development server exited with code {failed.returncode}. Stopping both servers.", file=sys.stderr)
                return failed.returncode or 1
            time.sleep(0.25)
    finally:
        for child in children:
            if child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
