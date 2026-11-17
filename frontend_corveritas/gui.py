"""Launch the local FastAPI and Vite development servers together."""

import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR / "backend"
FRONTEND_DIR = APP_DIR / "frontend"


def main() -> int:
    npm = shutil.which("npm")
    if npm is None:
        print("npm was not found. Install Node.js, then run `npm install` in frontend/.")
        return 1

    processes = []
    try:
        backend = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--reload",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=BACKEND_DIR,
        )
        processes.append(backend)
        time.sleep(1)
        if backend.poll() is not None:
            return int(backend.returncode or 1)

        frontend = subprocess.Popen(
            [npm, "run", "dev"],
            cwd=FRONTEND_DIR,
        )
        processes.append(frontend)
        print("CorVeritas API: http://127.0.0.1:8000")
        print("CorVeritas UI:  http://127.0.0.1:5173")
        while all(process.poll() is None for process in processes):
            time.sleep(0.25)
        return next(
            int(process.returncode or 0)
            for process in processes
            if process.returncode is not None
        )
    except KeyboardInterrupt:
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        for process in processes:
            if process.poll() is None:
                process.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())