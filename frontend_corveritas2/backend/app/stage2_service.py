"""Lazy, isolated Stage 2 runtime. Stage 1 never imports Torch or waits for it."""
import json
import os
from pathlib import Path
import selectors
import subprocess
import threading
import time

from .config import APP_ROOT, REPO_ROOT


class Stage2Unavailable(RuntimeError):
    pass


class Stage2Service:
    def __init__(self, model_dir=None, python=None, timeout=60):
        self.model_dir = Path(model_dir or os.getenv("CORVERITAS_STAGE2_MODEL_DIR", str(REPO_ROOT / "models_stage2" / "models")))
        self.python = Path(python or os.getenv("CORVERITAS_STAGE2_PYTHON", str(REPO_ROOT / "models_stage2" / ".venv" / "bin" / "python")))
        if not self.model_dir.is_absolute():
            self.model_dir = REPO_ROOT / self.model_dir
        if not self.python.is_absolute():
            self.python = REPO_ROOT / self.python
        self.timeout = timeout
        self._process = None
        self._metadata = None
        self._buffer = b""
        self._lock = threading.Lock()

    def _read(self):
        deadline = time.monotonic() + self.timeout
        with selectors.DefaultSelector() as selector:
            selector.register(self._process.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise Stage2Unavailable("Stage 2 timed out. Stage 1 remains available; retry Stage 2 when ready.")
                chunk = os.read(self._process.stdout.fileno(), 65536)
                if not chunk:
                    raise Stage2Unavailable("The Stage 2 worker stopped. Stage 1 remains available.")
                self._buffer += chunk
                if len(self._buffer) > 1024 * 1024:
                    raise Stage2Unavailable("Stage 2 returned an unsupported response.")
        line, self._buffer = self._buffer.split(b"\n", 1)
        response = json.loads(line)
        if not response.get("ok"):
            raise Stage2Unavailable("Stage 2 could not load its artifacts or complete this request. Verify its own environment and model files; Stage 1 remains available.")
        return response["result"]

    def _start(self):
        if self._process is not None and self._process.poll() is None and self._metadata:
            return
        self._stop()
        if not self.python.is_file() or not (self.model_dir / "metadata.json").is_file():
            raise Stage2Unavailable("Stage 2 is unavailable. Its Python environment or trained model files are missing. Stage 1 remains available.")
        environment = os.environ.copy()
        environment.update({"PYTHONDONTWRITEBYTECODE": "1", "MPLBACKEND": "Agg", "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
        environment["MPLCONFIGDIR"] = str(APP_ROOT / ".cache" / "stage2" / "matplotlib")
        environment["XDG_CACHE_HOME"] = str(APP_ROOT / ".cache" / "stage2")
        self._process = subprocess.Popen([str(self.python), "-B", "-u", str(Path(__file__).with_name("stage2_worker.py")), str(self.model_dir)],
                                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         cwd=REPO_ROOT, env=environment)
        self._metadata = self._read()

    def status(self):
        if not self._lock.acquire(timeout=1):
            return {"available": False, "reason": "Stage 2 is busy. Retry when the current classification finishes."}
        try:
            self._start()
            return dict(self._metadata)
        except Exception as error:
            self._stop()
            return {"available": False, "reason": str(error) if isinstance(error, Stage2Unavailable) else "Stage 2 could not start. Verify its environment and artifacts; Stage 1 remains available."}
        finally:
            self._lock.release()

    def predict(self, features, explain=True):
        if not self._lock.acquire(timeout=1):
            raise Stage2Unavailable("Stage 2 is busy. Wait for the current classification before retrying.")
        try:
            self._start()
            data = json.dumps({"features": features, "explain": explain}, allow_nan=False).encode() + b"\n"
            self._process.stdin.write(data)
            self._process.stdin.flush()
            return self._read()
        except Exception as error:
            self._stop()
            if isinstance(error, Stage2Unavailable):
                raise
            raise Stage2Unavailable("Stage 2 classification failed. Stage 1 remains available; retry Stage 2.") from None
        finally:
            self._lock.release()

    def _stop(self):
        process, self._process = self._process, None
        self._metadata, self._buffer = None, b""
        if process is not None:
            if process.poll() is None:
                try:
                    process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def close(self):
        with self._lock:
            self._stop()
