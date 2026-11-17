"""Download/cache CDC files, reject HTML masquerading as XPT, audit schema."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from .config import BASE_URL, MODULE_COLUMNS, RAW, ensure_dirs
except ImportError:
    from config import BASE_URL, MODULE_COLUMNS, RAW, ensure_dirs


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_module(path, module):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run download_data.py or supply --raw-dir with all seven XPT files.")
    with path.open("rb") as stream:
        if not stream.read(80).startswith(b"HEADER RECORD*******LIBRARY HEADER RECORD!!!!!!!"):
            raise ValueError(f"{path} is not a SAS transport file (possibly an HTML error page).")
    try:
        frame = pd.read_sas(path, format="xport", encoding="utf-8")
    except Exception as exc:
        raise ValueError(f"Cannot parse {path}: {exc}") from exc
    if frame.empty:
        raise ValueError(f"Empty NHANES module: {path}")
    # SAS column case differs between modules/readers. Canonicalize explicitly.
    lookup = {str(c).upper(): c for c in frame.columns}
    missing = [c for c in MODULE_COLUMNS[module] if c.upper() not in lookup]
    if missing:
        raise ValueError(f"{path.name}: required columns missing: {missing}")
    frame = frame.rename(columns={lookup[c.upper()]: c for c in MODULE_COLUMNS[module]})
    if frame.SEQN.isna().any() or frame.SEQN.duplicated().any():
        raise ValueError(f"{path.name}: SEQN must be nonmissing and unique before merging.")
    if not (frame.SEQN % 1 == 0).all():
        raise ValueError(f"{path.name}: noninteger participant identifier.")
    return frame


def make_session():
    session = requests.Session()
    session.headers["User-Agent"] = "CorVeritas-Stage2-academic/1.0"
    session.mount("https://", HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1,
                  status_forcelist=[429, 500, 502, 503, 504])))
    return session


def fetch(session, url, path):
    try:
        response = session.get(url, timeout=(15, 90))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(f"Download failed: {url}. Put an official local copy at {path}. Details: {exc}") from exc
    if not response.content:
        raise ValueError(f"Empty download: {url}")
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(response.content)
    return temporary


def download_all(raw_dir=RAW, offline=False):
    ensure_dirs()
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"cycle": "August 2021-August 2023", "audited_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    with make_session() as session:
        for module, columns in MODULE_COLUMNS.items():
            path = raw_dir / f"{module}.xpt"
            if not path.exists() and not offline:
                part = fetch(session, BASE_URL + path.name, path)
                read_module(part, module)
                part.replace(path)
            frame = read_module(path, module)
            document = raw_dir / f"{module}.htm"
            if not document.exists() and not offline:
                part = fetch(session, BASE_URL + document.name, document)
                content = part.read_text(errors="replace")
                if "August 2021" not in content or any(c.upper() not in content.upper() for c in columns):
                    raise ValueError(f"Codebook cycle/schema check failed: {document.name}")
                part.replace(document)
            if document.exists():
                content = document.read_text(errors="replace")
                if "August 2021" not in content or any(c.upper() not in content.upper() for c in columns):
                    raise ValueError(f"Cached codebook cycle/schema check failed: {document}")
            manifest["files"][module] = {"url": BASE_URL + path.name, "sha256": sha256(path),
                "rows": len(frame), "columns": list(frame.columns), "codebook_url": BASE_URL + document.name,
                "codebook_sha256": sha256(document) if document.exists() else None}
            print(f"{path.name}: {len(frame):,} participants, schema checked", flush=True)
    (raw_dir / "download_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=RAW)
    parser.add_argument("--offline", action="store_true", help="Use local XPTs only; codebook snapshots optional (see CODEBOOK.md).")
    args = parser.parse_args()
    download_all(args.raw_dir, args.offline)


if __name__ == "__main__":
    main()
