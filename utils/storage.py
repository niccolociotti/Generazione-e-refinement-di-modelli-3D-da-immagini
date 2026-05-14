import os
import uuid
import base64
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = BASE_DIR / 'runs'


def ensure_dirs():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def make_session_dir(session_id=None):
    ensure_dirs()
    if not session_id:
        session_id = str(uuid.uuid4())
    session_dir = RUNS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    return str(session_dir)


def save_base64_image(data, output_dir, filename):
    if ',' in data:
        data = data.split(',', 1)[1]
    raw = base64.b64decode(data)
    path = os.path.join(output_dir, filename)
    with open(path, 'wb') as f:
        f.write(raw)
    return path