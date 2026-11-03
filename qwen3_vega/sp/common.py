import hashlib
import json
import os
from pathlib import Path


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    part = path.with_suffix(path.suffix + ".tmp")
    part.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    os.replace(part, path)
