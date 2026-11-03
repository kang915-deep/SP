"""Resume a public training archive across fresh HTTP connections."""
from pathlib import Path
import io
import os
import shutil
import subprocess
import tempfile
import time
from uuid import uuid4


class ArchiveStream(io.RawIOBase):
    """Expose consecutive verified download pieces to hashlib and streaming tarfile."""
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            n = self.current.readinto(buffer)
            if n:
                return n
            self.current.close()
            self.current = None

    def close(self):
        if self.current is not None:
            self.current.close()
        super().close()


def download_pieces(url, part, size, chunk_bytes=128 << 20, fallback_url=None):
    """Keep the existing prefix read-only. Every new write starts at byte zero of a new object."""
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Partial file exceeds expected size: {part}")
    paths = [part] if offset else []
    folder = part.with_name(part.name + ".ranges")
    folder.mkdir(exist_ok=True)
    print(f"Range download: {part.name}, retained {offset} bytes, remaining {size-offset} bytes", flush=True)
    for start in range(offset, size, chunk_bytes):
        end = min(start + chunk_bytes, size) - 1
        length = end - start + 1
        path = folder / f"{start}-{end}"
        paths.append(path)
        if path.is_file() and path.stat().st_size == length:
            continue
        with tempfile.TemporaryDirectory(prefix="vega-range-") as tmp:
            scratch = Path(tmp) / "chunk"
            for attempt in range(1, 9):
                result = subprocess.run(
                    ["curl", "--http1.1", "-fsSL", "--connect-timeout", "20",
                     "--speed-limit", "1", "--speed-time", "90", "--max-filesize", str(length),
                     "--range", f"{start}-{end}", "-o", str(scratch), "-w", "%{http_code}",
                     f"{url}?download=true&range={start}-{end}-{time.time_ns()}"],
                    stdout=subprocess.PIPE, text=True)
                if result.returncode == 0 and result.stdout == "206" and scratch.stat().st_size == length:
                    break
                retry = (result.returncode in (5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92)
                         or result.stdout in ("408", "429", "500", "502", "503", "504"))
                if fallback_url and (retry or result.stdout in ("401","403","404") or result.returncode in (0,63)):
                    print(f"Retrying range {start}-{end} from ModelScope", flush=True)
                    url, fallback_url = fallback_url, None
                    retry = True
                if not retry or attempt == 8:
                    raise RuntimeError(f"Range {start}-{end} failed in {tmp}: curl {result.returncode}, "
                                       f"HTTP {result.stdout}. Original data retained.")
                time.sleep(min(2 ** attempt, 30))
            stage = folder / f"{path.name}.{uuid4().hex}.partial"
            print(f"Writing new range {start}-{end} to {folder}", flush=True)
            with scratch.open("rb") as src, stage.open("xb") as dst:
                shutil.copyfileobj(src, dst, 4 << 20)
            os.replace(stage, path)
    return paths


def download(url, part: Path, size=None):
    for attempt in range(1, 9):
        offset = part.stat().st_size if part.exists() else 0
        if size is not None and offset >= size:
            if offset == size:
                return  # Caller still verifies the complete SHA-256 before publishing.
            raise ValueError(f"Partial file exceeds expected size: {part}")
        print(f"Downloading {part.name} from byte {offset} (attempt {attempt}/8)", flush=True)
        # Reopen curl for each attempt so -C - uses the updated on-disk length.
        # A fresh resolver URL also obtains a new signed CDN redirect for this range.
        request_url = f"{url}?download=true&resume={offset}-{time.time_ns()}"
        result = subprocess.run(
            ["curl", "--http1.1", "-fsSL", "--connect-timeout", "20", "--speed-limit", "1",
             "--speed-time", "90", "-C", "-", "-o", str(part), "-w", "%{http_code}", request_url],
            stdout=subprocess.PIPE, text=True,
        )
        received = part.stat().st_size if part.exists() else 0
        if result.returncode == 0 and (size is None or received == size):
            return
        retry = (result.returncode in (5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92)
                 or result.stdout in ("408", "429", "500", "502", "503", "504")
                 or (result.returncode == 0 and received > offset))
        if not retry or attempt == 8:
            raise RuntimeError(f"Download stopped: {part.name}, HTTP {result.stdout}, "
                               f"curl {result.returncode}, {received} bytes retained. Rerun to resume.")
        time.sleep(min(2 ** attempt, 30))
