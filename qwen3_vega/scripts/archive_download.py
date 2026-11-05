"""Resume a public training archive across fresh HTTP connections."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import io
import os
import shutil
import subprocess
import tempfile
import time
import threading
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


def download_pieces(url, part, size, chunk_bytes=128 << 20, fallback_url=None, workers=None):
    """Keep the existing prefix read-only. Every new write starts at byte zero of a new object."""
    workers = int(os.environ.get("VEGA_DOWNLOAD_WORKERS", "4")) if workers is None else workers
    if not 1 <= workers <= 8:
        raise ValueError("VEGA_DOWNLOAD_WORKERS must be between 1 and 8")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        raise ValueError(f"Partial file exceeds expected size: {part}")
    paths = [part] if offset else []
    folder = part.with_name(part.name + ".ranges")
    folder.mkdir(exist_ok=True)
    pending = []
    for start in range(offset, size, chunk_bytes):
        end = min(start + chunk_bytes, size) - 1
        length = end - start + 1
        path = folder / f"{start}-{end}"
        paths.append(path)
        if path.is_file() and path.stat().st_size == length:
            continue
        pending.append((start, end, path))
    retained = size - sum(end-start+1 for start, end, _ in pending)
    print(f"Range download: {part.name}, retained {retained} bytes, remaining {size-retained} bytes, "
          f"workers={workers}, scratch={tempfile.gettempdir()}", flush=True)
    # Keep object-store commits sequential; overlap them with independent network reads.
    writer = threading.Lock()
    source_lock = threading.Lock()
    preferred = [url, fallback_url]
    started = time.monotonic()
    completed = retained

    def fetch_piece(item):
        nonlocal completed
        start, end, path = item
        length = end-start+1
        with source_lock:
            source, alternative = preferred
        with tempfile.TemporaryDirectory(prefix="vega-range-") as tmp:
            scratch = Path(tmp) / "chunk"
            network_started = time.monotonic()
            for attempt in range(1, 9):
                received = scratch.stat().st_size if scratch.exists() else 0
                remaining = length-received
                response = Path(tmp) / f"response-{attempt}"
                result = subprocess.run(
                    ["curl", "--http1.1", "-fsSL", "--connect-timeout", "20",
                     "--speed-limit", "65536", "--speed-time", "60", "--max-time", "600",
                     "--max-filesize", str(remaining),
                     "--range", f"{start+received}-{end}", "-o", str(response), "-w", "%{http_code}",
                     f"{source}?download=true&range={start+received}-{end}-{time.time_ns()}"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                retry = (result.returncode in (5, 6, 7, 16, 18, 28, 35, 52, 55, 56, 92)
                         or result.stdout in ("408", "429", "500", "502", "503", "504"))
                amount = response.stat().st_size if response.exists() else 0
                # Resume a dropped HTTP 206 inside the local scratch file, never append to BOS.
                progress = result.stdout == "206" and 0 < amount <= remaining and (retry or result.returncode == 0)
                if progress:
                    with response.open("rb") as src, scratch.open("ab") as dst:
                        shutil.copyfileobj(src, dst, 4 << 20)
                    received += amount
                if received == length:
                    break
                retry = retry or (progress and result.returncode == 0)
                if alternative and (retry or result.stdout in ("401","403","404") or result.returncode in (0,63)):
                    print(f"Retrying range {start}-{end} from ModelScope", flush=True)
                    with source_lock:
                        preferred[:] = [alternative, None]
                        source, alternative = preferred
                    retry = True
                if not retry or attempt == 8:
                    raise RuntimeError(f"Range {start}-{end} failed in {tmp}: curl {result.returncode}, "
                                       f"HTTP {result.stdout}. Original data retained. {result.stderr.strip()[-500:]}")
                print(f"RETRY {attempt}/8: range {start}-{end}, curl={result.returncode}, "
                      f"HTTP={result.stdout}, retained {received}/{length} bytes locally; "
                      f"continuing at {start+received}", flush=True)
                time.sleep(1 if progress else min(2 ** attempt, 30))
            network_seconds = time.monotonic()-network_started
            with writer:
                stage = folder / f"{path.name}.{uuid4().hex}.partial"
                print(f"Writing new range {start}-{end} to {folder}; "
                      f"network={network_seconds:.1f}s ({length/2**20/network_seconds:.2f} MiB/s)", flush=True)
                write_started = time.monotonic()
                with scratch.open("rb") as src, stage.open("xb") as dst:
                    shutil.copyfileobj(src, dst, 4 << 20)
                os.replace(stage, path)
                write_seconds = time.monotonic()-write_started
                completed += length
                elapsed = time.monotonic()-started
                print(f"Committed {part.name} {completed}/{size} bytes; "
                      f"write={write_seconds:.1f}s ({length/2**20/max(write_seconds,1e-6):.2f} MiB/s); "
                      f"session={(completed-retained)/2**20/elapsed:.2f} MiB/s", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # Consume in archive order even when network requests finish out of order.
        for _ in pool.map(fetch_piece, pending):
            pass
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
