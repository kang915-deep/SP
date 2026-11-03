"""Resume a public training archive across fresh HTTP connections."""
from pathlib import Path
import subprocess
import time


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
