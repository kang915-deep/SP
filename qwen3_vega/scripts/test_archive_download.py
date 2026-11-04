"""Exercise real curl against a local server that drops the first connection."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import hashlib
import io
import random
import tarfile
import tempfile
import threading
import unittest

from archive_download import ArchiveStream, download, download_pieces


class ArchiveDownloadTest(unittest.TestCase):
    def test_read_only_prefix_and_reusable_ranges(self):
        member = random.Random(42).randbytes(72000)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            info = tarfile.TarInfo("video.mp4")
            info.size = len(member)
            tar.addfile(info, io.BytesIO(member))
        archive = buf.getvalue()
        requests = []
        routes = []

        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                first, last = map(int, self.headers["Range"][6:].split("-"))
                requests.append((first, last))
                routes.append(self.path.split("?")[0])
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(archive)}")
                self.send_header("Content-Length", str(last-first+1))
                self.end_headers()
                stop = first + 200 if len(requests) == 1 else last + 1
                self.wfile.write(archive[first:stop])

        with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server, tempfile.TemporaryDirectory() as folder:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            part = Path(folder) / "video.tar.gz.part"
            part.write_bytes(archive[:7])
            part.chmod(0o444)
            url = f"http://127.0.0.1:{server.server_port}/file"
            try:
                paths = download_pieces(url, part, len(archive), 16384, fallback_url=url+"-mirror", workers=1)
                self.assertEqual(requests[0], requests[1])
                self.assertEqual(routes, ["/file"] + ["/file-mirror"] * (len(routes)-1))
                count = len(requests)
                self.assertEqual(download_pieces(url, part, len(archive), 16384), paths)
                self.assertEqual(count, len(requests))
                self.assertEqual(part.read_bytes(), archive[:7])
                with ArchiveStream(paths) as stream:
                    self.assertEqual(hashlib.file_digest(stream, "sha256").digest(), hashlib.sha256(archive).digest())
                with ArchiveStream(paths) as stream, tarfile.open(fileobj=stream, mode="r|gz") as tar:
                    self.assertEqual(tar.extractfile(tar.next()).read(), member)
            finally:
                server.shutdown()

    def test_parallel_ranges_keep_archive_order(self):
        import time
        payload = random.Random(19).randbytes(65536)
        active = 0
        peak = 0
        lock = threading.Lock()

        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal active, peak
                first, last = map(int, self.headers["Range"][6:].split("-"))
                with lock:
                    active += 1
                    peak = max(peak, active)
                time.sleep(.15 if first == 0 else .03)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
                self.send_header("Content-Length", str(last-first+1))
                self.end_headers()
                self.wfile.write(payload[first:last+1])
                with lock:
                    active -= 1

        with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server, tempfile.TemporaryDirectory() as folder:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            part = Path(folder) / "archive.part"
            url = f"http://127.0.0.1:{server.server_port}/archive"
            try:
                paths = download_pieces(url, part, len(payload), 16384, workers=4)
                self.assertGreaterEqual(peak, 2)
                with ArchiveStream(paths) as stream:
                    self.assertEqual(hashlib.file_digest(stream,"sha256").digest(), hashlib.sha256(payload).digest())
                self.assertEqual(paths, sorted(paths, key=lambda p: int(p.name.split('-')[0])))
            finally:
                server.shutdown()

    def test_existing_partial_and_interrupted_resume(self):
        payload = bytes(range(256)) * 8192
        offsets = []

        class Server(BaseHTTPRequestHandler):
            def do_GET(self):
                offset = int(self.headers["Range"].split("=")[1].rstrip("-"))
                offsets.append(offset)
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {offset}-{len(payload)-1}/{len(payload)}")
                self.send_header("Content-Length", str(len(payload) - offset))
                self.end_headers()
                end = 524288 if len(offsets) == 1 else len(payload)
                self.wfile.write(payload[offset:end])

        with ThreadingHTTPServer(("127.0.0.1", 0), Server) as server, tempfile.TemporaryDirectory() as folder:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            part = Path(folder) / "archive.part"
            part.write_bytes(payload[:65536])
            url = f"http://127.0.0.1:{server.server_port}/archive"
            try:
                download(url, part, len(payload))
                download(url, part, len(payload))  # Complete partial awaits the caller's hash check.
                self.assertEqual(offsets, [65536, 524288])
                self.assertEqual(part.read_bytes(), payload)
            finally:
                server.shutdown()


if __name__ == "__main__":
    unittest.main()
