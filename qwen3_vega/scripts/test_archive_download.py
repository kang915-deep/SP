"""Exercise real curl against a local server that drops the first connection."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import tempfile
import threading
import unittest

from archive_download import download


class ArchiveDownloadTest(unittest.TestCase):
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
