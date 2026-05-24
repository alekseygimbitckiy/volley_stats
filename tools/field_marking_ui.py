#!/usr/bin/env python3
"""Serve the volleyball field marking UI and save calibration outputs."""

from __future__ import annotations

import base64
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
OUTPUT_DIR = ROOT / "data" / "processed" / "calibrations"
HTML_FILE = TOOLS_DIR / "field_marking_ui.html"


class FieldMarkingHandler(BaseHTTPRequestHandler):
    server_version = "FieldMarkingUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/field_marking_ui.html"):
            self._send_file(HTML_FILE)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/save":
            self.send_error(404, "Not found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            metadata = payload["metadata"]
            annotated_png = payload["annotated_png"]
            frame_png = payload.get("frame_png")
            self._save_outputs(metadata, annotated_png, frame_png)
        except Exception as exc:  # noqa: BLE001 - report readable errors to the UI.
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json(
            {
                "ok": True,
                "files": {
                    "layout": str(OUTPUT_DIR / "field_layout.json"),
                    "annotated_frame": str(OUTPUT_DIR / "field_layout_annotated.png"),
                    "source_frame": str(OUTPUT_DIR / "field_layout_source_frame.png"),
                },
            }
        )

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, "File not found")
            return

        content = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict, status: int = 200) -> None:
        content = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _save_outputs(self, metadata: dict, annotated_png: str, frame_png: str | None) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        metadata = dict(metadata)
        metadata["saved_files"] = {
            "layout": str(OUTPUT_DIR / "field_layout.json"),
            "annotated_frame": str(OUTPUT_DIR / "field_layout_annotated.png"),
            "source_frame": str(OUTPUT_DIR / "field_layout_source_frame.png"),
        }

        (OUTPUT_DIR / "field_layout.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _write_data_url_png(OUTPUT_DIR / "field_layout_annotated.png", annotated_png)
        if frame_png:
            _write_data_url_png(OUTPUT_DIR / "field_layout_source_frame.png", frame_png)


def _write_data_url_png(path: Path, data_url: str) -> None:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError(f"Expected PNG data URL for {path.name}")
    path.write_bytes(base64.b64decode(data_url[len(prefix) :]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the volleyball field marking UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if not HTML_FILE.exists():
        raise SystemExit(f"Missing UI file: {HTML_FILE}")

    server = ThreadingHTTPServer((args.host, args.port), FieldMarkingHandler)
    print(f"Field marking UI: http://{args.host}:{args.port}/")
    print(f"Outputs will be saved in: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping field marking UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
