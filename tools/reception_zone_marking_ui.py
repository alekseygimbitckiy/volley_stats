#!/usr/bin/env python3
"""Serve a UI for marking reception pass-origin scoring zones."""

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
HTML_FILE = TOOLS_DIR / "reception_zone_marking_ui.html"
SOURCE_FRAME = OUTPUT_DIR / "field_layout_source_frame.png"
FIELD_LAYOUT = OUTPUT_DIR / "field_layout.json"
ZONES_JSON = OUTPUT_DIR / "reception_zones.json"
ZONES_ANNOTATED = OUTPUT_DIR / "reception_zones_annotated.png"


class ReceptionZoneMarkingHandler(BaseHTTPRequestHandler):
    server_version = "ReceptionZoneMarkingUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/reception_zone_marking_ui.html"):
            self._send_file(HTML_FILE)
            return
        if parsed.path == "/frame":
            self._send_file(SOURCE_FRAME)
            return
        if parsed.path == "/layout":
            self._send_json(read_json_file(FIELD_LAYOUT))
            return
        if parsed.path == "/zones":
            self._send_json(read_json_file(ZONES_JSON) if ZONES_JSON.exists() else {"zones": []})
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
            self._save_outputs(metadata, annotated_png)
        except Exception as exc:  # noqa: BLE001 - report readable errors to UI.
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json(
            {
                "ok": True,
                "files": {
                    "zones": str(ZONES_JSON),
                    "annotated_frame": str(ZONES_ANNOTATED),
                },
            }
        )

    def log_message(self, fmt: str, *args: object) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def _send_file(self, path: Path) -> None:
        if not path.exists():
            self.send_error(404, f"File not found: {path}")
            return
        content = path.read_bytes()
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", f"{mime}; charset=utf-8" if mime.startswith("text/") else mime)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, data: dict, status: int = 200) -> None:
        content = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _save_outputs(self, metadata: dict, annotated_png: str) -> None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metadata = dict(metadata)
        metadata["saved_files"] = {
            "zones": str(ZONES_JSON),
            "annotated_frame": str(ZONES_ANNOTATED),
        }
        ZONES_JSON.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        write_data_url_png(ZONES_ANNOTATED, annotated_png)


def read_json_file(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_data_url_png(path: Path, data_url: str) -> None:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError(f"Expected PNG data URL for {path.name}")
    path.write_bytes(base64.b64decode(data_url[len(prefix) :]))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the reception scoring zone marking UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    if not HTML_FILE.exists():
        raise SystemExit(f"Missing UI file: {HTML_FILE}")
    if not SOURCE_FRAME.exists():
        raise SystemExit(f"Missing source frame. Run field marking first: {SOURCE_FRAME}")

    server = ThreadingHTTPServer((args.host, args.port), ReceptionZoneMarkingHandler)
    print(f"Reception zone marking UI: http://{args.host}:{args.port}/")
    print(f"Outputs will be saved in: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping reception zone marking UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
