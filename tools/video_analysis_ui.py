#!/usr/bin/env python3
"""Serve a small UI for launching video analysis."""

from __future__ import annotations

import json
import mimetypes
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
HTML_FILE = TOOLS_DIR / "video_analysis_ui.html"
ANALYZER = TOOLS_DIR / "analyze_videos.py"

STATE = {
    "running": False,
    "returncode": None,
    "log": "",
    "command": "",
}


class VideoAnalysisHandler(BaseHTTPRequestHandler):
    server_version = "VideoAnalysisUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/video_analysis_ui.html"):
            self._send_file(HTML_FILE)
            return
        if parsed.path == "/status":
            self._send_json(STATE)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path != "/start":
            self.send_error(404, "Not found")
            return

        if STATE["running"]:
            self._send_json({"ok": False, "error": "Analysis is already running."}, status=409)
            return

        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        command = build_command(payload)

        STATE.update({"running": True, "returncode": None, "log": "", "command": " ".join(command)})
        thread = threading.Thread(target=run_command, args=(command,), daemon=True)
        thread.start()
        self._send_json({"ok": True, "command": STATE["command"]})

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


def build_command(payload: dict) -> list[str]:
    command = [
        sys.executable,
        str(ANALYZER),
        "--videos-dir",
        payload.get("videos_dir") or "data/game",
        "--layout",
        payload.get("layout") or "data/processed/calibrations/field_layout.json",
        "--embeddings",
        payload.get("embeddings") or "data/processed/player_embeddings/player_embeddings.json",
        "--ball-tracks-dir",
        payload.get("ball_tracks_dir") or "data/processed/vball_net_raw",
        "--output-dir",
        payload.get("output_dir") or "data/processed/analysis",
        "--yolo-model",
        payload.get("yolo_model") or "yolov8n.pt",
        "--frame-stride",
        str(int(payload.get("frame_stride") or 5)),
        "--match-threshold",
        str(float(payload.get("match_threshold") or 0.65)),
    ]
    if payload.get("limit"):
        command.extend(["--limit", str(int(payload["limit"]))])
    if payload.get("vball_net_command"):
        command.extend(["--vball-net-command", payload["vball_net_command"]])
    return command


def run_command(command: list[str]) -> None:
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            STATE["log"] += line
        returncode = process.wait()
    except Exception as exc:  # noqa: BLE001
        STATE["log"] += f"\n{exc}\n"
        returncode = 1
    STATE["returncode"] = returncode
    STATE["running"] = False


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the video analysis UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()

    if not HTML_FILE.exists():
        raise SystemExit(f"Missing UI file: {HTML_FILE}")
    if not ANALYZER.exists():
        raise SystemExit(f"Missing analyzer: {ANALYZER}")

    server = ThreadingHTTPServer((args.host, args.port), VideoAnalysisHandler)
    print(f"Video analysis UI: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping video analysis UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
