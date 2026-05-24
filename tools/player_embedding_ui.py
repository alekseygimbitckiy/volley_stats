#!/usr/bin/env python3
"""Serve the player snapshot and embedding UI."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT / "tools"
OUTPUT_DIR = ROOT / "data" / "processed" / "player_embeddings"
HTML_FILE = TOOLS_DIR / "player_embedding_ui.html"


class PlayerEmbeddingHandler(BaseHTTPRequestHandler):
    server_version = "PlayerEmbeddingUI/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/player_embedding_ui.html"):
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
            files = self._save_outputs(payload)
        except Exception as exc:  # noqa: BLE001 - return readable UI errors.
            self._send_json({"ok": False, "error": str(exc)}, status=400)
            return

        self._send_json({"ok": True, "files": files})

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

    def _save_outputs(self, payload: dict) -> dict:
        metadata = payload["metadata"]
        samples = payload["samples"]
        players = payload["players"]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_root = OUTPUT_DIR / "snapshots"
        snapshot_root.mkdir(parents=True, exist_ok=True)

        saved_samples = []
        for index, sample in enumerate(samples, start=1):
            player_id = _safe_name(sample["player_id"])
            player_dir = snapshot_root / player_id
            player_dir.mkdir(parents=True, exist_ok=True)

            file_name = f"{index:04d}_{_safe_name(sample['source_video'])}_{sample['time_ms']}ms.png"
            path = player_dir / file_name
            _write_data_url_png(path, sample["crop_png"])

            sample_metadata = dict(sample)
            sample_metadata.pop("crop_png", None)
            sample_metadata["snapshot_path"] = str(path)
            saved_samples.append(sample_metadata)

        output = {
            "schema_version": 1,
            "metadata": metadata,
            "players": players,
            "samples": saved_samples,
            "embedding": {
                "type": "browser_color_luma_descriptor_v1",
                "distance": "cosine or euclidean after L2 normalization",
                "notes": (
                    "This is a lightweight bootstrap descriptor. Replace with a neural "
                    "person ReID embedding later while keeping player_id/sample metadata."
                ),
            },
            "saved_files": {
                "embeddings": str(OUTPUT_DIR / "player_embeddings.json"),
                "snapshots": str(snapshot_root),
            },
        }

        embeddings_path = OUTPUT_DIR / "player_embeddings.json"
        embeddings_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        return {
            "embeddings": str(embeddings_path),
            "snapshots": str(snapshot_root),
        }


def _write_data_url_png(path: Path, data_url: str) -> None:
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        raise ValueError(f"Expected PNG data URL for {path.name}")
    path.write_bytes(base64.b64decode(data_url[len(prefix) :]))


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned.strip("._") or "unknown"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run the player snapshot and embedding UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8767)
    args = parser.parse_args()

    if not HTML_FILE.exists():
        raise SystemExit(f"Missing UI file: {HTML_FILE}")

    server = ThreadingHTTPServer((args.host, args.port), PlayerEmbeddingHandler)
    print(f"Player embedding UI: http://{args.host}:{args.port}/")
    print(f"Outputs will be saved in: {OUTPUT_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping player embedding UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
