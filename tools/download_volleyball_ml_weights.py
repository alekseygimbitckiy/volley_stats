#!/usr/bin/env python3
"""Download masouduut94/volleyball-ml-models weights into the cloned repo."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DRIVE_ID = "1__zkTmGwZo2z0EgbJvC14I_3kOpgQx3o"
EXPECTED = {
    "action": "action/weights/best.pt",
    "court": "court/weights/best.pt",
    "ball": "ball/weights/best.pt",
    "game_state": "game_state",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download volleyball-ml-models weights from Google Drive.")
    parser.add_argument(
        "--weights-dir",
        default="external/volleyball-ml-models/weights",
        help="Directory where the ZIP should be extracted.",
    )
    parser.add_argument("--force", action="store_true", help="Download even if expected weights already exist.")
    args = parser.parse_args()

    gdown = import_gdown()
    weights_dir = resolve_path(Path(args.weights_dir))
    weights_dir.mkdir(parents=True, exist_ok=True)

    existing = check_existing(weights_dir)
    if all(existing.values()) and not args.force:
        print(f"All expected weights already exist in {weights_dir}")
        return 0

    zip_path = weights_dir / "all_weights.zip"
    print(f"Downloading volleyball-ml-models weights to {zip_path}")
    gdown.download(id=DRIVE_ID, output=str(zip_path), quiet=False)
    if not zip_path.exists():
        raise SystemExit(f"Download failed: {zip_path} was not created")

    print(f"Extracting to {weights_dir}")
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(weights_dir)
    zip_path.unlink()

    existing = check_existing(weights_dir)
    for name, is_present in existing.items():
        status = "ok" if is_present else "missing"
        print(f"{name}: {status}")

    if not existing.get("action") or not existing.get("court"):
        raise SystemExit("Action or court weights are still missing after extraction.")
    return 0


def import_gdown():
    try:
        import gdown  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: gdown\nInstall with: ./venv/bin/python -m pip install gdown") from exc
    return gdown


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def check_existing(weights_dir: Path) -> dict[str, bool]:
    return {name: (weights_dir / relative).exists() for name, relative in EXPECTED.items()}


if __name__ == "__main__":
    raise SystemExit(main())
