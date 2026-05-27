#!/usr/bin/env python3
"""Download the SportsReID SoccerNet OSNet checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOCCERNET_OSNET_DRIVE_ID = "1To0Ww6_HxU2ITAlb4kQEgYExV-orwit8"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SoccerNet-trained OSNet ReID weights.")
    parser.add_argument("--output", default="external/soccernet-reid/osnet/model.osnet.pth.tar")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output = resolve_project_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.stat().st_size > 900_000_000 and not args.force:
        print(f"SoccerNet ReID checkpoint already exists: {output}")
        return 0

    try:
        import gdown  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing package: gdown\nInstall with: ./venv/bin/python -m pip install gdown") from exc

    print(f"Downloading SoccerNet ReID checkpoint to {output}")
    gdown.download(id=SOCCERNET_OSNET_DRIVE_ID, output=str(output), quiet=False)
    if not output.exists():
        raise SystemExit(f"Download failed: {output} was not created")
    print(f"Saved: {output}")
    return 0


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


if __name__ == "__main__":
    raise SystemExit(main())
