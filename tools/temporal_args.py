"""Helpers for FPS-independent command-line timing options."""

from __future__ import annotations

import math
import sys
from argparse import Namespace
from typing import Any


def seconds_to_frames(seconds: float, fps: float, minimum: int = 0) -> int:
    """Convert elapsed seconds to the nearest frame count.

    ``round`` is not used because Python rounds exact halves to the nearest even
    integer. Timing thresholds are easier to reason about with conventional
    half-up rounding.
    """

    seconds = float(seconds)
    fps = float(fps)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError(f"seconds must be a finite non-negative value, got {seconds!r}")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"fps must be a finite positive value, got {fps!r}")
    frames = int(math.floor(seconds * fps + 0.5))
    return max(int(minimum), frames)


def resolve_temporal_option(
    args: Namespace,
    *,
    fps: float,
    target_attr: str,
    seconds_attr: str,
    legacy_frames_attr: str,
    seconds_option: str,
    legacy_frames_option: str,
    minimum_frames: int = 0,
) -> dict[str, Any]:
    """Resolve a seconds-first option and an optional legacy frame override.

    The resolved integer is stored on ``args`` under ``target_attr`` so the
    frame-indexed processing code can remain simple. Legacy frame options take
    precedence when explicitly supplied.
    """

    legacy_frames = getattr(args, legacy_frames_attr, None)
    if legacy_frames is not None:
        frames = int(legacy_frames)
        if frames < minimum_frames:
            raise ValueError(
                f"{legacy_frames_option} must be at least {minimum_frames}, got {frames}"
            )
        requested_seconds = frames / float(fps)
        source = legacy_frames_option
        print(
            f"Warning: {legacy_frames_option} is deprecated; use {seconds_option}. "
            f"At {fps:.3f} FPS, {frames} frames is {requested_seconds:.6f} seconds.",
            file=sys.stderr,
        )
    else:
        requested_seconds = float(getattr(args, seconds_attr))
        frames = seconds_to_frames(requested_seconds, fps, minimum=minimum_frames)
        source = seconds_option

    setattr(args, target_attr, frames)
    return {
        "requested_seconds": requested_seconds,
        "frames": frames,
        "effective_seconds": frames / float(fps),
        "source": source,
    }
