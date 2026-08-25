from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from temporal_args import resolve_temporal_option, seconds_to_frames  # noqa: E402


class SecondsToFramesTests(unittest.TestCase):
    def test_same_duration_scales_with_fps(self) -> None:
        self.assertEqual(seconds_to_frames(0.2, 30), 6)
        self.assertEqual(seconds_to_frames(0.2, 60), 12)
        self.assertEqual(seconds_to_frames(0.2, 120), 24)

    def test_half_frames_round_up(self) -> None:
        self.assertEqual(seconds_to_frames(0.05, 30), 2)

    def test_minimum_is_applied(self) -> None:
        self.assertEqual(seconds_to_frames(0.0, 120, minimum=1), 1)

    def test_invalid_values_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            seconds_to_frames(-0.1, 30)
        with self.assertRaises(ValueError):
            seconds_to_frames(0.1, 0)


class ResolveTemporalOptionTests(unittest.TestCase):
    def test_seconds_are_the_default_source(self) -> None:
        args = Namespace(gap_sec=0.2, gap_frames=None)
        result = resolve_temporal_option(
            args,
            fps=60,
            target_attr="gap",
            seconds_attr="gap_sec",
            legacy_frames_attr="gap_frames",
            seconds_option="--gap-sec",
            legacy_frames_option="--gap-frames",
        )
        self.assertEqual(args.gap, 12)
        self.assertEqual(result["source"], "--gap-sec")
        self.assertAlmostEqual(result["effective_seconds"], 0.2)

    def test_legacy_frames_override_seconds_and_warn(self) -> None:
        args = Namespace(gap_sec=0.2, gap_frames=5)
        stderr = StringIO()
        with redirect_stderr(stderr):
            result = resolve_temporal_option(
                args,
                fps=30,
                target_attr="gap",
                seconds_attr="gap_sec",
                legacy_frames_attr="gap_frames",
                seconds_option="--gap-sec",
                legacy_frames_option="--gap-frames",
            )
        self.assertEqual(args.gap, 5)
        self.assertEqual(result["source"], "--gap-frames")
        self.assertIn("deprecated", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
