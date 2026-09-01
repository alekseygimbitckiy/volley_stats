from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from resize_yolo_dataset import convert_label_text  # noqa: E402


class ResizeYoloDatasetTests(unittest.TestCase):
    def test_square_image_label_accounts_for_horizontal_letterbox(self) -> None:
        converted = convert_label_text(
            "3 0.5 0.5 0.25 0.5\n",
            source_width=640,
            source_height=640,
            target_width=960,
            target_height=640,
            scale=1.0,
            pad_x=160,
            pad_y=0,
        ).split()
        self.assertEqual(converted[0], "3")
        self.assertAlmostEqual(float(converted[1]), 0.5)
        self.assertAlmostEqual(float(converted[2]), 0.5)
        self.assertAlmostEqual(float(converted[3]), 1.0 / 6.0, places=9)
        self.assertAlmostEqual(float(converted[4]), 0.5)

    def test_wide_image_label_accounts_for_vertical_letterbox(self) -> None:
        converted = convert_label_text(
            "4 0.25 0.5 0.1 0.2\n",
            source_width=640,
            source_height=416,
            target_width=960,
            target_height=640,
            scale=1.5,
            pad_x=0,
            pad_y=8,
        ).split()
        self.assertAlmostEqual(float(converted[1]), 0.25)
        self.assertAlmostEqual(float(converted[2]), 0.5)
        self.assertAlmostEqual(float(converted[3]), 0.1)
        self.assertAlmostEqual(float(converted[4]), 0.195)


if __name__ == "__main__":
    unittest.main()
