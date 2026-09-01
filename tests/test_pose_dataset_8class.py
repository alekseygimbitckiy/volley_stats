from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, train_test_split


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labeling"))

from build_pose_dataset_8class import (  # noqa: E402
    normalized_xywh_to_xyxy,
    parse_excluded_classes,
    source_frame_name,
)
from train_pose_svm import (  # noqa: E402
    append_additional_dataset_rows,
    filter_rows_by_excluded_labels,
    filter_rows_by_source_prefix,
    split_indices,
)


class EightClassPoseDatasetTests(unittest.TestCase):
    def test_roboflow_variants_share_one_source_group(self) -> None:
        self.assertEqual(
            source_frame_name("modena_tour_mp4-1526_jpg.rf.123abc"),
            "modena_tour_mp4-1526_jpg",
        )

    def test_normalized_box_is_converted_and_clipped(self) -> None:
        self.assertEqual(normalized_xywh_to_xyxy((0.5, 0.5, 0.2, 0.4), 100, 200), (40.0, 60.0, 60.0, 140.0))
        self.assertEqual(normalized_xywh_to_xyxy((0.0, 0.0, 0.2, 0.2), 100, 100), (0.0, 0.0, 10.0, 10.0))

    def test_defense_can_be_excluded(self) -> None:
        self.assertEqual(parse_excluded_classes(" defense "), {"defense"})

    def test_group_split_never_places_a_group_on_both_sides(self) -> None:
        rows = []
        labels = []
        for label in ("wait", "set"):
            for group_index in range(8):
                for _ in range(2):
                    rows.append({"source_group": f"{label}-{group_index}"})
                    labels.append(label)
        y = np.array(labels)
        args = argparse.Namespace(
            split_mode="group",
            group_column="source_group",
            test_size=0.25,
            random_state=42,
        )
        deps = {
            "np": np,
            "train_test_split": train_test_split,
            "StratifiedGroupKFold": StratifiedGroupKFold,
        }
        train_indices, test_indices, metadata = split_indices(rows, y, args, deps)
        train_groups = {rows[index]["source_group"] for index in train_indices}
        test_groups = {rows[index]["source_group"] for index in test_indices}
        self.assertFalse(train_groups & test_groups)
        self.assertEqual(metadata["mode"], "group")

    def test_source_prefix_filter_keeps_manual_rows(self) -> None:
        rows = [
            {"source_image": "", "action_label": "wait"},
            {"source_image": "train/images/modena_1.jpg", "action_label": "set"},
            {"source_image": "train/images/-E3-82-B9_2.jpg", "action_label": "spike"},
            {"source_image": "train/images/other_3.jpg", "action_label": "block"},
        ]
        filtered, metadata = filter_rows_by_source_prefix(
            rows,
            prefixes=["-E3-82-B9", "modena"],
            source_image_column="source_image",
        )
        self.assertEqual([row["action_label"] for row in filtered], ["wait", "set", "spike"])
        self.assertEqual(metadata["manual_rows_retained"], 1)
        self.assertEqual(metadata["matched_source_rows"], 2)

    def test_additional_dataset_selector_drops_manual_and_unmatched_rows(self) -> None:
        import tempfile

        path = Path(tempfile.gettempdir()) / "pose_additional_dataset_test.csv"
        path.write_text(
            "sample_id,source_image,action_label\n"
            "manual,,wait\n"
            "modena,train/images/modena_1.jpg,set\n"
            "other,train/images/other_1.jpg,spike\n",
            encoding="utf-8",
        )
        combined, metadata = append_additional_dataset_rows(
            [{"sample_id": "primary", "source_image": "", "action_label": "wait"}],
            dataset_paths=[str(path)],
            prefixes=["modena"],
            source_image_column="source_image",
        )
        self.assertEqual([row["sample_id"] for row in combined], ["primary", "modena"])
        self.assertEqual(metadata[0]["rows_added"], 1)

    def test_excluded_labels_are_removed(self) -> None:
        rows = [
            {"action_label": "wait"},
            {"action_label": "recive_top"},
            {"action_label": "set"},
            {"action_label": "spike"},
        ]
        filtered, metadata = filter_rows_by_excluded_labels(
            rows,
            label_column="action_label",
            excluded_labels=["wait", "recive_top"],
        )
        self.assertEqual([row["action_label"] for row in filtered], ["set", "spike"])
        self.assertEqual(metadata["rows_removed"], 2)


if __name__ == "__main__":
    unittest.main()
