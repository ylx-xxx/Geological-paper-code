import unittest
import numpy as np
from reproducibility.sen12_protocol import (
    choose_time,
    rectangle_gap,
    select_support,
    assert_training_regions,
    bootstrap_blocks,
)
from reproducibility.core import normalize, normalize_channels


class ExternalProtocolTests(unittest.TestCase):
    def test_clear_post_anchor_selection_and_tie(self):
        dates = [
            "2019-03-14T00:00:00",
            "2019-03-15T00:00:00",
            "2019-03-20T00:00:00",
            "2019-03-25T00:00:00",
        ]
        scl = np.full((4, 4, 4), 4)
        self.assertEqual(choose_time(dates, scl, "2019-03-15"), (2, None))
        scl[2] = 255
        self.assertEqual(choose_time(dates, scl, "2019-03-15"), (3, None))
        scl[3] = 8
        self.assertEqual(
            choose_time(dates, scl, "2019-03-15"), (None, "insufficient_clear_pixels")
        )

    def test_post_anchor_window_is_bounded(self):
        self.assertEqual(
            choose_time(["2020-01-01T00:00:00"], np.ones((1, 4, 4)) * 4, "2019-03-15")[
                1
            ],
            "no_frame_in_post_anchor_window",
        )

    def test_external_normalization_matches_shared_band_math(self):
        x = np.random.default_rng(7).normal(size=(14, 128, 128)).astype(np.float32)
        np.testing.assert_array_equal(
            normalize(x)[[3, 2, 1]], normalize_channels(x[[3, 2, 1]])
        )

    def test_holdout_region_cannot_enter_training(self):
        assert_training_regions([{"region": "chimanimani"}])
        with self.assertRaises(ValueError):
            assert_training_regions([{"region": "dominicamaria"}])

    def test_label_blind_support_selection_has_buffer(self):
        rows = [
            {
                "sample": f"s{i:03d}",
                "region": "chimanimani",
                "excluded_reason": None,
                "bbox": [i * 1280, 0, (i + 1) * 1280, 1280],
                "positive_pixels": i + 1,
            }
            for i in range(120)
        ]
        train, val, summary = select_support(rows)
        self.assertEqual((len(train), len(val)), (40, 10))
        self.assertGreaterEqual(summary["minimum_train_val_footprint_gap_m"], 1280)
        for r in rows:
            r["positive_pixels"] = 200 - r["positive_pixels"]
        other_train, other_val, _ = select_support(rows)
        self.assertEqual(
            [r["sample"] for r in train], [r["sample"] for r in other_train]
        )
        self.assertEqual([r["sample"] for r in val], [r["sample"] for r in other_val])

    def test_block_bootstrap_refuses_single_cluster(self):
        rows = [
            {"block_id": "same", "TN": 100, "FP": 3, "FN": 2, "TP": 5}
            for _ in range(20)
        ]
        self.assertIsNone(bootstrap_blocks(rows)["interval"])

    def test_footprint_gap_uses_edges(self):
        self.assertEqual(rectangle_gap([0, 0, 1280, 1280], [1280, 0, 2560, 1280]), 0)
        self.assertEqual(rectangle_gap([0, 0, 1280, 1280], [2560, 0, 3840, 1280]), 1280)


if __name__ == "__main__":
    unittest.main()
