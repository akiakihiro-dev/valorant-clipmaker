"""highlight_detector.compute_highlight_ranges のテスト。"""

import unittest

from src.highlight_detector import compute_highlight_ranges


class ComputeHighlightRangesTest(unittest.TestCase):
    def test_adds_buffer_before_and_after(self):
        result = compute_highlight_ranges([(10.0, 12.0)], buffer_sec=1.5)
        self.assertEqual(result, [(8.5, 13.5)])

    def test_clamps_start_to_zero(self):
        result = compute_highlight_ranges([(0.5, 2.0)], buffer_sec=1.5)
        self.assertEqual(result, [(0.0, 3.5)])

    def test_clamps_end_to_video_duration(self):
        result = compute_highlight_ranges([(58.0, 59.5)], buffer_sec=1.5, video_duration=60.0)
        self.assertEqual(result, [(56.5, 60.0)])

    def test_merges_overlapping_ranges_after_buffering(self):
        # 5.0-6.0と7.0-8.0はバッファ(1.5s)込みで重なるため1区間に統合される
        result = compute_highlight_ranges([(5.0, 6.0), (7.0, 8.0)], buffer_sec=1.5)
        self.assertEqual(result, [(3.5, 9.5)])

    def test_keeps_far_apart_ranges_separate(self):
        result = compute_highlight_ranges([(5.0, 6.0), (30.0, 31.0)], buffer_sec=1.5)
        self.assertEqual(result, [(3.5, 7.5), (28.5, 32.5)])

    def test_empty_input_returns_empty_list(self):
        self.assertEqual(compute_highlight_ranges([]), [])


if __name__ == "__main__":
    unittest.main()
