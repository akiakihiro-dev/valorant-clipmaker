"""kill_detector内の、動画ファイル無しでテストできる純粋なロジックのテスト。"""

import unittest

import numpy as np

from src.kill_detector import ROIConfig, compute_kill_mark_score


class ROIConfigToPixelsTest(unittest.TestCase):
    def test_converts_ratio_to_pixels_for_1920x1080(self):
        roi = ROIConfig(x_ratio=0.5, y_ratio=0.25, width_ratio=0.25, height_ratio=0.1)
        x, y, w, h = roi.to_pixels(frame_width=1920, frame_height=1080)
        self.assertEqual((x, y, w, h), (960, 270, 480, 108))


class ComputeKillMarkScoreTest(unittest.TestCase):
    def test_solid_warm_color_scores_high(self):
        # 明るい赤（暖色・高彩度・高明度）で塗りつぶした画像はスコアが高くなる
        roi_frame = np.full((20, 20, 3), (0, 0, 255), dtype=np.uint8)  # BGR: 赤
        self.assertGreater(compute_kill_mark_score(roi_frame), 0.9)

    def test_solid_green_scores_low(self):
        # 緑（暖色でも白でもない）で塗りつぶした画像はスコアが低くなる
        roi_frame = np.full((20, 20, 3), (0, 255, 0), dtype=np.uint8)  # BGR: 緑
        self.assertLess(compute_kill_mark_score(roi_frame), 0.1)


if __name__ == "__main__":
    unittest.main()
