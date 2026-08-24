"""kill_detector内の、動画ファイル無しでテストできる純粋なロジックのテスト。"""

import unittest

import numpy as np

from src.kill_detector import ROIConfig, _white_text_mask


class ROIConfigToPixelsTest(unittest.TestCase):
    def test_converts_ratio_to_pixels_for_1920x1080(self):
        roi = ROIConfig(x_ratio=0.5, y_ratio=0.25, width_ratio=0.25, height_ratio=0.1)
        x, y, w, h = roi.to_pixels(frame_width=1920, frame_height=1080)
        self.assertEqual((x, y, w, h), (960, 270, 480, 108))


class WhiteTextMaskTest(unittest.TestCase):
    def test_white_pixel_is_masked(self):
        # 彩度が低く明度が高い白色はキルフィードの文字色としてマスクされる
        bgr = np.full((10, 10, 3), (255, 255, 255), dtype=np.uint8)
        mask = _white_text_mask(bgr)
        self.assertTrue((mask == 255).all())

    def test_colored_background_is_not_masked(self):
        # 緑ハイライトや赤背景など彩度の高い色は文字マスクから除外される
        bgr = np.full((10, 10, 3), (0, 200, 0), dtype=np.uint8)  # BGR: 緑
        mask = _white_text_mask(bgr)
        self.assertTrue((mask == 0).all())


if __name__ == "__main__":
    unittest.main()
