"""キル区間からハイライトクリップの切り出し範囲を決定するモジュール。

`kill_detector.detect_own_kill_windows` が返す「キルフィードに自分のキルが
表示されている区間」は、キルの瞬間そのものではなく表示され続けている期間なので、
前後にバッファを付けてハイライトの切り出し範囲とする。
"""

from typing import List, Optional, Tuple

import cv2


def get_video_duration_sec(video_path: str) -> float:
    """動画の長さ（秒）を取得する。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {video_path}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        return frame_count / fps
    finally:
        cap.release()


def compute_highlight_ranges(
    kill_windows: List[Tuple[float, float]],
    buffer_sec: float = 1.5,
    video_duration: Optional[float] = None,
) -> List[Tuple[float, float]]:
    """キル区間の前後にバッファを付け、ハイライトの切り出し範囲を決定する。

    バッファを付けた結果、隣接する区間同士が重なった場合は1つの範囲に統合する。
    """
    buffered_ranges = []
    for start, end in kill_windows:
        clip_start = max(0.0, start - buffer_sec)
        clip_end = end + buffer_sec
        if video_duration is not None:
            clip_end = min(video_duration, clip_end)
        buffered_ranges.append((clip_start, clip_end))

    merged_ranges: List[Tuple[float, float]] = []
    for start, end in buffered_ranges:
        if merged_ranges and start <= merged_ranges[-1][1]:
            merged_ranges[-1] = (merged_ranges[-1][0], max(merged_ranges[-1][1], end))
        else:
            merged_ranges.append((start, end))

    return merged_ranges
