"""動画からのフレーム抽出とキル検出を行うモジュール。

キル検出は、画面中央下部（クロスヘアの少し下）にキル時のみ出現する「キルマーク」
演出を対象とする。キルマークの絵柄はプレイヤーのクロスヘア設定やキルの種類によって
複数パターンが存在するため、特定の絵柄へのテンプレートマッチングではなく、
彩度・明度に基づく「UI要素らしさ」の変化を検出する形状非依存の方式を採る。
"""

from dataclasses import dataclass
from typing import Generator, List, Tuple

import cv2
import numpy as np


@dataclass
class ROIConfig:
    """キルフィードが表示される領域の設定。

    解像度やHUDスケール設定によって表示位置・サイズが変わるため、絶対ピクセルではなく
    フレームサイズに対する比率（0.0〜1.0）で保持する。デフォルト値は
    `clipsample/` のサンプルクリップ（1920x1080）を目視確認して設定したもので、
    キルフィード最大6行分を余裕を持ってカバーする範囲。解像度・HUD設定が異なる
    環境では別途調整が必要。
    """

    x_ratio: float = 0.74
    y_ratio: float = 0.07
    width_ratio: float = 0.26
    height_ratio: float = 0.28

    def to_pixels(self, frame_width: int, frame_height: int) -> Tuple[int, int, int, int]:
        """フレームサイズに対する比率から実ピクセル座標 (x, y, w, h) を計算する。"""
        x = int(self.x_ratio * frame_width)
        y = int(self.y_ratio * frame_height)
        w = int(self.width_ratio * frame_width)
        h = int(self.height_ratio * frame_height)
        return x, y, w, h


def extract_frames(
    video_path: str, sample_fps: float = 5.0
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """動画から一定間隔でフレームをサンプリングする。

    動画全体を一度にメモリへ載せないよう、ジェネレータとして
    (タイムスタンプ[秒], フレーム画像) を順次yieldする。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"動画を開けませんでした: {video_path}")

    try:
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if not video_fps or video_fps <= 0:
            video_fps = sample_fps

        # sample_fpsが動画自体のfpsを超える場合は全フレームを対象にする
        frame_interval = max(1, round(video_fps / sample_fps))

        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % frame_interval == 0:
                timestamp_sec = frame_index / video_fps
                yield timestamp_sec, frame

            frame_index += 1
    finally:
        cap.release()


def crop_roi(frame: np.ndarray, roi: ROIConfig) -> np.ndarray:
    """フレームからROI設定に基づいて領域を切り出す。"""
    frame_height, frame_width = frame.shape[:2]
    x, y, w, h = roi.to_pixels(frame_width, frame_height)
    return frame[y : y + h, x : x + w]


def extract_roi_frames(
    video_path: str, roi: ROIConfig, sample_fps: float = 5.0
) -> Generator[Tuple[float, np.ndarray], None, None]:
    """フレーム抽出とROI切り出しをまとめて行うヘルパー。

    キル検出はこの関数が返す (タイムスタンプ, ROI画像) の列をそのまま利用できる。
    """
    for timestamp_sec, frame in extract_frames(video_path, sample_fps):
        yield timestamp_sec, crop_roi(frame, roi)


# 画面中央下部、クロスヘアの少し下に出現するキルマークのROI。
# `clipsample/` のサンプルクリップ（1920x1080）を目視確認して設定したもので、
# キルマークが展開しきった状態（周囲の円形装飾込み）を余裕を持ってカバーする範囲。
KILL_MARK_ROI = ROIConfig(
    x_ratio=0.458,
    y_ratio=0.72,
    width_ratio=0.09,
    height_ratio=0.093,
)


def compute_kill_mark_score(roi_frame: np.ndarray) -> float:
    """ROI画像に対して「UI要素らしさ」のスコア（0.0〜1.0）を計算する。

    キルマークは彩度・明度が高い暖色（赤・金色等）、または低彩度高明度の白色で
    描かれており、壁や床などの背景テクスチャはこの条件に当てはまりにくい。
    この性質を利用し、キルマークの絵柄そのものに依存せずに出現を検出する。

    マップ内の緑・青系オブジェクト（設置物やアビリティエフェクト等）は彩度・明度が
    高くても誤検出の原因になるため、色相（Hue）を赤〜金・白系に限定して除外する。
    """
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.int32)
    saturation = hsv[:, :, 1].astype(np.int32)
    value = hsv[:, :, 2].astype(np.int32)

    is_warm_hue = (hue <= 35) | (hue >= 160)
    colorful_mask = (saturation > 120) & (value > 120) & is_warm_hue
    white_mask = (saturation < 60) & (value > 200)
    mask = colorful_mask | white_mask
    return float(mask.sum()) / mask.size


def detect_kill_windows(
    video_path: str,
    roi: ROIConfig = KILL_MARK_ROI,
    sample_fps: float = 10.0,
    threshold: float = 0.015,
    min_duration_sec: float = 0.3,
    merge_gap_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """キルマークが出現している区間（開始・終了時刻）を検出する。

    連続キル（マルチキル）の場合はキルマークが表示され続けるため、
    まとめて1つの区間として返る。区間の開始時刻をキル発生時刻の目安として扱う想定。

    マルチキル展開アニメーションの途中でスコアが一瞬閾値を割る、あるいは環境オブジェクト
    による単発ノイズが挟まることがあるため、`merge_gap_sec` 以下の間隔は同一区間として
    結合し、`min_duration_sec` 未満しか続かない区間はノイズとして除外する。
    """
    raw_windows: List[Tuple[float, float]] = []
    window_start = None
    prev_timestamp = None

    for timestamp, roi_frame in extract_roi_frames(video_path, roi, sample_fps):
        score = compute_kill_mark_score(roi_frame)
        is_mark_visible = score > threshold

        if is_mark_visible and window_start is None:
            window_start = timestamp
        elif not is_mark_visible and window_start is not None:
            raw_windows.append((window_start, prev_timestamp))
            window_start = None

        prev_timestamp = timestamp

    if window_start is not None:
        raw_windows.append((window_start, prev_timestamp))

    merged_windows: List[Tuple[float, float]] = []
    for start, end in raw_windows:
        if merged_windows and start - merged_windows[-1][1] <= merge_gap_sec:
            merged_windows[-1] = (merged_windows[-1][0], end)
        else:
            merged_windows.append((start, end))

    return [w for w in merged_windows if (w[1] - w[0]) >= min_duration_sec]
