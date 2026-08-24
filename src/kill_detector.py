"""動画からのフレーム抽出とキル検出を行うモジュール。

キル検出は、キルフィード内で自分のプレイヤー名がキラー側（行の右寄り）に
表示されているかどうかをテンプレートマッチングで判定する方式を採る。
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

    # x_ratio・width_ratioは右端（画面右端に一致）を固定したまま、左に約100px分の
    # 余白を追加している。バトルレポート画面では自分の名前の表示位置がライブ中の
    # キルフィードより左に寄ることがあり、余白が無いと名前の先頭文字がROIの左端で
    # 切れてテンプレート一致度が大きく下がるため。
    x_ratio: float = 0.6879
    y_ratio: float = 0.07
    width_ratio: float = 0.3121
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


# キルフィードの1行あたりの高さ・先頭オフセット（ROI全体に対する比率）。
# サンプルクリップの複数フレームでクロスヘアアイコンのY座標を実測して求めた値。
# 単純に行数で均等割り（1/6、オフセット無し）すると実際の行ピッチ（約1/7.7、
# 先頭に高さ3%分のマージンあり）とのズレが行が進むほど蓄積し、3行目以降で
# 隣接行の内容が混ざってしまうため、実測値を使う。
KILL_FEED_ROW_HEIGHT_RATIO = 39.3 / 302
KILL_FEED_ROW_TOP_OFFSET_RATIO = 9.85 / 302


# 自分のプレイヤー名をキルフィード内で切り出したテンプレート画像。
# サンプルクリップ（1920x1080）のキルフィード1行分から名前部分のみを切り出したもの。
# 利用時は各自の環境で自分のプレイヤー名部分を切り出した画像に差し替える。
#
# バトルレポート画面での名前の切れはROIConfig側の余白拡張で解消済み（一致度
# 0.40→0.91程度）のため、テンプレートはこの1種類のみで足りている。背景色が
# 赤くなる場合があるが、白色文字マスクによる照合のため影響を受けない。
# 行の解像度が低く他プレイヤーの短い名前とも紛らわしいため、パターン別の
# テンプレートを追加で持つより現状の1枚構成の方が誤検出が少ない。
OWN_NAME_TEMPLATE_PATHS = [
    "assets/templates/own_name.png",
]

_own_name_template_masks: list[np.ndarray] | None = None


def _white_text_mask(bgr: np.ndarray) -> np.ndarray:
    """白色の文字らしいピクセルを二値マスクとして抽出する。

    キルフィードの文字は彩度が低く明度が高い白色で描画されている。背景の色
    (緑・赤のハイライトやマップの風景)を無視して文字の形だけを取り出すことで、
    背景色がたまたま似ていても文字の形が違えばマッチしないようにする。
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.int32)
    value = hsv[:, :, 2].astype(np.int32)
    mask = (saturation < 60) & (value > 180)
    return (mask.astype(np.uint8)) * 255


def _get_own_name_template_masks() -> list[np.ndarray]:
    """自分の名前テンプレート（複数パターン）の二値マスクを読み込む（初回のみ）。"""
    global _own_name_template_masks
    if _own_name_template_masks is None:
        masks = []
        for path in OWN_NAME_TEMPLATE_PATHS:
            template_bgr = cv2.imread(path)
            if template_bgr is None:
                raise IOError(f"名前テンプレート画像を読み込めませんでした: {path}")
            masks.append(_white_text_mask(template_bgr))
        _own_name_template_masks = masks
    return _own_name_template_masks


# キル/被害の左右判定基準を、行の右端（画面右端に固定されている）からの
# 距離で定義するための参照値。ROIConfigは左側に余白を追加して拡張されているため
# 行の中央（row_width/2）を基準にすると、余白の分だけ本来の判定境界からずれてしまう。
# 拡張前のROI幅（frame_width比0.26）の半分を、右端からの距離の基準として使う。
KILL_FEED_RIGHT_HALF_WIDTH_RATIO = 0.26 / 2


def _find_own_name_side(row_bgr: np.ndarray, match_threshold: float, frame_width: int) -> str | None:
    """行内で自分の名前テンプレートに最も一致する位置を探し、左右どちらかを返す。

    行全体に対してテンプレートマッチングをかけ、最も一致度が高い位置が
    行の右端からKILL_FEED_RIGHT_HALF_WIDTH_RATIO*frame_width以上離れていれば
    "kill"（自分がキラー側）、それ未満なら"death"（自分が被害者側）とみなす。
    一致度がmatch_threshold未満なら該当なしとしてNoneを返す。
    """
    row_mask = _white_text_mask(row_bgr)

    best_val = -1.0
    best_loc = None
    best_width = 0
    for template_mask in _get_own_name_template_masks():
        if row_mask.shape[0] < template_mask.shape[0] or row_mask.shape[1] < template_mask.shape[1]:
            continue
        result = cv2.matchTemplate(row_mask, template_mask, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val > best_val:
            best_val = max_val
            best_loc = max_loc
            best_width = template_mask.shape[1]

    if best_loc is None or best_val < match_threshold:
        return None

    match_center_x = best_loc[0] + best_width / 2
    distance_from_right = row_bgr.shape[1] - match_center_x
    split_distance = KILL_FEED_RIGHT_HALF_WIDTH_RATIO * frame_width
    return "kill" if distance_from_right > split_distance else "death"


# 行の高さ（KILL_FEED_ROW_HEIGHT_RATIOから計算）は丸め誤差で39px/40pxの間で
# ばらつくが、名前テンプレートは40px丁度のため39pxの行ではテンプレートが
# 収まらず必ず不一致になる。名前照合の際は上下に余白を持たせて切り出すことで
# この丸め誤差を吸収する。
_NAME_MATCH_ROW_PADDING_PX = 4


def classify_own_kill_feed_state(
    roi_frame: np.ndarray,
    match_threshold: float = 0.5,
    width_ratio: float = ROIConfig().width_ratio,
) -> str | None:
    """キルフィードROI内で自分の名前がある行を探し、キラー側("kill")か
    被害者側("death")かを判定する。

    自分のプレイヤー名のテンプレート画像を各行に対してテンプレートマッチングし、
    最も一致する位置が行の右端から一定距離以上離れていれば"kill"（自分が
    キラー側）、それ未満なら"death"（自分が被害者側）とみなす
    （`_find_own_name_side`参照）。名前の背景色はハイライトやリプレイモードで
    変わりうるため、色ではなく名前の形そのものの一致で判定する。

    キルフィードは新しい行が常に固定の高さで上から順に埋まっていくため、
    固定の行高さで分割し、行ごとに判定してから統合する。

    戻り値は "kill"（自分のキル）、"death"（自分の被害）、どちらの気配もなければ None。
    いずれかの行が "kill" ならそれを優先して返す。
    """
    height = roi_frame.shape[0]
    width = roi_frame.shape[1]
    frame_width = width / width_ratio
    row_height = height * KILL_FEED_ROW_HEIGHT_RATIO
    top_offset = height * KILL_FEED_ROW_TOP_OFFSET_RATIO
    num_rows = int((height - top_offset) / row_height)

    row_states = []
    for i in range(num_rows):
        row_top = max(0, int(top_offset + i * row_height) - _NAME_MATCH_ROW_PADDING_PX)
        row_bottom = min(height, int(top_offset + (i + 1) * row_height) + _NAME_MATCH_ROW_PADDING_PX)
        row = roi_frame[row_top:row_bottom, :]
        state = _find_own_name_side(row, match_threshold, frame_width)
        if state:
            row_states.append(state)

    if "kill" in row_states:
        return "kill"
    if "death" in row_states:
        return "death"
    return None


def detect_own_kill_windows(
    video_path: str,
    roi: ROIConfig = ROIConfig(),
    sample_fps: float = 10.0,
    min_duration_sec: float = 0.3,
    merge_gap_sec: float = 0.5,
) -> List[Tuple[float, float]]:
    """キルフィード内に「自分のキル」を示す行が出現している区間を検出する。

    行の出現から一定時間はキルフィードに残り続けるため、連続キル（マルチキル）の
    場合はまとめて1つの区間として返る。短いギャップの結合・最小持続時間未満の
    ノイズ除去を行う。
    """
    raw_windows: List[Tuple[float, float]] = []
    window_start = None
    prev_timestamp = None

    for timestamp, roi_frame in extract_roi_frames(video_path, roi, sample_fps):
        is_own_kill_visible = classify_own_kill_feed_state(roi_frame, width_ratio=roi.width_ratio) == "kill"

        if is_own_kill_visible and window_start is None:
            window_start = timestamp
        elif not is_own_kill_visible and window_start is not None:
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
