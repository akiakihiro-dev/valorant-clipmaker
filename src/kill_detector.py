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


def _teal_ratio(
    hsv_region: np.ndarray, hue_low: int, hue_high: int, saturation_threshold: int
) -> float:
    """HSV領域内で、緑（ティール）色相のピクセルが彩度のあるピクセルに占める比率を返す。"""
    hue = hsv_region[:, :, 0].astype(np.int32)
    saturation = hsv_region[:, :, 1].astype(np.int32)
    colored = saturation > saturation_threshold
    if colored.sum() == 0:
        return 0.0
    teal_mask = (hue >= hue_low) & (hue <= hue_high) & colored
    return float(teal_mask.sum()) / colored.sum()


# キルフィードの1行あたりの高さ・先頭オフセット（ROI全体に対する比率）。
# クロスヘアアイコン（彩度が低く明度が高い白色部分）のY座標をサンプルクリップの
# 複数フレームで実測して求めた値。ROIConfig()のデフォルト値（1920x1080基準）では
# 実際の1行のピッチはROI高さの約1/6ではなく約1/7.7で、先頭行の手前にも
# 高さの約3%分のマージンがある。以前は1/6・オフセット無しで割っていたため、
# row_idxが大きくなるほど実際の行境界とのズレが蓄積し、3行目以降で隣接行の
# 内容が混ざってキル/被害の判定が不安定になる問題があった。
KILL_FEED_ROW_HEIGHT_RATIO = 39.3 / 302
KILL_FEED_ROW_TOP_OFFSET_RATIO = 9.85 / 302


def _classify_row(
    row_bgr: np.ndarray, hue_low: int, hue_high: int, saturation_threshold: int, ratio_threshold: float
) -> str | None:
    """1行分の画像から、自分がキラー側("kill")か被害者側("death")かを判定する（色ハイライトのみ）。

    左右の判定領域は行の中央（クロスヘアアイコン）から離した位置を見る。
    ハイライトの緑〜黄色のグラデーションは中央のクロスヘアアイコンの少し先まで
    続いており、中央に近い位置（45%〜55%付近）まで判定領域を広げると、
    どちらの色でもない側にもグラデーションの残りが入り込み、僅差で
    kill/deathの判定がフレームごとに反転してしまう（キルフィードが表示され
    続けているのに区間が不自然に分割される原因になっていた）。

    マップ内の緑色の背景（植物・ガラス張りの建物等）がこの判定域に写り込むと、
    実際のキルフィードより強い緑色反応を示すことがあり、色情報だけでは
    誤検出を避けられない。そのため`_classify_row_by_name`で名前の
    テンプレートマッチングと組み合わせて使う想定。
    """
    hsv = cv2.cvtColor(row_bgr, cv2.COLOR_BGR2HSV)
    height, width = row_bgr.shape[:2]
    left = hsv[:, int(width * 0.15) : int(width * 0.35)]
    right = hsv[:, int(width * 0.70) : int(width * 0.90)]

    left_ratio = _teal_ratio(left, hue_low, hue_high, saturation_threshold)
    right_ratio = _teal_ratio(right, hue_low, hue_high, saturation_threshold)

    if max(left_ratio, right_ratio) < ratio_threshold:
        return None
    return "kill" if left_ratio > right_ratio else "death"


# 自分のプレイヤー名「火事場のバカ」をキルフィード内で切り出したテンプレート画像。
# サンプルクリップ（1920x1080）のキルフィード1行分から名前部分のみを切り出したもの。
OWN_NAME_TEMPLATE_PATH = "assets/templates/own_name.png"

_own_name_template_mask: np.ndarray | None = None


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


def _get_own_name_template_mask() -> np.ndarray:
    """自分の名前テンプレートの二値マスクを読み込む（初回のみ）。"""
    global _own_name_template_mask
    if _own_name_template_mask is None:
        template_bgr = cv2.imread(OWN_NAME_TEMPLATE_PATH)
        if template_bgr is None:
            raise IOError(f"名前テンプレート画像を読み込めませんでした: {OWN_NAME_TEMPLATE_PATH}")
        _own_name_template_mask = _white_text_mask(template_bgr)
    return _own_name_template_mask


def _find_own_name_side(row_bgr: np.ndarray, match_threshold: float) -> str | None:
    """行内で自分の名前テンプレートに最も一致する位置を探し、左右どちらかを返す。

    行全体に対してテンプレートマッチングをかけ、最も一致度が高い位置のx座標が
    行の左半分にあれば"kill"（自分がキラー側）、右半分にあれば"death"
    （自分が被害者側）とみなす。一致度がmatch_threshold未満なら該当なしとしてNoneを返す。
    """
    template_mask = _get_own_name_template_mask()
    row_mask = _white_text_mask(row_bgr)

    if row_mask.shape[0] < template_mask.shape[0] or row_mask.shape[1] < template_mask.shape[1]:
        return None

    result = cv2.matchTemplate(row_mask, template_mask, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val < match_threshold:
        return None

    match_center_x = max_loc[0] + template_mask.shape[1] / 2
    return "kill" if match_center_x < row_bgr.shape[1] / 2 else "death"


# 行の高さ（KILL_FEED_ROW_HEIGHT_RATIOから計算）は丸め誤差で39px/40pxの間で
# ばらつく。名前テンプレートは40px丁度で作成したため、39pxになった行では
# テンプレートがそもそも収まらず必ず不一致になっていた（行インデックスの偶奇で
# 名前検出が半分近く機能しない不具合の原因）。名前照合の際は上下に余白を
# 持たせて切り出すことで、丸め誤差を吸収する。
_NAME_MATCH_ROW_PADDING_PX = 4


def _classify_row_by_name(
    roi_frame: np.ndarray,
    row_top: int,
    row_bottom: int,
    match_threshold: float,
    hue_low: int,
    hue_high: int,
    saturation_threshold: int,
    ratio_threshold: float,
) -> str | None:
    """まず色ハイライトで候補側を絞り込み、見つかった場合のみ名前の一致を確認する。

    色ハイライトだけでは、マップ内の緑色の背景（植物・ガラス張りの建物等）が
    実際のキルフィードより強い反応を示すことがあり、誤検出の原因になっていた。
    一方、名前のテンプレートマッチングだけを独立に行ごとに走らせると、行の
    高さの丸め誤差でテンプレートが収まらない行が生じ、実際にキルフィードが
    表示されているのに検出が抜け落ちることがあった。
    そのため、まず色ハイライト（行の厳密な境界で判定、既存ロジックのまま）で
    候補側を絞り込み、候補が見つかった行だけ上下に余白を持たせて切り出し、
    名前テンプレートと照合して確認する。
    """
    row = roi_frame[row_top:row_bottom, :]
    color_side = _classify_row(row, hue_low, hue_high, saturation_threshold, ratio_threshold)
    if color_side is None:
        return None

    padded_top = max(0, row_top - _NAME_MATCH_ROW_PADDING_PX)
    padded_bottom = min(roi_frame.shape[0], row_bottom + _NAME_MATCH_ROW_PADDING_PX)
    padded_row = roi_frame[padded_top:padded_bottom, :]

    name_side = _find_own_name_side(padded_row, match_threshold)
    if name_side != color_side:
        return None

    return color_side


def classify_own_kill_feed_state(
    roi_frame: np.ndarray,
    match_threshold: float = 0.5,
    hue_low: int = 50,
    hue_high: int = 82,
    saturation_threshold: int = 40,
    ratio_threshold: float = 0.15,
) -> str | None:
    """キルフィードROI内で自分が関わる行があるか、あるなら自分がキラー側か被害者側かを判定する。

    Valorantのキルフィードは、自分が関わった行を緑（ティール、Hue≈70-75）で
    ハイライトする。このハイライトは行全体ではなく、行内で自分の名前がある側
    （左＝自分がキラー、右＝自分が被害者）に強く寄ることをサンプルクリップで確認した。

    色ハイライトだけでは、マップ内の緑色の背景（植物・ガラス張りの建物等）を
    誤検出することがあるため、自分のプレイヤー名「火事場のバカ」のテンプレート
    マッチングと組み合わせ、両方が一致した場合のみ検出とする
    （`_classify_row_by_name`参照）。

    hue_highの上限は95ではなく82に絞っている。KAY/Oなど一部エージェントの
    アイコン自体がシアン系の見た目（Hue≈80-90）を持ち、95まで許容すると
    キャラクター側の色を誤ってハイライトと誤認し、本来「kill」と判定すべき
    行が「death」側に誤判定されてキル区間の検出が途切れる問題があったため。

    キルフィードは新しい行が常に固定の高さで上から順に埋まっていく（試合序盤で
    行数が少なくても1行の高さは変わらない）ため、ROI全体をまとめて見ると
    埋まっていない/無関係な行の分だけ比率が薄まってしまう。そのため固定の行高さで
    分割し、行ごとに判定してから統合する。

    戻り値は "kill"（自分のキル）、"death"（自分の被害）、どちらの気配もなければ None。
    いずれかの行が "kill" ならそれを優先して返す。
    """
    height = roi_frame.shape[0]
    row_height = height * KILL_FEED_ROW_HEIGHT_RATIO
    top_offset = height * KILL_FEED_ROW_TOP_OFFSET_RATIO
    num_rows = int((height - top_offset) / row_height)

    row_states = []
    for i in range(num_rows):
        row_top = int(top_offset + i * row_height)
        row_bottom = int(top_offset + (i + 1) * row_height)
        state = _classify_row_by_name(
            roi_frame, row_top, row_bottom, match_threshold, hue_low, hue_high, saturation_threshold, ratio_threshold
        )
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
    """キルフィード内に「自分のキル」を示す緑色の行が出現している区間を検出する。

    行の出現から一定時間はキルフィードに残り続けるため、連続キル（マルチキル）の
    場合はまとめて1つの区間として返る。`detect_kill_windows`（キルマーク方式）と
    同様に、短いギャップの結合・最小持続時間未満のノイズ除去を行う。
    """
    raw_windows: List[Tuple[float, float]] = []
    window_start = None
    prev_timestamp = None

    for timestamp, roi_frame in extract_roi_frames(video_path, roi, sample_fps):
        is_own_kill_visible = classify_own_kill_feed_state(roi_frame) == "kill"

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
