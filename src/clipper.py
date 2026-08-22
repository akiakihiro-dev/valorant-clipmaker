"""検出区間から動画クリップを切り出すモジュール。

`ffmpeg`コマンドを外部プロセスとして呼び出し、区間の切り出し・結合を行う。
再エンコードなし（`-c copy`）はコピーのみのため高速・無劣化だが、キーフレームの
位置によっては失敗する場合があるため、失敗時は再エンコードにフォールバックする。
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple


def _run_ffmpeg_cut(
    video_path: str, start: float, duration: float, output_path: str, copy: bool
) -> bool:
    """ffmpegで1区間を切り出す。成功した場合はTrueを返す。"""
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        # -ssを-iより前に置く高速シークだと、-c copyでは直前のキーフレームまで
        # 遡って開始するため、複数区間を結合したときに前の区間の末尾と同じ
        # 内容が重複再生されることがあった。-iの後に置く正確シークでは
        # 指定時刻以降の最初のキーフレームから開始する（遡らない）ため、
        # 区間同士が重複しない。
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        # 音声側が映像のキーフレーム位置より手前のフレームからコピーされ、
        # 音声PTSが負の値になることがある。これを補正しないと再生開始直後に
        # 音声・映像がズレてカクつくため、タイムスタンプを0基準に揃える。
        "-avoid_negative_ts",
        "make_zero",
    ]
    if copy:
        command += ["-c", "copy"]
    else:
        command += ["-c:v", "libx264", "-c:a", "aac"]
    command.append(output_path)

    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0:
        return False
    return os.path.exists(output_path) and os.path.getsize(output_path) > 0


def _cut_segment(video_path: str, start: float, end: float, output_path: str) -> None:
    """入力動画から区間を切り出す。`-c copy`に失敗した場合のみ再エンコードする。"""
    duration = end - start
    if _run_ffmpeg_cut(video_path, start, duration, output_path, copy=True):
        return

    if not _run_ffmpeg_cut(video_path, start, duration, output_path, copy=False):
        raise RuntimeError(f"ffmpegでの切り出しに失敗しました ({start:.2f}s - {end:.2f}s)")


def _concat_segments(segment_paths: List[str], output_path: str) -> None:
    """複数の動画ファイルをffmpegのconcat demuxerで1本に結合する。"""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as list_file:
        for path in segment_paths:
            escaped_path = os.path.abspath(path).replace("'", "'\\''")
            list_file.write(f"file '{escaped_path}'\n")
        list_file_path = list_file.name

    try:
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file_path,
            "-c",
            "copy",
            output_path,
        ]
        result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError(f"動画の結合に失敗しました: {result.stderr.decode(errors='ignore')}")
    finally:
        os.remove(list_file_path)


def create_highlight_clip(
    video_path: str,
    highlight_ranges: List[Tuple[float, float]],
    output_path: str,
) -> bool:
    """ハイライト区間の一覧から、区間ごとに切り出して1本のクリップに結合する。

    区間が1件も無い場合は何も出力せずFalseを返す。
    """
    if not highlight_ranges:
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        segment_paths = []
        for i, (start, end) in enumerate(highlight_ranges):
            segment_path = os.path.join(temp_dir, f"segment_{i:03d}.mp4")
            _cut_segment(video_path, start, end, segment_path)
            segment_paths.append(segment_path)

        if len(segment_paths) == 1:
            shutil.move(segment_paths[0], output_path)
        else:
            _concat_segments(segment_paths, output_path)

    return True
