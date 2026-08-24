"""検出区間から動画クリップを切り出すモジュール。

`ffmpeg`コマンドを外部プロセスとして呼び出し、区間の切り出し・結合を行う。

区間の切り出しは再エンコード（`-c:v libx264 -c:a aac`）で行う。`-c copy`は
高速・無劣化だが、映像はキーフレームの位置でしか開始できないのに対し音声は
任意の位置から開始できるため、指定した開始時刻と映像のキーフレーム位置が
ずれるほど音声と映像の開始タイミングがずれてしまい（`-ss`を`-i`の前に置けば
音声が映像より早く始まり、後に置けば映像が音声より遅れて始まる）、
どちらの置き方でも解消できなかった。再エンコードであれば任意の時刻を
基準に音声・映像を揃えて出力できるため、この問題が起きない。
"""

import os
import shutil
import subprocess
import tempfile
from typing import List, Tuple


def _cut_segment(video_path: str, start: float, end: float, output_path: str) -> None:
    """入力動画から区間を切り出す（再エンコード）。"""
    duration = end - start
    command = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-ss",
        f"{start:.3f}",
        "-t",
        f"{duration:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        output_path,
    ]
    result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if result.returncode != 0 or not (os.path.exists(output_path) and os.path.getsize(output_path) > 0):
        raise RuntimeError(
            f"ffmpegでの切り出しに失敗しました ({start:.2f}s - {end:.2f}s): "
            f"{result.stderr.decode(errors='ignore')}"
        )


def _concat_segments(segment_paths: List[str], output_path: str) -> None:
    """複数の動画ファイルをffmpegのconcat demuxerで1本に結合する（再エンコード）。"""
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
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
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
