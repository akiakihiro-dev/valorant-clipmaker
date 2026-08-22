"""検出区間から動画クリップを切り出すモジュール。

`ffmpeg`コマンドを外部プロセスとして呼び出し、区間の切り出しを行う。
再エンコードなし（`-c copy`）はコピーのみのため高速・無劣化だが、キーフレームの
位置によっては失敗する場合があるため、失敗時は再エンコードにフォールバックする。
"""

import os
import subprocess
from typing import List, Tuple


def _run_ffmpeg_cut(
    video_path: str, start: float, duration: float, output_path: str, copy: bool
) -> bool:
    """ffmpegで1区間を切り出す。成功した場合はTrueを返す。"""
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        video_path,
        "-t",
        f"{duration:.3f}",
        # -ssを-iより前に置く高速シークでは、音声側が映像のキーフレーム位置より
        # 手前のフレームからコピーされ、音声PTSが負の値になることがある。
        # これを補正しないと再生開始直後に音声・映像がズレてカクつくため、
        # タイムスタンプを0基準に揃える。
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


def extract_clips(
    video_path: str,
    windows: List[Tuple[float, float]],
    output_dir: str,
    prefix: str = "clip",
) -> List[str]:
    """検出区間ごとに動画を切り出し、`output_dir`にmp4として保存する。

    区間ファイル名には開始・終了時刻（秒）を含め、目視確認しやすくする。
    まず`-c copy`（再エンコードなし）で切り出しを試み、失敗した場合のみ再エンコードする。
    """
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []
    for i, (start, end) in enumerate(windows):
        duration = end - start
        output_path = os.path.join(
            output_dir, f"{prefix}_{i:03d}_{start:.2f}-{end:.2f}.mp4"
        )

        if not _run_ffmpeg_cut(video_path, start, duration, output_path, copy=True):
            print(f"  -c copyでの切り出しに失敗したため再エンコードします: {output_path}")
            if not _run_ffmpeg_cut(video_path, start, duration, output_path, copy=False):
                raise RuntimeError(f"ffmpegでの切り出しに失敗しました: {output_path}")

        output_paths.append(output_path)

    return output_paths
