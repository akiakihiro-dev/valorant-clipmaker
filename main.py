"""キルフィードで自分のキルが表示されている区間を抜き出すPoCスクリプト。

`clipsample/`内の動画に対し、キルフィード内の緑色ハイライト（自分のキル）が
左側に表示されている区間を検出し、それらを1本に結合したハイライトクリップとして
`output/`へ保存する。
"""

import glob
import os

from src.clipper import create_highlight_clip
from src.highlight_detector import compute_highlight_ranges, get_video_duration_sec
from src.kill_detector import detect_own_kill_windows

CLIP_SAMPLE_DIR = "clipsample"
OUTPUT_DIR = "output"


def main() -> None:
    video_paths = sorted(glob.glob(os.path.join(CLIP_SAMPLE_DIR, "*.mp4")))
    if not video_paths:
        print(f"{CLIP_SAMPLE_DIR} にmp4ファイルが見つかりませんでした。")
        return

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"[{video_name}] キルフィードの自分キル区間を検出中...")

        windows = detect_own_kill_windows(video_path)
        if not windows:
            print(f"[{video_name}] 自分のキル区間は検出されませんでした。")
            continue

        video_duration = get_video_duration_sec(video_path)
        highlight_ranges = compute_highlight_ranges(windows, video_duration=video_duration)

        for start, end in highlight_ranges:
            print(f"  {start:.2f}s - {end:.2f}s (duration {end - start:.2f}s)")

        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_highlight.mp4")
        create_highlight_clip(video_path, highlight_ranges, output_path)
        print(f"[{video_name}] {output_path}に出力しました。")


if __name__ == "__main__":
    main()
