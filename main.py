"""キルフィードで自分のキルが表示されている区間を抜き出すPoCスクリプト。

`clipsample/`内の動画に対し、キルフィード内の緑色ハイライト（自分のキル）が
左側に表示されている区間を検出し、区間ごとに動画を切り出して`output/`へ保存する。
"""

import glob
import os

from src.clipper import extract_clips
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

        clip_output_dir = os.path.join(OUTPUT_DIR, video_name)
        output_paths = extract_clips(video_path, highlight_ranges, clip_output_dir, prefix="own_kill")
        print(f"[{video_name}] {len(output_paths)}件のクリップを{clip_output_dir}に出力しました。")


if __name__ == "__main__":
    main()
