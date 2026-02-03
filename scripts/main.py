from __future__ import annotations

import argparse
import os

from .ingest_video import ingest_video
from .search import search_text

DEFAULT_QUERY = "\u043a\u0440\u0430\u0441\u043d\u0430\u044f \u043c\u0430\u0448\u0438\u043d\u0430"


def _print_results(results: list[dict]) -> None:
    if not results:
        print("No results.")
        return
    for i, item in enumerate(results, start=1):
        score = item.get("score", 0.0)
        print(
            f"{i}. score={score:.4f} camera_id={item.get('camera_id')} "
            f"video_path={item.get('video_path')} timestamp={item.get('timestamp')} "
            f"frame_index={item.get('frame_index')} frame_id={item.get('frame_id')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=os.getenv("VIDEO_PATH", "data/videos/sample.mp4"))
    parser.add_argument("--camera-id", default=os.getenv("CAMERA_ID", "cam01"))
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        print(f"Video file not found: {args.video}")
        print("Set --video or VIDEO_PATH.")
        return 1

    print(f"Ingesting: {args.video} (camera_id={args.camera_id}, fps={args.fps})")
    indexed = ingest_video(args.video, args.camera_id, fps=args.fps)
    print(f"Indexed frames: {indexed}")

    print(f"Searching: {args.query}")
    results = search_text(args.query, top_k=args.top_k)
    _print_results(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
