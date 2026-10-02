"""Extract the three screen-recording clips used in the film as numbered JPEG frames.

The recording itself is not committed (it is a phone capture). Pass its path:

    uv run python video/extract_frames.py path/to/Screen_Recording.mp4

Clips (seconds in the source recording):
  A  answer arrives in Telegram, with "[1] Hooks"     12.5 - 15.0
  B  tapping the source: "Open Link" dialog           16.0 - 17.8
  C  the docs page scrolls up to the cited sentence   33.8 - 39.0
Frames are upscaled 2x with Lanczos + light sharpening so they stay crisp at 1080p.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "assets" / "rec"
FPS = 30
CLIPS = {"A": (12.5, 15.0), "B": (16.0, 17.8), "C": (33.8, 39.0)}


def main(src: str) -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    meta = {}
    for name, (start, end) in CLIPS.items():
        d = OUT / name
        d.mkdir(parents=True)
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-i",
                src,
                "-vf",
                f"fps={FPS},scale=iw*2:ih*2:flags=lanczos,unsharp=5:5:0.6",
                "-q:v",
                "2",
                str(d / "%04d.jpg"),
            ],
            check=True,
        )
        meta[name] = {"start": start, "end": end, "frames": len(list(d.glob("*.jpg")))}
    (OUT / "clips.json").write_text(json.dumps({"fps": FPS, "clips": meta}, indent=2))
    print(json.dumps(meta))


if __name__ == "__main__":
    main(sys.argv[1])
