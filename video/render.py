"""Render video/comp.html to MP4 (or a contact sheet of stills).

The composition is a pure function of time (window.__seek(t)), so every frame is a
deterministic screenshot. Motion blur is a real 180-degree shutter: each output frame
averages `--samples` captures spread over the first half of the frame interval.

    uv run --with playwright --with numpy --with pillow python video/render.py --stills 1,5,9 --out stills.png
    uv run --with playwright --with numpy --with pillow python video/render.py --out showcase.mp4 --audio sfx.wav

Uses the locally installed Chrome (`channel="chrome"`), so no browser download is needed.
"""

from __future__ import annotations

import argparse
import io
import multiprocessing as mp
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
COMP = (HERE / "comp.html").resolve().as_uri()
W, H = 1920, 1080


def _open(p):  # type: ignore[no-untyped-def]
    browser = p.chromium.launch(channel="chrome", headless=True, args=["--force-color-profile=srgb"])
    page = browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
    page.goto(COMP)
    page.wait_for_function("window.__ready === true", timeout=120_000)
    return browser, page


def _capture(page, t: float) -> np.ndarray:  # type: ignore[no-untyped-def]
    page.evaluate(f"window.__seek({t:.5f})")
    png = page.screenshot(type="png", clip={"x": 0, "y": 0, "width": W, "height": H})
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.float32)


def _worker(args: tuple[list[int], int, int, str]) -> int:
    frames, fps, samples, outdir = args
    with sync_playwright() as p:
        browser, page = _open(p)
        for f in frames:
            t0 = f / fps
            if samples <= 1:
                acc = _capture(page, t0)
            else:
                # 180-degree shutter: average in linear light across half a frame interval
                acc = np.zeros((H, W, 3), np.float32)
                for s in range(samples):
                    img = _capture(page, t0 + (s / samples) * (0.5 / fps))
                    acc += (img / 255.0) ** 2.2
                acc = (acc / samples) ** (1 / 2.2) * 255.0
            Image.fromarray(np.clip(acc + 0.5, 0, 255).astype(np.uint8)).save(
                Path(outdir) / f"{f:05d}.png", compress_level=1
            )
        browser.close()
    return len(frames)


def stills(times: list[float], out: Path) -> None:
    with sync_playwright() as p:
        browser, page = _open(p)
        shots = [Image.fromarray(_capture(page, t).astype(np.uint8)) for t in times]
        browser.close()
    cols = min(3, len(shots))
    rows = (len(shots) + cols - 1) // cols
    tw, th = W // 3, H // 3
    sheet = Image.new("RGB", (cols * tw, rows * th), "black")
    for i, im in enumerate(shots):
        sheet.paste(im.resize((tw, th), Image.LANCZOS), ((i % cols) * tw, (i // cols) * th))
    sheet.save(out)
    print(f"sheet -> {out}")


def render(out: Path, fps: int, samples: int, workers: int, audio: Path | None,
           start: float, end: float | None) -> None:
    with sync_playwright() as p:
        browser, page = _open(p)
        duration = float(page.evaluate("window.__duration"))
        browser.close()
    end = end if end is not None else duration
    frames = list(range(int(start * fps), int(end * fps)))
    tmp = Path(tempfile.mkdtemp(prefix="docrag_render_"))
    chunks = [frames[i::workers] for i in range(workers)]
    t0 = time.time()
    print(f"rendering {len(frames)} frames x {samples} samples on {workers} workers ...")
    with mp.Pool(workers) as pool:
        done = sum(pool.map(_worker, [(c, fps, samples, str(tmp)) for c in chunks]))
    print(f"captured {done} frames in {time.time() - t0:.0f}s")
    cmd = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-start_number", str(frames[0]),
           "-i", str(tmp / "%05d.png")]
    if audio:
        cmd += ["-ss", f"{start:.3f}", "-i", str(audio), "-af", "loudnorm=I=-18:TP=-2:LRA=11",
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-tune", "film", str(out)]
    subprocess.run(cmd, check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"video -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stills", type=str, default="")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--audio", type=Path, default=None)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    a = ap.parse_args()
    if a.stills:
        stills([float(x) for x in a.stills.split(",")], a.out)
    else:
        render(a.out, a.fps, a.samples, a.workers, a.audio, a.start, a.end)


if __name__ == "__main__":
    main()
