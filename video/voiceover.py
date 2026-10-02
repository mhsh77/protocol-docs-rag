"""AI voice-over for the showcase film: Orpheus text-to-speech served by Groq (free tier).

Each line is synthesised separately, measured, checked against the scene it belongs to,
placed at its start time, and mixed over the sound design with the effects ducked under
the voice. The rendered picture is reused as is; only the audio track changes.

    uv run --with openai python video/voiceover.py --voices autumn,diana,hannah,austin,daniel,troy
    uv run --with openai python video/voiceover.py --voice troy \
        --video video/build/showcase.mp4 --sfx video/build/sfx.wav --out video/build/showcase_vo.mp4

The model's terms must be accepted once in the Groq console. Viewers are told the voice is
AI-generated (a line on the end card of the voiced cut).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI

HERE = Path(__file__).parent
BUILD = HERE / "build" / "vo"

# (start s, latest end s, text) on the 47.2 s timeline (title opening = first 4.2 s).
LINES: list[tuple[float, float, str]] = [
    (0.35, 4.1, "I built a RAG assistant for the Uniswap developer docs."),
    (
        4.6,
        13.9,
        "Most AI bots answer anyway, even when the docs don't say. "
        "Ask this one something the docs don't cover, and it tells you so.",
    ),
    (
        20.6,
        30.9,
        "When it does answer, every claim has a citation. "
        "Tap it, and you land on the exact sentence it used.",
    ),
    (
        31.6,
        40.9,
        "I tested it on eighty held-out questions. "
        "It declined every one it couldn't answer, "
        "and under one percent of its claims were unsupported.",
    ),
    (41.4, 46.6, "It's live on Telegram. The code and the full evaluation are on GitHub."),
]

STYLE = (
    "Calm, confident and warm, like an engineer presenting their own project to a founder. "
    "Natural pace, clear diction, no salesy hype. Slight emphasis on 'every claim', "
    "'tells you so' and the numbers."
)


MODEL = "canopylabs/orpheus-v1-english"
VOICES = ["autumn", "diana", "hannah", "austin", "daniel", "troy"]


def client() -> OpenAI:
    # Read the key from the project .env only, never from unrelated system-wide variables.
    key = dotenv_values(HERE.parent / ".env").get("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set in .env")
    return OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(out.stdout.strip())


def synth(c: OpenAI, model: str, voice: str, text: str, out: Path) -> Path:
    kwargs: dict[str, object] = {
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav",
    }
    if "gpt" in model:  # instruction-steerable TTS models
        kwargs["instructions"] = STYLE
    with c.audio.speech.with_streaming_response.create(**kwargs) as resp:  # type: ignore[arg-type]
        resp.stream_to_file(out)
    return out


def make_lines(c: OpenAI, model: str, voice: str) -> list[dict[str, object]]:
    d = BUILD / voice
    d.mkdir(parents=True, exist_ok=True)
    report = []
    for i, (start, end, text) in enumerate(LINES):
        f = synth(c, model, voice, text, d / f"line{i}.wav")
        dur = duration(f)
        slot = end - start
        report.append(
            {
                "line": i,
                "start": start,
                "dur": round(dur, 2),
                "slot": round(slot, 2),
                "fits": dur <= slot,
                "file": str(f),
            }
        )
    (d / "report.json").write_text(json.dumps(report, indent=2))
    return report


def mix(report: list[dict[str, object]], sfx: Path, video: Path, out: Path, tempo: float) -> None:
    """Place lines, duck the effects under the voice, normalise, and remux onto the video."""
    inputs = ["-i", str(video), "-i", str(sfx)]
    chains = []
    for k, r in enumerate(report):
        inputs += ["-i", str(r["file"])]
        delay = int(float(r["start"]) * 1000)  # type: ignore[arg-type]
        speed = f"atempo={tempo}," if tempo != 1.0 else ""
        place = f"adelay={delay}|{delay},pan=stereo|c0=c0|c1=c0"
        chains.append(f"[{k + 2}:a]{speed}aresample=48000,{place}[v{k}]")
    n = len(report)
    # pad the voice to the film's length: the ducker ends with its sidechain, and -shortest
    # would otherwise cut the end card where the last line ends
    full = duration(video)
    vo_mix = (
        "".join(f"[v{k}]" for k in range(n))
        + f"amix=inputs={n}:normalize=0,apad=whole_dur={full:.3f}[vo]"
    )
    duck = "sidechaincompress=threshold=0.02:ratio=6:attack=15:release=350:makeup=1"
    graph = ";".join(
        [
            *chains,
            vo_mix,
            "[vo]asplit=2[vo1][vokey]",
            # effects ducked by ~8 dB while the voice speaks
            "[1:a]aresample=48000[fx]",
            f"[fx][vokey]{duck}[fxd]",
            "[fxd][vo1]amix=inputs=2:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]",
        ]
    )
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            *inputs,
            "-filter_complex",
            graph,
            "-map",
            "0:v",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-ar",
            "48000",
            "-shortest",
            "-movflags",
            "+faststart",
            str(out),
        ],
        check=True,
    )
    print(f"video with voice-over -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--voices", default="", help="comma list: synthesise one line in each")
    ap.add_argument("--voice", default="troy", choices=VOICES)
    ap.add_argument("--tempo", type=float, default=1.0, help="atempo if a line runs long")
    ap.add_argument("--video", type=Path)
    ap.add_argument("--sfx", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    c = client()
    model = a.model
    print(f"model: {model}")
    if a.voices:
        BUILD.mkdir(parents=True, exist_ok=True)
        for v in a.voices.split(","):
            f = synth(c, model, v.strip(), LINES[1][2], BUILD / f"audition_{v.strip()}.wav")
            print(f"{v.strip():8} {duration(f):.2f}s -> {f}")
        return
    report = make_lines(c, model, a.voice)
    for r in report:
        verdict = "ok" if r["fits"] else "TOO LONG"
        print(f"line {r['line']}: {r['dur']}s of {r['slot']}s slot {verdict}")
    if a.video and a.sfx and a.out:
        mix(report, a.sfx, a.video, a.out, a.tempo)


if __name__ == "__main__":
    main()
