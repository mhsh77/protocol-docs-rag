# Showcase film

A ~47 s kinetic-typography film (1920×1080, 30 fps) for LinkedIn, X and the GitHub README.
The piece is code: one HTML composition where every frame is a pure function of time, rendered
frame by frame in headless Chrome with a real shutter-based motion blur, then encoded with ffmpeg.

## Story

| Time | Beat | Source of the content |
|---|---|---|
| 0–4 s | Title opening, readable from frame 0 (LinkedIn autoplay and thumbnail): "A RAG assistant for the Uniswap docs, that cites every answer [1] and won't guess" + what is inside | |
| 4–14 s | A question the docs can't answer: a naive RAG setup invents an answer, this assistant declines | Verbatim from eval record `q108` (`naive_rag` vs `hybrid_rerank`), run `run-8726a14b6f` |
| 14–17 s | "It knows what it doesn't know." | |
| 17–20 s | "How many hooks can a Uniswap v4 pool have?" → cited answer `[1]` | The live bot's answer |
| 20–31 s | The `[1]` opens into the real Telegram session: answer → tap the source → the docs page lands on the quoted sentence | Screen recording of the live bot (not committed) |
| 31–41 s | Measured, not claimed: 80 held-out questions, 23 unanswerable, 23/23 declined, unsupported claims 5.0% (naive RAG) vs 0.8% | `eval/runs/run-8726a14b6f/results.json` |
| 41–47 s | Hybrid search · Reranking · Cited answers · Honest refusals → end card | |

The `[1]` citation chip is the thread through the film: it appears in the answer, grows into the
source, becomes the mark of the evidence section, and carries into the end card.

## Rebuild

```bash
# 1. frames from the screen recording (kept out of git)
uv run python video/extract_frames.py path/to/Screen_Recording.mp4
# 2. preview in a browser: video/comp.html?play  (or ?t=17.5 to pin a frame)
# 3. sound design (synthesised, from the composition's cue list)
uv run --with playwright --with numpy python video/sfx.py --out video/build/sfx.wav
# 4. stills to check the look, then the film
uv run --with playwright --with numpy --with pillow python video/render.py --stills 6.4,17.6,29.4 --out video/build/stills.png
uv run --with playwright --with numpy --with pillow python video/render.py --out video/build/showcase.mp4 --audio video/build/sfx.wav
```

### Voice-over (optional second cut)

An AI voice (Orpheus `troy`, served free by Groq; accept the model's terms once in the Groq
console) reads five short lines timed to the scenes. The picture is reused; a small
"Voice: AI-generated" line is added to the end card, and the effects are ducked under the voice.

```bash
uv run --with openai python video/voiceover.py --voice troy   # lines + fit check against each scene
ffmpeg -i video/build/showcase_draft.mp4 -vf "hqdn3d=0:0:8:8,drawtext=fontfile=video/fonts/Inter-Variable.ttf:text='Voice\: AI-generated (Orpheus TTS)':fontsize=21:fontcolor=0x625c53:x=(w-text_w)/2:y=742:alpha='if(lt(t,44.4),0,min(1,(t-44.4)/0.6))'" -an -c:v libx264 -crf 18 video/build/showcase_vo_picture.mp4
uv run --with openai python video/voiceover.py --voice troy --video video/build/showcase_vo_picture.mp4 --sfx video/build/sfx.wav --out video/build/showcase_vo.mp4
```

Each generated line was checked by transcribing it back with Whisper (all five matched the
script), and the final mix was transcribed with timestamps to confirm every line lands on its scene.

Rendering uses the locally installed Chrome (`channel="chrome"`). On a 4-core laptop the full
film (1,416 frames × 4 shutter samples) took about 45 minutes on 3 parallel browsers.

## Credits and licences

- Fonts (SIL Open Font License, licence files in `fonts/`): Instrument Serif, Inter, JetBrains Mono.
- Sound: synthesised in `sfx.py`; no samples or music. Voice-over (second cut): AI-generated with Orpheus TTS on Groq, disclosed on the end card.
- Text-reveal timings follow the MIT-licensed
  [kinetic-typography skill](https://github.com/iart-ai/kinetic-typography-skills) (masked line
  reveals, expo-out easing, 30–70 ms staggers). No code from non-commercially licensed sources is used.
- Uniswap is named only as the subject of the docs; no Uniswap logo or brand assets are used, and
  the end card states that the project is unofficial.
