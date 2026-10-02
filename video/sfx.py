"""Synthesised sound design for the showcase film (no samples, no licensing questions).

Reads the cue list from the composition (window.__events()), renders each cue from a small
palette of materials, places everything in one shared room (a synthetic reverb), adds a quiet
ambient bed, and masters to a -3 dBFS peak.

    uv run --with playwright --with numpy python video/sfx.py --out video/build/sfx.wav
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
from playwright.sync_api import sync_playwright

SR = 48_000
HERE = Path(__file__).parent
rng = np.random.default_rng(7)


def env(n: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / SR
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-np.maximum(t - attack, 0) / decay)


def tone(freq: float, dur: float, attack: float = 0.005, decay: float = 0.2, shape: str = "sine",
         glide: float = 0.0) -> np.ndarray:
    n = int(dur * SR)
    t = np.arange(n) / SR
    f = freq * (1 + glide * (1 - np.exp(-t / 0.03)))
    ph = 2 * np.pi * np.cumsum(f) / SR
    w = np.sin(ph) if shape == "sine" else 2 / np.pi * np.arcsin(np.sin(ph))  # triangle
    return w * env(n, attack, decay)


def noise(dur: float) -> np.ndarray:
    return rng.standard_normal(int(dur * SR))


def onepole(x: np.ndarray, cutoff: np.ndarray | float, high: bool = False) -> np.ndarray:
    """One-pole low/high-pass with a (possibly time-varying) cutoff in Hz."""
    c = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    a = np.exp(-2 * np.pi * c / SR)
    y = np.empty_like(x)
    s = 0.0
    for i in range(len(x)):  # small buffers only
        s = (1 - a[i]) * x[i] + a[i] * s
        y[i] = s
    return x - y if high else y


def mix(*parts: np.ndarray) -> np.ndarray:
    """Sum signals of different lengths (zero-padded to the longest)."""
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[: len(p)] += p
    return out


def db(v: float) -> float:
    return float(10 ** (v / 20))


# ---- the palette: few materials, reused ----
def key() -> np.ndarray:
    x = onepole(noise(0.03), 4200) * env(int(0.03 * SR), 0.0008, 0.006)
    return onepole(x, 1800, high=True) * db(-30) * rng.uniform(0.7, 1.0)


def pop() -> np.ndarray:
    return mix(tone(620, 0.18, 0.003, 0.05, glide=0.35) * 0.8, tone(1240, 0.12, 0.002, 0.03) * 0.2) * db(-21)


def chip() -> np.ndarray:
    return mix(tone(1568, 0.35, 0.002, 0.09, "triangle"), 0.5 * tone(2349, 0.3, 0.002, 0.06)) * db(-24)


def tap() -> np.ndarray:
    x = onepole(noise(0.05), 2500) * env(int(0.05 * SR), 0.001, 0.012)
    return x * db(-20)


def swoosh(dur: float = 0.45, up: bool = True, level: float = -27) -> np.ndarray:
    n = int(dur * SR)
    k = np.linspace(0, 1, n)
    cut = (300 + 3200 * k) if up else (3500 - 3200 * k)
    shape = np.sin(np.pi * k) ** 1.5
    return onepole(noise(dur), cut) * shape * db(level)


def confirm() -> np.ndarray:
    a = tone(659.3, 0.9, 0.006, 0.28, "triangle")
    b = np.concatenate([np.zeros(int(0.09 * SR)), tone(987.8, 0.81, 0.006, 0.3, "triangle")])
    return mix(a, 0.8 * b) * db(-25)


def flag() -> np.ndarray:
    a = tone(233.1, 0.5, 0.004, 0.16)
    b = tone(246.9, 0.5, 0.004, 0.14)  # a soft minor second: "not right"
    thunk = onepole(noise(0.08), 400) * env(int(0.08 * SR), 0.001, 0.02)
    out = (a + b) * 0.5
    out[: len(thunk)] += thunk * 0.8
    return out * db(-19)


def swell(dur: float = 0.9) -> np.ndarray:
    n = int(dur * SR)
    k = np.linspace(0, 1, n)
    pad = sum(tone(f, dur, dur * 0.7, 0.6) for f in (220.0, 329.6, 440.0)) / 3
    return mix(pad * np.sin(np.pi * k) ** 0.8, swoosh(dur, True, -34)) * db(-24)


def count() -> np.ndarray:
    out = np.zeros(int(0.8 * SR))
    times = 0.6 * (1 - (1 - np.linspace(0, 1, 13)) ** 2.2)   # fast, then settling
    for i, t0 in enumerate(times):
        c = tone(1900 + 40 * i, 0.04, 0.0005, 0.008) * db(-27)
        s = int(t0 * SR)
        out[s : s + len(c)] += c[: len(out) - s]
    return out


def bar() -> np.ndarray:
    return tone(392.0, 0.7, 0.18, 0.25, "triangle", glide=0.12) * db(-27)


def hit() -> np.ndarray:
    sub = tone(58, 0.4, 0.002, 0.12, glide=0.8) * 0.9
    click = onepole(noise(0.02), 5000) * env(int(0.02 * SR), 0.0005, 0.004) * 0.35
    sub[: len(click)] += click
    return sub * db(-13)


def resolve() -> np.ndarray:
    chord = [220.0, 277.2, 329.6, 440.0, 554.4]
    return mix(*(tone(f, 3.4, 0.02 + 0.015 * i, 1.1, "triangle") for i, f in enumerate(chord))) / 5 * db(-15)


PALETTE = {
    "key": key, "pop": pop, "chip": chip, "tap": tap, "confirm": confirm, "flag": flag,
    "swell": swell, "count": count, "bar": bar, "hit": hit, "resolve": resolve,
    "swoosh": lambda: swoosh(0.45, True), "swoosh_soft": lambda: swoosh(0.5, True, -31),
    "open": lambda: swoosh(0.6, True, -24), "close": lambda: swoosh(0.5, False, -25),
}


def room(dur: float = 1.6) -> np.ndarray:
    n = int(dur * SR)
    ir = rng.standard_normal(n) * np.exp(-np.arange(n) / SR / 0.32)
    ir = np.fft.irfft(np.fft.rfft(ir) * (1 / (1 + (np.fft.rfftfreq(n, 1 / SR) / 3500) ** 2)), n)
    return ir / np.sqrt(np.sum(ir**2))


def bed(n: int) -> np.ndarray:
    t = np.arange(n) / SR
    lfo = 0.5 + 0.5 * np.sin(2 * np.pi * 0.07 * t)
    drone = (np.sin(2 * np.pi * 110 * t) + 0.6 * np.sin(2 * np.pi * 164.8 * t + 1.0)) * (0.6 + 0.4 * lfo)
    air = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / (1 + (np.fft.rfftfreq(n, 1 / SR) / 900) ** 2), n)
    air /= np.max(np.abs(air))
    fade = np.clip(t / 1.5, 0, 1) * np.clip((t[-1] - t) / 2.0, 0, 1)
    return (drone * 0.5 + air * 0.35) * fade * db(-36)


def events_from_comp() -> tuple[list[tuple[float, str]], float]:
    with sync_playwright() as p:
        b = p.chromium.launch(channel="chrome", headless=True)
        page = b.new_page()
        page.goto((HERE / "comp.html").resolve().as_uri())
        page.wait_for_function("window.__ready === true", timeout=120_000)
        ev = page.evaluate("window.__events()")
        dur = page.evaluate("window.__duration")
        b.close()
    return [(float(t), str(k)) for t, k in ev], float(dur)


def build(out: Path) -> None:
    events, dur = events_from_comp()
    n = int((dur + 0.5) * SR)
    dry = np.zeros(n)
    for t0, kind in events:
        s = PALETTE[kind]()
        i = int(t0 * SR)
        dry[i : i + len(s)] += s[: max(0, n - i)]
    ir = room()
    size = 1 << int(np.ceil(np.log2(n + len(ir))))
    wet = np.fft.irfft(np.fft.rfft(dry, size) * np.fft.rfft(ir, size), size)[:n]
    mix = dry * 0.82 + wet * 0.30 + bed(n)
    # gentle stereo: slightly delayed, decorrelated right channel for width
    d = int(0.011 * SR)
    left = mix
    right = np.concatenate([mix[:d] * 0.0, mix[:-d]]) * 0.35 + mix * 0.65
    stereo = np.stack([left, right], axis=1)
    stereo *= db(-3) / np.max(np.abs(stereo))           # peak at -3 dBFS
    end_fade = np.clip((n - np.arange(n)) / (0.6 * SR), 0, 1)
    stereo *= end_fade[:, None]
    out.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((stereo * 32767).astype("<i2").tobytes())
    print(f"{len(events)} cues, {dur:.1f}s -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "build" / "sfx.wav")
    build(ap.parse_args().out)
