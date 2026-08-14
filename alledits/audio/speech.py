"""Finding speech in a clip's own audio, so music can move out of its way.

This is deliberately NOT transcription. Transcription (Whisper-class) is a
deferred dependency and is unavailable in this build; claiming it would be a
lie. What the mix actually needs is narrower and answerable from signal: *when
is someone talking?* — which is a voice-activity question, not a words question.

Two properties separate speech from a musical bed:

1. **Band energy.** Speech concentrates in roughly 180-3400 Hz.
2. **Syllable-rate modulation.** The envelope of speech fluctuates at about
   2-8 Hz. A sustained note or a pad does not. Band energy alone would flag any
   midrange-heavy music, so the modulation test is what makes this specific.

The output is a list of speaking windows. If the audio cannot be read, the
answer is "no speech detected", never a guess.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..core.ffmpeg import ffmpeg

VOICE_LOW_HZ = 180.0
VOICE_HIGH_HZ = 3400.0
SYLLABLE_LOW_HZ = 2.0
SYLLABLE_HIGH_HZ = 8.0

# A window must clear BOTH tests. Calibrated so a music-only bed scores zero.
BAND_RATIO_MIN = 0.45      # fraction of energy inside the voice band
MODULATION_MIN = 0.14      # syllable-rate energy as a share of envelope energy
MIN_SPEECH_S = 0.35        # shorter than this is a transient, not a phrase
MERGE_GAP_S = 0.45         # gaps below this are pauses within one phrase


@dataclass
class SpeechAnalysis:
    has_speech: bool = False
    windows: list = field(default_factory=list)     # [(start, end)] seconds
    speech_fraction: float = 0.0
    reason: str = ""

    def to_dict(self):
        return asdict(self)


def _decode_mono(path, sr=16000, duration=None) -> np.ndarray | None:
    tmp = Path("/tmp") / f"_vad_{abs(hash((str(path), duration))) % (10 ** 10)}.raw"
    args = ["-i", str(path)]
    if duration:
        args = ["-t", f"{duration:.3f}"] + args
    try:
        ffmpeg([*args, "-vn", "-ac", "1", "-ar", str(sr),
                "-f", "f32le", str(tmp)])
        data = np.fromfile(tmp, dtype="<f4")
        return data if data.size else None
    except Exception:
        return None
    finally:
        tmp.unlink(missing_ok=True)


def detect_speech(path, duration: float | None = None,
                  sr: int = 16000, hop_s: float = 0.02) -> SpeechAnalysis:
    """Locate speaking windows in an audio or video file."""
    x = _decode_mono(path, sr=sr, duration=duration)
    if x is None or x.size < sr // 4:
        return SpeechAnalysis(reason="no readable audio stream")

    hop = max(1, int(sr * hop_s))
    win = hop * 2
    n = (len(x) - win) // hop
    if n <= 4:
        return SpeechAnalysis(reason="audio too short to analyse")

    frames = np.lib.stride_tricks.as_strided(
        x, shape=(n, win), strides=(x.strides[0] * hop, x.strides[0])).copy()
    frames *= np.hanning(win)
    spec = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1.0 / sr)

    band = (freqs >= VOICE_LOW_HZ) & (freqs <= VOICE_HIGH_HZ)
    total = spec.sum(axis=1) + 1e-9
    band_ratio = spec[:, band].sum(axis=1) / total
    envelope = spec[:, band].sum(axis=1)

    # Syllable-rate content of the envelope, over a ~1 s sliding window.
    fps_env = 1.0 / hop_s
    seg = max(8, int(fps_env))
    modulation = np.zeros(n)
    for i in range(0, n, seg // 2 or 1):
        chunk = envelope[i:i + seg]
        if len(chunk) < 8:
            break
        c = chunk - chunk.mean()
        if not np.any(c):
            continue
        E = np.abs(np.fft.rfft(c))
        f = np.fft.rfftfreq(len(c), hop_s)
        syl = (f >= SYLLABLE_LOW_HZ) & (f <= SYLLABLE_HIGH_HZ)
        share = E[syl].sum() / (E.sum() + 1e-9)
        modulation[i:i + seg] = np.maximum(modulation[i:i + seg], share)

    # Silence must not qualify on ratio alone: a near-silent frame can have a
    # high band ratio simply because there is nothing outside the band either.
    loud = envelope > max(envelope.max() * 0.06, 1e-6)
    voiced = (band_ratio >= BAND_RATIO_MIN) & (modulation >= MODULATION_MIN) & loud

    windows = _to_windows(voiced, hop_s)
    windows = _merge(windows, MERGE_GAP_S)
    windows = [(a, b) for a, b in windows if b - a >= MIN_SPEECH_S]

    dur = n * hop_s
    frac = sum(b - a for a, b in windows) / dur if dur > 0 else 0.0
    if not windows:
        return SpeechAnalysis(reason=(
            "no window passed both the voice-band and syllable-rate tests; "
            "midrange-heavy music fails the second"))
    return SpeechAnalysis(
        has_speech=True, windows=windows, speech_fraction=float(frac),
        reason=(f"{len(windows)} speaking window(s) covering {frac * 100:.0f}% "
                f"of the audio, by voice-band energy with syllable-rate "
                f"modulation"))


def _to_windows(mask, hop_s):
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        out.append((start * hop_s, len(mask) * hop_s))
    return out


def _merge(windows, gap):
    if not windows:
        return []
    out = [list(windows[0])]
    for a, b in windows[1:]:
        if a - out[-1][1] <= gap:
            out[-1][1] = b
        else:
            out.append([a, b])
    return [(round(a, 3), round(b, 3)) for a, b in out]
