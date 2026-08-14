"""Music/sound intelligence (Spec §17).

Implemented from first principles on numpy+scipy so the system carries no
unavailable dependency:

  onset envelope : half-wave-rectified spectral flux on a mel-ish band grouping
  tempo          : autocorrelation of the onset envelope over 60-190 BPM
  beat grid      : phase search maximising onset energy on the grid, then
                   refined by snapping each beat to the nearest local onset peak
  downbeats      : best of 4 phases by summed onset strength (4/4 assumption,
                   recorded as an assumption in the result)
  energy / RMS   : short-window loudness curve
  sections       : novelty peaks on a self-similarity matrix of band energies
  drops          : large sustained low-band energy jumps following a lull

Everything downstream (cut placement, transition timing, intensity curve) is
driven by these measurements, so the edit is genuinely synchronised to the music.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field, asdict
from scipy import signal, ndimage

from .decode import decode_pcm

HOP = 512
NFFT = 2048


@dataclass
class AudioAnalysis:
    duration: float = 0.0
    sr: int = 22050
    bpm: float = 0.0
    bpm_confidence: float = 0.0
    beats: list = field(default_factory=list)        # seconds
    downbeats: list = field(default_factory=list)    # seconds
    beat_interval: float = 0.0
    onset_times: list = field(default_factory=list)
    energy_times: list = field(default_factory=list)
    energy: list = field(default_factory=list)       # 0..1
    sections: list = field(default_factory=list)     # [{start,end,energy,label}]
    drops: list = field(default_factory=list)        # seconds
    silence: list = field(default_factory=list)      # [{start,end}]
    assumptions: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    def energy_at(self, t: float) -> float:
        if not self.energy:
            return 0.5
        i = int(np.searchsorted(self.energy_times, t))
        return float(self.energy[min(max(i, 0), len(self.energy) - 1)])

    def nearest_beat(self, t: float) -> float:
        if not self.beats:
            return t
        arr = np.asarray(self.beats)
        return float(arr[int(np.argmin(np.abs(arr - t)))])

    def beats_in(self, start: float, end: float) -> list:
        return [b for b in self.beats if start <= b <= end]


def _stft_mag(y, sr):
    f, t, Z = signal.stft(y, fs=sr, nperseg=NFFT, noverlap=NFFT - HOP,
                          window="hann", padded=False, boundary=None)
    return f, t, np.abs(Z)


def _band_energies(f, S, n_bands=24):
    """Group linear FFT bins into log-spaced bands (mel-like, no external dep)."""
    fmin, fmax = 40.0, min(8000.0, f[-1])
    edges = np.logspace(np.log10(fmin), np.log10(fmax), n_bands + 1)
    out = np.zeros((n_bands, S.shape[1]), dtype=np.float32)
    for i in range(n_bands):
        m = (f >= edges[i]) & (f < edges[i + 1])
        if m.any():
            out[i] = S[m].mean(0)
    return out, edges


def analyze_audio(path, sr: int = 22050) -> AudioAnalysis:
    y, sr = decode_pcm(path, sr)
    a = AudioAnalysis(sr=sr, duration=len(y) / sr)
    if len(y) < sr // 2:
        return a

    f, times, S = _stft_mag(y, sr)
    bands, edges = _band_energies(f, S)
    logb = np.log1p(bands * 100.0)

    # ---------- onset envelope: half-wave rectified spectral flux ----------
    flux = np.diff(logb, axis=1, prepend=logb[:, :1])
    onset = np.maximum(flux, 0).sum(0)
    onset = onset - ndimage.uniform_filter1d(onset, size=32)
    onset = np.maximum(onset, 0)
    if onset.max() > 0:
        onset /= onset.max()
    frame_rate = sr / HOP

    # ---------- tempo via autocorrelation ----------
    ac = np.correlate(onset - onset.mean(), onset - onset.mean(), mode="full")
    ac = ac[len(ac) // 2:]
    lo = int(frame_rate * 60.0 / 190.0)      # 190 BPM
    hi = int(frame_rate * 60.0 / 60.0)       # 60 BPM
    hi = min(hi, len(ac) - 1)
    if hi > lo + 2:
        seg = ac[lo:hi]
        # prefer stronger, lower-lag (faster) peaks slightly to avoid half-tempo
        peak = int(np.argmax(seg * np.linspace(1.06, 1.0, len(seg)))) + lo
        a.bpm = float(60.0 * frame_rate / peak)
        a.bpm_confidence = float(np.clip(seg.max() / (np.abs(ac[lo:hi]).mean() * 6 + 1e-9), 0, 1))
        a.beat_interval = 60.0 / a.bpm

        # ---------- phase search ----------
        period = peak
        best_phase, best_score = 0, -1e9
        for ph in range(period):
            idx = np.arange(ph, len(onset), period)
            sc = onset[idx].sum()
            if sc > best_score:
                best_score, best_phase = sc, ph
        grid = np.arange(best_phase, len(onset), period)

        # ---------- snap each beat to nearest local onset peak ----------
        tol = max(1, int(period * 0.14))
        beats = []
        for g in grid:
            s, e = max(0, g - tol), min(len(onset), g + tol + 1)
            if e > s:
                beats.append((s + int(np.argmax(onset[s:e]))) / frame_rate)
        a.beats = [float(b) for b in beats]

        # ---------- downbeats: strongest of 4 phases ----------
        if len(a.beats) >= 8:
            strengths = []
            for ph in range(4):
                idx = [int(b * frame_rate) for b in a.beats[ph::4]]
                idx = [i for i in idx if 0 <= i < len(onset)]
                strengths.append(onset[idx].sum() if idx else 0.0)
            dph = int(np.argmax(strengths))
            a.downbeats = a.beats[dph::4]
            a.assumptions.append("4/4 time assumed for downbeat inference")

    # ---------- onsets ----------
    pk, _ = signal.find_peaks(onset, height=max(0.12, float(onset.mean() * 2.2)),
                              distance=max(1, int(frame_rate * 0.08)))
    a.onset_times = [float(p / frame_rate) for p in pk]

    # ---------- energy curve ----------
    win = int(sr * 0.10)
    if win > 1:
        n = len(y) // win
        rms = np.sqrt(np.array([np.mean(y[i * win:(i + 1) * win] ** 2) for i in range(n)]) + 1e-12)
        rms_db = 20 * np.log10(rms + 1e-9)
        lo_db, hi_db = np.percentile(rms_db, [5, 98])
        norm = np.clip((rms_db - lo_db) / max(hi_db - lo_db, 1e-6), 0, 1)
        norm = ndimage.uniform_filter1d(norm, size=5)
        a.energy = [float(v) for v in norm]
        a.energy_times = [float(i * win / sr) for i in range(n)]
        # silence
        sil, cur = [], None
        for t, v in zip(a.energy_times, a.energy):
            if v < 0.08 and cur is None:
                cur = t
            elif v >= 0.08 and cur is not None:
                if t - cur > 0.25:
                    sil.append({"start": cur, "end": t})
                cur = None
        if cur is not None:
            sil.append({"start": cur, "end": a.duration})
        a.silence = sil

    # ---------- sections via self-similarity novelty ----------
    B = logb / (np.linalg.norm(logb, axis=0, keepdims=True) + 1e-9)
    step = max(1, B.shape[1] // 400)
    Bs = B[:, ::step]
    Ssim = Bs.T @ Bs
    L = 16
    if Ssim.shape[0] > 4 * L:
        kern = np.zeros((2 * L, 2 * L), np.float32)
        kern[:L, :L] = 1; kern[L:, L:] = 1; kern[:L, L:] = -1; kern[L:, :L] = -1
        kern *= np.outer(signal.windows.gaussian(2 * L, L / 2),
                         signal.windows.gaussian(2 * L, L / 2))
        nov = np.zeros(Ssim.shape[0], np.float32)
        for i in range(L, Ssim.shape[0] - L):
            nov[i] = (Ssim[i - L:i + L, i - L:i + L] * kern).sum()
        nov = np.maximum(nov - ndimage.uniform_filter1d(nov, 64), 0)
        if nov.max() > 0:
            nov /= nov.max()
        sp, _ = signal.find_peaks(nov, height=0.25, distance=max(2, int(4.0 * frame_rate / step)))
        bounds = [0.0] + [float(times[min(p * step, len(times) - 1)]) for p in sp] + [a.duration]
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            if e - s < 1.0:
                continue
            eng = float(np.mean([a.energy_at(t) for t in np.linspace(s, e, 8)]))
            label = "high" if eng > 0.66 else "low" if eng < 0.33 else "mid"
            a.sections.append({"start": s, "end": e, "energy": eng, "label": label})

    # ---------- drops: sustained low-band jump after a lull ----------
    low = logb[:6].mean(0)
    low = ndimage.uniform_filter1d(low, 8)
    if len(low) > 40:
        d = np.diff(low, prepend=low[:1])
        thr = float(np.percentile(d, 99.2))
        dp, _ = signal.find_peaks(d, height=max(thr, 1e-6),
                                  distance=max(2, int(frame_rate * 3)))
        for p in dp:
            pre = low[max(0, p - int(frame_rate)):p].mean()
            post = low[p:p + int(frame_rate)].mean()
            if post > pre * 1.22:
                a.drops.append(float(times[min(p, len(times) - 1)]))
    return a
