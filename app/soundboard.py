"""
soundboard.py
Manages loading mp3s into numpy arrays and playing them mixed into the
outgoing audio stream. Uses pydub (ffmpeg) purely for DECODING, then does
our own high-quality resampling with scipy.signal.resample_poly instead of
relying on pydub's built-in frame rate conversion (a crude linear
interpolation resampler that causes audible wobble).

NOTE: an earlier version of this file had a "safety cap" that fell back to
a cheap interpolation resampler whenever the up/down conversion factors
were "too large". Unfortunately the single most common real-world case --
44100 Hz (typical mp3) -> 48000 Hz (typical virtual cable rate) -- reduces
to up=160, down=147, which tripped that cap every time. That meant the
high-quality resampler was silently never actually used, and the same
wobble bug persisted. The cap has been removed: resample_poly is only run
once per unique file (result is cached), so there's no real performance
concern even with larger factors (a full 3-minute song resamples in well
under half a second, one time only).

Playback is tracked per sound_id so that pressing Play again on a sound
that's already playing RESTARTS it instead of stacking a second overlapping
copy (which previously caused phasing artifacts).
"""

import math
import numpy as np
from pydub import AudioSegment
from pathlib import Path
import threading


def _high_quality_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample (frames, channels) float32 audio from src_rate to dst_rate
    using a proper polyphase filter (scipy.signal.resample_poly). Always
    used regardless of the size of the up/down factors -- this only runs
    once per unique file (cached afterward), so it's cheap enough even for
    "unfriendly" rate ratios like 44100->48000 (up=160, down=147)."""
    if src_rate == dst_rate:
        return samples

    from scipy.signal import resample_poly

    g = math.gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g

    resampled_channels = [
        resample_poly(samples[:, ch], up, down)
        for ch in range(samples.shape[1])
    ]
    out = np.stack(resampled_channels, axis=1).astype(np.float32)
    return out


class SoundboardPlayer:
    def __init__(self, samplerate: int = 48000, channels: int = 2):
        self.samplerate = samplerate
        self.channels = channels
        self._lock = threading.Lock()
        self._active_clips = []
        self._cache = {}

    def _decode(self, filepath: Path) -> np.ndarray:
        key = (str(filepath), self.samplerate, self.channels)
        if key in self._cache:
            return self._cache[key]

        seg = AudioSegment.from_file(filepath)
        native_rate = seg.frame_rate
        native_channels = seg.channels

        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (2 ** (8 * seg.sample_width - 1))

        if native_channels > 1:
            samples = samples.reshape((-1, native_channels))
        else:
            samples = samples.reshape((-1, 1))

        if native_channels != self.channels:
            if native_channels == 1 and self.channels == 2:
                samples = np.repeat(samples, 2, axis=1)
            elif native_channels == 2 and self.channels == 1:
                samples = samples.mean(axis=1, keepdims=True)
            else:
                fixed = np.zeros((samples.shape[0], self.channels), dtype=np.float32)
                n = min(samples.shape[1], self.channels)
                fixed[:, :n] = samples[:, :n]
                samples = fixed

        if native_rate != self.samplerate:
            samples = _high_quality_resample(samples, native_rate, self.samplerate)

        samples = np.ascontiguousarray(samples.astype(np.float32))
        self._cache[key] = samples
        return samples

    def play(self, filepath: Path, volume: float = 1.0, sound_id: str = None):
        audio = self._decode(filepath)
        with self._lock:
            if sound_id is not None:
                self._active_clips = [c for c in self._active_clips if c["id"] != sound_id]
            self._active_clips.append({
                "id": sound_id,
                "audio": audio,
                "pos": 0,
                "volume": volume
            })

    def stop_all(self):
        with self._lock:
            self._active_clips.clear()

    def stop_sound(self, sound_id: str):
        with self._lock:
            self._active_clips = [c for c in self._active_clips if c["id"] != sound_id]

    def is_playing(self) -> bool:
        with self._lock:
            return len(self._active_clips) > 0

    def read_block(self, num_frames: int) -> np.ndarray:
        out = np.zeros((num_frames, self.channels), dtype=np.float32)
        with self._lock:
            still_active = []
            for clip in self._active_clips:
                audio = clip["audio"]
                pos = clip["pos"]
                remaining = len(audio) - pos
                take = min(num_frames, remaining)
                if take > 0:
                    out[:take] += audio[pos:pos + take] * clip["volume"]
                    clip["pos"] += take
                if clip["pos"] < len(audio):
                    still_active.append(clip)
            self._active_clips = still_active

        np.clip(out, -1.0, 1.0, out=out)
        return out
