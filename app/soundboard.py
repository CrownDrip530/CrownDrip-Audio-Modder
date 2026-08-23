"""
soundboard.py
Manages loading mp3s into numpy arrays and playing them mixed into the
outgoing audio stream. Uses pydub (ffmpeg) purely for DECODING, then does
our own high-quality resampling with scipy instead of relying on pydub's
built-in frame rate conversion (which uses Python's old `audioop` module --
a crude linear-interpolation resampler that introduces audible wobble/
warping artifacts, especially when the source rate doesn't cleanly divide
into the target rate, e.g. 44100 -> 48000).

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
    using a proper polyphase filter (scipy.signal.resample_poly), which is
    far cleaner than simple linear interpolation and avoids the "wobbly"
    artifacts that come from lower quality resamplers."""
    if src_rate == dst_rate:
        return samples

    from scipy.signal import resample_poly

    g = math.gcd(src_rate, dst_rate)
    up = dst_rate // g
    down = src_rate // g

    if up > 32 or down > 32:
        out = np.empty((int(round(len(samples) * dst_rate / src_rate)), samples.shape[1]), dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=len(out), endpoint=False)
        for ch in range(samples.shape[1]):
            out[:, ch] = np.interp(x_new, x_old, samples[:, ch])
        return out.astype(np.float32)

    resampled_channels = []
    for ch in range(samples.shape[1]):
        resampled_channels.append(resample_poly(samples[:, ch], up, down))
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
