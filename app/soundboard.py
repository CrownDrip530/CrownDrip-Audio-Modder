"""
soundboard.py
Decodes and caches MP3/WAV files (resampled ONCE to the live output
stream's actual rate/channels using a proper polyphase resampler), then
exposes a read_block() method that the single output audio callback in
audio_engine.py calls to mix soundboard playback directly into the same
stream as mic passthrough.

DEFENSIVE DECODING: some encoders/tools (including several online tone
generators) produce WAV files with slightly nonstandard headers or an
incomplete trailing frame. If we blindly trust the reported channel count
and force-reshape the decoded sample array into that many channels, an
odd/mismatched total sample count causes either a hard crash ("cannot
reshape array of size X into shape (2)") or, worse, silent channel
misalignment that can sound like garbling/wobbling. We now verify the
total sample count actually divides evenly by the reported channel count,
and trim any leftover partial frame before reshaping, so this is always
safe regardless of how the source file was encoded.

WHY A SINGLE SHARED OUTPUT STREAM (not one stream per sound):
Python's GIL means multiple independent real-time audio callback threads
fight over the same lock roughly every ~20ms. Mixing soundboard playback
into the SAME output callback that handles mic passthrough (via
read_block(), cheap numpy slicing/summation) avoids that entirely.

Playback is tracked per sound_id so pressing Play again on a sound that's
already playing RESTARTS it instead of stacking an overlapping copy.
"""

import math
import threading
import numpy as np
from pydub import AudioSegment
from pathlib import Path

from effects import GainEffect, DeepFryEffect


def _high_quality_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample (frames, channels) float32 audio using scipy's polyphase
    filter (resample_poly). Runs once per unique (file, rate) combo, then
    cached -- negligible cost even for ratios like 44100->48000."""
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
    return np.stack(resampled_channels, axis=1).astype(np.float32)


class SoundboardPlayer:
    def __init__(self, samplerate: int = 48000, channels: int = 2):
        self.samplerate = samplerate
        self.channels = channels
        self._lock = threading.Lock()
        self._active_clips = []
        self._cache = {}

        self.gain = GainEffect(enabled=True, gain_db=0.0)
        self.deep_fry = DeepFryEffect(enabled=False)

    def _decode(self, filepath: Path) -> np.ndarray:
        key = (str(filepath), self.samplerate, self.channels)
        if key in self._cache:
            return self._cache[key]

        seg = AudioSegment.from_file(filepath)
        native_rate = seg.frame_rate
        native_channels = max(1, seg.channels)

        raw = np.array(seg.get_array_of_samples()).astype(np.float32)
        raw /= (2 ** (8 * seg.sample_width - 1))

        # DEFENSIVE: verify the total sample count actually divides evenly
        # by the reported channel count. Some encoders (including certain
        # online tone/WAV generators) produce files with a slightly
        # nonstandard header or an incomplete trailing frame, which can
        # cause a mismatch here. If we blindly reshape anyway, it either
        # crashes ("cannot reshape array of size X into shape (N)") or
        # silently misaligns channels, which can sound like
        # garbling/wobbling rather than throwing a clear error. Instead,
        # trim any leftover partial frame so the reshape always succeeds
        # correctly.
        remainder = len(raw) % native_channels
        if remainder != 0:
            raw = raw[:len(raw) - remainder]

        if native_channels > 1:
            samples = raw.reshape((-1, native_channels))
        else:
            samples = raw.reshape((-1, 1))

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

            out = self.gain.process(out, self.samplerate)
            if self.deep_fry.enabled:
                out = self.deep_fry.process(out, self.samplerate)

        np.clip(out, -1.0, 1.0, out=out)
        return out

    def set_gain_db(self, gain_db: float):
        with self._lock:
            self.gain.params["gain_db"] = gain_db

    def set_deep_fry_enabled(self, enabled: bool):
        with self._lock:
            self.deep_fry.enabled = enabled

    def update_deep_fry_params(self, **kwargs):
        with self._lock:
            self.deep_fry.params.update(kwargs)
