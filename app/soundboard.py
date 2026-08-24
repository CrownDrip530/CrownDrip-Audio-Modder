"""
soundboard.py
Plays MP3 soundboard clips by opening a dedicated sounddevice OutputStream
DIRECTLY to the virtual cable device. This lets Windows' own WASAPI
shared-mode audio engine mix soundboard playback together with the mic
passthrough stream at the OS level.

IMPORTANT: VB-Cable's WASAPI endpoint only accepts ONE specific sample
rate (whatever Windows has it configured to -- almost always 48000Hz) and
will immediately fail with PaErrorCode -9997 "Invalid sample rate" if you
try to open a stream at a different rate (e.g. a typical mp3's native
44100Hz). So we can't just open the stream at the file's native rate.

Fix: query the output device's actual accepted default sample rate, and
resample each mp3 to that rate ONCE at decode time (cached afterward)
using a proper polyphase resampler (scipy.signal.resample_poly), then open
the playback stream at that same device rate. This avoids both the
"invalid sample rate" error AND the wobble caused by cheap resamplers.

Each sound plays on a short-lived independent stream that closes itself
automatically when the clip finishes. Pressing Play again on a sound
that's already playing restarts it (stops the old stream, starts a fresh
one) instead of stacking an overlapping copy.
"""

import math
import threading
import uuid
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pathlib import Path

from effects import GainEffect, DeepFryEffect


def _high_quality_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Resample (frames, channels) float32 audio from src_rate to dst_rate
    using scipy's polyphase filter. Only runs once per unique (file, rate)
    combination -- cached afterward -- so cost is negligible even for
    unfriendly ratios like 44100->48000 (up=160, down=147)."""
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


class _ActiveClip:
    __slots__ = ("key", "stream")

    def __init__(self, key, stream):
        self.key = key
        self.stream = stream


class SoundboardPlayer:
    def __init__(self):
        self._lock = threading.Lock()
        self._active = {}
        self._cache = {}
        self.output_device = None

        self.gain = GainEffect(enabled=True, gain_db=0.0)
        self.deep_fry = DeepFryEffect(enabled=False)

    def _get_output_format(self):
        try:
            info = sd.query_devices(self.output_device)
            rate = int(round(info.get("default_samplerate", 48000) or 48000))
            max_out = int(info.get("max_output_channels", 2) or 2)
            channels = 2 if max_out >= 2 else max(1, max_out)
            return rate, channels
        except Exception:
            return 48000, 2

    def _decode_and_prepare(self, filepath: Path, device_rate: int, device_channels: int) -> np.ndarray:
        cache_key = (str(filepath), device_rate, device_channels)
        if cache_key in self._cache:
            return self._cache[cache_key]

        seg = AudioSegment.from_file(filepath)
        native_rate = seg.frame_rate
        native_channels = seg.channels

        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (2 ** (8 * seg.sample_width - 1))

        if native_channels > 1:
            samples = samples.reshape((-1, native_channels))
        else:
            samples = samples.reshape((-1, 1))

        if native_channels != device_channels:
            if native_channels == 1 and device_channels == 2:
                samples = np.repeat(samples, 2, axis=1)
            elif native_channels == 2 and device_channels == 1:
                samples = samples.mean(axis=1, keepdims=True)
            else:
                fixed = np.zeros((samples.shape[0], device_channels), dtype=np.float32)
                n = min(samples.shape[1], device_channels)
                fixed[:, :n] = samples[:, :n]
                samples = fixed

        if native_rate != device_rate:
            samples = _high_quality_resample(samples, native_rate, device_rate)

        samples = np.ascontiguousarray(samples.astype(np.float32))
        self._cache[cache_key] = samples
        return samples

    def play(self, filepath: Path, volume: float = 1.0, sound_id: str = None):
        device_rate, device_channels = self._get_output_format()
        data = self._decode_and_prepare(filepath, device_rate, device_channels)

        key = sound_id if sound_id is not None else str(uuid.uuid4())
        if sound_id is not None:
            self.stop_sound(sound_id)

        pos = {"i": 0}

        def callback(outdata, frames, time_info, status):
            start = pos["i"]
            end = start + frames
            chunk = data[start:end]

            finished = False
            if len(chunk) < frames:
                fixed = np.zeros((frames, device_channels), dtype=np.float32)
                fixed[:len(chunk)] = chunk
                chunk = fixed
                finished = True

            processed = chunk * volume
            with self._lock:
                processed = self.gain.process(processed, device_rate)
                if self.deep_fry.enabled:
                    processed = self.deep_fry.process(processed, device_rate)

            outdata[:] = np.clip(processed, -1.0, 1.0)
            pos["i"] = end

            if finished:
                raise sd.CallbackStop()

        stream = sd.OutputStream(
            device=self.output_device,
            channels=device_channels,
            samplerate=device_rate,
            dtype="float32",
            callback=callback,
            finished_callback=lambda k=key: self._on_finished(k),
        )

        with self._lock:
            self._active[key] = _ActiveClip(key, stream)

        stream.start()

    def _on_finished(self, key):
        with self._lock:
            clip = self._active.pop(key, None)
        if clip is not None:
            try:
                clip.stream.close()
            except Exception:
                pass

    def stop_sound(self, sound_id):
        with self._lock:
            clip = self._active.pop(sound_id, None)
        if clip is not None:
            try:
                clip.stream.stop()
                clip.stream.close()
            except Exception:
                pass

    def stop_all(self):
        with self._lock:
            clips = list(self._active.values())
            self._active.clear()
        for clip in clips:
            try:
                clip.stream.stop()
                clip.stream.close()
            except Exception:
                pass

    def is_playing(self) -> bool:
        with self._lock:
            return len(self._active) > 0

    def set_gain_db(self, gain_db: float):
        with self._lock:
            self.gain.params["gain_db"] = gain_db

    def set_deep_fry_enabled(self, enabled: bool):
        with self._lock:
            self.deep_fry.enabled = enabled

    def update_deep_fry_params(self, **kwargs):
        with self._lock:
            self.deep_fry.params.update(kwargs)
