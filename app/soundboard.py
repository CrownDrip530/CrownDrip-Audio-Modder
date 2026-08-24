"""
soundboard.py
Plays MP3 soundboard clips by opening a dedicated sounddevice OutputStream
DIRECTLY to the virtual cable device, at the file's own native sample rate
and channel count. This lets Windows' own WASAPI shared-mode audio engine
handle sample-rate conversion and mixing with the mic passthrough stream,
instead of us doing our own resampling/mixing in Python.

This is what actually fixes the "wobbly after a few seconds" MP3 bug for
good -- WASAPI shared mode natively supports multiple simultaneous streams
to the same device (this is how e.g. Discord + a game + a Windows
notification sound can all play through the same speaker at once), so mic
passthrough and soundboard playback mix together correctly at the OS
level with zero custom resampling code required.

Each sound plays on a short-lived independent stream that closes itself
automatically when the clip finishes. Pressing Play again on a sound
that's already playing restarts it (stops the old stream, starts a fresh
one) instead of stacking an overlapping copy.
"""

import threading
import uuid
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pathlib import Path

from effects import GainEffect, DeepFryEffect


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

    def _decode(self, filepath: Path):
        key = str(filepath)
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

        samples = np.ascontiguousarray(samples)
        self._cache[key] = (samples, native_rate, native_channels)
        return self._cache[key]

    def _resolve_playback_channels(self, native_channels: int) -> int:
        try:
            info = sd.query_devices(self.output_device)
            max_out = int(info.get("max_output_channels", native_channels) or native_channels)
        except Exception:
            max_out = native_channels

        if max_out <= 0:
            return native_channels
        if native_channels == 1 and max_out >= 2:
            return 2
        return min(native_channels, max_out)

    @staticmethod
    def _convert_channels(samples: np.ndarray, native_channels: int, target_channels: int) -> np.ndarray:
        if native_channels == target_channels:
            return samples
        if native_channels == 1 and target_channels == 2:
            return np.repeat(samples, 2, axis=1)
        if native_channels == 2 and target_channels == 1:
            return samples.mean(axis=1, keepdims=True)
        fixed = np.zeros((samples.shape[0], target_channels), dtype=np.float32)
        n = min(samples.shape[1], target_channels)
        fixed[:, :n] = samples[:, :n]
        return fixed

    def play(self, filepath: Path, volume: float = 1.0, sound_id: str = None):
        samples, native_rate, native_channels = self._decode(filepath)

        key = sound_id if sound_id is not None else str(uuid.uuid4())
        if sound_id is not None:
            self.stop_sound(sound_id)

        playback_channels = self._resolve_playback_channels(native_channels)
        data = self._convert_channels(samples, native_channels, playback_channels)

        pos = {"i": 0}

        def callback(outdata, frames, time_info, status):
            start = pos["i"]
            end = start + frames
            chunk = data[start:end]

            finished = False
            if len(chunk) < frames:
                fixed = np.zeros((frames, playback_channels), dtype=np.float32)
                fixed[:len(chunk)] = chunk
                chunk = fixed
                finished = True

            processed = chunk * volume
            with self._lock:
                processed = self.gain.process(processed, native_rate)
                if self.deep_fry.enabled:
                    processed = self.deep_fry.process(processed, native_rate)

            outdata[:] = np.clip(processed, -1.0, 1.0)
            pos["i"] = end

            if finished:
                raise sd.CallbackStop()

        stream = sd.OutputStream(
            device=self.output_device,
            channels=playback_channels,
            samplerate=native_rate,
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
