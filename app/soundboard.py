"""
soundboard.py
Decodes and caches MP3/WAV files, then exposes read_block() for mixing
into the shared output audio callback.

WAV files decode via Python's built-in `wave` module (no ffmpeg
subprocess -- avoids the console-flash + multi-second delay problem).
MP3/other formats still go through pydub/ffmpeg since there's no
pure-Python MP3 decoder built in.

Tracks the last file's native rate/channels vs the live device's
rate/channels so the GUI can display real diagnostic info instead of
guessing at causes.
"""

import math
import wave
import threading
import numpy as np
from pathlib import Path

from effects import GainEffect, DeepFryEffect


def _high_quality_resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
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


def _decode_wav_native(filepath: Path):
    """Decode a .wav file using Python's built-in wave module -- no
    ffmpeg subprocess involved at all."""
    with wave.open(str(filepath), 'rb') as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_bytes = wf.readframes(n_frames)

    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128) / 128.0
    elif sample_width == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    remainder = len(samples) % n_channels
    if remainder != 0:
        samples = samples[:len(samples) - remainder]

    samples = samples.reshape((-1, n_channels))
    return samples, framerate, n_channels


def _decode_via_pydub(filepath: Path):
    """Fallback decoder for non-WAV formats (mp3, ogg, etc.) that still
    require ffmpeg via pydub."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(filepath)
    native_rate = seg.frame_rate
    native_channels = max(1, seg.channels)

    raw = np.array(seg.get_array_of_samples()).astype(np.float32)
    raw /= (2 ** (8 * seg.sample_width - 1))

    remainder = len(raw) % native_channels
    if remainder != 0:
        raw = raw[:len(raw) - remainder]

    if native_channels > 1:
        samples = raw.reshape((-1, native_channels))
    else:
        samples = raw.reshape((-1, 1))

    return samples, native_rate, native_channels


class SoundboardPlayer:
    def __init__(self, samplerate: int = 48000, channels: int = 2):
        self.samplerate = samplerate
        self.channels = channels
        self._lock = threading.Lock()
        self._active_clips = []
        self._cache = {}
        self._decode_info_cache = {}  # cache_key -> (native_rate, native_channels)

        self.gain = GainEffect(enabled=True, gain_db=0.0)
        self.deep_fry = DeepFryEffect(enabled=False)

        # Diagnostics: last decoded file's native format vs live device
        # format, so the GUI can show real evidence instead of guesses.
        self._last_native_rate = None
        self._last_native_channels = None

    def _decode(self, filepath: Path) -> np.ndarray:
        key = (str(filepath), self.samplerate, self.channels)

        if key in self._cache:
            native_rate, native_channels = self._decode_info_cache.get(key, (None, None))
            self._last_native_rate = native_rate
            self._last_native_channels = native_channels
            return self._cache[key]

        if filepath.suffix.lower() == ".wav":
            samples, native_rate, native_channels = _decode_wav_native(filepath)
        else:
            samples, native_rate, native_channels = _decode_via_pydub(filepath)

        self._last_native_rate = native_rate
        self._last_native_channels = native_channels

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
        self._decode_info_cache[key] = (native_rate, native_channels)

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

    def get_last_decode_info(self):
        return {
            "native_rate": self._last_native_rate,
            "native_channels": self._last_native_channels,
            "device_rate": self.samplerate,
            "device_channels": self.channels,
        }
