"""
soundboard.py
Manages loading mp3s into numpy arrays and playing them mixed into the
outgoing audio stream. Uses pydub to decode mp3 -> PCM.

Playback is tracked per sound_id so that pressing Play again on a sound
that's already playing RESTARTS it instead of stacking a second overlapping
copy (which caused phasing/comb-filter artifacts over time).
"""

import numpy as np
from pydub import AudioSegment
from pathlib import Path
import threading


class SoundboardPlayer:
    def __init__(self, samplerate: int = 48000, channels: int = 2):
        self.samplerate = samplerate
        self.channels = channels
        self._lock = threading.Lock()
        self._active_clips = []
        self._cache = {}

    def _decode(self, filepath: Path) -> np.ndarray:
        key = str(filepath)
        if key in self._cache:
            return self._cache[key]

        seg = AudioSegment.from_file(filepath)
        seg = seg.set_frame_rate(self.samplerate).set_channels(self.channels)
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        samples /= (2 ** (8 * seg.sample_width - 1))

        if self.channels > 1:
            samples = samples.reshape((-1, self.channels))
        else:
            samples = samples.reshape((-1, 1))

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
