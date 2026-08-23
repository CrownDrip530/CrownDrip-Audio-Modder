"""
audio_engine.py
Real-time audio engine: captures your real mic, runs it through the
EffectChain, mixes in any playing soundboard clips, and streams the result
out to the virtual cable device.

IMPORTANT: Windows exposes the same physical/virtual device through
multiple audio host APIs at once (old MME/DirectSound, and modern WASAPI).
MME is a legacy compatibility layer that does its own internal resampling
with a lower quality algorithm -- a classic cause of pitch "warble" that
settles in over the first couple seconds. We restrict device selection to
WASAPI only, which reports accurate native sample rates and has much
better real-time timing behavior.
"""

import sounddevice as sd
import numpy as np
import threading
import queue

from effects import EffectChain, GainEffect, DeepFryEffect
from soundboard import SoundboardPlayer

BLOCK_SIZE = 1024
CHANNELS = 2
DEFAULT_SAMPLE_RATE = 48000

TARGET_QUEUE_DEPTH = 4
MAX_QUEUE_DEPTH = 10


def _get_wasapi_hostapi_index():
    for i, api in enumerate(sd.query_hostapis()):
        if "wasapi" in api["name"].lower():
            return i
    return None


class AudioEngine:
    def __init__(self, mic_device=None, output_device=None, monitor_device=None):
        self.mic_device = mic_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.monitor_enabled = False

        self.chain = EffectChain()

        self.soundboard_gain = GainEffect(enabled=True, gain_db=0.0)
        self.soundboard_deep_fry = DeepFryEffect(enabled=False)

        self.input_rate = DEFAULT_SAMPLE_RATE
        self.output_rate = DEFAULT_SAMPLE_RATE

        self.soundboard = SoundboardPlayer(samplerate=DEFAULT_SAMPLE_RATE, channels=CHANNELS)

        self._input_stream = None
        self._output_stream = None
        self._lock = threading.Lock()
        self._running = False
        self._buffer = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        self._last_block = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    @staticmethod
    def list_input_devices():
        """Only lists WASAPI input devices -- avoids picking up legacy
        MME/DirectSound duplicates that cause resampling artifacts."""
        wasapi_idx = _get_wasapi_hostapi_index()
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                if wasapi_idx is None or d["hostapi"] == wasapi_idx:
                    result.append({"index": i, "name": d["name"]})
        return result

    @staticmethod
    def list_output_devices():
        """Only lists WASAPI output devices -- avoids picking up legacy
        MME/DirectSound duplicates that cause resampling artifacts."""
        wasapi_idx = _get_wasapi_hostapi_index()
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0:
                if wasapi_idx is None or d["hostapi"] == wasapi_idx:
                    result.append({"index": i, "name": d["name"]})
        return result

    @staticmethod
    def find_vb_cable_output_index():
        for d in AudioEngine.list_output_devices():
            if "CABLE Input" in d["name"]:
                return d["index"]
        return None

    @staticmethod
    def _get_device_default_rate(device_index, fallback=DEFAULT_SAMPLE_RATE):
        try:
            info = sd.query_devices(device_index)
            rate = info.get("default_samplerate")
            if rate and rate > 0:
                return int(round(rate))
        except Exception:
            pass
        return fallback

    def _input_callback(self, indata, frames, time_info, status):
        with self._lock:
            mic_signal = indata.copy()
            if mic_signal.shape[1] == 1 and CHANNELS == 2:
                mic_signal = np.repeat(mic_signal, 2, axis=1)
            processed = self.chain.process(mic_signal, self.input_rate)

        while self._buffer.qsize() > TARGET_QUEUE_DEPTH:
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break

        try:
            self._buffer.put_nowait(processed)
        except queue.Full:
            try:
                self._buffer.get_nowait()
                self._buffer.put_nowait(processed)
            except queue.Empty:
                pass

    def _output_callback(self, outdata, frames, time_info, status):
        try:
            block = self._buffer.get_nowait()
            self._last_block = block
        except queue.Empty:
            block = self._last_block

        if block.shape[0] != frames:
            fixed = np.zeros((frames, CHANNELS), dtype=np.float32)
            n = min(frames, block.shape[0])
            fixed[:n] = block[:n]
            block = fixed

        sb_block = self.soundboard.read_block(frames)

        with self._lock:
            sb_block = self.soundboard_gain.process(sb_block, self.output_rate)
            if self.soundboard_deep_fry.enabled:
                sb_block = self.soundboard_deep_fry.process(sb_block, self.output_rate)

        mixed = block + sb_block
        mixed = self._soft_limit(mixed)
        outdata[:] = mixed

    @staticmethod
    def _soft_limit(x: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        out = np.copy(x)
        over_mask = np.abs(x) > threshold
        if np.any(over_mask):
            sign = np.sign(x[over_mask])
            excess = np.abs(x[over_mask]) - threshold
            compressed = threshold + (1.0 - threshold) * np.tanh(excess / (1.0 - threshold))
            out[over_mask] = sign * compressed
        return np.clip(out, -1.0, 1.0)

    def start(self):
        if self._running:
            return

        self.input_rate = self._get_device_default_rate(self.mic_device)
        self.output_rate = self._get_device_default_rate(self.output_device)

        self.soundboard = SoundboardPlayer(samplerate=self.output_rate, channels=CHANNELS)

        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break
        self._last_block = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

        self._input_stream = sd.InputStream(
            device=self.mic_device,
            channels=CHANNELS,
            samplerate=self.input_rate,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            device=self.output_device,
            channels=CHANNELS,
            samplerate=self.output_rate,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._output_callback,
        )

        self._input_stream.start()
        self._output_stream.start()
        self._running = True

    def stop(self):
        if self._input_stream is not None:
            self._input_stream.stop()
            self._input_stream.close()
            self._input_stream = None
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None
        self._running = False

    def is_running(self):
        return self._running

    def set_mic_gain_db(self, gain_db: float):
        with self._lock:
            self.chain.set_mic_gain_db(gain_db)

    def set_deep_fry_enabled(self, enabled: bool):
        with self._lock:
            self.chain.set_deep_fry_enabled(enabled)

    def update_deep_fry_params(self, **kwargs):
        with self._lock:
            self.chain.deep_fry.params.update(kwargs)

    def set_soundboard_gain_db(self, gain_db: float):
        with self._lock:
            self.soundboard_gain.params["gain_db"] = gain_db

    def set_soundboard_deep_fry_enabled(self, enabled: bool):
        with self._lock:
            self.soundboard_deep_fry.enabled = enabled

    def update_soundboard_deep_fry_params(self, **kwargs):
        with self._lock:
            self.soundboard_deep_fry.params.update(kwargs)

    def play_sound(self, filepath, volume=1.0, sound_id=None):
        self.soundboard.play(filepath, volume, sound_id=sound_id)

    def stop_sound(self, sound_id):
        self.soundboard.stop_sound(sound_id)

    def stop_all_sounds(self):
        self.soundboard.stop_all()

    def to_config_dict(self) -> dict:
        d = self.chain.to_config_dict()
        d["soundboard_volume_db"] = self.soundboard_gain.params.get("gain_db", 0.0)
        d["effects"]["soundboard_deep_fry"] = self.soundboard_deep_fry.to_dict()
        return d

    def load_config_dict(self, config: dict):
        self.chain.load_config_dict(config)
        self.soundboard_gain.params["gain_db"] = config.get("soundboard_volume_db", 0.0)
        sfx_cfg = config.get("effects", {}).get("soundboard_deep_fry", {})
        self.soundboard_deep_fry.load_dict(sfx_cfg)
