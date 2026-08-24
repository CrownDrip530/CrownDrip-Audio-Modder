"""
audio_engine.py
Real-time audio engine.

Mic passthrough: captures your real mic, runs it through the EffectChain,
and streams it out to the virtual cable device via an ElasticBuffer that
bridges the two independently-clocked devices smoothly (no audible warble
from clock drift).

Soundboard (MP3) playback: handled entirely by soundboard.py, which opens
its OWN short-lived stream(s) directly to the virtual cable device at each
file's native rate. Windows' WASAPI shared-mode audio engine natively
mixes multiple simultaneous streams to the same device, so this plays
alongside mic passthrough correctly with zero custom resampling/mixing
code on our end -- this is what actually fixed the "wobbly after a few
seconds" mp3 bug for good, versus manually resampling/mixing it ourselves.
"""

import sounddevice as sd
import numpy as np
import threading

from effects import EffectChain
from soundboard import SoundboardPlayer

BLOCK_SIZE = 1024
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2

TARGET_LATENCY_SEC = 0.12
MAX_LATENCY_SEC = 0.6


def _get_wasapi_hostapi_index():
    for i, api in enumerate(sd.query_hostapis()):
        if "wasapi" in api["name"].lower():
            return i
    return None


class ElasticBuffer:
    def __init__(self, channels: int, samplerate: int):
        self.channels = channels
        self.samplerate = samplerate
        self.target_frames = int(TARGET_LATENCY_SEC * samplerate)
        self.max_frames = int(MAX_LATENCY_SEC * samplerate)
        self._lock = threading.Lock()
        self._data = np.zeros((0, channels), dtype=np.float32)

    def write(self, block: np.ndarray):
        with self._lock:
            self._data = np.concatenate([self._data, block], axis=0)
            if len(self._data) > self.max_frames:
                excess = len(self._data) - self.max_frames
                self._data = self._data[excess:]

    def read(self, num_frames: int) -> np.ndarray:
        with self._lock:
            available = len(self._data)

            if available < 4:
                return np.zeros((num_frames, self.channels), dtype=np.float32)

            error = available - self.target_frames
            correction = np.clip(error / max(self.target_frames, 1) * 0.15, -0.02, 0.02)
            ratio = 1.0 + correction

            src_len_needed = max(2, int(round(num_frames * ratio)))
            src_len_needed = min(src_len_needed, available)

            src = self._data[:src_len_needed]
            self._data = self._data[src_len_needed:]

        if src_len_needed == num_frames:
            return src.astype(np.float32, copy=True)

        x_old = np.linspace(0.0, 1.0, num=src_len_needed, endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=num_frames, endpoint=False)
        out = np.empty((num_frames, self.channels), dtype=np.float32)
        for ch in range(self.channels):
            out[:, ch] = np.interp(x_new, x_old, src[:, ch])
        return out


class AudioEngine:
    def __init__(self, mic_device=None, output_device=None, monitor_device=None):
        self.mic_device = mic_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.monitor_enabled = False

        self.chain = EffectChain()
        self.soundboard = SoundboardPlayer()

        self.input_rate = DEFAULT_SAMPLE_RATE
        self.output_rate = DEFAULT_SAMPLE_RATE
        self.input_channels = DEFAULT_CHANNELS
        self.output_channels = DEFAULT_CHANNELS

        self._input_stream = None
        self._output_stream = None
        self._lock = threading.Lock()
        self._running = False
        self._elastic = None

    @staticmethod
    def list_input_devices():
        wasapi_idx = _get_wasapi_hostapi_index()
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and (wasapi_idx is None or d["hostapi"] == wasapi_idx):
                result.append({"index": i, "name": d["name"]})
        return result

    @staticmethod
    def list_output_devices():
        wasapi_idx = _get_wasapi_hostapi_index()
        devices = sd.query_devices()
        result = []
        for i, d in enumerate(devices):
            if d["max_output_channels"] > 0 and (wasapi_idx is None or d["hostapi"] == wasapi_idx):
                result.append({"index": i, "name": d["name"]})
        return result

    @staticmethod
    def find_vb_cable_output_index():
        for d in AudioEngine.list_output_devices():
            if "CABLE Input" in d["name"]:
                return d["index"]
        return None

    @staticmethod
    def _get_device_info(device_index):
        try:
            return sd.query_devices(device_index)
        except Exception:
            return {}

    def _input_callback(self, indata, frames, time_info, status):
        with self._lock:
            mic_signal = indata.copy()
            processed = self.chain.process(mic_signal, self.input_rate)

        if processed.shape[1] != self.output_channels:
            if processed.shape[1] == 1 and self.output_channels == 2:
                processed = np.repeat(processed, 2, axis=1)
            elif processed.shape[1] == 2 and self.output_channels == 1:
                processed = processed.mean(axis=1, keepdims=True)
            else:
                fixed = np.zeros((processed.shape[0], self.output_channels), dtype=np.float32)
                n = min(processed.shape[1], self.output_channels)
                fixed[:, :n] = processed[:, :n]
                processed = fixed

        if self._elastic is not None:
            self._elastic.write(processed)

    def _output_callback(self, outdata, frames, time_info, status):
        if self._elastic is not None:
            block = self._elastic.read(frames)
        else:
            block = np.zeros((frames, self.output_channels), dtype=np.float32)
        outdata[:] = np.clip(block, -1.0, 1.0)

    def start(self):
        if self._running:
            return

        mic_info = self._get_device_info(self.mic_device)
        out_info = self._get_device_info(self.output_device)

        requested_input_rate = int(round(mic_info.get("default_samplerate", DEFAULT_SAMPLE_RATE)))
        requested_output_rate = int(round(out_info.get("default_samplerate", DEFAULT_SAMPLE_RATE)))

        self.input_channels = max(1, min(2, mic_info.get("max_input_channels", 1)))
        self.output_channels = max(1, min(2, out_info.get("max_output_channels", 2)))

        self._input_stream = sd.InputStream(
            device=self.mic_device,
            channels=self.input_channels,
            samplerate=requested_input_rate,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            device=self.output_device,
            channels=self.output_channels,
            samplerate=requested_output_rate,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._output_callback,
        )

        self.input_rate = int(round(self._input_stream.samplerate))
        self.output_rate = int(round(self._output_stream.samplerate))

        self._elastic = ElasticBuffer(channels=self.output_channels, samplerate=self.output_rate)

        self.soundboard.output_device = self.output_device

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
        self._elastic = None
        self._running = False
        self.soundboard.stop_all()

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
        self.soundboard.set_gain_db(gain_db)

    def set_soundboard_deep_fry_enabled(self, enabled: bool):
        self.soundboard.set_deep_fry_enabled(enabled)

    def update_soundboard_deep_fry_params(self, **kwargs):
        self.soundboard.update_deep_fry_params(**kwargs)

    def play_sound(self, filepath, volume=1.0, sound_id=None):
        if self.soundboard.output_device is None:
            self.soundboard.output_device = self.output_device
        self.soundboard.play(filepath, volume, sound_id=sound_id)

    def stop_sound(self, sound_id):
        self.soundboard.stop_sound(sound_id)

    def stop_all_sounds(self):
        self.soundboard.stop_all()

    def to_config_dict(self) -> dict:
        d = self.chain.to_config_dict()
        d["soundboard_volume_db"] = self.soundboard.gain.params.get("gain_db", 0.0)
        d["effects"]["soundboard_deep_fry"] = self.soundboard.deep_fry.to_dict()
        return d

    def load_config_dict(self, config: dict):
        self.chain.load_config_dict(config)
        self.soundboard.gain.params["gain_db"] = config.get("soundboard_volume_db", 0.0)
        sfx_cfg = config.get("effects", {}).get("soundboard_deep_fry", {})
        self.soundboard.deep_fry.load_dict(sfx_cfg)
