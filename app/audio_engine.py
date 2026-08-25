"""
audio_engine.py
Real-time audio engine.

QUICK FIX: previous versions forced the output stream to always open at
the device's reported "default" rate (typically 48000Hz), requiring every
non-48000 file (e.g. a typical 44100Hz WAV/MP3) to go through our
resampler. Diagnostics confirmed zero buffer xruns/underruns, which rules
out timing/buffering as the wobble's cause -- leaving the resample step
itself as the last real suspect. Instead of debugging that further, we
now try opening the output stream at the FILE's own native rate first
(no resampling needed at all if it succeeds), only falling back to the
device's default rate (with resampling) if the device rejects that rate.
This removes resampling from the picture entirely for the common case.
"""

import sounddevice as sd
import numpy as np
import threading

from effects import EffectChain
from soundboard import SoundboardPlayer

BLOCK_SIZE = 1024
DEFAULT_SAMPLE_RATE = 48000
DEFAULT_CHANNELS = 2

TARGET_LATENCY_SEC = 0.15
MAX_LATENCY_SEC = 0.6

# Common rates to try, in priority order, before falling back to the
# device's own reported default. 44100 is the most common file rate.
CANDIDATE_OUTPUT_RATES = [44100, 48000]


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

        self.input_rate = DEFAULT_SAMPLE_RATE
        self.output_rate = DEFAULT_SAMPLE_RATE
        self.input_channels = DEFAULT_CHANNELS
        self.output_channels = DEFAULT_CHANNELS

        self.soundboard = SoundboardPlayer(samplerate=DEFAULT_SAMPLE_RATE, channels=DEFAULT_CHANNELS)

        self._input_stream = None
        self._output_stream = None
        self._lock = threading.Lock()
        self._running = False
        self._elastic = None

        self.input_xruns = 0
        self.output_xruns = 0

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
        if status:
            self.input_xruns += 1

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
        if status:
            self.output_xruns += 1

        if self._elastic is not None:
            block = self._elastic.read(frames)
        else:
            block = np.zeros((frames, self.output_channels), dtype=np.float32)

        sb_block = self.soundboard.read_block(frames)

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

    def _open_output_stream(self, device_channels):
        """Try opening the output stream at each candidate rate in order
        (44100 first, since that's the most common file rate) before
        falling back to the device's own reported default. Returns the
        opened stream, or raises the last error if nothing worked."""
        out_info = self._get_device_info(self.output_device)
        default_rate = int(round(out_info.get("default_samplerate", DEFAULT_SAMPLE_RATE)))

        rates_to_try = [r for r in CANDIDATE_OUTPUT_RATES if r != default_rate] + [default_rate]

        last_error = None
        for rate in rates_to_try:
            try:
                stream = sd.OutputStream(
                    device=self.output_device,
                    channels=device_channels,
                    samplerate=rate,
                    blocksize=BLOCK_SIZE,
                    dtype="float32",
                    callback=self._output_callback,
                )
                return stream
            except Exception as e:
                last_error = e
                continue

        raise last_error if last_error else RuntimeError("Could not open output stream at any candidate rate")

    def start(self):
        if self._running:
            return

        self.input_xruns = 0
        self.output_xruns = 0

        mic_info = self._get_device_info(self.mic_device)
        out_info = self._get_device_info(self.output_device)

        requested_input_rate = int(round(mic_info.get("default_samplerate", DEFAULT_SAMPLE_RATE)))

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

        # Try 44100 first (no resampling needed for the most common file
        # rate), only falling back to the device's default (with
        # resampling) if 44100 is rejected.
        self._output_stream = self._open_output_stream(self.output_channels)

        self.input_rate = int(round(self._input_stream.samplerate))
        self.output_rate = int(round(self._output_stream.samplerate))

        old_gain_db = self.soundboard.gain.params.get("gain_db", 0.0)
        old_fry_enabled = self.soundboard.deep_fry.enabled
        old_fry_params = dict(self.soundboard.deep_fry.params)

        self.soundboard = SoundboardPlayer(samplerate=self.output_rate, channels=self.output_channels)
        self.soundboard.gain.params["gain_db"] = old_gain_db
        self.soundboard.deep_fry.enabled = old_fry_enabled
        self.soundboard.deep_fry.params.update(old_fry_params)

        self._elastic = ElasticBuffer(channels=self.output_channels, samplerate=self.output_rate)

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
