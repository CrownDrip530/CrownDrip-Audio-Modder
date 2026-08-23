"""
audio_engine.py
Real-time audio engine: captures your real mic, runs it through the
EffectChain, mixes in any playing soundboard clips (with their own
volume/deep fry controls), and streams the result out to the virtual
cable device.

Uses two independent streams (InputStream + OutputStream) linked by a
thread-safe queue, since Windows won't allow one combined duplex stream
across two different devices/host APIs (real mic + VB-Cable).

Because the mic and the virtual cable are separate devices, they run on
independent hardware clocks. Even at the "same" 48000Hz sample rate, tiny
clock drift between them causes the buffer to slowly overflow or run dry
over a few seconds -> crackling / pitch wobble. We keep the queue depth
near a target watermark to compensate for this drift.
"""

import sounddevice as sd
import numpy as np
import threading
import queue

from effects import EffectChain, GainEffect, DeepFryEffect
from soundboard import SoundboardPlayer

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
CHANNELS = 2

TARGET_QUEUE_DEPTH = 4
MAX_QUEUE_DEPTH = 10


class AudioEngine:
    def __init__(self, mic_device=None, output_device=None, monitor_device=None):
        self.mic_device = mic_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.monitor_enabled = False

        self.chain = EffectChain()

        self.soundboard_gain = GainEffect(enabled=True, gain_db=0.0)
        self.soundboard_deep_fry = DeepFryEffect(enabled=False)

        self.soundboard = SoundboardPlayer(samplerate=SAMPLE_RATE, channels=CHANNELS)

        self._input_stream = None
        self._output_stream = None
        self._lock = threading.Lock()
        self._running = False
        self._buffer = queue.Queue(maxsize=MAX_QUEUE_DEPTH)
        self._last_block = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

    @staticmethod
    def list_input_devices():
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    @staticmethod
    def list_output_devices():
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"]}
            for i, d in enumerate(devices)
            if d["max_output_channels"] > 0
        ]

    @staticmethod
    def find_vb_cable_output_index():
        for d in AudioEngine.list_output_devices():
            if "CABLE Input" in d["name"]:
                return d["index"]
        return None

    def _input_callback(self, indata, frames, time_info, status):
        with self._lock:
            mic_signal = indata.copy()
            if mic_signal.shape[1] == 1 and CHANNELS == 2:
                mic_signal = np.repeat(mic_signal, 2, axis=1)
            processed = self.chain.process(mic_signal, SAMPLE_RATE)

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
            sb_block = self.soundboard_gain.process(sb_block, SAMPLE_RATE)
            if self.soundboard_deep_fry.enabled:
                sb_block = self.soundboard_deep_fry.process(sb_block, SAMPLE_RATE)

        mixed = np.clip(block + sb_block, -1.0, 1.0)
        outdata[:] = mixed

    def start(self):
        if self._running:
            return

        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except queue.Empty:
                break
        self._last_block = np.zeros((BLOCK_SIZE, CHANNELS), dtype=np.float32)

        self._input_stream = sd.InputStream(
            device=self.mic_device,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._input_callback,
        )
        self._output_stream = sd.OutputStream(
            device=self.output_device,
            channels=CHANNELS,
            samplerate=SAMPLE_RATE,
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
