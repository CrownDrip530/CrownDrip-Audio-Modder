"""
audio_engine.py
Real-time audio engine: captures your real mic, runs it through the
EffectChain, mixes in any playing soundboard clips, and streams the result
out to the virtual cable device.
"""

import sounddevice as sd
import numpy as np
import threading

from effects import EffectChain
from soundboard import SoundboardPlayer

SAMPLE_RATE = 48000
BLOCK_SIZE = 1024
CHANNELS = 2


class AudioEngine:
    def __init__(self, mic_device=None, output_device=None, monitor_device=None):
        self.mic_device = mic_device
        self.output_device = output_device
        self.monitor_device = monitor_device
        self.monitor_enabled = False

        self.chain = EffectChain()
        self.soundboard = SoundboardPlayer(samplerate=SAMPLE_RATE, channels=CHANNELS)

        self._stream = None
        self._lock = threading.Lock()
        self._running = False

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

    def start(self):
        if self._running:
            return

        def callback(indata, outdata, frames, time_info, status):
            if status:
                pass

            with self._lock:
                mic_signal = indata.copy()
                if mic_signal.shape[1] == 1 and CHANNELS == 2:
                    mic_signal = np.repeat(mic_signal, 2, axis=1)

                processed = self.chain.process(mic_signal, SAMPLE_RATE)

                sb_block = self.soundboard.read_block(frames)
                mixed = np.clip(processed + sb_block, -1.0, 1.0)

                outdata[:] = mixed

        self._stream = sd.Stream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            channels=CHANNELS,
            device=(self.mic_device, self.output_device),
            callback=callback,
        )
        self._stream.start()
        self._running = True

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
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

    def play_sound(self, filepath, volume=1.0):
        self.soundboard.play(filepath, volume)

    def stop_all_sounds(self):
        self.soundboard.stop_all()
