"""
effects.py
Extensible audio effects chain.
"""

import numpy as np


class Effect:
    name = "base"

    def __init__(self, enabled=False, params=None):
        self.enabled = enabled
        self.params = params or {}

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        raise NotImplementedError

    def to_dict(self):
        return {"enabled": self.enabled, "params": self.params}

    def load_dict(self, data: dict):
        self.enabled = data.get("enabled", self.enabled)
        self.params.update(data.get("params", {}))


class DeepFryEffect(Effect):
    name = "deep_fry"

    def __init__(self, enabled=False, drive=8.0, bitcrush_depth=6, eq_boost_db=10.0):
        super().__init__(enabled, {
            "drive": drive,
            "bitcrush_depth": bitcrush_depth,
            "eq_boost_db": eq_boost_db,
        })

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        drive = float(self.params.get("drive", 8.0))
        bit_depth = int(self.params.get("bitcrush_depth", 6))
        eq_boost_db = float(self.params.get("eq_boost_db", 10.0))

        x = signal.copy()
        x = np.tanh(x * drive)
        x = self._mid_boost(x, samplerate, eq_boost_db)

        levels = 2 ** max(2, bit_depth)
        x = np.round(x * levels) / levels

        x = np.clip(x * 1.5, -1.0, 1.0)
        return x

    @staticmethod
    def _mid_boost(x: np.ndarray, samplerate: int, boost_db: float) -> np.ndarray:
        if boost_db == 0:
            return x

        freq = 2000.0
        Q = 1.2
        A = 10 ** (boost_db / 40.0)
        w0 = 2 * np.pi * freq / samplerate
        alpha = np.sin(w0) / (2 * Q)
        cos_w0 = np.cos(w0)

        b0 = 1 + alpha * A
        b1 = -2 * cos_w0
        b2 = 1 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2 * cos_w0
        a2 = 1 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([1.0, a1 / a0, a2 / a0])

        from scipy.signal import lfilter
        if x.ndim == 1:
            return lfilter(b, a, x).astype(np.float32)
        else:
            out = np.zeros_like(x)
            for ch in range(x.shape[1]):
                out[:, ch] = lfilter(b, a, x[:, ch])
            return out.astype(np.float32)


class GainEffect(Effect):
    name = "gain"

    def __init__(self, enabled=True, gain_db=0.0):
        super().__init__(enabled, {"gain_db": gain_db})

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        gain_db = float(self.params.get("gain_db", 0.0))
        factor = 10 ** (gain_db / 20.0)
        return np.clip(signal * factor, -1.0, 1.0)


class EffectChain:
    def __init__(self):
        self.gain = GainEffect(enabled=True, gain_db=0.0)
        self.deep_fry = DeepFryEffect(enabled=False)
        self.effects = [self.deep_fry]

    def register(self, effect: Effect):
        self.effects.append(effect)

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        out = self.gain.process(signal, samplerate)
        for fx in self.effects:
            if fx.enabled:
                out = fx.process(out, samplerate)
        return out

    def set_deep_fry_enabled(self, enabled: bool):
        self.deep_fry.enabled = enabled

    def set_mic_gain_db(self, gain_db: float):
        self.gain.params["gain_db"] = gain_db

    def to_config_dict(self) -> dict:
        return {
            "mic_gain_db": self.gain.params.get("gain_db", 0.0),
            "effects": {
                "deep_fry": self.deep_fry.to_dict()
            }
        }

    def load_config_dict(self, config: dict):
        self.gain.params["gain_db"] = config.get("mic_gain_db", 0.0)
        fx_cfg = config.get("effects", {}).get("deep_fry", {})
        self.deep_fry.load_dict(fx_cfg)
