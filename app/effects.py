"""
effects.py
Extensible audio effects chain.

Design:
- Each effect is a class implementing `process(signal, samplerate) -> signal`
- Effects have `enabled` flag and a `params` dict for GUI-tweakable settings
- EffectChain runs enabled effects in order, mic gain always applied first

To add a new effect later (echo, robot/pitch, reverb, etc.):
1. Subclass Effect
2. Implement process()
3. Register it in EffectChain.__init__ (or dynamically via register())
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
    """Aggressive 'deep fried meme audio' distortion chain:

    1. Pre-gain drive boost
    2. Waveshaping distortion (tanh) + extra hard clip fold
    3. Sample-rate reduction / zero-order-hold decimation (classic lo-fi
       aliasing crunch you hear in deep fried memes)
    4. Two stacked harsh mid/high peaking EQ boosts
    5. Bitcrush (bit depth reduction)
    6. Final saturation + hard clip for loudness/crispiness
    """
    name = "deep_fry"

    def __init__(self, enabled=False, drive=18.0, bitcrush_depth=4,
                 eq_boost_db=16.0, sample_reduction=3):
        super().__init__(enabled, {
            "drive": drive,
            "bitcrush_depth": bitcrush_depth,
            "eq_boost_db": eq_boost_db,
            "sample_reduction": sample_reduction,
        })

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        drive = float(self.params.get("drive", 18.0))
        bit_depth = int(self.params.get("bitcrush_depth", 4))
        eq_boost_db = float(self.params.get("eq_boost_db", 16.0))
        sr_reduction = int(self.params.get("sample_reduction", 3))

        x = signal.copy()

        # 1. Pre-gain drive boost
        x = x * drive

        # 2. Waveshaping distortion + extra hard clip fold for harsher edges
        x = np.tanh(x)
        x = np.clip(x * 1.8, -1.0, 1.0)

        # 3. Sample-rate reduction (zero-order-hold decimation) -- this is
        #    what gives deep fried audio its characteristic lo-fi aliasing
        #    crunch, distinct from just distortion.
        if sr_reduction > 1:
            x = self._sample_reduce(x, sr_reduction)

        # 4. Stacked harsh mid/high peaking EQ boosts (nasty deep fry range)
        x = self._peaking_boost(x, samplerate, eq_boost_db, freq=2000.0, Q=1.2)
        x = self._peaking_boost(x, samplerate, eq_boost_db * 0.6, freq=3500.0, Q=1.5)

        # 5. Bitcrush - reduce bit depth for crunchy digital artifacts
        levels = 2 ** max(2, bit_depth)
        x = np.round(x * levels) / levels

        # 6. Final saturation + hard clip so it's loud and crispy
        x = np.tanh(x * 1.5)
        x = np.clip(x * 1.6, -1.0, 1.0)

        return x

    @staticmethod
    def _sample_reduce(x: np.ndarray, factor: int) -> np.ndarray:
        """Zero-order-hold decimation: holds each sample for `factor` frames,
        simulating a much lower sample rate and introducing aliasing crunch."""
        out = np.empty_like(x)
        n = len(x)
        for i in range(0, n, factor):
            end = min(i + factor, n)
            out[i:end] = x[i]
        return out

    @staticmethod
    def _peaking_boost(x: np.ndarray, samplerate: int, boost_db: float,
                        freq: float, Q: float) -> np.ndarray:
        """Peaking EQ boost at a given frequency using a biquad filter."""
        if boost_db == 0:
            return x

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
    """Simple dB gain stage, applied first in the chain (mic boost)."""
    name = "gain"

    def __init__(self, enabled=True, gain_db=0.0):
        super().__init__(enabled, {"gain_db": gain_db})

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        gain_db = float(self.params.get("gain_db", 0.0))
        factor = 10 ** (gain_db / 20.0)
        return np.clip(signal * factor, -1.0, 1.0)


class EffectChain:
    """Ordered pipeline of effects applied to the mic signal.

    Order: Gain (always first, always enabled) -> other effects in
    registration order (Deep Fry, and anything added later).
    """

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
