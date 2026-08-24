"""
effects.py
Extensible audio effects chain.
"""

import numpy as np
from scipy.signal import lfilter, lfilter_zi


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


class _PersistentPeakingFilter:
    """Wraps a peaking EQ biquad filter and keeps its internal state (zi)
    persistent ACROSS audio blocks, instead of resetting to silence at the
    start of every ~21ms block.

    WHY THIS MATTERS: lfilter() with no explicit zi starts from zero
    internal memory every single call. Called repeatedly on sequential
    chunks of one continuous audio stream (as we do, once per audio
    callback), that produces a tiny artificial ringing/decay artifact at
    every block boundary. Stacked 40+ times per second, this is audible as
    a smeared, echo/reverb-like trail behind the voice -- NOT true delay
    based echo, but a filter discontinuity artifact that sounds similar.
    Persisting zi between calls (per channel) makes the filter behave
    exactly as if it were processing one continuous unbroken signal,
    which eliminates this."""

    def __init__(self):
        self._b = None
        self._a = None
        self._zi_per_channel = {}  # channel index -> zi state array
        self._last_params = None

    def _maybe_rebuild_coeffs(self, samplerate, boost_db, freq, Q):
        params_key = (samplerate, round(boost_db, 3), freq, Q)
        if params_key == self._last_params and self._b is not None:
            return

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

        self._b = np.array([b0, b1, b2]) / a0
        self._a = np.array([1.0, a1 / a0, a2 / a0])
        self._last_params = params_key
        # Coefficients changed (e.g. user moved a slider) -- reset zi so
        # the filter re-settles cleanly rather than using stale state
        # computed for different coefficients.
        self._zi_per_channel = {}

    def process(self, x: np.ndarray, samplerate: int, boost_db: float, freq: float, Q: float) -> np.ndarray:
        if boost_db == 0:
            return x

        self._maybe_rebuild_coeffs(samplerate, boost_db, freq, Q)

        out = np.zeros_like(x)
        for ch in range(x.shape[1]):
            if ch not in self._zi_per_channel:
                self._zi_per_channel[ch] = lfilter_zi(self._b, self._a) * x[0, ch]
            filtered, zf = lfilter(self._b, self._a, x[:, ch], zi=self._zi_per_channel[ch])
            self._zi_per_channel[ch] = zf
            out[:, ch] = filtered

        return out.astype(np.float32)


class DeepFryEffect(Effect):
    """Aggressive 'deep fried meme audio' distortion chain:

    1. Pre-gain drive boost
    2. Waveshaping distortion (tanh) + extra hard clip fold
    3. Sample-rate reduction / zero-order-hold decimation
    4. Two stacked harsh mid/high peaking EQ boosts (state-persistent
       across blocks -- this is what fixes the echo/smearing artifact)
    5. Bitcrush (bit depth reduction)
    6. Final saturation + hard clip
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
        # Two independent persistent filters (different frequencies), each
        # keeping its own state across calls.
        self._filter1 = _PersistentPeakingFilter()
        self._filter2 = _PersistentPeakingFilter()

    def process(self, signal: np.ndarray, samplerate: int) -> np.ndarray:
        drive = float(self.params.get("drive", 18.0))
        bit_depth = int(self.params.get("bitcrush_depth", 4))
        eq_boost_db = float(self.params.get("eq_boost_db", 16.0))
        sr_reduction = int(self.params.get("sample_reduction", 3))

        x = signal.copy()

        x = x * drive
        x = np.tanh(x)
        x = np.clip(x * 1.8, -1.0, 1.0)

        if sr_reduction > 1:
            x = self._sample_reduce(x, sr_reduction)

        x = self._filter1.process(x, samplerate, eq_boost_db, freq=2000.0, Q=1.2)
        x = self._filter2.process(x, samplerate, eq_boost_db * 0.6, freq=3500.0, Q=1.5)

        levels = 2 ** max(2, bit_depth)
        x = np.round(x * levels) / levels

        x = np.tanh(x * 1.5)
        x = np.clip(x * 1.6, -1.0, 1.0)

        return x

    @staticmethod
    def _sample_reduce(x: np.ndarray, factor: int) -> np.ndarray:
        out = np.empty_like(x)
        n = len(x)
        for i in range(0, n, factor):
            end = min(i + factor, n)
            out[i:end] = x[i]
        return out


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
