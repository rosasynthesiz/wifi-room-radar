"""Streaming short-time Fourier transform for micro-Doppler analysis.

A reflector at path length ``d(t)`` contributes a phasor
``exp(-j * 2*pi * d(t) / lambda)`` to the dynamic CSI component. Its
instantaneous Doppler frequency is ``f_D = -(1/lambda) * d'(t)``: a person
walking towards the link shortens the reflected path (``d' < 0``) and shows
up at *positive* Doppler; walking away shows up at *negative* Doppler. A
chest moving with breathing traces a slow +-0.1 Hz oscillation around DC.

Because the CSI-derived motion series is **complex** (we keep the full
phasor, not just its magnitude), the spectrum is genuinely two-sided:
positive and negative frequency bins are independent, so approaching and
receding motion are distinguishable. Had we taken a real signal, the
spectrum would be conjugate-symmetric and that sign information would be
lost — which is exactly why the pipeline carries complex samples all the
way into the STFT.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

#: Amplitude floor inside the log so silent bins clip near -240 dB instead
#: of producing -inf.
_DB_EPS = 1e-12


def doppler_bins(n_fft: int, fs: float) -> np.ndarray:
    """FFT-shifted Doppler frequency axis.

    Args:
        n_fft: FFT length.
        fs: Sample rate of the (complex) motion series in Hz.

    Returns:
        Frequencies in Hz, shape ``[n_fft]``, ordered from most negative
        (``-fs/2``, receding) through 0 (static) to most positive
        (approaching), matching the column layout of
        :class:`StreamingSTFT`.
    """
    return np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / fs))


class StreamingSTFT:
    """Incremental STFT that emits spectrum columns as samples arrive.

    Samples are appended to an internal buffer; whenever ``n_fft`` samples
    are available a column is emitted and the analysis position advances by
    ``hop`` (so consecutive columns overlap by ``n_fft - hop`` samples).
    Pushing arbitrary chunk sizes — including chunks of a single sample —
    yields exactly the same columns as one big offline STFT.

    Each column is the fft-shifted **magnitude in dB** of the windowed
    block, normalised by the window's coherent gain (``sum(window)``) so a
    unit-amplitude complex tone reads ~0 dB regardless of ``n_fft`` or
    window choice. Because the input is complex, the column is two-sided:
    index 0 is ``-fs/2`` (receding motion), the centre bin is DC (static
    reflections), and the last index approaches ``+fs/2`` (approaching
    motion) — see :func:`doppler_bins` and the module docstring for the
    sign convention.

    Args:
        n_fft: Analysis block / FFT length.
        hop: Samples to advance between consecutive columns.
        fs: Input sample rate in Hz.
        window: Window name accepted by :func:`scipy.signal.get_window`
            (default Hann: -31 dB sidelobes keep strong body Doppler from
            masking nearby faint vitals lines).
    """

    def __init__(self, n_fft: int, hop: int, fs: float, window: str = "hann"):
        if n_fft < 2:
            raise ValueError(f"n_fft must be >= 2, got {n_fft}")
        if hop < 1:
            raise ValueError(f"hop must be >= 1, got {hop}")
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.fs = float(fs)
        self._window = signal.get_window(window, self.n_fft, fftbins=True).astype(np.float64)
        self._gain = float(np.sum(self._window))
        self._freqs = doppler_bins(self.n_fft, self.fs)
        self._buf = np.zeros(0, dtype=np.complex128)
        self._skip = 0  # deficit carried over when hop > n_fft

    @property
    def freqs(self) -> np.ndarray:
        """Doppler axis of every emitted column (copy), shape ``[n_fft]``."""
        return self._freqs.copy()

    @property
    def seconds_per_column(self) -> float:
        """Time advance between consecutive columns."""
        return self.hop / self.fs

    def reset(self) -> None:
        """Drop all buffered samples."""
        self._buf = np.zeros(0, dtype=np.complex128)
        self._skip = 0

    def push(self, samples: np.ndarray) -> list[np.ndarray]:
        """Append samples; return all spectrum columns now computable.

        Args:
            samples: 1-D array of complex samples (real input is promoted
                to complex; its spectrum will then simply be symmetric).

        Returns:
            Zero or more columns, each a float64 array of length ``n_fft``:
            fft-shifted magnitudes in dB (``20*log10(|X|/gain + eps)``),
            ordered negative to positive Doppler.
        """
        s = np.asarray(samples)
        if s.ndim != 1:
            raise ValueError(f"expected 1-D samples, got shape {s.shape}")
        s = s.astype(np.complex128, copy=False)

        if self._skip:  # consume deficit from a previous hop > n_fft advance
            drop = min(self._skip, s.size)
            s = s[drop:]
            self._skip -= drop
        if s.size:
            self._buf = np.concatenate((self._buf, s))

        columns: list[np.ndarray] = []
        start = 0
        while start + self.n_fft <= self._buf.size:
            block = self._buf[start : start + self.n_fft] * self._window
            spec = np.fft.fftshift(np.fft.fft(block)) / self._gain
            columns.append(20.0 * np.log10(np.abs(spec) + _DB_EPS))
            start += self.hop

        if start:
            if start >= self._buf.size:
                self._skip = start - self._buf.size
                self._buf = np.zeros(0, dtype=np.complex128)
            else:
                self._buf = self._buf[start:].copy()
        return columns
