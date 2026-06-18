"""DSP unit tests: ratio, alignment, background subtraction, filters, STFT."""
from __future__ import annotations

import numpy as np
from scipy import signal

from wifi_room_radar.processing.filters import Decimator, StreamingSOSFilter, design_bandpass
from wifi_room_radar.processing.preprocessing import (
    BackgroundSubtractor,
    align_to_background,
    csi_ratio,
    pca_first_component,
)
from wifi_room_radar.processing.spectrogram import StreamingSTFT

RNG = np.random.default_rng(5)


def _random_csi(n_rx=3, n_tx=1, n_sub=32):
    return RNG.standard_normal((n_rx, n_tx, n_sub)) + 1j * RNG.standard_normal((n_rx, n_tx, n_sub))


def test_csi_ratio_cancels_common_phase():
    base = _random_csi()
    rot = base * np.exp(1j * RNG.uniform(0, 2 * np.pi))  # common CFO phase
    np.testing.assert_allclose(csi_ratio(rot), csi_ratio(base), atol=1e-10)


def test_align_to_background_recovers_offset_and_slope():
    bg = _random_csi()
    k = np.arange(bg.shape[-1])
    corrupted = bg * np.exp(1j * (0.7 + 0.03 * k))[None, None, :]
    aligned = align_to_background(corrupted, bg)
    np.testing.assert_allclose(aligned, bg, atol=1e-6)


def test_background_subtractor_isolates_dynamic():
    static = _random_csi()
    sub = BackgroundSubtractor(alpha=0.05)
    for _ in range(300):
        dynamic, _ = sub.update(static)
    assert np.linalg.norm(dynamic) < 1e-6 * np.linalg.norm(static)
    # a small superimposed perturbation must survive subtraction
    pert = 0.05 * _random_csi()
    dynamic, _ = sub.update(static + pert)
    assert np.linalg.norm(dynamic) > 0.5 * np.linalg.norm(pert)


def test_pca_first_component_recovers_rank1_time_course():
    t = np.linspace(0, 4, 200)
    course = np.exp(1j * 2 * np.pi * 1.5 * t) * (1 + 0.3 * np.sin(2 * np.pi * 0.3 * t))
    direction = RNG.standard_normal(24) + 1j * RNG.standard_normal(24)
    X = np.outer(course, direction) + 0.05 * (
        RNG.standard_normal((200, 24)) + 1j * RNG.standard_normal((200, 24))
    )
    comp = pca_first_component(X)
    c = np.abs(np.vdot(comp - comp.mean(), course - course.mean()))
    c /= np.linalg.norm(comp - comp.mean()) * np.linalg.norm(course - course.mean())
    assert c > 0.95


def test_streaming_stft_maps_signed_doppler():
    fs, n_fft, hop = 200.0, 128, 32
    stft = StreamingSTFT(n_fft, hop, fs)
    t = np.arange(int(4 * fs)) / fs
    for f_true in (+30.0, -30.0):
        stft = StreamingSTFT(n_fft, hop, fs)
        cols = stft.push(np.exp(2j * np.pi * f_true * t))
        assert cols, "expected at least one spectrum column"
        peak_f = stft.freqs[int(np.argmax(cols[-1]))]
        assert abs(peak_f - f_true) <= fs / n_fft


def test_streaming_sos_filter_matches_batch():
    """Chunked output must equal one-shot filtering with the same priming.

    StreamingSOSFilter primes its state with the first sample's DC steady
    state (suppressing the cold-start step transient), so the batch
    reference must be primed identically rather than starting from zeros.
    """
    sos = design_bandpass(1.0, 8.0, 50.0)
    x = RNG.standard_normal(1000)
    batch, _ = signal.sosfilt(sos, x, zi=signal.sosfilt_zi(sos) * x[0])
    f = StreamingSOSFilter(sos)
    chunks = [f.push(c) for c in np.split(x, [100, 350, 720])]
    np.testing.assert_allclose(np.concatenate(chunks), batch, atol=1e-10)


def test_decimator_preserves_in_band_tone():
    fs_in, fs_out = 200.0, 20.0
    dec = Decimator(fs_in, fs_out)
    t = np.arange(int(20 * fs_in)) / fs_in
    x = np.sin(2 * np.pi * 3.0 * t)  # 3 Hz, safely below the 10 Hz output Nyquist
    out = np.concatenate([dec.push(c) for c in np.split(x, 40)])
    assert abs(len(out) - len(x) * fs_out / fs_in) <= 2
    f, p = signal.welch(out[int(2 * fs_out):], fs=fs_out, nperseg=256)
    assert abs(f[np.argmax(p)] - 3.0) < 0.2
