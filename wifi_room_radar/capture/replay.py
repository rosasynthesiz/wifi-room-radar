"""Record / replay CSI streams to and from ``.npz`` files.

A recording is a single compressed numpy archive with five arrays:

==================  =======================  =====================================
key                 shape / dtype            meaning
==================  =======================  =====================================
``timestamps``      ``[T] float64``          seconds since stream start
``seq``             ``[T] int64``            packet sequence numbers (gaps = drops)
``csi``             ``[T, n_rx, n_tx, n_sub] complex128``  stacked channel matrices
``rssi``            ``[T] float64``          per-packet RSSI (dBm)
``ground_truth``    ``[T] unicode``          per-frame JSON string (``"null"`` when
                                             the frame had no ground truth)
==================  =======================  =====================================

Storing ground truth as JSON strings keeps the archive loadable with
``allow_pickle=False`` (no arbitrary-object security risk) while preserving
the full nested dict.

:class:`ReplayCSISource` plays a recording back through the standard
:class:`~wifi_room_radar.capture.base.CSISource` interface, so the whole pipeline
can be exercised offline against saved (simulated or real) captures. When no
:class:`~wifi_room_radar.config.RadioConfig` is supplied, a best-effort config is
reconstructed: array dimensions from the CSI shape and the sample rate from
the median timestamp spacing (the median is robust to gaps left by dropped
packets); geometry fields fall back to the package defaults.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterator, Optional, Sequence, Union

import numpy as np

from ..config import RadioConfig
from ..types import CSIFrame
from .base import CSISource

PathLike = Union[str, os.PathLike]


def save_recording(frames: Sequence[CSIFrame], path: PathLike) -> Path:
    """Write a list of CSI frames to a compressed ``.npz`` recording.

    Args:
        frames: Frames in time order (e.g. collected from any
            :class:`CSISource`). All frames must share one CSI shape.
        path: Destination file. A ``.npz`` suffix is appended by numpy if
            missing.

    Returns:
        The path actually written (with the ``.npz`` suffix resolved).

    Raises:
        ValueError: If ``frames`` is empty or CSI shapes are inconsistent.
    """
    if not frames:
        raise ValueError("save_recording: cannot save an empty frame list")
    shape0 = frames[0].csi.shape
    for f in frames:
        if f.csi.shape != shape0:
            raise ValueError(
                f"save_recording: inconsistent CSI shapes {shape0} vs {f.csi.shape}"
            )
    timestamps = np.array([f.timestamp for f in frames], dtype=np.float64)
    seq = np.array([f.seq for f in frames], dtype=np.int64)
    csi = np.stack([np.asarray(f.csi, dtype=np.complex128) for f in frames])
    rssi = np.array([f.rssi for f in frames], dtype=np.float64)
    ground_truth = np.array([json.dumps(f.ground_truth) for f in frames])

    path = Path(path)
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=timestamps,
        seq=seq,
        csi=csi,
        rssi=rssi,
        ground_truth=ground_truth,
    )
    return path


class ReplayCSISource(CSISource):
    """Play a saved ``.npz`` recording back as a :class:`CSISource`.

    Args:
        path: Recording written by :func:`save_recording`.
        radio: Radio/link config to attach to the stream. If ``None``, a
            config is reconstructed from the recording itself: ``n_rx``,
            ``n_tx`` and ``n_subcarriers`` from the CSI array shape, and
            ``sample_rate`` from the median inter-frame interval (robust to
            sequence gaps from dropped packets). Geometry (positions,
            carrier) keeps the :class:`RadioConfig` defaults — fine for
            replaying recordings made with the default geometry; pass an
            explicit config otherwise.
        realtime: If True, pace playback to wall-clock time using the
            recorded timestamps; if False (default) yield as fast as the
            consumer pulls.
    """

    def __init__(
        self,
        path: PathLike,
        radio: Optional[RadioConfig] = None,
        realtime: bool = False,
    ) -> None:
        self.path = Path(path)
        with np.load(self.path, allow_pickle=False) as data:
            self._timestamps = np.asarray(data["timestamps"], dtype=np.float64)
            self._seq = np.asarray(data["seq"], dtype=np.int64)
            self._csi = np.asarray(data["csi"], dtype=np.complex128)
            self._rssi = np.asarray(data["rssi"], dtype=np.float64)
            self._ground_truth = [str(s) for s in data["ground_truth"]]
        if self._csi.ndim != 4:
            raise ValueError(
                f"{self.path}: expected csi of shape [T, n_rx, n_tx, n_sub], "
                f"got shape {self._csi.shape}"
            )
        if radio is None:
            _, n_rx, n_tx, n_sub = self._csi.shape
            radio = RadioConfig(
                n_rx=int(n_rx),
                n_tx=int(n_tx),
                n_subcarriers=int(n_sub),
                sample_rate=self._infer_sample_rate(),
            )
        super().__init__(radio)
        self.realtime = realtime
        self._closed = False

    def _infer_sample_rate(self) -> float:
        """Sample rate from the median timestamp spacing (gap-robust)."""
        if self._timestamps.shape[0] < 2:
            return float(RadioConfig.sample_rate)
        dts = np.diff(self._timestamps)
        dts = dts[dts > 0]
        if dts.size == 0:
            return float(RadioConfig.sample_rate)
        return float(1.0 / np.median(dts))

    def __len__(self) -> int:
        """Number of frames in the recording."""
        return int(self._timestamps.shape[0])

    def frames(self) -> Iterator[CSIFrame]:
        """Yield the recorded frames in order (paced when ``realtime``)."""
        n = len(self)
        if n == 0:
            return
        t0 = float(self._timestamps[0])
        start_wall = time.perf_counter()
        for i in range(n):
            if self._closed:
                return
            t = float(self._timestamps[i])
            if self.realtime:
                delay = (start_wall + (t - t0)) - time.perf_counter()
                if delay > 0.0:
                    time.sleep(delay)
            gt = json.loads(self._ground_truth[i])
            yield CSIFrame(
                timestamp=t,
                seq=int(self._seq[i]),
                csi=self._csi[i],
                rssi=float(self._rssi[i]),
                ground_truth=gt,
            )

    def close(self) -> None:
        """Stop playback at the next frame. Idempotent."""
        self._closed = True

    @property
    def info(self) -> dict:
        base = super().info
        duration = 0.0
        if len(self) >= 2:
            duration = float(self._timestamps[-1] - self._timestamps[0])
        base.update(
            {
                "path": str(self.path),
                "n_frames": len(self),
                "duration_sec": duration,
                "realtime": self.realtime,
            }
        )
        return base
