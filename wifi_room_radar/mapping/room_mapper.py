"""Bartlett matched-field room mapping from dynamic WiFi CSI.

Physical model
--------------
A person at position ``c`` scatters the TX signal toward the RX array. The
scattered component of the channel seen by RX element ``e`` on a subcarrier of
absolute RF frequency ``f`` is (up to a complex amplitude)::

    h_e(f) = a * exp(-1j * 2*pi * f * tau_e(c)),
    tau_e(c) = (|tx - c| + |c - rx_e|) / SPEED_OF_LIGHT

i.e. the bistatic TX->scatterer->RX delay. After the pipeline removes the
static (furniture / direct-path) channel with a background subtractor and
aligns CFO/STO nuisance phases, what is left of each CSI frame is a
superposition of such scatterer responses plus noise.

The mapper inverts this with a *Bartlett matched-field beamformer*: for each
cell of a grid covering the room it precomputes the expected unit response
``s_c`` over all (RX element, subcarrier) pairs and scores the cell by the
coherent matched-filter power averaged over a short time window::

    P(c) = mean_t | s_c^H x_t |^2

where ``x_t`` is the flattened dynamic CSI vector of frame ``t``. A real
scatterer at ``c`` makes all ``n_rx * n_sub`` terms add in phase, so ``P``
peaks at occupied cells.

Honest limitations (read before tuning)
---------------------------------------
With 20 MHz of bandwidth the *range* (delay) resolution is roughly
``c / (2 * B) ~ 7.5 m`` -- wider than a typical room. Almost all of the
localisation information therefore comes from:

* **Angle of arrival** across the small RX array (phase gradient over the
  elements at the carrier wavelength) -- this constrains the *bearing* of the
  scatterer from the array well, and
* the **coarse delay slope** across the 20 MHz band, which weakly constrains
  the total bistatic path length.

The resulting per-window likelihood maps are ridge-shaped: sharp across
bearing, smeared along range (an ellipse of constant ``|tx-c| + |c-rx|``).
Expect ghost lobes and range ambiguity in a single snapshot. Three mechanisms
recover usable positions anyway: the EMA across windows (a moving person's
ridge sweeps and only the true position stays consistently hot), the robust
percentile normalisation (kills the diffuse pedestal), and the downstream
Kalman tracker (imposes motion continuity). Treat the grid as a heat map of
evidence, not a calibrated probability.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from ..config import SPEED_OF_LIGHT, RadioConfig


class RoomMapper:
    """Streaming occupancy-map estimator over a fixed room grid.

    The mapper is stateful: each call to :meth:`update` consumes one window of
    background-subtracted dynamic CSI and refreshes the exponentially smoothed
    occupancy grid ``self.grid``. The map is gated so an empty room decays to
    an all-zero grid instead of amplifying noise into phantom blobs; a window
    counts as "active" when either

    * its total dynamic power exceeds an adaptive (minimum-tracking) noise
      floor, or
    * its lag-1 temporal coherence is high. At 200 frames/s consecutive CSI
      frames of a moving person are nearly identical (normalised correlation
      ~1), while white noise decorrelates completely (~1/sqrt(T*F)). This
      scale-free test rescues the cold-start case where someone is already
      moving when the mapper starts and the power floor therefore initialises
      at person-level power.

    Args:
        radio: Link geometry and OFDM parameters. Must be the same
            :class:`~wifi_room_radar.config.RadioConfig` the capture source uses,
            since the steering matrix is built from ``radio.tx_pos``,
            ``radio.rx_positions()`` and ``radio.subcarrier_freqs()``.
        room_width: Room extent along x, metres.
        room_depth: Room extent along y, metres.
        grid_resolution: Cell edge length, metres. Cell ``(row, col)`` has its
            centre at ``((col + 0.5) * res, (row + 0.5) * res)``; rows index
            y/depth, columns index x/width.
        ema: Smoothing weight given to the *newest* map in the exponential
            moving average (0 < ema <= 1; higher = more responsive).

    Attributes:
        grid: Current smoothed occupancy map, shape ``[n_rows, n_cols]``,
            float64 in 0..1. All zeros until the first update.
        n_rows / n_cols: Grid dimensions.
        cell_centres: ``[n_cells, 2]`` (x, y) centres, row-major
            (``cell = row * n_cols + col``), matching ``grid.ravel()``.
    """

    #: Gate: the window is considered "active" (someone is moving) when its
    #: total dynamic power exceeds this multiple of the tracked noise floor.
    NOISE_GATE_FACTOR: float = 2.5

    #: Alternative gate: lag-1 temporal coherence above this opens the map
    #: regardless of the power floor (white noise sits near 0; a person at
    #: 200 frames/s sits near 1).
    COHERENCE_GATE: float = 0.6

    def __init__(
        self,
        radio: RadioConfig,
        room_width: float,
        room_depth: float,
        grid_resolution: float = 0.25,
        ema: float = 0.4,
    ) -> None:
        if grid_resolution <= 0.0:
            raise ValueError("grid_resolution must be positive")
        if not 0.0 < ema <= 1.0:
            raise ValueError("ema must be in (0, 1]")
        self.radio = radio
        self.room_width = float(room_width)
        self.room_depth = float(room_depth)
        self.resolution = float(grid_resolution)
        self.ema = float(ema)

        # Grid geometry. ceil so the grid always covers the full room even if
        # the room size is not an exact multiple of the resolution.
        self.n_cols = max(1, int(np.ceil(self.room_width / self.resolution - 1e-9)))
        self.n_rows = max(1, int(np.ceil(self.room_depth / self.resolution - 1e-9)))
        n_cells = self.n_rows * self.n_cols

        cols = (np.arange(self.n_cols) + 0.5) * self.resolution  # x of each column
        rows = (np.arange(self.n_rows) + 0.5) * self.resolution  # y of each row
        xx, yy = np.meshgrid(cols, rows)  # [n_rows, n_cols] each
        #: [n_cells, 2] (x, y), row-major so index i*n_cols+j <-> grid[i, j].
        self.cell_centres: np.ndarray = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Bearing of each cell as seen from the RX array centre, used by
        # detect_peaks: range resolution at 20 MHz is far coarser than the
        # room, so the map's reliable resolving dimension is bearing, and
        # peak suppression must operate there (see detect_peaks).
        rxc = np.asarray(radio.rx_center, dtype=float)
        self._cell_bearings: np.ndarray = np.arctan2(
            self.cell_centres[:, 1] - rxc[1], self.cell_centres[:, 0] - rxc[0]
        )

        # --- Steering matrices, one per receiver node ------------------------
        # Bistatic path length TX -> cell -> RX element, per (cell, element).
        # Each node is an independent radio chain (own LO/clock), so the
        # absolute phase BETWEEN nodes is unobservable: matched-field
        # processing is coherent only within a node, and node power maps are
        # fused incoherently in update(). Multiple viewing geometries are
        # what collapse the single-link range ridge into a point.
        tx = np.asarray(radio.tx_pos, dtype=float)
        all_rx = radio.rx_positions()  # [total_rx, 2], node blocks stacked
        freqs = radio.subcarrier_freqs()  # [n_sub] absolute RF Hz
        d_tx = np.linalg.norm(self.cell_centres - tx[None, :], axis=1)  # [n_cells]
        self.n_nodes = int(radio.n_nodes)
        self._per_node_rx = int(radio.n_rx)
        self._steering_h: list[np.ndarray] = []  # per node, [n_cells, F_node]
        for node in range(self.n_nodes):
            rx = all_rx[radio.node_slice(node)]  # [n_rx, 2]
            d_rx = np.linalg.norm(
                self.cell_centres[:, None, :] - rx[None, :, :], axis=2
            )  # [n_cells, n_rx]
            path = d_tx[:, None] + d_rx  # [n_cells, n_rx]
            # Expected unit-amplitude scatterer response, [n_cells, n_rx, n_sub]:
            #   s = exp(-1j * 2*pi * f * path / c)
            phase = (
                (2.0 * np.pi / SPEED_OF_LIGHT) * path[:, :, None] * freqs[None, None, :]
            )
            steering = np.exp(-1j * phase).reshape(
                n_cells, radio.n_rx * radio.n_subcarriers
            )
            # Row-normalise to unit L2 norm (each entry is unit modulus, so
            # this is a constant 1/sqrt(F), kept explicit for generality),
            # and store the Hermitian transpose rows: the matched filter is
            # s^H x, so the matrix we actually multiply with is conj(s).
            steering /= np.linalg.norm(steering, axis=1, keepdims=True)
            self._steering_h.append(np.conj(steering))

        #: Smoothed occupancy map in 0..1.
        self.grid: np.ndarray = np.zeros((self.n_rows, self.n_cols), dtype=float)
        self._have_grid = False  # first update sets the grid directly

        # Adaptive noise floor on mean |dynamic CSI|^2 per window, tracked
        # PER NODE. Per-node matters beyond mesh noise scaling: each node
        # sees a breathing phasor with its own geometry, so power troughs
        # (which the minimum-tracking floor calibrates on) occur at different
        # breathing phases per node — the summed power never dips to the
        # noise level, but each node's own power does. Snaps down to dips
        # instantly and creeps up slowly — slower still while the gate is
        # open, so a person present from the very first window cannot drag
        # the floor up to their own level.
        self._noise_floor: Optional[np.ndarray] = None  # [n_nodes]
        self._last_total_power: float = 0.0

        # Cells inside ~0.5 m of the TX or any RX node are excluded from the
        # map: their near-field steering vectors are quasi-degenerate (they
        # correlate broadly with any spatially structured signal and can
        # out-score the true cell), and a person cannot stand inside the
        # radio anyway.
        radios = np.vstack([tx[None, :], np.asarray(radio.node_centers(), dtype=float)])
        d_radio = np.min(
            np.linalg.norm(self.cell_centres[:, None, :] - radios[None, :, :], axis=2),
            axis=1,
        )
        self._radio_mask: np.ndarray = d_radio < 0.5  # [n_cells]

    @property
    def n_cells(self) -> int:
        """Total number of grid cells (= n_rows * n_cols)."""
        return self.n_rows * self.n_cols

    def cell_index(self, x: float, y: float) -> int:
        """Flat index of the grid cell containing room point ``(x, y)`` (clamped)."""
        col = int(np.clip(x / self.resolution, 0, self.n_cols - 1))
        row = int(np.clip(y / self.resolution, 0, self.n_rows - 1))
        return row * self.n_cols + col

    def matched_filter(self, node: int, cell_idx: int) -> np.ndarray:
        """Conjugated unit-norm steering row for (node, cell), ``[n_rx * n_sub]``.

        Projecting one node's flattened dynamic CSI onto this vector yields a
        complex time series spatially filtered toward that cell — the basis
        of per-track vital-sign extraction.
        """
        return self._steering_h[node][cell_idx]

    def update(self, dynamic_window: np.ndarray) -> np.ndarray:
        """Fold one window of dynamic CSI into the occupancy grid.

        Args:
            dynamic_window: Complex array ``[T, n_rx, n_sub]`` of
                background-subtracted, background-aligned CSI frames (the
                pipeline produces this via ``align_to_background`` +
                ``BackgroundSubtractor``; this class never touches raw CSI).
                A leading-1 TX axis (``[T, n_rx, 1, n_sub]``) is tolerated and
                squeezed.

        Returns:
            The updated smoothed grid, shape ``[n_rows, n_cols]``, 0..1.
        """
        x = np.asarray(dynamic_window)
        if x.ndim == 4 and x.shape[2] == 1:  # tolerate an un-squeezed n_tx axis
            x = x[:, :, 0, :]
        if x.ndim != 3:
            raise ValueError(
                f"dynamic_window must be [T, total_rx, n_sub], got shape {x.shape}"
            )
        n_frames = x.shape[0]
        flat = x.reshape(n_frames, -1)  # [T, F_total] (rx slow, subcarrier fast)
        self._last_total_power = float(np.mean(np.abs(flat) ** 2)) if flat.size else 0.0

        # Per-node activity statistics for the gate (see _track_noise_floor).
        node_flats = []
        node_powers = np.zeros(self.n_nodes)
        node_coh = np.zeros(self.n_nodes)
        for node in range(self.n_nodes):
            sl = slice(node * self._per_node_rx, (node + 1) * self._per_node_rx)
            nf = x[:, sl, :].reshape(n_frames, -1)  # [T, F_node]
            node_flats.append(nf)
            node_powers[node] = float(np.mean(np.abs(nf) ** 2)) if nf.size else 0.0
            node_coh[node] = self._temporal_coherence(nf)
        active = self._track_noise_floor(node_powers, node_coh)

        if active and n_frames > 0:
            # Bartlett matched-field power per node (coherent within a node),
            # fused across nodes in the LOG domain (weighted product of the
            # per-node likelihood maps). Sum fusion lets the highest-power
            # node dominate, so its range ridge survives in the fused map;
            # a product requires a cell to score well at EVERY node that
            # sees signal, which is exactly the geometric intersection of
            # the per-node ridges — the point of having a mesh. Weights are
            # each node's relative dynamic power (a node that barely sees
            # the scene contributes little either way), and a small floor
            # keeps unseen cells from collapsing the product to -inf. With
            # one node this reduces to the plain Bartlett map (w = 1).
            log_fused = np.zeros(self.n_cells)
            w_total = float(np.max(node_powers)) + 1e-30
            for node in range(self.n_nodes):
                projection = self._steering_h[node] @ node_flats[node].T  # [n_cells, T]
                p = np.mean(np.abs(projection) ** 2, axis=1)
                p_norm = p / (float(p.max()) + 1e-30)
                weight = float(node_powers[node]) / w_total
                log_fused += weight * np.log(p_norm + 0.05)
            power = np.exp(log_fused)
            power[self._radio_mask] = 0.0  # degenerate near-field cells
            raw = self._normalise(power).reshape(self.n_rows, self.n_cols)
        else:
            # Below the noise floor: feed zeros so the EMA drains the map and
            # an empty room shows an empty grid.
            raw = np.zeros((self.n_rows, self.n_cols), dtype=float)

        if self._have_grid:
            self.grid = self.ema * raw + (1.0 - self.ema) * self.grid
        else:
            self.grid = raw
            self._have_grid = True
        return self.grid

    def detect_peaks(
        self,
        max_peaks: int,
        min_separation: float = 0.8,
        min_value: float = 0.45,
        rel_threshold: float = 0.7,
        min_bearing_sep: float = 0.35,
    ) -> list[tuple[float, float, float]]:
        """Greedy non-max suppression on the current grid.

        Cells are visited strongest-first; a cell becomes a peak if it clears
        ``min_value``, clears ``rel_threshold`` times the strongest peak, and
        is separated from every accepted peak both by ``min_separation``
        metres *and* by ``min_bearing_sep`` radians of bearing as seen from
        the RX array.

        The bearing gate is what suppresses the map's dominant artefact: with
        20 MHz of bandwidth the bistatic range resolution (~7.5 m) exceeds
        the room, so a single person produces a near-constant ridge of cells
        along their bearing from the array. Euclidean suppression alone
        leaves several ridge cells more than ``min_separation`` apart, which
        the tracker then confirms as a queue of phantom people. The flip side
        is physical honesty: two real people at the same bearing but
        different ranges are genuinely unresolvable by a 3-element array on
        one 20 MHz link, and this detector reports the stronger one.

        Args:
            max_peaks: Maximum number of peaks to return.
            min_separation: Minimum Euclidean distance between peaks, metres.
            min_value: Minimum grid value (0..1) for a cell to qualify.
            rel_threshold: Minimum value as a fraction of the strongest
                accepted peak (0 disables the relative gate).
            min_bearing_sep: Minimum |bearing difference| (radians, wrapped)
                from every accepted peak. The default ~20 degrees sits well
                inside the ~1 rad beamwidth of the default 3-element
                half-wavelength array (0 disables the bearing gate).

        Returns:
            ``[(x_m, y_m, strength), ...]`` strongest first; possibly empty.
        """
        if max_peaks <= 0:
            return []
        flat = self.grid.ravel()
        candidates = np.flatnonzero(flat >= min_value)
        if candidates.size == 0:
            return []
        # Stable sort for determinism when values tie.
        order = candidates[np.argsort(flat[candidates], kind="stable")[::-1]]

        peaks: list[tuple[float, float, float]] = []
        peak_xy = np.empty((0, 2), dtype=float)
        peak_bearing: list[float] = []
        for idx in order:
            if len(peaks) >= max_peaks:
                break
            if peaks and flat[idx] < rel_threshold * peaks[0][2]:
                break  # order is strongest-first: nothing later qualifies
            pos = self.cell_centres[idx]
            if peak_xy.shape[0]:
                if np.min(np.linalg.norm(peak_xy - pos[None, :], axis=1)) < min_separation:
                    continue
                # Bearing suppression is a SINGLE-node necessity (one link's
                # range ridge). With a multistatic mesh the ridges intersect
                # into compact blobs, and the gate would wrongly suppress a
                # real second person who happens to share node-0's bearing.
                if self.n_nodes == 1:
                    b = self._cell_bearings[idx]
                    db = np.abs(np.angle(np.exp(1j * (np.asarray(peak_bearing) - b))))
                    if float(np.min(db)) < min_bearing_sep:
                        continue
            peaks.append((float(pos[0]), float(pos[1]), float(flat[idx])))
            peak_xy = np.vstack([peak_xy, pos[None, :]])
            peak_bearing.append(float(self._cell_bearings[idx]))
        return peaks

    # ------------------------------------------------------------------ #
    # internals                                                          #
    # ------------------------------------------------------------------ #

    def _track_noise_floor(
        self, node_powers: np.ndarray, node_coherence: np.ndarray
    ) -> bool:
        """Update per-node noise floors; return True if any node is active.

        Minimum-statistics style tracker per node: each floor snaps down to
        any power dip immediately and rises only slowly (~10x slower while
        the gate is open, since the observed power then measures *signal*,
        not noise). A window counts as active when ANY node's power exceeds
        ``NOISE_GATE_FACTOR`` times that node's floor, *or* when any node's
        temporal coherence alone proves structured signal (the cold-start
        escape hatch -- see the class docstring).

        Per-node rather than summed on purpose: a person close to node 2 but
        far from node 0 must open the gate via node 2, and each node's
        breathing-power troughs (which calibrate the minimum-tracking floor)
        occur at different times, so the summed power never reaches the true
        noise level while any one node's power does.
        """
        eps = 1e-30
        if self._noise_floor is None:
            # First window: conservative init at the observed power. If the
            # room is genuinely empty this is exactly the noise level; if a
            # person is already moving, the coherence gate below still opens
            # the map and the floors snap down at the first quiet window.
            self._noise_floor = node_powers + eps
        active = bool(
            np.any(node_powers > self.NOISE_GATE_FACTOR * self._noise_floor)
            or np.any(node_coherence > self.COHERENCE_GATE)
        )
        alpha_up = 0.01 if active else 0.1
        risen = self._noise_floor + alpha_up * (node_powers - self._noise_floor)
        # Snap down to new minima elementwise; rise slowly otherwise.
        self._noise_floor = np.where(node_powers < self._noise_floor, node_powers + eps, risen)
        return active

    @staticmethod
    def _temporal_coherence(flat: np.ndarray) -> float:
        """Normalised lag-1 correlation of the window across time, in 0..1.

        The per-(rx, subcarrier) temporal mean of the window is removed first
        so a constant residual from an imperfect background estimate (which
        is perfectly self-correlated but carries no motion information)
        cannot open the gate. What remains is the within-window *variation*:
        for a moving (or breathing) person consecutive 5 ms frames are nearly
        identical, giving a coherence near 1; white noise gives ~0.
        """
        if flat.shape[0] < 4 or flat.size == 0:
            return 0.0
        centred = flat - flat.mean(axis=0, keepdims=True)
        a, b = centred[:-1], centred[1:]
        num = np.abs(np.sum(np.conj(a) * b))
        den = np.sqrt(
            float(np.sum(np.abs(a) ** 2)) * float(np.sum(np.abs(b) ** 2))
        )
        if den <= 1e-30:
            return 0.0
        return float(num / den)

    @staticmethod
    def _normalise(power: np.ndarray) -> np.ndarray:
        """Map raw Bartlett power to 0..1, preserving the peak ordering.

        Subtract the median (the diffuse multipath / noise pedestal that the
        beamformer spreads over every cell) and scale by the spread up to the
        maximum. The median keeps a flat (empty) map near zero instead of
        amplifying its noise texture; scaling by the max — rather than a high
        percentile — is deliberate: a percentile cap clips the entire peak
        region to a flat tie at 1.0, after which argmax/peak ordering decay
        into row-major index order and side lobes can outrank the true peak.
        """
        med = float(np.median(power))
        scale = float(power.max()) - med
        if scale <= 1e-30:
            return np.zeros_like(power)
        return np.clip((power - med) / scale, 0.0, 1.0)
