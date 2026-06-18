"""Adapters for real CSI-capable hardware.

Three families of commodity hardware can emit CSI:

* **ESP32 + ESP32-CSI-Tool** (https://github.com/StevenMHernandez/ESP32-CSI-Tool):
  the ESP32 prints one CSV line per received packet over USB serial.
  :class:`ESP32CSISource` is fully implemented (the line parser
  :func:`parse_esp32_csi_line` is pure and unit-testable; only the serial
  port itself needs hardware).
* **Broadcom + Nexmon CSI** (Raspberry Pi 3B+/4, bcm43455c0 etc.):
  :class:`NexmonCSISource` — wire format documented below, capture loop left
  for a contributor with the hardware.
* **Intel 5300 + linux-80211n-csitool**: :class:`IntelCSISource` — log
  format documented below, parser left for a contributor.

All adapters produce :class:`~wifi_room_radar.types.CSIFrame` with
``ground_truth=None`` (there is no oracle in the real world) and timestamps
rebased to seconds since stream start, matching the simulator's convention so
the downstream pipeline is source-agnostic.
"""
from __future__ import annotations

import time
from typing import Iterator, Optional

import numpy as np

from ..config import RadioConfig
from ..types import CSIFrame
from .base import CSISource

__all__ = [
    "parse_esp32_csi_line",
    "ESP32CSISource",
    "NexmonCSISource",
    "IntelCSISource",
]


# ---------------------------------------------------------------------------
# ESP32 (ESP32-CSI-Tool)
# ---------------------------------------------------------------------------

def parse_esp32_csi_line(line: str) -> Optional[CSIFrame]:
    """Parse one ESP32-CSI-Tool serial CSV line into a :class:`CSIFrame`.

    ESP32-CSI-Tool prints one line per received packet::

        CSI_DATA,<role>,<mac>,<rssi>,<rate>,<sig_mode>,<mcs>,<bandwidth>,
        <smoothing>,<not_sounding>,<aggregation>,<stbc>,<fec_coding>,<sgi>,
        <noise_floor>,<ampdu_cnt>,<channel>,<secondary_channel>,
        <local_timestamp>,<ant>,<sig_len>,<rx_state>,<real_time_set>,
        <real_timestamp>,<len>,"[i0 q0 i1 q1 ...]"

    (single physical line; field 0 is the literal tag ``CSI_DATA``). The
    trailing bracketed block is a space-separated array of small signed
    integers — interleaved in-phase / quadrature pairs, one pair per
    subcarrier. This function maps pair ``(i, q)`` to the complex sample
    ``i + 1j*q`` and returns CSI of shape ``[1, 1, n_sub]`` (the ESP32 is a
    single-antenna, single-chain radio; ``n_sub`` is whatever the firmware
    emitted, typically 64 for HT20 including guard/null subcarriers).

    Best-effort metadata extraction:

    * ``rssi`` from CSV field 3 (dBm), ``0.0`` if absent/unparseable.
    * ``timestamp`` from field 23 (``real_timestamp``, seconds) when set,
      else field 18 (``local_timestamp``, microseconds since boot) / 1e6,
      else ``0.0``. The caller (:meth:`ESP32CSISource.frames`) rebases this
      to seconds-since-stream-start.
    * ``seq`` is always 0 here; the streaming loop assigns monotonically
      increasing sequence numbers.

    Returns ``None`` for non-CSI lines (boot logs, prompts), truncated lines,
    or malformed integer arrays — the serial stream is noisy in practice and
    the caller just skips those lines.
    """
    line = line.strip()
    if not line.startswith("CSI_DATA"):
        return None
    lb = line.find("[")
    rb = line.rfind("]")
    if lb < 0 or rb <= lb:
        return None
    try:
        values = [int(tok) for tok in line[lb + 1 : rb].split()]
    except ValueError:
        return None
    if len(values) < 2 or len(values) % 2 != 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    csi = (arr[0::2] + 1j * arr[1::2]).astype(np.complex128).reshape(1, 1, -1)

    fields = line[:lb].split(",")
    rssi = 0.0
    if len(fields) > 3:
        try:
            rssi = float(fields[3])
        except ValueError:
            pass
    timestamp = 0.0
    if len(fields) > 23:
        try:
            timestamp = float(fields[23])  # real_timestamp, seconds
        except ValueError:
            pass
    if timestamp <= 0.0 and len(fields) > 18:
        try:
            timestamp = float(fields[18]) / 1e6  # local_timestamp, us
        except ValueError:
            pass
    return CSIFrame(timestamp=timestamp, seq=0, csi=csi, rssi=rssi, ground_truth=None)


class ESP32CSISource(CSISource):
    """Live CSI from an ESP32 running ESP32-CSI-Tool, over USB serial.

    Hardware setup:

    1. Flash an ESP32 dev board with ESP32-CSI-Tool (``active_sta``,
       ``active_ap`` or ``passive`` project; ``idf.py flash``).
    2. Plug it in over USB; note the serial port (Windows: ``COM5``-style,
       Linux: ``/dev/ttyUSB0``, macOS: ``/dev/cu.usbserial-*``).
    3. The tool prints at 921600 baud by default.

    Args:
        port: Serial port name.
        radio: Link config. Defaults to a single-chain 64-subcarrier 2.4 GHz
            HT20 setup matching the ESP32's radio; positions/geometry should
            be edited to match your physical deployment for the mapping
            stages to mean anything.
        baudrate: Serial baud rate (ESP32-CSI-Tool default 921600).

    ``pyserial`` is imported lazily inside :meth:`frames` so the rest of the
    package works without it installed.
    """

    def __init__(
        self,
        port: str,
        radio: Optional[RadioConfig] = None,
        baudrate: int = 921600,
    ) -> None:
        if radio is None:
            radio = RadioConfig(
                carrier_freq=2.437e9,  # 2.4 GHz channel 6 (ESP32 is 2.4 GHz only)
                bandwidth=20e6,
                n_subcarriers=64,
                n_rx=1,
                n_tx=1,
                sample_rate=100.0,
            )
        super().__init__(radio)
        self.port = port
        self.baudrate = baudrate
        self._serial = None
        self._closed = False

    def frames(self) -> Iterator[CSIFrame]:
        """Read, parse and yield CSI lines from the serial port forever.

        Raises:
            RuntimeError: If ``pyserial`` is not installed, with install and
                wiring instructions.
        """
        try:
            import serial  # lazy: pyserial is an optional dependency
        except ImportError as exc:
            raise RuntimeError(
                "ESP32CSISource needs the 'pyserial' package.\n"
                "  Install it with:   pip install pyserial\n"
                "Hardware checklist:\n"
                "  1. Flash an ESP32 with ESP32-CSI-Tool "
                "(https://github.com/StevenMHernandez/ESP32-CSI-Tool).\n"
                "  2. Connect it over USB and find the port "
                "(Windows: Device Manager -> COMx; Linux: /dev/ttyUSB0).\n"
                "  3. Construct ESP32CSISource(port='COM5') (or your port) "
                "with baudrate 921600 (the tool's default).\n"
            ) from exc

        self._serial = serial.Serial(self.port, self.baudrate, timeout=1.0)
        seq = 0
        t0: Optional[float] = None
        try:
            while not self._closed:
                raw = self._serial.readline()
                if not raw:
                    continue  # timeout — keep polling until close()
                frame = parse_esp32_csi_line(raw.decode("utf-8", errors="replace"))
                if frame is None:
                    continue
                # Rebase timestamps to seconds since stream start. Fall back
                # to the host monotonic clock for firmware builds that do not
                # report usable timestamps.
                ts = frame.timestamp if frame.timestamp > 0.0 else time.monotonic()
                if t0 is None:
                    t0 = ts
                frame.timestamp = ts - t0
                frame.seq = seq
                seq += 1
                yield frame
        finally:
            self.close()

    def close(self) -> None:
        """Close the serial port and stop the stream. Idempotent."""
        self._closed = True
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None


# ---------------------------------------------------------------------------
# Broadcom / Nexmon CSI
# ---------------------------------------------------------------------------

class NexmonCSISource(CSISource):
    """CSI from Broadcom radios patched with Nexmon CSI. **Stub.**

    Nexmon CSI (https://github.com/seemoo-lab/nexmon_csi) patches the
    firmware of Broadcom chips (Raspberry Pi 3B+/4: bcm43455c0; Asus
    RT-AC86U: bcm4366c0; various smartphones) to extract per-packet CSI. The
    patched firmware *broadcasts each CSI measurement as a UDP datagram to
    255.255.255.255:5500*, so capture is just a UDP socket bound to port
    5500 on the device (or a host sniffing its traffic / reading a pcap of
    it).

    UDP payload wire format (per seemoo-lab/nexmon_csi README and
    ``utils/matlab/unpack_float.m``), all multi-byte fields little-endian::

        offset  size  field
        ------  ----  -----------------------------------------------------
        0       2     magic bytes 0x1111
        2       1     RSSI (signed int8, dBm)
        3       1     frame control byte of the sniffed packet
        4       6     source MAC of the sniffed packet
        10      2     packet sequence number
        12      2     core and spatial-stream index (core = bits 0-2,
                      spatial stream = bits 3-5)
        14      2     chanspec (channel + bandwidth encoding)
        16      2     chip version
        18      ...   CSI data, n_sub complex values; n_sub = 64/128/256
                      for 20/40/80 MHz.
                      * bcm43455c0: interleaved int16 pairs (real, imag)
                      * bcm4366c0: packed custom floating point; decode per
                        unpack_float.m (9-bit mantissas, shared exponent)

    Note that one *physical* packet produces one datagram per (core,
    spatial-stream) pair; a full ``[n_rx, n_tx, n_sub]`` CSI matrix must be
    assembled by grouping datagrams with the same sequence number.

    What a contributor with the hardware needs to implement:

    * ``parse_nexmon_payload(payload: bytes, chip: str) -> tuple[int, int, int, np.ndarray, float]``
      — decode one datagram into ``(seq, core, spatial_stream, csi_vector,
      rssi)`` following the table above.
    * ``_assemble(self, parts) -> CSIFrame`` — group datagrams by sequence
      number into the full ``[n_rx, n_tx, n_sub]`` matrix (emit when all
      core/stream pairs arrived or on timeout).
    * the UDP receive loop in :meth:`frames` (``socket.socket(AF_INET,
      SOCK_DGRAM)``, bind ``(listen_addr, port)``, ``recvfrom``; rebase
      timestamps to stream start, honour ``self._closed``).

    Args:
        radio: Link config matching the patched device (e.g. ``n_rx=1`` for
            a Raspberry Pi, 5 GHz channel per your ``makecsiparams`` call).
        listen_addr: Address to bind the UDP socket to.
        port: UDP port the firmware sends to (5500 unless repatched).
        chip: Chip identifier selecting the CSI number format
            ("bcm43455c0" or "bcm4366c0").
    """

    def __init__(
        self,
        radio: RadioConfig,
        listen_addr: str = "0.0.0.0",
        port: int = 5500,
        chip: str = "bcm43455c0",
    ) -> None:
        super().__init__(radio)
        self.listen_addr = listen_addr
        self.port = port
        self.chip = chip
        self._closed = False

    def frames(self) -> Iterator[CSIFrame]:
        """Not implemented — needs a Nexmon-patched device to develop against."""
        raise NotImplementedError(
            "NexmonCSISource is a documented stub: the UDP wire format is "
            "described in the class docstring, but the capture loop needs a "
            "Nexmon-CSI-patched Broadcom device (e.g. Raspberry Pi 4 with "
            "seemoo-lab/nexmon_csi) to develop and test against. Implement "
            "parse_nexmon_payload() and the UDP receive loop per the class "
            "docstring, or use SimulatedCSISource / ReplayCSISource instead."
        )

    def close(self) -> None:
        self._closed = True


# ---------------------------------------------------------------------------
# Intel 5300 / linux-80211n-csitool
# ---------------------------------------------------------------------------

class IntelCSISource(CSISource):
    """CSI from the Intel 5300 NIC (linux-80211n-csitool logs). **Stub.**

    The classic CSI platform (Halperin et al.,
    https://dhalperi.github.io/linux-80211n-csitool/): a modified iwlwifi
    driver delivers "beamforming feedback" records which ``log_to_file``
    writes to a binary ``.dat`` file. This class is intended to parse such a
    file (live netlink capture would be a separate Linux-only effort).

    ``.dat`` log format — a sequence of records::

        2 bytes   record length N (big-endian, excludes these 2 bytes)
        1 byte    record code; 0xBB = beamforming matrix (CSI), others skip
        N-1 bytes record payload

    Payload of a code-0xBB record (little-endian, per the csitool's
    ``read_bfee.c``)::

        offset  size  field
        ------  ----  -----------------------------------------------------
        0       4     timestamp_low: 1 MHz NIC clock (wraps ~72 min)
        4       2     bfee_count: running record counter
        6       2     reserved
        8       1     Nrx (antennas, 1-3)
        9       1     Ntx (space-time streams, 1-3)
        10      1     rssi_a   (raw RSSI, antenna A)
        11      1     rssi_b
        12      1     rssi_c
        13      1     noise (signed int8, dBm)
        14      1     agc (dB)
        15      1     antenna_sel: RX antenna permutation, 2 bits per chain
        16      2     len: payload bytes of the CSI matrix
        18      2     fake_rate_n_flags
        20      len   CSI matrix, *bit-packed*: for each of the 30
                      subcarrier groups: skip 3 bits, then for each of the
                      Nrx*Ntx entries read signed 8-bit real and imag parts
                      at the current (arbitrary, non-byte-aligned) bit
                      offset. Total bits = 30 * (3 + 16 * Nrx * Ntx).

    Post-processing required for physically meaningful CSI (mirrors the
    csitool's MATLAB helpers):

    * permute RX rows according to ``antenna_sel`` (``hex2dec`` pairs),
    * scale to absolute linear units using rssi_a/b/c, ``agc``, ``noise``
      and Ntx (see ``get_scaled_csi.m``: total RSS in dBm minus 44 dB and
      AGC, divided by the matrix's total power, with an SNR-dependent noise
      correction),
    * note the 5300 reports 30 *grouped* subcarriers for both 20 and
      40 MHz — set ``radio.n_subcarriers = 30``.

    What a contributor needs to implement:

    * ``read_bfee(payload: bytes) -> dict`` — the bit-unpacking described
      above (a numpy implementation: build a bit array with
      ``np.unpackbits`` and gather the 8-bit fields at computed offsets).
    * ``scale_csi(entry: dict) -> np.ndarray`` — port of
      ``get_scaled_csi.m`` returning ``[n_rx, n_tx, 30]`` complex128.
    * the record loop in :meth:`frames` — walk the ``.dat`` file, skip
      non-0xBB records, unwrap ``timestamp_low`` (1 MHz, 32-bit) into
      seconds since stream start, yield frames.

    Args:
        path: ``.dat`` log file produced by
            ``log_to_file``/``netlink_to_file``.
        radio: Link config (``n_subcarriers`` should be 30; ``n_rx``/``n_tx``
            per your antenna setup).
    """

    def __init__(self, path: str, radio: Optional[RadioConfig] = None) -> None:
        if radio is None:
            radio = RadioConfig(n_subcarriers=30, n_rx=3, n_tx=1)
        super().__init__(radio)
        self.path = path
        self._closed = False

    def frames(self) -> Iterator[CSIFrame]:
        """Not implemented — needs the bit-packed bfee parser (see docstring)."""
        raise NotImplementedError(
            "IntelCSISource is a documented stub: the linux-80211n-csitool "
            ".dat record layout and the bit-packed bfee matrix format are "
            "described in the class docstring, but the parser (read_bfee + "
            "get_scaled_csi ports) still needs to be written and validated "
            "against real Intel 5300 logs. Implement them per the docstring, "
            "or use SimulatedCSISource / ReplayCSISource instead."
        )

    def close(self) -> None:
        self._closed = True
