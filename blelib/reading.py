"""One BLE RSSI sample, and the ring buffer that holds a stream of them.

Everything in this lab that touches a real radio -- ``scan.py``, ``hci.py``,
``weblive.py`` -- ultimately reduces to a stream of :class:`Reading`. Keeping
that type tiny and frozen means the whole rest of the pure pipeline
(``signal.py``, ``pathloss.py``, ``hunt.py``) can be exercised with a plain
Python list built by hand, with no BlueZ, no event loop, and no socket in the
loop at all.

``ReadingLog`` is deliberately *not* a general-purpose time-series container.
A live hunt session can run for hours at 3-5 readings/second; without a bound
the process grows without limit for a demo nobody remembers to stop. A
``deque(maxlen=...)`` gives O(1) append with automatic eviction of the oldest
sample, for free, from the standard library -- no reason to hand-roll it.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

#: What ``rssi_dbm`` actually is, for the two units this lab ever produces.
#: See :attr:`Reading.units` -- every consumer that turns ``rssi_dbm`` into a
#: distance/band/path-loss claim MUST check this field before doing so.
UNIT_DBM = "dbm"
UNIT_GOLDEN_RANGE_DB = "golden_range_db"


@dataclass(frozen=True, slots=True)
class Reading:
    """One RSSI sample, as reported -- no smoothing, no correction, no units
    conversion. Everything downstream that *interprets* the number (medians,
    bands, trends, path-loss fits) lives in :mod:`signal` and
    :mod:`pathloss`; this dataclass only carries the fact of the measurement.
    """

    rssi_dbm: int
    """As reported by the source. Integer because that is what the BlueZ /
    HCI layers actually hand back -- inventing sub-dB precision here would
    misrepresent a number the Core Spec itself only promises to +/-6 dB.

    NOTE the name is a historical holdover from when every source this lab
    had was in dBm: it is NOT always dBm. Check :attr:`units` before
    interpreting the number -- see that attribute's docstring."""

    t: float
    """Monotonic seconds (``time.monotonic()`` domain), never wall-clock.
    Wall-clock can jump (NTP step, DST); a hunt session's windows (live vs.
    trend) must never see a fake gap or a fake overlap because the OS clock
    moved."""

    source: str
    """``"advert" | "link" | "sim" | "replay"`` -- kept as a plain string
    rather than an enum because it is round-tripped through SigMF JSON
    (``capture.py``) and displayed verbatim in the honest-counters line; a
    string needs no translation layer at either boundary."""

    addr: str = ""
    """Device address, already redacted by the caller if redaction is on.
    This module never redacts anything itself -- ``scan.py`` decides that,
    once, at the point a real BlueZ address is first seen. A pure module has
    no business making a privacy decision it cannot audit."""

    units: str = UNIT_DBM
    """What :attr:`rssi_dbm` actually IS -- the single most load-bearing field
    added for connected-link (classic BR/EDR/ACL) support (``blelib/hci.py``).

    ``"dbm"`` (:data:`UNIT_DBM`, the default -- the only value every source
    before link mode ever produced): real received signal strength in dBm,
    from a BLE advert (``scan.py``) or from ``HCI_Read_RSSI`` on an LE
    transport.

    ``"golden_range_db"`` (:data:`UNIT_GOLDEN_RANGE_DB`): ``HCI_Read_RSSI`` on
    a classic BR/EDR (ACL) link, which the Bluetooth Core Spec (Vol 4, Part
    E, section 7.5.4) defines completely differently from the LE case: the
    returned value is a SIGNED difference from the controller's Golden
    Receive Power Range. Zero means the RSSI is INSIDE the golden range; a
    negative value is how many dB it sits BELOW the range's lower limit; a
    positive value is how many dB it sits ABOVE the range's upper limit. It
    is a three-zone indicator with a wide dead zone around 0, NOT a dBm
    figure, and NOT comparable to anything else this lab shows -- see
    ``blelib/hci.py``'s module docstring for the full citation and the
    real 2026-08-06 walk-test evidence that proved the read itself works.

    Every function that turns ``rssi_dbm`` into a distance, a proximity
    band, or a path-loss fit (``signal.band_for``, ``pathloss.*``, the
    calibration panel in ``weblive.py``) MUST check this field first and
    refuse to run on anything but ``"dbm"`` -- treating a golden-range
    number as dBm would silently misrepresent what the instrument actually
    measured, which is the one thing this whole lab exists to never do."""


class ReadingLog:
    """Append-only, time-ordered, bounded ring buffer of :class:`Reading`.

    "Time-ordered" is an invariant the caller must uphold by only ever
    calling :meth:`append` with non-decreasing ``t`` -- readings arrive from
    a live source in arrival order, so this holds naturally. The payoff is
    that :meth:`since` can stop as soon as it walks past its cutoff instead
    of visiting every element, which matters once the buffer is thousands of
    samples deep and ``since`` is called every display tick.
    """

    def __init__(self, maxlen: int = 4096) -> None:
        self._buf: deque[Reading] = deque(maxlen=maxlen)

    def append(self, r: Reading) -> None:
        self._buf.append(r)

    def since(self, seconds: float, now: float) -> list[Reading]:
        """Readings with ``t >= now - seconds``, oldest first.

        Walks from the newest entry backwards and stops at the first reading
        older than the cutoff, rather than filtering the whole deque. The
        live window this feeds (`signal.LIVE_WINDOW_S`, a handful of
        seconds) is always a small tail of a buffer that can hold over an
        hour of history at BLE advertising rates -- scanning all of it on
        every tick would waste far more work than the buffer itself costs.
        """
        cutoff = now - seconds
        out: list[Reading] = []
        for r in reversed(self._buf):
            if r.t < cutoff:
                break
            out.append(r)
        out.reverse()
        return out

    def window(self, t0: float, t1: float) -> list[Reading]:
        """Readings with ``t0 <= t <= t1``, oldest first.

        Used for arbitrary historical slices (plots, calibration review)
        rather than the live-display path, so a full linear scan is fine --
        it is not on the hot loop that :meth:`since` is.
        """
        return [r for r in self._buf if t0 <= r.t <= t1]

    def __len__(self) -> int:
        return len(self._buf)

    @property
    def last(self) -> Reading | None:
        return self._buf[-1] if self._buf else None
