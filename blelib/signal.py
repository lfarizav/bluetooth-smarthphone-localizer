"""Turn a stream of raw RSSI readings into the numbers a human can act on.

This module is the whole reason the meter is trustworthy at all. Bluetooth
SIG's own guidance (verified 2026-08-06,
https://www.bluetooth.com/blog/proximity-and-rssi/) is blunt: RSSI has no
standardised relationship to distance, varies by chipset, and should be read
as a *trend*, never an absolute. Every function here follows from that:

* :func:`median_rssi` -- never the mean. A single wall reflection can spike
  one packet by 10+ dB; a median shrugs it off where a mean carries it
  straight into the display.
* :func:`trend_of` -- compares two medians, not two raw samples, and refuses
  to answer (``Trend.UNKNOWN``) until both windows hold enough evidence.
  Guessing off one packet is how a meter sends someone the wrong way down a
  corridor.
* :func:`distinct_measurements` -- a BlueZ poller that returns its last
  cached value looks identical, on the wire, to a fresh confirming read. Only
  a value *change* is new information; anything else is padding the count.

None of this needs a radio, a clock, or Flask -- see ``blelib/__init__.py``
and SPEC section 4 for why that split is load-bearing for the whole lab.
"""

from __future__ import annotations

import enum
import statistics
from dataclasses import dataclass
from typing import Sequence

from .reading import Reading

#: What the displayed number is drawn from. Short enough that walking a few
#: steps changes it, long enough that one reflected packet does not dominate
#: the median of the window it lands in.
LIVE_WINDOW_S: float = 4.0

#: The comparison window immediately behind the live one. Three times the
#: live window gives the "before" median a comparable sample count without
#: reaching so far back that a WARMER call is really describing where the
#: student was a minute ago.
TREND_WINDOW_S: float = 12.0

#: dB of median-to-median change before the trend arrow moves off STEADY.
#: Set below the Core Spec's own +/-6 dB `HCI_Read_RSSI` accuracy budget and
#: it will flicker on measurement noise alone; this is the decision-task
#: default a student is meant to question, not treat as correct.
TREND_THRESHOLD_DB: int = 3


@dataclass(frozen=True, slots=True)
class ProximityBand:
    """One rung of the proximity ladder shown on the meter.

    ``floor_dbm`` is the *lower* bound of the band -- a reading qualifies for
    a band if it is at or above that floor and below every floor above it.
    Both ``es`` and ``en`` are mandatory on every rung; a hardcoded English
    string anywhere on this ladder is exactly the bug SPEC section 9 forbids.
    """

    floor_dbm: float
    key: str
    es: str
    en: str


#: Ordered high to low; the last floor is -inf so every finite RSSI value
#: always lands in exactly one band. Thresholds are not a measured constant
#: of BLE -- they are a rough, openly-approximate ladder for "which way to
#: walk", consistent with the lab's honest finding that RSSI is not a ruler.
PROXIMITY_BANDS: tuple[ProximityBand, ...] = (
    ProximityBand(floor_dbm=-45.0, key="arms_reach",
                  es="A tu alcance", en="Arm's reach"),
    ProximityBand(floor_dbm=-60.0, key="same_table",
                  es="Misma mesa", en="Same table"),
    ProximityBand(floor_dbm=-72.0, key="same_room",
                  es="Misma habitación", en="Same room"),
    ProximityBand(floor_dbm=-85.0, key="far",
                  es="Lejos / detrás de un obstáculo", en="Far / behind cover"),
    ProximityBand(floor_dbm=float("-inf"), key="very_far",
                  es="Muy lejos / blindado", en="Very far / shielded"),
)

#: Sentinel used INSTEAD OF a real PROXIMITY_BANDS entry whenever the newest
#: reading is not in dBm -- concretely, link-mode Golden Receive Power Range
#: readings (``blelib/hci.py``, ``reading.UNIT_GOLDEN_RANGE_DB``). The floors
#: above are calibrated to this lab's own indoor dBm scale (README §2.1);
#: running them against a golden-range number would silently relabel an
#: honest "how far outside the golden range" reading as a distance claim it
#: never was. ``hunt.py``'s units-aware branch returns this instead of
#: calling :func:`band_for` whenever ``Reading.units != "dbm"``.
NOT_APPLICABLE_BAND = ProximityBand(
    floor_dbm=float("-inf"), key="not_applicable",
    es="No aplica (unidades de enlace, no dBm)",
    en="Not applicable (link units, not dBm)")

#: :func:`fraction` / :func:`sparkline_levels` span for link-mode Golden
#: Receive Power Range readings -- the dBm defaults (-100..-30) would clip
#: almost every golden-range value to one end of the bar. Not derived from
#: any spec constant (the Core Spec does not bound how far outside the
#: golden range a controller may report); sized with headroom around the one
#: real walk test this lab has evidence for
#: (``evidence/link-walk-2026-08-06.json``: -8..+6 over a real 2-minute
#: walk). This only positions the bar/sparkline honestly -- it does NOT turn
#: the value into a distance estimate, which is exactly why band_for is
#: never called for these units (see NOT_APPLICABLE_BAND above).
GOLDEN_RANGE_LO: float = -20.0
GOLDEN_RANGE_HI: float = 20.0


def band_index(rssi_dbm: float) -> int:
    """Index into :data:`PROXIMITY_BANDS` for a raw RSSI value.

    Linear scan over five entries beats a bisect setup for a table this
    short, and keeps the "ordered high to low, last floor -inf" invariant
    trivially visible at the call site.
    """
    for i, band in enumerate(PROXIMITY_BANDS):
        if rssi_dbm >= band.floor_dbm:
            return i
    return len(PROXIMITY_BANDS) - 1  # unreachable: the last floor is -inf


def band_for(rssi_dbm: float) -> ProximityBand:
    return PROXIMITY_BANDS[band_index(rssi_dbm)]


def band_label(rssi_dbm: float, lang: str = "en") -> str:
    band = band_for(rssi_dbm)
    return band.es if lang == "es" else band.en


def median_rssi(readings: Sequence[Reading]) -> float | None:
    """The true median of ``rssi_dbm`` across ``readings``, or ``None`` if empty.

    Never the mean -- see the module docstring. ``statistics.median`` is used
    rather than a hand-rolled sort-and-index because it already does the
    even/odd averaging correctly and is exactly as fast as this list ever
    needs (the live window is a handful of samples).
    """
    if not readings:
        return None
    return float(statistics.median(r.rssi_dbm for r in readings))


def peak_rssi(readings: Sequence[Reading]) -> int | None:
    """Strongest (least negative) single reading, or ``None`` if empty."""
    if not readings:
        return None
    return max(r.rssi_dbm for r in readings)


def fraction(rssi_dbm: float, lo: float = -100.0, hi: float = -30.0) -> float:
    """Position of ``rssi_dbm`` in ``[lo, hi]``, clamped to 0..1.

    Feeds the bar and sparkline, both of which need a value they can draw
    unconditionally -- a reading outside the "useful span" pins to an end of
    the bar rather than under- or overflowing it.
    """
    span = hi - lo
    if span <= 0:
        return 0.0
    f = (rssi_dbm - lo) / span
    return max(0.0, min(1.0, f))


class Trend(enum.Enum):
    WARMER = "warmer"
    COLDER = "colder"
    STEADY = "steady"
    UNKNOWN = "unknown"


def trend_of(readings: Sequence[Reading], now: float, *,
             live_window: float = LIVE_WINDOW_S,
             trend_window: float = TREND_WINDOW_S,
             threshold_db: int = TREND_THRESHOLD_DB) -> Trend:
    """Median of the recent window vs. the median of the window before it.

    Two medians, not two raw samples, and not "now vs. one packet ago": a
    single-packet comparison is exactly the reflection-yanks-the-display
    failure mode :func:`median_rssi` exists to prevent, just moved one layer
    up. ``UNKNOWN`` until *both* windows hold at least two samples -- a hunt
    that has just started, or has gone quiet, has no business claiming a
    direction.
    """
    recent = [r for r in readings if now - live_window <= r.t <= now]
    prior = [r for r in readings
             if now - trend_window <= r.t < now - live_window]
    if len(recent) < 2 or len(prior) < 2:
        return Trend.UNKNOWN

    recent_med = median_rssi(recent)
    prior_med = median_rssi(prior)
    delta = recent_med - prior_med
    if delta >= threshold_db:
        return Trend.WARMER
    if delta <= -threshold_db:
        return Trend.COLDER
    return Trend.STEADY


def is_stale(readings: Sequence[Reading], now: float, timeout_s: float) -> bool:
    """True once more than ``timeout_s`` has passed since the newest reading.

    Relies on the append-order invariant readings arrive in (see
    ``ReadingLog``): the newest reading is the last element, so no scan or
    ``max()`` over the whole sequence is needed.
    """
    if not readings:
        return True
    return (now - readings[-1].t) > timeout_s


def distinct_measurements(readings: Sequence[Reading]) -> int:
    """Count RUNS of identical values, not raw polls and not bare transitions.

    A poller that returns a cached number does not produce information --
    same reasoning as findphone's macOS-cache note (its README, design fact
    only, re-derived independently here for BlueZ; see SPEC section 0 on why
    no code from that project is a permitted input). Poll 112 times and get
    runs of 8-31 identical values back, and the honest number of
    measurements is the number of distinct *runs*: each run started with one
    genuine observation, and every repeat within that run is the cache
    handing back the same answer again, carrying no new information.

    Concretely: the *first* reading in the whole sequence is itself a real
    observation and always counts (a track of one value is one measurement,
    not zero); after that, each value that differs from the one before it
    starts a new run and adds one more. So this is "number of runs", which
    is "number of value-to-value transitions" plus one for a non-empty
    sequence -- not the transition count on its own::

        []                    -> 0
        [-60]                 -> 1     (one real observation)
        [-60, -60, -60]       -> 1     (one observation, cached twice)
        [-60, -61, -62]       -> 3
        [-60, -60, -61, -61]  -> 2
    """
    if not readings:
        return 0
    count = 1
    prev = readings[0].rssi_dbm
    for r in readings[1:]:
        if r.rssi_dbm != prev:
            count += 1
        prev = r.rssi_dbm
    return count


def sparkline_levels(readings: Sequence[Reading], n_levels: int = 8,
                      lo: float = -100.0, hi: float = -30.0) -> list[int]:
    """Map each reading to a bar level 0..``n_levels - 1``.

    Pure mapping only -- the browser owns the actual glyph/height rendering
    (SPEC 5.9). Keeping the mapping here means the sparkline logic is
    unit-testable without a page.
    """
    top = max(n_levels - 1, 0)
    return [int(round(fraction(r.rssi_dbm, lo, hi) * top)) for r in readings]
