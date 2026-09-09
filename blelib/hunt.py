"""The hunt state machine -- turns a stream of Readings into one snapshot.

``weblive.py`` (SPEC 5.9) pushes ``HuntSnapshot`` over Server-Sent Events on
every tick; everything the browser draws (the big dBm readout, the band
label, the bar, the sparkline, the honest counters) is exactly the fields of
one snapshot. Keeping :meth:`HuntEngine.snapshot` pure in ``now`` -- no
``time.monotonic()`` call anywhere inside it -- is what lets that whole page
be tested without Flask, without SSE, and without a socket: a test just feeds
readings and asks for a snapshot at a chosen instant, the same way a browser
tick would.

This module owns none of the *interpretation* math (medians, bands, trends,
staleness, the honest measurement count) -- all of that lives in
:mod:`signal`, per SPEC section 4's "pure module" split. ``hunt.py`` only
decides which windows of the log to hand to each of those functions, and
assembles the result.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass

from .reading import UNIT_DBM, Reading, ReadingLog
from .signal import (
    GOLDEN_RANGE_HI,
    GOLDEN_RANGE_LO,
    LIVE_WINDOW_S,
    NOT_APPLICABLE_BAND,
    TREND_THRESHOLD_DB,
    TREND_WINDOW_S,
    band_for,
    distinct_measurements,
    fraction,
    is_stale,
    median_rssi,
    peak_rssi,
    sparkline_levels,
    trend_of,
)

__all__ = ["HuntConfig", "HuntSnapshot", "HuntEngine"]

#: History window fed to the sparkline -- long enough to draw a real shape
#: (well past the handful of samples the live median is drawn from), short
#: enough that the browser's little bar chart does not scroll back into
#: ancient history. Not in HuntConfig because it is a *display* choice, not
#: part of the decision task's committed settings (SPEC 6.1: only
#: ``live_window_s`` and ``trend_threshold_db`` are committed-before-you-look).
SPARK_WINDOW_S: float = 20.0

#: The "1 minute" in ``peak_1min_dbm`` / ``n_last_min`` is literal, not tied
#: to any other configurable window.
ONE_MINUTE_S: float = 60.0

#: How many of the target's most recent inter-arrival gaps
#: :meth:`HuntEngine._effective_stale_after_s` tracks to estimate the
#: source's own update period -- just enough to be robust to one dropped or
#: delayed packet without smoothing over a real cadence change (e.g. bredr's
#: own irregular ~10s+ cycle, bredr.py's module docstring) for too long.
_PERIOD_WINDOW = 5

#: Below this many observed gaps there is not enough evidence to trust a
#: period estimate -- a fast source's first two packets could coincidentally
#: be far apart (radio warm-up, first scan window) and adapting off that
#: alone would needlessly widen the very staleness check it exists to keep
#: honest. Fewer samples means "trust the configured default instead."
_MIN_GAP_SAMPLES = 2

#: How far past the OBSERVED period the effective staleness threshold
#: widens to -- comfortably above 1x so a single slightly-late packet does
#: not itself trip staleness, without being so generous that a source that
#: really has gone quiet stays "fine" for multiple missed cycles.
_STALE_PERIOD_MULTIPLIER = 2.5


@dataclass
class HuntConfig:
    """Everything the decision task (SPEC 6.1) can tune, plus the
    thresholds it does not: ``stale_after_s``, ``vanished_after_s`` and
    ``redact`` are safety / privacy defaults, never something a hunt should
    be won or lost by picking the "wrong" value for.

    Deliberately NOT frozen: ``weblive.handle_control`` (SPEC 5.9) mutates a
    running session's config live, from the browser's calibration panel --
    freezing it would force every control tick to rebuild a whole new
    ``HuntEngine`` just to change one number.
    """

    live_window_s: float = LIVE_WINDOW_S
    trend_window_s: float = TREND_WINDOW_S
    trend_threshold_db: int = TREND_THRESHOLD_DB
    stale_after_s: float = 5.0

    vanished_after_s: float = 45.0
    """How long the target must stay silent before the meter stops
    implying a weak-but-present signal and instead says the address itself
    may be gone (``HuntSnapshot.vanished``). Sized to sit comfortably above
    an ordinary advertising gap or a few missed BR/EDR inquiry cycles (both
    already covered by ``stale_after_s``/the adaptive threshold below) and
    comfortably below even the fast end of a BLE address-rotation period
    (8 minutes = 480s, per
    https://www.bluetooth.com/blog/enhancing-device-privacy-and-energy-efficiency-with-bluetooth-randomized-rpa-updates/,
    checked 2026-08-06) -- so "vanished" fires well before a rotation
    plausibly could have happened, never after."""

    redact: bool = True


@dataclass(frozen=True)
class HuntSnapshot:
    rssi_dbm: float | None
    raw_dbm: int | None
    band_key: str
    band_es: str
    band_en: str
    fraction: float
    trend: str
    stale: bool
    age_s: float | None
    peak_1min_dbm: int | None
    n_total: int
    n_last_min: int
    source: str
    spark: list[int]
    elapsed_s: float
    units: str = UNIT_DBM
    """What the newest reading's ``rssi_dbm``-shaped fields (``rssi_dbm``,
    ``raw_dbm``, ``peak_1min_dbm``) actually are -- ``"dbm"`` or
    ``"golden_range_db"`` (``reading.py``). Defaults to ``"dbm"`` so a
    snapshot taken before any reading has arrived reads the same as every
    pre-link-mode snapshot always did. Every field derived from a
    dBm-calibrated table (``band_key``/``band_es``/``band_en``,
    ``fraction``) is computed accordingly -- see :meth:`HuntEngine.snapshot`."""

    vanished: bool = False
    """True once the target has been silent for ``cfg.vanished_after_s`` --
    a state distinct from (and always a STRICTER condition than) ``stale``.
    ``stale`` means "no fresh number to show right now"; ``vanished`` means
    "this has gone on long enough that the address itself may be dead", the
    distinction the bug report that motivated this field was about: a
    student reading a frozen "stale" number assumed the meter was broken,
    when the real cause was a rotated BLE private address that will never
    advertise again under this address. Defaults to False so a snapshot
    built the old way (no ``vanished=`` kwarg, exactly what every
    pre-existing test in this file already does) behaves exactly as it
    always did."""


class HuntEngine:
    """Feed it Readings; ask it for a snapshot at any ``now``.

    ``target``, when non-empty, is an exact address filter: a reading whose
    ``addr`` is set and does not equal ``target`` is dropped before it ever
    reaches the log. Two consequences worth being explicit about, since
    neither is a bug:

    * a reading with ``addr == ""`` always passes -- ``sim.py`` readings and
      a not-yet-address-aware source both look like this, and a target
      filter has no business rejecting a source that never claimed an
      address in the first place;
    * an RPA rotation (SPEC 5.7/5.4) changes ``addr`` outright, so a strict
      ``target`` set to a pre-rotation address stops matching afterwards.
      This is not a shortcut this engine takes -- it is the same real
      limitation a live BlueZ/bleak scan has with no IRK to resolve a
      rotated address back to the same device, which is exactly why
      challenge-06b (SPEC 6.1) uses one mid-hunt rotation to make the
      default settings fail.
    """

    def __init__(self, cfg: HuntConfig, target: str = "", maxlen: int = 4096) -> None:
        self.cfg = cfg
        self.target = target
        self._maxlen = maxlen
        self._log = ReadingLog(maxlen=maxlen)
        self._t_start: float | None = None
        self._n_total = 0
        self._last_value: int | None = None
        #: Recent inter-arrival gaps between readings that pass the target
        #: filter -- feeds :meth:`_effective_stale_after_s`'s adaptive
        #: threshold. See that method's docstring for why this exists.
        self._gap_log: deque[float] = deque(maxlen=_PERIOD_WINDOW)
        self._last_feed_t: float | None = None

    def feed(self, r: Reading) -> None:
        if self.target and r.addr and r.addr != self.target:
            return
        if self._last_feed_t is not None:
            gap = r.t - self._last_feed_t
            if gap > 0:  # guard against duplicate/out-of-order timestamps
                self._gap_log.append(gap)
        self._last_feed_t = r.t
        if self._t_start is None:
            self._t_start = r.t
        self._log.append(r)
        # Honest count, tracked incrementally rather than recomputed from the
        # (bounded) log at snapshot time: `signal.distinct_measurements`
        # counts runs of identical values over whatever sequence it is
        # handed, and the log's ring buffer can evict readings a long hunt
        # has long since scrolled past. Feeding one reading at a time here
        # reproduces exactly the same "did the value change" rule, but keeps
        # counting for the whole session instead of only the last `maxlen`
        # readings.
        if self._last_value is None or r.rssi_dbm != self._last_value:
            self._n_total += 1
        self._last_value = r.rssi_dbm

    def _effective_stale_after_s(self) -> float:
        """``cfg.stale_after_s``, widened if the target's OWN observed
        inter-arrival period needs more room than the flat configured
        default gives it.

        Found 2026-08-06: bredr mode's ~10.24s BR/EDR inquiry cycle
        (bredr.py's module docstring) against the 5.0s fast-source default
        marked the display stale between almost every single reading --
        honest in the narrow sense of never lying, but flickering "SIN
        SEÑAL" on a source that is working exactly as designed is not a
        working meter either (this project's own no-placeholder rule).
        Widening from the SOURCE'S OWN observed cadence, rather than
        hardcoding a per-mode number here, means any slow or irregular
        source self-corrects -- not just the one this lab happens to name;
        ``run.py``'s own ``MODE_CONFIG_DEFAULTS`` for ``--mode bredr``
        (25.0s) is an independent, CLI-level instance of the same
        reasoning, sized off the same evidence, and the two happen to land
        in the same neighbourhood (10.24s * 2.5 ~= 25.6s).

        Never NARROWS ``cfg.stale_after_s`` -- it is always a floor, never
        a target: a fast source with a tiny observed gap must still use the
        student's own configured (or default 5.0s) staleness timeout, not
        something smaller derived from how fast BLE happens to advertise.
        """
        if len(self._gap_log) < _MIN_GAP_SAMPLES:
            return self.cfg.stale_after_s
        observed_period = statistics.median(self._gap_log)
        return max(self.cfg.stale_after_s, observed_period * _STALE_PERIOD_MULTIPLIER)

    def snapshot(self, now: float) -> HuntSnapshot:
        """Pure in ``now``: no clock read anywhere in this method.

        Calling this twice with the same ``now`` after the same sequence of
        :meth:`feed` calls must return an identical snapshot -- that is the
        whole contract SPEC 5.5 asks for, and it is what lets
        ``weblive.py``'s SSE loop be driven by a test harness instead of a
        real timer.
        """
        cfg = self.cfg
        live = self._log.since(cfg.live_window_s, now)
        # trend_of needs both its "recent" and "before" windows in one call;
        # handing it exactly their combined span means it never sees a
        # reading it has no business considering, and never has to guess
        # where the log's other readings went.
        trend_history = self._log.since(cfg.live_window_s + cfg.trend_window_s, now)
        effective_stale_after_s = self._effective_stale_after_s()
        stale_check = self._log.since(effective_stale_after_s, now)
        spark_readings = self._log.since(SPARK_WINDOW_S, now)
        peak_readings = self._log.since(ONE_MINUTE_S, now)
        last_min_readings = self._log.since(ONE_MINUTE_S, now)

        rssi_dbm = median_rssi(live)
        last = self._log.last
        raw_dbm = last.rssi_dbm if last is not None else None
        age_s = (now - last.t) if last is not None else None
        # vanished is a STRICTER, separate threshold from stale -- see
        # HuntSnapshot.vanished's own docstring for the distinction. No
        # reading ever fed means nothing has vanished FROM anything yet, so
        # this is False rather than True the way an empty-log `stale` is.
        vanished = age_s is not None and age_s >= cfg.vanished_after_s

        # Which units the CURRENT reading regime is in. No reading yet ->
        # "dbm" by construction (HuntSnapshot.units default), so an empty
        # engine behaves exactly as it did before this field existed.
        units = last.units if last is not None else UNIT_DBM

        # Before the first reading ever lands, there is nothing to report a
        # band for -- band_for's last floor is -inf (signal.py), so handing
        # it -inf here lands on the same "very_far" rung a genuinely weak
        # signal would, rather than inventing a sixth "unknown" band that
        # would need its own bilingual strings and its own place on the
        # meter.
        reference = (
            rssi_dbm if rssi_dbm is not None
            else float(raw_dbm) if raw_dbm is not None
            else float("-inf")
        )

        # PROXIMITY_BANDS and the dBm fraction span are calibrated to real
        # dBm (README §2.1) -- running them against a link-mode Golden Range
        # delta would silently relabel an honest "how far outside the golden
        # range" number as a distance/band claim it never was. See
        # signal.NOT_APPLICABLE_BAND and signal.GOLDEN_RANGE_LO/HI.
        if units == UNIT_DBM:
            band = band_for(reference)
            frac = fraction(reference)
            spark = sparkline_levels(spark_readings)
        else:
            band = NOT_APPLICABLE_BAND
            frac = fraction(reference, lo=GOLDEN_RANGE_LO, hi=GOLDEN_RANGE_HI)
            spark = sparkline_levels(spark_readings, lo=GOLDEN_RANGE_LO,
                                     hi=GOLDEN_RANGE_HI)

        trend = trend_of(
            trend_history, now,
            live_window=cfg.live_window_s,
            trend_window=cfg.trend_window_s,
            threshold_db=cfg.trend_threshold_db,
        )

        return HuntSnapshot(
            rssi_dbm=rssi_dbm,
            raw_dbm=raw_dbm,
            band_key=band.key,
            band_es=band.es,
            band_en=band.en,
            fraction=frac,
            trend=trend.value,
            stale=is_stale(stale_check, now, effective_stale_after_s),
            age_s=age_s,
            peak_1min_dbm=peak_rssi(peak_readings),
            n_total=self._n_total,
            n_last_min=distinct_measurements(last_min_readings),
            source=last.source if last is not None else "",
            spark=spark,
            elapsed_s=(now - self._t_start) if self._t_start is not None else 0.0,
            units=units,
            vanished=vanished,
        )

    def reset(self) -> None:
        """Back to a freshly constructed engine -- same cfg/target/maxlen."""
        self._log = ReadingLog(maxlen=self._maxlen)
        self._t_start = None
        self._n_total = 0
        self._last_value = None
        self._gap_log.clear()
        self._last_feed_t = None
