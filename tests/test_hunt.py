"""Tests for blelib.hunt -- purity of snapshot(now), and that HuntEngine wires
readings through to exactly the signal.py functions SPEC 5.5 names.

Where a value depends on a threshold signal.py itself owns (band floors,
trend threshold), the test computes the expected value by calling the very
same signal.py function the engine calls, rather than hardcoding a number --
this file has no business asserting an exact dBm threshold that is another
module's decision.
"""

from __future__ import annotations

import pytest

from blelib import hunt, signal
from blelib.reading import UNIT_DBM, UNIT_GOLDEN_RANGE_DB, Reading


def _r(rssi_dbm: int, t: float, source: str = "sim", addr: str = "",
       units: str = UNIT_DBM) -> Reading:
    return Reading(rssi_dbm=rssi_dbm, t=t, source=source, addr=addr, units=units)


def _engine(**cfg_kwargs) -> hunt.HuntEngine:
    return hunt.HuntEngine(hunt.HuntConfig(**cfg_kwargs))


# ------------------------------------------------------------------- purity
def test_snapshot_is_pure_given_now_and_the_fed_readings():
    """Calling snapshot() twice with the same `now`, after the same feed
    sequence, must be identical -- no clock read, no hidden state advances
    on its own between calls."""
    eng = _engine()
    for i in range(10):
        eng.feed(_r(-50 - i, t=float(i)))
    s1 = eng.snapshot(now=20.0)
    s2 = eng.snapshot(now=20.0)
    assert s1 == s2


def test_snapshot_depends_only_on_the_now_argument_not_wall_clock():
    """The engine must not read time.monotonic() / time.time() itself:
    calling snapshot() far away from real "now" in wall-clock terms must
    still produce a coherent, self-consistent result driven purely by the
    `now` passed in."""
    eng = _engine()
    eng.feed(_r(-55, t=1000.0))
    # `now` chosen nowhere near any real wall-clock value.
    snap = eng.snapshot(now=1_000_000.0)
    assert snap.age_s == pytest.approx(1_000_000.0 - 1000.0)
    assert snap.stale is True   # obviously long past stale_after_s


def test_empty_engine_snapshot_has_no_reading_yet():
    eng = _engine()
    snap = eng.snapshot(now=0.0)
    assert snap.rssi_dbm is None
    assert snap.raw_dbm is None
    assert snap.age_s is None
    assert snap.stale is True
    assert snap.n_total == 0
    assert snap.n_last_min == 0
    assert snap.source == ""
    assert snap.spark == []
    assert snap.elapsed_s == 0.0
    # band_for(-inf) still returns a real, bilingual band -- the last rung,
    # not a made-up sixth "unknown" state (see hunt.py's design note).
    assert snap.band_key == signal.PROXIMITY_BANDS[-1].key
    assert snap.fraction == 0.0


# --------------------------------------------------------------- live value
def test_rssi_dbm_is_the_median_of_the_live_window():
    eng = _engine(live_window_s=4.0)
    now = 10.0
    values = [-70, -60, -65]
    for i, v in enumerate(values):
        eng.feed(_r(v, t=now - 1.0 - i))   # all inside the 4s live window
    snap = eng.snapshot(now=now)
    live = [_r(v, now - 1.0 - i) for i, v in enumerate(values)]
    expected = signal.median_rssi(live)
    assert snap.rssi_dbm == pytest.approx(expected)


def test_raw_dbm_is_always_the_newest_single_reading_even_outside_the_window():
    eng = _engine(live_window_s=1.0)
    eng.feed(_r(-80, t=0.0))     # will fall outside the live window at now=10
    eng.feed(_r(-40, t=9.9))
    snap = eng.snapshot(now=10.0)
    assert snap.raw_dbm == -40


# ------------------------------------------------------------------- bands
def test_band_matches_signal_band_for_applied_to_the_live_median():
    eng = _engine(live_window_s=4.0)
    now = 5.0
    readings = [_r(-50, now - 0.5), _r(-52, now - 1.0), _r(-48, now - 1.5)]
    for r in readings:
        eng.feed(r)
    snap = eng.snapshot(now=now)
    expected_band = signal.band_for(signal.median_rssi(readings))
    assert snap.band_key == expected_band.key
    assert snap.band_es == expected_band.es
    assert snap.band_en == expected_band.en
    assert snap.fraction == pytest.approx(signal.fraction(signal.median_rssi(readings)))


def test_bilingual_band_strings_are_both_present_and_distinct():
    eng = _engine()
    eng.feed(_r(-40, t=0.0))
    snap = eng.snapshot(now=0.5)
    assert snap.band_es and snap.band_en
    assert snap.band_es != snap.band_en


# -------------------------------------------------------------------- trend
def test_trend_matches_signal_trend_of_over_the_same_combined_window():
    cfg = hunt.HuntConfig(live_window_s=4.0, trend_window_s=12.0, trend_threshold_db=3)
    eng = hunt.HuntEngine(cfg)
    now = 20.0
    # prior window is [now-12, now-4) = [8, 16); recent window is [16, 20].
    # prior: weak signal. recent: strong signal -- a closing-in walk.
    readings = [
        _r(-80, now - 10.0), _r(-79, now - 8.0),     # prior
        _r(-50, now - 2.0), _r(-49, now - 1.0),      # recent
    ]
    for r in readings:
        eng.feed(r)
    snap = eng.snapshot(now=now)
    expected = signal.trend_of(readings, now, live_window=cfg.live_window_s,
                               trend_window=cfg.trend_window_s,
                               threshold_db=cfg.trend_threshold_db)
    assert snap.trend == expected.value
    assert snap.trend == "warmer"   # sanity: this track is closing in


def test_trend_is_unknown_with_fewer_than_two_samples_per_window():
    eng = _engine()
    eng.feed(_r(-60, t=0.0))
    snap = eng.snapshot(now=1.0)
    assert snap.trend == signal.Trend.UNKNOWN.value


# ------------------------------------------------------------------- stale
def test_stale_flips_true_once_the_timeout_elapses():
    eng = _engine(stale_after_s=5.0)
    eng.feed(_r(-60, t=0.0))
    assert eng.snapshot(now=4.9).stale is False
    assert eng.snapshot(now=5.1).stale is True


# ----------------------------------------------------------- honest counts
def test_n_total_counts_only_value_changes_not_every_feed():
    eng = _engine()
    for t, v in enumerate([-60, -60, -60, -61, -61, -62]):
        eng.feed(_r(v, t=float(t)))
    snap = eng.snapshot(now=10.0)
    assert snap.n_total == 3   # runs: [-60,-60,-60] [-61,-61] [-62]


def test_n_total_matches_signal_distinct_measurements_while_under_maxlen():
    """Cross-check: HuntEngine's incremental counter and a direct call to
    signal.distinct_measurements over everything fed must agree, as long as
    the ring buffer has not yet evicted anything."""
    eng = hunt.HuntEngine(hunt.HuntConfig(), maxlen=4096)
    values = [-70, -70, -69, -68, -68, -68, -50, -50, -49]
    fed = [_r(v, t=float(i)) for i, v in enumerate(values)]
    for r in fed:
        eng.feed(r)
    snap = eng.snapshot(now=100.0)
    assert snap.n_total == signal.distinct_measurements(fed)


def test_n_total_keeps_counting_past_the_ring_buffer_capacity():
    """The whole reason n_total is tracked incrementally (hunt.py's design
    note) rather than recomputed from the bounded log: it must not reset or
    undercount once the ring buffer starts evicting old readings."""
    eng = hunt.HuntEngine(hunt.HuntConfig(), maxlen=8)
    for t in range(50):
        eng.feed(_r(-60 - t, t=float(t)))   # every value distinct
    snap = eng.snapshot(now=100.0)
    assert len(eng._log) == 8         # buffer really did evict
    assert snap.n_total == 50         # but the honest count did not shrink


def test_peak_1min_dbm_is_the_strongest_reading_in_the_last_minute():
    eng = _engine()
    now = 100.0
    eng.feed(_r(-70, t=now - 200.0))   # outside the 60s window
    eng.feed(_r(-65, t=now - 30.0))
    eng.feed(_r(-40, t=now - 10.0))    # strongest, inside the window
    eng.feed(_r(-55, t=now - 5.0))
    snap = eng.snapshot(now=now)
    assert snap.peak_1min_dbm == -40


# ------------------------------------------------------------------ source
def test_source_reflects_the_newest_reading():
    eng = _engine()
    eng.feed(_r(-60, t=0.0, source="sim"))
    eng.feed(_r(-58, t=1.0, source="advert"))
    snap = eng.snapshot(now=1.5)
    assert snap.source == "advert"


# -------------------------------------------------------------- sparkline
def test_spark_is_a_valid_sparkline_of_recent_history():
    eng = _engine()
    for i in range(5):
        eng.feed(_r(-90 + 10 * i, t=float(i)))
    snap = eng.snapshot(now=10.0)
    assert len(snap.spark) == 5
    assert all(0 <= level <= 7 for level in snap.spark)
    # a monotonically improving signal should give a non-decreasing spark
    assert snap.spark == sorted(snap.spark)


# --------------------------------------------------------------------- addr
def test_target_filter_drops_readings_from_other_addresses():
    eng = hunt.HuntEngine(hunt.HuntConfig(), target="AA:BB:CC:DD:EE:01")
    eng.feed(_r(-40, t=0.0, addr="AA:BB:CC:DD:EE:01"))
    eng.feed(_r(-90, t=1.0, addr="11:22:33:44:55:66"))   # someone else's phone
    snap = eng.snapshot(now=1.5)
    assert snap.raw_dbm == -40
    assert snap.n_total == 1


def test_target_filter_accepts_readings_with_no_address():
    """sim.py-style readings with addr="" must not be rejected by a target
    filter -- a source that never claimed an address cannot fail to match
    one."""
    eng = hunt.HuntEngine(hunt.HuntConfig(), target="AA:BB:CC:DD:EE:01")
    eng.feed(_r(-55, t=0.0, addr=""))
    snap = eng.snapshot(now=0.5)
    assert snap.raw_dbm == -55


def test_no_target_means_no_address_filtering():
    eng = hunt.HuntEngine(hunt.HuntConfig())   # target="" default
    eng.feed(_r(-40, t=0.0, addr="AA:AA:AA:AA:AA:AA"))
    eng.feed(_r(-90, t=1.0, addr="BB:BB:BB:BB:BB:BB"))
    snap = eng.snapshot(now=1.5)
    assert snap.n_total == 2


# -------------------------------------------------------------------- reset
def test_reset_clears_everything():
    eng = _engine()
    eng.feed(_r(-60, t=0.0))
    eng.feed(_r(-61, t=1.0))
    eng.reset()
    snap = eng.snapshot(now=5.0)
    assert snap.n_total == 0
    assert snap.raw_dbm is None
    assert snap.elapsed_s == 0.0
    assert len(eng._log) == 0


# ---------------------------------------------------------------- elapsed_s
def test_elapsed_s_measures_from_the_first_fed_reading():
    eng = _engine()
    eng.feed(_r(-60, t=100.0))
    eng.feed(_r(-61, t=101.0))
    snap = eng.snapshot(now=130.0)
    assert snap.elapsed_s == pytest.approx(30.0)


# -------------------------------------------------------- config signature
def test_hunt_config_matches_the_agreed_defaults():
    cfg = hunt.HuntConfig()
    assert cfg.live_window_s == signal.LIVE_WINDOW_S
    assert cfg.trend_window_s == signal.TREND_WINDOW_S
    assert cfg.trend_threshold_db == signal.TREND_THRESHOLD_DB
    assert cfg.stale_after_s == 5.0
    assert cfg.redact is True


def test_hunt_snapshot_is_frozen():
    eng = _engine()
    eng.feed(_r(-60, t=0.0))
    snap = eng.snapshot(now=1.0)
    with pytest.raises(Exception):
        snap.rssi_dbm = -70  # type: ignore[misc]


# ------------------------------------------------------- units-aware branch
def test_empty_engine_snapshot_units_defaults_to_dbm():
    eng = _engine()
    snap = eng.snapshot(now=0.0)
    assert snap.units == UNIT_DBM


def test_units_follows_the_newest_reading():
    eng = _engine()
    eng.feed(_r(-60, t=0.0, units=UNIT_DBM))
    eng.feed(_r(-3, t=1.0, units=UNIT_GOLDEN_RANGE_DB))
    snap = eng.snapshot(now=1.5)
    assert snap.units == UNIT_GOLDEN_RANGE_DB


def test_link_units_suppress_the_proximity_band():
    """A golden-range reading must NEVER get a distance-band label
    (arms_reach/same_table/...) -- see signal.NOT_APPLICABLE_BAND."""
    eng = _engine(live_window_s=4.0)
    eng.feed(_r(6, t=0.0, units=UNIT_GOLDEN_RANGE_DB))  # would read "arms_reach"
                                                          # if wrongly treated as dBm
    snap = eng.snapshot(now=0.5)
    assert snap.band_key == signal.NOT_APPLICABLE_BAND.key
    assert snap.band_key != "arms_reach"
    assert snap.band_es == signal.NOT_APPLICABLE_BAND.es
    assert snap.band_en == signal.NOT_APPLICABLE_BAND.en


def test_link_units_use_the_golden_range_fraction_span_not_the_dbm_one():
    eng = _engine(live_window_s=4.0)
    eng.feed(_r(0, t=0.0, units=UNIT_GOLDEN_RANGE_DB))  # 0 = inside the golden range
    snap = eng.snapshot(now=0.5)
    expected = signal.fraction(0.0, lo=signal.GOLDEN_RANGE_LO, hi=signal.GOLDEN_RANGE_HI)
    assert snap.fraction == pytest.approx(expected)
    # Sanity: NOT the dBm fraction, which would clamp 0 to the top (1.0) --
    # a 0 dBm reading is essentially unreachable, but a 0 golden-range delta
    # (this lab's evidence file has plenty) is exactly the ambiguous middle.
    dbm_fraction = signal.fraction(0.0)
    assert dbm_fraction == 1.0
    assert snap.fraction != dbm_fraction


def test_link_units_use_the_golden_range_span_for_the_sparkline_too():
    eng = _engine(live_window_s=4.0)
    for i, v in enumerate([-8, -3, 0, 3, 6]):
        eng.feed(_r(v, t=float(i), units=UNIT_GOLDEN_RANGE_DB))
    snap = eng.snapshot(now=10.0)
    expected = signal.sparkline_levels(
        [Reading(rssi_dbm=v, t=float(i), source="sim", units=UNIT_GOLDEN_RANGE_DB)
         for i, v in enumerate([-8, -3, 0, 3, 6])],
        lo=signal.GOLDEN_RANGE_LO, hi=signal.GOLDEN_RANGE_HI)
    assert snap.spark == expected


def test_dbm_units_still_use_the_ordinary_band_and_fraction():
    """Regression guard: the units-aware branch must not change ANY
    existing dBm behaviour -- every pre-link-mode test in this file already
    asserts this, this is just an explicit belt-and-suspenders check."""
    eng = _engine(live_window_s=4.0)
    eng.feed(_r(-50, t=0.0))  # units=UNIT_DBM by _r's own default
    snap = eng.snapshot(now=0.5)
    assert snap.units == UNIT_DBM
    assert snap.band_key == signal.band_for(-50).key
    assert snap.fraction == pytest.approx(signal.fraction(-50))
