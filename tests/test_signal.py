"""blelib.signal -- smoothing, proximity bands, trend, and the honest counters.

The two properties SPEC calls out explicitly get their own tests, each with
an explanation of *why* the naive alternative fails:

* :func:`median_rssi` must be a true median so one reflected spike cannot
  yank the display (mean would fail this).
* :func:`distinct_measurements` must count value CHANGES only, so a cached
  repeated poll reports zero new information (a raw len() would fail this).
"""

from __future__ import annotations

import pytest

from blelib.reading import Reading
from blelib.signal import (
    LIVE_WINDOW_S,
    PROXIMITY_BANDS,
    TREND_THRESHOLD_DB,
    TREND_WINDOW_S,
    Trend,
    band_for,
    band_index,
    band_label,
    distinct_measurements,
    fraction,
    is_stale,
    median_rssi,
    peak_rssi,
    sparkline_levels,
    trend_of,
)


def mk(rssi: int, t: float, source: str = "sim") -> Reading:
    return Reading(rssi_dbm=rssi, t=t, source=source)


# ------------------------------------------------------------- median vs mean
def test_median_rssi_is_a_true_median_not_a_mean():
    # A track that is flat at -60 except one reflected spike at -30.
    readings = [mk(-60, 0.0), mk(-61, 1.0), mk(-30, 2.0), mk(-59, 3.0), mk(-60, 4.0)]
    med = median_rssi(readings)
    mean = sum(r.rssi_dbm for r in readings) / len(readings)
    # The mean is dragged up by the spike; the median must not be.
    assert mean > -58.0          # spike visibly pulls the mean
    assert med == -60.0          # median ignores it entirely
    assert med != pytest.approx(mean)


def test_median_rssi_even_count_averages_the_two_middle_values():
    readings = [mk(-70, 0.0), mk(-60, 1.0), mk(-50, 2.0), mk(-40, 3.0)]
    assert median_rssi(readings) == -55.0


def test_median_rssi_empty_returns_none():
    assert median_rssi([]) is None


def test_median_rssi_single_reading():
    assert median_rssi([mk(-72, 0.0)]) == -72.0


def test_peak_rssi_is_the_strongest_reading():
    readings = [mk(-70, 0.0), mk(-40, 1.0), mk(-90, 2.0)]
    assert peak_rssi(readings) == -40


def test_peak_rssi_empty_returns_none():
    assert peak_rssi([]) is None


# ------------------------------------------------------------ proximity bands
def test_proximity_bands_are_ordered_high_to_low_with_minus_inf_floor():
    floors = [b.floor_dbm for b in PROXIMITY_BANDS]
    assert floors == sorted(floors, reverse=True)
    assert floors[-1] == float("-inf")


def test_every_band_has_a_real_accented_spanish_label_and_english_label():
    expected_es = {
        "arms_reach": "A tu alcance",
        "same_table": "Misma mesa",
        "same_room": "Misma habitación",
        "far": "Lejos / detrás de un obstáculo",
        "very_far": "Muy lejos / blindado",
    }
    for band in PROXIMITY_BANDS:
        assert band.es == expected_es[band.key]
        assert band.en                      # non-empty
        assert band.es != band.en           # not silently reusing English


@pytest.mark.parametrize("rssi,key", [
    (-30, "arms_reach"),
    (-45, "arms_reach"),   # floor is inclusive
    (-46, "same_table"),
    (-60, "same_table"),
    (-61, "same_room"),
    (-72, "same_room"),
    (-73, "far"),
    (-85, "far"),
    (-86, "very_far"),
    (-1000, "very_far"),
])
def test_band_for_classifies_rssi_into_the_right_rung(rssi, key):
    assert band_for(rssi).key == key
    assert PROXIMITY_BANDS[band_index(rssi)].key == key


def test_band_label_follows_lang_argument():
    assert band_label(-40, lang="en") == "Arm's reach"
    assert band_label(-40, lang="es") == "A tu alcance"


def test_band_label_defaults_to_english():
    assert band_label(-40) == band_label(-40, lang="en")


# ---------------------------------------------------------------- fraction
def test_fraction_clamps_to_0_1_and_is_monotonic():
    assert fraction(-30.0) == 1.0          # at/above hi
    assert fraction(-20.0) == 1.0          # above hi, clamped
    assert fraction(-100.0) == 0.0         # at/below lo
    assert fraction(-110.0) == 0.0         # below lo, clamped
    mid = fraction(-65.0)
    assert 0.0 < mid < 1.0
    assert fraction(-70.0) < fraction(-40.0)


def test_fraction_custom_bounds():
    assert fraction(-50.0, lo=-100.0, hi=0.0) == 0.5


# ------------------------------------------------------------------- trend
def test_trend_is_unknown_with_fewer_than_two_samples_per_window():
    # Only one sample in the whole recent window.
    readings = [mk(-60, 9.0)]
    assert trend_of(readings, now=10.0) is Trend.UNKNOWN


def test_trend_is_unknown_when_prior_window_is_empty_even_with_full_recent_window():
    readings = [mk(-60, 8.0), mk(-60, 9.0), mk(-60, 10.0)]
    assert trend_of(readings, now=10.0) is Trend.UNKNOWN


def test_trend_warmer_when_recent_median_beats_prior_by_threshold():
    now = 20.0
    prior = [mk(-70, now - TREND_WINDOW_S + 1), mk(-70, now - LIVE_WINDOW_S - 1)]
    recent = [mk(-60, now - LIVE_WINDOW_S + 1), mk(-60, now)]
    assert trend_of(prior + recent, now=now) is Trend.WARMER


def test_trend_colder_when_recent_median_drops_by_threshold():
    now = 20.0
    prior = [mk(-60, now - TREND_WINDOW_S + 1), mk(-60, now - LIVE_WINDOW_S - 1)]
    recent = [mk(-70, now - LIVE_WINDOW_S + 1), mk(-70, now)]
    assert trend_of(prior + recent, now=now) is Trend.COLDER


def test_trend_steady_when_change_is_below_threshold():
    now = 20.0
    prior = [mk(-60, now - TREND_WINDOW_S + 1), mk(-60, now - LIVE_WINDOW_S - 1)]
    recent = [mk(-60 + (TREND_THRESHOLD_DB - 1), now - LIVE_WINDOW_S + 1),
              mk(-60 + (TREND_THRESHOLD_DB - 1), now)]
    assert trend_of(prior + recent, now=now) is Trend.STEADY


def test_trend_a_single_reflected_spike_in_the_recent_window_does_not_flip_it():
    # Recent window is genuinely steady at -60 except one spike to -30; the
    # median-based comparison must not read this as WARMER.
    now = 20.0
    prior = [mk(-60, now - TREND_WINDOW_S + 1), mk(-60, now - LIVE_WINDOW_S - 1)]
    recent = [mk(-60, now - LIVE_WINDOW_S + 0.5), mk(-30, now - 1.0), mk(-60, now)]
    assert trend_of(prior + recent, now=now) is Trend.STEADY


# --------------------------------------------------------------- is_stale
def test_is_stale_true_on_empty_readings():
    assert is_stale([], now=100.0, timeout_s=5.0) is True


def test_is_stale_false_within_timeout():
    readings = [mk(-60, 96.0)]
    assert is_stale(readings, now=100.0, timeout_s=5.0) is False


def test_is_stale_true_past_timeout():
    readings = [mk(-60, 90.0)]
    assert is_stale(readings, now=100.0, timeout_s=5.0) is True


def test_is_stale_boundary_is_not_stale_at_exactly_timeout():
    readings = [mk(-60, 95.0)]
    assert is_stale(readings, now=100.0, timeout_s=5.0) is False


# ----------------------------------------------------- distinct_measurements
def test_distinct_measurements_counts_runs_not_transitions():
    # A poller that re-reports the same cached number three times in a row
    # must not inflate the count: the run counts once, as one observation.
    readings = [mk(-60, 0.0), mk(-60, 0.3), mk(-60, 0.6),   # run 1 (cached x2)
                mk(-58, 0.9),                                 # run 2 starts
                mk(-58, 1.2), mk(-58, 1.5),                   # run 2 (cached x2)
                mk(-65, 1.8)]                                  # run 3 starts
    assert distinct_measurements(readings) == 3


@pytest.mark.parametrize("values,expected", [
    ([], 0),
    ([-60], 1),                    # one real observation
    ([-60, -60, -60], 1),          # one observation, cached twice
    ([-60, -61, -62], 3),
    ([-60, -60, -61, -61], 2),
])
def test_distinct_measurements_run_counting_cases(values, expected):
    readings = [mk(v, float(i)) for i, v in enumerate(values)]
    assert distinct_measurements(readings) == expected


def test_distinct_measurements_every_reading_different_counts_all_of_them():
    readings = [mk(-60 - i, float(i)) for i in range(6)]
    assert distinct_measurements(readings) == 6


# ------------------------------------------------------------ sparkline
def test_sparkline_levels_are_in_range_and_track_rssi_order():
    readings = [mk(-100, 0.0), mk(-65, 1.0), mk(-30, 2.0)]
    levels = sparkline_levels(readings, n_levels=8)
    assert levels[0] == 0
    assert levels[-1] == 7
    assert levels[0] <= levels[1] <= levels[2]
    assert all(0 <= lv <= 7 for lv in levels)


def test_sparkline_levels_empty_readings():
    assert sparkline_levels([]) == []


def test_sparkline_levels_respects_custom_bounds():
    readings = [mk(-50, 0.0)]
    levels = sparkline_levels(readings, n_levels=4, lo=-100.0, hi=0.0)
    # fraction = 0.5 -> level 0.5 * 3 = 1.5 -> rounds to 2
    assert levels == [2]
