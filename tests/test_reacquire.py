"""Tests for the "re-acquire after a BLE private-address rotation" feature
added to fix the bug report "I have to reboot the app to see the change in
power of the bluetooth signal" -- the meter was not broken, the target's
rotating private address had simply changed and the meter had no way to
say so, let alone find the new one.

Two independent things are tested here, kept in one file because they were
delivered together and neither ``tests/test_scan.py`` nor
``tests/test_weblive.py`` is the obvious single home for both:

* :mod:`blelib.scan`'s PURE re-acquire scorer -- :func:`scan.score_reacquire`
  and :func:`scan.rank_reacquire_candidates`, exercised against hand-built
  :class:`scan.DeviceFingerprint` objects, no radio, no event loop (see
  SPEC's own testing-seam discipline, mirrored here).
* :mod:`blelib.hunt`'s two other additions from the same fix:
  ``HuntSnapshot.vanished`` (distinct from ``stale``) and the adaptive
  staleness threshold that stops a slow, regular source (bredr's ~10s
  cycle) from flickering "SIN SEÑAL" between every reading.
"""

from __future__ import annotations

import pytest

from blelib import hunt, scan
from blelib.reading import Reading

# ============================================================== scan.py ==


def _fp(addr: str = "AA:BB:CC:DD:EE:01", addr_kind: str = "rpa",
        rssi_dbm: int = -60, t: float = 0.0, local_name: str = "",
        manufacturer_ids: frozenset[int] = frozenset(),
        mfg_payload_len: int = 0, service_uuids: frozenset[str] = frozenset(),
        tx_power: int | None = None) -> scan.DeviceFingerprint:
    return scan.DeviceFingerprint(
        addr=addr, addr_kind=addr_kind, rssi_dbm=rssi_dbm, t=t,
        local_name=local_name, manufacturer_ids=manufacturer_ids,
        mfg_payload_len=mfg_payload_len, service_uuids=service_uuids,
        tx_power=tx_power)


# --------------------------------------------------------------- fingerprint
def test_device_fingerprint_equality_is_by_value():
    a = _fp(local_name="Phone")
    b = _fp(local_name="Phone")
    assert a == b


def test_reacquire_candidate_is_frozen():
    cand = scan.ReacquireCandidate(addr="X", score=0.5, reasons=["name"],
                                   rssi_dbm=-60, addr_kind="rpa")
    with pytest.raises(Exception):
        cand.score = 0.9  # type: ignore[misc]


# ----------------------------------------------------------- score_reacquire
def test_score_reacquire_matching_name_is_the_single_strongest_signal():
    last_seen = _fp(local_name="Luis's Phone")
    same_name = _fp(addr="new-addr", local_name="Luis's Phone")
    result = scan.score_reacquire(last_seen, same_name, gap_s=5.0)
    assert "name" in result.reasons
    assert result.score >= scan._W_LOCAL_NAME


def test_score_reacquire_no_signals_at_all_scores_zero():
    last_seen = _fp(rssi_dbm=-60, t=0.0)
    candidate = _fp(addr="new-addr", rssi_dbm=-60 - scan.RSSI_CONTINUITY_MAX_DELTA_DB - 1,
                    t=scan.TIME_PROXIMITY_MAX_GAP_S + 1)
    result = scan.score_reacquire(last_seen, candidate,
                                  gap_s=candidate.t - last_seen.t)
    assert result.score == 0.0
    assert result.reasons == []


def test_score_reacquire_manufacturer_id_overlap_is_credited():
    last_seen = _fp(manufacturer_ids=frozenset({0x004C, 0x0075}))
    candidate = _fp(addr="new-addr", manufacturer_ids=frozenset({0x004C}))
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "manufacturer_id" in result.reasons


def test_score_reacquire_disjoint_manufacturer_ids_are_not_credited():
    last_seen = _fp(manufacturer_ids=frozenset({0x004C}))
    candidate = _fp(addr="new-addr", manufacturer_ids=frozenset({0x0075}))
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "manufacturer_id" not in result.reasons


def test_score_reacquire_service_uuid_overlap_is_credited():
    last_seen = _fp(service_uuids=frozenset({"180d", "180f"}))
    candidate = _fp(addr="new-addr", service_uuids=frozenset({"180d"}))
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "service_uuid" in result.reasons


def test_score_reacquire_payload_shape_within_tolerance_is_credited():
    last_seen = _fp(mfg_payload_len=10)
    candidate = _fp(addr="new-addr",
                    mfg_payload_len=10 + scan.PAYLOAD_LEN_TOLERANCE_BYTES)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "payload_shape" in result.reasons


def test_score_reacquire_payload_shape_outside_tolerance_is_not_credited():
    last_seen = _fp(mfg_payload_len=10)
    candidate = _fp(addr="new-addr",
                    mfg_payload_len=10 + scan.PAYLOAD_LEN_TOLERANCE_BYTES + 1)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "payload_shape" not in result.reasons


def test_score_reacquire_payload_shape_both_zero_is_not_credited():
    """Two devices that advertise NO manufacturer data at all match on
    "zero equals zero" by construction, which is not evidence of anything
    -- most BLE devices in a room have no manufacturer data."""
    last_seen = _fp(mfg_payload_len=0)
    candidate = _fp(addr="new-addr", mfg_payload_len=0)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "payload_shape" not in result.reasons


def test_score_reacquire_tx_power_exact_match_is_credited():
    last_seen = _fp(tx_power=4)
    candidate = _fp(addr="new-addr", tx_power=4)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "tx_power" in result.reasons


def test_score_reacquire_tx_power_mismatch_is_not_credited():
    last_seen = _fp(tx_power=4)
    candidate = _fp(addr="new-addr", tx_power=0)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "tx_power" not in result.reasons


def test_score_reacquire_tx_power_missing_on_either_side_is_not_credited():
    last_seen = _fp(tx_power=None)
    candidate = _fp(addr="new-addr", tx_power=4)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "tx_power" not in result.reasons


def test_score_reacquire_rssi_continuity_decays_with_distance():
    last_seen = _fp(rssi_dbm=-60)
    close = scan.score_reacquire(last_seen, _fp(addr="a", rssi_dbm=-60), gap_s=0.0)
    near = scan.score_reacquire(last_seen, _fp(addr="b", rssi_dbm=-64), gap_s=0.0)
    far = scan.score_reacquire(
        last_seen, _fp(addr="c", rssi_dbm=-60 - scan.RSSI_CONTINUITY_MAX_DELTA_DB),
        gap_s=0.0)
    assert close.score > near.score > far.score
    assert "rssi_continuity" in close.reasons
    assert "rssi_continuity" not in far.reasons  # exactly at the cutoff: 0 credit


def test_score_reacquire_rssi_beyond_max_delta_gets_no_continuity_credit():
    last_seen = _fp(rssi_dbm=-60)
    candidate = _fp(addr="new-addr",
                    rssi_dbm=-60 - scan.RSSI_CONTINUITY_MAX_DELTA_DB - 5)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert "rssi_continuity" not in result.reasons


def test_score_reacquire_time_proximity_decays_with_gap():
    last_seen = _fp()
    soon = scan.score_reacquire(last_seen, _fp(addr="a"), gap_s=1.0)
    later = scan.score_reacquire(last_seen, _fp(addr="b"), gap_s=60.0)
    assert soon.score > later.score
    assert "time_proximity" in soon.reasons


def test_score_reacquire_time_proximity_uses_the_absolute_gap():
    """A candidate that started advertising slightly BEFORE the target's
    last packet (overlapping rotation) must score on magnitude, not be
    rejected for a negative gap."""
    last_seen = _fp()
    negative_gap = scan.score_reacquire(last_seen, _fp(addr="a"), gap_s=-2.0)
    positive_gap = scan.score_reacquire(last_seen, _fp(addr="b"), gap_s=2.0)
    assert negative_gap.score == pytest.approx(positive_gap.score)
    assert "time_proximity" in negative_gap.reasons


def test_score_reacquire_beyond_max_gap_gets_no_time_credit():
    last_seen = _fp()
    candidate = _fp(addr="new-addr")
    result = scan.score_reacquire(last_seen, candidate,
                                  gap_s=scan.TIME_PROXIMITY_MAX_GAP_S + 1)
    assert "time_proximity" not in result.reasons


def test_score_reacquire_score_never_exceeds_one():
    last_seen = _fp(local_name="Phone", manufacturer_ids=frozenset({1}),
                    service_uuids=frozenset({"a"}), mfg_payload_len=5,
                    tx_power=4, rssi_dbm=-60, t=0.0)
    candidate = _fp(addr="new-addr", local_name="Phone",
                    manufacturer_ids=frozenset({1}), service_uuids=frozenset({"a"}),
                    mfg_payload_len=5, tx_power=4, rssi_dbm=-60, t=0.0)
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert result.score <= 1.0
    # every signal fired -- a strong, "everything matches" case.
    assert set(result.reasons) == {"name", "manufacturer_id", "service_uuid",
                                    "payload_shape", "tx_power",
                                    "rssi_continuity", "time_proximity"}


def test_score_reacquire_carries_the_candidates_own_addr_and_rssi_and_kind():
    last_seen = _fp()
    candidate = _fp(addr="NEW:ADDR", rssi_dbm=-70, addr_kind="nrpa")
    result = scan.score_reacquire(last_seen, candidate, gap_s=0.0)
    assert result.addr == "NEW:ADDR"
    assert result.rssi_dbm == -70
    assert result.addr_kind == "nrpa"


# ------------------------------------------------------ rank_reacquire_candidates
def test_rank_reacquire_candidates_sorts_best_match_first():
    last_seen = _fp(local_name="Phone", rssi_dbm=-60, t=0.0,
                    manufacturer_ids=frozenset({0x004C}))
    # Both stay above MIN_REACQUIRE_SCORE, but "strong" matches on name
    # (the single heaviest-weighted signal) as well as RSSI/time proximity,
    # while "medium" only matches on manufacturer ID plus weaker RSSI/time
    # proximity credit -- strictly less evidence, so it must rank second.
    medium = _fp(addr="medium", manufacturer_ids=frozenset({0x004C}),
                 rssi_dbm=-64, t=5.0)
    strong = _fp(addr="strong", local_name="Phone", rssi_dbm=-61, t=1.0)
    ranked = scan.rank_reacquire_candidates(last_seen, [medium, strong])
    assert [c.addr for c in ranked] == ["strong", "medium"]


def test_rank_reacquire_candidates_excludes_the_target_itself():
    last_seen = _fp(addr="SAME", local_name="Phone")
    same_addr_again = _fp(addr="SAME", local_name="Phone", t=1.0)
    ranked = scan.rank_reacquire_candidates(last_seen, [same_addr_again])
    assert ranked == []


def test_rank_reacquire_candidates_drops_scores_below_the_floor():
    last_seen = _fp(rssi_dbm=-60, t=0.0)
    hopeless = _fp(addr="hopeless",
                   rssi_dbm=-60 - scan.RSSI_CONTINUITY_MAX_DELTA_DB - 10,
                   t=scan.TIME_PROXIMITY_MAX_GAP_S + 100)
    ranked = scan.rank_reacquire_candidates(last_seen, [hopeless])
    assert ranked == []


def test_rank_reacquire_candidates_keeps_a_candidate_at_or_above_the_floor():
    last_seen = _fp(local_name="Phone", rssi_dbm=-60, t=0.0)
    plausible = _fp(addr="plausible", local_name="Phone", rssi_dbm=-95, t=200.0)
    ranked = scan.rank_reacquire_candidates(last_seen, [plausible])
    assert len(ranked) == 1
    assert ranked[0].score >= scan.MIN_REACQUIRE_SCORE


def test_rank_reacquire_candidates_empty_input_returns_empty():
    assert scan.rank_reacquire_candidates(_fp(), []) == []


def test_rank_reacquire_candidates_gap_s_is_derived_from_t_difference():
    last_seen = _fp(t=100.0, local_name="Phone")
    candidate = _fp(addr="new-addr", t=101.0, local_name="Phone")  # 1s after
    direct = scan.score_reacquire(last_seen, candidate, gap_s=1.0)
    [ranked] = scan.rank_reacquire_candidates(last_seen, [candidate])
    assert ranked.score == pytest.approx(direct.score)


# =============================================================== hunt.py ==
# HuntEngine's other two additions from this same fix: the `vanished` state
# distinct from `stale`, and the adaptive staleness threshold. Placed here
# (not tests/test_hunt.py) because that file is out of scope for this fix.

def _r(rssi_dbm: int, t: float, source: str = "sim", addr: str = "") -> Reading:
    return Reading(rssi_dbm=rssi_dbm, t=t, source=source, addr=addr)


# --------------------------------------------------------------- vanished
def test_hunt_config_vanished_after_s_defaults_to_45s():
    assert hunt.HuntConfig().vanished_after_s == 45.0


def test_vanished_is_false_with_no_reading_ever():
    eng = hunt.HuntEngine(hunt.HuntConfig())
    snap = eng.snapshot(now=100000.0)
    assert snap.vanished is False   # nothing to vanish FROM yet


def test_vanished_flips_true_only_after_the_configured_timeout():
    eng = hunt.HuntEngine(hunt.HuntConfig(vanished_after_s=45.0))
    eng.feed(_r(-50, t=0.0))
    assert eng.snapshot(now=44.9).vanished is False
    assert eng.snapshot(now=45.1).vanished is True


def test_vanished_is_a_stricter_condition_than_stale():
    """A device quiet for a few seconds is STALE, not VANISHED -- the
    whole point of the distinct state (the bug report conflated the two:
    a student read a frozen stale number and assumed the app was broken)."""
    eng = hunt.HuntEngine(hunt.HuntConfig(stale_after_s=5.0, vanished_after_s=45.0))
    eng.feed(_r(-50, t=0.0))
    snap = eng.snapshot(now=10.0)   # well past stale, nowhere near vanished
    assert snap.stale is True
    assert snap.vanished is False


def test_vanished_defaults_to_false_on_a_hand_built_snapshot():
    """Regression guard: a HuntSnapshot built the old way (no `vanished=`
    kwarg) must behave exactly as it did before this field existed."""
    snap = hunt.HuntSnapshot(
        rssi_dbm=-60.0, raw_dbm=-60, band_key="same_room", band_es="x",
        band_en="y", fraction=0.5, trend="steady", stale=False, age_s=1.0,
        peak_1min_dbm=-55, n_total=5, n_last_min=5, source="advert",
        spark=[1, 2], elapsed_s=10.0)
    assert snap.vanished is False


# ----------------------------------------------------- adaptive staleness
def test_effective_stale_after_adapts_to_a_slow_observed_period():
    """A source with a genuinely slow, regular update cadence (like
    bredr's ~10.24s BR/EDR inquiry cycle) must not flicker stale between
    every single reading just because HuntConfig.stale_after_s defaults to
    5.0s -- the exact bug this fix addresses."""
    eng = hunt.HuntEngine(hunt.HuntConfig())   # stale_after_s stays 5.0
    eng.feed(_r(-40, t=0.0))
    eng.feed(_r(-41, t=10.0))
    eng.feed(_r(-42, t=20.0))   # two observed ~10s gaps -- enough to adapt
    # 6s after the last reading: the FLAT 5.0s default would already call
    # this stale; the observed ~10s cadence means it must not be.
    assert eng.snapshot(now=26.0).stale is False
    # Comfortably past even the adapted (~25s) threshold now.
    assert eng.snapshot(now=60.0).stale is True


def test_effective_stale_after_does_not_shrink_the_configured_default():
    """A fast source's tiny observed gaps must never NARROW the configured
    (or default) staleness timeout below what the student actually set."""
    eng = hunt.HuntEngine(hunt.HuntConfig(stale_after_s=5.0))
    for t in (0.0, 0.2, 0.4, 0.6):
        eng.feed(_r(-50, t=t))
    assert eng.snapshot(now=5.5).stale is False   # 4.9s since last reading
    assert eng.snapshot(now=5.7).stale is True    # 5.1s since last reading


def test_effective_stale_after_untouched_with_fewer_than_two_gaps():
    eng = hunt.HuntEngine(hunt.HuntConfig(stale_after_s=5.0))
    eng.feed(_r(-50, t=0.0))
    eng.feed(_r(-50, t=8.0))   # one big gap -- not enough samples to trust
    assert eng.snapshot(now=13.5).stale is True   # 5.5s since last reading


def test_reset_clears_gap_tracking_too():
    """If `_last_feed_t` leaked across reset, the first post-reset gap
    would be computed against ancient pre-reset history and could inflate
    the adaptive threshold enormously, silently hiding real staleness."""
    eng = hunt.HuntEngine(hunt.HuntConfig())
    eng.feed(_r(-50, t=0.0))
    eng.feed(_r(-50, t=10.0))
    eng.reset()
    eng.feed(_r(-50, t=1000.0))
    eng.feed(_r(-50, t=1000.2))
    eng.feed(_r(-50, t=1000.4))
    assert eng.snapshot(now=1005.5).stale is True   # 5.1s since last reading
