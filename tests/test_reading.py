"""blelib.reading -- Reading and ReadingLog.

Pure-data tests: no clock, no radio. Every timestamp here is a plain float
chosen by the test, exactly the point of keeping this module clockless.
"""

from __future__ import annotations

from blelib.reading import UNIT_DBM, UNIT_GOLDEN_RANGE_DB, Reading, ReadingLog


def mk(rssi: int, t: float, source: str = "sim", addr: str = "") -> Reading:
    return Reading(rssi_dbm=rssi, t=t, source=source, addr=addr)


# --------------------------------------------------------------- Reading
def test_reading_is_frozen_and_carries_all_fields():
    r = Reading(rssi_dbm=-60, t=1.5, source="advert", addr="XX:XX:B1:42")
    assert r.rssi_dbm == -60
    assert r.t == 1.5
    assert r.source == "advert"
    assert r.addr == "XX:XX:B1:42"
    try:
        r.rssi_dbm = -50  # type: ignore[misc]
        assert False, "Reading must be immutable"
    except AttributeError:
        pass


def test_reading_addr_defaults_to_empty_string():
    r = Reading(rssi_dbm=-70, t=0.0, source="sim")
    assert r.addr == ""


def test_reading_units_defaults_to_dbm():
    """Every source before link mode ever produced dBm -- a Reading built
    the old way (no units= kwarg) must keep meaning exactly what it always
    meant, unchanged."""
    r = Reading(rssi_dbm=-70, t=0.0, source="sim")
    assert r.units == UNIT_DBM


def test_reading_units_carries_golden_range_db_for_link_mode():
    r = Reading(rssi_dbm=-3, t=0.0, source="link", units=UNIT_GOLDEN_RANGE_DB)
    assert r.units == UNIT_GOLDEN_RANGE_DB
    assert r.rssi_dbm == -3


# ------------------------------------------------------------- ReadingLog
def test_append_and_len_and_last():
    log = ReadingLog()
    assert len(log) == 0
    assert log.last is None
    log.append(mk(-60, 1.0))
    log.append(mk(-58, 2.0))
    assert len(log) == 2
    assert log.last == mk(-58, 2.0)


def test_maxlen_evicts_oldest_first():
    log = ReadingLog(maxlen=3)
    for i in range(5):
        log.append(mk(-60 + i, float(i)))
    assert len(log) == 3
    # Only the three newest survive, oldest-first.
    kept = [r.rssi_dbm for r in log.window(0.0, 100.0)]
    assert kept == [-58, -57, -56]


def test_since_returns_only_readings_within_the_window_oldest_first():
    log = ReadingLog()
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        log.append(mk(-60, t))
    out = log.since(seconds=2.0, now=4.0)
    # cutoff = now - seconds = 2.0, so t in {2.0, 3.0, 4.0}
    assert [r.t for r in out] == [2.0, 3.0, 4.0]


def test_since_on_empty_log_returns_empty_list():
    log = ReadingLog()
    assert log.since(seconds=5.0, now=10.0) == []


def test_since_walks_back_from_the_end_not_a_full_scan():
    """A cutoff near "now" must not force a walk through old history.

    We cannot inspect the internal loop directly without over-fitting the
    test to the implementation, so this asserts the *externally visible*
    contract that makes the backward walk correct in the first place: the
    buffer is time-ordered, and readings older than the cutoff never leak
    into the result even when there are thousands of them ahead of the
    (small) recent window.
    """
    log = ReadingLog(maxlen=10_000)
    for t in range(5000):
        log.append(mk(-60, float(t)))
    out = log.since(seconds=1.5, now=4999.0)
    assert [r.t for r in out] == [4998.0, 4999.0]


def test_window_is_inclusive_on_both_ends():
    log = ReadingLog()
    for t in (0.0, 1.0, 2.0, 3.0):
        log.append(mk(-60, t))
    out = log.window(1.0, 2.0)
    assert [r.t for r in out] == [1.0, 2.0]


def test_window_on_empty_log_returns_empty_list():
    log = ReadingLog()
    assert log.window(0.0, 10.0) == []
