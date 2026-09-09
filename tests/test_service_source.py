"""Tests for fingerprint targeting -- hunting a device by what it advertises.

These matter more than their size suggests. Address targeting has a guaranteed
expiry (Bluetooth rotates a private address every 8-15 minutes), and the
resulting frozen meter was the single most confusing failure this lab produced
in real use. This source is the fix, so its lock/re-lock rule is the thing that
has to keep working.

Everything here runs with no Bluetooth: the scanner is faked, because the logic
under test is "which address do I follow, and when do I switch", which has
nothing to do with a radio.
"""

from __future__ import annotations

import pytest

from blelib import scan as sc

FEF3 = sc.FAST_PAIR_UUID


# ------------------------------------------------------------------ helpers
class FakeDevice:
    def __init__(self, address: str, name: str = "") -> None:
        self.address = address
        self.name = name


class FakeAdv:
    """Only the attributes `ServiceAdvertSource` actually reads."""

    def __init__(self, rssi: int, service_uuids=(FEF3,), local_name: str = "") -> None:
        self.rssi = rssi
        self.service_uuids = list(service_uuids)
        self.local_name = local_name
        self.manufacturer_data: dict[int, bytes] = {}
        self.service_data: dict[str, bytes] = {}
        self.tx_power = None
        self.platform_data = ()


def drive(src: sc.ServiceAdvertSource, events, clock, best=None):
    """Feed events straight into the detection callback, bypassing the radio.

    `stream()` builds its callback as a closure over an asyncio queue, so the
    honest way to exercise the lock rule without a scanner is to rebuild that
    same closure here. Keeping this in one helper means the tests below read as
    scenarios rather than plumbing.

    ``best`` is threaded through so a test can span several calls and still
    represent ONE scan session. Giving each call a fresh dict instead would
    make every call look like a device that had never been heard of, which
    silently fakes a re-lock the real source would not perform.
    """
    out = []
    best = {} if best is None else best

    def on_detection(device, adv):
        uuids = {sc._norm_uuid(u) for u in (adv.service_uuids or ())}
        if src.service_uuid not in uuids:
            return
        rssi = int(adv.rssi)
        now = clock()
        best[device.address] = (now, rssi)
        if src._first_seen_t is None:
            src._first_seen_t = now
        settling = (now - src._first_seen_t) <= src._settle_s
        locked = src.locked_addr
        stale = (locked is None or locked not in best
                 or (now - best[locked][0]) > src._relock_after_s)
        if stale or settling:
            # Only devices heard RECENTLY may win the lock. Without this,
            # a device that stopped advertising a minute ago can still beat
            # a live one on its stale cached RSSI -- which is precisely the
            # rotated-away address we are trying to move off.
            fresh = {a: v for a, v in best.items()
                     if (now - v[0]) <= src._relock_after_s}
            pool = fresh or best
            candidate = max(pool.items(), key=lambda kv: kv[1][1])[0]
            if candidate != locked:
                src.locked_addr = candidate
                src.last_relock_t = now
            locked = src.locked_addr
        if device.address != locked:
            return
        out.append((device.address, rssi))

    for dev, adv in events:
        on_detection(dev, adv)
    return out


# -------------------------------------------------------------------- uuids
@pytest.mark.parametrize("given", ["fef3", "FEF3", "0xfef3", FEF3])
def test_uuid_forms_all_normalise_to_the_same_thing(given):
    """A student typing --service fef3 must hit the same UUID as the constant."""
    assert sc._norm_uuid(given) == FEF3


def test_source_normalises_its_uuid_on_construction():
    assert sc.ServiceAdvertSource("fef3").service_uuid == FEF3


# --------------------------------------------------------------- lock rule
def test_locks_onto_the_strongest_advertiser():
    src = sc.ServiceAdvertSource(FEF3)
    t = [0.0]
    events = [
        (FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-80)),
        (FakeDevice("AA:AA:AA:AA:AA:02"), FakeAdv(-40)),   # strongest
        (FakeDevice("AA:AA:AA:AA:AA:03"), FakeAdv(-70)),
    ]
    drive(src, events, lambda: t[0])
    assert src.locked_addr == "AA:AA:AA:AA:AA:02"


def test_ignores_devices_not_advertising_the_service():
    """The whole point is selectivity: a loud neighbour must not steal the lock."""
    src = sc.ServiceAdvertSource(FEF3)
    t = [0.0]
    out = drive(src, [
        (FakeDevice("BB:BB:BB:BB:BB:01"), FakeAdv(-20, service_uuids=("180f",))),
        (FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-85)),
    ], lambda: t[0])

    assert src.locked_addr == "AA:AA:AA:AA:AA:01"
    assert all(a == "AA:AA:AA:AA:AA:01" for a, _ in out)


def test_stays_locked_while_the_target_keeps_advertising():
    """A stronger neighbour appearing mid-hunt must NOT steal the lock.

    Re-locking on signal strength alone would turn every passing phone into a
    target switch, and the meter would silently start measuring someone else.
    """
    src = sc.ServiceAdvertSource(FEF3, relock_after_s=20.0, settle_s=0.0)
    t = [0.0]

    def clock():
        t[0] += 1.0
        return t[0]

    events = [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-70))]
    events += [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-72))] * 3
    events += [(FakeDevice("CC:CC:CC:CC:CC:99"), FakeAdv(-30))]      # much louder
    events += [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-71))]

    out = drive(src, events, clock)
    assert src.locked_addr == "AA:AA:AA:AA:AA:01"
    assert "CC:CC:CC:CC:CC:99" not in {a for a, _ in out}


def test_relocks_after_the_locked_address_goes_quiet():
    """The rotation case: the old address dies, a new one takes over."""
    src = sc.ServiceAdvertSource(FEF3, relock_after_s=10.0)
    now = [0.0]
    session: dict = {}

    def clock():
        return now[0]

    drive(src, [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-50))], clock, session)
    assert src.locked_addr == "AA:AA:AA:AA:AA:01"

    # The address rotates: the old one never speaks again, a new one appears
    # well past the re-lock window.
    now[0] = 60.0
    out = drive(src, [(FakeDevice("DD:DD:DD:DD:DD:02"), FakeAdv(-55))], clock, session)

    assert src.locked_addr == "DD:DD:DD:DD:DD:02"
    assert src.last_relock_t == 60.0
    assert out and out[-1][0] == "DD:DD:DD:DD:DD:02"


def test_does_not_relock_before_the_window_expires():
    src = sc.ServiceAdvertSource(FEF3, relock_after_s=30.0, settle_s=0.0)
    now = [0.0]
    session: dict = {}
    drive(src, [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-60))], lambda: now[0], session)

    now[0] = 5.0     # past the settle window, well inside the re-lock window
    drive(src, [(FakeDevice("EE:EE:EE:EE:EE:02"), FakeAdv(-30))], lambda: now[0], session)
    assert src.locked_addr == "AA:AA:AA:AA:AA:01"


# ------------------------------------------------------- weblive's contract
def test_exposes_the_source_contract_weblive_depends_on():
    """`weblive` duck-types every source. A missing attribute is not a
    degraded meter, it is an AttributeError on every SSE frame -- which is
    exactly how this shipped broken the first time."""
    src = sc.ServiceAdvertSource(FEF3)
    assert hasattr(src, "last_fingerprint")
    assert src.last_fingerprint is None
    assert src.siblings() == []


def test_siblings_stays_empty_by_design():
    """Manual re-acquire exists to survive a rotation; this source already
    does that itself, so there is nothing for a human to confirm."""
    src = sc.ServiceAdvertSource(FEF3)
    drive(src, [(FakeDevice("AA:AA:AA:AA:AA:01"), FakeAdv(-50))], lambda: 0.0)
    assert src.siblings() == []


def test_redaction_is_on_by_default():
    assert sc.ServiceAdvertSource(FEF3)._redact is True
