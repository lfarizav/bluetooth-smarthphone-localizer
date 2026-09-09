"""Real BLE advertisement scanning via ``bleak`` -- privacy-critical.

Everything this module returns to the rest of the lab has already had one
question asked of it: *does a student need to see this to find their own
device, or is it someone else's private hardware identifier riding along for
free?* A public or static-random BLE address is exactly as stable and
trackable as a Wi-Fi MAC -- the whole reason Apple/Google/Microsoft rotate
resolvable private addresses every ~15 minutes (SPEC section 0, module-09) is
that an unredacted address left on a screen or in a log is a standing way to
recognise a specific phone across time and space. So:

* :func:`redact_addr` is applied by default everywhere an address reaches a
  caller. ``--no-redact`` (SPEC 5.7) must be an explicit, informed opt-in.
* Nothing here decodes or stores manufacturer/service *payload* bytes --
  :class:`Discovered` does not even have a field for them. The lab identifies
  a device by address, name and RSSI only, never by what is inside its
  advertisement.
* :func:`discover` is for finding *your own* device among what is in the air.
  It is not, and must never become, a general-purpose tracker; see the
  ``README`` for the plain statement of that boundary.

Import safety
--------------
``bleak`` (and the ``dbus-fast`` it pulls in on Linux) is an optional runtime
dependency from this module's point of view: a machine with no Bluetooth
stack at all must still be able to ``import blelib.scan`` so the pure test
suite and ``validate.py`` run without it. The import is therefore guarded,
and only the functions that actually need a radio raise -- with a plain
``ImportError`` telling the caller what to install, not a traceback from deep
inside this module.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Sequence

from .reading import Reading

try:  # pragma: no cover - exercised implicitly whenever bleak is present
    from bleak import BleakScanner as _BleakScanner
    _BLEAK_IMPORT_ERROR: Exception | None = None
except Exception as _exc:  # pragma: no cover - exercised only without bleak
    _BleakScanner = None
    _BLEAK_IMPORT_ERROR = _exc


def _require_bleak() -> None:
    """Raise a clear, actionable ``ImportError`` -- only called by the
    functions that actually need a radio, never at module import time."""
    if _BleakScanner is None:
        raise ImportError(
            "bleak is not installed, so this machine cannot scan for real "
            "BLE advertisements. Install it with `pip install bleak` (see "
            "requirements.txt) or run the lab in --mode sim instead. / "
            "bleak no esta instalado, asi que esta maquina no puede escanear "
            "anuncios BLE reales. Instalalo con `pip install bleak` (ver "
            "requirements.txt) o ejecuta el laboratorio en --mode sim."
        ) from _BLEAK_IMPORT_ERROR


def bleak_version() -> str | None:
    """Installed ``bleak`` version, or ``None`` if it cannot be determined.

    Reads package metadata rather than ``bleak.__version__`` -- bleak 3.x
    does not export that attribute (confirmed on this machine, bleak 3.0.2)
    -- and never raises: ``run.py --probe`` calls this to report honestly on
    a machine that may not have bleak installed at all.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("bleak")
        except PackageNotFoundError:
            return None
    except Exception:
        return None


class NoAdapterError(RuntimeError):
    """No usable Bluetooth adapter was found through BlueZ.

    Raised by :func:`discover` (and, transitively, by :class:`AdvertSource`)
    so ``run.py`` can print one helpful bilingual line instead of a D-Bus
    traceback -- the failure mode of "no adapter" is common (a laptop with
    Bluetooth disabled in firmware, a container with no D-Bus system bus) and
    deserves a message a student can act on.
    """


@dataclass(frozen=True)
class Discovered:
    """One device seen during :func:`discover` -- identity only, no payload.

    ``addr`` is already redacted if the caller asked for redaction; this
    dataclass carries whatever :func:`discover` decided to hand back, it does
    not redact anything itself (same split of responsibility as
    :class:`~blelib.reading.Reading`).
    """

    addr: str
    name: str
    rssi_dbm: int
    tx_power: int | None
    n_seen: int
    addr_kind: str  # "public" | "random-static" | "rpa" | "nrpa"


@dataclass(frozen=True)
class DeviceFingerprint:
    """One device's observable advertisement SHAPE, for re-acquire matching
    ONLY -- never surfaced to a student as "here is what this device's
    payload contains" the way a general-purpose tracker would (exactly what
    this module's own docstring forbids). :func:`score_reacquire` is the
    ONLY consumer this lab has for these fields; ``weblive.py`` turns a
    score into a numeric confidence plus a short list of "why" reason KEYS
    (resolved to bilingual text there, never a raw payload dump on screen).

    Still no service_data VALUES (the actual payload bytes) and no raw
    manufacturer_data bytes -- only shape: WHICH company IDs and service
    UUIDs were present, and how many total manufacturer-data bytes there
    were. That is enough to tell "probably the same phone, new random
    address" from "some other device" without decoding what the phone is
    actually broadcasting, keeping the same privacy discipline
    :class:`Discovered` holds for identity fields.
    """

    addr: str
    """Raw (never redacted) -- fingerprints are ephemeral, in-process
    matching state, not something a caller displays as-is; the one place a
    raw address survives past this dataclass is the retarget action a
    HUMAN explicitly confirms (weblive.py), same "own device only" boundary
    the rest of this module holds."""

    addr_kind: str
    rssi_dbm: int

    t: float
    """When this fingerprint was FIRST captured (monotonic seconds) --
    :class:`AdvertSource` keeps this fixed at first sighting even as later
    fields (rssi, name, ...) get refreshed on repeat detections, because
    "when did this address first show up" is the signal
    :func:`score_reacquire` needs for time-proximity-to-disappearance
    scoring, not "when did we last see it"."""

    local_name: str
    manufacturer_ids: frozenset[int]

    mfg_payload_len: int
    """Total bytes across every manufacturer_data value -- a coarse
    "shape", not the bytes themselves (see the class docstring)."""

    service_uuids: frozenset[str]
    tx_power: int | None


def _fingerprint_of(addr: str, adv: Any, *, addr_kind: str, name: str,
                    rssi: int, t: float) -> DeviceFingerprint:
    """Build one :class:`DeviceFingerprint` from a live detection.

    Reads exactly the ``AdvertisementData`` fields :func:`score_reacquire`
    scores on -- verified against installed bleak 3.0.2's
    ``bleak/backends/scanner.py:AdvertisementData`` (a ``NamedTuple`` with
    ``manufacturer_data: dict[int, bytes]``, ``service_uuids: list[str]``,
    ``tx_power: int | None``), the same object ``_addr_kind``/``discover``
    already rely on for ``rssi``/``local_name``/``platform_data``.
    ``rssi``/``t`` are taken as parameters rather than re-read from ``adv``/
    the clock here, so the caller's own already-validated RSSI int and
    single ``time.monotonic()`` read are reused instead of parsing twice.

    Service UUIDs are lower-cased before being frozen: different platforms
    are not guaranteed to report the same UUID casing, and a case mismatch
    would fail an exact-set match between two readings of the SAME device
    for a reason that has nothing to do with whether they are the same
    device.
    """
    mfg = getattr(adv, "manufacturer_data", None) or {}
    services = getattr(adv, "service_uuids", None) or []
    tx_power = getattr(adv, "tx_power", None)
    return DeviceFingerprint(
        addr=addr, addr_kind=addr_kind, rssi_dbm=rssi, t=t, local_name=name,
        manufacturer_ids=frozenset(mfg.keys()),
        mfg_payload_len=sum(len(v) for v in mfg.values()),
        service_uuids=frozenset(u.lower() for u in services),
        tx_power=tx_power,
    )


@dataclass(frozen=True)
class ReacquireCandidate:
    """One scored guess at "this address might be the target's new one
    after a private-address rotation" -- see :func:`score_reacquire`.

    ``reasons`` is an ordered list of REASON KEYS (e.g. ``"name"``,
    ``"rssi_continuity"``), not display text -- resolving a key to a
    bilingual sentence is presentation, and presentation belongs entirely
    in ``weblive.py`` (same split :mod:`signal`'s ``Trend`` enum already
    uses: this module hands back a fact, the web module is the only place
    that ever turns a fact into English or Spanish). Never present a guess
    as a certainty: this is why every candidate carries its reasons, not
    just a bare score -- so the page can show its work.
    """

    addr: str
    score: float
    reasons: list[str]
    rssi_dbm: int
    addr_kind: str


#: score_reacquire's weight table -- how much confidence each independently
#: observable signal contributes toward "this is probably the same physical
#: device under a new rotated address." A REASONED heuristic, not a
#: measured constant (no dataset of confirmed real-world rotations exists
#: to fit against) -- same honesty standard as signal.py's
#: TREND_THRESHOLD_DB and this file's own weblive.py sound-cue thresholds.
#: Ordered by how hard the signal is to fake by coincidence:
#:  * local_name is weighted highest -- this app only ever hunts a device
#:    the STUDENT set the name of; two different phones sharing the exact
#:    same user-chosen name, in the same room, at the same moment, is rare
#:    enough that a match here is close to dispositive on its own.
#:  * manufacturer company ID(s) and service UUIDs are OS/app/vendor
#:    fingerprints (an iPhone's Apple company ID, a fitness app's service
#:    UUID) that survive an address rotation on the SAME phone, but are
#:    shared by every OTHER phone of the same make running the same apps --
#:    real signal, weaker alone than a matching name.
#:  * payload length and tx_power are coarse and easy to coincide on by
#:    chance (many BLE stacks default to the same tx_power), so each gets a
#:    small weight.
#:  * RSSI continuity and time-proximity are continuous, distance-like
#:    signals rather than same/different facts -- scored as partial credit
#:    that decays to zero past a threshold (see the two _MAX constants
#:    below), never all-or-nothing.
#: The seven weights sum to exactly 1.0, the scale score_reacquire reports
#: confidence on.
_W_LOCAL_NAME = 0.35
_W_MANUFACTURER_IDS = 0.20
_W_SERVICE_UUIDS = 0.15
_W_PAYLOAD_SHAPE = 0.10
_W_TX_POWER = 0.05
_W_RSSI_CONTINUITY = 0.10
_W_TIME_PROXIMITY = 0.05

#: Payload-length "similar enough" tolerance, in bytes -- a counter/nonce
#: byte or two shifting between two advertisements from the SAME app on the
#: SAME phone is normal; a bigger difference is more likely a genuinely
#: different payload shape (a different app, a different device).
PAYLOAD_LEN_TOLERANCE_BYTES = 2

#: Past this RSSI delta, a candidate gets NO continuity credit at all -- a
#: rotated address belonging to the same phone, observed within seconds to
#: low-minutes of the old one going quiet, should still be roughly the same
#: distance from the scanner. 10 dB is comfortably wider than the Core
#: Spec's own +/-6 dB HCI_Read_RSSI accuracy budget (README section 2.1;
#: also cited in weblive.py's sound-cue docstring), so ordinary measurement
#: noise alone should not zero out this signal.
RSSI_CONTINUITY_MAX_DELTA_DB = 10.0

#: Past this many seconds between the target going quiet and a candidate's
#: first sighting, time proximity gives no credit at all. Sized well under
#: even the *fast* end of a BLE address rotation period (8 minutes = 480s,
#: per https://www.bluetooth.com/blog/enhancing-device-privacy-and-energy-efficiency-with-bluetooth-randomized-rpa-updates/,
#: checked 2026-08-06) -- a candidate that only shows up minutes later is
#: far more likely a device that was simply out of range earlier, not this
#: hunt's target reappearing under a new address.
TIME_PROXIMITY_MAX_GAP_S = 90.0

#: Below this combined score, :func:`rank_reacquire_candidates` drops a
#: candidate entirely rather than suggesting it -- an unfiltered list
#: padded with near-zero-confidence guesses would train a student to click
#: through noise instead of reading the reasons, exactly the "guess
#: presented as a certainty" failure this feature exists to avoid.
MIN_REACQUIRE_SCORE = 0.15

#: Confidence label boundary the page uses -- anything below this is
#: labelled "possible", never "likely". Never present a guess as a
#: certainty.
REACQUIRE_LIKELY_THRESHOLD = 0.55


def score_reacquire(last_seen: DeviceFingerprint, candidate: DeviceFingerprint,
                    *, gap_s: float) -> ReacquireCandidate:
    """How well ``candidate`` matches ``last_seen`` as "the same physical
    device, now advertising under ``candidate.addr`` instead" -- pure, no
    radio, unit-tested directly against hand-built fingerprints (see
    ``tests/test_reacquire.py``).

    ``gap_s`` is the caller's own measure of how long after ``last_seen``
    went quiet ``candidate`` first appeared -- kept as an explicit
    parameter (rather than derived here from the two fingerprints' own
    ``t`` fields) so a test can probe the time-proximity signal in
    isolation, and so :func:`rank_reacquire_candidates` (the one real
    caller) has one obvious place to define what "gap" means. May be
    negative (a candidate that started advertising slightly before the
    target's last-seen packet, e.g. an overlapping rotation) -- scored on
    magnitude, never rejected for being negative.
    """
    reasons: list[str] = []
    score = 0.0

    if last_seen.local_name and candidate.local_name and \
            last_seen.local_name == candidate.local_name:
        score += _W_LOCAL_NAME
        reasons.append("name")

    if last_seen.manufacturer_ids & candidate.manufacturer_ids:
        score += _W_MANUFACTURER_IDS
        reasons.append("manufacturer_id")

    if last_seen.service_uuids & candidate.service_uuids:
        score += _W_SERVICE_UUIDS
        reasons.append("service_uuid")

    if (last_seen.mfg_payload_len or candidate.mfg_payload_len) and \
            abs(last_seen.mfg_payload_len - candidate.mfg_payload_len) <= PAYLOAD_LEN_TOLERANCE_BYTES:
        score += _W_PAYLOAD_SHAPE
        reasons.append("payload_shape")

    if last_seen.tx_power is not None and candidate.tx_power is not None \
            and last_seen.tx_power == candidate.tx_power:
        score += _W_TX_POWER
        reasons.append("tx_power")

    # Strict "<", not "<=": at exactly the max delta/gap the decay formula
    # below already computes zero credit, so this also keeps the reason out
    # of the list at the boundary -- a listed reason should always mean a
    # non-zero contribution, never "technically still in range for 0 points."
    delta_rssi = abs(candidate.rssi_dbm - last_seen.rssi_dbm)
    if delta_rssi < RSSI_CONTINUITY_MAX_DELTA_DB:
        score += _W_RSSI_CONTINUITY * (1.0 - delta_rssi / RSSI_CONTINUITY_MAX_DELTA_DB)
        reasons.append("rssi_continuity")

    gap = abs(gap_s)
    if gap < TIME_PROXIMITY_MAX_GAP_S:
        score += _W_TIME_PROXIMITY * (1.0 - gap / TIME_PROXIMITY_MAX_GAP_S)
        reasons.append("time_proximity")

    return ReacquireCandidate(addr=candidate.addr, score=round(min(score, 1.0), 4),
                              reasons=reasons, rssi_dbm=candidate.rssi_dbm,
                              addr_kind=candidate.addr_kind)


def rank_reacquire_candidates(last_seen: DeviceFingerprint,
                               candidates: Sequence[DeviceFingerprint],
                               ) -> list[ReacquireCandidate]:
    """Score every candidate against ``last_seen`` and return the ones worth
    showing a student, best match first.

    Never includes ``last_seen.addr`` itself (a device cannot be its own
    replacement) and drops anything under :data:`MIN_REACQUIRE_SCORE` --
    see that constant's docstring for why a long tail of near-zero guesses
    is worse than an honest "nothing found". ``gap_s`` for each candidate is
    simply ``candidate.t - last_seen.t`` -- the caller's own
    :class:`AdvertSource` already fixes ``t`` at first-sighting for both
    the target and its siblings (see each dataclass field's docstring), so
    this difference is exactly "how long after the target's last packet did
    this OTHER address first show up."
    """
    scored = [
        score_reacquire(last_seen, c, gap_s=c.t - last_seen.t)
        for c in candidates if c.addr != last_seen.addr
    ]
    scored = [s for s in scored if s.score >= MIN_REACQUIRE_SCORE]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored


def classify_address(addr: str) -> str:
    """BLE random-address subtype from the top two bits of the MSB octet.

    Per the Bluetooth Core Specification, Vol 6 ("Low Energy Controller"),
    Part B ("Link Layer Specification"), Section 1.3.2 "Random Device
    Address": a Random Device Address encodes its subtype in its two most
    significant bits -- ``11`` static, ``01`` resolvable private (RPA),
    ``00`` non-resolvable private (NRPA); ``10`` is reserved. Independently
    corroborated (checked 2026-08-06) against Novel Bits' worked examples
    ("Bluetooth Addresses & Privacy", https://novelbits.io/bluetooth-address-privacy-ble/,
    e.g. static example ``C3:5A:...`` -> ``C3`` = ``1100 0011``, top two bits
    ``11``) and against BlueZ's own address-printing convention
    (``lib/bluetooth/bluetooth.c:ba2str`` prints ``b[5]:b[4]:...:b[0]``, i.e.
    the *left-most* octet of the familiar colon-form string is the most
    significant octet -- the same convention ``hcitool``/``bluetoothctl``
    use, and the one this function relies on).

    This is deliberately a pure bit-decoder and nothing more: a *public*
    (IEEE OUI-assigned) address has no such reserved-bit convention at all,
    so this function cannot and does not claim to detect "public" from the
    address alone. :func:`discover` only calls this once BlueZ has already
    flagged a device's ``AddressType`` as ``"random"`` over D-Bus; an address
    BlueZ has not flagged random is read as ``"public"`` upstream, without
    ever reaching here. The one address-only case this function does resolve
    on its own is the reserved ``10`` pattern: since that is not a legal
    random-address subtype, an address landing there is read as ``"public"``
    -- the conservative reading, rather than inventing a fourth random kind
    the Core Spec does not define.
    """
    first_octet_hex = addr.split(":")[0]
    first_octet = int(first_octet_hex, 16)
    top_bits = (first_octet >> 6) & 0b11
    if top_bits == 0b11:
        return "random-static"
    if top_bits == 0b01:
        return "rpa"
    if top_bits == 0b00:
        return "nrpa"
    return "public"  # 0b10 is reserved -- not a legal random subtype


def redact_addr(addr: str) -> str:
    """Keep only the OUI-free tail: ``"AA:BB:CC:DD:EE:FF"`` -> ``"XX:XX:XX:XX:EE:FF"``.

    A public BLE address is a stable hardware identifier -- exactly the kind
    of thing that must never reach a screen recording or a log a student
    shares later. Only the last two octets survive,
    which is enough to tell "same device, different scan" apart on-screen
    without being enough to look the device up.
    """
    parts = addr.split(":")
    if len(parts) != 6:
        return addr  # not a colon-form MAC address -- nothing we know how to redact
    return "XX:XX:XX:XX:" + ":".join(parts[4:])


def _address_type(adv: Any) -> str | None:
    """BlueZ's own ``AddressType`` ("public" | "random") for this advertisement.

    ``bleak``'s BlueZ backend stashes ``(object_path, device1_props)`` in
    ``AdvertisementData.platform_data`` (verified against
    ``bleak/backends/bluezdbus/scanner.py:_handle_advertising_data``,
    installed bleak 3.0.2: ``platform_data=(path, props)`` where ``props`` is
    the raw ``org.bluez.Device1`` property dict). That dict is where the
    controller's own public/random flag lives -- this function never guesses
    it from the address bytes.
    """
    platform_data = getattr(adv, "platform_data", None)
    if not platform_data or len(platform_data) < 2:
        return None
    props = platform_data[1]
    if not isinstance(props, dict):
        return None
    value = props.get("AddressType")
    return value if isinstance(value, str) else None


def _addr_kind(addr: str, adv: Any) -> str:
    """"public" unless BlueZ has flagged this device's address as random."""
    if _address_type(adv) == "random":
        return classify_address(addr)
    return "public"


#: Printed once by :func:`discover` -- discovery exists so a student can pick
#: their own device out of what is in the air, not as an invitation to watch
#: someone else's. See the privacy notice in README.md.
OWN_DEVICE_ONLY_NOTICE = (
    "Reminder: hunt YOUR OWN device only. This list shows what is in the "
    "air because you cannot pick a target any other way -- it is not an "
    "invitation to track someone else's phone. / "
    "Recordatorio: caza SOLO tu propio dispositivo. Esta lista muestra lo "
    "que hay en el aire porque no hay otra forma de elegir un objetivo -- "
    "no es una invitacion a rastrear el telefono de otra persona."
)

_NO_ADAPTER_MSG = (
    "No usable Bluetooth adapter was found through BlueZ (org.bluez exposed "
    "no org.bluez.Adapter1 object, or the D-Bus call itself failed). Check "
    "`bluetoothctl list` and that bluetooth.service is running. / "
    "No se encontro ningun adaptador Bluetooth utilizable a traves de "
    "BlueZ (org.bluez no expuso ningun objeto org.bluez.Adapter1, o la "
    "llamada D-Bus fallo). Verifica `bluetoothctl list` y que "
    "bluetooth.service este activo."
)


async def _list_adapters_async() -> list[dict]:
    """Core adapter enumeration -- may raise; :func:`adapters` never does.

    Queries ``org.bluez``'s standard ``org.freedesktop.DBus.ObjectManager``
    (``GetManagedObjects`` on path ``/``) rather than shelling out to
    ``hciconfig`` -- this is the same call BlueZ's own ``bluetoothctl`` uses
    to enumerate adapters, and it is what ``bleak`` already depends on
    (``dbus-fast``, verified installed alongside bleak 3.0.2). Live-verified
    on this machine 2026-08-06: returns exactly one adapter,
    ``/org/bluez/hci0`` with ``Address: E0:D5:5D:71:A8:B7``, ``Powered: True``
    -- matching SPEC section 3's recorded facts.
    """
    from dbus_fast import BusType, Message, MessageType, unpack_variants
    from dbus_fast.aio import MessageBus

    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    try:
        reply = await bus.call(Message(
            destination="org.bluez",
            path="/",
            interface="org.freedesktop.DBus.ObjectManager",
            member="GetManagedObjects",
        ))
        if reply.message_type != MessageType.METHOD_RETURN:
            raise RuntimeError(
                f"GetManagedObjects on org.bluez failed: "
                f"{reply.error_name or reply.message_type}")
        objects = unpack_variants(reply.body[0])
    finally:
        bus.disconnect()

    out: list[dict] = []
    for path, ifaces in objects.items():
        props = ifaces.get("org.bluez.Adapter1")
        if props is None:
            continue
        raw_addr = props.get("Address", "")
        out.append({
            "name": path.rsplit("/", 1)[-1],
            "address": redact_addr(raw_addr) if raw_addr else raw_addr,
            "powered": bool(props.get("Powered", False)),
        })
    return out


def adapters() -> list[dict]:
    """What Bluetooth adapters this machine has, via BlueZ over D-Bus.

    Never raises -- returns ``[]`` on any failure (no system bus, no BlueZ,
    permission trouble). This feeds ``run.py --probe``, whose entire job is
    to report the truth rather than crash (same ethos as
    :func:`~blelib.hci.probe_link_rssi`). Addresses are always redacted
    (:func:`redact_addr`): this contract has no opt-out, and a laptop's own
    adapter address is exactly the kind of stable hardware identifier the
    privacy rule this module exists for is about, regardless of whose it is.
    """
    try:
        return asyncio.run(_list_adapters_async())
    except Exception:
        return []


async def discover(duration_s: float = 8.0, *, redact: bool = True) -> list[Discovered]:
    """Passive-listen for ``duration_s`` seconds and report what was heard.

    For finding *your own* device among what is in the air -- see
    :data:`OWN_DEVICE_ONLY_NOTICE`, printed once per call. Only identity
    fields reach :class:`Discovered`: no manufacturer-data or service-data
    dump, ever (SPEC 5.7).

    Raises :class:`NoAdapterError` up front if BlueZ reports no adapter at
    all, rather than letting ``bleak`` fail deep inside its D-Bus plumbing
    with a much less actionable traceback.
    """
    _require_bleak()
    print(OWN_DEVICE_ONLY_NOTICE)

    try:
        found = await _list_adapters_async()
    except Exception as exc:
        raise NoAdapterError(_NO_ADAPTER_MSG) from exc
    if not found:
        raise NoAdapterError(_NO_ADAPTER_MSG)

    seen: dict[str, dict] = {}

    def _on_detection(device: Any, adv: Any) -> None:
        addr = device.address
        try:
            rssi = int(adv.rssi)
        except (TypeError, ValueError):
            # Same guard as AdvertSource.stream(): a detection that arrives
            # with a missing or unparseable RSSI is skipped, not fatal. Without
            # it a single malformed advertisement takes down the whole
            # discovery pass -- and discovery is the step the student runs
            # first, on whatever happens to be advertising in the room.
            return
        name = device.name or getattr(adv, "local_name", None) or ""
        entry = seen.get(addr)
        if entry is None:
            seen[addr] = {
                "name": name, "rssi": rssi, "tx": adv.tx_power,
                "n": 1, "kind": _addr_kind(addr, adv),
            }
        else:
            entry["rssi"] = rssi
            if name:
                entry["name"] = name
            entry["n"] += 1

    async with _BleakScanner(_on_detection):
        await asyncio.sleep(duration_s)

    out = []
    for addr, entry in seen.items():
        shown_addr = redact_addr(addr) if redact else addr
        out.append(Discovered(
            addr=shown_addr, name=entry["name"], rssi_dbm=entry["rssi"],
            tx_power=entry["tx"], n_seen=entry["n"], addr_kind=entry["kind"]))
    return out


class AdvertSource:
    """Streams :class:`~blelib.reading.Reading` for ONE target address.

    Two ways to consume it, both built on the same underlying scan:

    * :meth:`stream` -- an async generator, what ``run.py``'s live CLI mode
      consumes directly.
    * :meth:`feed_into` -- pushes the same readings into a
      :class:`~blelib.hunt.HuntEngine`, matching the SPEC 5.7 docstring's
      description of this class ("streams Readings for ONE target address
      into a HuntEngine"). It is a thin wrapper over :meth:`stream`, not a
      second scan implementation.

    Filtering happens on the *unredacted* address (the target the caller
    asked to hunt is necessarily known to them already); redaction is only
    applied to what ends up on a :class:`~blelib.reading.Reading`, i.e. what
    could reach a screen or a log.
    """

    def __init__(self, target: str, *, redact: bool = True,
                 track_siblings: bool = True, siblings_maxlen: int = 64) -> None:
        self._target = target
        self._redact = redact
        self._track_siblings = track_siblings
        self._siblings_maxlen = siblings_maxlen

        #: Best-effort, latest-known fingerprint of the TARGET itself --
        #: read by weblive.LiveSession from OUTSIDE this coroutine's own
        #: thread, purely to know the target's own addr_kind and shape for
        #: re-acquire scoring once it vanishes. Deliberately a plain
        #: attribute, not behind a lock: the same "eventually consistent,
        #: worst case one stale read, never a crash" convention
        #: weblive.LiveSession.error already uses for the same reason (one
        #: writer thread, one best-effort reader).
        self.last_fingerprint: DeviceFingerprint | None = None

        #: Every OTHER device seen while this source has been streaming,
        #: keyed by raw address, ``t`` fixed at FIRST sighting -- the raw
        #: material :func:`rank_reacquire_candidates`'s caller
        #: (weblive.LiveSession) ranks once the target goes quiet. No extra
        #: scan is needed for this: BleakScanner already receives every
        #: device's detection while this source streams, this just also
        #: keeps what it was already discarding. Bounded the same way
        #: ``ReadingLog`` is bounded (reading.py's own docstring): a room is
        #: not empty, and a hunt can run for a long time.
        self._siblings: dict[str, DeviceFingerprint] = {}

    def siblings(self) -> list[DeviceFingerprint]:
        """Snapshot of every non-target device fingerprint seen so far.

        A ``list(...)`` copy of the current values so a caller iterating
        this on another thread (weblive.LiveSession's SSE/request thread,
        while this source's own feed thread keeps mutating the dict) never
        sees it change mid-iteration.
        """
        return list(self._siblings.values())

    async def stream(self) -> AsyncIterator[Reading]:
        """Yield a :class:`~blelib.reading.Reading` each time the target advertises.

        Runs until the consumer stops iterating. Because this is an async
        generator wrapping an ``async with BleakScanner(...)`` block, a
        caller that breaks out of ``async for`` early should call
        ``await agen.aclose()`` (or use ``contextlib.aclosing``) so the scan
        is actually stopped rather than left running until garbage
        collection -- plain early ``break`` does not trigger that cleanup.
        """
        _require_bleak()
        queue: asyncio.Queue[Reading] = asyncio.Queue()

        def _on_detection(device: Any, adv: Any) -> None:
            addr = device.address
            try:
                rssi = int(adv.rssi)
            except (TypeError, ValueError):
                return
            now = time.monotonic()
            name = device.name or getattr(adv, "local_name", None) or ""

            if addr == self._target:
                kind = _addr_kind(addr, adv)
                self.last_fingerprint = _fingerprint_of(
                    addr, adv, addr_kind=kind, name=name, rssi=rssi, t=now)
                out_addr = redact_addr(addr) if self._redact else addr
                queue.put_nowait(Reading(rssi_dbm=rssi, t=now,
                                         source="advert", addr=out_addr))
            elif self._track_siblings:
                # A device this hunt is NOT targeting, seen while scanning
                # for the one it is -- kept only as fingerprint shape (never
                # surfaced raw), so a rotated target can later be matched
                # against it. See DeviceFingerprint's own docstring for why
                # this does not violate the module's "identify by address/
                # name/RSSI only" discipline: matching signals, not a
                # payload dump.
                kind = _addr_kind(addr, adv)
                existing = self._siblings.get(addr)
                first_t = existing.t if existing is not None else now
                self._siblings[addr] = _fingerprint_of(
                    addr, adv, addr_kind=kind, name=name, rssi=rssi, t=first_t)
                if len(self._siblings) > self._siblings_maxlen:
                    oldest_addr = min(self._siblings,
                                      key=lambda a: self._siblings[a].t)
                    if oldest_addr != addr:
                        del self._siblings[oldest_addr]

        async with _BleakScanner(_on_detection):
            while True:
                yield await queue.get()

    async def feed_into(self, engine: Any, duration_s: float | None = None) -> None:
        """Push readings for the target into ``engine.feed`` (a ``HuntEngine``).

        ``engine`` is typed as ``Any`` rather than importing
        :class:`~blelib.hunt.HuntEngine` at module load time -- this module
        must stay importable (SPEC section 4: ``scan.py`` is the one side of
        the ``[pure]``/I-O split) even before ``hunt.py`` exists in a given
        checkout, and duck-typing ``engine.feed(reading)`` costs nothing at
        the one call site that needs it.
        """
        agen = self.stream()
        deadline = None if duration_s is None else time.monotonic() + duration_s
        try:
            async for reading in agen:
                engine.feed(reading)
                if deadline is not None and time.monotonic() >= deadline:
                    break
        finally:
            await agen.aclose()


# ------------------------------------------------------------------ identity
@dataclass(frozen=True)
class TargetIdentity:
    """Who the meter is pointed at, as far as BlueZ can tell us.

    The meter spent its whole development showing a number and never saying
    WHOSE number it was. That is a real usability failure and not a cosmetic
    one: a reading of 0 next to no device name is indistinguishable from a
    broken app, and the first person to run it against their own connected
    phone reasonably concluded it was not working at all.

    Every field is best-effort. BlueZ knows the alias and the connection
    state for devices it has paired or discovered; for a bare advertising
    address it may know nothing but the address itself, and that is a fine
    answer -- an honest "unknown name" beats a confident wrong one.
    """

    addr: str
    name: str = ""
    connected: bool = False
    paired: bool = False
    known_to_bluez: bool = False

    @property
    def display(self) -> str:
        """What to put on screen: the name when there is one, else the address."""
        return self.name or self.addr


async def target_identity_async(addr: str, *, redact: bool = True) -> TargetIdentity:
    """Look up ``addr``'s BlueZ alias and connection state. May raise."""
    from dbus_fast import BusType, Message, MessageType, unpack_variants
    from dbus_fast.aio import MessageBus

    shown = redact_addr(addr) if redact else addr

    bus = await MessageBus(bus_type=BusType.SYSTEM).connect()
    try:
        reply = await bus.call(Message(
            destination="org.bluez", path="/",
            interface="org.freedesktop.DBus.ObjectManager",
            member="GetManagedObjects"))
        if reply.message_type != MessageType.METHOD_RETURN:
            return TargetIdentity(addr=shown)
        objects = unpack_variants(reply.body[0])
    finally:
        bus.disconnect()

    for _path, ifaces in objects.items():
        props = ifaces.get("org.bluez.Device1")
        if not props:
            continue
        if str(props.get("Address", "")).upper() != addr.upper():
            continue
        # Alias is the right field, not Name: BlueZ falls back to Name when no
        # alias is set, so Alias is always at least as informative, and it is
        # what bluetoothctl shows the user.
        return TargetIdentity(
            addr=shown,
            name=str(props.get("Alias") or props.get("Name") or ""),
            connected=bool(props.get("Connected", False)),
            paired=bool(props.get("Paired", False)),
            known_to_bluez=True,
        )
    return TargetIdentity(addr=shown)


def target_identity(addr: str, *, redact: bool = True) -> TargetIdentity:
    """Never-raising wrapper. Identity is a nicety; a meter must not die for it."""
    if not addr:
        return TargetIdentity(addr="")
    try:
        import asyncio
        return asyncio.run(target_identity_async(addr, redact=redact))
    except Exception:                                  # noqa: BLE001
        return TargetIdentity(addr=redact_addr(addr) if redact else addr)


# --------------------------------------------------- fingerprint targeting
#: Google Fast Pair / Nearby. Modern Android phones advertise it continuously
#: even when otherwise idle, which makes it the one reliable handle on a phone
#: that refuses to advertise anything else. Confirmed on this machine
#: 2026-08-06: the connected phone's own `org.bluez.Device1.ServiceData`
#: carried `0000fef3-...`, and the strongest nearby advertiser carried the same
#: UUID at -44..-39 dBm while the phone sat next to the laptop.
FAST_PAIR_UUID = "0000fef3-0000-1000-8000-00805f9b34fb"


def _norm_uuid(u: str) -> str:
    """Accept 'fef3', '0xfef3' or the full 128-bit form; return the full form."""
    from bleak.uuids import normalize_uuid_str

    s = str(u).strip().lower().removeprefix("0x")
    if len(s) == 4:
        s = f"0000{s}-0000-1000-8000-00805f9b34fb"
    return normalize_uuid_str(s)


class ServiceAdvertSource:
    """Hunt a device by what it ADVERTISES, not by the address it wears.

    A phone's advertising address is a resolvable/non-resolvable private
    address that Bluetooth rotates every 8-15 minutes precisely so it cannot
    be followed. Targeting one is therefore targeting something with a
    guaranteed expiry date, and the meter's most confusing failure -- a number
    that freezes forever -- is what that expiry looks like from the outside.

    A service UUID survives the rotation, because it describes what the device
    *is* rather than what it is currently called. So this source locks onto the
    strongest advertiser carrying ``service_uuid`` and keeps following it; when
    that address goes quiet past ``relock_after_s`` it re-locks onto whatever
    is now strongest with the same UUID, which after a rotation is the same
    physical device under its new name.

    The honest limit, stated because it decides whether this is safe to use:
    this is proximity-based re-identification, NOT cryptographic identity. If
    two devices in range advertise the same UUID, "strongest" can pick the
    wrong one. It is right for finding your own phone in your own office, and
    it is not evidence of identity. Where correctness matters more than
    convenience, target an address and confirm re-acquisitions by hand
    (:func:`score_reacquire`).
    """

    def __init__(self, service_uuid: str = FAST_PAIR_UUID, *,
                 redact: bool = True, relock_after_s: float = 20.0,
                 settle_s: float = 5.0) -> None:
        self.service_uuid = _norm_uuid(service_uuid)
        self._redact = redact
        self._relock_after_s = float(relock_after_s)
        # Locking onto the FIRST device heard would be a coin toss: the first
        # advertisement to arrive is whichever device happened to transmit
        # next, not the nearest one. So the lock stays open for a short settle
        # window and keeps taking the strongest, which is the phone in your
        # hand rather than the one two offices away.
        self._settle_s = float(settle_s)
        self._first_seen_t: float | None = None
        #: Address currently being followed, unredacted. Published so a caller
        #: can show which one, and so a re-lock is observable rather than silent.
        self.locked_addr: str | None = None
        #: Monotonic time of the most recent re-lock, for the same reason.
        self.last_relock_t: float | None = None
        #: Same contract as :class:`AdvertSource`. `weblive` reads this to
        #: decide whether a vanished target's address was a private one, so a
        #: source that does not publish it makes the browser meter fail with an
        #: AttributeError rather than degrade -- found the hard way, 2026-08-06.
        self.last_fingerprint: DeviceFingerprint | None = None

    def siblings(self) -> list[DeviceFingerprint]:
        """Always empty -- same contract as :meth:`AdvertSource.siblings`.

        The manual re-acquire flow exists to survive an address rotation. This
        source already survives one on its own by re-locking on the service
        UUID, so there is nothing for a human to confirm and no candidate list
        to offer. Returning an empty list (rather than omitting the method)
        keeps `weblive`'s duck-typed source contract satisfied.
        """
        return []

    async def stream(self) -> AsyncIterator[Reading]:
        """Yield readings from whichever address currently carries the UUID."""
        _require_bleak()
        queue: asyncio.Queue[Reading] = asyncio.Queue()
        best: dict[str, tuple[float, int]] = {}      # addr -> (last_seen_t, rssi)

        def _on_detection(device: Any, adv: Any) -> None:
            uuids = {_norm_uuid(u) for u in (adv.service_uuids or ())}
            if self.service_uuid not in uuids:
                return
            try:
                rssi = int(adv.rssi)
            except (TypeError, ValueError):
                return

            now = time.monotonic()
            addr = device.address
            best[addr] = (now, rssi)

            # Lock on first sight, or re-lock when the locked address has gone
            # quiet long enough that a rotation is the likely explanation.
            if self._first_seen_t is None:
                self._first_seen_t = now
            settling = (now - self._first_seen_t) <= self._settle_s

            locked = self.locked_addr
            stale = (locked is None
                     or locked not in best
                     or (now - best[locked][0]) > self._relock_after_s)
            if stale or settling:
                # Only devices heard RECENTLY may win the lock. Without this,
                # a device that stopped advertising a minute ago can still beat
                # a live one on its stale cached RSSI -- which is precisely the
                # rotated-away address we are trying to move off.
                fresh = {a: v for a, v in best.items()
                         if (now - v[0]) <= self._relock_after_s}
                pool = fresh or best
                candidate = max(pool.items(), key=lambda kv: kv[1][1])[0]
                if candidate != locked:
                    self.locked_addr = candidate
                    self.last_relock_t = now
                locked = self.locked_addr

            if addr != locked:
                return
            name = device.name or getattr(adv, "local_name", None) or ""
            self.last_fingerprint = _fingerprint_of(
                addr, adv, addr_kind=_addr_kind(addr, adv), name=name,
                rssi=rssi, t=now)
            shown = redact_addr(addr) if self._redact else addr
            queue.put_nowait(Reading(rssi_dbm=rssi, t=now,
                                     source="advert", addr=shown))

        async with _BleakScanner(_on_detection):
            while True:
                yield await queue.get()
