# Technical details

The [README](../README.md) is deliberately short and non-technical. This
page is where the engineering detail, the safety fine print, and the
design credit live instead.

## Command-line flags

`find_phone.py` has exactly one option that matters, `--find-phone` — pass
it or the tool refuses to run. Everything else is a minor knob:

| Flag | Default | What it does |
|---|---|---|
| `--port N` | `8646` | web meter port |
| `--host ADDR` | `127.0.0.1` | web meter bind address (keep it localhost unless you know you want it reachable from other machines) |
| `--lang es\|en` | `en` | language of the browser page and console text |
| `--no-redact` | off | show the full Bluetooth address once locked on, instead of the last two octets only. A public Bluetooth address is a stable hardware identifier — do not screen-record with this on |

## Safety and privacy, in full

- Hunt only a device you own. This finds a phone by radio signal
  strength, not by pairing or by asking permission — that is exactly why
  it must never be pointed at someone else's device.
- Addresses are shown **redacted** (last two octets only) by default.
  `--no-redact` reveals the full address; don't screen-record with that on.
- This needs the target's Bluetooth radio to be on and actively
  advertising — usually meaning the screen is on, or it was used
  recently. It is not GPS and does not work through airplane mode.
- Signal strength (RSSI) is a rough, noisy proxy for distance, not a
  ruler — expect the number to wobble. The trend (getting warmer/colder as
  you walk) is the reliable part.

## Why Google Fast Pair

A phone that is not currently paired or connected to anything mostly
advertises **nothing identifying** on purpose — that's the privacy design
Bluetooth address rotation exists for. Google Fast Pair is the practical
exception: modern Android phones keep broadcasting the Fast Pair service
UUID (`0xFEF3`) on their own, so that any earbuds or accessory nearby can
offer a one-tap pairing sheet. Hunting by that service UUID, instead of by
a fixed Bluetooth address, is also what makes this survive an address
rotation mid-hunt: the address changes, the fact that it's advertising
Fast Pair does not.

## How it works, briefly

- `blelib/scan.py` — real BLE scanning over `bleak`. `ServiceAdvertSource`
  filters advertisements down to the strongest device currently advertising
  the target service UUID, and keeps "locking on" again by service UUID
  every time the address underneath it rotates.
- `blelib/hunt.py` / `blelib/signal.py` — turn a stream of raw RSSI
  readings into one honest snapshot: a median over a short live window, a
  warmer/colder trend, staleness detection, and a "this address may be
  gone, not just weak" distinction for when a rotation actually happens.
- `blelib/pathloss.py` — an optional calibration panel in the browser
  page: feed it a couple of (distance, signal) points you measure
  yourself, and it fits a log-distance path-loss model and reports how
  wide the resulting distance estimate really is (indoor multipath means:
  wide — the honest answer is "closer/further", not "N.N metres").
- `blelib/weblive.py` — the browser meter itself: Flask + Server-Sent
  Events, one self-contained HTML page, no external JavaScript or CDN.
- `find_phone.py` — the entire CLI surface. It wires the above together
  with one fixed configuration (`--mode live --service fef3`, in the
  vocabulary the modules use internally) and refuses to do anything else.

Run the test suite with `pytest` — the `blelib` modules carry their own
tests (`tests/test_*.py`), ported unmodified from the source project.
Comments in the source cite an internal design spec by section number
(`SPEC N.N`); those numbers are historical context from that project, not
something a reader here needs to chase down.

## Design origin

This tool's idea — find a lost phone by walking around with a live
Bluetooth signal-strength meter — is not new. It follows a case reported
publicly in August 2026, in which a phone with Find My disabled by MDM
was located by walking an office while watching a Bluetooth
signal-strength meter. The tool that was publicized for that is
[`ben-z/findphone`](https://github.com/ben-z/findphone) (macOS, Swift;
that repository has no `LICENSE` file, meaning all rights are reserved on
its code). Its README was read **only for the design idea** — no line of
its code was used, and none would have been useful here: this is an
independent implementation, in Python, over BlueZ/`bleak`, targeting Fast
Pair instead of a fixed address.

## License

Code in this repository is licensed under the [Apache License 2.0](../LICENSE).

Copyright © 2026 Luis Felipe Ariza Vesga.
