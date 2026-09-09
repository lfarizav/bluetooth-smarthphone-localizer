#!/usr/bin/env python3
"""find_phone.py -- hunt YOUR OWN nearby phone by Bluetooth signal strength.

This tool has exactly one real mode: --find-phone.

It scans for Google Fast Pair's advertised service UUID (0xFEF3) -- the one
thing a modern Android phone reliably broadcasts even when it advertises
nothing else of its own -- locks onto the strongest one nearby, and follows
it across BLE private-address rotations (a phone quietly changes its
Bluetooth address every ~15 minutes for privacy; a plain address-based
tracker loses the target the moment that happens, this one does not). Point
a browser at the URL it prints and walk your own room until the number gets
small.

Read "Before you run this" in README.md first: this hunts by radio signal
strength, not GPS, it only works on a device that is currently advertising
Bluetooth Low Energy (screen usually needs to be on, or the device recently
used), and it is built to help you find YOUR OWN phone -- not to track
anyone else's.

Usage:
    python find_phone.py --find-phone
"""

from __future__ import annotations

import argparse
import sys

from blelib import hunt as hu
from blelib import signal as sg
from blelib import weblive

#: Google Fast Pair's advertised GATT service UUID. Chosen because it is
#: the one BLE service modern Android phones reliably keep broadcasting on
#: their own, with no pairing step and no app installed -- see README.md
#: section "Why Fast Pair" for how this was verified against a real phone.
FAST_PAIR_SERVICE_UUID = "fef3"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="find_phone.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--find-phone",
        action="store_true",
        help="hunt the strongest nearby phone advertising Google Fast Pair "
             "(service UUID 0xFEF3) and follow it across BLE private-"
             "address rotations. This is the only thing this tool does -- "
             "pass it to confirm you mean it.",
    )
    p.add_argument("--port", type=int, default=None,
                    help=f"web meter port (default: {weblive.DEFAULT_PORT})")
    p.add_argument("--host", type=str, default="127.0.0.1",
                    help="web meter bind address (default: 127.0.0.1, "
                         "localhost only -- change only if you know you "
                         "want the meter reachable from other machines)")
    p.add_argument("--lang", choices=("es", "en"), default="en",
                    help="language for the browser meter and console text")
    p.add_argument("--no-redact", action="store_true",
                    help="show the full Bluetooth address once locked on, "
                         "instead of only its last two octets. A Bluetooth "
                         "address is a stable hardware identifier -- do not "
                         "screen-record with this on.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.find_phone:
        print(
            "This tool only does one thing: pass --find-phone to hunt your "
            "own nearby phone by Bluetooth signal strength.\n\n"
            "  python find_phone.py --find-phone\n",
            file=sys.stderr,
        )
        return 2

    try:
        from blelib import scan as sc  # noqa: F401 -- import-checks bleak early
    except ImportError:
        print(
            "bleak is not installed. Install the dependencies first:\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    cfg = hu.HuntConfig(
        live_window_s=sg.LIVE_WINDOW_S,
        trend_window_s=sg.TREND_WINDOW_S,
        trend_threshold_db=sg.TREND_THRESHOLD_DB,
        stale_after_s=5.0,
        redact=not args.no_redact,
    )
    port = args.port or weblive.DEFAULT_PORT
    session = weblive.LiveSession(
        cfg=cfg,
        mode="live",
        target="",
        source_label="advert",
        lang=args.lang,
        service=FAST_PAIR_SERVICE_UUID,
    )
    app = weblive.create_app(session)

    print(
        "find_phone.py -- creado por Luis Felipe Ariza Vesga, con amor."
        if args.lang == "es"
        else "find_phone.py -- created by Luis Felipe Ariza Vesga, with love."
    )
    print(
        "REMEMBER: hunt ONLY a device you own. Addresses are shown "
        "redacted by default; --no-redact reveals them and the result "
        "must not be screen-recorded."
    )
    print(f"Browser meter:  http://{args.host}:{port}/   (Ctrl-C to stop)")
    print("Move a few metres, then stand still ~10 s and let it settle.")

    session.start()
    try:
        app.run(host=args.host, port=port, threaded=True, debug=False)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop()
        print("Stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
