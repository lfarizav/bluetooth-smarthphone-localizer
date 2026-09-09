"""find_phone.py's CLI surface: exactly one real option, --find-phone.

These tests stay at the argparse/return-code layer on purpose -- anything
past that point opens a Bluetooth adapter and a web server, which has no
business running under pytest (and would hang or fail on a CI box with no
Bluetooth hardware). blelib's own test suite (test_scan.py,
test_service_source.py, test_reacquire.py, test_hunt.py, test_signal.py,
test_pathloss.py, test_reading.py) covers everything find_phone.py wires
together.
"""

from __future__ import annotations

import find_phone as fp


def test_find_phone_flag_defaults_to_false():
    args = fp.build_parser().parse_args([])
    assert args.find_phone is False


def test_find_phone_flag_can_be_set():
    args = fp.build_parser().parse_args(["--find-phone"])
    assert args.find_phone is True


def test_default_host_is_localhost_only():
    args = fp.build_parser().parse_args([])
    assert args.host == "127.0.0.1"


def test_default_port_is_none_so_weblive_default_wins():
    args = fp.build_parser().parse_args([])
    assert args.port is None


def test_default_lang_is_english():
    args = fp.build_parser().parse_args([])
    assert args.lang == "en"


def test_no_redact_defaults_to_false_ie_redaction_is_on():
    args = fp.build_parser().parse_args([])
    assert args.no_redact is False


def test_lang_rejects_unknown_language():
    import pytest

    with pytest.raises(SystemExit):
        fp.build_parser().parse_args(["--lang", "fr"])


def test_main_without_find_phone_refuses_and_returns_2(capsys):
    rc = fp.main([])
    assert rc == 2
    err = capsys.readouterr().err
    assert "--find-phone" in err


def test_main_with_unrelated_flags_still_refuses_without_find_phone(capsys):
    rc = fp.main(["--port", "9999", "--lang", "es"])
    assert rc == 2


def test_fast_pair_service_uuid_matches_the_documented_value():
    # Google Fast Pair's GATT service UUID -- see README.md "Why Fast Pair".
    assert fp.FAST_PAIR_SERVICE_UUID == "fef3"
