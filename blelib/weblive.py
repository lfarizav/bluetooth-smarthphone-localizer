"""Browser dashboard for the BLE proximity hunt: Flask + Server-Sent Events +
(SPEC 5.9 -- an internal design note kept for context):

* :func:`build_payload`       -- pure: a :class:`~blelib.hunt.HuntSnapshot`
  (plus the accumulated calibration state) -> the JSON the page renders.
* :func:`handle_control`      -- pure: a control-panel action -> a staged
  change on a :class:`~blelib.hunt.HuntConfig`.
* :func:`add_calibration_point` / :func:`calibration_payload` -- pure:
  accumulate (distance, RSSI) pairs and fit `pathloss.fit_log_distance`
  against them -- this is where the lab's honest finding (SPEC section 1)
  becomes a number on screen instead of a claim in a README.
* :class:`LiveSession`        -- owns the :class:`HuntEngine`, the feeding
  thread (sim / replay / live), and the calibration accumulator.
* :func:`create_app`          -- wraps a session in Flask: `/`, `/stream`
  (SSE), `/control` (POST), `/healthz`.

Every user-visible string lives in :data:`LIVE_STRINGS`, a single nested
`{"es": ..., "en": ...}` table, and every string the page renders is resolved
to a language in Python before it reaches JSON -- the browser never chooses
between two spans of text, it only ever displays what :func:`build_payload`
handed it. :func:`bilingual_gaps` walks that table so completeness is a build
gate, not a habit.

No CDN, no external JavaScript, no chart library, no web fonts: the page is
one file and works with the network off, same discipline as lab-01.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np

from .hunt import HuntConfig, HuntEngine, HuntSnapshot
from .pathloss import (
    PathLossFit,
    distance_interval,
    fit_log_distance,
    range_ratio,
)
from .reading import UNIT_DBM, Reading

DEFAULT_PORT = 8646
STREAM_HZ = 8.0

#: Hard floor on the SSE generator's per-message sleep -- same reasoning as
#: lab-01: bounds the loop even if building one snapshot somehow takes
#: longer than the stream period, so a slow serialiser can never turn the
#: streamer into a busy-wait.
MIN_STREAM_SLEEP = 0.01

#: How many discrete levels :func:`blelib.signal.sparkline_levels` maps a
#: reading into. HuntSnapshot carries the mapped values (SPEC 5.5, `spark:
#: list[int]`) but not the level count they were mapped against -- the page
#: needs it to turn a level back into a bar height, so it rides along here
#: rather than being a magic number baked into the JavaScript.
SPARK_N_LEVELS = 8

#: pathloss.fit_log_distance's own floor (see its docstring): with fewer
#: than three points sigma cannot be measured at all -- there is no honest
#: fit to show yet, only a count of how many more points are needed.
CALIB_MIN_POINTS = 3

# --------------------------------------------------------------- sound cue
#: Geiger-counter proximity audio: closer = faster clicks (and higher
#: pitch). The MAPPING (this section) is pure Python and unit-tested
#: directly (`test_weblive.py`); actually producing sound is entirely
#: client-side Web Audio API in PAGE's JS below (SPEC 5.9's own split: no
#: browser-only behaviour lives in Python, no interpretation logic lives in
#: JS). Nobody has listened to this and confirmed it sounds right -- that
#: is not something a headless test can do; only the human ear can.
#:
#: dBm zones deliberately reuse `signal.PROXIMITY_BANDS`' own thresholds
#: (via `HuntSnapshot.band_key`, already computed upstream) rather than a
#: second, independent table -- the sound and the on-screen band label can
#: then never disagree. One audible step is therefore worth exactly one
#: band gap, 12-15 dB (PROXIMITY_BANDS' own floors), comfortably wider
#: than the Core Spec's own +-6 dB HCI_Read_RSSI accuracy budget (README
#: section 2.1). That is deliberate: mapping every 1 dB to an audibly
#: distinct rate would have the ear chasing instrument noise, not real
#: proximity change.
_SOUND_BY_BAND_KEY: dict[str, tuple[float, float]] = {
    # band_key -> (click_hz, pitch_hz)
    "arms_reach":  (8.0, 1400.0),
    "same_table":  (5.0, 1100.0),
    "same_room":   (2.5, 850.0),
    "far":         (1.2, 650.0),
    "very_far":    (0.5, 450.0),
}
_SOUND_FALLBACK_ZONE = "very_far"

#: Link-mode (golden-range) sound thresholds. NOT a measured constant --
#: the Bluetooth Core Spec does not size the Golden Receive Power Range
#: itself, so +-2 dB is a reasoned, documented choice (roughly a third of
#: the -8..+6 dB spread the one real walk test in evidence/
#: link-walk-2026-08-06.json actually observed), not a spec value. Exactly
#: THREE zones on purpose -- hci.py's own module docstring calls this a
#: "three-zone indicator"; a fourth or fifth zone here would invent
#: resolution the golden-range delta does not have. See PAGE's JS: this
#: mode intentionally does NOT get a smooth click-rate gradient.
GOLDEN_SOUND_CLOSE_FLOOR = 2.0
GOLDEN_SOUND_FAR_CEILING = -2.0
_SOUND_GOLDEN_CLOSE = ("golden_close", 8.0, 1400.0)
_SOUND_GOLDEN_INSIDE = ("golden_inside", 3.0, 900.0)
_SOUND_GOLDEN_FAR = ("golden_far", 1.0, 550.0)

#: Played instead of a proximity click whenever the display is stale or has
#: no reading yet. NEVER derived from the last known value (this is the
#: non-negotiable part): a click rate computed from a stale reading would
#: keep telling the walker "you are still this close" when nothing has
#: actually been measured recently -- exactly the kind of overstated
#: instrument this whole lab argues against. `click_hz=0.0` tells the
#: browser "no repeating click"; it plays this sparse, distinctly
#: different-sounding waiting pulse instead (see PAGE's JS `soundTick`),
#: never silence that could be mistaken for a bug and never a proximity
#: click at any rate.
SOUND_WAITING_PITCH_HZ = 300.0
SOUND_WAITING_PULSE_HZ = 0.4  # one soft pulse every 2.5 s while waiting


def sound_cue(snap: HuntSnapshot) -> dict:
    """Pure: one :class:`~blelib.hunt.HuntSnapshot` -> the Web Audio
    proximity cue the browser should play right now.

    Decides WHAT to play (a click rate, a pitch, a zone key), never HOW --
    no ``AudioContext``, nothing that touches a browser, so this is
    testable headlessly with a hand-built snapshot exactly like
    :func:`build_payload` (see this module's own testing-seam discipline,
    SPEC 5.9). Staleness always wins over the reading, and link-mode
    (golden-range) readings only ever resolve to one of three zones -- see
    the constants above for why both of those are non-negotiable, not
    stylistic choices.
    """
    if snap.stale or snap.rssi_dbm is None:
        return {"active": False, "zone": "waiting", "click_hz": 0.0,
                "pitch_hz": SOUND_WAITING_PITCH_HZ,
                "pulse_hz": SOUND_WAITING_PULSE_HZ}

    if snap.units == UNIT_DBM:
        click_hz, pitch_hz = _SOUND_BY_BAND_KEY.get(
            snap.band_key, _SOUND_BY_BAND_KEY[_SOUND_FALLBACK_ZONE])
        return {"active": True, "zone": snap.band_key, "click_hz": click_hz,
                "pitch_hz": pitch_hz, "pulse_hz": 0.0}

    # golden_range_db: three zones ONLY (module docstring above) -- band_key
    # is already the suppressed NOT_APPLICABLE sentinel here (hunt.py), so
    # this classifies the raw value directly instead of reusing band_key.
    value = float(snap.rssi_dbm)
    if value >= GOLDEN_SOUND_CLOSE_FLOOR:
        zone, click_hz, pitch_hz = _SOUND_GOLDEN_CLOSE
    elif value <= GOLDEN_SOUND_FAR_CEILING:
        zone, click_hz, pitch_hz = _SOUND_GOLDEN_FAR
    else:
        zone, click_hz, pitch_hz = _SOUND_GOLDEN_INSIDE
    return {"active": True, "zone": zone, "click_hz": click_hz,
            "pitch_hz": pitch_hz, "pulse_hz": 0.0}

#: Canonical Heritage palette (course-builder/course_builder/render/themes),
#: copied verbatim from lab-01's weblive.py -- same repo, same author, one
#: definition of "what dark/light mode look like" across every live lab.
THEME = {
    "dark": {"bg": "#1E1E1E", "card": "#161b22", "text": "#F5F5F5",
             "muted": "#BDBDBD", "heading": "#FFFFFF", "accent": "#81D4FA",
             "accent2": "#F48FB1", "accent3": "#A5D6A7", "border": "#30363d",
             "vivid": "#00D9FF"},
    "light": {"bg": "#F7F9FC", "card": "#FFFFFF", "text": "#1E1E1E",
              "muted": "#546E7A", "heading": "#14213D", "accent": "#0277BD",
              "accent2": "#AD1457", "accent3": "#2E7D32", "border": "#D8DEE9",
              "vivid": "#0277BD"},
}


# --------------------------------------------------------------- strings
#: The single source of truth for every user-visible string in this module,
#: nested by section. :func:`bilingual_gaps` walks this dict rather than a
#: hand-maintained checklist, so a table added later (a new calibration
#: field, a new trend word) is covered automatically -- SPEC section 9's "no
#: hardcoded English anywhere" is enforced here, not just followed.
LIVE_STRINGS: dict[str, dict] = {
    "ui": {
        "title": {"es": "lab-06 · caza BLE en vivo",
                  "en": "lab-06 · live BLE hunt"},
        "instruction": {"es": "Camina unos metros, luego quédate quieto ~10 s.",
                        "en": "Move a few metres, then stand still ~10 s."},
        "stale_label": {"es": "SIN SEÑAL", "en": "NO SIGNAL"},
        # Who the meter is pointed at. Absent from the page for most of this
        # lab's development, which made a legitimate reading of 0 next to no
        # device name look exactly like a broken app.
        "target_label": {"es": "Objetivo", "en": "Target"},
        "target_none": {"es": "sin objetivo — modo simulación",
                        "en": "no target — simulation mode"},
        "target_connected": {"es": "conectado", "en": "connected"},
        "target_paired": {"es": "emparejado", "en": "paired"},
        "target_unknown_device": {"es": "no conocido por BlueZ",
                                  "en": "not known to BlueZ"},
        # Deliberately holds the OTHER language's code: the button always
        # shows what clicking it switches TO, same convention as lab-01's
        # `#lang` button. That makes this table's own {es, en} shape do
        # double duty as the toggle logic -- no special-casing needed
        # anywhere this table is walked (rendering, or bilingual_gaps).
        "lang_button": {"es": "EN", "en": "ES"},
        "sound_enable_button": {"es": "🔈 Activar sonido", "en": "🔈 Enable sound"},
        "sound_on": {"es": "🔊 sonido: activado", "en": "🔊 sound: on"},
        "sound_off": {"es": "🔇 sonido: desactivado", "en": "🔇 sound: off"},
        "sound_step_note": {
            "es": "Cada paso de sonido cubre varios dB (igual que las bandas "
                  "de proximidad) — el RSSI en sí sólo tiene ±6 dB de "
                  "exactitud (Core Spec), así que un paso por cada 1 dB "
                  "perseguiría ruido, no señal.",
            "en": "Each sound step spans several dB (same as the proximity "
                  "bands) — RSSI itself is only ±6 dB accurate (Core "
                  "Spec), so a step per 1 dB would chase noise, not "
                  "signal.",
        },
        # -- vanished / re-scan / re-acquire (bug: "I have to reboot the
        # app to see the change" -- an RPA rotation, not a streaming bug;
        # this whole group is the in-page fix, see LiveSession) ------------
        "vanished_badge": {"es": "OBJETIVO PERDIDO", "en": "TARGET LOST"},
        "rescan_button": {"es": "Volver a escanear", "en": "Re-scan"},
        "rescan_scanning_label": {"es": "Escaneando…", "en": "Scanning…"},
        "rescan_heading": {"es": "Dispositivos encontrados",
                            "en": "Devices found"},
        "rescan_pick_button": {"es": "Usar este dispositivo",
                                "en": "Use this device"},
        "rescan_no_candidates": {
            "es": "No se encontró ningún dispositivo. Inténtalo de nuevo.",
            "en": "No devices found. Try again."},
        "no_name_placeholder": {"es": "(sin nombre)", "en": "(no name)"},
        "reacquire_heading": {"es": "Posible coincidencia para tu objetivo",
                               "en": "Possible match for your target"},
        "reacquire_confirm_button": {
            "es": "Sí, es este — cambiar objetivo",
            "en": "Yes, this one — switch target"},
        "reacquire_dismiss_button": {"es": "No es este", "en": "Not this one"},
        "reacquire_none_label": {
            "es": "No se encontró ninguna coincidencia probable cerca.",
            "en": "No likely match found nearby."},
        "reacquire_confidence_likely": {"es": "coincidencia probable",
                                         "en": "likely match"},
        "reacquire_confidence_possible": {"es": "coincidencia posible",
                                           "en": "possible match"},
    },
    "sound": {
        # Zone labels ONLY for the cases sound_cue() cannot borrow a label
        # from elsewhere: "waiting" (no band exists yet) and the three
        # golden-range zones (band_key is suppressed to NOT_APPLICABLE for
        # link mode -- see hunt.py). dBm zones reuse `band_label`, already
        # resolved in this same payload, on purpose -- see build_payload.
        "waiting": {"es": "esperando datos…", "en": "waiting for data…"},
        "golden_far": {"es": "lejos (por debajo del rango dorado)",
                       "en": "far (below the golden range)"},
        "golden_inside": {"es": "dentro del rango dorado — ambiguo",
                           "en": "inside the golden range — ambiguous"},
        "golden_close": {"es": "muy cerca (por encima del rango dorado)",
                          "en": "very close (above the golden range)"},
    },
    "trend": {
        "warmer": {"es": "más caliente", "en": "warmer"},
        "colder": {"es": "más frío", "en": "colder"},
        "steady": {"es": "estable", "en": "steady"},
        "unknown": {"es": "sin datos suficientes", "en": "not enough data"},
    },
    "source": {
        "advert": {"es": "anuncio BLE", "en": "advert"},
        "link": {"es": "enlace conectado", "en": "link"},
        "bredr": {"es": "descubrimiento BR/EDR", "en": "BR/EDR discovery"},
        "sim": {"es": "simulación", "en": "sim"},
        "replay": {"es": "reproducción", "en": "replay"},
    },
    "units": {
        # Shown next to the big readout AND used to decide whether the
        # band/bar/calibration panel (all dBm-calibrated) may run at all --
        # see build_payload's "is_link_units"/"show_calibration" and
        # hunt.py's units-aware branch. golden_range_db is link mode ONLY;
        # bredr mode's readings carry units="dbm" like every other source
        # and get the ordinary dBm treatment.
        "golden_range_note": {
            "es": "Δ dB respecto al Rango Dorado de Recepción (Golden Receive "
                  "Power Range) del enlace clásico — NO es dBm. 0 significa "
                  "que la señal está DENTRO del rango; negativo, por debajo "
                  "(más lejos); positivo, por encima (más cerca). Hay una "
                  "zona muerta ancha alrededor de 0 — esto NO es una regla "
                  "de distancia.",
            "en": "Δ dB relative to the classic link's Golden Receive Power "
                  "Range — NOT dBm. 0 means the signal is INSIDE the range; "
                  "negative is below it (farther); positive is above it "
                  "(closer). There is a wide dead zone around 0 — this is "
                  "NOT a distance ruler.",
        },
    },
    "calibration": {
        "heading": {"es": "Calibración", "en": "Calibration"},
        "distance_label": {"es": "Distancia real (m)",
                            "en": "True distance (m)"},
        "record_button": {"es": "Registrar punto", "en": "Record point"},
        "reset_button": {"es": "Reiniciar", "en": "Reset"},
        "points_label": {"es": "puntos registrados", "en": "points recorded"},
        "fit_a": {"es": "A (RSSI a 1 m)", "en": "A (RSSI at 1 m)"},
        "fit_n": {"es": "n (exponente de pérdida)",
                  "en": "n (path-loss exponent)"},
        "fit_sigma": {"es": "σ (sombreado)", "en": "σ (shadowing)"},
        "fit_r2": {"es": "bondad de ajuste R²", "en": "goodness of fit R²"},
        "range_ratio_label": {"es": "razón de rango (90%)",
                               "en": "range ratio (90%)"},
        # Verified 2026-08-06, https://www.bluetooth.com/blog/proximity-and-rssi/
        # -- the English is the SIG's own wording verbatim; the Spanish is a
        # translation, labelled as such rather than passed off as a second
        # primary quote.
        "quote": {
            "es": "«Evita usar el valor absoluto del RSSI — usa la tendencia» "
                  "— Bluetooth SIG (traducido)",
            "en": "“Avoid using the absolute value of the RSSI — use "
                  "the trend instead.” — Bluetooth SIG",
        },
        "core_spec_note": {
            "es": "El propio Core Spec sólo promete ±6 dB de exactitud en "
                  "HCI_Read_RSSI, antes de cualquier efecto de propagación.",
            "en": "The Core Spec itself only promises ±6 dB accuracy on "
                  "HCI_Read_RSSI, before any propagation effect.",
        },
        "need_more": {
            "es": "Se necesitan al menos {min} puntos para ajustar; hay {n}. "
                  "Camina a una nueva distancia conocida y registra otro punto.",
            "en": "At least {min} points are needed to fit; {n} recorded so "
                  "far. Walk to a new known distance and record another point.",
        },
        "meaning": {
            "es": "Una lectura de {rssi} dBm es consistente con cualquier "
                  "distancia entre {lo:.1f} m y {hi:.1f} m — esa es la razón "
                  "de rango de tu propio ajuste, no un error del medidor.",
            "en": "A {rssi} dBm reading is consistent with anything from "
                  "{lo:.1f} m to {hi:.1f} m — that is your own fit's "
                  "range ratio, not a meter error.",
        },
    },
    "templates": {
        "counters_line": {
            "en": "1 block = 1 measurement · {n_last_min} last min · "
                  "{n_total} total · peak/min {peak} · via {source} "
                  "· refreshed {age} ago",
            "es": "1 bloque = 1 medición · {n_last_min} último min · "
                  "{n_total} total · pico/min {peak} · vía {source} "
                  "· actualizado hace {age}",
        },
        "stale_message": {
            "en": "no packets for {age} — this is SILENCE, not that the "
                  "device is gone",
            "es": "sin paquetes hace {age} — esto es SILENCIO, no que el "
                  "dispositivo no exista",
        },
        "rescan_candidate_line": {
            "en": "{addr} · {name} · {rssi} dBm · {kind} · seen {n}×",
            "es": "{addr} · {name} · {rssi} dBm · {kind} · visto {n}×",
        },
        "reacquire_candidate_line": {
            "en": "{addr} · {rssi} dBm · {kind} · {confidence} ({score:.0f}%)",
            "es": "{addr} · {rssi} dBm · {kind} · {confidence} ({score:.0f}%)",
        },
        "reacquire_reasons_line": {
            "en": "why: {reasons}",
            "es": "por qué: {reasons}",
        },
    },
    # -- vanished: distinct from "stale" -- see hunt.HuntSnapshot.vanished's
    # own docstring for the state this reports. Which of the two messages
    # below is shown depends on whether the target's OWN address was ever
    # empirically observed (via BlueZ's AddressType flag, scan.classify_address)
    # to be a private one -- never guessed from the address bytes alone.
    "vanished": {
        "message_private": {
            "es": "El objetivo dejó de anunciarse. Puede que su dirección "
                  "privada haya rotado — Bluetooth la cambia cada 8-15 "
                  "minutos para que no se pueda rastrear. Vuelve a escanear "
                  "para recuperarlo.",
            "en": "Target stopped advertising. Its private address may "
                  "have rotated — Bluetooth changes it every 8-15 minutes "
                  "so devices cannot be tracked. Re-scan to pick it up "
                  "again.",
        },
        "message_generic": {
            "es": "El objetivo dejó de anunciarse. Es más probable que "
                  "haya salido de rango o que se le haya apagado el "
                  "Bluetooth. Vuelve a escanear para buscarlo de nuevo.",
            "en": "Target stopped advertising. It more likely went out of "
                  "range or had Bluetooth turned off. Re-scan to look for "
                  "it again.",
        },
    },
    # -- BLE random-address subtype labels (scan.classify_address's own
    # four return values), shown next to rescan/reacquire candidates so a
    # student can see WHY a device is a plausible (private) or implausible
    # (public/static) rotation candidate.
    "addr_kind": {
        "public": {"es": "pública", "en": "public"},
        "random-static": {"es": "aleatoria estática", "en": "random-static"},
        "rpa": {"es": "privada resoluble (RPA)",
                "en": "resolvable private (RPA)"},
        "nrpa": {"es": "privada no resoluble (NRPA)",
                 "en": "non-resolvable private (NRPA)"},
    },
    # -- scan.score_reacquire's reason KEYS, resolved to bilingual text
    # here and ONLY here (scan.py stays presentation-free, same split
    # signal.py's Trend enum already uses -- see ReacquireCandidate's own
    # docstring).
    "reacquire_reasons": {
        "name": {"es": "mismo nombre anunciado", "en": "same advertised name"},
        "manufacturer_id": {"es": "mismo ID de fabricante",
                             "en": "same manufacturer ID"},
        "service_uuid": {"es": "mismos UUID de servicio",
                          "en": "same service UUID(s)"},
        "payload_shape": {"es": "forma de payload similar",
                           "en": "similar payload shape"},
        "tx_power": {"es": "misma potencia de transmisión (tx_power)",
                     "en": "same tx power"},
        "rssi_continuity": {"es": "continuidad de RSSI",
                             "en": "RSSI continuity"},
        "time_proximity": {"es": "apareció cerca de cuando el objetivo se "
                                  "perdió",
                            "en": "appeared close to when the target "
                                  "vanished"},
    },
}

_UI_STRINGS = LIVE_STRINGS["ui"]
_TREND_STRINGS = LIVE_STRINGS["trend"]
_SOURCE_STRINGS = LIVE_STRINGS["source"]
_UNITS_STRINGS = LIVE_STRINGS["units"]
_SOUND_STRINGS = LIVE_STRINGS["sound"]
_CALIB_STRINGS = LIVE_STRINGS["calibration"]
_COUNTERS_TEMPLATE = LIVE_STRINGS["templates"]["counters_line"]
_STALE_TEMPLATE = LIVE_STRINGS["templates"]["stale_message"]
_RESCAN_LINE_TEMPLATE = LIVE_STRINGS["templates"]["rescan_candidate_line"]
_REACQUIRE_LINE_TEMPLATE = LIVE_STRINGS["templates"]["reacquire_candidate_line"]
_REACQUIRE_REASONS_LINE_TEMPLATE = LIVE_STRINGS["templates"]["reacquire_reasons_line"]
_VANISHED_STRINGS = LIVE_STRINGS["vanished"]
_ADDR_KIND_STRINGS = LIVE_STRINGS["addr_kind"]
_REACQUIRE_REASON_STRINGS = LIVE_STRINGS["reacquire_reasons"]

#: Trend glyphs. Not language text (a triangle means the same thing in both
#: languages) so this deliberately lives outside LIVE_STRINGS and outside
#: :func:`bilingual_gaps`'s walk.
_TREND_ARROW = {"warmer": "▲", "colder": "▼",
                 "steady": "■", "unknown": "?"}

#: Static labels the page's `data-ui="<key>"` attributes look up directly,
#: as opposed to the calibration table's templated entries ("need_more",
#: "meaning"), which need runtime numbers substituted before they mean
#: anything -- see :func:`calibration_payload`.
_CALIB_STATIC_KEYS = (
    "heading", "distance_label", "record_button", "reset_button",
    "points_label", "fit_a", "fit_n", "fit_sigma", "fit_r2",
    "range_ratio_label", "quote", "core_spec_note",
)


def _redact_addr_tail(addr: str) -> str:
    """Keep only the last two octets: same rule as ``scan.redact_addr`` /
    ``hci._redact`` / ``bredr.py``'s reuse of ``scan.redact_addr`` --
    duplicated here in miniature rather than importing ``scan.py`` at module
    level, matching this module's own "heavy/radio imports stay local"
    discipline (every ``_run_*`` method below already imports its own
    source module locally). Used ONLY to compute what a real source's
    redacted ``Reading.addr`` will look like, so :class:`LiveSession` can
    hand :class:`~blelib.hunt.HuntEngine` a target filter that actually
    matches (see ``LiveSession.__init__``)."""
    parts = addr.split(":")
    if len(parts) != 6:
        return addr
    return "XX:XX:XX:XX:" + ":".join(parts[4:])


def _text(entry: dict[str, str], lang: str) -> str:
    """One bilingual leaf, resolved to `lang` with an English fallback."""
    return entry.get(lang, entry.get("en", ""))


def _bilingual_gaps_of(table: dict, prefix: str) -> list[str]:
    """Recursive walker behind :func:`bilingual_gaps`.

    Factored out from the no-argument public function so a test can also
    run it against a deliberately broken table and check the walker itself
    finds the gap, not just that the real table happens to be complete.
    """
    gaps: list[str] = []
    for key, value in table.items():
        path = f"{prefix}.{key}"
        if isinstance(value, dict) and set(value.keys()) <= {"es", "en"}:
            # A leaf: exactly the {"es": ..., "en": ...} shape (or a subset
            # of it, which IS the gap -- a table missing "es" entirely still
            # has to be walked as a leaf, not silently skipped as "not a
            # bilingual entry"). LIVE_STRINGS' vocabulary never uses "es" or
            # "en" as a subsection name, so this check is unambiguous.
            for lang in ("es", "en"):
                text = value.get(lang)
                if not isinstance(text, str) or not text.strip():
                    gaps.append(f"{path}.{lang}")
        elif isinstance(value, dict):
            gaps.extend(_bilingual_gaps_of(value, path))
        else:
            gaps.append(f"{path} (not a bilingual table)")
    return gaps


def bilingual_gaps() -> list[str]:
    """Every string in :data:`LIVE_STRINGS` missing, or blank in, 'es' or 'en'.

    Dotted paths, e.g. ``"LIVE_STRINGS.calibration.record_button.es"``.
    Empty list means complete. Walks the dict itself rather than a
    hand-maintained checklist -- see the module docstring and SPEC section
    9 -- so this is what ``validate.py`` gates the build on (mirrors the
    Heritage platform's own ``collect_bilingual_gaps``) and what
    ``test_weblive.py`` asserts against directly. A present-but-blank string
    is reported too: a blank Spanish label renders as a silent empty tag,
    which is worse than a crash because nobody notices it.
    """
    return _bilingual_gaps_of(LIVE_STRINGS, "LIVE_STRINGS")


def _ui_table(lang: str) -> dict[str, str]:
    """Every STATIC label the page needs, already resolved to `lang`.

    Flat and prefixed (``calib_*``) rather than nested, because the page's
    `data-ui="<key>"` attributes do one flat lookup per element -- nesting
    here would just move the flattening into JavaScript, which is exactly
    the rendering logic SPEC 5.9 says belongs in Python.
    """
    out = {k: _text(v, lang) for k, v in _UI_STRINGS.items()}
    for key in _CALIB_STATIC_KEYS:
        out[f"calib_{key}"] = _text(_CALIB_STRINGS[key], lang)
    return out


def _format_counters(snap: HuntSnapshot, lang: str) -> str:
    """The honest counters line: distinct measurements, not poll count."""
    peak = "—" if snap.peak_1min_dbm is None else f"{snap.peak_1min_dbm:d}"
    age = "—" if snap.age_s is None else f"{snap.age_s:.0f}s"
    source_label = _text(
        _SOURCE_STRINGS.get(snap.source, {"es": snap.source, "en": snap.source}),
        lang)
    tmpl = _COUNTERS_TEMPLATE.get(lang, _COUNTERS_TEMPLATE["en"])
    return tmpl.format(n_last_min=snap.n_last_min, n_total=snap.n_total,
                        peak=peak, source=source_label, age=age)


def _format_stale(snap: HuntSnapshot, lang: str) -> str:
    """Silence reads as 'no signal', never as 'no device' -- see SPEC's
    front-end brief. The wording says SILENCE explicitly rather than
    leaving a frozen number on screen with no explanation."""
    age = "?" if snap.age_s is None else f"{snap.age_s:.0f}s"
    tmpl = _STALE_TEMPLATE.get(lang, _STALE_TEMPLATE["en"])
    return tmpl.format(age=age)


def _format_mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60}:{total % 60:02d}"


# --------------------------------------------------------- vanished / re-acquire
def _format_vanished(is_private_addr: bool, lang: str) -> str:
    """Which of the two vanished messages to show -- see LIVE_STRINGS
    "vanished"'s own comment for why the choice is never guessed from the
    address bytes: `is_private_addr` must come from something that actually
    OBSERVED the target's BlueZ AddressType (scan.AdvertSource.last_fingerprint
    .addr_kind), not from classify_address applied blind to an address of
    unknown provenance."""
    key = "message_private" if is_private_addr else "message_generic"
    return _text(_VANISHED_STRINGS[key], lang)


def _addr_kind_label(addr_kind: str, lang: str) -> str:
    return _text(_ADDR_KIND_STRINGS.get(addr_kind, {"es": addr_kind, "en": addr_kind}),
                lang)


def _resolve_rescan_candidates(candidates: Sequence[dict], lang: str) -> list[dict]:
    """Turn the raw (language-agnostic) candidates :class:`LiveSession`
    stashed from its last ``"rescan"`` control action into what the page
    renders -- kept as a free function (not a method) so it is testable
    with a plain hand-built list, same testing-seam discipline as
    :func:`build_payload`. Resolved fresh on every call (i.e. every SSE
    tick) rather than once at scan time, so a language toggle re-translates
    an already-found candidate list instead of leaving it stuck in
    whichever language was active when the scan ran.
    """
    tmpl = _RESCAN_LINE_TEMPLATE.get(lang, _RESCAN_LINE_TEMPLATE["en"])
    no_name = _text(_UI_STRINGS["no_name_placeholder"], lang)
    out = []
    for c in candidates:
        line = tmpl.format(addr=c["addr_display"], name=c["name"] or no_name,
                           rssi=c["rssi_dbm"], kind=_addr_kind_label(c["addr_kind"], lang),
                           n=c["n_seen"])
        out.append({"id": c["id"], "line": line})
    return out


def _resolve_reacquire_candidates(candidates: Sequence[dict], lang: str) -> list[dict]:
    """Same purpose/discipline as :func:`_resolve_rescan_candidates`, for
    the auto-reacquire suggestion list -- additionally resolves each
    candidate's ``reasons`` (scan.score_reacquire's reason KEYS) to a
    bilingual "why" line, so the page can show its work instead of a bare
    percentage (never present a guess as a certainty)."""
    tmpl = _REACQUIRE_LINE_TEMPLATE.get(lang, _REACQUIRE_LINE_TEMPLATE["en"])
    reasons_tmpl = _REACQUIRE_REASONS_LINE_TEMPLATE.get(
        lang, _REACQUIRE_REASONS_LINE_TEMPLATE["en"])
    out = []
    for c in candidates:
        confidence_key = f"reacquire_confidence_{c['confidence']}"
        confidence_label = _text(
            _UI_STRINGS.get(confidence_key, {"es": c["confidence"], "en": c["confidence"]}),
            lang)
        line = tmpl.format(addr=c["addr_display"], rssi=c["rssi_dbm"],
                           kind=_addr_kind_label(c["addr_kind"], lang),
                           confidence=confidence_label, score=c["score"] * 100)
        reason_labels = [_text(_REACQUIRE_REASON_STRINGS.get(r, {"es": r, "en": r}), lang)
                         for r in c["reasons"]]
        reasons_line = reasons_tmpl.format(reasons=", ".join(reason_labels))
        out.append({"id": c["id"], "line": line, "reasons_line": reasons_line,
                    "score": c["score"]})
    return out


# --------------------------------------------------------------- calibration
def add_calibration_point(points: Sequence[tuple[float, int]],
                           distance_m: float, rssi_dbm: int,
                           ) -> list[tuple[float, int]]:
    """Append one (distance, RSSI) pair. Pure: returns a NEW list.

    The caller (:class:`LiveSession`) is the only place calibration state is
    actually stored -- same split as ``HuntEngine.feed`` vs.
    ``HuntEngine.snapshot``, kept here so the accumulate-and-fit path is
    testable with plain lists and no session, no Flask, no socket.
    """
    d = float(distance_m)
    if not (d > 0):
        raise ValueError(
            "distance_m must be a positive number of metres / "
            "distance_m debe ser un número positivo de metros")
    return [*points, (d, int(rssi_dbm))]


def calibration_payload(points: Sequence[tuple[float, int]],
                         lang: str = "en") -> dict:
    """The calibration panel's content: fit quality, range_ratio, and its
    plain-language meaning -- the lab's honest finding (SPEC section 1),
    made visible while the student is still standing there.

    Below :data:`CALIB_MIN_POINTS` there is no honest fit yet (three points
    is `pathloss.fit_log_distance`'s own floor -- two points would report a
    perfect, and false, sigma of 0): report the count and how many more are
    needed instead of a division-by-zero-flavoured guess.
    """
    lang = lang if lang in ("es", "en") else "en"
    n = len(points)
    out: dict = {
        "n_points": n,
        "min_points": CALIB_MIN_POINTS,
        "ready": n >= CALIB_MIN_POINTS,
        "points": [{"distance_m": round(float(d), 3), "rssi_dbm": int(r)}
                   for d, r in points],
        "fit": None,
        "example": None,
    }
    if n < CALIB_MIN_POINTS:
        out["message"] = _text(_CALIB_STRINGS["need_more"], lang).format(
            min=CALIB_MIN_POINTS, n=n)
        return out

    distances = [d for d, _ in points]
    rssis = [r for _, r in points]
    fit: PathLossFit = fit_log_distance(distances, rssis)
    ratio = range_ratio(fit.sigma_db, fit.n)
    # Anchored on the MOST RECENT reading, not an arbitrary one: "your
    # meter currently says X, and that means..." is the sentence that lands
    # while the student is still standing at the spot they just measured.
    last_rssi = rssis[-1]
    d_lo, d_hi = distance_interval(last_rssi, fit)
    out["fit"] = {"a_dbm": round(fit.a_dbm, 2), "n": round(fit.n, 3),
                  "sigma_db": round(fit.sigma_db, 2),
                  "r_squared": round(fit.r_squared, 4),
                  "n_points": fit.n_points, "range_ratio": round(ratio, 2)}
    out["example"] = {"rssi_dbm": last_rssi, "d_lo_m": round(d_lo, 2),
                       "d_hi_m": round(d_hi, 2)}
    out["message"] = _text(_CALIB_STRINGS["meaning"], lang).format(
        rssi=last_rssi, lo=d_lo, hi=d_hi)
    return out


# --------------------------------------------------------------- payload
def _short_uuid(u: str) -> str:
    """The 16-bit short form of a Bluetooth UUID, for display.

    Accepts both what the CLI stores ("fef3") and the normalised 128-bit form,
    because the target line is fed from whichever the caller happened to pass.
    """
    s = str(u).strip().lower().removeprefix("0x")
    if len(s) >= 8 and "-" in s:
        return s[4:8]
    return s


def _format_target(identity, lang: str) -> dict:
    """The "who am I reading?" line.

    Kept separate and pure so it can be tested without a radio, and so the
    honest cases are explicit: no target at all (sim mode), a target BlueZ has
    never heard of (a bare advertising address), and a paired/connected device
    whose real name we can show.
    """
    ui = LIVE_STRINGS["ui"]

    def s(key: str) -> str:
        return ui[key][lang]

    if identity is None or not getattr(identity, "addr", ""):
        return {"label": s("target_label"), "display": s("target_none"),
                "addr": "", "flags": [], "known": False}

    flags: list[str] = []
    if getattr(identity, "connected", False):
        flags.append(s("target_connected"))
    if getattr(identity, "paired", False):
        flags.append(s("target_paired"))
    if not getattr(identity, "known_to_bluez", False):
        flags.append(s("target_unknown_device"))

    name = getattr(identity, "name", "") or ""
    addr = getattr(identity, "addr", "")
    return {
        "label": s("target_label"),
        # Show BOTH when we have a name: the name is what a human recognises,
        # the address is what they typed and what identifies it unambiguously.
        "display": f"{name} · {addr}" if name else addr,
        "addr": addr,
        "name": name,
        "flags": flags,
        "known": bool(getattr(identity, "known_to_bluez", False)),
    }


def build_payload(snap: HuntSnapshot, lang: str = "en", *,
                   mode: str = "", calibration: dict | None = None,
                   target_addr_kind: str | None = None,
                   target_identity=None) -> dict:
    """Turn one :class:`HuntSnapshot` into the JSON the page renders.

    Pure and side-effect free: every user-visible string is resolved to the
    requested language HERE, in Python, so the page only ever displays what
    this function handed it -- no template logic in JavaScript, no
    hardcoded English, and the whole payload is exercised by
    ``test_weblive.py`` with no socket and no ``HuntEngine`` at all (a
    ``HuntSnapshot`` is a plain frozen dataclass, easy to hand-build).

    `mode`, `calibration` and `target_addr_kind` are additive keyword-only
    extensions past the SPEC-bound `(snap, lang)` signature -- positional
    callers, and the bilingual-completeness tests, still work with just the
    first two args. `target_addr_kind` is `None` unless something has
    actually OBSERVED the current target's BlueZ AddressType (only
    `--mode live` ever populates it, via `AdvertSource.last_fingerprint`) --
    it decides, when `snap.vanished`, whether the "may have rotated"
    message is honest to show (see `_format_vanished`).
    """
    lang = lang if lang in ("es", "en") else "en"
    trend_key = snap.trend if snap.trend in _TREND_STRINGS else "unknown"
    band_label = snap.band_es if lang == "es" else snap.band_en
    stale = bool(snap.stale)
    vanished = bool(snap.vanished)
    vanished_message = (
        _format_vanished(target_addr_kind in ("rpa", "nrpa"), lang)
        if vanished else "")

    # units-aware display split: link mode's Golden Receive Power Range delta
    # (reading.UNIT_GOLDEN_RANGE_DB) is NOT dBm and must never be shown, or
    # treated, as if it were -- see hci.py's module docstring and
    # hunt.py's units-aware branch, which already suppressed band_key/
    # fraction upstream. This is the second half of that same discipline:
    # the calibration/path-loss panel (which assumes real dBm end to end)
    # is switched off entirely for these units, not just relabelled.
    is_link_units = snap.units != UNIT_DBM
    golden_range_note = (
        _text(_UNITS_STRINGS["golden_range_note"], lang) if is_link_units else "")

    # Sound cue: dBm zones reuse band_label (already resolved above); the
    # "waiting" and golden-range zones have no band_label to borrow (band
    # is suppressed for link mode, and there is no band at all before the
    # first reading), so those resolve from LIVE_STRINGS["sound"] instead.
    cue = sound_cue(snap)
    zone_label = (_text(_SOUND_STRINGS[cue["zone"]], lang)
                 if cue["zone"] in _SOUND_STRINGS else band_label)
    sound_payload = {**cue, "zone_label": zone_label}

    return {
        "lang": lang,
        "mode": mode,
        "rssi_dbm": None if snap.rssi_dbm is None else round(float(snap.rssi_dbm), 1),
        "raw_dbm": snap.raw_dbm,
        "units": snap.units,
        "unit_symbol": "dBm" if snap.units == UNIT_DBM else "dB",
        "is_link_units": is_link_units,
        "golden_range_note": golden_range_note,
        "show_calibration": not is_link_units,
        "sound": sound_payload,
        "band_key": snap.band_key,
        "band_label": band_label,
        "fraction": round(float(snap.fraction), 4),
        "trend": snap.trend,
        "trend_word": _text(_TREND_STRINGS.get(trend_key, _TREND_STRINGS["unknown"]), lang),
        "trend_arrow": _TREND_ARROW.get(trend_key, _TREND_ARROW["unknown"]),
        "stale": stale,
        "age_s": None if snap.age_s is None else round(float(snap.age_s), 2),
        "stale_message": _format_stale(snap, lang) if stale else "",
        "vanished": vanished,
        "vanished_message": vanished_message,
        "peak_1min_dbm": snap.peak_1min_dbm,
        "n_total": int(snap.n_total),
        "n_last_min": int(snap.n_last_min),
        "source": snap.source,
        "source_label": _text(
            _SOURCE_STRINGS.get(snap.source, {"es": snap.source, "en": snap.source}),
            lang),
        "spark": [int(v) for v in snap.spark],
        "spark_levels": SPARK_N_LEVELS,
        "elapsed_s": round(float(snap.elapsed_s), 2),
        "elapsed_label": _format_mmss(snap.elapsed_s),
        "counters_line": _format_counters(snap, lang),
        "target": _format_target(target_identity, lang),
        "ui": _ui_table(lang),
        "calibration": calibration if calibration is not None else calibration_payload([], lang),
    }


# --------------------------------------------------------------- controls
#: Every action :func:`handle_control` can stage on a HuntConfig, and its
#: validator. HuntConfig is a plain (non-frozen) dataclass -- see SPEC
#: 5.5 -- so staging means mutating the SAME object the caller handed in;
#: :class:`LiveSession` relies on that to make a control change land on the
#: next `snapshot()` with no extra plumbing.
def _positive_float(value: object) -> float:
    f = float(value)  # type: ignore[arg-type]
    if not (f > 0):
        raise ValueError("must be a positive number / debe ser un número positivo")
    return f


def _nonneg_int(value: object) -> int:
    f = float(value)  # type: ignore[arg-type]
    if f < 0:
        raise ValueError("must be zero or positive / debe ser cero o positivo")
    return int(round(f))


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on", "si", "sí")
    return bool(value)


_CFG_FIELD_VALIDATORS: dict[str, Callable[[object], object]] = {
    "live_window_s": _positive_float,
    "trend_window_s": _positive_float,
    "trend_threshold_db": _nonneg_int,
    "stale_after_s": _positive_float,
    "redact": _as_bool,
}


def handle_control(action: dict, cfg: HuntConfig) -> dict:
    """Apply one control action to `cfg`. Pure with respect to the network
    and thread layers: no Flask, no socket, no HuntEngine reference -- a
    test builds a bare ``HuntConfig()`` and asserts the field it expects
    changed, exactly like SPEC 5.9's testing-seam split demands.

    ``"reset"`` is a sentinel, not a cfg field: it does not touch `cfg` at
    all, because resetting the reading history needs the HuntEngine, which
    this function deliberately does not have. :meth:`LiveSession.control`
    is where that sentinel gets acted on.
    """
    action = action or {}
    name = str(action.get("action", ""))
    if name == "reset":
        return {"ok": True, "reset": True}
    validator = _CFG_FIELD_VALIDATORS.get(name)
    if validator is None:
        return {"ok": False, "error": f"unknown action {name!r} / "
                                       f"acción desconocida {name!r}"}
    try:
        value = validator(action.get("value"))
    except (TypeError, ValueError) as exc:
        return {"ok": False, "error": f"{name}: {exc}"}
    setattr(cfg, name, value)
    return {"ok": True, "field": name, "value": value}


# --------------------------------------------------------------- sim feed
#: One synthetic "leg" of continuous sim mode: a walk to a new random
#: target distance. Long enough that a student sees a full proximity-band
#: transition, short enough that a fresh target arrives before the hunt
#: gets boring -- and, unlike a single fixed demo track, this never runs
#: out, which matters because a live meter session can run far longer than
#: any one hand-written walk.
_SIM_LEG_S = 45.0
_SIM_LEG_D_MIN_M = 0.3
_SIM_LEG_D_MAX_M = 15.0


def _iter_sim_readings(cfg: object, seed: int = 0,
                        leg_s: float = _SIM_LEG_S,
                        d_min: float = _SIM_LEG_D_MIN_M,
                        d_max: float = _SIM_LEG_D_MAX_M) -> Iterator[Reading]:
    """Endless synthetic hunt: one random-walk leg after another.

    ``sim.simulate()`` only knows how to render ONE finite track (SPEC 5.4
    is explicit that it takes a fixed ``(track_t, track_d)`` pair). This
    stitches legs together forever: each leg starts where the last one
    ended (distance-continuous, so there is no visible jump at the seam)
    and picks a new random target, deterministic given the session's own
    seed so a bug report ("seed 7 breaks at minute 3") is reproducible.
    Readings carry the module's own relative time, offset by the
    cumulative leg time -- :meth:`LiveSession._run_feed_loop` re-anchors
    them to the real clock at delivery time, this generator only has to
    keep them internally consistent leg to leg.
    """
    from . import sim as _sim  # local: keeps this module importable even
                                # before sim.py exists / for callers that
                                # never touch continuous sim mode.

    rng = np.random.default_rng(seed)
    t_offset = 0.0
    d_start = float(rng.uniform(d_min, d_max))
    leg_index = 0
    while True:
        d_end = float(rng.uniform(d_min, d_max))
        waypoints = (_sim.Waypoint(t=0.0, distance_m=d_start),
                     _sim.Waypoint(t=leg_s, distance_m=d_end))
        track_t, track_d = _sim.walk(waypoints, dt=0.1)
        result = _sim.simulate(track_t, track_d, cfg, seed=seed + leg_index)
        for r in result.readings:
            yield Reading(rssi_dbm=r.rssi_dbm, t=r.t + t_offset,
                          source=r.source, addr=r.addr)
        t_offset += leg_s
        d_start = d_end
        leg_index += 1


# --------------------------------------------------------------- session
_VALID_MODES = ("sim", "replay", "live", "link", "bredr")


@dataclass(frozen=True)
class _IdentityCache:
    """One resolved target identity, keyed by the address it was resolved for."""

    addr_key: str
    identity: object | None


class LiveSession:
    """Owns one :class:`HuntEngine`, the feeding thread for whichever mode
    is active, the calibration accumulator, and the display language.

    Mirrors lab-01's ``LiveSession`` threading discipline: :meth:`start`
    spawns exactly one daemon feeding thread, :meth:`stop` clears a stop
    event and joins it with a bounded timeout so Ctrl-C in ``run.py`` never
    hangs. The thread is the ONLY thing here that is not pure --
    :meth:`snapshot`, :meth:`feed`, and every string-composing function in
    this module stay testable by hand-building a ``HuntEngine`` and feeding
    it Readings directly, with no thread, no clock but the one passed in,
    and (for "sim" / "replay") no radio at all.
    """

    def __init__(self, *, cfg: HuntConfig, mode: str,
                 readings: list[Reading] | None = None,
                 target: str = "", source_label: str = "sim",
                 speed: float = 1.0, lang: str = "en",
                 sim_config: object = None, seed: int = 0,
                 service: str | None = None,
                 now: Callable[[], float] = time.monotonic) -> None:
        if mode not in _VALID_MODES:
            raise ValueError(
                f"unknown mode {mode!r}, expected one of {_VALID_MODES} / "
                f"modo desconocido {mode!r}, se esperaba uno de {_VALID_MODES}")
        self.mode = mode
        self._cfg = cfg
        # HuntEngine's OWN target filter must match what Reading.addr will
        # actually contain, not the raw address the caller passed in. For
        # "live"/"link"/"bredr", AdvertSource/LinkSource/BredrSource already
        # filter to exactly this one address themselves, on the RAW address,
        # before a Reading is ever created -- only the OUTPUT Reading.addr
        # gets redacted afterward, by default (see each class's own
        # docstring). Handing the raw target straight to HuntEngine while
        # cfg.redact is True (the default) made every single reading fail
        # `addr != target` and get silently dropped -- found 2026-08-06
        # while proving --mode link end-to-end against real hardware.
        # "sim"/"replay" never reach this branch in practice (run.py never
        # forwards --target to _serve() for those modes), so this cannot
        # change their behaviour.
        engine_target = _redact_addr_tail(target) if target and cfg.redact else target
        self.engine = HuntEngine(cfg, target=engine_target, maxlen=4096)
        self._target = target
        #: Hunt by advertised service UUID instead of by address when set.
        #: A service UUID says what a device IS, so it survives the private-
        #: address rotation that otherwise ends an address-targeted hunt.
        self._service = service
        self.source_label = source_label
        self.speed = float(speed) if speed and speed > 0 else 1.0
        self.lang = lang if lang in ("es", "en") else "en"
        self._readings = list(readings) if readings is not None else None
        self._sim_config = sim_config
        self._seed = int(seed)
        self._now = now

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self.error: str | None = None
        self._calib_points: list[tuple[float, int]] = []

        # -- vanished / re-scan / re-acquire (SPEC-adjacent fix: "reboot
        # the app to see the change" was an RPA rotation, not a streaming
        # bug -- see the module docstring above and
        # `_maybe_trigger_reacquire`) --------------------------------------
        #: Set once `_run_live`'s pump starts (mode == "live" only) --
        #: `None` for every other mode, which is exactly the guard
        #: `_maybe_trigger_reacquire`/`_target_addr_kind` need: RPA rotation
        #: is a BLE-advertisement-only phenomenon, so "vanished may have
        #: rotated" and auto-reacquire only ever make sense here.
        self._advert_source = None
        self._identity_cache: _IdentityCache | None = None
        #: Opaque id -> raw address, for whichever candidate list is
        #: currently on screen. Two SEPARATE maps (not one shared one) so
        #: clearing one (e.g. a fresh rescan) never invalidates ids the
        #: other panel is still showing.
        self._rescan_candidates: list[dict] = []
        self._rescan_index: dict[str, str] = {}
        self._reacquire_candidates: list[dict] = []
        self._reacquire_index: dict[str, str] = {}
        #: Guards `_maybe_trigger_reacquire` so it computes suggestions
        #: ONCE per vanish episode (not on every ~8Hz SSE tick) -- holds the
        #: target fingerprint's address the suggestions were computed for;
        #: `None` means "not computed for the current episode yet".
        self._reacquire_computed_for: str | None = None
        self._candidate_seq = 0

    @property
    def cfg(self) -> HuntConfig:
        return self._cfg

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop_evt.clear()
            self.error = None
            if self.mode == "live":
                target_fn: Callable[..., None] = self._run_live
                args: tuple = ()
            elif self.mode == "link":
                target_fn = self._run_link
                args = ()
            elif self.mode == "bredr":
                target_fn = self._run_bredr
                args = ()
            else:
                target_fn = self._run_feed_loop
                args = (self._build_sim_or_replay_iter(), self.speed)
            self._running = True
            self._thread = threading.Thread(target=target_fn, args=args,
                                             daemon=True, name="lab06-feed")
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._running = False
            self._stop_evt.set()
            thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=2.0)

    # -- feeding ------------------------------------------------------------
    def feed(self, r: Reading) -> None:
        with self._lock:
            self.engine.feed(r)

    def _build_sim_or_replay_iter(self) -> Iterable[Reading]:
        """"sim" with a given track and "replay" share one playback path:
        both walk a finite, pre-recorded list of Readings on ITS OWN
        timestamps, scaled by `speed`. Only "sim" with no `readings` is
        different -- it has no track to exhaust, by design (SPEC: "a hunt
        meter that stops after 90 seconds is useless")."""
        if self._readings is not None:
            return list(self._readings)
        cfg = self._sim_config
        if cfg is None:
            from . import sim as _sim
            cfg = _sim.SimConfig()
        return _iter_sim_readings(cfg, self._seed)

    def _run_feed_loop(self, reading_iter: Iterable[Reading], speed: float) -> None:
        """Paces a finite OR infinite Reading iterable against the real
        clock. When the iterable ends (replay, or "sim" with a given
        track) this simply returns: no crash, no silent loop-back. The
        engine's own `is_stale()` then does the honest thing on its own --
        once `stale_after_s` passes with nothing new fed in, the display
        goes stale exactly as it would for a real device that stopped
        advertising. Nothing special has to happen here for that.
        """
        prev_t: float | None = None
        try:
            for r in reading_iter:
                if self._stop_evt.is_set():
                    return
                if prev_t is not None:
                    dt = max(0.0, (r.t - prev_t) / speed)
                    if self._stop_evt.wait(dt):
                        return
                prev_t = r.t
                # Re-anchored to the real clock at delivery time: a
                # recording's timestamps are relative to when IT was made,
                # but HuntEngine's live/trend windows and is_stale() are
                # only meaningful measured against the SAME clock
                # snapshot() reads -- self._now, not the file's own t=0.
                self.feed(Reading(rssi_dbm=r.rssi_dbm, t=self._now(),
                                  source=r.source, addr=r.addr))
        except Exception as exc:                      # keep the server alive
            self.error = f"{type(exc).__name__}: {exc}"

    def _run_live(self) -> None:
        """Real BLE, in its own thread with its own asyncio loop.

        The import is local and guarded on purpose: `blelib.scan` pulls in
        `bleak`, which is not installed on every machine this module needs
        to import cleanly on (tests, `validate.py`, a laptop with no
        Bluetooth stack at all). "live" mode is the only path that ever
        touches it, and only once :meth:`start` actually runs it.
        """
        try:
            import asyncio
            from .scan import AdvertSource
        except Exception as exc:
            self.error = (f"live mode necesita blelib.scan / bleak, no "
                          f"disponible en esta máquina -- live mode needs "
                          f"blelib.scan / bleak, not available on this "
                          f"machine: {type(exc).__name__}: {exc}")
            return

        async def _pump() -> None:
            if self._service:
                from .scan import ServiceAdvertSource
                source = ServiceAdvertSource(self._service,
                                             redact=self._cfg.redact)
            else:
                source = AdvertSource(self._target, redact=self._cfg.redact)
            # Published for `_target_addr_kind`/`_maybe_trigger_reacquire`
            # (running on the SSE/request thread) to read -- same
            # eventually-consistent, no-lock convention as `self.error`
            # (one writer thread, one best-effort reader) and as
            # `AdvertSource.last_fingerprint`'s own docstring.
            self._advert_source = source
            async for r in source.stream():
                if self._stop_evt.is_set():
                    return
                self.feed(r)

        try:
            asyncio.run(_pump())
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _run_link(self) -> None:
        """Connected-link RSSI (blelib.hci) -- golden-range dB, NOT dBm.

        Same import-local, own-event-loop, self.error-not-a-crash discipline
        as :meth:`_run_live`. ``LinkNotConnectedError`` (no active ACL
        connection) and ``LinkDroppedError`` (the connection vanished
        mid-hunt) are both bilingual and actionable -- surfaced via
        ``self.error`` -> :meth:`payload`'s ``session_error`` key, not a
        crash, and not a frozen number pretending nothing happened.
        """
        try:
            import asyncio
            from .hci import LinkDroppedError, LinkNotConnectedError, LinkSource
        except Exception as exc:
            self.error = (f"link mode necesita blelib.hci, no disponible en "
                          f"esta máquina -- link mode needs blelib.hci, not "
                          f"available on this machine: "
                          f"{type(exc).__name__}: {exc}")
            return

        async def _pump() -> None:
            source = LinkSource(self._target, redact=self._cfg.redact)
            async for r in source.stream():
                if self._stop_evt.is_set():
                    return
                self.feed(r)

        try:
            asyncio.run(_pump())
        except (LinkNotConnectedError, LinkDroppedError) as exc:
            self.error = str(exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def _run_bredr(self) -> None:
        """BR/EDR discovery-mode RSSI (blelib.bredr) -- real dBm, slow (~0.1 Hz).

        Same discipline as :meth:`_run_link`. ``NoAdapterError`` and
        ``BredrNotDiscoverableError`` (no RSSI update arrived -- the target
        is not actively discoverable right now) are both surfaced via
        ``self.error`` rather than a crash or a silently-forever-stale
        display with no explanation of why.
        """
        try:
            import asyncio
            from .bredr import BredrNotDiscoverableError, BredrSource
            from .scan import NoAdapterError
        except Exception as exc:
            self.error = (f"bredr mode necesita blelib.bredr / dbus_fast, no "
                          f"disponible en esta máquina -- bredr mode needs "
                          f"blelib.bredr / dbus_fast, not available on this "
                          f"machine: {type(exc).__name__}: {exc}")
            return

        async def _pump() -> None:
            source = BredrSource(self._target, redact=self._cfg.redact)
            async for r in source.stream():
                if self._stop_evt.is_set():
                    return
                self.feed(r)

        try:
            asyncio.run(_pump())
        except (BredrNotDiscoverableError, NoAdapterError) as exc:
            self.error = str(exc)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    # -- reads --------------------------------------------------------------
    def snapshot(self, now: float | None = None) -> HuntSnapshot:
        t = self._now() if now is None else now
        with self._lock:
            return self.engine.snapshot(t)

    def calibration(self) -> dict:
        with self._lock:
            points = list(self._calib_points)
        return calibration_payload(points, self.lang)

    def _target_addr_kind(self) -> str | None:
        """The target's OWN observed BlueZ AddressType, or `None` if it has
        never been seen (or this session's mode never has visibility into
        it -- only "live" does). Feeds `build_payload`'s `target_addr_kind`
        so the vanished message never claims "may have rotated" without
        evidence."""
        source = self._advert_source
        if source is None or source.last_fingerprint is None:
            return None
        return source.last_fingerprint.addr_kind

    def _identity(self):
        """Who this session is pointed at, resolved once and cached.

        Cached because it is a D-Bus round trip and the SSE loop asks eight
        times a second, but invalidated on retarget so a re-acquired device
        shows its own name rather than the previous target's. Never raises:
        `scan.target_identity` already guarantees that, and a meter that
        crashed because it could not look up a name would be a worse bug than
        the missing name this fixes.
        """
        # Fingerprint hunts have no fixed address by design, so report the
        # service being followed plus whichever address currently carries it.
        # Falling through to the address branch here showed "no target --
        # simulation mode" during a working live hunt, which is exactly the
        # kind of "is this thing on?" ambiguity the target line exists to end.
        if self._service:
            src = self._advert_source
            locked = getattr(src, "locked_addr", None) if src else None
            fp = getattr(src, "last_fingerprint", None) if src else None
            from .scan import TargetIdentity, redact_addr
            shown = ""
            if locked:
                shown = redact_addr(locked) if self._cfg.redact else locked
            return TargetIdentity(
                addr=shown or self._service,
                name=(getattr(fp, "local_name", "") or "")
                     or f"service {_short_uuid(self._service)}",
                connected=False, paired=False,
                known_to_bluez=bool(locked))

        target = self._target
        if not target:
            return None
        cached = self._identity_cache
        if cached is not None and cached.addr_key == target:
            return cached.identity
        try:
            from .scan import target_identity
            ident = target_identity(target, redact=self._cfg.redact)
        except Exception:                              # noqa: BLE001
            ident = None
        self._identity_cache = _IdentityCache(addr_key=target, identity=ident)
        return ident

    def _next_candidate_id(self) -> str:
        """Opaque id for one rescan/reacquire candidate -- the click that
        confirms a new target goes through this id, NEVER the (possibly
        redacted, possibly just plain sensitive) address string itself, so
        the redacted-display / raw-retarget split (module docstring, "own
        device only") holds even in the browser's DOM."""
        self._candidate_seq += 1
        return f"c{self._candidate_seq}"

    def _maybe_trigger_reacquire(self, snap: HuntSnapshot) -> None:
        """Edge-triggered: the first time a live-mode hunt's target goes
        VANISHED (not merely stale), rank whatever OTHER devices the same
        background scan has already seen against the target's own
        last-known fingerprint -- no extra scan needed, `AdvertSource` has
        been watching everything in the room the whole time (see its own
        `siblings()` docstring). This is the "try to find the SAME physical
        device under its new address" feature -- it only ever COMPUTES a
        ranked suggestion list; a human still has to click "confirm"
        (`control(action="retarget")`) before the hunt actually switches,
        because a wrong auto-switch would silently corrupt a graded hunt.

        Fires at most once per vanish episode (`_reacquire_computed_for`
        guards against recomputing on every ~8Hz SSE tick); re-arms once
        `retarget`/`reset` clears the guard.
        """
        source = self._advert_source
        if source is None or source.last_fingerprint is None or not snap.vanished:
            return
        fp = source.last_fingerprint
        with self._lock:
            if self._reacquire_computed_for == fp.addr:
                return
            self._reacquire_computed_for = fp.addr
            from . import scan as sc
            ranked = sc.rank_reacquire_candidates(fp, source.siblings())
            index: dict[str, str] = {}
            candidates = []
            for cand in ranked[:5]:
                cid = self._next_candidate_id()
                index[cid] = cand.addr
                candidates.append({
                    "id": cid,
                    "addr_display": (_redact_addr_tail(cand.addr) if self._cfg.redact
                                     else cand.addr),
                    "score": cand.score,
                    "confidence": ("likely" if cand.score >= sc.REACQUIRE_LIKELY_THRESHOLD
                                  else "possible"),
                    "reasons": list(cand.reasons),
                    "rssi_dbm": cand.rssi_dbm,
                    "addr_kind": cand.addr_kind,
                })
            self._reacquire_index = index
            self._reacquire_candidates = candidates

    def payload(self) -> dict:
        """One full `/stream` message: `build_payload` fed with this
        session's live snapshot, language, and calibration state, plus
        `session_error` if the feeding thread hit one (link mode dropped,
        bredr mode's target never went discoverable, ...). Kept OUTSIDE
        `build_payload`'s pure signature deliberately -- `session_error` is
        session/thread state, not something a hand-built HuntSnapshot can
        express, and build_payload's own tests construct snapshots with no
        session at all. `rescan`/`reacquire` are the same kind of
        session-only addition, for the same reason."""
        snap = self.snapshot()
        self._maybe_trigger_reacquire(snap)
        data = build_payload(snap, self.lang, mode=self.mode,
                             calibration=self.calibration(),
                             target_addr_kind=self._target_addr_kind(),
                             target_identity=self._identity())
        if self.error:
            data["session_error"] = self.error
        with self._lock:
            rescan_candidates = list(self._rescan_candidates)
            reacquire_candidates = list(self._reacquire_candidates)
        data["rescan"] = {"candidates": _resolve_rescan_candidates(rescan_candidates, self.lang)}
        data["reacquire"] = {"candidates": _resolve_reacquire_candidates(reacquire_candidates, self.lang)}
        return data

    # -- re-scan / re-target ----------------------------------------------
    def _do_rescan(self, action: dict) -> dict:
        """Fresh discovery scan from INSIDE a running session -- the whole
        point of this action is that the student never has to Ctrl-C and
        relaunch `run.py` just to see what is advertising right now (the
        bug report this feature exists to fix: "I have to reboot the app").

        Blocking, same as `run.py cmd_scan`'s own
        `asyncio.run(sc.discover(...))` -- a POST from the browser waiting
        a few seconds for a real BLE scan is the honest cost of a real
        scan, not a bug; `run.py --mode scan` already accepts the same
        wait for the exact same reason.
        """
        try:
            from . import scan as sc
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        try:
            duration_s = float(action.get("duration_s", 6.0))
        except (TypeError, ValueError):
            duration_s = 6.0
        duration_s = max(1.0, min(duration_s, 30.0))  # sane bounds either way

        import asyncio
        try:
            found = asyncio.run(sc.discover(duration_s, redact=False))
        except (sc.NoAdapterError, ImportError) as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        found = sorted(found, key=lambda d: -d.rssi_dbm)
        with self._lock:
            index: dict[str, str] = {}
            candidates = []
            for d in found:
                cid = self._next_candidate_id()
                index[cid] = d.addr
                candidates.append({
                    "id": cid,
                    "addr_display": _redact_addr_tail(d.addr) if self._cfg.redact else d.addr,
                    "name": d.name,
                    "rssi_dbm": d.rssi_dbm,
                    "addr_kind": d.addr_kind,
                    "n_seen": d.n_seen,
                })
            self._rescan_index = index
            self._rescan_candidates = candidates
        return {"ok": True, "candidates": _resolve_rescan_candidates(candidates, self.lang)}

    def _do_retarget(self, action: dict) -> dict:
        """Confirm ONE candidate (from either an explicit `rescan` or an
        auto-`reacquire` suggestion) as the new target, entirely from the
        browser -- no restart. This is what removes "reboot the app to
        pick up the phone's new rotated address" from the bug report: the
        student clicks a candidate, the SAME running session starts
        hunting it.

        Requires an explicit id from an explicit click -- never switches
        automatically, even for a very high reacquire score, because a
        silent auto-switch could silently corrupt a graded hunt (module
        docstring, "own device only" boundary applies here too: a
        confirmed retarget is the ONE place a raw address, chosen by the
        student, is allowed to become the new filter).
        """
        cid = str(action.get("id", ""))
        with self._lock:
            raw_addr = self._rescan_index.get(cid) or self._reacquire_index.get(cid)
        if not raw_addr:
            return {"ok": False, "error": f"unknown candidate id {cid!r} / "
                                           f"id de candidato desconocido {cid!r}"}

        was_running = self._running
        if was_running:
            self.stop()

        with self._lock:
            self._target = raw_addr
            self.engine.target = (_redact_addr_tail(raw_addr) if self._cfg.redact
                                  else raw_addr)
            self.engine.reset()
            self._advert_source = None
            self._reacquire_candidates = []
            self._reacquire_index = {}
            self._reacquire_computed_for = None
            self.error = None

        if was_running:
            self.start()

        display = _redact_addr_tail(raw_addr) if self._cfg.redact else raw_addr
        return {"ok": True, "target_addr_display": display}

    # -- controls -------------------------------------------------------
    def control(self, action: dict) -> dict:
        """Dispatch one `/control` POST.

        Cfg-field staging (live_window_s, trend_threshold_db, ...) is
        delegated to the pure :func:`handle_control`. Everything that needs
        the engine or the calibration accumulator -- language, calibration
        add/reset, the "reset" sentinel -- is handled here, where session
        state actually lives.
        """
        action = action or {}
        name = str(action.get("action", ""))

        if name == "lang":
            value = str(action.get("value", "")).lower()
            if value not in ("es", "en"):
                return {"ok": False, "error": f"unknown lang {value!r} / "
                                               f"idioma desconocido {value!r}"}
            self.lang = value
            return {"ok": True, "lang": value}

        if name == "rescan":
            return self._do_rescan(action)

        if name == "retarget":
            return self._do_retarget(action)

        if name == "reacquire_dismiss":
            # Clears the suggestion WITHOUT switching -- the explicit "not
            # this one" the student needs so a wrong guess does not just sit
            # there forever (the panel would otherwise never go away on its
            # own: the target is still vanished, so _maybe_trigger_reacquire
            # would not recompute anything new either).
            with self._lock:
                self._reacquire_candidates = []
                self._reacquire_index = {}
            return {"ok": True}

        if name == "calibration_add":
            try:
                distance_m = float(action.get("distance_m"))
            except (TypeError, ValueError):
                return {"ok": False,
                        "error": "distance_m must be a number / "
                                 "distance_m debe ser un número"}
            snap = self.snapshot()
            if snap.rssi_dbm is None:
                return {"ok": False,
                        "error": "no live reading yet -- wait for a packet / "
                                 "no hay lectura en vivo todavía -- espera "
                                 "un paquete"}
            if snap.units != UNIT_DBM:
                # Link mode's number is a Golden Receive Power Range delta,
                # not dBm (hci.py's module docstring) -- fitting the
                # log-distance model against it would fabricate a distance
                # claim from a number that was never a signal-strength
                # measurement in dBm to begin with. Refuse server-side, not
                # just hide the button, so a stale client / direct POST
                # cannot smuggle a golden-range value into the fit either.
                return {"ok": False,
                        "error": "calibration needs real dBm -- this "
                                 "session's readings are in "
                                 f"{snap.units!r} (link mode), not dBm; use "
                                 "--mode live or --mode bredr for "
                                 "calibration / "
                                 "la calibracion necesita dBm real -- las "
                                 "lecturas de esta sesion estan en "
                                 f"{snap.units!r} (modo enlace), no en dBm; "
                                 "usa --mode live o --mode bredr para "
                                 "calibrar"}
            # The RSSI half of the pair is the meter's OWN live reading at
            # the moment of the click, not something typed by hand -- a
            # calibration point is only honest if both halves came from the
            # same measurement discipline the rest of the meter uses
            # (median of the live window), not from whatever the student
            # remembers the number being a second ago.
            with self._lock:
                try:
                    self._calib_points = add_calibration_point(
                        self._calib_points, distance_m, round(snap.rssi_dbm))
                except ValueError as exc:
                    return {"ok": False, "error": str(exc)}
                points = list(self._calib_points)
            return {"ok": True, "calibration": calibration_payload(points, self.lang)}

        if name == "calibration_reset":
            with self._lock:
                self._calib_points = []
            return {"ok": True, "calibration": calibration_payload([], self.lang)}

        result = handle_control(action, self._cfg)
        if result.get("ok") and result.get("reset"):
            with self._lock:
                self.engine.reset()
        return result


# ------------------------------------------------------------------- app
def create_app(session: LiveSession):
    """Flask app factory. Import is local so Flask stays an optional extra
    for every pure module in this package (SPEC section 4)."""
    from flask import Flask, Response, jsonify, request

    app = Flask(__name__)
    app.config["SESSION"] = session

    @app.get("/")
    def index():
        return Response(PAGE, mimetype="text/html; charset=utf-8")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True, "mode": session.mode})

    @app.post("/control")
    def control():
        body = request.get_json(silent=True) or {}
        try:
            result = session.control(body)
        except Exception as exc:            # a control click must never 500
            return jsonify({"ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}), 400
        return jsonify(result), (200 if result.get("ok", True) else 400)

    @app.get("/stream")
    def stream():
        def gen():
            period = 1.0 / STREAM_HZ
            while True:
                t0 = time.perf_counter()
                try:
                    data = json.dumps(session.payload())
                except Exception as exc:  # never kill the stream
                    data = json.dumps({"error": f"{type(exc).__name__}: {exc}"})
                yield f"data: {data}\n\n"
                dt = period - (time.perf_counter() - t0)
                time.sleep(max(dt, MIN_STREAM_SLEEP))

        return Response(gen(), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no"})

    return app


# ------------------------------------------------------------------- page
#: One self-contained page. Every element that carries text either comes
#: straight from the JSON (`data.*`) or is looked up generically from
#: `data.ui[key]` via `data-ui="<key>"` -- there is no hardcoded English (or
#: Spanish) string anywhere below, including in the initial markup: every
#: `data-ui` element starts EMPTY and is filled by the first SSE message,
#: which arrives within one stream period (<= 1/STREAM_HZ s) of page load.
#: A blank first paint is the honest alternative to guessing a language.
PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>lab-06 &middot; caza BLE en vivo / live BLE hunt</title>
<style>
:root{
  --bg:#1E1E1E; --card:#161b22; --text:#F5F5F5; --muted:#BDBDBD;
  --heading:#FFFFFF; --accent:#81D4FA; --accent2:#F48FB1; --accent3:#A5D6A7;
  --border:#30363d; --vivid:#00D9FF;
  --band-near:#69F0AE; --band-mid:#FFD54F; --band-far:#FFAB40; --band-gone:#FF5252;
}
@media (prefers-color-scheme: light){
  :root{ --bg:#F7F9FC; --card:#FFFFFF; --text:#1E1E1E; --muted:#546E7A;
    --heading:#14213D; --accent:#0277BD; --accent2:#AD1457; --accent3:#2E7D32;
    --border:#D8DEE9; --vivid:#0277BD;
    --band-near:#2E7D32; --band-mid:#F9A825; --band-far:#EF6C00; --band-gone:#C62828; }
}
*{box-sizing:border-box}
html,body{margin:0;padding:0}
body{background:var(--bg);color:var(--text);min-height:100vh;
  font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
header{display:flex;align-items:center;gap:10px;padding:10px 16px;
  border-bottom:1px solid var(--border);background:var(--card);flex-wrap:wrap}
h1{font-size:14px;margin:0;color:var(--heading);font-weight:650;flex:1;min-width:160px}
.badge{font:11px/1 ui-monospace,"SF Mono",Menlo,monospace;padding:5px 9px;
  border-radius:99px;border:1px solid var(--border);color:var(--muted);
  white-space:nowrap}
.badge.stale{display:none;border-color:var(--band-gone);color:var(--band-gone)}
button{background:transparent;border:1px solid var(--border);color:var(--text);
  padding:9px 16px;border-radius:8px;cursor:pointer;font-size:14px}
button:hover{border-color:var(--accent);color:var(--accent)}
button.ghost{opacity:.75}
main{max-width:920px;margin:0 auto;padding:14px 16px 40px;display:flex;
  flex-direction:column;gap:14px}
.card{background:var(--card);border:1px solid var(--border);border-radius:14px;
  padding:16px 18px}

/* -- the meter: legible across a room ---------------------------------- */
.meter{text-align:center;padding:22px 10px 10px}
.mega-row{display:flex;align-items:flex-end;justify-content:center;gap:10px}
.mega{font-family:ui-monospace,"SF Mono","Cascadia Code",Menlo,monospace;
  font-weight:800;font-variant-numeric:tabular-nums;line-height:.85;
  font-size:clamp(76px,24vw,240px);color:var(--band-near);transition:color .25s}
.unit{font:700 20px/1 ui-monospace,monospace;color:var(--muted);padding-bottom:.24em}
.mega.band-arms_reach,.mega.band-same_table{color:var(--band-near)}
.mega.band-same_room{color:var(--band-mid)}
.mega.band-far{color:var(--band-far)}
.mega.band-very_far{color:var(--band-gone)}
.mega.band-not_applicable{color:var(--muted)}
.mega.stale{color:var(--muted)!important;opacity:.5}
.unit-note{display:none;text-align:center;font:12.5px/1.5 system-ui,sans-serif;
  color:var(--accent);max-width:640px;margin:8px auto 0;padding:0 12px}
.unit-note.show{display:block}
.session-error{display:none;text-align:center;font:12.5px/1.6 system-ui,sans-serif;
  color:var(--band-gone);background:var(--bg);border:1px solid var(--band-gone);
  border-radius:10px;padding:10px 14px;margin:0 0 4px}
.session-error.show{display:block}

/* -- vanished: distinct from stale, see hunt.HuntSnapshot.vanished ----- */
.vanished-banner{display:none;text-align:center;font:13px/1.6 system-ui,sans-serif;
  color:var(--band-gone);background:var(--bg);border:2px solid var(--band-gone);
  border-radius:10px;padding:14px 16px;margin:0 0 4px}
.vanished-banner.show{display:block}
.vanished-banner .msg{margin:0 0 10px}

/* -- re-scan / re-acquire candidate lists ------------------------------- */
.card > h2{font-size:12px;margin:0 0 10px;color:var(--accent);font-weight:700;
  letter-spacing:.05em;text-transform:uppercase}
.candidate-list{display:flex;flex-direction:column;gap:8px;margin-top:10px}
.candidate-row{display:flex;align-items:center;justify-content:space-between;gap:10px;
  flex-wrap:wrap;border:1px solid var(--border);border-radius:8px;padding:8px 10px}
.candidate-row .line{flex:1;min-width:200px;font:12.5px/1.5 ui-monospace,monospace;
  color:var(--text)}
.candidate-row .reasons{width:100%;font:11.5px/1.4 system-ui,sans-serif;color:var(--muted)}
.candidate-actions{display:flex;gap:6px;flex-wrap:wrap}
.candidate-actions button{padding:5px 10px;font-size:12.5px}
#reacquire{display:none}
#reacquire.show{display:block}

/* -- sound: the primary output while walking, not decoration ----------- */
.sound-row{display:flex;align-items:center;justify-content:center;gap:12px;
  flex-wrap:wrap;margin-top:14px}
#sound-toggle{font-weight:700}
.sound-zone{font:12.5px/1.4 system-ui,sans-serif;color:var(--muted)}
.sound-footnote{text-align:center}
.band-row{display:flex;align-items:center;justify-content:center;gap:14px;
  flex-wrap:wrap;margin-top:8px}
.band-label{font:800 20px/1.1 system-ui,sans-serif;letter-spacing:.03em;
  text-transform:uppercase;color:var(--heading)}
.trend{display:flex;align-items:center;gap:6px;font:700 15px/1 ui-monospace,monospace;
  color:var(--muted);text-transform:uppercase}
.trend .arrow{font-size:18px}
.trend.warmer{color:var(--band-near)}
.trend.colder{color:var(--band-gone)}
#stale-line{display:none;margin-top:12px;font:12.5px/1.5 system-ui,sans-serif;
  color:var(--band-gone)}
#stale-line.show{display:block}

/* -- bar + sparkline ------------------------------------------------------ */
.bar-track{position:relative;height:24px;border-radius:12px;overflow:hidden;
  border:1px solid var(--border);
  background-image:repeating-linear-gradient(45deg,var(--border) 0 6px,transparent 6px 12px)}
.bar-fill{position:absolute;inset:0 auto 0 0;width:0%;background:var(--vivid);
  border-radius:12px 0 0 12px;transition:width .12s linear}
.spark{display:flex;align-items:flex-end;gap:2px;height:38px;margin-top:10px}
.spark-bar{flex:1;min-width:2px;background:var(--accent);border-radius:2px 2px 0 0;
  transition:height .12s linear}
.instruction{text-align:center;font:13px/1.5 system-ui,sans-serif;color:var(--muted);
  margin:12px 0 0}
.counters{text-align:center;font:11.5px/1.5 ui-monospace,monospace;color:var(--muted);
  margin-top:8px}

/* -- calibration: the panel that carries the lab's finding --------------- */
.calib .quote{font:italic 12.5px/1.5 system-ui,sans-serif;color:var(--accent);
  border-left:3px solid var(--accent);padding-left:10px;margin:0 0 14px}
.calib h2{font-size:12px;margin:0 0 10px;color:var(--accent);font-weight:700;
  letter-spacing:.05em;text-transform:uppercase}
.calib-form{display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;margin-bottom:14px}
.calib-form label{font:11px/1.4 system-ui,sans-serif;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em;display:block;margin-bottom:4px}
.calib-form input{background:var(--bg);color:var(--text);border:1px solid var(--border);
  border-radius:8px;padding:9px 10px;font-size:15px;width:120px}
.calib-body{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:620px){.calib-body{grid-template-columns:1fr}}
.calib-points{font:12px/1.7 ui-monospace,monospace;color:var(--muted);
  max-height:120px;overflow-y:auto}
.calib-points .pt{color:var(--text)}
.calib-msg{font:12px/1.5 system-ui,sans-serif;color:var(--muted);margin-top:8px}
.fit-row{display:flex;justify-content:space-between;font:12.5px/1.7 ui-monospace,monospace}
.fit-row .k{color:var(--muted)}
.ratio-block{text-align:center;padding:12px;border:1px solid var(--border);
  border-radius:10px;background:var(--bg)}
.ratio-label{font:11px/1.4 system-ui,sans-serif;color:var(--muted);
  text-transform:uppercase;letter-spacing:.04em}
.ratio-num{font:800 44px/1.2 ui-monospace,monospace;color:var(--accent2)}
.ratio-msg{font:12.5px/1.5 system-ui,sans-serif;color:var(--muted);margin-top:4px}
.footnote{font:11px/1.5 system-ui,sans-serif;color:var(--muted);margin:14px 0 0}
.langwrap{text-align:center;padding:4px 0 22px}
.target-line{margin:.15rem 0 .6rem;font-size:.95rem;opacity:.85;
  display:flex;gap:.45rem;flex-wrap:wrap;align-items:baseline}
.target-line .target-key{opacity:.7}
.target-line .target-val{font-weight:600;font-family:ui-monospace,Menlo,Consolas,monospace}
.target-line .target-flags{opacity:.7;font-size:.85rem}
</style></head>
<body>
<header>
  <h1 data-ui="title"></h1>
  <span class="badge" id="source-badge">&mdash;</span>
  <span class="badge stale" id="stale-badge" data-ui="stale_label"></span>
</header>
<p class="target-line" id="target-line">
  <span class="target-key" id="target-label"></span>
  <span class="target-val" id="target-display">&mdash;</span>
  <span class="target-flags" id="target-flags"></span>
</p>
<main>
  <p class="session-error" id="session-error"></p>
  <div class="vanished-banner" id="vanished-banner">
    <p class="msg" id="vanished-message"></p>
    <button id="vanished-rescan" data-ui="rescan_button"></button>
  </div>
  <section class="card meter">
    <div class="mega-row">
      <div class="mega" id="mega">&mdash;</div>
      <div class="unit" id="unit"></div>
    </div>
    <div class="band-row">
      <span class="band-label" id="band-label">&mdash;</span>
      <span class="trend" id="trend"><span class="arrow" id="trend-arrow">?</span><span id="trend-word"></span></span>
    </div>
    <p class="unit-note" id="unit-note"></p>
    <div id="stale-line"></div>
    <div class="sound-row">
      <button id="sound-toggle"></button>
      <span class="sound-zone" id="sound-zone-label"></span>
    </div>
    <p class="footnote sound-footnote" data-ui="sound_step_note"></p>
  </section>

  <section class="card">
    <div class="bar-track"><div class="bar-fill" id="bar-fill"></div></div>
    <div class="spark" id="spark"></div>
    <p class="instruction" id="instruction" data-ui="instruction"></p>
    <p class="counters" id="counters"></p>
  </section>

  <section class="card" id="reacquire">
    <h2 data-ui="reacquire_heading"></h2>
    <div class="candidate-list" id="reacquire-list"></div>
    <p class="calib-msg" id="reacquire-none" data-ui="reacquire_none_label" style="display:none"></p>
    <button id="reacquire-dismiss" class="ghost" data-ui="reacquire_dismiss_button"></button>
  </section>

  <section class="card" id="rescan-panel">
    <h2 data-ui="rescan_heading"></h2>
    <button id="rescan-button" data-ui="rescan_button"></button>
    <p class="calib-msg" id="rescan-status"></p>
    <div class="candidate-list" id="rescan-list"></div>
  </section>

  <section class="card calib" id="calib">
    <h2 data-ui="calib_heading"></h2>
    <p class="quote" data-ui="calib_quote"></p>
    <div class="calib-form">
      <div>
        <label data-ui="calib_distance_label"></label>
        <input type="number" id="calib-distance" min="0.1" step="0.1"
               inputmode="decimal" value="1.0">
      </div>
      <button id="calib-record" data-ui="calib_record_button"></button>
      <button id="calib-reset" class="ghost" data-ui="calib_reset_button"></button>
    </div>
    <div class="calib-body">
      <div>
        <div class="calib-points" id="calib-points"></div>
        <div id="calib-fit" style="margin-top:8px"></div>
      </div>
      <div class="ratio-block">
        <div class="ratio-label" data-ui="calib_range_ratio_label"></div>
        <div class="ratio-num" id="calib-ratio">&mdash;</div>
        <div class="ratio-msg" id="calib-message"></div>
      </div>
    </div>
    <p class="calib-msg" id="calib-status"></p>
    <p class="footnote" data-ui="calib_core_spec_note"></p>
  </section>

  <div class="langwrap"><button id="lang" data-ui="lang_button"></button></div>
</main>
<script>
(function(){
  "use strict";
  function $(id){ return document.getElementById(id); }
  var lastUi = {};

  function post(action, params){
    var body = Object.assign({action: action}, params || {});
    return fetch("/control", {method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(body)}).then(function(r){ return r.json(); });
  }

  function fillUi(ui){
    lastUi = ui || {};
    document.querySelectorAll("[data-ui]").forEach(function(el){
      var key = el.getAttribute("data-ui");
      if (lastUi && Object.prototype.hasOwnProperty.call(lastUi, key)) {
        el.textContent = lastUi[key];
      }
    });
  }

  function row(k, v){
    return '<div class="fit-row"><span class="k">' + k + "</span><span>" + v + "</span></div>";
  }

  function renderCalib(c){
    var pts = $("calib-points");
    pts.innerHTML = (c.points || []).map(function(p){
      return '<div class="pt">' + p.distance_m.toFixed(2) + " m &rarr; " + p.rssi_dbm + " dBm</div>";
    }).join("") || "";
    var fitEl = $("calib-fit"), ratioEl = $("calib-ratio"), statusEl = $("calib-status");
    if (c.fit) {
      fitEl.innerHTML =
        row(lastUi.calib_fit_a || "A", c.fit.a_dbm.toFixed(1) + " dBm") +
        row(lastUi.calib_fit_n || "n", c.fit.n.toFixed(2)) +
        row(lastUi.calib_fit_sigma || "σ", c.fit.sigma_db.toFixed(1) + " dB") +
        row(lastUi.calib_fit_r2 || "R²", c.fit.r_squared.toFixed(3));
      ratioEl.textContent = c.fit.range_ratio.toFixed(1) + "×";
    } else {
      fitEl.innerHTML = "";
      ratioEl.textContent = "—";
    }
    statusEl.textContent = c.message || "";
  }

  // ------------------------------------------------- vanished / re-acquire
  // Candidates arrive PRE-RENDERED (data.rescan/data.reacquire's own
  // `line`/`reasons_line` are already the right language -- weblive.py's
  // own discipline, see build_payload's docstring: no template logic in
  // JavaScript). This just lays them out and wires the one action a click
  // can take (`retarget`, via the opaque `id` -- never the visible
  // address string, see LiveSession._next_candidate_id's docstring).
  function renderCandidateList(containerId, candidates, pickLabelKey, showReasons){
    var el = $(containerId);
    var pickLabel = lastUi[pickLabelKey] || "Use this device";
    el.innerHTML = (candidates || []).map(function(c){
      var reasonsHtml = (showReasons && c.reasons_line)
        ? '<div class="reasons">' + c.reasons_line + "</div>" : "";
      return '<div class="candidate-row">' +
        '<div class="line">' + c.line + "</div>" + reasonsHtml +
        '<div class="candidate-actions"><button class="pick-btn" data-id="' +
        c.id + '">' + pickLabel + "</button></div></div>";
    }).join("");
  }

  function renderVanished(data){
    var banner = $("vanished-banner");
    if (data.vanished) {
      $("vanished-message").textContent = data.vanished_message || "";
      banner.className = "vanished-banner show";
    } else {
      banner.className = "vanished-banner";
    }
  }

  function renderRescan(rescan){
    renderCandidateList("rescan-list", (rescan && rescan.candidates) || [],
      "rescan_pick_button", false);
  }

  function renderReacquire(reacquire){
    var cands = (reacquire && reacquire.candidates) || [];
    var panel = $("reacquire");
    if (cands.length > 0) {
      panel.className = "show";
      renderCandidateList("reacquire-list", cands, "reacquire_confirm_button", true);
      $("reacquire-none").style.display = "none";
    } else {
      panel.className = "";
      $("reacquire-list").innerHTML = "";
    }
  }

  // One delegated listener catches every "pick" button in either
  // candidate list -- rows are rebuilt wholesale on each SSE frame, so a
  // per-row listener would need re-attaching every tick.
  document.addEventListener("click", function(ev){
    var btn = ev.target.closest && ev.target.closest(".pick-btn");
    if (!btn) return;
    var id = btn.getAttribute("data-id");
    if (!id) return;
    post("retarget", {id: id});
  });

  function render(data){
    document.documentElement.lang = data.lang || "en";
    fillUi(data.ui);

    var errEl = $("session-error");
    if (data.session_error) {
      errEl.textContent = data.session_error;
      errEl.className = "session-error show";
    } else {
      errEl.textContent = "";
      errEl.className = "session-error";
    }

    var mega = $("mega");
    if (data.rssi_dbm === null || data.rssi_dbm === undefined) {
      mega.textContent = "—";
    } else if (data.is_link_units) {
      // Golden Receive Power Range delta: sign is the whole meaning (0 =
      // inside the range, negative = below it, positive = above it), so it
      // is shown explicitly rather than relying on the usual negative-dBm
      // convention where a bare "-" would be assumed anyway.
      var rounded = Math.round(data.rssi_dbm);
      mega.textContent = (rounded > 0 ? "+" : "") + rounded;
    } else {
      mega.textContent = Math.round(data.rssi_dbm);
    }
    mega.className = "mega band-" + data.band_key + (data.stale ? " stale" : "");
    $("unit").textContent = data.unit_symbol || "dBm";

    var noteEl = $("unit-note");
    if (data.golden_range_note) {
      noteEl.textContent = data.golden_range_note;
      noteEl.className = "unit-note show";
    } else {
      noteEl.textContent = "";
      noteEl.className = "unit-note";
    }

    $("calib").style.display = (data.show_calibration === false) ? "none" : "";

    $("band-label").textContent = data.band_label || "";
    $("trend").className = "trend " + data.trend;
    $("trend-arrow").textContent = data.trend_arrow || "?";
    $("trend-word").textContent = data.trend_word || "";

    $("bar-fill").style.width = (data.fraction * 100).toFixed(1) + "%";

    var levels = Math.max((data.spark_levels || 8) - 1, 1);
    $("spark").innerHTML = (data.spark || []).map(function(level){
      var pct = Math.max(6, (level / levels) * 100);
      return '<div class="spark-bar" style="height:' + pct.toFixed(0) + '%"></div>';
    }).join("");

    $("counters").textContent = data.counters_line || "";
    var tgt = data.target || {};
    $("target-label").textContent = tgt.label ? (tgt.label + ":") : "";
    $("target-display").textContent = tgt.display || "\u2014";
    $("target-flags").textContent = (tgt.flags && tgt.flags.length)
        ? ("\u00b7 " + tgt.flags.join(" \u00b7 ")) : "";
    $("source-badge").textContent = data.source_label || "";

    var staleBadge = $("stale-badge"), staleLine = $("stale-line");
    if (data.stale) {
      staleBadge.style.display = "";
      staleLine.textContent = data.stale_message || "";
      staleLine.className = "show";
    } else {
      staleBadge.style.display = "none";
      staleLine.className = "";
      staleLine.textContent = "";
    }

    renderCalib(data.calibration || {});
    renderVanished(data);
    renderRescan(data.rescan);
    renderReacquire(data.reacquire);

    $("sound-zone-label").textContent = (data.sound && data.sound.zone_label) || "";
    soundState = data.sound || soundState;
    $("sound-toggle").textContent = soundEnabled
      ? (lastUi.sound_on || "sound: on")
      : (lastUi.sound_off || lastUi.sound_enable_button || "Enable sound");
  }

  // ------------------------------------------------------------- sound
  // Geiger-counter proximity cue: closer = faster clicks (Web Audio API,
  // no libraries, no audio files -- generated in-browser). soundState is
  // set from every SSE message's `data.sound` (the pure Python mapping in
  // weblive.sound_cue); this block only decides HOW to play it.
  //
  // Browsers refuse to start audio without a user gesture (autoplay
  // policy) -- the AudioContext is created/resumed ONLY inside the button
  // click handler below, never on page load.
  var audioCtx = null;
  var soundEnabled = false;
  var soundState = {active: false, zone: "waiting", click_hz: 0,
                     pitch_hz: 300, pulse_hz: 0.4};
  var nextSoundEventTime = 0;

  function playClick(freqHz, durationMs, gainPeak){
    if (!audioCtx) return;
    var t = audioCtx.currentTime;
    var osc = audioCtx.createOscillator();
    var gain = audioCtx.createGain();
    osc.type = "square";
    osc.frequency.setValueAtTime(freqHz, t);
    // Fast attack, exponential decay -- a short percussive click, not a
    // continuous tone. A continuous tone gets unbearable within a minute
    // of hunting and the ear reads RATE changes far better than pitch.
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(gainPeak, t + 0.004);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + durationMs / 1000);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start(t);
    osc.stop(t + durationMs / 1000 + 0.02);
  }

  function soundTick(){
    if (!soundEnabled || !audioCtx) return;
    var now = audioCtx.currentTime;
    if (now < nextSoundEventTime) return;
    if (soundState.active && soundState.click_hz > 0) {
      // Proximity click: short, sharp, rate = distance-coded.
      playClick(soundState.pitch_hz || 800, 25, 0.35);
      nextSoundEventTime = now + 1 / soundState.click_hz;
    } else {
      // STALE or no reading yet: NEVER keep clicking at the last known
      // rate -- that would claim a measurement that did not happen. This
      // is a sparse, longer, lower-pitched pulse -- deliberately a
      // different RHYTHM and TIMBRE from any proximity click, so it can
      // never be mistaken for "still this close".
      var pulseHz = soundState.pulse_hz > 0 ? soundState.pulse_hz : 0.4;
      playClick(soundState.pitch_hz || 300, 90, 0.22);
      nextSoundEventTime = now + 1 / pulseHz;
    }
  }
  setInterval(soundTick, 40);

  $("sound-toggle").addEventListener("click", function(){
    if (!audioCtx) {
      var Ctx = window.AudioContext || window.webkitAudioContext;
      if (!Ctx) { return; }  // Web Audio unavailable -- button stays inert
      audioCtx = new Ctx();
    }
    if (audioCtx.state === "suspended") { audioCtx.resume(); }
    soundEnabled = !soundEnabled;
    nextSoundEventTime = 0;
    $("sound-toggle").textContent = soundEnabled
      ? (lastUi.sound_on || "sound: on")
      : (lastUi.sound_off || lastUi.sound_enable_button || "Enable sound");
  });

  var es = new EventSource("/stream");
  es.onmessage = function(ev){
    try { render(JSON.parse(ev.data)); }
    catch (e) { /* one bad frame: skip it, keep the stream open */ }
  };

  $("lang").addEventListener("click", function(){
    var target = ($("lang").textContent || "").trim().toLowerCase();
    if (target === "es" || target === "en") post("lang", {value: target});
  });
  function doRescan(){
    $("rescan-status").textContent = lastUi.rescan_scanning_label || "Scanning...";
    post("rescan", {}).then(function(r){
      $("rescan-status").textContent = (r && r.ok === false)
        ? (r.error || "") : (lastUi.rescan_no_candidates && r && r.candidates &&
           r.candidates.length === 0 ? lastUi.rescan_no_candidates : "");
    });
  }
  $("rescan-button").addEventListener("click", doRescan);
  $("vanished-rescan").addEventListener("click", doRescan);
  $("reacquire-dismiss").addEventListener("click", function(){
    post("reacquire_dismiss", {}).then(function(){
      renderReacquire({candidates: []});
    });
  });
  $("calib-record").addEventListener("click", function(){
    var v = parseFloat($("calib-distance").value);
    if (!isFinite(v) || v <= 0) return;
    post("calibration_add", {distance_m: v}).then(function(r){
      if (r && r.calibration) renderCalib(r.calibration);
      else if (r && r.error) $("calib-status").textContent = r.error;
    });
  });
  $("calib-reset").addEventListener("click", function(){
    post("calibration_reset", {}).then(function(r){
      if (r && r.calibration) renderCalib(r.calibration);
    });
  });
})();
</script>
</body></html>
"""
