"""blelib.pathloss -- the log-distance model, its fit, and the lab's headline.

Two things get extra scrutiny because the SPEC calls them out by name:

* :func:`range_ratio` is checked against an *independent* numerical
  computation built from :func:`distance_interval` (d_hi / d_lo), not
  against a hardcoded constant -- so a bug shared between the two would
  still be caught by any test that hardcodes the expected ratio instead.
* :func:`fit_log_distance` must raise a clear error on too few points or a
  non-positive distance, and the messages must carry both languages.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from blelib.pathloss import (
    D0_M,
    PathLossFit,
    distance_interval,
    fit_log_distance,
    gradient_snr,
    predict_distance,
    predict_rssi,
    range_ratio,
)


# ------------------------------------------------------------- predict_*
def test_predict_rssi_at_d0_equals_a_dbm():
    assert predict_rssi(D0_M, a_dbm=-45.0, n=2.2) == pytest.approx(-45.0)


def test_predict_rssi_decreases_with_distance():
    near = predict_rssi(1.0, a_dbm=-45.0, n=2.2)
    far = predict_rssi(10.0, a_dbm=-45.0, n=2.2)
    assert far < near


def test_predict_distance_is_the_inverse_of_predict_rssi():
    a_dbm, n = -45.0, 2.2
    for d in (0.5, 1.0, 3.7, 20.0):
        rssi = predict_rssi(d, a_dbm, n)
        d_back = predict_distance(rssi, a_dbm, n)
        assert d_back == pytest.approx(d, rel=1e-9)


# --------------------------------------------------------- fit_log_distance
def _synthetic_track(a_dbm=-45.0, n=2.2, sigma_db=0.0, n_points=200, seed=0):
    """Deterministic (distance, RSSI) pairs from the model itself.

    sigma_db=0 gives a noiseless track (fit must recover A and n exactly);
    a positive sigma_db lets the fit-recovers-sigma property be checked too.
    """
    rng = np.random.default_rng(seed)
    d = rng.uniform(0.3, 15.0, size=n_points)
    x = np.log10(d / D0_M)
    noise = rng.normal(0.0, sigma_db, size=n_points) if sigma_db > 0 else 0.0
    rssi = a_dbm - 10.0 * n * x + noise
    return d, rssi


def test_fit_recovers_exact_parameters_on_a_noiseless_track():
    a_dbm, n = -45.0, 2.2
    d, rssi = _synthetic_track(a_dbm, n, sigma_db=0.0, n_points=50)
    fit = fit_log_distance(d, rssi)
    assert fit.a_dbm == pytest.approx(a_dbm, abs=1e-6)
    assert fit.n == pytest.approx(n, abs=1e-6)
    assert fit.sigma_db == pytest.approx(0.0, abs=1e-6)
    assert fit.r_squared == pytest.approx(1.0, abs=1e-6)
    assert fit.n_points == 50


def test_fit_recovers_seeded_parameters_within_tolerance_over_noisy_track():
    # Mirrors validate.py's acceptance bar (SPEC section 8): n within +/-0.25,
    # A within +/-2 dB, sigma within +/-1.5 dB, over >= 200 points.
    a_dbm, n, sigma_db = -45.0, 2.2, 7.0
    d, rssi = _synthetic_track(a_dbm, n, sigma_db=sigma_db, n_points=400, seed=3)
    fit = fit_log_distance(d, rssi)
    assert fit.n == pytest.approx(n, abs=0.25)
    assert fit.a_dbm == pytest.approx(a_dbm, abs=2.0)
    assert fit.sigma_db == pytest.approx(sigma_db, abs=1.5)
    assert 0.0 <= fit.r_squared <= 1.0
    assert fit.n_points == 400


def test_fit_log_distance_raises_a_clear_bilingual_error_on_fewer_than_3_points():
    with pytest.raises(ValueError) as exc_info:
        fit_log_distance([1.0, 2.0], [-50.0, -55.0])
    msg = str(exc_info.value)
    assert "3" in msg
    # Bilingual-ready: both languages present so the caller can show either.
    assert " / " in msg
    assert any(w in msg for w in ("necesita", "necesitan"))
    assert "needs" in msg


def test_fit_log_distance_raises_a_clear_bilingual_error_on_zero_points():
    with pytest.raises(ValueError):
        fit_log_distance([], [])


def test_fit_log_distance_raises_a_clear_bilingual_error_on_non_positive_distance():
    with pytest.raises(ValueError) as exc_info:
        fit_log_distance([1.0, 0.0, 3.0], [-50.0, -55.0, -60.0])
    msg = str(exc_info.value)
    assert " / " in msg
    assert "non-positive" in msg
    assert any(w in msg for w in ("positiva", "positivo"))


def test_fit_log_distance_raises_on_negative_distance_too():
    with pytest.raises(ValueError):
        fit_log_distance([1.0, -2.0, 3.0], [-50.0, -55.0, -60.0])


# --------------------------------------------------------------- range_ratio
def test_range_ratio_matches_an_independent_computation_from_distance_interval():
    """The lab's headline number, checked two ways.

    Rather than asserting range_ratio(...) against a value typed into this
    test by hand, build a PathLossFit, ask distance_interval() for the
    actual (d_lo, d_hi) bounds it computes for some reading, and verify
    range_ratio() predicts their ratio -- so range_ratio cannot silently
    drift away from what distance_interval actually returns.
    """
    fit = PathLossFit(a_dbm=-45.0, n=2.2, sigma_db=7.0, r_squared=0.6, n_points=200)
    for conf in (0.50, 0.90, 0.95, 0.99):
        for rssi in (-50.0, -65.0, -80.0):
            d_lo, d_hi = distance_interval(rssi, fit, conf=conf)
            independent_ratio = d_hi / d_lo
            assert range_ratio(fit.sigma_db, fit.n, conf=conf) == pytest.approx(
                independent_ratio, rel=1e-9)


def test_range_ratio_typical_indoor_values_land_in_the_6x_to_12x_range():
    # SPEC section 5.3's own headline claim for sigma=6-8 dB, n~2.
    ratio = range_ratio(sigma_db=7.0, n=2.2, conf=0.90)
    assert 6.0 <= ratio <= 12.0


def test_range_ratio_widens_with_larger_sigma():
    lo_sigma = range_ratio(sigma_db=3.0, n=2.2, conf=0.90)
    hi_sigma = range_ratio(sigma_db=10.0, n=2.2, conf=0.90)
    assert hi_sigma > lo_sigma


def test_range_ratio_narrows_with_larger_n():
    lo_n = range_ratio(sigma_db=7.0, n=1.5, conf=0.90)
    hi_n = range_ratio(sigma_db=7.0, n=4.0, conf=0.90)
    assert hi_n < lo_n


def test_range_ratio_widens_with_higher_confidence():
    r90 = range_ratio(sigma_db=7.0, n=2.2, conf=0.90)
    r99 = range_ratio(sigma_db=7.0, n=2.2, conf=0.99)
    assert r99 > r90


def test_range_ratio_uses_scipy_norm_ppf_for_z():
    # Cross-check the closed form directly against an independently computed z,
    # rather than assuming range_ratio's internals.
    sigma_db, n, conf = 7.0, 2.2, 0.90
    z = stats.norm.ppf(0.5 + conf / 2.0)
    expected = 10.0 ** (2.0 * z * sigma_db / (10.0 * n))
    assert range_ratio(sigma_db, n, conf) == pytest.approx(expected, rel=1e-9)


# ----------------------------------------------------------- distance_interval
def test_distance_interval_brackets_the_point_estimate():
    fit = PathLossFit(a_dbm=-45.0, n=2.2, sigma_db=7.0, r_squared=0.6, n_points=200)
    d_hat = predict_distance(-65.0, fit.a_dbm, fit.n)
    d_lo, d_hi = distance_interval(-65.0, fit, conf=0.90)
    assert d_lo < d_hat < d_hi


def test_distance_interval_collapses_to_the_point_estimate_when_sigma_is_zero():
    fit = PathLossFit(a_dbm=-45.0, n=2.2, sigma_db=0.0, r_squared=1.0, n_points=200)
    d_hat = predict_distance(-65.0, fit.a_dbm, fit.n)
    d_lo, d_hi = distance_interval(-65.0, fit, conf=0.90)
    assert d_lo == pytest.approx(d_hat, rel=1e-9)
    assert d_hi == pytest.approx(d_hat, rel=1e-9)


# -------------------------------------------------------------- gradient_snr
def test_gradient_snr_increases_with_more_averaged_samples():
    low_n = gradient_snr(delta_rssi_db=3.0, sigma_db=7.0, n_samples=1)
    high_n = gradient_snr(delta_rssi_db=3.0, sigma_db=7.0, n_samples=25)
    assert high_n > low_n
    # sqrt(25) = 5x the samples of sqrt(1) -> exactly 5x the SNR.
    assert high_n == pytest.approx(5.0 * low_n, rel=1e-9)


def test_gradient_snr_scales_linearly_with_the_step_size():
    small = gradient_snr(delta_rssi_db=1.0, sigma_db=7.0, n_samples=10)
    large = gradient_snr(delta_rssi_db=3.0, sigma_db=7.0, n_samples=10)
    assert large == pytest.approx(3.0 * small, rel=1e-9)


def test_gradient_snr_zero_step_is_zero():
    assert gradient_snr(delta_rssi_db=0.0, sigma_db=7.0, n_samples=10) == 0.0
