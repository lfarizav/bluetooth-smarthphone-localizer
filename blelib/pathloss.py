"""The log-distance path-loss model, and what it honestly tells us about distance.

This module exists to produce exactly one number the lab is built around
(SPEC section 1): given a student's own calibration data, how wide is the
distance interval implied by their own measured shadowing? The answer is
"close to an order of magnitude" for typical indoor sigma -- not because the
lab says so, but because :func:`fit_log_distance` measured it from readings
the student took themselves, and :func:`range_ratio` turns that measurement
into a single headline ratio.

Model (module-02, `d0 = 1 m`)::

    RSSI(d) = A - 10*n*log10(d / d0) + X,   X ~ N(0, sigma^2)

``A`` is the RSSI intercept at one metre, ``n`` the path-loss exponent
(indoor offices run higher than free-space's 2.0 because of walls and
furniture), and ``X`` the log-normal shadowing term whose spread, ``sigma``,
is what makes a single RSSI reading an unreliable ruler no matter how well
``A`` and ``n`` are known.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy import stats

#: Reference distance the model is anchored to. 1 m is the Bluetooth SIG's
#: own convention (`tx_power` in an advertisement is specified at 1 m), so
#: using anything else here would make ``A`` incomparable to that field.
D0_M: float = 1.0


@dataclass(frozen=True, slots=True)
class PathLossFit:
    """One calibration fit -- what a student's (distance, RSSI) pairs imply."""

    a_dbm: float
    """RSSI at ``d0`` (1 m)."""
    n: float
    """Path-loss exponent."""
    sigma_db: float
    """Residual standard deviation -- the shadowing term. This, not ``A`` or
    ``n``, is the number that drives :func:`range_ratio`: it is the direct
    measurement of how much a single reading can lie."""
    r_squared: float
    n_points: int


def predict_rssi(d_m, a_dbm: float, n: float, d0: float = D0_M):
    """The model's mean RSSI at distance ``d_m`` -- ``X`` set to its mean, 0.

    Accepts a scalar or an array of distances, and returns the matching shape:
    a plain ``float`` for a scalar, an ``ndarray`` for anything array-like.
    The unconditional ``float(...)`` this used to end with raised
    ``TypeError: only 0-dimensional arrays can be converted to Python scalars``
    for *any* array argument -- including a one-element one -- which meant the
    curve in ``plots.pathloss_fit`` and the whole of ``plots.write_all``
    crashed on every real run, since both pass the distance array straight in.
    """
    out = a_dbm - 10.0 * n * np.log10(np.asarray(d_m, dtype=float) / d0)
    return float(out) if np.ndim(out) == 0 else out


def predict_distance(rssi_dbm: float, a_dbm: float, n: float,
                     d0: float = D0_M) -> float:
    """Invert the model for the distance whose *mean* RSSI equals ``rssi_dbm``.

    This is a point estimate only -- it assumes the shadowing term was
    exactly 0 for this particular reading, which it never is. That is
    precisely the gap :func:`distance_interval` quantifies rather than hides.
    """
    return float(d0 * 10.0 ** ((a_dbm - rssi_dbm) / (10.0 * n)))


def fit_log_distance(distances_m: Sequence[float],
                     rssi_dbm: Sequence[float]) -> PathLossFit:
    """Ordinary least squares of RSSI on ``log10(d)``.

    Two points make a line with no residual at all -- ``sigma_db`` would
    read exactly 0, which would silently hand the lab's headline number a
    false "the meter is perfectly precise" answer. Three is the minimum that
    leaves at least one degree of freedom to measure shadowing honestly, so
    that is where the floor sits. A non-positive distance is rejected before
    the fit even runs: ``log10`` of it is undefined, and a distance of 0 or
    less was never a real calibration point to begin with.
    """
    d = np.asarray(distances_m, dtype=float)
    y = np.asarray(rssi_dbm, dtype=float)

    if d.size < 3:
        raise ValueError(
            f"fit_log_distance needs at least 3 (distance, RSSI) points to "
            f"estimate both the path-loss line and its residual spread; got "
            f"{d.size}. / fit_log_distance necesita al menos 3 puntos "
            f"(distancia, RSSI) para estimar la recta y su dispersión "
            f"residual; se recibieron {d.size}.")
    if np.any(d <= 0):
        raise ValueError(
            "fit_log_distance received a non-positive distance; every "
            "calibration point must be a real, positive distance in metres "
            "(log10 of a distance <= 0 is undefined). / "
            "fit_log_distance recibió una distancia no positiva; cada punto "
            "de calibración debe ser una distancia real y positiva en "
            "metros (log10 de una distancia <= 0 no está definido).")

    x = np.log10(d / D0_M)
    # y = b0 + b1*x, with b0 = A (intercept at d0) and b1 = -10n (slope).
    b1, b0 = np.polyfit(x, y, 1)
    n = -b1 / 10.0

    y_pred = b0 + b1 * x
    resid = y - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # ddof=2: two parameters (A, n) were estimated from the same data used
    # to measure their own residual, so dividing by n_points would quietly
    # understate sigma -- the classic OLS unbiased-variance correction.
    dof = d.size - 2
    sigma_db = float(np.sqrt(ss_res / dof)) if dof > 0 else 0.0

    return PathLossFit(a_dbm=float(b0), n=float(n), sigma_db=sigma_db,
                       r_squared=float(r_squared), n_points=int(d.size))


def range_ratio(sigma_db: float, n: float, conf: float = 0.90) -> float:
    """Closed form for how wide the distance CI is, independent of the reading.

        ratio = d_hi / d_lo = 10 ** (2 * z * sigma_db / (10 * n))

    ``z`` is the two-sided normal quantile for ``conf`` -- ``scipy.stats``
    already implements the inverse normal CDF correctly (including its
    numerically stable tail behaviour near ``conf`` close to 1), so this
    reimplements none of that; only the closed form above is new. This is
    the lab's headline number: with a typical indoor sigma of 6-8 dB and
    ``n`` ~ 2, it lands between roughly 6x and 12x.
    """
    z = stats.norm.ppf(0.5 + conf / 2.0)
    return float(10.0 ** (2.0 * z * sigma_db / (10.0 * n)))


def distance_interval(rssi_dbm: float, fit: PathLossFit,
                      conf: float = 0.90) -> tuple[float, float]:
    """The distance CI implied by the student's own fit, for one reading.

    Built from first principles rather than by calling :func:`range_ratio`
    directly: inverting the model shows ``X`` enters as an *additive* term
    on ``log10(d)``, so the point estimate ``predict_distance(...)`` is the
    geometric centre of the interval and the half-width in log-space is
    ``z * sigma_db / (10 * n)`` on each side. The ratio of the two bounds
    this produces is, by construction, exactly :func:`range_ratio` -- which
    is what makes that closed form trustworthy rather than a second,
    independently-guessed formula that happens to look similar.
    """
    d_hat = predict_distance(rssi_dbm, fit.a_dbm, fit.n, D0_M)
    z = stats.norm.ppf(0.5 + conf / 2.0)
    half_width_log = z * fit.sigma_db / (10.0 * fit.n)
    d_lo = d_hat * 10.0 ** (-half_width_log)
    d_hi = d_hat * 10.0 ** (half_width_log)
    return (float(d_lo), float(d_hi))


def gradient_snr(delta_rssi_db: float, sigma_db: float, n_samples: int) -> float:
    """Why the TREND survives what the absolute value does not.

    Averaging ``n_samples`` independent readings shrinks the shadowing
    term's standard deviation by ``sqrt(n)`` (central limit theorem) while a
    genuine walk-driven change in mean RSSI is untouched by that averaging.
    The ratio below is that step size measured in units of the *averaged*
    noise: at gradient_snr >> 1 the step is detectable; a typical 3 dB
    decision-task threshold (``TREND_THRESHOLD_DB`` in :mod:`signal`)
    clears sigma/sqrt(n) long before the equivalent 1 m-scale distance
    change would clear the raw sigma used by :func:`range_ratio`.
    """
    n = max(int(n_samples), 1)
    if sigma_db <= 0:
        return float("inf") if delta_rssi_db != 0 else 0.0
    noise_after_averaging = sigma_db / np.sqrt(n)
    return float(delta_rssi_db / noise_after_averaging)
