"""Validation of the ratio-pyrometry module against Wien-limit closed forms."""

import math

import numpy as np
import pytest

from ircam.constants import C2
from ircam.pyrometry import (
    NotchFilter,
    PyroCamera,
    RatioPyrometer,
    electron_rate,
    equivalent_wavelength,
    exposure_for_well_fill,
    misregistered_ratio_temperature_wien,
    misregistration_gain,
    ratio_temperature_error_wien,
    silicon_qe,
    single_band_temperature_error,
)


@pytest.fixture
def pyrometer():
    return RatioPyrometer(
        filter_short=NotchFilter(620e-9, 30e-9),
        filter_long=NotchFilter(870e-9, 50e-9),
        camera=PyroCamera(),
    )


def test_ratio_monotonic_and_round_trip(pyrometer):
    temps = np.array([1200.0, 1800.0, 2400.0, 3000.0, 3600.0])
    ratios = np.array([pyrometer.ratio(t) for t in temps])
    # Short band gains on long band as T rises: ratio strictly increasing.
    assert np.all(np.diff(ratios) > 0)
    for t in temps:
        assert pyrometer.temperature_from_ratio(pyrometer.ratio(t)) == \
            pytest.approx(t, abs=0.01)


def test_dlnratio_matches_wien_slope(pyrometer):
    """Full-integral slope must match the Wien form (c2/T^2)(1/l1 - 1/l2)."""
    for t in (1773.0, 2773.0):
        wien = C2 / t**2 * (1.0 / pyrometer.filter_short.center
                            - 1.0 / pyrometer.filter_long.center)
        assert pyrometer.dlnratio_dT(t) == pytest.approx(wien, rel=0.03)


def test_sigma_t_consistent_with_wien_formula(pyrometer):
    """Numeric NEdT ~ (lam_eq T^2/c2) * sigma_R/R when read noise negligible."""
    t = 2273.0
    t_exp = 1e-4
    n1 = electron_rate(t, pyrometer.filter_short, pyrometer.camera) * t_exp
    n2 = electron_rate(t, pyrometer.filter_long, pyrometer.camera) * t_exp
    rel_ratio = math.sqrt(1.0 / n1 + 1.0 / n2)  # pure shot noise
    expected = ratio_temperature_error_wien(
        t, pyrometer.filter_short.center, pyrometer.filter_long.center, rel_ratio
    )
    quiet = PyroCamera(read_noise=0.0)
    quiet_pyro = RatioPyrometer(pyrometer.filter_short, pyrometer.filter_long, quiet)
    assert quiet_pyro.sigma_T(t, t_exp, t_exp) == pytest.approx(expected, rel=0.03)
    # Averaging 4 frames halves the noise.
    assert quiet_pyro.sigma_T(t, t_exp, t_exp, frames=4) == pytest.approx(
        0.5 * quiet_pyro.sigma_T(t, t_exp, t_exp), rel=1e-6
    )


def test_emissivity_bias_sign_and_wien_magnitude(pyrometer):
    t = 3000.0
    assert pyrometer.emissivity_bias(t, 1.0) == pytest.approx(0.0, abs=0.01)
    # eps_short/eps_long > 1 -> ratio reads high -> apparent T too hot.
    bias = pyrometer.emissivity_bias(t, 1.05)
    assert bias > 0
    wien = (equivalent_wavelength(pyrometer.filter_short.center,
                                  pyrometer.filter_long.center)
            * t**2 / C2 * math.log(1.05))
    assert bias == pytest.approx(wien, rel=0.05)


def test_single_band_error_formula():
    # Classic result: at 650 nm, 10% signal/emissivity error at 3273 K ~ 48 K.
    err = single_band_temperature_error(3273.0, 650e-9, 0.10)
    assert err == pytest.approx(48.4, rel=0.02)


def test_exposure_for_well_fill(pyrometer):
    t_exp = exposure_for_well_fill(3273.0, pyrometer.filter_long,
                                   pyrometer.camera, fill=0.6)
    n_e = electron_rate(3273.0, pyrometer.filter_long, pyrometer.camera) * t_exp
    assert n_e == pytest.approx(0.6 * pyrometer.camera.well_capacity, rel=1e-9)
    # At 3000 C the required exposure is microseconds -- saturation regime.
    assert t_exp < 1e-4


def test_silicon_qe_range():
    assert silicon_qe(550e-9) == pytest.approx(0.75)
    assert silicon_qe(1200e-9) == 0.0
    assert silicon_qe(300e-9) == 0.0


def test_misregistered_ratio_wien_vs_full_integrals(pyrometer):
    """Two cameras registered imperfectly: the bands view spots dT apart.
    Wien closed form 1/T = [lam2/T_s - lam1/T_l]/(lam2 - lam1) against the
    exact inversion with full band integrals; gains lam2/(lam2 - lam1) and
    lam1/(lam2 - lam1) for the short- and long-band offsets."""
    lam1, lam2 = 620e-9, 870e-9
    t, d = 1500.0 + 273.15, 10.0
    assert misregistered_ratio_temperature_wien(t, t, lam1, lam2) == pytest.approx(t)
    assert pyrometer.misregistered_temperature(t, t) == pytest.approx(t, abs=0.02)
    g_long = misregistration_gain(lam1, lam2, "long")
    g_short = misregistration_gain(lam1, lam2, "short")
    assert g_long == pytest.approx(620.0 / 250.0)
    assert g_short == pytest.approx(870.0 / 250.0)
    assert g_short - g_long == pytest.approx(1.0)   # the weights sum to one
    with pytest.raises(ValueError):
        misregistration_gain(lam1, lam2, "middle")
    # Long band views the hotter spot -> reported colder by ~g_long * d.
    wien = misregistered_ratio_temperature_wien(t, t + d, lam1, lam2) - t
    full = pyrometer.misregistered_temperature(t, t + d) - t
    assert wien < 0 and full < 0
    assert wien == pytest.approx(-g_long * d, rel=0.03)
    assert full == pytest.approx(wien, rel=0.05)
    # Short band views the hotter spot -> reported hotter by ~g_short * d.
    wien = misregistered_ratio_temperature_wien(t + d, t, lam1, lam2) - t
    full = pyrometer.misregistered_temperature(t + d, t) - t
    assert wien > 0 and full > 0
    assert wien == pytest.approx(g_short * d, rel=0.03)
    assert full == pytest.approx(wien, rel=0.05)
    # Vectorised over an array of long-band temperatures.
    arr = misregistered_ratio_temperature_wien(t, t + np.array([0.0, d, 2 * d]), lam1, lam2)
    assert arr.shape == (3,) and np.all(np.diff(arr) < 0)
