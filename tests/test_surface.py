"""Viewing-geometry and reflected-sun model checks."""

import math

import numpy as np
import pytest

from ircam.pyrometry import NotchFilter, RatioPyrometer
from ircam.sensors import imx900_camera
from ircam.surface import (
    MATERIALS,
    OpticalConstants,
    directional_emissivity,
    fresnel_reflectances,
    one_band_apparent_temperature,
    ratio_apparent_temperature,
    solar_reflected_electron_rate,
    solar_spectral_irradiance,
    specular_glint_ratio,
)

CAM = imx900_camera(4.0, 0.85)
F620, F870 = NotchFilter(620e-9, 30e-9), NotchFilter(870e-9, 50e-9)
PYRO = RatioPyrometer(F620, F870, CAM)
GRAPHITE = MATERIALS["graphite-like"]
OXIDE = MATERIALS["oxide-like dielectric"]


def test_normal_incidence_closed_form():
    """eps(0) = 1 - ((n-1)^2 + k^2) / ((n+1)^2 + k^2)."""
    for n, k in ((1.7, 0.1), (2.7, 1.4), (3.6, 2.9)):
        expected = 1.0 - ((n - 1) ** 2 + k**2) / ((n + 1) ** 2 + k**2)
        assert float(directional_emissivity(0.0, n, k)) == pytest.approx(expected, rel=1e-12)


def test_brewster_angle_for_lossless_dielectric():
    """R_p vanishes at atan(n) when k = 0."""
    n = 1.7
    _, r_p = fresnel_reflectances(math.atan(n), n, 0.0)
    assert float(r_p) == pytest.approx(0.0, abs=1e-12)


def test_grazing_limit_and_energy_bounds():
    theta = np.radians(np.linspace(0, 89.9, 200))
    for mat in MATERIALS.values():
        eps = mat.emissivity(theta, 620e-9)
        assert np.all((eps >= 0.0) & (eps <= 1.0))
        assert eps[-1] < 0.05                       # -> 0 at grazing incidence
        assert np.all(np.diff(eps[theta > math.radians(80)]) < 0)  # collapsing


def test_one_band_round_trip_and_signs():
    t_true = 2273.15
    # Same angle as calibration, no sun: exact recovery.
    assert one_band_apparent_temperature(t_true, F620, CAM, GRAPHITE, 0.0) == \
        pytest.approx(t_true, abs=0.01)
    # Grazing view with a normal-incidence calibration reads cold.
    assert one_band_apparent_temperature(t_true, F620, CAM, GRAPHITE,
                                         math.radians(85)) < t_true - 100
    # Full sun on a normally viewed surface reads hot.
    assert one_band_apparent_temperature(1773.15, F620, CAM, GRAPHITE, 0.0,
                                         theta_s=0.0, sun_factor=1.0) > 1773.15


def test_ratio_cancels_angular_factor_for_dielectric():
    """Same n, k in both bands -> eps_1/eps_2 is angle-independent -> no bias."""
    t_true = 2273.15
    for deg in (45, 75, 85):
        assert ratio_apparent_temperature(t_true, PYRO, OXIDE, math.radians(deg)) == \
            pytest.approx(t_true, abs=0.05)


def test_sun_biases_ratio_more_than_one_band_at_1500c():
    """The sun term is differential (stronger at 620 than 870) and propagates
    through lam_eq, so under full sun the ratio bias exceeds the one-band bias."""
    t_true = 1773.15
    theta_v, theta_s = math.radians(45), math.radians(45)
    one = one_band_apparent_temperature(t_true, F620, CAM, GRAPHITE, theta_v,
                                        theta_s, 1.0, cal_theta=theta_v) - t_true
    two = ratio_apparent_temperature(t_true, PYRO, GRAPHITE, theta_v, theta_s, 1.0,
                                     cal_theta=theta_v) - t_true
    assert one > 0 and two > 0
    assert two > 2.0 * one
    # A 90% pre-ignition subtraction scales the bias down roughly tenfold.
    two_sub = ratio_apparent_temperature(t_true, PYRO, GRAPHITE, theta_v, theta_s,
                                         1.0, cal_theta=theta_v, sun_residual=0.1) - t_true
    assert 0.05 * two < two_sub < 0.15 * two


def test_solar_terms():
    assert float(solar_spectral_irradiance(620e-9)) == pytest.approx(1.44e9, rel=0.05)
    assert float(solar_spectral_irradiance(300e-9)) == 0.0
    # Reflected-sun electrons vanish when shaded or when the sun is behind the surface.
    assert solar_reflected_electron_rate(F620, CAM, GRAPHITE, 0.0, 0.0, 0.0) == 0.0
    assert solar_reflected_electron_rate(F620, CAM, GRAPHITE, 0.0, math.radians(95), 1.0) == 0.0
    # Specular glint outshines even a 3000 C surface at 620 nm.
    assert specular_glint_ratio(math.radians(45), 620e-9, 3273.15, GRAPHITE) > 3.0


def test_optical_constants_interpolate():
    mat = OpticalConstants("t", (600e-9, 800e-9), (2.0, 3.0), (1.0, 2.0))
    n, k = mat.nk(700e-9)
    assert float(n) == pytest.approx(2.5) and float(k) == pytest.approx(1.5)
