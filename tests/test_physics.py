"""Physics validation tests.

Each test checks the package against an independent reference:
closed-form laws (Stefan-Boltzmann, Wien), the series expansion of the
blackbody fraction function, per-photon energy consistency, and known
scaling behaviour of the system-level formulas.
"""

import math

import numpy as np
import pytest

from ircam import (
    CLEAR_SEA_LEVEL,
    LWIR,
    MWIR,
    Fpa,
    Optics,
    PhotonDetectorChain,
    dstar_from_jones,
    netd_from_dstar,
    planck,
)
from ircam.constants import C, C2, H, SIGMA
from ircam.range_performance import (
    JOHNSON_N50,
    contrast_limited_range,
    cycles_on_target,
    focal_length_for_task,
    sampling_limited_range,
)


# ---------------------------------------------------------------- Planck law

def test_planck_integrates_to_stefan_boltzmann():
    """Integral of Planck radiance over all wavelengths = sigma T^4 / pi."""
    for temperature in (200.0, 300.0, 500.0, 1000.0):
        total = planck.band_radiance(temperature, 1e-8, 1e-2, order=512)
        expected = SIGMA * temperature**4 / math.pi
        assert total == pytest.approx(expected, rel=1e-6)


def test_wien_peak_matches_numerical_maximum():
    temperature = 300.0
    lam_peak = planck.wien_peak_wavelength(temperature)
    lam = np.linspace(0.8 * lam_peak, 1.2 * lam_peak, 20001)
    radiance = planck.spectral_radiance(lam, temperature)
    assert lam[np.argmax(radiance)] == pytest.approx(lam_peak, rel=1e-4)
    # 300 K scene peaks near 9.66 um -- squarely in LWIR.
    assert LWIR.lam1 < lam_peak < LWIR.lam2


def _fraction_series(lam_t: float) -> float:
    """Independent blackbody fraction F(0 -> lambda*T) by series expansion."""
    x = C2 / lam_t
    total = 0.0
    for n in range(1, 200):
        total += math.exp(-n * x) / n * (
            x**3 + 3.0 * x**2 / n + 6.0 * x / n**2 + 6.0 / n**3
        )
    return 15.0 / math.pi**4 * total


@pytest.mark.parametrize("band", [MWIR, LWIR])
def test_band_fraction_matches_series_expansion(band):
    temperature = 300.0
    expected = _fraction_series(band.lam2 * temperature) - _fraction_series(
        band.lam1 * temperature
    )
    computed = planck.blackbody_fraction(temperature, band.lam1, band.lam2)
    assert computed == pytest.approx(expected, rel=1e-6)


def test_photon_and_energy_radiance_consistent():
    """L_lambda / L_q,lambda must equal the photon energy hc/lambda."""
    lam = np.array([1e-6, 4e-6, 10e-6])
    ratio = planck.spectral_radiance(lam, 300.0) / planck.spectral_photon_radiance(
        lam, 300.0
    )
    assert np.allclose(ratio, H * C / lam, rtol=1e-12)


def test_thermal_derivative_matches_finite_difference():
    temperature, dt = 300.0, 1e-3
    for band in (MWIR, LWIR):
        analytic = planck.band_radiance_dT(temperature, band.lam1, band.lam2)
        numeric = (
            planck.band_radiance(temperature + dt, band.lam1, band.lam2)
            - planck.band_radiance(temperature - dt, band.lam1, band.lam2)
        ) / (2.0 * dt)
        assert analytic == pytest.approx(numeric, rel=1e-7)


def test_known_300k_band_values():
    """Order-of-magnitude anchors from standard blackbody tables at 300 K."""
    lwir = planck.band_radiance(300.0, LWIR.lam1, LWIR.lam2)
    mwir = planck.band_radiance(300.0, MWIR.lam1, MWIR.lam2)
    # 8-14 um holds ~37-38% of the total 146 W/m^2/sr; 3-5 um ~1.3%.
    assert 50.0 < lwir < 60.0
    assert 1.5 < mwir < 2.5
    # MWIR relative contrast (~3.5-4 %/K) beats LWIR (~1.5-1.8 %/K).
    rel_mwir = planck.band_radiance_dT(300.0, MWIR.lam1, MWIR.lam2) / mwir
    rel_lwir = planck.band_radiance_dT(300.0, LWIR.lam1, LWIR.lam2) / lwir
    assert 0.030 < rel_mwir < 0.045
    assert 0.013 < rel_lwir < 0.020
    assert rel_mwir > 2.0 * rel_lwir


# ------------------------------------------------------------------- optics

def test_pixel_solid_angle_limits():
    fast = Optics(focal_length=0.05, f_number=1.0)
    slow = Optics(focal_length=0.05, f_number=8.0)
    assert fast.pixel_solid_angle == pytest.approx(math.pi / 5.0)
    # Slow optics approach the small-angle form pi/(4 F#^2).
    assert slow.pixel_solid_angle == pytest.approx(
        math.pi / (4.0 * 8.0**2), rel=5e-3
    )


def test_ifov_and_fov():
    optics = Optics(focal_length=0.040, f_number=1.0)
    assert optics.ifov(12e-6) == pytest.approx(300e-6)  # 0.3 mrad
    fov = optics.field_of_view(640, 12e-6)
    assert fov == pytest.approx(2.0 * math.atan(640 * 12e-6 / (2 * 0.040)))


# ------------------------------------------------------------- NETD scaling

def test_netd_dstar_scaling():
    """Lloyd NETD scales as (4F^2+1) and 1/D*."""
    kwargs = dict(
        pixel_pitch=12e-6,
        band_exitance_contrast=2.6,
        noise_bandwidth=15.0,
        optics_transmittance=0.9,
    )
    d_star = dstar_from_jones(1e9)
    netd_f1 = netd_from_dstar(1.0, d_star, **kwargs)
    netd_f2 = netd_from_dstar(2.0, d_star, **kwargs)
    assert netd_f2 / netd_f1 == pytest.approx(17.0 / 5.0)
    assert netd_from_dstar(1.0, 2 * d_star, **kwargs) == pytest.approx(netd_f1 / 2)
    # Typical uncooled LWIR configuration lands in the tens of millikelvin.
    assert 0.02 < netd_f1 < 0.15


def test_photon_chain_netd_reasonable():
    """Cooled MWIR chain: well fill sane, NETD in the 10-30 mK class."""
    chain = PhotonDetectorChain(
        optics=Optics(focal_length=0.048, f_number=2.5, transmittance=0.85),
        fpa=Fpa(pixel_pitch=15e-6, n_columns=640, n_rows=512,
                quantum_efficiency=0.7, well_capacity=7e6, read_noise=300.0),
        band=MWIR,
        integration_time=5e-3,
        scene_temperature=300.0,
    )
    assert 0.1 < chain.well_fill() < 0.9
    assert 0.005 < chain.netd() < 0.030
    # Shot-noise-limited: doubling integration time improves NETD ~sqrt(2).
    longer = PhotonDetectorChain(
        optics=chain.optics, fpa=chain.fpa, band=chain.band,
        integration_time=10e-3,
    )
    assert longer.netd() < chain.netd()
    assert longer.netd() / chain.netd() == pytest.approx(1 / math.sqrt(2), rel=0.05)
    # SNR is linear in dT and consistent with NETD.
    assert chain.snr(chain.netd()) == pytest.approx(1.0)


# -------------------------------------------------------- range performance

def test_sampling_range_relations():
    f = focal_length_for_task(0.75, 400.0, 12e-6, JOHNSON_N50["recognize"])
    assert f == pytest.approx(0.0384)
    # Round-trip: that focal length gives back the 400 m recognition range.
    assert sampling_limited_range(0.75, f, 12e-6, 3.0) == pytest.approx(400.0)
    assert cycles_on_target(0.75, f, 12e-6, 400.0) == pytest.approx(3.0)
    # Range scales linearly with focal length, inversely with pitch and N50.
    assert sampling_limited_range(0.75, 2 * f, 12e-6, 3.0) == pytest.approx(800.0)


def test_contrast_limited_range():
    atm = CLEAR_SEA_LEVEL["LWIR"]
    r = contrast_limited_range(2.0, 0.05, atm, snr_required=2.5)
    expected = 1e3 * math.log(2.0 / 0.125) / 0.17
    assert r == pytest.approx(expected)
    # Impossible task (needed contrast > available) -> zero range.
    assert contrast_limited_range(0.1, 0.05, atm, snr_required=2.5) == 0.0
