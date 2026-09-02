"""Formula-by-formula verification against closed-form and textbook references.

Each test names the formula it checks and the independent reference it is
checked against. Companion document: docs/verification.md.
"""

import math

import numpy as np
import pytest
from scipy.integrate import dblquad
from scipy.optimize import brentq
from scipy.special import jn_zeros, zeta

from ircam import planck
from ircam.bands import Band
from ircam.constants import C, C1L, C2, H, KB, SIGMA, WIEN_B
from ircam.detector import Fpa, blip_dstar, netd_from_dstar
from ircam.optics import Optics
from ircam.pyrometry import (
    NotchFilter,
    PyroCamera,
    RatioPyrometer,
    equivalent_wavelength,
    ratio_temperature_error_wien,
    single_band_temperature_error,
)
from ircam.radiometry import PhotonDetectorChain


# ------------------------------------------------------------ constants

def test_radiation_constants_match_codata():
    """c1L = 2hc^2 and c2 = hc/k against CODATA 2018 values."""
    assert C1L == pytest.approx(1.191042972e-16, rel=1e-8)
    assert C2 == pytest.approx(1.438776877e-2, rel=1e-8)


def test_stefan_boltzmann_constant_identity():
    """sigma = 2 pi^5 k^4 / (15 h^3 c^2)."""
    assert SIGMA == pytest.approx(2 * math.pi**5 * KB**4 / (15 * H**3 * C**2),
                                  rel=1e-9)


def test_wien_constant_from_transcendental_root():
    """b = hc / (k x*) with x* the root of x = 5 (1 - exp(-x))."""
    x_star = brentq(lambda x: x - 5.0 * (1.0 - math.exp(-x)), 1.0, 10.0)
    assert x_star == pytest.approx(4.965114231744276, rel=1e-9)
    assert WIEN_B == pytest.approx(H * C / (KB * x_star), rel=1e-8)


# --------------------------------------------------------- Planck integrals

def test_photon_stefan_boltzmann_law():
    """Total photon exitance = 4 pi zeta(3) k^3 T^3 / (c^2 h^3) ~ 1.5205e15 T^3."""
    coeff = 4 * math.pi * zeta(3) * KB**3 / (C**2 * H**3)
    assert coeff == pytest.approx(1.5205e15, rel=2e-4)
    # The photon spectrum's Rayleigh-Jeans tail falls only as lam^-3, so the
    # integral must run to ~1 m to capture all but 1e-9 of the photons.
    for temperature in (300.0, 1773.0):
        numeric = math.pi * planck.band_photon_radiance(temperature, 1e-8, 1.0,
                                                         order=1024)
        assert numeric == pytest.approx(coeff * temperature**3, rel=1e-6)


def test_blackbody_fraction_textbook_anchors():
    """F(0 -> lambda_peak T) = 0.2501 and F(0 -> 4107 um K) = 0.5 (median)."""
    temperature = 1000.0
    below_peak = planck.blackbody_fraction(temperature, 1e-9,
                                           planck.wien_peak_wavelength(temperature))
    assert below_peak == pytest.approx(0.2501, abs=5e-4)
    median = planck.blackbody_fraction(temperature, 1e-9, 4.1072e-3 / temperature)
    assert median == pytest.approx(0.5, abs=1e-3)


def test_wien_limit_of_planck():
    """For x = c2/(lam T) >> 1, Planck -> Wien: L = c1L lam^-5 exp(-x)."""
    lam, temperature = 620e-9, 2273.0  # x ~ 10.2
    x = C2 / (lam * temperature)
    wien = C1L / lam**5 * math.exp(-x)
    assert float(planck.spectral_radiance(lam, temperature)) == \
        pytest.approx(wien * (1 + math.exp(-x)), rel=1e-6)


# ------------------------------------------------------------------ optics

def test_airy_coefficient_from_bessel_zero():
    """1.22 lam F# uses the first zero of J1: j_{1,1}/pi = 1.2197."""
    assert jn_zeros(1, 1)[0] / math.pi == pytest.approx(1.2197, abs=1e-4)
    optics = Optics(focal_length=0.05, f_number=2.0)
    assert optics.airy_radius(1e-6) == pytest.approx(
        jn_zeros(1, 1)[0] / math.pi * 1e-6 * 2.0, rel=5e-4)


@pytest.mark.parametrize("f_number", [1.0, 2.0, 4.0])
def test_projected_solid_angle_by_integration(f_number):
    """Omega = int cos(th) dOmega over a cone with tan(th_max) = 1/(2F#) = pi/(4F#^2+1)."""
    theta_max = math.atan(1.0 / (2.0 * f_number))
    omega, _ = dblquad(lambda th, ph: math.cos(th) * math.sin(th),
                       0.0, 2.0 * math.pi, 0.0, theta_max)
    assert omega == pytest.approx(math.pi / (4 * f_number**2 + 1), rel=1e-8)
    assert Optics(0.05, f_number).pixel_solid_angle == pytest.approx(omega)
    cam = PyroCamera(pixel_pitch=5e-6, f_number=f_number)
    assert cam.pixel_etendue == pytest.approx(omega * 25e-12)


# ----------------------------------------------------------- NETD routes

def test_lloyd_netd_equals_photon_chain_at_blip():
    """The D*-route (Lloyd) and the photon-counting route must agree when
    D* = D*_BLIP and df = 1/(2 t_int): this cross-checks the Lloyd formula,
    the BLIP D* expression, the pupil solid angle and the photon chain in
    one shot. A 20 nm band keeps the lam/hc energy conversion exact to
    <0.3%."""
    band = Band("narrow", 4.00e-6, 4.02e-6)
    lam_c = band.center
    optics = Optics(focal_length=0.05, f_number=2.0, transmittance=0.8)
    fpa = Fpa(pixel_pitch=15e-6, n_columns=64, n_rows=64,
              quantum_efficiency=0.7, well_capacity=1e12, read_noise=0.0)
    t_int = 1e-3
    chain = PhotonDetectorChain(optics=optics, fpa=fpa, band=band,
                                integration_time=t_int, scene_temperature=300.0)
    # Background photon irradiance on the detector through the same optics.
    photon_irradiance = chain.photon_rate() / fpa.pixel_area
    d_star = blip_dstar(lam_c, fpa.quantum_efficiency, photon_irradiance)
    dm_dt = planck.band_exitance_dT(300.0, band.lam1, band.lam2)
    lloyd = netd_from_dstar(optics.f_number, d_star, fpa.pixel_pitch, dm_dt,
                            noise_bandwidth=1.0 / (2.0 * t_int),
                            optics_transmittance=optics.transmittance)
    assert lloyd == pytest.approx(chain.netd(), rel=5e-3)


# ------------------------------------------------------ ratio pyrometry

def test_ratio_slope_converges_to_wien_for_narrow_filters():
    """d(lnR)/dT from full band integrals -> (c2/T^2)(1/l1 - 1/l2) as the
    filters narrow and x >> 1."""
    pyro = RatioPyrometer(NotchFilter(620e-9, 2e-9), NotchFilter(870e-9, 2e-9),
                          PyroCamera(quantum_efficiency=lambda lam: np.ones_like(lam)))
    temperature = 1773.0
    wien = C2 / temperature**2 * (1 / 620e-9 - 1 / 870e-9)
    assert pyro.dlnratio_dT(temperature) == pytest.approx(wien, rel=2e-3)


def test_error_laws_reduce_consistently():
    """Ratio error law with lam_eq equals the single-band law at lam = lam_eq,
    and lam_eq = l1 l2/(l2 - l1)."""
    l1, l2, temperature = 620e-9, 870e-9, 2273.0
    lam_eq = equivalent_wavelength(l1, l2)
    assert lam_eq == pytest.approx(620e-9 * 870e-9 / 250e-9)
    assert ratio_temperature_error_wien(temperature, l1, l2, 0.01) == \
        pytest.approx(single_band_temperature_error(temperature, lam_eq, 0.01))
