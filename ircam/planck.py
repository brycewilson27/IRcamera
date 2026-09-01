"""Blackbody radiometry: Planck's law, band integrals, thermal derivatives.

All quantities are SI: wavelengths in metres, temperatures in kelvin.
Spectral radiance is W m^-2 sr^-1 m^-1 (per metre of wavelength);
band-integrated radiance is W m^-2 sr^-1. Photon quantities replace watts
with photons per second.

Numerical notes
---------------
The Planck factor 1/(exp(x) - 1) is evaluated as exp(-x)/(1 - exp(-x))
via ``expm1`` so it neither overflows for short-wavelength/cold scenes
(x >> 1) nor loses precision in the Rayleigh-Jeans limit (x << 1).

Band integrals use Gauss-Legendre quadrature in ln(wavelength), which is
essentially exact for these smooth integrands at the default order.
"""

from __future__ import annotations

import numpy as np

from .constants import C, C1L, C2, SIGMA, WIEN_B

__all__ = [
    "spectral_radiance",
    "spectral_radiance_dT",
    "spectral_photon_radiance",
    "spectral_photon_radiance_dT",
    "band_radiance",
    "band_radiance_dT",
    "band_photon_radiance",
    "band_photon_radiance_dT",
    "band_exitance",
    "band_exitance_dT",
    "blackbody_fraction",
    "total_radiance",
    "total_exitance",
    "wien_peak_wavelength",
]


def _planck_factor(x):
    """Stable 1/(exp(x) - 1) for x > 0."""
    return np.exp(-x) / (-np.expm1(-x))


def spectral_radiance(lam, temperature):
    """Planck spectral radiance L_lambda [W m^-2 sr^-1 m^-1]."""
    lam = np.asarray(lam, dtype=float)
    x = C2 / (lam * np.asarray(temperature, dtype=float))
    return C1L / lam**5 * _planck_factor(x)


def spectral_radiance_dT(lam, temperature):
    """Analytic dL_lambda/dT [W m^-2 sr^-1 m^-1 K^-1].

    dL/dT = L * (x/T) * exp(x)/(exp(x)-1) with x = c2/(lam*T).
    """
    lam = np.asarray(lam, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    x = C2 / (lam * temperature)
    # exp(x)/(exp(x)-1) = 1/(1-exp(-x))
    return spectral_radiance(lam, temperature) * (x / temperature) / (-np.expm1(-x))


def spectral_photon_radiance(lam, temperature):
    """Photon spectral radiance L_q,lambda [photons s^-1 m^-2 sr^-1 m^-1]."""
    lam = np.asarray(lam, dtype=float)
    x = C2 / (lam * np.asarray(temperature, dtype=float))
    return 2.0 * C / lam**4 * _planck_factor(x)


def spectral_photon_radiance_dT(lam, temperature):
    """Analytic dL_q,lambda/dT [photons s^-1 m^-2 sr^-1 m^-1 K^-1]."""
    lam = np.asarray(lam, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    x = C2 / (lam * temperature)
    return (
        spectral_photon_radiance(lam, temperature)
        * (x / temperature)
        / (-np.expm1(-x))
    )


def _log_lambda_nodes(lam1, lam2, order):
    """Gauss-Legendre nodes/weights for integrating f(lam) dlam over [lam1, lam2]
    with the substitution u = ln(lam)."""
    if not (0.0 < lam1 < lam2):
        raise ValueError("require 0 < lam1 < lam2")
    x, w = np.polynomial.legendre.leggauss(order)
    u1, u2 = np.log(lam1), np.log(lam2)
    u = 0.5 * (u2 - u1) * x + 0.5 * (u2 + u1)
    lam = np.exp(u)
    weights = w * 0.5 * (u2 - u1) * lam  # dlam = lam du
    return lam, weights


def _band_integral(spectral_fn, temperature, lam1, lam2, order):
    lam, weights = _log_lambda_nodes(lam1, lam2, order)
    temperature = np.asarray(temperature, dtype=float)
    lam = lam.reshape((-1,) + (1,) * temperature.ndim)
    weights = weights.reshape(lam.shape)
    values = np.sum(weights * spectral_fn(lam, temperature), axis=0)
    return values.item() if np.ndim(values) == 0 else values


def band_radiance(temperature, lam1, lam2, order=128):
    """In-band radiance L(lam1->lam2, T) [W m^-2 sr^-1]."""
    return _band_integral(spectral_radiance, temperature, lam1, lam2, order)


def band_radiance_dT(temperature, lam1, lam2, order=128):
    """In-band thermal contrast dL/dT [W m^-2 sr^-1 K^-1]."""
    return _band_integral(spectral_radiance_dT, temperature, lam1, lam2, order)


def band_photon_radiance(temperature, lam1, lam2, order=128):
    """In-band photon radiance [photons s^-1 m^-2 sr^-1]."""
    return _band_integral(spectral_photon_radiance, temperature, lam1, lam2, order)


def band_photon_radiance_dT(temperature, lam1, lam2, order=128):
    """In-band photon contrast dL_q/dT [photons s^-1 m^-2 sr^-1 K^-1]."""
    return _band_integral(spectral_photon_radiance_dT, temperature, lam1, lam2, order)


def band_exitance(temperature, lam1, lam2, order=128):
    """In-band exitance M = pi * L for a Lambertian blackbody [W m^-2]."""
    return np.pi * band_radiance(temperature, lam1, lam2, order)


def band_exitance_dT(temperature, lam1, lam2, order=128):
    """In-band exitance derivative dM/dT [W m^-2 K^-1]."""
    return np.pi * band_radiance_dT(temperature, lam1, lam2, order)


def total_exitance(temperature):
    """Stefan-Boltzmann total exitance sigma*T^4 [W m^-2]."""
    return SIGMA * np.asarray(temperature, dtype=float) ** 4


def total_radiance(temperature):
    """Total blackbody radiance sigma*T^4/pi [W m^-2 sr^-1]."""
    return total_exitance(temperature) / np.pi


def blackbody_fraction(temperature, lam1, lam2, order=128):
    """Fraction of total blackbody power emitted between lam1 and lam2."""
    return band_radiance(temperature, lam1, lam2, order) / total_radiance(temperature)


def wien_peak_wavelength(temperature):
    """Wavelength of peak spectral radiance (Wien displacement law) [m]."""
    return WIEN_B / np.asarray(temperature, dtype=float)
