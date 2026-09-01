"""Multi-band (ratio) pyrometry with a visible/NIR silicon camera.

The measurement: a camera behind narrowband filters images an incandescent
surface (or luminous soot); each band's signal samples the Planck curve, and
temperature is inferred from the signals.

Single band ("brightness pyrometry"): signal S = K eps(lam) L(lam, T) with
K an absolute radiometric chain constant. T follows only if eps and K are
known and stable; every multiplicative unknown (emissivity, window fouling,
partial pixel fill, vignetting) reads as a temperature error:

    dT = (lam T^2 / c2) * (dS/S).

Two bands ("ratio pyrometry"): the per-pixel ratio R = S1/S2 cancels all
common-mode multiplicative factors. In the Wien limit (excellent below
1 um for T < 3600 K),

    ln R = ln(eps1/eps2) + ln(k1/k2) + 5 ln(lam2/lam1) - (c2/T)(1/lam1 - 1/lam2)

so with a gray body (eps1 = eps2) and one calibration constant, 1/T is
linear in ln R. Two measurements determine the two unknowns (T and the
combined scale factor) exactly -- the "two-point Planck fit". Noise and
non-gray emissivity both propagate through the *equivalent wavelength*

    lam_eq = 1 / (1/lam1 - 1/lam2) = lam1 lam2 / (lam2 - lam1),

    sigma_T = (lam_eq T^2 / c2) * sigma_R/R,
    T_bias  ~ (lam_eq T^2 / c2) * ln(eps1/eps2):

widely spaced bands (small lam_eq) suppress BOTH noise amplification and
gray-assumption bias per unit of eps mismatch -- but widen the wavelength
span over which eps must be modelled and the optics stay achromatic.
All numerics below use full Planck integrals over the actual filter and
QE curves, not the Wien/center-wavelength approximation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import brentq

from . import planck
from .constants import C2

__all__ = [
    "NotchFilter",
    "PyroCamera",
    "RatioPyrometer",
    "silicon_qe",
    "electron_rate",
    "exposure_for_well_fill",
    "single_band_temperature_error",
    "ratio_temperature_error_wien",
    "equivalent_wavelength",
    "PLUME_EMISSION_LINES",
]

# Typical front-illuminated CMOS quantum efficiency (fraction) vs wavelength.
_SI_QE_LAM = np.array([350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
                       850, 900, 950, 1000, 1050, 1100]) * 1e-9
_SI_QE = np.array([0.20, 0.45, 0.60, 0.70, 0.75, 0.75, 0.72, 0.65, 0.55,
                   0.45, 0.35, 0.25, 0.15, 0.07, 0.02, 0.0])


def silicon_qe(lam):
    """Representative silicon CMOS quantum efficiency vs wavelength [m]."""
    return np.interp(lam, _SI_QE_LAM, _SI_QE, left=0.0, right=0.0)


#: Common plume/flame emission features to keep notch filters AWAY from
#: (thermal continuum pyrometry assumes the band contains no line emission).
#: Wavelengths in metres; H2O values are NIR band centres, not lines.
PLUME_EMISSION_LINES = {
    "CH": (431.4e-9,),
    "C2 Swan": (473.7e-9, 516.5e-9, 563.5e-9),
    "Na D": (589.0e-9, 589.6e-9),
    "H-alpha": (656.3e-9,),
    "Li": (670.8e-9,),
    "K": (766.5e-9, 769.9e-9),
    "H2O bands": (720e-9, 820e-9, 940e-9),
}


@dataclass(frozen=True)
class NotchFilter:
    """Ideal top-hat bandpass filter: centre, full width, peak transmission.

    Real interference filters also need deep out-of-band blocking
    (OD >= 4-5 over the full silicon response); see the analysis document.
    """

    center: float
    width: float
    peak_transmission: float = 0.9

    @property
    def lam_min(self) -> float:
        return self.center - 0.5 * self.width

    @property
    def lam_max(self) -> float:
        return self.center + 0.5 * self.width


@dataclass(frozen=True)
class PyroCamera:
    """Silicon camera + objective for pyrometry."""

    pixel_pitch: float = 5e-6
    f_number: float = 4.0
    optics_transmittance: float = 0.85  # objective + standoff window, excl. filter
    quantum_efficiency: Callable = field(default=silicon_qe)
    read_noise: float = 5.0  # electrons rms
    well_capacity: float = 20e3  # electrons

    @property
    def pixel_etendue(self) -> float:
        """A_pix * Omega [m^2 sr]."""
        return self.pixel_pitch**2 * np.pi / (4.0 * self.f_number**2 + 1.0)


def _as_eps_fn(emissivity) -> Callable:
    if callable(emissivity):
        return emissivity
    return lambda lam: np.full_like(np.asarray(lam, dtype=float), float(emissivity))


def electron_rate(temperature, filt: NotchFilter, camera: PyroCamera,
                  emissivity=1.0, n_lam: int = 101):
    """Photoelectrons per second from a pixel-filling source at temperature.

    Integrates QE(lam) * filter * eps(lam) * photon radiance over the band.
    """
    eps = _as_eps_fn(emissivity)
    lam = np.linspace(filt.lam_min, filt.lam_max, n_lam)
    temperature = np.asarray(temperature, dtype=float)
    lam_b = lam.reshape((-1,) + (1,) * temperature.ndim)
    integrand = (
        camera.quantum_efficiency(lam_b)
        * filt.peak_transmission
        * eps(lam_b)
        * planck.spectral_photon_radiance(lam_b, temperature)
    )
    rate = np.trapezoid(integrand, lam, axis=0) * camera.pixel_etendue \
        * camera.optics_transmittance
    return rate.item() if np.ndim(rate) == 0 else rate


def exposure_for_well_fill(temperature, filt: NotchFilter, camera: PyroCamera,
                           fill: float = 0.6, emissivity=1.0) -> float:
    """Exposure time [s] that fills the well to `fill` at the given scene T."""
    return fill * camera.well_capacity / electron_rate(temperature, filt, camera,
                                                       emissivity)


def equivalent_wavelength(lam1: float, lam2: float) -> float:
    """Ratio-pyrometry equivalent wavelength lam1*lam2/(lam2-lam1) [m]."""
    return lam1 * lam2 / (lam2 - lam1)


def single_band_temperature_error(temperature: float, lam: float,
                                  relative_signal_error: float) -> float:
    """Brightness-pyrometry error dT for a fractional signal/emissivity error."""
    return lam * temperature**2 / C2 * relative_signal_error


def ratio_temperature_error_wien(temperature: float, lam1: float, lam2: float,
                                 relative_ratio_error: float) -> float:
    """Ratio-pyrometry error dT for a fractional ratio error (Wien limit)."""
    return equivalent_wavelength(lam1, lam2) * temperature**2 / C2 \
        * relative_ratio_error


@dataclass(frozen=True)
class RatioPyrometer:
    """Two-band ratio pyrometer: short-wavelength and long-wavelength notches."""

    filter_short: NotchFilter
    filter_long: NotchFilter
    camera: PyroCamera = field(default_factory=PyroCamera)

    def __post_init__(self):
        if self.filter_short.center >= self.filter_long.center:
            raise ValueError("filter_short must have the shorter centre wavelength")

    @property
    def equivalent_wavelength(self) -> float:
        return equivalent_wavelength(self.filter_short.center,
                                     self.filter_long.center)

    def ratio(self, temperature, emissivity=1.0):
        """Signal ratio short/long at equal exposure (monotonic in T)."""
        return (
            electron_rate(temperature, self.filter_short, self.camera, emissivity)
            / electron_rate(temperature, self.filter_long, self.camera, emissivity)
        )

    def temperature_from_ratio(self, ratio: float, t_lo: float = 400.0,
                               t_hi: float = 6000.0) -> float:
        """Invert the (calibrated) ratio to temperature [K]."""
        return brentq(lambda t: self.ratio(t) - ratio, t_lo, t_hi, xtol=1e-3)

    def dlnratio_dT(self, temperature: float) -> float:
        """d(ln R)/dT, computed from the full band integrals."""
        dt = max(0.01, 1e-4 * temperature)
        return (np.log(self.ratio(temperature + dt))
                - np.log(self.ratio(temperature - dt))) / (2.0 * dt)

    def sigma_T(self, temperature: float, exposure_short: float,
                exposure_long: float, binning: int = 1, frames: int = 1) -> float:
        """Shot + read noise NEdT of the ratio measurement [K].

        `binning` x `binning` spatial pixels and `frames` video frames are
        averaged; signals add, read-noise variances add.
        """
        n_avg = binning**2 * frames
        rel_var = 0.0
        for filt, t_exp in ((self.filter_short, exposure_short),
                            (self.filter_long, exposure_long)):
            n_e = electron_rate(temperature, filt, self.camera) * t_exp * n_avg
            var = n_e + n_avg * self.camera.read_noise**2
            rel_var += var / n_e**2
        return np.sqrt(rel_var) / abs(self.dlnratio_dT(temperature))

    def emissivity_bias(self, temperature: float, eps_ratio: float) -> float:
        """Temperature bias [K] if the true eps_short/eps_long is `eps_ratio`
        but the instrument assumes a gray body (ratio calibrated with
        eps_ratio = 1)."""
        measured = eps_ratio * self.ratio(temperature)
        return self.temperature_from_ratio(measured) - temperature
