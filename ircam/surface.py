"""Viewing geometry and reflected sunlight for surface pyrometry.

Directional emissivity
----------------------
For an optically smooth, opaque medium with complex refractive index
m = n + i k viewed from vacuum at angle theta from the surface normal, the
Fresnel reflectances are

    r_s = (cos th - m cos th_t) / (cos th + m cos th_t)
    r_p = (m cos th - cos th_t) / (m cos th + cos th_t),   cos th_t = sqrt(1 - (sin th / m)^2)

and Kirchhoff's law for an opaque body gives the unpolarised directional
emissivity eps(th) = 1 - (|r_s|^2 + |r_p|^2) / 2. Rough surfaces are more
Lambertian than this smooth-surface model up to fairly grazing angles, so
the Fresnel curve is a worst case for the angular collapse. The optical
constants shipped here are ILLUSTRATIVE class values, not measurements of
any nozzle material; a coupon measurement replaces them.

Reflected sunlight
------------------
Diffuse approximation: a surface with directional-hemispherical
reflectance rho(th_v) = 1 - eps(th_v) under collimated solar irradiance
E_sun(lam) at incidence th_s reflects radiance

    L_refl(lam) = rho(th_v) E_sun(lam) cos(th_s) / pi.

This is an additive term in each band. The specular glint (the sun's
image reflected into the camera) is treated separately as a hazard ratio
because it exceeds the radiance of a 3000 C surface.

Solar spectral irradiance is a coarse ASTM G173-03 global-tilt (AM1.5G)
table, 400-1000 nm, accurate to roughly 10% and smoothing over the narrow
O2 and H2O absorption features.

Plume illumination
------------------
A luminous plume (soot or particle laden, radiating as a gray body of
emissivity eps_pl at temperature T_pl) illuminates the wall. A diffuse
wall element that sees the plume over a cosine-weighted hemisphere
fraction F (the view factor) reflects

    L_refl(lam) = rho(th_v) F eps_pl(lam) L_bb(lam, T_pl),

and the same plume radiance also reaches every pixel through veiling
glare in the optics, L_glare = g eps_pl L_bb(T_pl), with g the veiling
glare index times the plume's area fraction in the field. Both are
additive; unlike sunlight they exist only during the burn, so they cannot
be captured by a pre-ignition frame -- the plume radiance must be measured
from plume pixels in the same image and the view factor modelled. Clean
(non-sooty) plumes have band emissivities of order 1e-3 to 1e-2 in the
notches; particle-laden plumes reach 0.1 to 0.9.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from . import planck
from .constants import C, H
from .pyrometry import NotchFilter, PyroCamera, RatioPyrometer, electron_rate

__all__ = [
    "fresnel_reflectances",
    "directional_emissivity",
    "OpticalConstants",
    "MATERIALS",
    "solar_spectral_irradiance",
    "SUN_SOLID_ANGLE",
    "thermal_electron_rate",
    "solar_reflected_electron_rate",
    "one_band_apparent_temperature",
    "ratio_apparent_temperature",
    "specular_glint_ratio",
    "foreshortening",
    "PlumeSource",
    "plume_reflected_electron_rate",
]

#: Solid angle of the solar disc [sr].
SUN_SOLID_ANGLE = 6.8e-5


def fresnel_reflectances(theta, n, k):
    """Unpolarised-component reflectances (R_s, R_p) at angle theta [rad]."""
    theta = np.asarray(theta, dtype=float)
    m = np.asarray(n, dtype=float) + 1j * np.asarray(k, dtype=float)
    cos_i = np.cos(theta)
    cos_t = np.sqrt(1.0 - (np.sin(theta) / m) ** 2)
    r_s = (cos_i - m * cos_t) / (cos_i + m * cos_t)
    r_p = (m * cos_i - cos_t) / (m * cos_i + cos_t)
    return np.abs(r_s) ** 2, np.abs(r_p) ** 2


def directional_emissivity(theta, n, k):
    """Unpolarised directional emissivity of an opaque smooth surface."""
    r_s, r_p = fresnel_reflectances(theta, n, k)
    return 1.0 - 0.5 * (r_s + r_p)


@dataclass(frozen=True)
class OpticalConstants:
    """Complex refractive index tabulated at a few wavelengths [m]."""

    name: str
    lam: tuple
    n: tuple
    k: tuple
    note: str = ""

    def nk(self, lam):
        lam = np.asarray(lam, dtype=float)
        return np.interp(lam, self.lam, self.n), np.interp(lam, self.lam, self.k)

    def emissivity(self, theta, lam):
        """Directional emissivity at viewing angle theta [rad], wavelength(s) lam."""
        n, k = self.nk(lam)
        return directional_emissivity(theta, n, k)


#: Illustrative class values at 620 and 870 nm (see module docstring).
MATERIALS = {
    "oxide-like dielectric": OpticalConstants(
        "oxide-like dielectric", (620e-9, 870e-9), (1.7, 1.7), (0.1, 0.1),
        "flat to ~60 deg, then collapses; angular factor identical in both bands"),
    "graphite-like": OpticalConstants(
        "graphite-like", (620e-9, 870e-9), (2.7, 2.9), (1.4, 1.6),
        "semi-metal; angular factor differs slightly between bands"),
    "tungsten-like metal": OpticalConstants(
        "tungsten-like metal", (620e-9, 870e-9), (3.6, 3.3), (2.9, 3.2),
        "emissivity rises toward 75 deg before collapsing"),
}

# ASTM G173-03 global tilt, approximate [nm, W m^-2 nm^-1].
_SOLAR_NM = np.array([400, 425, 450, 475, 500, 525, 550, 575, 600, 625, 650,
                      675, 700, 725, 750, 775, 800, 825, 850, 875, 900, 925,
                      950, 975, 1000], dtype=float)
_SOLAR_W_M2_NM = np.array([1.12, 1.45, 1.60, 1.66, 1.55, 1.60, 1.53, 1.50, 1.47,
                           1.44, 1.44, 1.38, 1.36, 1.20, 1.24, 1.10, 1.07, 0.95,
                           0.95, 0.90, 0.75, 0.55, 0.35, 0.55, 0.72])


def solar_spectral_irradiance(lam):
    """Approximate AM1.5G solar spectral irradiance [W m^-2 m^-1] at lam [m]."""
    lam_nm = np.asarray(lam, dtype=float) * 1e9
    return np.interp(lam_nm, _SOLAR_NM, _SOLAR_W_M2_NM, left=0.0, right=0.0) * 1e9


def foreshortening(theta):
    """Pixel-footprint stretch along the line of sight, 1/cos(theta)."""
    return 1.0 / np.cos(theta)


def thermal_electron_rate(temperature, filt: NotchFilter, cam: PyroCamera,
                          material: OpticalConstants, theta_v: float):
    """Emitted-signal electron rate with directional emissivity at theta_v."""
    return electron_rate(temperature, filt, cam,
                         emissivity=lambda lam: material.emissivity(theta_v, lam))


def solar_reflected_electron_rate(filt: NotchFilter, cam: PyroCamera,
                                  material: OpticalConstants, theta_v: float,
                                  theta_s: float, sun_factor: float = 1.0,
                                  n_lam: int = 101) -> float:
    """Electron rate from diffusely reflected sunlight [e-/s].

    sun_factor scales the AM1.5G table (0 = night or shaded, 1 = full sun).
    """
    cos_s = max(math.cos(theta_s), 0.0)
    if sun_factor <= 0.0 or cos_s <= 0.0:
        return 0.0
    lam = np.linspace(filt.lam_min, filt.lam_max, n_lam)
    rho = 1.0 - material.emissivity(theta_v, lam)
    radiance = rho * solar_spectral_irradiance(lam) * sun_factor * cos_s / math.pi
    photon_radiance = radiance * lam / (H * C)
    integrand = cam.quantum_efficiency(lam) * filt.peak_transmission * photon_radiance
    return float(np.trapezoid(integrand, lam) * cam.pixel_etendue
                 * cam.optics_transmittance)


@dataclass(frozen=True)
class PlumeSource:
    """Luminous plume illuminating the wall and the optics.

    emissivity is the plume band emissivity at lam_ref; alpha gives its
    spectral slope, eps(lam) = emissivity (lam_ref/lam)^alpha (0 = gray,
    ~1 = Rayleigh soot). view_factor is the cosine-weighted fraction of the
    wall element's hemisphere filled by plume; stray_light is the veiling
    glare index times the plume's area fraction in the field; residual is
    the fraction left after an in-frame subtraction.
    """

    temperature: float
    emissivity: float = 0.0
    alpha: float = 0.0
    lam_ref: float = 870e-9
    view_factor: float = 0.2
    stray_light: float = 0.0
    residual: float = 1.0

    def band_emissivity(self, lam):
        return self.emissivity * (self.lam_ref / np.asarray(lam, dtype=float)) ** self.alpha

    def radiance(self, lam):
        """Plume spectral radiance [W m^-2 sr^-1 m^-1]."""
        return self.band_emissivity(lam) * planck.spectral_radiance(lam, self.temperature)


def plume_reflected_electron_rate(filt: NotchFilter, cam: PyroCamera,
                                  material: OpticalConstants, theta_v: float,
                                  plume: PlumeSource | None, n_lam: int = 101) -> float:
    """Electron rate from plume light reflected by the wall plus veiling glare."""
    if plume is None or plume.emissivity <= 0.0 or plume.residual <= 0.0:
        return 0.0
    lam = np.linspace(filt.lam_min, filt.lam_max, n_lam)
    rho = 1.0 - material.emissivity(theta_v, lam)
    radiance = ((rho * plume.view_factor + plume.stray_light) * plume.radiance(lam)
                * plume.residual)
    photon_radiance = radiance * lam / (H * C)
    integrand = cam.quantum_efficiency(lam) * filt.peak_transmission * photon_radiance
    return float(np.trapezoid(integrand, lam) * cam.pixel_etendue
                 * cam.optics_transmittance)


def one_band_apparent_temperature(t_true: float, filt: NotchFilter, cam: PyroCamera,
                                  material: OpticalConstants, theta_v: float,
                                  theta_s: float = 0.0, sun_factor: float = 0.0,
                                  cal_theta: float = 0.0,
                                  sun_residual: float = 1.0,
                                  plume: PlumeSource | None = None) -> float:
    """Temperature a one-band instrument reports.

    The instrument is calibrated with the material's emissivity at
    ``cal_theta`` (a normal-incidence coupon by default). ``sun_residual``
    is the fraction of the reflected-sun signal left after any background
    subtraction (1 = none, 0.1 = a pre-ignition frame removed 90%).
    """
    s_meas = (thermal_electron_rate(t_true, filt, cam, material, theta_v)
              + sun_residual * solar_reflected_electron_rate(
                  filt, cam, material, theta_v, theta_s, sun_factor)
              + plume_reflected_electron_rate(filt, cam, material, theta_v, plume))

    def eps_cal(lam):
        return material.emissivity(cal_theta, lam)

    return brentq(lambda t: electron_rate(t, filt, cam, emissivity=eps_cal) - s_meas,
                  300.0, 20000.0, xtol=1e-3)


def ratio_apparent_temperature(t_true: float, pyro: RatioPyrometer,
                               material: OpticalConstants, theta_v: float,
                               theta_s: float = 0.0, sun_factor: float = 0.0,
                               cal_theta: float = 0.0,
                               sun_residual: float = 1.0,
                               plume: PlumeSource | None = None) -> float:
    """Temperature a two-band ratio instrument reports.

    Calibration absorbs eps_1/eps_2 at ``cal_theta``; only the angular
    change of that ratio and the additive sun term survive.
    """
    cam = pyro.camera
    signals = []
    for filt in (pyro.filter_short, pyro.filter_long):
        signals.append(thermal_electron_rate(t_true, filt, cam, material, theta_v)
                       + sun_residual * solar_reflected_electron_rate(
                           filt, cam, material, theta_v, theta_s, sun_factor)
                       + plume_reflected_electron_rate(filt, cam, material, theta_v,
                                                       plume))
    r_meas = signals[0] / signals[1]

    def eps_cal(lam):
        return material.emissivity(cal_theta, lam)

    def r_cal(t):
        return (electron_rate(t, pyro.filter_short, cam, emissivity=eps_cal)
                / electron_rate(t, pyro.filter_long, cam, emissivity=eps_cal))

    return brentq(lambda t: r_cal(t) - r_meas, 300.0, 20000.0, xtol=1e-3)


def specular_glint_ratio(theta_v: float, lam: float, temperature: float,
                         material: OpticalConstants, sun_factor: float = 1.0) -> float:
    """Radiance of the sun's specular image relative to the surface's own
    thermal radiance at wavelength lam (a hazard ratio; > 1 means the glint
    outshines the surface)."""
    eps = float(material.emissivity(theta_v, lam))
    sun_radiance = float(solar_spectral_irradiance(lam)) * sun_factor / SUN_SOLID_ANGLE
    return (1.0 - eps) * sun_radiance / (eps * float(planck.spectral_radiance(lam, temperature)))
