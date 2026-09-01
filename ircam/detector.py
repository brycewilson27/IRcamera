"""Focal-plane-array description and detector-level figures of merit.

Two sensitivity routes are provided:

1. ``netd_from_dstar`` -- the classic D*-based NETD expression (Lloyd,
   *Thermal Imaging Systems*; Rogalski, *Infrared Detectors*), the natural
   description for uncooled microbolometers whose noise is characterised
   by a specific detectivity D*:

       NETD = (4 F#^2 + 1) sqrt(df) / (sqrt(A_d) tau_o D* (dM/dT))

   with A_d the pixel area, df the noise-equivalent bandwidth, tau_o the
   optics transmittance and dM/dT the in-band exitance contrast of the
   scene. (Many texts write 4 F#^2; the +1 is the exact pupil solid angle.)

2. A photon-counting chain (see :mod:`ircam.radiometry`) for cooled
   photon detectors, where NETD follows from shot noise on integrated
   photoelectrons -- the appropriate description for InSb / HgCdTe / T2SL
   FPAs that are background- or well-capacity-limited.

D* here is in SI units, m Hz^0.5 / W. Vendor datasheets quote
"Jones" = cm Hz^0.5 / W; use :func:`dstar_from_jones`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import C, H


def dstar_from_jones(d_star_jones: float) -> float:
    """Convert D* from Jones (cm Hz^0.5/W) to SI (m Hz^0.5/W)."""
    return d_star_jones * 1e-2


@dataclass(frozen=True)
class Fpa:
    """A staring focal-plane array.

    Photon-detector parameters (quantum_efficiency, well_capacity,
    read_noise, dark_current) are used by the photon-counting chain;
    d_star/bandwidth describe a thermal (bolometer) detector.
    """

    pixel_pitch: float
    n_columns: int
    n_rows: int
    quantum_efficiency: float = 0.7
    well_capacity: float = 7e6  # electrons
    read_noise: float = 300.0  # electrons rms
    dark_current: float = 0.0  # electrons/s per pixel
    d_star: float | None = None  # m Hz^0.5 / W (SI)
    noise_bandwidth: float | None = None  # Hz

    @property
    def pixel_area(self) -> float:
        """Pixel area [m^2] (100% fill factor assumed unless folded into QE)."""
        return self.pixel_pitch**2

    @property
    def nyquist_frequency(self) -> float:
        """Spatial Nyquist frequency 1/(2 p) [cycles/m at focal plane]."""
        return 1.0 / (2.0 * self.pixel_pitch)


def netd_from_dstar(
    f_number: float,
    d_star: float,
    pixel_pitch: float,
    band_exitance_contrast: float,
    noise_bandwidth: float,
    optics_transmittance: float = 0.9,
) -> float:
    """NETD [K] for a D*-characterised detector (classic Lloyd formula).

    Parameters
    ----------
    d_star : specific detectivity in SI units [m Hz^0.5/W].
    band_exitance_contrast : dM/dT of the scene in the sensor band [W m^-2 K^-1].
    noise_bandwidth : noise-equivalent electrical bandwidth [Hz]; for a
        staring array integrating for t_int, df ~ 1/(2 t_int).
    """
    a_d = pixel_pitch**2
    return ((4.0 * f_number**2 + 1.0) * math.sqrt(noise_bandwidth)) / (
        math.sqrt(a_d) * optics_transmittance * d_star * band_exitance_contrast
    )


def blip_dstar(lam: float, quantum_efficiency: float, background_photon_irradiance: float) -> float:
    """Background-limited (BLIP) D* [m Hz^0.5/W] for a photovoltaic detector.

        D*_BLIP = (lam / h c) sqrt(eta / (2 E_q))

    where E_q is the background photon irradiance on the detector
    [photons s^-1 m^-2]. With a cold shield matched to the optics,
    E_q = pi L_q / (4 F#^2 + 1) from the scene band photon radiance L_q.
    """
    return (lam / (H * C)) * math.sqrt(
        quantum_efficiency / (2.0 * background_photon_irradiance)
    )
