"""End-to-end radiometric signal chain for a photon-detector camera.

Scene (blackbody at T) -> atmosphere -> optics -> pixel -> photoelectrons.

The flux collected by one pixel staring at an extended (pixel-filling)
Lambertian scene of in-band photon radiance L_q is

    Phi_q = L_q * A_pix * Omega_pix * tau_atm * tau_opt   [photons/s]

with Omega_pix = pi / (4 F#^2 + 1) the pupil solid angle seen from the
focal plane. A cold shield matched to the optics F-number is assumed, so
out-of-cone background is negligible; optics self-emission is neglected
(valid for cooled or low-loss optics -- flag for refinement).

Noise model: shot noise on scene + dark electrons, plus read noise, in
quadrature. NETD is the temperature step that equals one noise sigma:

    NETD = sigma_n / (dN_e/dT).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from . import planck
from .atmosphere import SimpleAtmosphere
from .bands import Band
from .detector import Fpa
from .optics import Optics


@dataclass(frozen=True)
class PhotonDetectorChain:
    """Radiometric chain for a cooled photon-detector camera."""

    optics: Optics
    fpa: Fpa
    band: Band
    integration_time: float = 5e-3
    scene_temperature: float = 300.0
    atmosphere: SimpleAtmosphere | None = None
    range_m: float = 0.0

    def _path_transmittance(self) -> float:
        if self.atmosphere is None or self.range_m <= 0.0:
            return 1.0
        return float(self.atmosphere.transmittance(self.range_m))

    def photon_rate(self, temperature: float | None = None) -> float:
        """Scene photons per second reaching one pixel."""
        t = self.scene_temperature if temperature is None else temperature
        radiance_q = planck.band_photon_radiance(t, self.band.lam1, self.band.lam2)
        return (
            radiance_q
            * self.fpa.pixel_area
            * self.optics.pixel_solid_angle
            * self.optics.transmittance
            * self._path_transmittance()
        )

    def signal_electrons(self, temperature: float | None = None) -> float:
        """Photoelectrons integrated from the scene in one frame."""
        return (
            self.fpa.quantum_efficiency * self.photon_rate(temperature) * self.integration_time
        )

    def total_electrons(self, temperature: float | None = None) -> float:
        """Scene plus dark electrons integrated in one frame."""
        return self.signal_electrons(temperature) + self.fpa.dark_current * self.integration_time

    def well_fill(self, temperature: float | None = None) -> float:
        """Fraction of well capacity used."""
        return self.total_electrons(temperature) / self.fpa.well_capacity

    def max_integration_time(self, target_fill: float = 0.5) -> float:
        """Integration time that fills the well to target_fill at the scene T."""
        rate = (
            self.fpa.quantum_efficiency * self.photon_rate() + self.fpa.dark_current
        )
        return target_fill * self.fpa.well_capacity / rate

    def electrons_per_kelvin(self, temperature: float | None = None) -> float:
        """dN_e/dT: change in integrated electrons per kelvin of scene temperature."""
        t = self.scene_temperature if temperature is None else temperature
        contrast_q = planck.band_photon_radiance_dT(t, self.band.lam1, self.band.lam2)
        return (
            contrast_q
            * self.fpa.pixel_area
            * self.optics.pixel_solid_angle
            * self.optics.transmittance
            * self._path_transmittance()
            * self.fpa.quantum_efficiency
            * self.integration_time
        )

    def noise_electrons(self, temperature: float | None = None) -> float:
        """RMS noise electrons: shot (scene + dark) plus read, in quadrature."""
        return math.sqrt(self.total_electrons(temperature) + self.fpa.read_noise**2)

    def netd(self, temperature: float | None = None) -> float:
        """Noise-equivalent temperature difference [K]."""
        return self.noise_electrons(temperature) / self.electrons_per_kelvin(temperature)

    def snr(self, delta_t: float, temperature: float | None = None) -> float:
        """SNR for a small scene temperature difference delta_t [K]."""
        return delta_t * self.electrons_per_kelvin(temperature) / self.noise_electrons(temperature)
