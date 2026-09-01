"""First-order optics for a staring infrared camera.

Conventions: focal length and pitches in metres, angles in radians.
The solid angle subtended by the exit pupil at the focal plane uses the
exact form for a circular pupil,

    Omega = pi / (4 F#^2 + 1),

which reduces to the familiar pi/(4 F#^2) for slow optics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Optics:
    """An imaging objective described by focal length, F-number, transmittance."""

    focal_length: float
    f_number: float
    transmittance: float = 0.9

    @property
    def aperture_diameter(self) -> float:
        """Entrance-pupil diameter D = f / F# [m]."""
        return self.focal_length / self.f_number

    @property
    def pixel_solid_angle(self) -> float:
        """Solid angle of the pupil seen from the focal plane [sr]."""
        return math.pi / (4.0 * self.f_number**2 + 1.0)

    def airy_radius(self, lam: float) -> float:
        """Radius of the first Airy null at the focal plane, 1.22 lam F# [m]."""
        return 1.22 * lam * self.f_number

    def diffraction_cutoff(self, lam: float) -> float:
        """Incoherent MTF cutoff frequency 1/(lam F#) [cycles/m at focal plane]."""
        return 1.0 / (lam * self.f_number)

    def ifov(self, pixel_pitch: float) -> float:
        """Instantaneous field of view of one pixel [rad]."""
        return pixel_pitch / self.focal_length

    def field_of_view(self, n_pixels: int, pixel_pitch: float) -> float:
        """Full field of view across n_pixels [rad]."""
        return 2.0 * math.atan(0.5 * n_pixels * pixel_pitch / self.focal_length)

    def ground_sample_distance(self, pixel_pitch: float, range_m: float) -> float:
        """Pixel footprint at range [m] (small-angle)."""
        return self.ifov(pixel_pitch) * range_m

    def q_parameter(self, lam: float, pixel_pitch: float) -> float:
        """Optical Q = lam F# / pitch (ratio of Nyquist to optical cutoff, x2).

        Q = 2 means critically sampled (Nyquist at the diffraction cutoff);
        thermal imagers typically run Q ~ 0.5-1.2 (detector-limited).
        """
        return lam * self.f_number / pixel_pitch
