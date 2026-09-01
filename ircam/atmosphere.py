"""Simple horizontal-path atmospheric transmission model.

This is a single-parameter Beer-Lambert model,

    tau(R) = exp(-beta * R),

with a band-averaged extinction coefficient beta [1/km]. It captures the
first-order requirement trade (range vs. sensitivity) inside the
atmospheric windows but none of the spectral structure (CO2 notch at
4.2-4.4 um, water-vapour continuum, aerosol size effects). For real
system design the band-averaged tau must come from MODTRAN or measured
data for the specified operating environment.

Rough band-averaged extinction coefficients for a clear midlatitude
sea-level horizontal path (~15 km visibility, moderate humidity):

    SWIR: ~0.25 /km (aerosol-scattering dominated)
    MWIR: ~0.12 /km
    LWIR: ~0.17 /km (water-vapour continuum dominated)

MWIR degrades faster than LWIR in high humidity; LWIR degrades less in
dust/aerosol. These defaults are for framework-level trades only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimpleAtmosphere:
    """Beer-Lambert atmosphere with band-averaged extinction [1/km]."""

    extinction_per_km: float

    def transmittance(self, range_m):
        """Path transmittance over a horizontal range [m]."""
        return np.exp(-self.extinction_per_km * np.asarray(range_m, dtype=float) / 1e3)

    def range_for_transmittance(self, tau: float) -> float:
        """Range [m] at which the path transmittance falls to tau."""
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        return -1e3 * np.log(tau) / self.extinction_per_km


#: Clear midlatitude sea-level defaults, keyed by band name.
CLEAR_SEA_LEVEL = {
    "SWIR": SimpleAtmosphere(0.25),
    "MWIR": SimpleAtmosphere(0.12),
    "LWIR": SimpleAtmosphere(0.17),
}
