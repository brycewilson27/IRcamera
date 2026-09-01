"""Standard infrared spectral bands."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    """A spectral band defined by its wavelength limits [m]."""

    name: str
    lam1: float
    lam2: float

    @property
    def center(self) -> float:
        return 0.5 * (self.lam1 + self.lam2)

    def __str__(self) -> str:
        return f"{self.name} ({self.lam1 * 1e6:.1f}-{self.lam2 * 1e6:.1f} um)"


#: Short-wave IR: reflected-light band (needs illumination; thermal only for hot objects)
SWIR = Band("SWIR", 0.9e-6, 1.7e-6)
#: Mid-wave IR atmospheric window
MWIR = Band("MWIR", 3.0e-6, 5.0e-6)
#: Long-wave IR atmospheric window
LWIR = Band("LWIR", 8.0e-6, 14.0e-6)

STANDARD_BANDS = (SWIR, MWIR, LWIR)
