"""Sensor presets for pyrometry cameras.

Sony IMX900 (Pregius S, 4th-gen global shutter, back-illuminated stacked):
1/3.1-type, 3.21 MP (2064 x 1552), 2.25 um pixels, ~10 ke- saturation
capacity, enhanced NIR response (~2x a conventional front-illuminated
Pregius at 850 nm per Sony's marketing material).

The QE curve below is an APPROXIMATION assembled from Sony's published
qualitative claims and typical BSI global-shutter curves -- Sony does not
publish absolute QE for the IMX900 openly. Read noise is assumed at the
Pregius S class value (~2.5 e- rms). Replace both with datasheet /
measured values before using results for a design freeze; note that in
the saturation-capped video regime the QE largely cancels out of the
temperature-error budget (it rescales exposure, not electrons).
"""

from __future__ import annotations

import numpy as np

from .pyrometry import PyroCamera

__all__ = ["imx900_qe", "imx900_camera", "IMX900_SPECS"]

IMX900_SPECS = {
    "name": "Sony IMX900 (Pregius S, BSI stacked global shutter)",
    "format": "1/3.1-type, 2064 x 1552 (3.21 MP)",
    "pixel_pitch": 2.25e-6,
    "well_capacity": 10e3,   # electrons (~10 ke- saturation)
    "read_noise": 2.5,       # electrons rms, Pregius S class (assumed)
    "min_exposure": 2e-6,    # seconds, typical global-shutter floor (assumed)
}

# Approximate mono QE, enhanced-NIR BSI global shutter (see module docstring).
_IMX900_QE_LAM = np.array([350, 400, 450, 500, 550, 600, 650, 700, 750, 800,
                           850, 900, 950, 1000, 1050, 1100]) * 1e-9
_IMX900_QE = np.array([0.30, 0.62, 0.72, 0.80, 0.84, 0.84, 0.80, 0.73, 0.63,
                       0.52, 0.40, 0.29, 0.19, 0.10, 0.04, 0.0])


def imx900_qe(lam):
    """Approximate IMX900 mono quantum efficiency vs wavelength [m]."""
    return np.interp(lam, _IMX900_QE_LAM, _IMX900_QE, left=0.0, right=0.0)


def imx900_camera(f_number: float = 4.0, optics_transmittance: float = 0.85,
                  read_noise: float | None = None,
                  well_capacity: float | None = None) -> PyroCamera:
    """PyroCamera configured as an IMX900 behind the given optics."""
    return PyroCamera(
        pixel_pitch=IMX900_SPECS["pixel_pitch"],
        f_number=f_number,
        optics_transmittance=optics_transmittance,
        quantum_efficiency=imx900_qe,
        read_noise=IMX900_SPECS["read_noise"] if read_noise is None else read_noise,
        well_capacity=(IMX900_SPECS["well_capacity"] if well_capacity is None
                       else well_capacity),
    )
