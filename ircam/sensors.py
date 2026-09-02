"""Sensor presets for pyrometry cameras.

Sony IMX900-AMR-C
-----------------
1/3.1-type Pregius S (4th-gen, back-illuminated stacked) global shutter,
2.25 um pixels, 2063 x 1551 active array, 12-bit ADC. Values below are
taken verbatim from the StarTrackerCentroid repository
(``framegen/sensors/presets/imx900.py`` and
``matlab_digitaltwin/src/radiometry.py::imx900_qe_table``), which carries
source annotations for each number.

Conversion-gain modes (two in-pixel capacitors):

* HCG, high conversion gain: PTC-measured full well 2183 e-, read noise
  1.39 e- rms (FRAMOS EMVA 1288 HCG baseline: 1.954 e- rms).
* LCG, low conversion gain: FRAMOS EMVA 1288 LCG baseline, full well
  9458 e-, read noise 5.56 e- rms.

Pyrometry of a scene that contains 3000 C content is well-capacity
limited, not read-noise limited, so LCG is the default operating point
here; the star-tracker firmware uses HCG for the opposite reason.

Quantum efficiency: the 121-point curve (400-1000 nm, 5 nm) is the
locked shape reference of the source repository, ported from a Sony
spectral-response sheet, peaking at 0.958 near 505-530 nm. The source
repository's audit (``qe_audit.md``) flags that this peak is probably a
normalised spectral response, while the only publicly citable absolute
value is Basler's EMVA 1288 measurement of an IMX900 camera: 0.868 peak
at 525 nm. ``imx900_qe`` therefore applies an absolute-peak scale
(default 0.868; pass ``absolute_peak=None`` for the raw 0.958 curve).
In the saturation-capped regime the scale cancels out of the precision
budget (it rescales exposure, not electrons).

Dark current: 4.523 e-/s/pixel at 60 C (99.094 e-/s at 85 C). At the
microsecond-to-millisecond exposures of this application it is
negligible and is not modelled.
"""

from __future__ import annotations

import numpy as np

from .pyrometry import PyroCamera

__all__ = ["imx900_qe", "imx900_camera", "IMX900_SPECS", "IMX900_GAIN_MODES",
           "IMX900_QE_TABLE_PEAK", "IMX900_BASLER_EMVA_PEAK_QE"]

IMX900_SPECS = {
    "name": "Sony IMX900-AMR-C (Pregius S, BSI stacked global shutter)",
    "format": "1/3.1-type, 2063 x 1551 active (3.20 MP)",
    "pixel_pitch": 2.25e-6,
    "adc_bits": 12,
    "dark_current_ref_e_per_s": 4.523,  # at 60 C
    "dark_current_ref_temp_c": 60.0,
    "min_exposure": 2e-6,  # seconds; typical global-shutter floor (assumed)
}

#: Conversion-gain operating points: (full well [e-], read noise [e- rms]).
IMX900_GAIN_MODES = {
    "lcg": {"well_capacity": 9458.0, "read_noise": 5.56,
            "source": "FRAMOS EMVA 1288 LCG baseline, 0 dB"},
    "hcg": {"well_capacity": 2183.0, "read_noise": 1.39,
            "source": "PTC-measured HCG (datasheet baseline 1.954 e-)"},
}

# QE shape reference, percent, 400-1000 nm at 5 nm (121 points), verbatim.
_IMX900_QE_LAM_NM = np.arange(400, 1001, 5, dtype=float)
_IMX900_QE_PCT = np.array([
    83.9, 84.5, 85.0, 86.1, 87.2, 88.3, 89.4, 90.6, 91.7, 92.6,
    93.3, 93.8, 93.7, 93.5, 94.2, 94.3, 94.4, 95.1, 95.7, 95.6,
    95.7, 95.8, 95.6, 95.6, 95.6, 95.7, 95.7, 95.6, 95.5, 95.3,
    95.0, 94.9, 94.8, 94.6, 94.2, 93.8, 93.4, 93.0, 92.7, 92.4,
    91.9, 91.3, 90.5, 89.7, 89.0, 88.3, 87.7, 86.9, 86.0, 84.9,
    83.9, 83.0, 82.3, 81.8, 81.2, 80.6, 79.8, 78.6, 77.5, 76.4,
    75.6, 74.7, 73.9, 73.2, 72.3, 71.4, 70.5, 69.5, 68.4, 67.2,
    66.0, 64.8, 63.8, 62.7, 61.8, 60.8, 59.5, 58.3, 56.9, 55.5,
    54.1, 53.0, 52.0, 51.2, 50.4, 49.9, 48.9, 47.8, 46.8, 45.0,
    43.5, 42.1, 40.3, 39.3, 38.3, 36.9, 36.4, 35.7, 34.3, 33.6,
    33.0, 31.4, 30.0, 29.0, 27.6, 25.9, 25.0, 24.3, 22.8, 21.7,
    21.4, 20.4, 19.2, 18.7, 18.2, 17.1, 16.1, 15.4, 14.7, 14.2,
    13.7,
])

#: Peak of the shape-reference curve.
IMX900_QE_TABLE_PEAK = float(_IMX900_QE_PCT.max() / 100.0)
#: Basler EMVA 1288 absolute peak QE (a2A2048-37gmPRO, 525 nm).
IMX900_BASLER_EMVA_PEAK_QE = 0.868

# Interpolate in metres so a caller's `1000e-9` matches the table edge
# exactly (converting the query to nm can land at 1000.0000000000002).
_IMX900_QE_LAM_M = _IMX900_QE_LAM_NM * 1e-9


def imx900_qe(lam, absolute_peak: float | None = IMX900_BASLER_EMVA_PEAK_QE):
    """IMX900 quantum efficiency vs wavelength [m].

    The tabulated shape is scaled so its peak equals ``absolute_peak``
    (default: the Basler EMVA 1288 value); ``None`` returns the raw
    table. Zero outside 400-1000 nm, matching the source repository.
    """
    scale = 1.0 if absolute_peak is None else absolute_peak / IMX900_QE_TABLE_PEAK
    return np.interp(np.asarray(lam, dtype=float), _IMX900_QE_LAM_M,
                     _IMX900_QE_PCT / 100.0, left=0.0, right=0.0) * scale


def imx900_camera(f_number: float = 4.0, optics_transmittance: float = 0.85,
                  gain_mode: str = "lcg", read_noise: float | None = None,
                  well_capacity: float | None = None,
                  qe_absolute_peak: float | None = IMX900_BASLER_EMVA_PEAK_QE,
                  ) -> PyroCamera:
    """PyroCamera configured as an IMX900 behind the given optics.

    ``read_noise`` / ``well_capacity`` override the selected gain mode.
    """
    mode = IMX900_GAIN_MODES[gain_mode.lower()]
    return PyroCamera(
        pixel_pitch=IMX900_SPECS["pixel_pitch"],
        f_number=f_number,
        optics_transmittance=optics_transmittance,
        quantum_efficiency=lambda lam: imx900_qe(lam, qe_absolute_peak),
        read_noise=mode["read_noise"] if read_noise is None else read_noise,
        well_capacity=(mode["well_capacity"] if well_capacity is None
                       else well_capacity),
    )
