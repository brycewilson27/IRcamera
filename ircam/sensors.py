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

Other sensors
-------------
Only the IMX900 curve above is a tabulated measurement. For sensors not
yet chosen, ``parametric_silicon_qe`` gives an approximate QE from three
knobs (peak QE, effective absorption depth, blue edge) driven by the
crystalline-silicon absorption coefficient; ``SENSOR_PRESETS`` bundles a
few such class models (FSI, BSI, NIR-enhanced, large pixel) with typical
pitch / well / read-noise values, plus an ideal QE = 1 reference.
``tabulated_qe`` builds a QE function from user-entered points, and
``qe_from_spec`` / ``camera_from_spec`` turn a hashable description of any
of these into a callable / ``PyroCamera`` (the Streamlit app caches on it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.interpolate import PchipInterpolator

from .pyrometry import PyroCamera

__all__ = ["imx900_qe", "imx900_camera", "IMX900_SPECS", "IMX900_GAIN_MODES",
           "IMX900_QE_TABLE_PEAK", "IMX900_BASLER_EMVA_PEAK_QE",
           "silicon_absorption_coefficient", "parametric_silicon_qe",
           "tabulated_qe", "flat_qe", "SensorPreset", "SENSOR_PRESETS",
           "qe_from_spec", "camera_from_spec"]

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


# ---------------------------------------------------------------------------
# Generic silicon QE model, user-defined curves and sensor presets
# ---------------------------------------------------------------------------

# Crystalline-silicon absorption coefficient at 300 K [cm^-1], rounded from
# Green (2008), Sol. Energy Mater. Sol. Cells 92, 1305 ("Self-consistent
# optical parameters of intrinsic silicon at 300 K"). Good to ~10% over
# 400-1100 nm; it only drives the approximate parametric QE model below.
_SI_ALPHA_LAM_NM = np.array([
    300, 350, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620,
    640, 660, 680, 700, 720, 740, 760, 780, 800, 820, 840, 860, 880, 900,
    920, 940, 960, 980, 1000, 1020, 1040, 1060, 1080, 1100, 1150, 1200,
], dtype=float)
_SI_ALPHA_CM = np.array([
    1.73e6, 1.06e6, 9.52e4, 5.00e4, 3.11e4, 2.10e4, 1.48e4, 1.11e4, 8.80e3,
    7.05e3, 5.78e3, 4.79e3, 4.14e3, 3.52e3, 3.04e3, 2.58e3, 2.21e3, 1.90e3,
    1.66e3, 1.44e3, 1.22e3, 1.06e3, 8.50e2, 7.07e2, 6.00e2, 5.00e2, 4.20e2,
    3.06e2, 2.40e2, 1.83e2, 1.35e2, 9.34e1, 6.40e1, 4.30e1, 2.70e1, 1.60e1,
    8.7, 3.5, 0.5, 0.022,
])


_LOG_ALPHA_SPLINE = PchipInterpolator(_SI_ALPHA_LAM_NM, np.log(_SI_ALPHA_CM),
                                      extrapolate=False)


def silicon_absorption_coefficient(lam):
    """c-Si absorption coefficient alpha(lam) [1/m] at 300 K.

    Shape-preserving (PCHIP) interpolation of log alpha over the Green (2008)
    table, so the curve is smooth and stays monotone; held at the 300 nm
    value below it and effectively zero beyond 1200 nm (the band gap).
    """
    lam_nm = np.clip(np.asarray(lam, dtype=float) * 1e9, _SI_ALPHA_LAM_NM[0], None)
    log_alpha = _LOG_ALPHA_SPLINE(lam_nm)
    log_alpha = np.where(lam_nm > _SI_ALPHA_LAM_NM[-1], np.log(1e-9), log_alpha)
    return np.exp(log_alpha) * 100.0


_QE_NORM_GRID = np.linspace(350e-9, 1100e-9, 751)


def parametric_silicon_qe(lam, peak_qe: float = 0.80,
                          absorption_depth: float = 8e-6,
                          blue_edge: float = 400e-9,
                          blue_width: float = 25e-9):
    """Approximate silicon QE from three knobs.

        QE(lam) = A * [1 - exp(-alpha(lam) d)] * s(lam)

    ``alpha`` is the c-Si absorption coefficient; ``d`` is the *effective*
    absorption depth (photodiode thickness times any back-reflection or
    light-trapping path gain), which sets the NIR tail; ``s`` is a logistic
    short-wavelength roll-off centred on ``blue_edge`` (front-side layers on
    FSI sensors, surface losses on BSI); ``A`` is chosen so the curve peaks
    at ``peak_qe`` over 350-1100 nm. This reproduces the *shape* of
    published silicon QE curves to roughly +/-0.05-0.1 in QE. It is a class
    model for bracketing a sensor choice, not a measurement of any device.
    """
    def shape(x):
        x = np.asarray(x, dtype=float)
        absorbed = 1.0 - np.exp(-silicon_absorption_coefficient(x) * absorption_depth)
        rolloff = 1.0 / (1.0 + np.exp(-(x - blue_edge) / blue_width))
        return absorbed * rolloff

    out = peak_qe * shape(lam) / shape(_QE_NORM_GRID).max()
    return out.item() if np.ndim(out) == 0 else out


def tabulated_qe(points) -> Callable:
    """QE function from ((wavelength [m], QE fraction), ...) points.

    Linear interpolation between points, zero outside the table. Needs at
    least two points with distinct wavelengths; QE is clipped to [0, 1].
    """
    pts = sorted((float(lam), float(qe)) for lam, qe in points)
    if len(pts) < 2:
        raise ValueError("tabulated_qe needs at least two points")
    lam_t = np.array([lam for lam, _ in pts])
    if np.any(np.diff(lam_t) <= 0.0):
        raise ValueError("tabulated_qe wavelengths must be distinct")
    qe_t = np.clip([qe for _, qe in pts], 0.0, 1.0)

    def qe(lam):
        return np.interp(np.asarray(lam, dtype=float), lam_t, qe_t,
                         left=0.0, right=0.0)

    return qe


def flat_qe(lam, value: float = 1.0):
    """Wavelength-independent QE (reference detector)."""
    return np.full_like(np.asarray(lam, dtype=float), float(value))


@dataclass(frozen=True)
class SensorPreset:
    """A sensor the app can pick: QE curve plus typical pixel and noise values.

    ``approximate`` is True for class models whose numbers are typical
    published values rather than a datasheet or measurement of the device;
    ``qe_params`` holds (peak, absorption depth, blue edge) when the QE
    comes from ``parametric_silicon_qe``.
    """

    key: str
    name: str
    short_name: str
    pixel_pitch: float
    well_capacity: float
    read_noise: float
    min_exposure: float
    qe: Callable
    provenance: str
    approximate: bool = True
    qe_params: tuple | None = None


def _parametric_preset(key, name, short_name, pitch, well, read, min_exp,
                       peak, depth, edge, provenance):
    return SensorPreset(
        key=key, name=name, short_name=short_name, pixel_pitch=pitch,
        well_capacity=well, read_noise=read, min_exposure=min_exp,
        qe=lambda lam, _p=peak, _d=depth, _e=edge: parametric_silicon_qe(lam, _p, _d, _e),
        provenance=provenance, approximate=True, qe_params=(peak, depth, edge))


#: Sensors the designer can choose from (insertion order = menu order).
SENSOR_PRESETS: dict[str, SensorPreset] = {
    "imx900": SensorPreset(
        key="imx900",
        name="Sony IMX900 (tabulated QE)",
        short_name="Sony IMX900",
        pixel_pitch=IMX900_SPECS["pixel_pitch"],
        well_capacity=IMX900_GAIN_MODES["lcg"]["well_capacity"],
        read_noise=IMX900_GAIN_MODES["lcg"]["read_noise"],
        min_exposure=IMX900_SPECS["min_exposure"],
        qe=imx900_qe,
        provenance=("Tabulated 121-point QE and EMVA 1288 gain modes transcribed "
                    "from the StarTrackerCentroid repository; the only preset "
                    "backed by a measured curve (see module docstring)."),
        approximate=False),
    "fsi_3p45": _parametric_preset(
        "fsi_3p45", "FSI CMOS 3.45 um (approx.)", "FSI CMOS 3.45 um",
        3.45e-6, 10500.0, 2.4, 5e-6, 0.65, 4.5e-6, 430e-9,
        "Approximate class model of a front-illuminated global-shutter "
        "machine-vision sensor (Pregius Gen 2 class): peak QE 0.65, effective "
        "absorption depth 4.5 um, blue edge 430 nm. Pitch, well and read noise "
        "are typical published values, not a datasheet."),
    "bsi_2p74": _parametric_preset(
        "bsi_2p74", "BSI CMOS 2.74 um (approx.)", "BSI CMOS 2.74 um",
        2.74e-6, 10000.0, 2.5, 2e-6, 0.85, 10e-6, 360e-9,
        "Approximate class model of a back-illuminated global-shutter sensor "
        "(Pregius S class): peak QE 0.85, effective absorption depth 10 um, "
        "blue edge 360 nm. Pitch, well and read noise are typical published "
        "values, not a datasheet."),
    "nir_2p9": _parametric_preset(
        "nir_2p9", "NIR-enhanced 2.9 um (approx.)", "NIR-enhanced CMOS 2.9 um",
        2.9e-6, 12000.0, 3.0, 2e-6, 0.80, 16e-6, 370e-9,
        "Approximate class model of a thick-photodiode NIR-enhanced sensor: "
        "peak QE 0.80, effective absorption depth 16 um (light trapping), blue "
        "edge 370 nm. Pitch, well and read noise are typical published values, "
        "not a datasheet."),
    "large_5p86": _parametric_preset(
        "large_5p86", "Large-pixel 5.86 um (approx.)", "large-pixel CMOS 5.86 um",
        5.86e-6, 32000.0, 6.5, 10e-6, 0.77, 7e-6, 400e-9,
        "Approximate class model of a large-pixel global-shutter sensor "
        "(IMX174 class): peak QE 0.77, effective absorption depth 7 um, blue "
        "edge 400 nm. Its large full well is what changes the precision budget. "
        "Pitch, well and read noise are typical published values, not a datasheet."),
    "ideal": SensorPreset(
        key="ideal", name="Ideal QE = 1 (reference)", short_name="ideal detector",
        pixel_pitch=3.0e-6, well_capacity=10000.0, read_noise=1.0,
        min_exposure=1e-6, qe=flat_qe,
        provenance=("Reference only: unit QE at every wavelength, so the notch "
                    "trade shows pure Planck-curve behaviour."),
        approximate=True),
}


def qe_from_spec(spec) -> Callable:
    """QE function from a hashable description (the app caches on these).

    * ``("preset", key)`` -- a ``SENSOR_PRESETS`` entry
    * ``("parametric", peak_qe, absorption_depth, blue_edge)``
    * ``("table", ((lam_m, qe), ...))`` -- user points, see ``tabulated_qe``
    * ``("flat", value)``
    """
    kind = spec[0]
    if kind == "preset":
        return SENSOR_PRESETS[spec[1]].qe
    if kind == "parametric":
        _, peak, depth, edge = spec
        return lambda lam: parametric_silicon_qe(lam, peak, depth, edge)
    if kind == "table":
        return tabulated_qe(spec[1])
    if kind == "flat":
        return lambda lam: flat_qe(lam, spec[1])
    raise ValueError(f"unknown QE spec kind {kind!r}")


def camera_from_spec(spec, pixel_pitch: float, f_number: float = 4.0,
                     optics_transmittance: float = 0.85,
                     read_noise: float = 5.0, well_capacity: float = 10e3,
                     ) -> PyroCamera:
    """PyroCamera with the QE described by ``spec`` (see ``qe_from_spec``)."""
    return PyroCamera(
        pixel_pitch=pixel_pitch, f_number=f_number,
        optics_transmittance=optics_transmittance,
        quantum_efficiency=qe_from_spec(spec),
        read_noise=read_noise, well_capacity=well_capacity)
