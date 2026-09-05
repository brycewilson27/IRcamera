"""Generic silicon QE model, sensor presets and hashable QE specs."""

import numpy as np
import pytest

from ircam.pyrometry import NotchFilter, electron_rate
from ircam.sensors import (
    SENSOR_PRESETS,
    camera_from_spec,
    imx900_qe,
    parametric_silicon_qe,
    qe_from_spec,
    silicon_absorption_coefficient,
    tabulated_qe,
)

LAM = np.linspace(350e-9, 1100e-9, 151)


def test_silicon_absorption_table():
    # Green (2008) anchors: alpha(600 nm) = 4.14e3 cm^-1, alpha(1000 nm) = 64 cm^-1;
    # the smoothing spline stays within 5% of every tabulated value.
    assert silicon_absorption_coefficient(600e-9) == pytest.approx(4.14e5, rel=0.05)
    assert silicon_absorption_coefficient(1000e-9) == pytest.approx(6.4e3, rel=0.05)
    from ircam.sensors import _SI_ALPHA_CM, _SI_ALPHA_LAM_NM
    fitted = silicon_absorption_coefficient(_SI_ALPHA_LAM_NM * 1e-9) / 100.0
    assert np.all(np.abs(fitted / _SI_ALPHA_CM - 1.0) < 0.05)
    alpha = silicon_absorption_coefficient(np.linspace(300e-9, 1200e-9, 500))
    assert np.all(np.diff(alpha) < 0)  # monotone toward the band gap
    assert silicon_absorption_coefficient(1300e-9) < 1.0  # past the band gap


def test_parametric_qe_knobs():
    qe = parametric_silicon_qe(LAM, 0.7, 8e-6, 400e-9)
    assert qe.max() == pytest.approx(0.7, abs=1e-3)  # peak normalisation
    assert np.all((qe >= 0.0) & (qe <= 0.7 + 1e-9))
    # Absorption depth sets the NIR tail.
    thin = parametric_silicon_qe(900e-9, 0.7, 4e-6, 400e-9)
    thick = parametric_silicon_qe(900e-9, 0.7, 16e-6, 400e-9)
    assert thick > 1.5 * thin
    # Blue edge sets the short-wavelength roll-off.
    early = parametric_silicon_qe(400e-9, 0.7, 8e-6, 360e-9)
    late = parametric_silicon_qe(400e-9, 0.7, 8e-6, 440e-9)
    assert early > late
    assert parametric_silicon_qe(1300e-9, 0.7, 8e-6, 400e-9) < 1e-3
    assert isinstance(parametric_silicon_qe(600e-9), float)


def test_parametric_model_brackets_imx900_shape():
    """The class model with the IMX900's peak and a 10 um effective depth
    follows the tabulated IMX900 curve to within 0.1 in QE over the notch
    range -- the level of fidelity the approximate presets claim."""
    lam = np.arange(450, 951, 10) * 1e-9
    model = parametric_silicon_qe(lam, 0.868, 10e-6, 350e-9)
    assert np.max(np.abs(model - imx900_qe(lam))) < 0.10


def test_presets_are_physical():
    assert list(SENSOR_PRESETS)[0] == "imx900"
    for key, preset in SENSOR_PRESETS.items():
        qe = preset.qe(LAM)
        assert np.all((qe >= 0.0) & (qe <= 1.0 + 1e-9)), key
        assert preset.pixel_pitch > 0 and preset.well_capacity > preset.read_noise > 0
        assert preset.min_exposure > 0 and preset.provenance
        if preset.qe_params is not None:
            assert qe.max() == pytest.approx(preset.qe_params[0], abs=1e-3), key
    imx = SENSOR_PRESETS["imx900"]
    assert not imx.approximate
    assert np.allclose(imx.qe(LAM), imx900_qe(LAM))
    assert imx.well_capacity == pytest.approx(9458.0)
    assert np.all(SENSOR_PRESETS["ideal"].qe(LAM) == 1.0)


def test_tabulated_qe_and_specs():
    pts = ((400e-9, 0.4), (600e-9, 0.8), (1000e-9, 0.1))
    qe = tabulated_qe(pts)
    assert qe(500e-9) == pytest.approx(0.6)
    assert qe(300e-9) == 0.0 and qe(1100e-9) == 0.0
    assert tabulated_qe(((500e-9, 1.7), (600e-9, -0.2)))(550e-9) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        tabulated_qe(((500e-9, 0.5),))
    with pytest.raises(ValueError):
        tabulated_qe(((500e-9, 0.5), (500e-9, 0.6)))

    assert qe_from_spec(("table", pts))(600e-9) == pytest.approx(0.8)
    assert qe_from_spec(("flat", 1.0))(np.array([500e-9, 900e-9])).tolist() == [1.0, 1.0]
    assert qe_from_spec(("preset", "imx900"))(620e-9) == pytest.approx(imx900_qe(620e-9))
    assert qe_from_spec(("parametric", 0.8, 8e-6, 400e-9))(550e-9) == pytest.approx(
        parametric_silicon_qe(550e-9, 0.8, 8e-6, 400e-9))
    with pytest.raises(ValueError):
        qe_from_spec(("nope",))

    cam = camera_from_spec(("parametric", 0.8, 8e-6, 400e-9), 3.45e-6, 2.8, 0.9,
                           2.4, 10500.0)
    assert cam.pixel_pitch == 3.45e-6 and cam.f_number == 2.8
    assert cam.optics_transmittance == 0.9
    assert cam.read_noise == 2.4 and cam.well_capacity == 10500.0
    assert cam.quantum_efficiency(550e-9) == pytest.approx(
        parametric_silicon_qe(550e-9, 0.8, 8e-6, 400e-9))


def test_qe_level_cancels_when_saturation_capped():
    """With the exposure pinned by the hottest pixel, the electron count at
    the evaluation temperature depends on the full well and the QE *shape*,
    not on the QE level: doubling the peak QE halves the exposure."""
    filt = NotchFilter(620e-9, 30e-9)
    counts, exposures = [], []
    for peak in (0.4, 0.8):
        cam = camera_from_spec(("parametric", peak, 8e-6, 400e-9), 2.25e-6, 4.0,
                               0.85, 5.0, 9458.0)
        t_exp = 0.6 * cam.well_capacity / electron_rate(3273.15, filt, cam)
        exposures.append(t_exp)
        counts.append(electron_rate(1773.15, filt, cam) * t_exp)
    assert counts[0] == pytest.approx(counts[1], rel=1e-9)
    assert exposures[0] == pytest.approx(2.0 * exposures[1], rel=1e-9)
