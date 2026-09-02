"""Smoke tests: IMX900 sensor preset and the Streamlit app."""

import numpy as np
import pytest

from ircam.sensors import IMX900_SPECS, imx900_camera, imx900_qe

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_imx900_preset():
    assert 0.7 < imx900_qe(575e-9) < 0.9  # peak in the visible
    assert imx900_qe(850e-9) >= 0.3  # enhanced NIR claim
    assert imx900_qe(1200e-9) == 0.0
    cam = imx900_camera(f_number=4.0)
    assert cam.pixel_pitch == pytest.approx(2.25e-6)
    assert cam.well_capacity == pytest.approx(IMX900_SPECS["well_capacity"])
    # QE is monotone-decreasing beyond the peak (sane interpolation).
    lam = np.linspace(600e-9, 1100e-9, 50)
    assert np.all(np.diff(imx900_qe(lam)) <= 1e-12)


def test_streamlit_app_runs_clean():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    # All three story tabs render.
    assert len(at.tabs) == 3
    # Headline metrics exist (lam_eq caption + NEdT metrics in tab 3).
    assert len(at.metric) >= 4
