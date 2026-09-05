"""Smoke tests: IMX900 sensor preset and the Streamlit app."""

import numpy as np
import pytest

from ircam.sensors import (
    IMX900_BASLER_EMVA_PEAK_QE,
    IMX900_GAIN_MODES,
    IMX900_QE_TABLE_PEAK,
    imx900_camera,
    imx900_qe,
)

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def test_imx900_qe_table():
    lam = np.arange(400, 1001, 5) * 1e-9
    raw = imx900_qe(lam, absolute_peak=None)
    # Verbatim table anchors from the source repository (percent / 100).
    assert raw[0] == pytest.approx(0.839)          # 400 nm
    assert raw.max() == pytest.approx(0.958)       # peak
    assert raw[(620 - 400) // 5] == pytest.approx(0.890)
    assert raw[(870 - 400) // 5] == pytest.approx(0.383)
    assert raw[-1] == pytest.approx(0.137)         # 1000 nm
    assert IMX900_QE_TABLE_PEAK == pytest.approx(0.958)
    # Default scaling anchors the peak to the Basler EMVA 1288 value.
    assert imx900_qe(lam).max() == pytest.approx(IMX900_BASLER_EMVA_PEAK_QE)
    assert imx900_qe(1200e-9) == 0.0 and imx900_qe(350e-9) == 0.0
    # Monotone decreasing beyond the peak region.
    assert np.all(np.diff(raw[(600 - 400) // 5:]) < 0)


def test_imx900_gain_modes():
    lcg = imx900_camera(f_number=4.0)  # default operating point
    assert lcg.pixel_pitch == pytest.approx(2.25e-6)
    assert lcg.well_capacity == pytest.approx(9458.0)
    assert lcg.read_noise == pytest.approx(5.56)
    hcg = imx900_camera(f_number=4.0, gain_mode="hcg")
    assert hcg.well_capacity == pytest.approx(2183.0)
    assert hcg.read_noise == pytest.approx(1.39)
    assert set(IMX900_GAIN_MODES) == {"lcg", "hcg"}
    # Overrides win over the mode defaults.
    assert imx900_camera(read_noise=2.0).read_noise == 2.0


def test_streamlit_app_runs_clean():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=120)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    # All five story tabs render.
    assert len(at.tabs) == 5
    # Headline metrics exist (lam_eq caption + NEdT metrics in tab 3).
    assert len(at.metric) >= 4


def test_streamlit_app_other_sensors_run_clean():
    at = AppTest.from_file("../streamlit_app.py", default_timeout=120)
    at.run()
    assert at.sidebar.selectbox[0].value == "imx900"
    for key in ("parametric", "table", "large_5p86", "ideal"):
        # Re-fetch the widget each time: element handles go stale after a rerun.
        at.sidebar.selectbox[0].select(key).run()
        assert at.sidebar.selectbox[0].value == key
        assert not at.exception, [str(e) for e in at.exception]
        assert len(at.tabs) == 5 and len(at.metric) >= 4
