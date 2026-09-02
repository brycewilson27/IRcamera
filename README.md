# IR Camera Project

Physics analysis of infrared camera requirements: a validated Python
toolkit (`ircam`) plus a requirements-flowdown analysis built on it.

## Interactive designer app

`streamlit_app.py` is an interactive two-notch pyrometry designer on the
Sony IMX900 sensor model: notch spacing/width vs temperature sensitivity,
the one-notch radiometric-uncertainty story, and temperature certainty vs
scene temperature for the selected pair.

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

To deploy on Streamlit Community Cloud: push this repo to GitHub, then at
share.streamlit.io create an app pointing at this repository/branch with
main file `streamlit_app.py` (requirements.txt is picked up automatically).

## Start here

**[docs/nozzle_pyrometry.md](docs/nozzle_pyrometry.md)** — the mission
analysis: video thermography of an engine nozzle (500–3000 °C, primary
1500–3000 °C) by two-notch visible/NIR ratio pyrometry — one vs two
notches, optimal notch placement and widths, dynamic range, and a
recommended design. Computed tables in
[docs/computed_nozzle_results.md](docs/computed_nozzle_results.md).

**[docs/viewing_geometry_and_sun.md](docs/viewing_geometry_and_sun.md)** —
the viewing-angle and reflected-sunlight error budget: directional
emissivity (multiplicative, mostly cancelled by the ratio) and reflected
sun (additive, amplified by the ratio), with the pre-ignition subtraction
mitigation. Tab 4 of the app.

**[docs/physics_analysis.md](docs/physics_analysis.md)** — the general
framework:
scene radiometry, band trade (MWIR vs LWIR), atmosphere, optics,
detector physics, NETD both ways (uncooled D\* and cooled photon chain),
Johnson-criteria DRI ranges, and a worked flowdown to a derived
requirement set. All quoted numbers are computed by the package and
regenerated into [docs/computed_results.md](docs/computed_results.md).

## Package

| Module | Contents |
|---|---|
| `ircam.planck` | Planck law, band integrals, thermal derivatives (analytic dL/dT), photon radiance |
| `ircam.bands` | SWIR / MWIR / LWIR band definitions |
| `ircam.atmosphere` | Beer–Lambert band-averaged path transmission |
| `ircam.optics` | F#, étendue, diffraction, IFOV/FOV, sampling Q |
| `ircam.detector` | FPA description, Lloyd D\*-NETD, BLIP D\* |
| `ircam.radiometry` | End-to-end photon-detector chain: electrons, noise, NETD, SNR |
| `ircam.range_performance` | Johnson-criteria DRI: sampling- and contrast-limited ranges |
| `ircam.pyrometry` | Visible/NIR ratio pyrometry: notch filters, silicon QE, NEdT, emissivity bias |
| `ircam.sensors` | Sony IMX900 preset: tabulated QE, LCG/HCG gain modes, provenance |
| `ircam.surface` | Fresnel directional emissivity, AM1.5G reflected sunlight, apparent-temperature inversions, glint hazard |

## Usage

```bash
pip install -e .
pytest                          # physics validation tests
python analysis/run_analysis.py # regenerate figures/ and docs/computed_results.md
```

```python
from ircam import LWIR, planck
planck.band_radiance(300.0, LWIR.lam1, LWIR.lam2)  # 54.9 W/m^2/sr
```

## Validation

[docs/verification.md](docs/verification.md) is the formula-by-formula
record: each formula, the independent reference it is checked against,
the test that enforces it, and which inputs are assumptions rather than
physics. Highlights: Planck integrals vs Stefan–Boltzmann (energy and
photon forms), Wien constant from its transcendental root, blackbody
fractions vs the series expansion and textbook anchors, the pupil solid
angle by numerical integration, the Airy coefficient from the Bessel
zero, and the Lloyd D\*-NETD route cross-checked against the
photon-counting route at the BLIP limit. Sensor values are transcription-
checked against their source. Run `pytest tests/`.
