# IR Camera Project

Physics analysis of infrared camera requirements: a validated Python
toolkit (`ircam`) plus a requirements-flowdown analysis built on it.

## Start here

**[docs/nozzle_pyrometry.md](docs/nozzle_pyrometry.md)** — the mission
analysis: video thermography of an engine nozzle (500–3000 °C, primary
1500–3000 °C) by two-notch visible/NIR ratio pyrometry — one vs two
notches, optimal notch placement and widths, dynamic range, and a
recommended design. Computed tables in
[docs/computed_nozzle_results.md](docs/computed_nozzle_results.md).

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

`tests/test_physics.py` checks the code against independent references:
Planck integral vs Stefan–Boltzmann, Wien peak, blackbody fractions vs
the series expansion, analytic vs finite-difference thermal derivatives,
photon/energy radiance consistency, NETD scaling laws, and
Johnson-range round trips.
