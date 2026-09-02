# Viewing Angle and Reflected Sunlight

Two error terms that the plain radiometric chain leaves out, quantified
for the nozzle instrument (IMX900 LCG, F/4, 620/870 nm). The first is
multiplicative and varies across the frame; the second is additive and,
counter-intuitively, hurts the two-band ratio more than a single band.
Numbers come from `analysis/nozzle_geometry_sun.py`
([`computed_geometry_sun_results.md`](computed_geometry_sun_results.md));
the model is `ircam.surface`, tested in `tests/test_surface.py`. The
interactive version is tab 4 of `streamlit_app.py`.

**Scope of the model.** Directional emissivity uses the smooth-surface
Fresnel equations with complex refractive indices that are *illustrative
class values* (an oxide-like dielectric, a graphite-like semi-metal, a
tungsten-like metal), not measurements of any nozzle material. Rough
surfaces are more Lambertian than this up to fairly grazing angles, so
the Fresnel curve is a worst case for the angular collapse. Sunlight is
the ASTM G173 AM1.5G spectrum (about 10% uncertain), reflected diffusely
with reflectance 1 − ε(θ_v). A coupon measurement of ε(θ, λ) at both
notch wavelengths replaces the first assumption; a pyranometer reading on
the day replaces the second.

---

## 1. Distance drops out; angle does not

For a pixel-filling surface the pixel irradiance is L·π/(4F²+1)·τ, with no
distance term: the flux from any patch falls as 1/R² while the pixel's
footprint grows as R², and the two cancel. A camera calibrated on a lamp
at 1 m reads a nozzle at 40 m without correction. Viewing angle enters
through three separate mechanisms instead.

**Foreshortening.** The footprint stretches by 1/cos θ along the line of
sight: 2× at 60°, 3.9× at 75°, 11.5× at 85°. Axial gradients smear inside
a pixel; circumferential resolution is unaffected. A truly parallel view
has zero projected area.

**Directional emissivity.** Emissivity is flat to roughly 60° from the
normal and then collapses toward zero at grazing incidence (metals first
rise slightly near 75°). Because the local viewing angle changes
continuously along a bell seen from its exit, this is a multiplicative
error that differs from pixel to pixel:

![Directional emissivity](../figures/geometry_emissivity.png)

**Reflectance.** Whatever emissivity loses, reflectance gains
(ρ = 1 − ε for an opaque surface). An oxide-like surface goes from 7%
reflective at normal incidence to 63% at 85°, so a grazing wall becomes a
mirror for the plume, the sky, and, inside the bell, the throat. Reflected
throat radiation makes the divergent wall read hot; it is additive and is
not cancelled by the ratio.

## 2. Angle-only bias: one band vs ratio

With the instrument calibrated against a normal-incidence coupon, the
apparent temperature at oblique views is:

| Material | Angle | One band, 1500 °C | Ratio, 1500 °C | One band, 3000 °C | Ratio, 3000 °C |
|---|---|---|---|---|---|
| oxide-like dielectric | 75° | −35 K | 0 K | −117 K | 0 K |
| oxide-like dielectric | 85° | −116 K | 0 K | −375 K | 0 K |
| graphite-like | 75° | −18 K | −16 K | −62 K | −55 K |
| graphite-like | 85° | −82 K | −33 K | −270 K | −111 K |
| tungsten-like metal | 75° | +8 K | −4 K | +28 K | −15 K |
| tungsten-like metal | 85° | −32 K | 0 K | −107 K | −2 K |

![Angle bias](../figures/geometry_angle_error.png)

The ratio cancels the angular factor exactly when it is the same in both
bands (the dielectric, and very nearly the metal) and only partly when the
optical constants differ between 620 and 870 nm (the graphite-like case
drifts 7% by 85°). For a graphite-like surface the ratio buys little
below 75°; its advantage appears beyond that. In every case the single
band is exposed to the full collapse, which reaches hundreds of kelvin at
3000 °C past 80°.

## 3. Reflected sunlight: additive, and worse for the ratio

Reflected sun is a fixed radiance added to each channel, while the thermal
signal falls steeply toward cooler scene temperatures, so the
contamination fraction rises fast toward the cold end. It is much larger
in the short band because the thermal spectrum at 1500 °C is ~9× weaker at
620 nm than at 870 nm while sunlight is 1.5× stronger there:

| Scene T | Sun / signal, 620 nm | Sun / signal, 870 nm | One band | Ratio | Ratio after 90% subtraction |
|---|---|---|---|---|---|
| 1200 °C | 79% | 3.4% | +57 K | +205 K | +24 K |
| 1500 °C | 5.5% | 0.5% | +7 K | +23 K | +2 K |
| 2000 °C | 0.3% | 0.07% | +1 K | +2 K | 0 K |
| 2500 °C | 0.05% | 0.02% | 0 K | 0 K | 0 K |

(Graphite-like, viewed at 45°, full sun at 45° from the surface normal.)

![Sun bias](../figures/sun_error_vs_temperature.png)

Two things follow. First, the ratio's immunity applies to *multiplicative*
common-mode terms only; an additive term that is different in the two
bands shifts the ratio, and the shift is then amplified by λ_eq, which is
3.5× the single band's λ. Under full sun the ratio bias is roughly three
times the one-band bias at every temperature. Second, above ~2000 °C the
sun is irrelevant for either method, and below ~1400 °C it dominates both.

**Mitigation is direct.** Before ignition the nozzle is cold, so a
pre-ignition frame in each band is a per-pixel measurement of the
reflected-sun term itself. Subtracting it removes the bias in proportion
to how well the reflectance and sun geometry hold during the burn; a 90%
subtraction takes the 1500 °C ratio bias from +23 K to +2 K. Shading the
nozzle or testing at night removes it entirely. Note also that sunlight
and an oblique view push in opposite directions (sun warm, angle cold);
the combined scenario below shows them nearly cancelling for one band at
1500 °C, which is a coincidence of the chosen geometry and not something
to design around.

**Specular glint is a separate hazard.** If the sun's reflection lands in
the camera, the glint radiance at 620 nm is 9–11× that of a 3000 °C
surface and saturates the pixel outright. The sun-surface-camera specular
geometry must be excluded by camera placement or shading; no calibration
recovers a clipped pixel.

## 4. Combined budget at the recommended geometry

Graphite-like surface viewed at 70°, full sun at 45°, normal-incidence
calibration, no background subtraction:

| Scene T | One band: angle / sun / both | Ratio: angle / sun / both | Shot noise (2×2 × 8 frames) |
|---|---|---|---|
| 1500 °C | −10 / +9 / −1 K | −11 / +29 / +17 K | 41 K |
| 2250 °C | −20 / 0 / −20 K | −23 / +1 / −22 K | 8 K |
| 3000 °C | −34 / 0 / −34 K | −39 / 0 / −38 K | 5 K |

At 70° with a graphite-like surface the angle term is similar for both
methods and comparable to the shot noise at the hot end, and the sun term
is only significant at the cold end, where it is larger for the ratio. With
the pre-ignition subtraction the sun column shrinks by an order of
magnitude and the angle term is what remains to be measured on a coupon.

## 5. Design consequences

- Keep the view within 60–70° of the surface normal where possible; use a
  periscope mirror to steepen the angle if the camera must sit below the
  exit plane. Past 75° axial resolution is gone and the emissivity
  collapse dominates.
- Record pre-ignition frames in both bands and subtract them; shade the
  nozzle from direct sun if the test is in daylight; exclude the specular
  sun geometry.
- Measure ε(θ) at 620 and 870 nm on a coupon of the actual nozzle material
  and use the measured band ratio in the calibration; the illustrative
  constants here only bound the effect.
- Treat the ratio's advantage precisely: it removes multiplicative
  common-mode terms (emissivity level, fouling, geometry, exposure when
  shared) and most of the angular factor, and it does not remove additive
  terms (sunlight, plume and throat reflections, veiling glare), which it
  amplifies.
