# Physics Verification Record

Formula-by-formula record of what the `ircam` package computes, what each
formula was checked against, and which test enforces the check. "Verified"
means an automated test compares the implementation with an *independent*
reference (closed form, textbook constant, numerical integration, or a
second derivation route). "Derivation-consistent" means the formula is a
standard definition or algebraic identity with no independent numerical
reference in the suite. "Assumption" means a parameter value, not physics.

Run: `pytest tests/` (44 tests). Test files: `test_physics.py`,
`test_pyrometry.py`, `test_verification.py`, `test_surface.py`, `test_app.py`.

## Constants

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| c₁L = 2hc², c₂ = hc/k | `constants.C1L`, `constants.C2` | CODATA 2018: 1.191042972×10⁻¹⁶ W m² sr⁻¹, 1.438776877×10⁻² m K | `test_radiation_constants_match_codata` | Verified |
| σ = 2π⁵k⁴/(15h³c²) | `constants.SIGMA` (scipy) | closed form | `test_stefan_boltzmann_constant_identity` | Verified |
| Wien b = hc/(k x*), x* : x = 5(1−e⁻ˣ) | `constants.WIEN_B` (scipy) | transcendental root 4.965114 | `test_wien_constant_from_transcendental_root` | Verified |

## Blackbody radiometry (`ircam.planck`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| L_λ = c₁L λ⁻⁵ /(e^{c₂/λT} − 1) | `spectral_radiance` | ∫₀^∞ L_λ dλ = σT⁴/π to 10⁻⁶; Wien limit for x ≫ 1 | `test_planck_integrates_to_stefan_boltzmann`, `test_wien_limit_of_planck` | Verified |
| L_q,λ = 2c λ⁻⁴ /(e^{x} − 1) | `spectral_photon_radiance` | L_λ / L_q,λ = hc/λ; photon Stefan–Boltzmann 4πζ(3)k³T³/(c²h³) = 1.5205×10¹⁵ T³ to 10⁻⁶ | `test_photon_and_energy_radiance_consistent`, `test_photon_stefan_boltzmann_law` | Verified |
| ∂L_λ/∂T = L_λ (x/T) e^x/(e^x − 1) | `spectral_radiance_dT` | central finite difference, 10⁻⁷ | `test_thermal_derivative_matches_finite_difference` | Verified |
| Band integrals (Gauss–Legendre in ln λ) | `band_radiance` etc. | blackbody fraction series F(0→λT) = (15/π⁴)Σ…, 10⁻⁶; textbook anchors F(λ_peak T) = 0.2501, F(4107 µm K) = 0.500 | `test_band_fraction_matches_series_expansion`, `test_blackbody_fraction_textbook_anchors` | Verified |
| λ_peak = b/T | `wien_peak_wavelength` | numerical maximum of L_λ | `test_wien_peak_matches_numerical_maximum` | Verified |
| 300 K band values (L, fractions, relative contrast) | — | standard table values (8–14 µm: 37.6%; 3–5 µm: 1.3%; MWIR/LWIR relative contrast 3.6 / 1.5 %/K) | `test_known_300k_band_values` | Verified (order-of-magnitude anchors) |

## Optics (`ircam.optics`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| Ω = π/(4F² + 1) | `Optics.pixel_solid_angle`, `PyroCamera.pixel_etendue` | numerical ∫cos θ dΩ over the cone tan θ_max = 1/(2F) | `test_projected_solid_angle_by_integration`, `test_pixel_solid_angle_limits` | Verified |
| Airy null radius 1.22 λF | `Optics.airy_radius` | j₁,₁/π = 1.2197 (first zero of J₁) | `test_airy_coefficient_from_bessel_zero` | Verified (0.03% coefficient rounding) |
| Incoherent cutoff 1/(λF) | `Optics.diffraction_cutoff` | standard result | — | Derivation-consistent |
| IFOV = p/f, FOV = 2 atan(Np/2f), GSD = IFOV·R | `Optics.ifov`, `field_of_view`, `ground_sample_distance` | geometry | `test_ifov_and_fov` | Verified |
| Q = λF/p | `Optics.q_parameter` | definition | — | Derivation-consistent |

## Detectors and sensitivity (`ircam.detector`, `ircam.radiometry`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| Lloyd NETD = (4F²+1)√Δf / (√A_d τ_o D* ∂M/∂T) | `netd_from_dstar` | derived in module docstring from NEP = √(AΔf)/D* and Φ = L A Ω τ; **independent route:** equals the photon-counting chain NETD when D* = D*_BLIP and Δf = 1/(2t_int), to 0.5% | `test_lloyd_netd_equals_photon_chain_at_blip`, `test_netd_dstar_scaling` | Verified |
| D*_BLIP = (λ/hc)√(η/(2E_q)) (photovoltaic) | `blip_dstar` | same cross-check as above | `test_lloyd_netd_equals_photon_chain_at_blip` | Verified |
| Photon chain: N = η L_q A Ω τ t; σ² = N + N_dark + σ_read²; NETD = σ/(dN/dT); SNR = ΔT·(dN/dT)/σ | `PhotonDetectorChain` | shot-noise √2 scaling with t_int; SNR(NETD) = 1; well fill and NETD in the published cooled-MWIR class | `test_photon_chain_netd_reasonable`, cross-check above | Verified |
| Δf = 1/(2 t_int) for a staring array | convention in `netd_from_dstar` docstring | standard | — | Derivation-consistent |

## Range performance (`ircam.range_performance`, `ircam.atmosphere`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| Sampling-limited range R = d_c f/(2p N₅₀) | `sampling_limited_range` | round trip with `focal_length_for_task` and `cycles_on_target` | `test_sampling_range_relations` | Verified (algebra) |
| Contrast-limited range R = ln(ΔT₀/(k·NETD))/β | `contrast_limited_range` | closed form | `test_contrast_limited_range` | Verified (algebra) |
| Johnson N₅₀ = 0.75 / 3.0 / 6.0 | `JOHNSON_N50` | NVESD 2-D target values | — | **Assumption** (literature values) |
| τ = exp(−βR); β = 0.12 / 0.17 / 0.25 km⁻¹ | `SimpleAtmosphere`, `CLEAR_SEA_LEVEL` | Beer–Lambert definition; β values are rough clear-air figures | — | **Assumption** (needs MODTRAN for a real specification) |

## Ratio pyrometry (`ircam.pyrometry`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| ln R = const − (c₂/T)(1/λ₁ − 1/λ₂); d ln R/dT = c₂/(T² λ_eq) | `RatioPyrometer.ratio`, `dlnratio_dT` (full integrals) | Wien-limit closed form: 3% for 30/50 nm filters, 0.2% for 2 nm filters | `test_dlnratio_matches_wien_slope`, `test_ratio_slope_converges_to_wien_for_narrow_filters` | Verified |
| R(T) monotonic; T(R) inversion | `temperature_from_ratio` | round trip to 0.01 K | `test_ratio_monotonic_and_round_trip` | Verified |
| σ_T = (λ_eq T²/c₂)·σ_R/R | `sigma_T`, `ratio_temperature_error_wien` | numeric shot-noise propagation with read noise = 0, 3%; √frames averaging | `test_sigma_t_consistent_with_wien_formula` | Verified |
| Gray-assumption bias ΔT ≈ T²(λ_eq/c₂) ln(ε₁/ε₂) | `emissivity_bias` (exact inversion) | Wien magnitude and sign | `test_emissivity_bias_sign_and_wien_magnitude` | Verified |
| Single-band error dT = (λT²/c₂)·dS/S | `single_band_temperature_error` | classic 650 nm / 3273 K / 10% → 48 K; identity with the ratio law at λ = λ_eq | `test_single_band_error_formula`, `test_error_laws_reduce_consistently` | Verified |
| N_e = A Ω τ ∫ QE·F·ε·L_q dλ | `electron_rate` | exposure-for-fill round trip; µs exposures at 3000 °C | `test_exposure_for_well_fill` | Verified (algebra) |
| Plume emission line list | `PLUME_EMISSION_LINES` | standard line wavelengths (Na D, Hα, K, Li, CH, C₂ Swan, H₂O bands) | — | Reference data, not tested |

## Sensor model (`ircam.sensors`)

| Item | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| IMX900 QE table, 121 points | `_IMX900_QE_PCT` | verbatim from StarTrackerCentroid `matlab_digitaltwin/src/radiometry.py::imx900_qe_table` (anchors at 400, 620, 870, 1000 nm and peak) | `test_imx900_qe_table` | Verified transcription |
| QE absolute peak 0.868 | `IMX900_BASLER_EMVA_PEAK_QE` | Basler EMVA 1288, a2A2048-37gmPRO; the raw table peak 0.958 is flagged in the source repository as a probable normalised response | `test_imx900_qe_table` | **Assumption** (choice between two published figures; cancels in the saturation-capped regime) |
| LCG 9458 e⁻ / 5.56 e⁻; HCG 2183 e⁻ / 1.39 e⁻ | `IMX900_GAIN_MODES` | FRAMOS EMVA 1288 (LCG); PTC measurement (HCG), via `framegen/sensors/presets/imx900.py` | `test_imx900_gain_modes` | Verified transcription |
| 2 µs minimum exposure | `IMX900_SPECS["min_exposure"]` | typical global-shutter floor | — | **Assumption** |

## Viewing geometry and sunlight (`ircam.surface`)

| Formula | Implementation | Reference | Test | Status |
|---|---|---|---|---|
| Fresnel r_s, r_p with complex Snell; ε(θ) = 1 − (R_s + R_p)/2 | `fresnel_reflectances`, `directional_emissivity` | normal-incidence closed form 1 − ((n−1)² + k²)/((n+1)² + k²); R_p = 0 at Brewster's angle for k = 0; ε → 0 at grazing incidence; 0 ≤ ε ≤ 1 | `test_normal_incidence_closed_form`, `test_brewster_angle_for_lossless_dielectric`, `test_grazing_limit_and_energy_bounds` | Verified |
| L_refl = (1 − ε(θ_v)) E_sun cos θ_s / π | `solar_reflected_electron_rate` | zero when shaded or sun behind the surface; warm bias sign | `test_solar_terms`, `test_one_band_round_trip_and_signs` | Verified (algebra, signs) |
| One-band and ratio apparent temperature by exact inversion | `one_band_apparent_temperature`, `ratio_apparent_temperature` | round trip at the calibration angle; ratio bias exactly zero when both bands share optical constants; ratio sun bias > 2× one-band sun bias at 1500 °C; 90% subtraction scales the bias ~10× | `test_one_band_round_trip_and_signs`, `test_ratio_cancels_angular_factor_for_dielectric`, `test_sun_biases_ratio_more_than_one_band_at_1500c` | Verified |
| Glint ratio (1 − ε) L_sun / (ε L_bb), L_sun = E_sun/Ω_sun | `specular_glint_ratio` | > 1 at 3000 °C | `test_solar_terms` | Verified (order of magnitude) |
| Optical constants (three material classes) | `MATERIALS` | illustrative class values | — | **Assumption** (coupon measurement required) |
| AM1.5G solar spectrum, 25-point table | `solar_spectral_irradiance` | ASTM G173-03, ±10%, narrow bands smoothed | `test_solar_terms` (620 nm value) | **Assumption** (site/day irradiance required) |
| Diffuse (Lambertian) reflection; smooth-surface Fresnel emissivity | model form | worst case for the angular collapse; real BRDFs lie between diffuse and specular | — | **Assumption** |

## Modelling assumptions not covered by tests

* Gray emissivity between the two notches (the bias formula quantifies
  departures; refractory-surface ε slopes are not measured here).
* Top-hat filter passbands with perfect out-of-band blocking.
* Cold-shield-matched optics; no optics self-emission or narcissus.
* Continuum emitter (surface or soot); no line emission inside a notch.
* Wien-limit slope in the Streamlit app (bounded at < 3% by test).
* No comparison against measured hardware data — every check above is
  against theory, tabulated constants, or a second derivation route.
