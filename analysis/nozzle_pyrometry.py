#!/usr/bin/env python3
"""Nozzle thermography mission: two-notch visible/NIR ratio pyrometry design.

Scenario: video thermal mapping of an engine nozzle, scene temperatures
500-3000 C present simultaneously, primary accuracy band 1500-3000 C.
Instrument: standoff silicon camera(s) with narrowband ("notch") filters.

Run from the repository root:  python analysis/nozzle_pyrometry.py
Writes figures to figures/ and docs/computed_nozzle_results.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ircam import planck
from ircam.constants import C2
from ircam.pyrometry import (
    PLUME_EMISSION_LINES,
    NotchFilter,
    RatioPyrometer,
    electron_rate,
    equivalent_wavelength,
    exposure_for_well_fill,
    ratio_temperature_error_wien,
    single_band_temperature_error,
)
from ircam.sensors import IMX900_GAIN_MODES, IMX900_SPECS, imx900_camera, imx900_qe

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

# ------------------------------------------------------------------ mission
T_MIN_C, T_MAX_C = 500.0, 3000.0
T_PRIMARY_LO = 1500.0 + 273.15  # K
T_PRIMARY_HI = 3000.0 + 273.15  # K
WELL_FILL = 0.6                  # exposure set so T_max hits 60% well
F_NUMBER = 4.0
TAU_OPTICS = 0.85                # objective + standoff window, excl. filter
GAIN_MODE = "lcg"                # large-well mode: the scene is well-limited
CAMERA = imx900_camera(F_NUMBER, TAU_OPTICS, GAIN_MODE)  # same instrument as the app

# Candidate pairs [centre_short, width_short, centre_long, width_long] (nm)
CANDIDATE_PAIRS = {
    "450/650 (initial proposal)": (450, 30, 650, 30),
    "620/870 (recommended)": (620, 30, 870, 50),
    "650/950 (max SNR)": (650, 30, 950, 50),
    "600/650 (close pair)": (600, 30, 650, 30),
}


def make_pyro(l1_nm, w1_nm, l2_nm, w2_nm, camera=None):
    return RatioPyrometer(
        NotchFilter(l1_nm * 1e-9, w1_nm * 1e-9),
        NotchFilter(l2_nm * 1e-9, w2_nm * 1e-9),
        CAMERA if camera is None else camera,
    )


def frame_exposures(pyro):
    """Per-band exposures anti-saturated for the hottest scene point."""
    return (
        exposure_for_well_fill(T_PRIMARY_HI, pyro.filter_short, pyro.camera,
                               WELL_FILL),
        exposure_for_well_fill(T_PRIMARY_HI, pyro.filter_long, pyro.camera,
                               WELL_FILL),
    )


# ----------------------------------------------------------------- figures

def fig_spectra():
    lam = np.linspace(350e-9, 1150e-9, 600)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for tc, color in ((500, "#4a6"), (900, "#293"), (1500, "#b58900"),
                      (2250, "#cb4b16"), (3000, "#dc322f")):
        t = tc + 273.15
        ax.semilogy(lam * 1e9, planck.spectral_radiance(lam, t) * 1e-9,
                    color=color, label=f"{tc} C")
    # IMX900 response window
    qe = imx900_qe(lam)
    ax2 = ax.twinx()
    ax2.fill_between(lam * 1e9, 0, qe, color="gray", alpha=0.15, lw=0)
    ax2.set_ylabel("IMX900 QE (shaded)", color="gray")
    ax2.set_ylim(0, 1.6)
    ax2.tick_params(axis="y", colors="gray")
    # Recommended notches
    for c_nm, w_nm in ((620, 30), (870, 50)):
        ax.axvspan(c_nm - w_nm / 2, c_nm + w_nm / 2, color="#268bd2", alpha=0.25)
        ax.text(c_nm, 3e7, f"{c_nm} nm\nnotch", ha="center", fontsize=8,
                color="#268bd2")
    # Plume emission features to avoid
    for name, lines in PLUME_EMISSION_LINES.items():
        for line in lines:
            if line < 1150e-9:
                ax.axvline(line * 1e9, color="crimson", ls=":", lw=0.8, alpha=0.7)
    ax.text(589, 2e-4, "Na", color="crimson", fontsize=7, ha="center")
    ax.text(656, 2e-4, "H$\\alpha$", color="crimson", fontsize=7, ha="center")
    ax.text(768, 2e-4, "K", color="crimson", fontsize=7, ha="center")
    ax.text(940, 2e-4, "H$_2$O", color="crimson", fontsize=7, ha="center")
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Spectral radiance [W m$^{-2}$ sr$^{-1}$ nm$^{-1}$]")
    ax.set_title("Nozzle blackbody spectra on the IMX900 response window; "
                 "notches placed off plume emission lines")
    ax.set_ylim(1e-6, 1e8)
    ax.legend(loc="upper left", fontsize=9, title="Scene temperature")
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "nozzle_spectra.png", dpi=110)
    plt.close(fig)


def worst_case_sigma(l1_nm, l2_nm, w1_nm=30.0, w2_nm=50.0,
                     binning=1, frames=1):
    """Single-frame NEdT at the cold end of the primary range, with the
    exposure anti-saturated for the hot end (the video/heterogeneity case)."""
    pyro = make_pyro(l1_nm, w1_nm, l2_nm, w2_nm)
    t1, t2 = frame_exposures(pyro)
    return pyro.sigma_T(T_PRIMARY_LO, t1, t2, binning=binning, frames=frames)


def fig_pair_optimization():
    l1_grid = np.linspace(420, 780, 46)
    l2_grid = np.linspace(620, 1000, 50)
    sigma = np.full((len(l2_grid), len(l1_grid)), np.nan)
    for i, l2 in enumerate(l2_grid):
        for j, l1 in enumerate(l1_grid):
            if l2 - l1 >= 60.0:
                sigma[i, j] = worst_case_sigma(l1, l2)
    fig, ax = plt.subplots(figsize=(9, 6))
    levels = [25, 50, 75, 100, 150, 200, 300, 500, 1000]
    cs = ax.contourf(l1_grid, l2_grid, sigma, levels=levels, cmap="viridis_r",
                     extend="max")
    fig.colorbar(cs, ax=ax, label="Single-frame NEdT at 1500 C [K]")
    cl = ax.contour(l1_grid, l2_grid, sigma, levels=[50, 100, 200],
                    colors="white", linewidths=0.8)
    ax.clabel(cl, fontsize=7, fmt="%.0f K")
    # Mark candidates
    marks = {"450/650": (450, 650, "s", "white"),
             "620/870": (620, 870, "*", "red"),
             "650/950": (650, 950, "o", "orange")}
    for label, (x, y, m, c) in marks.items():
        ax.plot(x, y, m, color=c, ms=11 if m == "*" else 7,
                mec="black", mew=0.5)
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(8, -4), fontsize=8, color="black")
    # Line-contamination guidance
    for lines in PLUME_EMISSION_LINES.values():
        for line in lines:
            nm = line * 1e9
            if 420 < nm < 780:
                ax.axvline(nm, color="crimson", ls=":", lw=0.8, alpha=0.6)
            if 700 < nm < 1000:
                ax.axhline(nm, color="crimson", ls=":", lw=0.8, alpha=0.6)
    ax.set_xlabel("Short notch centre $\\lambda_1$ [nm]  "
                  "(red dotted: plume emission features)")
    ax.set_ylabel("Long notch centre $\\lambda_2$ [nm]")
    ax.set_title("Notch-pair optimization: worst-case NEdT at 1500 C\n"
                 "(exposure anti-saturated for 3000 C in the same frame, "
                 "single pixel, single frame)")
    fig.tight_layout()
    fig.savefig(FIGURES / "notch_pair_optimization.png", dpi=110)
    plt.close(fig)
    return l1_grid, l2_grid, sigma


def fig_sigma_vs_temperature():
    temps_c = np.linspace(1300, 3050, 60)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = {"450/650 (initial proposal)": "#6c71c4",
              "620/870 (recommended)": "#268bd2",
              "650/950 (max SNR)": "#2aa198",
              "600/650 (close pair)": "#93a1a1"}
    for label, (l1, w1, l2, w2) in CANDIDATE_PAIRS.items():
        pyro = make_pyro(l1, w1, l2, w2)
        t1, t2 = frame_exposures(pyro)
        sig = [pyro.sigma_T(tc + 273.15, t1, t2) for tc in temps_c]
        ax.semilogy(temps_c, sig, label=label, color=colors[label])
    # Recommended pair with modest averaging (2x2 binning x 8 frames)
    pyro = make_pyro(*CANDIDATE_PAIRS["620/870 (recommended)"])
    t1, t2 = frame_exposures(pyro)
    sig_avg = [pyro.sigma_T(tc + 273.15, t1, t2, binning=2, frames=8)
               for tc in temps_c]
    ax.semilogy(temps_c, sig_avg, "--", color="#268bd2",
                label="620/870, 2x2 binning x 8 frames")
    ax.axvspan(1500, 3000, color="orange", alpha=0.08)
    ax.text(2250, 700, "primary range", ha="center", fontsize=9,
            color="darkorange")
    ax.set_xlabel("Scene temperature [C]")
    ax.set_ylabel("Ratio-pyrometry NEdT [K]")
    ax.set_title("Temperature precision vs scene temperature\n"
                 "(single frame exposed for 3000 C full-scale, per pixel "
                 "unless noted)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "nozzle_sigma_vs_temperature.png", dpi=110)
    plt.close(fig)


def fig_channel_dynamics():
    pyro = make_pyro(*CANDIDATE_PAIRS["620/870 (recommended)"])
    t1, t2 = frame_exposures(pyro)
    temps_c = np.linspace(600, 3050, 200)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for filt, t_exp, color, label in (
        (pyro.filter_short, t1, "#268bd2",
         f"620 nm notch (t_exp = {t1 * 1e6:.2f} us)"),
        (pyro.filter_long, t2, "#cb4b16",
         f"870 nm notch (t_exp = {t2 * 1e6:.2f} us)"),
    ):
        n_e = [electron_rate(tc + 273.15, filt, CAMERA) * t_exp
               for tc in temps_c]
        ax.semilogy(temps_c, n_e, color=color, label=label)
    ax.axhline(CAMERA.well_capacity, color="black", ls="-", lw=1)
    ax.text(700, CAMERA.well_capacity * 1.3,
            f"full well ({CAMERA.well_capacity / 1e3:.1f} ke-, "
            f"{GAIN_MODE.upper()})", fontsize=8)
    ax.axhline(WELL_FILL * CAMERA.well_capacity, color="black", ls="--", lw=0.8)
    ax.axhline(CAMERA.read_noise, color="gray", ls=":")
    ax.text(700, CAMERA.read_noise * 1.4,
            f"read noise ({CAMERA.read_noise:.2f} e-)", fontsize=8,
            color="gray")
    ax.axhline(100, color="gray", ls="-.", lw=0.8)
    ax.text(700, 130, "SNR ~ 10 floor (100 e-)", fontsize=8, color="gray")
    ax.axvspan(1500, 3000, color="orange", alpha=0.08)
    ax.set_xlabel("Scene temperature [C]")
    ax.set_ylabel("Electrons per pixel per frame")
    ax.set_title("In-frame dynamic range, 620/870 pair exposed for 3000 C "
                 "full scale")
    ax.set_ylim(1, 1e5)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "nozzle_channel_dynamics.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ report

def main():
    FIGURES.mkdir(exist_ok=True)
    lines: list[str] = []
    add = lines.append
    add("# Computed results: nozzle two-notch pyrometry\n")
    add("*Machine-generated by `analysis/nozzle_pyrometry.py` -- do not edit "
        "by hand.*\n")
    add(f"Camera model: {IMX900_SPECS['name']}, "
        f"{CAMERA.pixel_pitch * 1e6:.2f} um pixels, F/{CAMERA.f_number:g}, "
        f"{GAIN_MODE.upper()} mode (well {CAMERA.well_capacity / 1e3:.2f} ke-, "
        f"read noise {CAMERA.read_noise:.2f} e- rms), tabulated IMX900 QE "
        f"scaled to the Basler EMVA 1288 peak of 0.868, optics+window "
        f"transmittance {CAMERA.optics_transmittance}. Exposures anti-saturated "
        f"to {WELL_FILL * 100:.0f}% well at 3000 C. This is the same instrument "
        f"model the Streamlit app uses by default.\n")

    # -------- candidate pair comparison table
    add("## Candidate notch pairs (single pixel, single frame)\n")
    add("| Pair | lam_eq [nm] | Exposures (short/long) [us] | e- at 1500 C "
        "(short/long) | NEdT @ 1500 C | NEdT @ 2250 C | NEdT @ 3000 C |")
    add("|---|---|---|---|---|---|---|")
    for label, (l1, w1, l2, w2) in CANDIDATE_PAIRS.items():
        pyro = make_pyro(l1, w1, l2, w2)
        t1, t2 = frame_exposures(pyro)
        n1 = electron_rate(T_PRIMARY_LO, pyro.filter_short, CAMERA) * t1
        n2 = electron_rate(T_PRIMARY_LO, pyro.filter_long, CAMERA) * t2
        sigs = [pyro.sigma_T(tc + 273.15, t1, t2) for tc in (1500, 2250, 3000)]
        add(f"| {label} | {pyro.equivalent_wavelength * 1e9:.0f} | "
            f"{t1 * 1e6:.2f} / {t2 * 1e6:.2f} | {n1:.0f} / {n2:.0f} | "
            f"{sigs[0]:.0f} K | {sigs[1]:.0f} K | {sigs[2]:.0f} K |")
    add("")

    # -------- recommended pair with averaging
    pyro = make_pyro(*CANDIDATE_PAIRS["620/870 (recommended)"])
    t1, t2 = frame_exposures(pyro)
    add("## Recommended pair 620/870 nm with modest averaging\n")
    add("| Averaging | NEdT @ 1500 C | NEdT @ 2250 C | NEdT @ 3000 C |")
    add("|---|---|---|---|")
    for label, b, f in (("per pixel, per frame", 1, 1),
                        ("2x2 binning", 2, 1),
                        ("2x2 binning x 8 frames", 2, 8)):
        sigs = [pyro.sigma_T(tc + 273.15, t1, t2, binning=b, frames=f)
                for tc in (1500, 2250, 3000)]
        add(f"| {label} | {sigs[0]:.1f} K | {sigs[1]:.1f} K | {sigs[2]:.1f} K |")
    add("")

    # -------- conversion-gain mode comparison
    add("## Conversion-gain mode, 620/870 pair (single pixel, single frame)\n")
    add("| Mode | Well [e-] | Read noise [e-] | e- at 1500 C (short/long) | "
        "NEdT @ 1500 C | NEdT @ 2250 C | NEdT @ 3000 C |")
    add("|---|---|---|---|---|---|---|")
    for mode, spec in IMX900_GAIN_MODES.items():
        cam = imx900_camera(F_NUMBER, TAU_OPTICS, mode)
        p = make_pyro(*CANDIDATE_PAIRS["620/870 (recommended)"], camera=cam)
        e1, e2 = frame_exposures(p)
        n1 = electron_rate(T_PRIMARY_LO, p.filter_short, cam) * e1
        n2 = electron_rate(T_PRIMARY_LO, p.filter_long, cam) * e2
        sigs = [p.sigma_T(tc + 273.15, e1, e2) for tc in (1500, 2250, 3000)]
        add(f"| {mode.upper()} | {spec['well_capacity']:.0f} | "
            f"{spec['read_noise']:.2f} | {n1:.0f} / {n2:.0f} | {sigs[0]:.0f} K | "
            f"{sigs[1]:.0f} K | {sigs[2]:.0f} K |")
    add("")

    # -------- systematic errors
    add("## Systematic error scales (620/870 pair)\n")
    for tc in (1500, 3000):
        t = tc + 273.15
        bias = pyro.emissivity_bias(t, 1.05)
        add(f"- Non-gray emissivity, eps(620)/eps(870) = 1.05 assumed gray: "
            f"bias = **{bias:+.0f} K at {tc} C**")
    err1 = single_band_temperature_error(T_PRIMARY_HI, 650e-9, 0.10)
    add(f"- Single-band (one-notch) comparison: a 10% error in emissivity, "
        f"window fouling, or absolute calibration reads as "
        f"**{err1:.0f} K at 3000 C** (at 650 nm); the ratio cancels all such "
        f"common-mode terms.")
    add(f"- Wien-limit noise law: sigma_T = (lam_eq T^2/c2) x (sigma_R/R); "
        f"for 620/870, lam_eq = "
        f"{equivalent_wavelength(620e-9, 870e-9) * 1e9:.0f} nm -> 1% ratio "
        f"error = "
        f"{ratio_temperature_error_wien(T_PRIMARY_HI, 620e-9, 870e-9, 0.01):.0f}"
        f" K at 3000 C, "
        f"{ratio_temperature_error_wien(T_PRIMARY_LO, 620e-9, 870e-9, 0.01):.0f}"
        f" K at 1500 C.\n")

    # -------- low-temperature coverage
    add("## Low-temperature coverage (secondary range, 500-1500 C)\n")
    add("Best case with exposure adapted to the local scene temperature "
        "(HDR / exposure bracketing; anti-saturated at 60% well in the "
        "870 nm notch, capped at a 16.6 ms video frame):\n")
    add("| Scene T [C] | Exposure | 870 nm e-/frame | 620 nm e-/frame | "
        "ratio usable? |")
    add("|---|---|---|---|---|")
    for tc in (500, 600, 700, 800, 900, 1100, 1300):
        t = tc + 273.15
        t_exp = min(exposure_for_well_fill(t, pyro.filter_long, CAMERA,
                                           WELL_FILL), 1.0 / 60.0)
        n_l = electron_rate(t, pyro.filter_long, CAMERA) * t_exp
        n_s = electron_rate(t, pyro.filter_short, CAMERA) * t_exp
        usable = "yes" if n_s > 100 else (
            "marginal" if n_s > 10 else
            ("870 brightness mode only" if n_l > 25 else "no"))
        exp_str = (f"{t_exp * 1e3:.2f} ms" if t_exp > 1e-3
                   else f"{t_exp * 1e6:.0f} us")
        add(f"| {tc} | {exp_str} | {n_l:.3g} | {n_s:.3g} | {usable} |")
    add("")

    (DOCS / "computed_nozzle_results.md").write_text("\n".join(lines))

    fig_spectra()
    fig_pair_optimization()
    fig_sigma_vs_temperature()
    fig_channel_dynamics()

    print("\n".join(lines))
    print(f"\nFigures written to {FIGURES}/")


if __name__ == "__main__":
    main()
