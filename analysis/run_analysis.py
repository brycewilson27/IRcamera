#!/usr/bin/env python3
"""Generate the IR camera requirements analysis: figures + computed results.

Run from the repository root:

    python analysis/run_analysis.py

Writes figures to figures/ and a machine-generated results file to
docs/computed_results.md. The narrative analysis in
docs/physics_analysis.md cites the numbers produced here.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ircam import (
    CLEAR_SEA_LEVEL,
    LWIR,
    MWIR,
    SWIR,
    Fpa,
    Optics,
    PhotonDetectorChain,
    dstar_from_jones,
    netd_from_dstar,
    planck,
)
from ircam.range_performance import (
    CRITICAL_DIMENSION,
    JOHNSON_N50,
    dri_ranges,
    focal_length_for_task,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
DOCS = ROOT / "docs"

BAND_COLORS = {"SWIR": "#b58900", "MWIR": "#cb4b16", "LWIR": "#268bd2"}

# ----------------------------------------------------------------- scenario
# Reference surveillance scenario driving the worked example:
SCENE_T = 300.0            # background temperature [K]
TARGET_DT = 2.0            # inherent human-vs-background contrast [K]
HUMAN = CRITICAL_DIMENSION["human"]
RECOGNITION_RANGE = 400.0  # requirement: recognize a human at 400 m
SNR_REQUIRED = 2.5

# Candidate design A: uncooled LWIR microbolometer.
LWIR_PITCH = 12e-6
LWIR_FORMAT = (640, 512)
LWIR_FNUM = 1.0
LWIR_TAU_O = 0.9
LWIR_DSTAR_JONES = 1.2e9   # effective D*, achievable modern a-Si/VOx
LWIR_BANDWIDTH = 15.0      # ~1/(2 * 33 ms frame) [Hz]

# Candidate design B: cooled MWIR photon detector (InSb-class).
MWIR_PITCH = 15e-6
MWIR_FORMAT = (640, 512)
MWIR_FNUM = 2.5
MWIR_TAU_O = 0.85
MWIR_QE = 0.7
MWIR_WELL = 7e6
MWIR_READ = 300.0
MWIR_TINT = 5e-3


def fig_planck_curves():
    lam = np.logspace(np.log10(0.5e-6), np.log10(30e-6), 800)
    fig, ax = plt.subplots(figsize=(8, 5))
    for temp in (250, 300, 350, 500, 800):
        ax.loglog(lam * 1e6, planck.spectral_radiance(lam, temp) * 1e-6,
                  label=f"{temp} K")
    for band in (SWIR, MWIR, LWIR):
        ax.axvspan(band.lam1 * 1e6, band.lam2 * 1e6, alpha=0.12,
                   color=BAND_COLORS[band.name])
        ax.text(band.center * 1e6, 3e8, band.name, ha="center", fontsize=9,
                color=BAND_COLORS[band.name])
    peak = planck.wien_peak_wavelength(300.0)
    ax.axvline(peak * 1e6, ls=":", color="gray")
    ax.annotate(f"300 K peak\n{peak * 1e6:.2f} um", (peak * 1e6, 2e5),
                fontsize=8, ha="left")
    ax.set_xlabel("Wavelength [um]")
    ax.set_ylabel("Spectral radiance [W m$^{-2}$ sr$^{-1}$ um$^{-1}$]")
    ax.set_title("Planck spectral radiance and the IR imaging bands")
    ax.set_ylim(1e-2, 1e9)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "planck_spectral_radiance.png", dpi=110)
    plt.close(fig)


def fig_band_radiance_vs_temperature():
    temps = np.linspace(230, 1000, 400)
    fig, ax = plt.subplots(figsize=(8, 5))
    for band in (SWIR, MWIR, LWIR):
        ax.semilogy(temps, planck.band_radiance(temps, band.lam1, band.lam2),
                    label=str(band), color=BAND_COLORS[band.name])
    ax.axvline(300, ls=":", color="gray")
    ax.set_xlabel("Scene temperature [K]")
    ax.set_ylabel("In-band radiance [W m$^{-2}$ sr$^{-1}$]")
    ax.set_title("In-band blackbody radiance vs scene temperature")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "band_radiance_vs_temperature.png", dpi=110)
    plt.close(fig)


def fig_thermal_contrast():
    temps = np.linspace(240, 400, 300)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for band in (MWIR, LWIR):
        dl = planck.band_radiance_dT(temps, band.lam1, band.lam2)
        rel = dl / planck.band_radiance(temps, band.lam1, band.lam2)
        ax1.plot(temps, dl, label=str(band), color=BAND_COLORS[band.name])
        ax2.plot(temps, 100 * rel, label=str(band), color=BAND_COLORS[band.name])
    for ax in (ax1, ax2):
        ax.axvline(300, ls=":", color="gray")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_xlabel("Scene temperature [K]")
    ax1.set_ylabel("dL/dT [W m$^{-2}$ sr$^{-1}$ K$^{-1}$]")
    ax1.set_title("Absolute thermal contrast (drives signal)")
    ax2.set_ylabel("(1/L) dL/dT [%/K]")
    ax2.set_title("Relative contrast (drives photon-limited NETD)")
    fig.tight_layout()
    fig.savefig(FIGURES / "thermal_contrast.png", dpi=110)
    plt.close(fig)


def fig_atmosphere():
    ranges = np.linspace(0, 5000, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, atm in CLEAR_SEA_LEVEL.items():
        ax.plot(ranges / 1e3, atm.transmittance(ranges),
                label=f"{name} (beta = {atm.extinction_per_km} /km)",
                color=BAND_COLORS[name])
    ax.set_xlabel("Horizontal range [km]")
    ax.set_ylabel("Path transmittance")
    ax.set_title("Beer-Lambert band-averaged transmittance (clear sea level)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "atmosphere_transmittance.png", dpi=110)
    plt.close(fig)


def fig_netd_vs_fnumber(lwir_dm_dt):
    f_numbers = np.linspace(0.8, 4.0, 200)
    uncooled = [
        1e3 * netd_from_dstar(f, dstar_from_jones(LWIR_DSTAR_JONES), LWIR_PITCH,
                              lwir_dm_dt, LWIR_BANDWIDTH, LWIR_TAU_O)
        for f in f_numbers
    ]
    cooled = []
    for f in f_numbers:
        chain = PhotonDetectorChain(
            optics=Optics(focal_length=0.05, f_number=f, transmittance=MWIR_TAU_O),
            fpa=Fpa(pixel_pitch=MWIR_PITCH, n_columns=MWIR_FORMAT[0],
                    n_rows=MWIR_FORMAT[1], quantum_efficiency=MWIR_QE,
                    well_capacity=MWIR_WELL, read_noise=MWIR_READ),
            band=MWIR, integration_time=MWIR_TINT, scene_temperature=SCENE_T,
        )
        # Cap integration at half-well if the well would saturate.
        t_int = min(MWIR_TINT, chain.max_integration_time(0.5))
        chain = PhotonDetectorChain(
            optics=chain.optics, fpa=chain.fpa, band=MWIR,
            integration_time=t_int, scene_temperature=SCENE_T,
        )
        cooled.append(1e3 * chain.netd())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(f_numbers, uncooled, color=BAND_COLORS["LWIR"],
            label=f"Uncooled LWIR bolometer (D* = {LWIR_DSTAR_JONES:.1e} Jones)")
    ax.plot(f_numbers, cooled, color=BAND_COLORS["MWIR"],
            label=f"Cooled MWIR photon FPA (t_int <= {MWIR_TINT * 1e3:.0f} ms, half-well cap)")
    ax.axhline(50, ls="--", color="gray", lw=1)
    ax.text(3.2, 53, "50 mK requirement", fontsize=8, color="gray")
    ax.set_xlabel("Optics F-number")
    ax.set_ylabel("NETD [mK]")
    ax.set_title("Sensitivity vs optical speed: why uncooled LWIR needs F/1")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "netd_vs_fnumber.png", dpi=110)
    plt.close(fig)


def fig_dri_vs_focal_length(netd_lwir):
    focal_lengths = np.linspace(0.01, 0.12, 200)
    atm = CLEAR_SEA_LEVEL["LWIR"]
    fig, ax = plt.subplots(figsize=(8, 5))
    styles = {"detect": "-", "recognize": "--", "identify": ":"}
    for task, n50 in JOHNSON_N50.items():
        achievable = [
            dri_ranges(HUMAN, f, LWIR_PITCH, TARGET_DT, netd_lwir, atm,
                       SNR_REQUIRED)[task]["achievable_m"] / 1e3
            for f in focal_lengths
        ]
        ax.plot(focal_lengths * 1e3, achievable, styles[task],
                color=BAND_COLORS["LWIR"], label=f"{task} (N50 = {n50})")
    r_contrast = dri_ranges(HUMAN, 0.05, LWIR_PITCH, TARGET_DT, netd_lwir, atm,
                            SNR_REQUIRED)["detect"]["contrast_m"] / 1e3
    ax.set_ylim(0, 5.5)
    ax.text(0.98, 0.96,
            f"sensitivity (contrast) limit = {r_contrast:.1f} km, off scale:\n"
            f"dT tau(R) = {SNR_REQUIRED} x NETD -- sampling dominates",
            transform=ax.transAxes, fontsize=8, color="gray",
            ha="right", va="top")
    ax.axvline(38.4, color="black", ls=":", lw=1)
    ax.text(39, 3.4, "f = 38.4 mm sizing point", fontsize=8, rotation=90)
    ax.set_xlabel("Focal length [mm]")
    ax.set_ylabel("Achievable range [km]")
    ax.set_title(
        f"DRI ranges vs focal length -- human target, {LWIR_PITCH * 1e6:.0f} um "
        f"pitch LWIR, dT = {TARGET_DT} K"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES / "dri_vs_focal_length.png", dpi=110)
    plt.close(fig)


def build_designs():
    """Instantiate the two candidate designs sized for the recognition task."""
    f_lwir = focal_length_for_task(HUMAN, RECOGNITION_RANGE, LWIR_PITCH,
                                   JOHNSON_N50["recognize"])
    f_mwir = focal_length_for_task(HUMAN, RECOGNITION_RANGE, MWIR_PITCH,
                                   JOHNSON_N50["recognize"])
    lwir_optics = Optics(focal_length=f_lwir, f_number=LWIR_FNUM,
                         transmittance=LWIR_TAU_O)
    mwir_optics = Optics(focal_length=f_mwir, f_number=MWIR_FNUM,
                         transmittance=MWIR_TAU_O)
    mwir_chain = PhotonDetectorChain(
        optics=mwir_optics,
        fpa=Fpa(pixel_pitch=MWIR_PITCH, n_columns=MWIR_FORMAT[0],
                n_rows=MWIR_FORMAT[1], quantum_efficiency=MWIR_QE,
                well_capacity=MWIR_WELL, read_noise=MWIR_READ),
        band=MWIR, integration_time=MWIR_TINT, scene_temperature=SCENE_T,
    )
    return lwir_optics, mwir_optics, mwir_chain


def main():
    FIGURES.mkdir(exist_ok=True)
    DOCS.mkdir(exist_ok=True)

    # ---------------- scene radiometry at the reference temperature
    lines: list[str] = []
    add = lines.append
    add("# Computed results\n")
    add("*Machine-generated by `analysis/run_analysis.py` -- do not edit by "
        "hand. The narrative in `physics_analysis.md` cites these numbers.*\n")

    add(f"## Scene radiometry at {SCENE_T:.0f} K\n")
    add(f"- Total exitance sigma T^4 = "
        f"{planck.total_exitance(SCENE_T):.1f} W/m^2; total radiance = "
        f"{planck.total_radiance(SCENE_T):.1f} W/m^2/sr")
    peak = planck.wien_peak_wavelength(SCENE_T)
    add(f"- Wien peak wavelength = {peak * 1e6:.2f} um\n")
    add("| Band | L_band [W/m^2/sr] | Fraction of total | dL/dT [W/m^2/sr/K] "
        "| Relative contrast [%/K] | Photon radiance [ph/s/m^2/sr] |")
    add("|---|---|---|---|---|---|")
    for band in (SWIR, MWIR, LWIR):
        radiance = planck.band_radiance(SCENE_T, band.lam1, band.lam2)
        fraction = planck.blackbody_fraction(SCENE_T, band.lam1, band.lam2)
        contrast = planck.band_radiance_dT(SCENE_T, band.lam1, band.lam2)
        photon = planck.band_photon_radiance(SCENE_T, band.lam1, band.lam2)
        add(f"| {band} | {radiance:.3g} | {fraction:.3g} | {contrast:.3g} | "
            f"{100 * contrast / radiance:.2f} | {photon:.3g} |")
    add("")

    # ---------------- candidate designs
    lwir_optics, mwir_optics, mwir_chain = build_designs()
    lwir_dm_dt = planck.band_exitance_dT(SCENE_T, LWIR.lam1, LWIR.lam2)
    netd_lwir = netd_from_dstar(LWIR_FNUM, dstar_from_jones(LWIR_DSTAR_JONES),
                                LWIR_PITCH, lwir_dm_dt, LWIR_BANDWIDTH,
                                LWIR_TAU_O)
    netd_mwir = mwir_chain.netd()

    add("## Candidate designs (sized to recognize a human at "
        f"{RECOGNITION_RANGE:.0f} m)\n")
    for label, optics, fmt, pitch, band in (
        ("A: uncooled LWIR", lwir_optics, LWIR_FORMAT, LWIR_PITCH, LWIR),
        ("B: cooled MWIR", mwir_optics, MWIR_FORMAT, MWIR_PITCH, MWIR),
    ):
        add(f"### Design {label}")
        add(f"- Focal length {optics.focal_length * 1e3:.1f} mm at "
            f"F/{optics.f_number:g} -> aperture "
            f"{optics.aperture_diameter * 1e3:.1f} mm")
        add(f"- IFOV = {optics.ifov(pitch) * 1e3:.3f} mrad; FOV = "
            f"{math.degrees(optics.field_of_view(fmt[0], pitch)):.1f} deg x "
            f"{math.degrees(optics.field_of_view(fmt[1], pitch)):.1f} deg")
        add(f"- Airy null radius at band centre = "
            f"{optics.airy_radius(band.center) * 1e6:.1f} um vs pitch "
            f"{pitch * 1e6:.0f} um; Q = lam F#/p = "
            f"{optics.q_parameter(band.center, pitch):.2f}")
        add("")
    add(f"- Design A NETD (Lloyd, D* = {LWIR_DSTAR_JONES:.1e} Jones, "
        f"df = {LWIR_BANDWIDTH:.0f} Hz): **{netd_lwir * 1e3:.0f} mK**")
    add(f"- Design B NETD (shot-noise chain, t_int = {MWIR_TINT * 1e3:.0f} ms, "
        f"well fill {mwir_chain.well_fill() * 100:.0f}%, "
        f"{mwir_chain.signal_electrons():.3g} e-): "
        f"**{netd_mwir * 1e3:.1f} mK**")
    add(f"- Design B electrons per kelvin: "
        f"{mwir_chain.electrons_per_kelvin():.3g} e-/K; noise "
        f"{mwir_chain.noise_electrons():.0f} e- rms\n")

    # ---------------- DRI table
    add("## DRI ranges, human target "
        f"(d_c = {HUMAN} m, dT0 = {TARGET_DT} K, SNR >= {SNR_REQUIRED})\n")
    add("| Design | Task | Sampling-limited [m] | Contrast-limited [m] | "
        "Achievable [m] |")
    add("|---|---|---|---|---|")
    for label, optics, pitch, band, netd in (
        ("A (LWIR)", lwir_optics, LWIR_PITCH, "LWIR", netd_lwir),
        ("B (MWIR)", mwir_optics, MWIR_PITCH, "MWIR", netd_mwir),
    ):
        table = dri_ranges(HUMAN, optics.focal_length, pitch, TARGET_DT, netd,
                           CLEAR_SEA_LEVEL[band], SNR_REQUIRED)
        for task in ("detect", "recognize", "identify"):
            row = table[task]
            add(f"| {label} | {task} | {row['sampling_m']:.0f} | "
                f"{row['contrast_m']:.0f} | {row['achievable_m']:.0f} |")
    add("")

    (DOCS / "computed_results.md").write_text("\n".join(lines))

    fig_planck_curves()
    fig_band_radiance_vs_temperature()
    fig_thermal_contrast()
    fig_atmosphere()
    fig_netd_vs_fnumber(lwir_dm_dt)
    fig_dri_vs_focal_length(netd_lwir)

    print("\n".join(lines))
    print(f"\nFigures written to {FIGURES}/")


if __name__ == "__main__":
    main()
