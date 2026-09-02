#!/usr/bin/env python3
"""Viewing-angle and reflected-sunlight error budget for the nozzle pyrometer.

Instrument: the same IMX900 LCG / F/4 / 620+870 nm model as the rest of
the analysis. Emissivity follows the smooth-surface Fresnel model with
illustrative optical constants; sunlight is the AM1.5G table reflected
diffusely. Outputs figures/geometry_*.png, figures/sun_*.png and
docs/computed_geometry_sun_results.md.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ircam.pyrometry import NotchFilter, RatioPyrometer
from ircam.sensors import imx900_camera
from ircam.surface import (
    MATERIALS,
    one_band_apparent_temperature,
    ratio_apparent_temperature,
    solar_reflected_electron_rate,
    specular_glint_ratio,
    thermal_electron_rate,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES, DOCS = ROOT / "figures", ROOT / "docs"

# Palette (validated reference set, fixed order).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
MUTED, GRID = "#898781", "#e1e0d9"

CAM = imx900_camera(4.0, 0.85, "lcg")
F620, F870 = NotchFilter(620e-9, 30e-9), NotchFilter(870e-9, 50e-9)
PYRO = RatioPyrometer(F620, F870, CAM)
GRAPHITE = MATERIALS["graphite-like"]
MAT_COLORS = dict(zip(MATERIALS, (BLUE, ORANGE, AQUA)))


def style(ax, xlabel, ylabel, title=None):
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if title:
        ax.set_title(title, fontsize=11)


def bias_one(t, mat, tv, ts=0.0, sun=0.0, cal=0.0, resid=1.0):
    return one_band_apparent_temperature(t, F620, CAM, mat, tv, ts, sun, cal, resid) - t


def bias_two(t, mat, tv, ts=0.0, sun=0.0, cal=0.0, resid=1.0):
    return ratio_apparent_temperature(t, PYRO, mat, tv, ts, sun, cal, resid) - t


def fig_emissivity():
    theta = np.radians(np.linspace(0, 88, 120))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, mat in MATERIALS.items():
        e1, e2 = mat.emissivity(theta, 620e-9), mat.emissivity(theta, 870e-9)
        ax1.plot(np.degrees(theta), e1, color=MAT_COLORS[name], lw=2, label=name)
        ax1.plot(np.degrees(theta), e2, color=MAT_COLORS[name], lw=2, ls="--")
        ax2.plot(np.degrees(theta), (e1 / e2) / (e1[0] / e2[0]), color=MAT_COLORS[name],
                 lw=2, label=name)
    ax1.text(0.98, 0.97, "solid: 620 nm   dashed: 870 nm", fontsize=9, color=MUTED,
             transform=ax1.transAxes, ha="right", va="top")
    style(ax1, "Viewing angle from surface normal [deg]", "Directional emissivity",
          "Emissivity collapses at grazing angles (one-band sees this directly)")
    style(ax2, "Viewing angle from surface normal [deg]",
          "(eps620/eps870) relative to normal incidence",
          "What survives in the ratio")
    ax2.axhline(1.0, color=MUTED, lw=0.8)
    ax1.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "geometry_emissivity.png", dpi=110)
    plt.close(fig)


def fig_angle_error():
    angles = np.linspace(0, 88, 45)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True)
    for ax, (name, mat) in zip(axes, MATERIALS.items()):
        for t_c, ls in ((1500, "-"), (3000, "--")):
            t = t_c + 273.15
            one = [bias_one(t, mat, math.radians(a)) for a in angles]
            two = [bias_two(t, mat, math.radians(a)) for a in angles]
            ax.plot(angles, one, color=BLUE, lw=2, ls=ls, label=f"one band, {t_c} C")
            ax.plot(angles, two, color=ORANGE, lw=2, ls=ls, label=f"ratio, {t_c} C")
        ax.axhline(0, color=MUTED, lw=0.8)
        style(ax, "Viewing angle [deg]", "Apparent - true temperature [K]" if ax is axes[0] else "",
              name)
        ax.set_ylim(-450, 60)
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("Bias from a normal-incidence emissivity calibration vs viewing angle "
                 "(no sun)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "geometry_angle_error.png", dpi=110)
    plt.close(fig)


def fig_sun_vs_temperature():
    temps_c = np.linspace(1000, 3000, 41)
    tv = ts = math.radians(45)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Left: fraction of channel signal that is reflected sun.
    for filt, color, label in ((F620, BLUE, "620 nm channel"), (F870, ORANGE, "870 nm channel")):
        frac = [solar_reflected_electron_rate(filt, CAM, GRAPHITE, tv, ts, 1.0)
                / thermal_electron_rate(tc + 273.15, filt, CAM, GRAPHITE, tv)
                for tc in temps_c]
        ax1.semilogy(temps_c, np.array(frac) * 100, color=color, lw=2, label=label)
    ax1.axhline(1, color=MUTED, lw=0.8, ls=":")
    ax1.text(2700, 1.2, "1 %", color=MUTED, fontsize=8)
    style(ax1, "Scene temperature [C]", "Reflected sun / thermal signal [%]",
          "Sun contaminates the short band far more")
    ax1.legend(fontsize=9, frameon=False)
    # Right: resulting bias, with and without pre-ignition subtraction.
    for resid, ls, tag in ((1.0, "-", "no subtraction"), (0.1, ":", "90% subtracted")):
        one = [bias_one(tc + 273.15, GRAPHITE, tv, ts, 1.0, tv, resid) for tc in temps_c]
        two = [bias_two(tc + 273.15, GRAPHITE, tv, ts, 1.0, tv, resid) for tc in temps_c]
        ax2.semilogy(temps_c, one, color=BLUE, lw=2, ls=ls, label=f"one band, {tag}")
        ax2.semilogy(temps_c, two, color=ORANGE, lw=2, ls=ls, label=f"ratio, {tag}")
    style(ax2, "Scene temperature [C]", "Warm bias [K]",
          "Full sun at 45 deg, graphite-like, viewed at 45 deg")
    ax2.set_ylim(0.1, 2000)
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "sun_error_vs_temperature.png", dpi=110)
    plt.close(fig)


def main():
    FIGURES.mkdir(exist_ok=True)
    lines: list[str] = []
    add = lines.append
    add("# Computed results: viewing angle and reflected sunlight\n")
    add("*Machine-generated by `analysis/nozzle_geometry_sun.py` -- do not edit by "
        "hand. Instrument: IMX900 LCG, F/4, 620/870 nm (30/50 nm). Optical "
        "constants are illustrative class values; sunlight is AM1.5G, diffusely "
        "reflected.*\n")

    add("## A. Angle-only bias (normal-incidence emissivity calibration, no sun)\n")
    add("| Material | Angle | One band @1500 C | Ratio @1500 C | One band @3000 C | Ratio @3000 C |")
    add("|---|---|---|---|---|---|")
    for name, mat in MATERIALS.items():
        for deg in (60, 75, 85):
            tv = math.radians(deg)
            add(f"| {name} | {deg} deg | {bias_one(1773.15, mat, tv):+.0f} K | "
                f"{bias_two(1773.15, mat, tv):+.0f} K | {bias_one(3273.15, mat, tv):+.0f} K | "
                f"{bias_two(3273.15, mat, tv):+.0f} K |")
    add("")

    add("## B. Sun-only bias (graphite-like, viewed at 45 deg, full sun at 45 deg, "
        "calibrated at the viewing angle)\n")
    add("| Scene T | Sun / signal, 620 | Sun / signal, 870 | One band | Ratio | "
        "Ratio, 90% pre-ignition subtraction |")
    add("|---|---|---|---|---|---|")
    tv = ts = math.radians(45)
    for tc in (1200, 1500, 2000, 2500, 3000):
        t = tc + 273.15
        f1 = solar_reflected_electron_rate(F620, CAM, GRAPHITE, tv, ts, 1.0) / \
            thermal_electron_rate(t, F620, CAM, GRAPHITE, tv)
        f2 = solar_reflected_electron_rate(F870, CAM, GRAPHITE, tv, ts, 1.0) / \
            thermal_electron_rate(t, F870, CAM, GRAPHITE, tv)
        add(f"| {tc} C | {100 * f1:.2f} % | {100 * f2:.2f} % | "
            f"{bias_one(t, GRAPHITE, tv, ts, 1.0, tv):+.0f} K | "
            f"{bias_two(t, GRAPHITE, tv, ts, 1.0, tv):+.0f} K | "
            f"{bias_two(t, GRAPHITE, tv, ts, 1.0, tv, 0.1):+.0f} K |")
    add("")

    add("## C. Combined scenario: graphite-like, viewed at 70 deg, full sun at 45 deg, "
        "normal-incidence calibration\n")
    add("| Scene T | One band: angle | One band: sun | One band: both | Ratio: angle | "
        "Ratio: sun | Ratio: both | Shot noise, 2x2 x 8 frames |")
    add("|---|---|---|---|---|---|---|---|")
    tv, ts = math.radians(70), math.radians(45)
    from ircam.pyrometry import exposure_for_well_fill
    e1 = exposure_for_well_fill(3273.15, F620, CAM, 0.6)
    e2 = exposure_for_well_fill(3273.15, F870, CAM, 0.6)
    for tc in (1500, 2250, 3000):
        t = tc + 273.15
        add(f"| {tc} C | {bias_one(t, GRAPHITE, tv):+.0f} K | "
            f"{bias_one(t, GRAPHITE, tv, ts, 1.0, tv):+.0f} K | "
            f"{bias_one(t, GRAPHITE, tv, ts, 1.0):+.0f} K | "
            f"{bias_two(t, GRAPHITE, tv):+.0f} K | "
            f"{bias_two(t, GRAPHITE, tv, ts, 1.0, tv):+.0f} K | "
            f"{bias_two(t, GRAPHITE, tv, ts, 1.0):+.0f} K | "
            f"{PYRO.sigma_T(t, e1, e2, binning=2, frames=8):.0f} K |")
    add("")

    add("## D. Specular sun glint hazard at 3000 C (glint radiance / surface radiance)\n")
    add("| Viewing angle | 620 nm | 870 nm |")
    add("|---|---|---|")
    for deg in (30, 45, 70):
        add(f"| {deg} deg | {specular_glint_ratio(math.radians(deg), 620e-9, 3273.15, GRAPHITE):.1f}x | "
            f"{specular_glint_ratio(math.radians(deg), 870e-9, 3273.15, GRAPHITE):.1f}x |")
    add("")

    (DOCS / "computed_geometry_sun_results.md").write_text("\n".join(lines))
    fig_emissivity()
    fig_angle_error()
    fig_sun_vs_temperature()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
