"""Range performance: sampling-limited DRI ranges and contrast-limited range.

Johnson-criteria style analysis. N50 is the number of resolvable cycles
across the target critical dimension for a 50% probability of the task.
We use the NVESD two-dimensional target values:

    detect     N50 = 0.75
    recognize  N50 = 3.0
    identify   N50 = 6.0

(The classic 1957 one-dimensional Johnson values are 1.0 / 4.0 / 6.4;
modern practice uses the TTP metric, which needs full MTF and noise
spectra -- see the limitations section of the analysis document.)

Two limits are computed:

* Sampling (Nyquist) limit: the sensor resolves at best one cycle per
  two pixels, so the cycles across a target of critical dimension d_c at
  range R are  n = d_c f / (2 p R),  giving

      R_task = d_c f / (2 p N50).

  This assumes optics MTF is adequate at Nyquist (Q <~ 1); check the
  Q parameter separately.

* Contrast (sensitivity) limit: the apparent temperature difference
  after atmospheric extinction must exceed a multiple of NETD:

      dT0 * exp(-beta R) >= k * NETD  =>  R = ln(dT0 / (k NETD)) / beta.

The achievable task range is the smaller of the two.
"""

from __future__ import annotations

import math

from .atmosphere import SimpleAtmosphere

#: N50 cycles across the critical dimension (2-D target set).
JOHNSON_N50 = {"detect": 0.75, "recognize": 3.0, "identify": 6.0}

#: Typical critical dimensions [m]: sqrt(height * width) of the target.
CRITICAL_DIMENSION = {"human": 0.75, "vehicle": 2.3}


def cycles_on_target(critical_dim: float, focal_length: float, pixel_pitch: float,
                     range_m: float) -> float:
    """Resolvable cycles across the target critical dimension at range."""
    return critical_dim * focal_length / (2.0 * pixel_pitch * range_m)


def sampling_limited_range(critical_dim: float, focal_length: float,
                           pixel_pitch: float, n50: float) -> float:
    """Range [m] at which N50 cycles fit across the critical dimension."""
    return critical_dim * focal_length / (2.0 * pixel_pitch * n50)


def contrast_limited_range(delta_t0: float, netd: float,
                           atmosphere: SimpleAtmosphere,
                           snr_required: float = 2.5) -> float:
    """Range [m] at which the apparent dT falls to snr_required * NETD.

    delta_t0 is the inherent target-background temperature difference [K].
    Returns 0.0 when the required contrast exceeds delta_t0 even at zero
    range (task impossible at any range with this sensitivity).
    """
    ratio = delta_t0 / (snr_required * netd)
    if ratio <= 1.0:
        return 0.0
    return 1e3 * math.log(ratio) / atmosphere.extinction_per_km


def dri_ranges(critical_dim: float, focal_length: float, pixel_pitch: float,
               delta_t0: float, netd: float, atmosphere: SimpleAtmosphere,
               snr_required: float = 2.5) -> dict[str, dict[str, float]]:
    """Sampling, contrast, and combined ranges [m] for each DRI task."""
    r_contrast = contrast_limited_range(delta_t0, netd, atmosphere, snr_required)
    out: dict[str, dict[str, float]] = {}
    for task, n50 in JOHNSON_N50.items():
        r_sampling = sampling_limited_range(critical_dim, focal_length, pixel_pitch, n50)
        out[task] = {
            "sampling_m": r_sampling,
            "contrast_m": r_contrast,
            "achievable_m": min(r_sampling, r_contrast),
        }
    return out


def focal_length_for_task(critical_dim: float, range_m: float, pixel_pitch: float,
                          n50: float) -> float:
    """Focal length [m] required to support N50 cycles at the given range."""
    return 2.0 * pixel_pitch * n50 * range_m / critical_dim
