"""Two-notch ratio pyrometry designer -- interactive companion to
docs/nozzle_pyrometry.md.

Four stories:
  1. Notch spacing & width vs temperature sensitivity
  2. Why one calibrated notch is inaccurate under uncertain radiometry
  3. Temperature certainty vs scene temperature for the selected pair
  4. Viewing angle (directional emissivity), reflected sunlight and
     plume light reflected by the wall
Sensor: selectable in the sidebar -- the tabulated Sony IMX900 curve, a set
of approximate silicon classes from a three-parameter QE model, or a
user-defined QE table (see ircam/sensors.py).

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, point Streamlit Community Cloud at this file.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ircam.constants import C2
from ircam.pyrometry import (
    NotchFilter,
    PyroCamera,
    electron_rate,
    equivalent_wavelength,
)
from ircam.sensors import (
    IMX900_BASLER_EMVA_PEAK_QE,
    IMX900_GAIN_MODES,
    IMX900_QE_TABLE_PEAK,
    IMX900_SPECS,
    SENSOR_PRESETS,
    camera_from_spec,
    qe_from_spec,
)
from ircam.surface import (
    MATERIALS,
    OpticalConstants,
    PlumeSource,
    RatioPyrometer,
    one_band_apparent_temperature,
    plume_reflected_electron_rate,
    ratio_apparent_temperature,
    solar_reflected_electron_rate,
    specular_glint_ratio,
    thermal_electron_rate,
)

# ---------------------------------------------------------------- palette
# Reference dataviz palette (light mode), categorical slots in fixed order.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

st.set_page_config(page_title="Nozzle pyrometry designer", page_icon=":fire:",
                   layout="wide")


def hline_log(fig, y, label, dash="dash"):
    """Reference line + label on a log-y chart.

    plotly's add_hline(annotation_text=...) mis-places its annotation on log
    axes and wrecks the autorange, so draw the label ourselves with log10
    data coordinates.
    """
    fig.add_hline(y=y, line=dict(color=MUTED, width=1, dash=dash))
    fig.add_annotation(x=0.99, xref="x domain", y=math.log10(y), yref="y",
                       text=label, showarrow=False, yshift=8,
                       font=dict(color=MUTED, size=11))


def style(fig, xtitle, ytitle, logy=False, height=420):
    fig.update_layout(
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(color=INK, family='system-ui, -apple-system, "Segoe UI", sans-serif'),
        margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0,
                    font=dict(size=12, color=INK2)),
    )
    ax = dict(gridcolor=GRID, linecolor=BASELINE, tickfont=dict(color=MUTED),
              title_font=dict(color=INK2, size=13), zeroline=False)
    fig.update_xaxes(title_text=xtitle, **ax)
    fig.update_yaxes(title_text=ytitle, type="log" if logy else "linear",
                     dtick=1 if logy else None, **ax)
    return fig


# ---------------------------------------------------------------- physics
# Wien-limit slope d(lnR)/dT is used throughout for speed; it agrees with
# the full band integrals to <3% over this app's range (see tests).

def anti_sat_exposure(filt, cam, t_max_k, fill, fps):
    """Exposure anti-saturated for the hottest scene content, frame-capped."""
    return min(fill * cam.well_capacity / electron_rate(t_max_k, filt, cam),
               0.95 / fps)


def sigma_ratio_vs_t(temps_k, l1, w1, l2, w2, cam, t_max_k, fill, fps,
                     binning=1, frames=1):
    """NEdT [K] of the two-notch ratio vs scene temperature (vectorized)."""
    n_avg = binning**2 * frames
    rel_var = np.zeros_like(temps_k)
    exposures = []
    for center, width in ((l1, w1), (l2, w2)):
        filt = NotchFilter(center, width)
        t_exp = anti_sat_exposure(filt, cam, t_max_k, fill, fps)
        exposures.append(t_exp)
        n_e = electron_rate(temps_k, filt, cam) * t_exp * n_avg
        rel_var = rel_var + (n_e + n_avg * cam.read_noise**2) / n_e**2
    slope = C2 / temps_k**2 * (1.0 / l1 - 1.0 / l2)
    return np.sqrt(rel_var) / slope, exposures


@st.cache_data(show_spinner="Computing notch-pair map...")
def pair_map(t_eval_k, qe_spec, pixel_pitch, f_number, tau, read_noise, well,
             t_max_k, fill, fps):
    """Worst-case NEdT over a (lambda1, lambda2) grid."""
    cam = camera_from_spec(qe_spec, pixel_pitch, f_number, tau, read_noise, well)
    l1_grid = np.linspace(420e-9, 760e-9, 35)
    l2_grid = np.linspace(700e-9, 975e-9, 29)
    z = np.full((len(l2_grid), len(l1_grid)), np.nan)
    for i, l2 in enumerate(l2_grid):
        for j, l1 in enumerate(l1_grid):
            if l2 - l1 >= 60e-9:
                sig, _ = sigma_ratio_vs_t(np.array([t_eval_k]), l1, 30e-9,
                                          l2, 50e-9, cam, t_max_k, fill, fps)
                z[i, j] = sig[0]
    return l1_grid, l2_grid, z


def _material(name, nk):
    if name in MATERIALS:
        return MATERIALS[name]
    n1, k1, n2, k2 = nk
    return OpticalConstants("custom", (620e-9, 870e-9), (n1, n2), (k1, k2))


def _plume(pp):
    t_pl, eps, alpha, fview, glare, resid = pp
    if eps <= 0.0:
        return None
    return PlumeSource(t_pl, eps, alpha, 870e-9, fview, glare, resid)


@st.cache_data(show_spinner=False)
def bias_vs_angle(mat_name, nk, t_k, theta_s_deg, sun, resid, pp, l1, w1, l2, w2,
                  f_number, tau, qe_spec, pixel_pitch, read_noise, well):
    cam = camera_from_spec(qe_spec, pixel_pitch, f_number, tau, read_noise, well)
    mat = _material(mat_name, nk)
    f1, f2 = NotchFilter(l1, w1), NotchFilter(l2, w2)
    pyro = RatioPyrometer(f1, f2, cam)
    plume = _plume(pp)
    angles = np.linspace(0, 88, 45)
    ts = math.radians(theta_s_deg)
    one = [one_band_apparent_temperature(t_k, f1, cam, mat, math.radians(a), ts, sun,
                                         0.0, resid, plume) - t_k for a in angles]
    two = [ratio_apparent_temperature(t_k, pyro, mat, math.radians(a), ts, sun,
                                      0.0, resid, plume) - t_k for a in angles]
    return angles, np.array(one), np.array(two)


@st.cache_data(show_spinner=False)
def bias_vs_temperature(mat_name, nk, theta_v_deg, theta_s_deg, sun, resid, pp, l1, w1,
                        l2, w2, f_number, tau, qe_spec, pixel_pitch, read_noise, well):
    cam = camera_from_spec(qe_spec, pixel_pitch, f_number, tau, read_noise, well)
    mat = _material(mat_name, nk)
    f1, f2 = NotchFilter(l1, w1), NotchFilter(l2, w2)
    pyro = RatioPyrometer(f1, f2, cam)
    plume = _plume(pp)
    temps_c = np.linspace(1000, 3000, 41)
    tv, ts = math.radians(theta_v_deg), math.radians(theta_s_deg)
    one = [one_band_apparent_temperature(tc + 273.15, f1, cam, mat, tv, ts, sun, 0.0,
                                         resid, plume) - (tc + 273.15) for tc in temps_c]
    two = [ratio_apparent_temperature(tc + 273.15, pyro, mat, tv, ts, sun, 0.0,
                                      resid, plume) - (tc + 273.15) for tc in temps_c]
    return temps_c, np.array(one), np.array(two)


# ---------------------------------------------------------------- sidebar
SENSOR_CHOICES = {key: p.name for key, p in SENSOR_PRESETS.items()}
SENSOR_CHOICES["parametric"] = "Define by parameters"
SENSOR_CHOICES["table"] = "Define by QE table"

with st.sidebar.expander("Sensor & QE curve", expanded=True):
    sensor_key = st.selectbox(
        "Sensor / QE curve", list(SENSOR_CHOICES),
        format_func=lambda k: SENSOR_CHOICES[k],
        help="Only the IMX900 curve is a tabulated measurement. The other "
             "presets are approximate silicon classes from a three-parameter "
             "model (peak QE, effective absorption depth, blue edge). Use them "
             "to bracket a sensor you have not chosen yet, or define your own.")
    gain_mode = "lcg"
    if sensor_key == "imx900":
        preset = SENSOR_PRESETS["imx900"]
        gain_mode = st.radio(
            "Conversion-gain mode", ["lcg", "hcg"], horizontal=True,
            format_func=lambda m: {"lcg": "LCG (large well)",
                                   "hcg": "HCG (low noise)"}[m],
            help="LCG: FRAMOS EMVA 1288, 9458 e- well / 5.56 e- read. "
                 "HCG: PTC-measured, 2183 e- well / 1.39 e- read. A scene with "
                 "3000 C content is well-limited, so LCG is the default.")
        mode = IMX900_GAIN_MODES[gain_mode]
        default_read, default_well = mode["read_noise"], mode["well_capacity"]
        QE_SPEC = ("preset", "imx900")
        SENSOR_NOTE = preset.provenance
    elif sensor_key == "parametric":
        preset = SENSOR_PRESETS["bsi_2p74"]
        peak_qe = st.slider("Peak QE", 0.30, 1.00, 0.80, 0.01)
        depth_um = st.slider(
            "Effective absorption depth [um]", 2.0, 25.0, 8.0, 0.5,
            help="Photodiode thickness times any light-trapping path gain. "
                 "Silicon absorbs weakly past 800 nm, so this sets how much QE "
                 "survives at the long notch (FSI ~4-5 um, BSI ~10 um, "
                 "NIR-enhanced ~15+ um).")
        blue_edge_nm = st.slider(
            "Blue edge [nm]", 350, 480, 400, 5,
            help="Centre of the short-wavelength roll-off: front-side layers "
                 "absorb blue on FSI sensors (~430 nm); BSI rolls off later "
                 "(~360 nm).")
        default_read, default_well = preset.read_noise, preset.well_capacity
        QE_SPEC = ("parametric", peak_qe, depth_um * 1e-6, blue_edge_nm * 1e-9)
        SENSOR_NOTE = ("Approximate QE from the c-Si absorption coefficient: "
                       "QE = A [1 - exp(-alpha(lam) d)] s(lam) with peak "
                       f"{peak_qe:.2f}, d = {depth_um:g} um, blue edge "
                       f"{blue_edge_nm} nm. A class model, not a measured curve.")
    elif sensor_key == "table":
        preset = SENSOR_PRESETS["fsi_3p45"]
        st.caption("Edit the points (add rows as needed). Linear interpolation "
                   "between points, zero outside the table.")
        default_pts = pd.DataFrame({
            "wavelength_nm": [400, 500, 550, 600, 700, 800, 900, 1000],
            "qe_pct": [45, 70, 75, 75, 65, 45, 25, 7]})
        edited = st.data_editor(
            default_pts, num_rows="dynamic", hide_index=True, key="qe_table",
            column_config={
                "wavelength_nm": st.column_config.NumberColumn(
                    "Wavelength [nm]", min_value=300, max_value=1200, step=5),
                "qe_pct": st.column_config.NumberColumn(
                    "QE [%]", min_value=0, max_value=100, step=1)})
        by_lam = {}
        for lam_nm_pt, q in zip(edited["wavelength_nm"], edited["qe_pct"]):
            if pd.notna(lam_nm_pt) and pd.notna(q):
                by_lam[float(lam_nm_pt) * 1e-9] = float(q) / 100.0
        pts = tuple(sorted(by_lam.items()))
        if len(pts) < 2:
            st.error("Enter at least two (wavelength, QE) points.")
            st.stop()
        default_read, default_well = preset.read_noise, preset.well_capacity
        QE_SPEC = ("table", pts)
        SENSOR_NOTE = f"QE from {len(pts)} user-entered points, linearly interpolated."
    else:
        preset = SENSOR_PRESETS[sensor_key]
        default_read, default_well = preset.read_noise, preset.well_capacity
        QE_SPEC = ("preset", sensor_key)
        SENSOR_NOTE = preset.provenance
        st.caption(preset.provenance)
    SENSOR_NAME = ({"parametric": "parametric silicon sensor",
                    "table": "user-defined QE table"}.get(sensor_key, preset.short_name))
    pixel_pitch_um = st.number_input("Pixel pitch [um]", 1.0, 20.0,
                                     preset.pixel_pitch * 1e6, 0.05,
                                     key=f"pitch_{sensor_key}")
    read_noise = st.number_input("Read noise [e- rms]", 0.1, 50.0,
                                 float(default_read), 0.1,
                                 key=f"rn_{sensor_key}_{gain_mode}")
    well = st.number_input("Full well [e-]", 500.0, 500000.0,
                           float(default_well), 100.0,
                           key=f"well_{sensor_key}_{gain_mode}")
    st.caption("With a hot pixel in frame the exposure is saturation-capped, so "
               "the QE level cancels and the **full well** sets precision; the "
               "QE shape still weights the bands and sets where read noise and "
               "the minimum exposure bite.")


st.sidebar.header("Notch pair")
l1_nm = st.sidebar.slider("Short notch center [nm]", 420, 750, 620, 5)
w1_nm = st.sidebar.slider("Short notch width [nm]", 10, 80, 30, 5)
l2_nm = st.sidebar.slider("Long notch center [nm]", max(700, l1_nm + 60),
                          975, max(870, l1_nm + 60), 5)
w2_nm = st.sidebar.slider("Long notch width [nm]", 10, 80, 50, 5)

st.sidebar.header("Optics & exposure")
f_number = st.sidebar.select_slider("F-number", [1.4, 2, 2.8, 4, 5.6, 8, 11, 16],
                                    value=4.0)
tau_optics = st.sidebar.slider("Optics + window transmittance", 0.5, 0.95, 0.85)
nd_od = st.sidebar.slider("ND filter optical density", 0.0, 3.0, 0.0, 0.5,
                          help="Lengthens exposures without changing precision; "
                               "use it to stay above the sensor's minimum exposure.")
t_max_c = st.sidebar.slider("Hottest scene content [C] (sets exposure)",
                            2000, 3200, 3000, 50)
fill = st.sidebar.slider("Target well fill at hottest point", 0.3, 0.8, 0.6)
fps = st.sidebar.select_slider("Frame rate [fps]", [30, 60, 120], value=60)

st.sidebar.header("Averaging")
binning = st.sidebar.select_slider("Pixel binning", [1, 2, 4], value=2)
frames = st.sidebar.slider("Frames averaged", 1, 32, 8)

PIXEL_PITCH = pixel_pitch_um * 1e-6
MIN_EXPOSURE = preset.min_exposure
QE_FN = qe_from_spec(QE_SPEC)
TAU_EFF = tau_optics * 10.0**(-nd_od)
CAM = camera_from_spec(QE_SPEC, PIXEL_PITCH, f_number, TAU_EFF, read_noise, well)
L1, W1, L2, W2 = l1_nm * 1e-9, w1_nm * 1e-9, l2_nm * 1e-9, w2_nm * 1e-9
T_MAX_K = t_max_c + 273.15
LAM_EQ = equivalent_wavelength(L1, L2)

st.title("Two-notch ratio pyrometry designer")
MODE_NOTE = f", {gain_mode.upper()} mode" if sensor_key == "imx900" else ""
st.caption(f"Engine-nozzle thermography, 1500-3000 C primary band. Sensor: "
           f"**{SENSOR_NAME}** ({PIXEL_PITCH * 1e6:.2f} um pixels, "
           f"{well / 1e3:.2f} ke- well, {read_noise:g} e- read{MODE_NOTE}). "
           f"Selected pair: **{l1_nm} / {l2_nm} nm**, equivalent wavelength "
           f"lam_eq = {LAM_EQ * 1e9:.0f} nm.")

tab1, tab2, tab3, tab4 = st.tabs([
    "1 - Notch spacing & width",
    "2 - One notch vs two: uncertain radiometry",
    "3 - Temperature certainty vs temperature",
    "4 - Angle, sunlight & plume reflection",
])

# ============================================================ story 1
with tab1:
    st.subheader("Spacing wins -- until the moving notch runs out of photons")
    st.markdown(
        "Temperature error scales with the **equivalent wavelength** "
        "lam_eq = lam1 lam2/(lam2 - lam1): sigma_T = (lam_eq T^2/c2) x "
        "(ratio noise). Each solid curve moves **one** notch while the other "
        "stays at its sidebar setting: blue slides the short notch with the "
        f"long one held at {l2_nm} nm, orange slides the long notch with the "
        f"short one held at {l1_nm} nm. Both pass through the selected pair "
        "(circles) at the same NEdT. The dashed curves are what the lam_eq "
        "law alone predicts if the ratio noise stayed at its selected-pair "
        "value -- spreading the notches always helps there. The solid curves "
        "peel away from them where a notch runs out of photons: the exposure "
        "is pinned by the hottest pixel in frame, so the short notch collapses "
        "as it moves blue. Stars mark the best position of each notch with "
        "the other held."
    )
    ctl1, ctl2 = st.columns([3, 1])
    t_eval_c = ctl1.slider("Evaluate precision at scene temperature [C]",
                           1300, 3000, 1500, 50, key="teval1")
    show_law = ctl2.checkbox("Show lam_eq law", value=True, key="law1")
    t_eval_k = t_eval_c + 273.15

    def nedt_pair(l1, l2):
        return sigma_ratio_vs_t(np.array([t_eval_k]), l1, W1, l2, W2, CAM,
                                T_MAX_K, fill, fps)[0][0]

    sig_sel = nedt_pair(L1, L2)
    l1_sweep = np.linspace(420e-9, L2 - 60e-9, 60)
    l2_sweep = np.linspace(L1 + 60e-9, 975e-9, 60)
    sig_short = np.array([nedt_pair(l1, L2) for l1 in l1_sweep])
    sig_long = np.array([nedt_pair(L1, l2) for l2 in l2_sweep])
    law_short = sig_sel * equivalent_wavelength(l1_sweep, L2) / LAM_EQ
    law_long = sig_sel * equivalent_wavelength(L1, l2_sweep) / LAM_EQ
    i_s, i_l = int(np.argmin(sig_short)), int(np.argmin(sig_long))

    fig = go.Figure()
    for x, y, color, name in (
        (l1_sweep, sig_short, BLUE,
         f"short notch moves (long held at {l2_nm} nm)"),
        (l2_sweep, sig_long, ORANGE,
         f"long notch moves (short held at {l1_nm} nm)"),
    ):
        fig.add_trace(go.Scatter(
            x=x * 1e9, y=y, name=name, line=dict(color=color, width=2),
            hovertemplate="%{x:.0f} nm: NEdT = %{y:.0f} K<extra></extra>"))
    if show_law:
        for x, y, color in ((l1_sweep, law_short, BLUE),
                            (l2_sweep, law_long, ORANGE)):
            fig.add_trace(go.Scatter(
                x=x * 1e9, y=y, line=dict(color=color, width=1.2, dash="dash"),
                name="lam_eq law alone (ratio noise held at its selected-pair value)",
                legendgroup="law", showlegend=color == BLUE,
                hovertemplate="%{x:.0f} nm: %{y:.0f} K (lam_eq law)<extra></extra>"))
    for xs, ys, color, label in (
        (l1_sweep[i_s], sig_short[i_s], BLUE,
         f"best short notch {l1_sweep[i_s] * 1e9:.0f} nm"),
        (l2_sweep[i_l], sig_long[i_l], ORANGE,
         f"best long notch {l2_sweep[i_l] * 1e9:.0f} nm"),
    ):
        fig.add_trace(go.Scatter(
            x=[xs * 1e9], y=[ys], mode="markers+text",
            marker=dict(color=color, size=11, symbol="star"),
            text=[label], textposition="top center", cliponaxis=False,
            textfont=dict(color=INK2, size=11), showlegend=False,
            hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[l1_nm, l2_nm], y=[sig_sel, sig_sel], mode="markers",
        marker=dict(color=[BLUE, ORANGE], size=9,
                    line=dict(color=INK, width=1)),
        name=f"selected pair {l1_nm}/{l2_nm} nm: {sig_sel:.0f} K",
        hovertemplate="selected pair: %{y:.0f} K<extra></extra>"))
    for x, label in ((l1_nm, f"short notch {l1_nm} nm"),
                     (l2_nm, f"long notch {l2_nm} nm")):
        fig.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1),
                      annotation_text=label, annotation_position="bottom right",
                      annotation_font_color=MUTED)
    st.plotly_chart(style(
        fig, "Center wavelength of the notch being moved [nm]",
        f"NEdT at {t_eval_c} C [K]  (per pixel, per frame)", logy=True),
        width="stretch")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Width buys nothing once saturation-capped.** With a "
            f"{t_max_c} C pixel in frame, widening the filter just forces a "
            "shorter exposure -- electron counts and precision are unchanged "
            "(blue). Width only helps when the exposure is capped by frame "
            "time instead (dim scene / no hot content, aqua)."
        )
        w_sweep = np.linspace(5e-9, 80e-9, 40)
        sat, frozen = [], []
        filt_ref = NotchFilter(L1, 30e-9)
        t_frozen = anti_sat_exposure(filt_ref, CAM, T_MAX_K, fill, fps)
        for w in w_sweep:
            filt = NotchFilter(L1, w)
            t_exp = anti_sat_exposure(filt, CAM, T_MAX_K, fill, fps)
            slope = C2 / t_eval_k**2 * (1.0 / L1 - 1.0 / L2)
            n_long = electron_rate(t_eval_k, NotchFilter(L2, W2), CAM) * \
                anti_sat_exposure(NotchFilter(L2, W2), CAM, T_MAX_K, fill, fps)
            for target, texp_i in ((sat, t_exp), (frozen, t_frozen)):
                n_short = electron_rate(t_eval_k, filt, CAM) * texp_i
                rel = math.sqrt((n_short + read_noise**2) / n_short**2
                                + (n_long + read_noise**2) / n_long**2)
                target.append(rel / slope)
        figw = go.Figure()
        figw.add_trace(go.Scatter(
            x=w_sweep * 1e9, y=sat, name="hot pixel in frame (saturation-capped)",
            line=dict(color=BLUE, width=2),
            hovertemplate="width %{x:.0f} nm: %{y:.0f} K<extra></extra>"))
        figw.add_trace(go.Scatter(
            x=w_sweep * 1e9, y=frozen, name="exposure frozen (photon-starved)",
            line=dict(color=AQUA, width=2),
            hovertemplate="width %{x:.0f} nm: %{y:.0f} K<extra></extra>"))
        st.plotly_chart(style(
            figw, "Short notch width [nm]",
            f"NEdT at {t_eval_c} C [K]", logy=True, height=380),
            width="stretch")

    with col_b:
        st.markdown(
            "**The pair map.** Worst-case error at the evaluation "
            "temperature over all pair choices (30/50 nm widths). The basin "
            "is broad; dotted guides mark plume emission features (Na 589, "
            "H-alpha 656, K 767, H2O 940 nm) that notches must avoid."
        )
        l1g, l2g, zmap = pair_map(t_eval_k, QE_SPEC, PIXEL_PITCH, f_number,
                                  TAU_EFF, read_noise, well, T_MAX_K, fill, fps)
        figm = go.Figure(go.Heatmap(
            x=l1g * 1e9, y=l2g * 1e9, z=np.log10(zmap),
            colorscale="Blues", reversescale=False,
            colorbar=dict(title="NEdT [K]", tickvals=[1.4, 1.7, 2, 2.3, 2.7, 3],
                          ticktext=["25", "50", "100", "200", "500", "1000"]),
            zmin=1.3, zmax=3.0,
            customdata=zmap,
            hovertemplate=("lam1 %{x:.0f} nm, lam2 %{y:.0f} nm<br>"
                           "NEdT = %{customdata:.0f} K<extra></extra>")))
        figm.add_trace(go.Scatter(
            x=[l1_nm], y=[l2_nm], mode="markers",
            marker=dict(symbol="star", size=14, color=YELLOW,
                        line=dict(color=INK, width=1)),
            name="selected pair", hovertemplate="selected pair<extra></extra>"))
        for x in (589, 656.3):
            figm.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1))
        for y in (766.5, 940):
            figm.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1))
        figm.update_layout(showlegend=False)
        st.plotly_chart(style(
            figm, "Short notch lam1 [nm]", "Long notch lam2 [nm]", height=380),
            width="stretch")

# ============================================================ story 2
with tab2:
    st.subheader("Uncertain radiometry breaks one-notch pyrometry; "
                 "the ratio cancels it")
    st.markdown(
        "One notch infers T from an **absolute** signal S = K eps L(lam, T). "
        "Emissivity, window fouling and calibration drift all multiply K -- "
        "and every percent of unknown K reads as temperature error. Two "
        "notches divide out everything common to both bands."
    )
    c1, c2_, c3, c4 = st.columns(4)
    d_eps = c1.slider("Emissivity uncertainty [%]", 0, 30, 10)
    d_window = c2_.slider("Window fouling [%]", 0, 30, 10)
    d_cal = c3.slider("Calibration drift [%]", 0, 30, 5)
    t_true_c = c4.slider("True scene temperature [C]", 1200, 3000, 2000, 50)
    delta = math.sqrt(d_eps**2 + d_window**2 + d_cal**2) / 100.0
    t_true_k = t_true_c + 273.15

    filt1 = NotchFilter(L1, W1)
    t_exp1 = anti_sat_exposure(filt1, CAM, T_MAX_K, fill, fps)
    temps_k = np.linspace(1000 + 273.15, 3200 + 273.15, 200)
    temps_c = temps_k - 273.15
    signal = electron_rate(temps_k, filt1, CAM) * t_exp1
    s_true = electron_rate(t_true_k, filt1, CAM) * t_exp1

    # Implied temperature interval from the multiplicative band (Wien form).
    inv_lo = 1.0 / t_true_k + (L1 / C2) * math.log1p(delta)
    inv_hi = 1.0 / t_true_k + (L1 / C2) * math.log1p(-delta)
    t_lo_c, t_hi_c = 1.0 / inv_lo - 273.15, 1.0 / inv_hi - 273.15

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=temps_c, y=signal * (1 + delta), line=dict(width=0),
        showlegend=False, hoverinfo="skip"))
    fig2.add_trace(go.Scatter(
        x=temps_c, y=signal * (1 - delta), fill="tonexty",
        fillcolor="rgba(42,120,214,0.15)", line=dict(width=0),
        name=f"signal x (1 +/- {delta * 100:.0f}%) uncertainty band",
        hoverinfo="skip"))
    fig2.add_trace(go.Scatter(
        x=temps_c, y=signal, name=f"{l1_nm} nm notch signal S(T)",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x:.0f} C: %{y:.3s} e-<extra></extra>"))
    hline_log(fig2, s_true, "measured signal")
    fig2.add_vrect(x0=t_lo_c, x1=t_hi_c, fillcolor="rgba(235,104,52,0.18)",
                   line_width=0,
                   annotation_text=f"implied T: {t_lo_c:.0f}-{t_hi_c:.0f} C",
                   annotation_position="top left",
                   annotation_font_color=INK2)
    st.plotly_chart(style(
        fig2, "Scene temperature [C]",
        f"Electrons per frame in the {l1_nm} nm notch", logy=True),
        width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Combined radiometric uncertainty", f"+/- {delta * 100:.0f} %")
    m2.metric("One-notch temperature error",
              f"-{t_true_c - t_lo_c:.0f} / +{t_hi_c - t_true_c:.0f} K")
    m3.metric("Same uncertainty through the ratio", "0 K",
              help="Common-mode multiplicative factors cancel exactly in S1/S2.")

    st.markdown(
        "**And it gets worse with temperature.** The same fractional "
        "uncertainty maps to error as (lam T^2/c2) x delta. The ratio's only "
        "surviving systematic is the *differential* part -- non-gray "
        "emissivity between the two bands (typically a few % for refractory "
        "surfaces), which propagates through lam_eq instead:"
    )
    d_diff = st.slider("Differential (non-gray) residual between bands [%]",
                       0.0, 10.0, 1.0, 0.5)
    err_1notch = L1 * temps_k**2 / C2 * delta
    err_1notch_lab = L1 * temps_k**2 / C2 * 0.02
    err_ratio = LAM_EQ * temps_k**2 / C2 * (d_diff / 100.0)
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=temps_c, y=err_1notch,
        name=f"one notch, {delta * 100:.0f}% radiometric uncertainty",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x:.0f} C: %{y:.0f} K<extra></extra>"))
    fig3.add_trace(go.Scatter(
        x=temps_c, y=err_1notch_lab,
        name="one notch, heroic 2% absolute radiometry",
        line=dict(color=ORANGE, width=2, dash="dash"),
        hovertemplate="%{x:.0f} C: %{y:.0f} K<extra></extra>"))
    fig3.add_trace(go.Scatter(
        x=temps_c, y=err_ratio,
        name=f"two-notch ratio, {d_diff:g}% non-gray residual",
        line=dict(color=AQUA, width=2),
        hovertemplate="%{x:.0f} C: %{y:.0f} K<extra></extra>"))
    st.plotly_chart(style(
        fig3, "Scene temperature [C]", "Systematic temperature error [K]",
        logy=True), width="stretch")

# ============================================================ story 3
with tab3:
    st.subheader(f"Temperature certainty with the {l1_nm}/{l2_nm} nm pair "
                 f"on the {SENSOR_NAME}")
    temps_k = np.linspace(1300 + 273.15, 3050 + 273.15, 80)
    temps_c = temps_k - 273.15
    sig_pix, exposures = sigma_ratio_vs_t(temps_k, L1, W1, L2, W2, CAM,
                                          T_MAX_K, fill, fps)
    sig_avg, _ = sigma_ratio_vs_t(temps_k, L1, W1, L2, W2, CAM, T_MAX_K,
                                  fill, fps, binning=binning, frames=frames)

    def at(tc):
        return float(np.interp(tc + 273.15, temps_k, sig_avg))

    m = st.columns(4)
    m[0].metric("NEdT @ 1500 C", f"{at(1500):.0f} K")
    m[1].metric("NEdT @ 2250 C", f"{at(2250):.1f} K")
    m[2].metric("NEdT @ 3000 C", f"{at(3000):.1f} K")
    m[3].metric("Exposures (short / long)",
                f"{exposures[0] * 1e6:.1f} / {exposures[1] * 1e6:.1f} us")
    if min(exposures) < MIN_EXPOSURE:
        st.warning(
            f"Shortest exposure is below the ~{MIN_EXPOSURE * 1e6:.0f} us "
            "global-shutter floor assumed for this sensor -- add ND "
            "(sidebar) or stop down; precision is unaffected.")

    fig4 = go.Figure()
    fig4.add_vrect(x0=1500, x1=3000, fillcolor="rgba(237,161,0,0.07)",
                   line_width=0, annotation_text="primary band",
                   annotation_position="top left",
                   annotation_font_color=MUTED)
    fig4.add_trace(go.Scatter(
        x=temps_c, y=sig_pix, name="per pixel, per frame",
        line=dict(color=BLUE, width=2),
        hovertemplate="%{x:.0f} C: %{y:.1f} K<extra></extra>"))
    fig4.add_trace(go.Scatter(
        x=temps_c, y=sig_avg,
        name=f"{binning}x{binning} binning x {frames} frames",
        line=dict(color=AQUA, width=2, dash="dash"),
        hovertemplate="%{x:.0f} C: %{y:.1f} K<extra></extra>"))
    st.plotly_chart(style(
        fig4, "Scene temperature [C]", "Ratio NEdT [K]", logy=True),
        width="stretch")

    st.markdown(
        "**Why precision dies toward the cold end:** the exposure is pinned "
        "by the hottest scene content, so each channel's electrons collapse "
        "along the Planck curve. The short notch hits the read-noise floor "
        "first -- that sets the cold limit of a single video frame "
        "(exposure bracketing / HDR restores it)."
    )
    fig5 = go.Figure()
    for center, width, t_exp, color, name in (
        (L1, W1, exposures[0], BLUE, f"{l1_nm} nm notch"),
        (L2, W2, exposures[1], ORANGE, f"{l2_nm} nm notch"),
    ):
        n_e = electron_rate(temps_k, NotchFilter(center, width), CAM) * t_exp
        fig5.add_trace(go.Scatter(
            x=temps_c, y=n_e, name=name, line=dict(color=color, width=2),
            hovertemplate="%{x:.0f} C: %{y:.3s} e-<extra></extra>"))
    for y, label, dash in ((well, "full well", "solid"),
                           (fill * well, "exposure target", "dash"),
                           (100, "SNR ~ 10", "dashdot"),
                           (read_noise, "read noise", "dot")):
        hline_log(fig5, y, label, dash)
    st.plotly_chart(style(
        fig5, "Scene temperature [C]", "Electrons per pixel per frame",
        logy=True), width="stretch")

    with st.expander("Sensor model & provenance"):
        if sensor_key == "imx900":
            st.markdown(
                f"- {IMX900_SPECS['name']}: {IMX900_SPECS['format']}, "
                f"{IMX900_SPECS['pixel_pitch'] * 1e6:.2f} um pixels, 12-bit ADC. "
                "Sensor values are taken verbatim from the StarTrackerCentroid "
                "repository presets (`framegen/sensors/presets/imx900.py`).\n"
                f"- LCG: {IMX900_GAIN_MODES['lcg']['well_capacity']:.0f} e- well, "
                f"{IMX900_GAIN_MODES['lcg']['read_noise']:.2f} e- read "
                f"({IMX900_GAIN_MODES['lcg']['source']}). HCG: "
                f"{IMX900_GAIN_MODES['hcg']['well_capacity']:.0f} e- well, "
                f"{IMX900_GAIN_MODES['hcg']['read_noise']:.2f} e- read "
                f"({IMX900_GAIN_MODES['hcg']['source']}).\n"
                "- QE: the 121-point tabulated curve (400-1000 nm) from the same "
                f"repository, whose raw peak of {IMX900_QE_TABLE_PEAK:.3f} is "
                "flagged there as a probable normalised response; it is scaled "
                f"here to the Basler EMVA 1288 absolute peak of "
                f"{IMX900_BASLER_EMVA_PEAK_QE:.3f}.\n"
                f"- The {MIN_EXPOSURE * 1e6:.0f} us minimum exposure is an "
                "assumption. Dark current (4.5 e-/s at 60 C) is negligible at "
                "these exposures and not modelled."
            )
        else:
            st.markdown(
                f"- **{SENSOR_NAME}.** {SENSOR_NOTE}\n"
                f"- Pixel pitch {PIXEL_PITCH * 1e6:.2f} um, full well {well:.0f} e-, "
                f"read noise {read_noise:.2f} e- rms, assumed minimum exposure "
                f"{MIN_EXPOSURE * 1e6:.0f} us -- all editable in the sidebar. "
                "Dark current is not modelled."
            )
        st.markdown(
            "- With a hot pixel in frame the exposure is saturation-capped, so "
            "the QE *level* cancels out of the precision budget (it rescales "
            "the exposure, not the electron count); the QE *shape* still "
            "weights each band integral and sets where the read-noise floor "
            "and the minimum exposure bite.\n"
            "- Slope d(lnR)/dT uses the Wien limit (< 3% from full Planck "
            "integrals here); electron counts use full integrals of "
            "QE x filter x Planck."
        )
        lam_qe = np.linspace(350e-9, 1100e-9, 200)
        figq = go.Figure(go.Scatter(
            x=lam_qe * 1e9, y=QE_FN(lam_qe) * 100,
            line=dict(color=BLUE, width=2), name=f"{SENSOR_NAME} QE",
            hovertemplate="%{x:.0f} nm: %{y:.0f}%<extra></extra>"))
        for x, lbl in ((l1_nm, "lam1"), (l2_nm, "lam2")):
            figq.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1),
                           annotation_text=lbl, annotation_font_color=MUTED)
        st.plotly_chart(style(
            figq, "Wavelength [nm]", "Quantum efficiency [%]", height=300),
            width="stretch")

# ============================================================ story 4
with tab4:
    st.subheader("Viewing angle, reflected sunlight and plume reflection")
    st.markdown(
        "Three terms the plain radiometric chain leaves out. **Directional "
        "emissivity** falls toward grazing angles (Fresnel), so a normal-"
        "incidence calibration reads a tilted surface cold: a multiplicative "
        "error that varies across the frame. The ratio cancels it exactly when "
        "the angular factor is the same in both bands, and nearly so otherwise. "
        "**Reflected sunlight** is additive, hits the short band far harder than "
        "the long band, and propagates through lam_eq, so it biases the ratio "
        "*more* than a single band. A pre-ignition frame measures it directly "
        "and can be subtracted. **Plume light reflected by the wall** is the "
        "same kind of term but can be 10-100x larger for a luminous (soot or "
        "particle) plume, and it exists only during the burn: it must be "
        "estimated from plume pixels in the same frame and a view-factor "
        "model, then subtracted."
    )
    g1, g2, g3, g4 = st.columns(4)
    mat_name = g1.selectbox("Surface optical constants",
                            list(MATERIALS) + ["custom"], index=1)
    theta_v_deg = g2.slider("Viewing angle from surface normal [deg]", 0, 88, 45)
    theta_s_deg = g3.slider("Sun angle from surface normal [deg]", 0, 90, 45)
    sun = g4.slider("Sun factor (0 = shaded / night, 1 = full sun)", 0.0, 1.2, 1.0, 0.1)
    h1, h2 = st.columns(2)
    t_eval4_c = h1.slider("Evaluate at scene temperature [C]", 1000, 3000, 1500, 50,
                          key="teval4")
    resid_pct = h2.slider("Reflected-sun residual after pre-ignition subtraction [%]",
                          0, 100, 100, 5)
    st.markdown("**Plume illumination of the wall**")
    q1, q2, q3, q4, q5, q6 = st.columns(6)
    plume_t_c = q1.slider("Plume temperature [C]", 1500, 2800, 2200, 50)
    plume_eps = q2.slider("Plume band emissivity", 0.0, 1.0, 0.1, 0.01,
                          help="~0.001-0.01 for a clean plume in the notches; "
                               "0.1-0.9 for soot / Al2O3 laden plumes.")
    plume_f = q3.slider("View factor (wall sees plume)", 0.0, 1.0, 0.2, 0.05,
                        help="Cosine-weighted fraction of the wall element's "
                             "hemisphere filled by plume.")
    plume_alpha = q4.slider("Plume spectral slope alpha", 0.0, 1.5, 0.0, 0.25,
                            help="eps_pl(lam) ~ lam^-alpha: 0 gray (particles), "
                                 "~1 Rayleigh soot.")
    plume_glare = q5.slider("Veiling glare coefficient", 0.0, 0.05, 0.0, 0.005,
                            help="Lens veiling glare index x plume area fraction "
                                 "in the field; reaches every pixel.")
    plume_resid_pct = q6.slider("Plume residual after in-frame subtraction [%]",
                                0, 100, 100, 5)
    pp = (plume_t_c + 273.15, plume_eps, plume_alpha, plume_f, plume_glare,
          plume_resid_pct / 100.0)
    plume_src = _plume(pp)
    nk = (2.7, 1.4, 2.9, 1.6)
    if mat_name == "custom":
        k1, k2, k3, k4 = st.columns(4)
        nk = (k1.number_input("n at 620 nm", 1.0, 6.0, 2.7, 0.1),
              k2.number_input("k at 620 nm", 0.0, 6.0, 1.4, 0.1),
              k3.number_input("n at 870 nm", 1.0, 6.0, 2.9, 0.1),
              k4.number_input("k at 870 nm", 0.0, 6.0, 1.6, 0.1))
    material = _material(mat_name, nk)
    resid = resid_pct / 100.0
    t4 = t_eval4_c + 273.15
    tv, ts = math.radians(theta_v_deg), math.radians(theta_s_deg)
    f1, f2 = NotchFilter(L1, W1), NotchFilter(L2, W2)
    pyro4 = RatioPyrometer(f1, f2, CAM)
    one_now = one_band_apparent_temperature(t4, f1, CAM, material, tv, ts, sun, 0.0,
                                            resid, plume_src) - t4
    two_now = ratio_apparent_temperature(t4, pyro4, material, tv, ts, sun, 0.0,
                                         resid, plume_src) - t4
    pfrac1 = (plume_reflected_electron_rate(f1, CAM, material, tv, plume_src)
              / thermal_electron_rate(t4, f1, CAM, material, tv))
    pfrac2 = (plume_reflected_electron_rate(f2, CAM, material, tv, plume_src)
              / thermal_electron_rate(t4, f2, CAM, material, tv))
    frac1 = (resid * solar_reflected_electron_rate(f1, CAM, material, tv, ts, sun)
             / thermal_electron_rate(t4, f1, CAM, material, tv))
    frac2 = (resid * solar_reflected_electron_rate(f2, CAM, material, tv, ts, sun)
             / thermal_electron_rate(t4, f2, CAM, material, tv))
    glint = specular_glint_ratio(tv, L1, t4, material, max(sun, 1e-9))
    m4 = st.columns(4)
    m4[0].metric(f"One-band bias @ {t_eval4_c} C", f"{one_now:+.0f} K",
                 help="Normal-incidence emissivity calibration; includes angle and sun.")
    m4[1].metric(f"Ratio bias @ {t_eval4_c} C", f"{two_now:+.0f} K")
    m4[2].metric("Reflected sun / signal", f"{100 * frac1:.1f} % / {100 * frac2:.1f} %",
                 help="short band / long band")
    m4b = st.columns(4)
    m4b[0].metric("Reflected plume / signal",
                  f"{100 * pfrac1:.1f} % / {100 * pfrac2:.1f} %",
                  help="short band / long band, after the in-frame subtraction residual")
    m4[3].metric("Specular glint / surface radiance",
                 f"{glint:,.0f}x" if glint >= 10 else f"{glint:.1f}x",
                 help="Radiance of the sun's specular image relative to the surface "
                      f"at {t_eval4_c} C, if that reflection lands in the camera. "
                      "> 1 saturates the pixel; exclude the sun-surface-camera "
                      "specular geometry.")

    ca, cb = st.columns(2)
    with ca:
        theta_grid = np.radians(np.linspace(0, 88, 120))
        e1 = material.emissivity(theta_grid, L1)
        e2 = material.emissivity(theta_grid, L2)
        fige = go.Figure()
        fige.add_trace(go.Scatter(x=np.degrees(theta_grid), y=e1, name=f"{l1_nm} nm",
                                  line=dict(color=BLUE, width=2),
                                  hovertemplate="%{x:.0f} deg: %{y:.3f}<extra></extra>"))
        fige.add_trace(go.Scatter(x=np.degrees(theta_grid), y=e2, name=f"{l2_nm} nm",
                                  line=dict(color=ORANGE, width=2),
                                  hovertemplate="%{x:.0f} deg: %{y:.3f}<extra></extra>"))
        fige.add_trace(go.Scatter(x=np.degrees(theta_grid),
                                  y=(e1 / e2) / (e1[0] / e2[0]),
                                  name="band ratio, relative to normal incidence",
                                  line=dict(color=AQUA, width=2, dash="dash"),
                                  hovertemplate="%{x:.0f} deg: %{y:.3f}<extra></extra>"))
        fige.add_vline(x=theta_v_deg, line=dict(color=MUTED, dash="dot", width=1))
        st.plotly_chart(style(fige, "Viewing angle from surface normal [deg]",
                              "Directional emissivity / relative band ratio",
                              height=380), width="stretch")
    with cb:
        angles, one_a, two_a = bias_vs_angle(
            mat_name, nk, t4, theta_s_deg, sun, resid, pp, L1, W1, L2, W2, f_number,
            TAU_EFF, QE_SPEC, PIXEL_PITCH, read_noise, well)
        figa = go.Figure()
        figa.add_trace(go.Scatter(x=angles, y=one_a, name="one band (620 nm)",
                                  line=dict(color=BLUE, width=2),
                                  hovertemplate="%{x:.0f} deg: %{y:+.0f} K<extra></extra>"))
        figa.add_trace(go.Scatter(x=angles, y=two_a, name="two-band ratio",
                                  line=dict(color=ORANGE, width=2),
                                  hovertemplate="%{x:.0f} deg: %{y:+.0f} K<extra></extra>"))
        figa.add_hline(y=0, line=dict(color=MUTED, width=1))
        figa.add_vline(x=theta_v_deg, line=dict(color=MUTED, dash="dot", width=1))
        st.plotly_chart(style(figa, "Viewing angle from surface normal [deg]",
                              f"Apparent - true temperature at {t_eval4_c} C [K]",
                              height=380), width="stretch")

    temps4, one_t, two_t = bias_vs_temperature(
        mat_name, nk, theta_v_deg, theta_s_deg, sun, resid, pp, L1, W1, L2, W2, f_number,
        TAU_EFF, QE_SPEC, PIXEL_PITCH, read_noise, well)
    figt = go.Figure()
    figt.add_trace(go.Scatter(x=temps4, y=one_t, name="one band (620 nm)",
                              line=dict(color=BLUE, width=2),
                              hovertemplate="%{x:.0f} C: %{y:+.0f} K<extra></extra>"))
    figt.add_trace(go.Scatter(x=temps4, y=two_t, name="two-band ratio",
                              line=dict(color=ORANGE, width=2),
                              hovertemplate="%{x:.0f} C: %{y:+.0f} K<extra></extra>"))
    figt.add_hline(y=0, line=dict(color=MUTED, width=1))
    figt.add_vrect(x0=1500, x1=3000, fillcolor="rgba(237,161,0,0.07)", line_width=0)
    st.plotly_chart(style(figt, "Scene temperature [C]",
                          f"Apparent - true temperature at {theta_v_deg} deg view [K]"),
                    width="stretch")
    st.caption(
        "Smooth-surface Fresnel emissivity with illustrative optical constants "
        "(rough surfaces are more Lambertian, so this is the worst case for the "
        "angular collapse); AM1.5G sunlight reflected diffusely (about 10% "
        "uncertain); plume light as a gray or soot-sloped body reflected "
        "diffusely with a user-set view factor; the specular glint is reported "
        "as a hazard ratio only."
    )

st.caption(
    "Model: `ircam` package in this repo (validated against Stefan-Boltzmann, "
    "Wien and series-expansion references -- see tests/). Assumes a continuum "
    "emitter (nozzle surface or luminous soot) and gray emissivity between "
    "the bands; notches must avoid plume emission lines. Full analysis: "
    "docs/nozzle_pyrometry.md."
)
