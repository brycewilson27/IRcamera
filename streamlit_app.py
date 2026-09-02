"""Two-notch ratio pyrometry designer -- interactive companion to
docs/nozzle_pyrometry.md.

Three stories:
  1. Notch spacing & width vs temperature sensitivity
  2. Why one calibrated notch is inaccurate under uncertain radiometry
  3. Temperature certainty vs scene temperature for the chosen pair
Sensor: Sony IMX900 (2.25 um BSI stacked global shutter, ~10 ke- well,
enhanced NIR). QE curve is approximate -- see ircam/sensors.py.

Run locally:   streamlit run streamlit_app.py
Deploy:        push to GitHub, point Streamlit Community Cloud at this file.
"""

import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ircam.constants import C2
from ircam.pyrometry import (
    NotchFilter,
    PyroCamera,
    electron_rate,
    equivalent_wavelength,
)
from ircam.sensors import IMX900_SPECS, imx900_camera, imx900_qe

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
def pair_map(t_eval_k, f_number, tau, read_noise, well, t_max_k, fill, fps):
    """Worst-case NEdT over a (lambda1, lambda2) grid."""
    cam = imx900_camera(f_number, tau, read_noise, well)
    l1_grid = np.linspace(420e-9, 760e-9, 35)
    l2_grid = np.linspace(700e-9, 1000e-9, 31)
    z = np.full((len(l2_grid), len(l1_grid)), np.nan)
    for i, l2 in enumerate(l2_grid):
        for j, l1 in enumerate(l1_grid):
            if l2 - l1 >= 60e-9:
                sig, _ = sigma_ratio_vs_t(np.array([t_eval_k]), l1, 30e-9,
                                          l2, 50e-9, cam, t_max_k, fill, fps)
                z[i, j] = sig[0]
    return l1_grid, l2_grid, z


# ---------------------------------------------------------------- sidebar
st.sidebar.header("Notch pair")
l1_nm = st.sidebar.slider("Short notch center [nm]", 420, 750, 620, 5)
w1_nm = st.sidebar.slider("Short notch width [nm]", 10, 80, 30, 5)
l2_nm = st.sidebar.slider("Long notch center [nm]", max(700, l1_nm + 60),
                          1000, max(870, l1_nm + 60), 5)
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

with st.sidebar.expander("Sensor: Sony IMX900"):
    st.caption(IMX900_SPECS["name"] + " -- " + IMX900_SPECS["format"]
               + ". QE curve approximate; replace with datasheet values.")
    read_noise = st.number_input("Read noise [e- rms]", 1.0, 10.0,
                                 IMX900_SPECS["read_noise"], 0.5)
    well = st.number_input("Full well [e-]", 4000.0, 30000.0,
                           IMX900_SPECS["well_capacity"], 1000.0)

TAU_EFF = tau_optics * 10.0**(-nd_od)
CAM = imx900_camera(f_number, TAU_EFF, read_noise, well)
L1, W1, L2, W2 = l1_nm * 1e-9, w1_nm * 1e-9, l2_nm * 1e-9, w2_nm * 1e-9
T_MAX_K = t_max_c + 273.15
LAM_EQ = equivalent_wavelength(L1, L2)

st.title("Two-notch ratio pyrometry designer")
st.caption(f"Engine-nozzle thermography, 1500-3000 C primary band, on the "
           f"Sony IMX900 ({IMX900_SPECS['pixel_pitch'] * 1e6:.2f} um pixels, "
           f"{well / 1e3:.0f} ke- well, {read_noise:g} e- read). Current pair: "
           f"**{l1_nm} / {l2_nm} nm**, equivalent wavelength "
           f"lam_eq = {LAM_EQ * 1e9:.0f} nm.")

tab1, tab2, tab3 = st.tabs([
    "1 - Notch spacing & width",
    "2 - One notch vs two: uncertain radiometry",
    "3 - Temperature certainty vs temperature",
])

# ============================================================ story 1
with tab1:
    st.subheader("Spacing wins -- until the short notch runs out of photons")
    st.markdown(
        "Temperature error scales with the **equivalent wavelength** "
        "lam_eq = lam1 lam2/(lam2 - lam1): sigma_T = (lam_eq T^2/c2) x "
        "(ratio noise). Spreading the notches shrinks lam_eq (dashed line). "
        "But with the exposure pinned by the hottest pixel in frame, the "
        "short notch's photon count collapses as it moves blue -- the solid "
        "line turns back up. The optimum is the balance."
    )
    t_eval_c = st.slider("Evaluate precision at scene temperature [C]",
                         1300, 3000, 1500, 50, key="teval1")
    t_eval_k = t_eval_c + 273.15

    l1_sweep = np.linspace(420e-9, L2 - 60e-9, 60)
    sig_full = np.array([
        sigma_ratio_vs_t(np.array([t_eval_k]), l1, W1, L2, W2, CAM,
                         T_MAX_K, fill, fps)[0][0]
        for l1 in l1_sweep
    ])
    sig_law = (equivalent_wavelength(l1_sweep, L2) * t_eval_k**2 / C2) * 0.01
    i_opt = int(np.argmin(sig_full))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=l1_sweep * 1e9, y=sig_full, name="full photon-noise model",
        line=dict(color=BLUE, width=2),
        hovertemplate="lam1 = %{x:.0f} nm<br>NEdT = %{y:.0f} K<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=l1_sweep * 1e9, y=sig_law, name="lam_eq law alone (fixed 1% ratio noise)",
        line=dict(color=ORANGE, width=2, dash="dash"),
        hovertemplate="lam1 = %{x:.0f} nm<br>NEdT = %{y:.1f} K<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[l1_sweep[i_opt] * 1e9], y=[sig_full[i_opt]], mode="markers+text",
        marker=dict(color=BLUE, size=10, symbol="star"),
        text=[f"optimum {l1_sweep[i_opt] * 1e9:.0f} nm"], textposition="top center",
        textfont=dict(color=INK2, size=11), showlegend=False, hoverinfo="skip"))
    fig.add_vline(x=l1_nm, line=dict(color=MUTED, dash="dot", width=1),
                  annotation_text=f"your lam1 = {l1_nm} nm",
                  annotation_font_color=MUTED)
    st.plotly_chart(style(
        fig, f"Short notch center lam1 [nm]   (long notch fixed at {l2_nm} nm)",
        f"NEdT at {t_eval_c} C [K]  (per pixel, per frame)", logy=True),
        use_container_width=True)

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
            use_container_width=True)

    with col_b:
        st.markdown(
            "**The pair map.** Worst-case error at the evaluation "
            "temperature over all pair choices (30/50 nm widths). The basin "
            "is broad; dotted guides mark plume emission features (Na 589, "
            "H-alpha 656, K 767, H2O 940 nm) that notches must avoid."
        )
        l1g, l2g, zmap = pair_map(t_eval_k, f_number, TAU_EFF, read_noise,
                                  well, T_MAX_K, fill, fps)
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
            name="your pair", hovertemplate="your pair<extra></extra>"))
        for x in (589, 656.3):
            figm.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1))
        for y in (766.5, 940):
            figm.add_hline(y=y, line=dict(color=MUTED, dash="dot", width=1))
        figm.update_layout(showlegend=False)
        st.plotly_chart(style(
            figm, "Short notch lam1 [nm]", "Long notch lam2 [nm]", height=380),
            use_container_width=True)

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
        use_container_width=True)

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
        logy=True), use_container_width=True)

# ============================================================ story 3
with tab3:
    st.subheader(f"Temperature certainty with the {l1_nm}/{l2_nm} nm pair "
                 "on the IMX900")
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
    if min(exposures) < IMX900_SPECS["min_exposure"]:
        st.warning(
            f"Shortest exposure is below the ~"
            f"{IMX900_SPECS['min_exposure'] * 1e6:.0f} us global-shutter floor "
            "-- add ND (sidebar) or stop down; precision is unaffected.")

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
        use_container_width=True)

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
        logy=True), use_container_width=True)

    with st.expander("IMX900 model & assumptions"):
        st.markdown(
            f"- {IMX900_SPECS['name']}: {IMX900_SPECS['format']}, "
            f"{IMX900_SPECS['pixel_pitch'] * 1e6:.2f} um pixels, ~10 ke- "
            "saturation, enhanced NIR (~2x conventional at 850 nm).\n"
            f"- Read noise {read_noise:g} e- and the QE curve below are "
            "*assumptions* at Pregius-S class values -- swap in datasheet "
            "numbers in `ircam/sensors.py`. In the saturation-capped regime "
            "QE cancels out of the precision budget (it rescales exposure, "
            "not electrons).\n"
            "- Slope d(lnR)/dT uses the Wien limit (< 3% from full Planck "
            "integrals here); electron counts use full integrals of "
            "QE x filter x Planck."
        )
        lam_qe = np.linspace(350e-9, 1100e-9, 200)
        figq = go.Figure(go.Scatter(
            x=lam_qe * 1e9, y=imx900_qe(lam_qe) * 100,
            line=dict(color=BLUE, width=2), name="IMX900 QE (approx.)",
            hovertemplate="%{x:.0f} nm: %{y:.0f}%<extra></extra>"))
        for x, lbl in ((l1_nm, "lam1"), (l2_nm, "lam2")):
            figq.add_vline(x=x, line=dict(color=MUTED, dash="dot", width=1),
                           annotation_text=lbl, annotation_font_color=MUTED)
        st.plotly_chart(style(
            figq, "Wavelength [nm]", "Quantum efficiency [%]", height=300),
            use_container_width=True)

st.caption(
    "Model: `ircam` package in this repo (validated against Stefan-Boltzmann, "
    "Wien and series-expansion references -- see tests/). Assumes a continuum "
    "emitter (nozzle surface or luminous soot) and gray emissivity between "
    "the bands; notches must avoid plume emission lines. Full analysis: "
    "docs/nozzle_pyrometry.md."
)
