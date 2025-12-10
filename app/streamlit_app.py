import streamlit as st
import numpy as np

from utils import ui_components
from models.temperature_model import model as temp_model
from models.irradiation_model import model as irr_model
from models.yield_strength_model import model as ys_model

st.set_page_config(
    page_title="Fusion Materials Uncertainty Demo",
    layout="wide"
)

ui_components.app_header()

tab1, tab2, tab3 = st.tabs([
    "Model 1: Temperature",
    "Model 2: Irradiation",
    "Model 3: Yield Strength"
])

with tab1:
    ui_components.model_summary_box(
        "Model 1: Temperature",
        "Maps average heat flux (e.g. from plasma-facing loading) to peak temperature, "
        "with aleatoric uncertainty representing small-scale thermal fluctuations and unmodelled details."
    )

    geom = temp_model.get_default_geometry()
    ui_components.geometry_expander("Fixed thermal / geometric parameters", geom)

    st.markdown("### 🔧 Inputs")
    q_avg = st.slider(
        "Average heat flux q_avg (arbitrary units)",
        min_value=0.0,
        max_value=10.0,
        value=5.0,
        step=0.1,
        help="Represents average heat flux onto the component over a pulse."
    )
    n_samples = st.slider(
        "Number of model samples",
        min_value=1,
        max_value=200,
        value=50,
        step=1,
        help="Increase to see the spread of aleatoric uncertainty."
    )

    if st.button("Run temperature model"):
        rng = np.random.default_rng(123)
        samples = temp_model.sample_model({"q_avg": np.array([q_avg])}, n_samples=n_samples, rng=rng)[0]
        st.markdown("### 📈 Output: Peak temperature distribution")
        st.write(f"Mean T_peak: **{samples.mean():.2f} K**, Std: **{samples.std():.2f} K**")
        st.bar_chart(samples)

    ui_components.data_section("temperature", "data/temperature")

with tab2:
    ui_components.model_summary_box(
        "Model 2: Irradiation",
        "Maps average neutron flux to peak dpa at the component, incorporating shielding and geometric factors "
        "with stochastic variation to mimic transport and spectrum effects."
    )

    geom = irr_model.get_default_geometry()
    ui_components.geometry_expander("Fixed shielding / exposure parameters", geom)

    st.markdown("### 🔧 Inputs")
    flux_avg = st.number_input(
        "Average neutron flux (n/m²/s)",
        min_value=0.0,
        value=1e18,
        step=1e17,
        help="Represents average neutron flux at a location in the device."
    )
    n_samples2 = st.slider(
        "Number of model samples (irradiation)",
        min_value=1,
        max_value=200,
        value=50,
        step=1,
        help="Increase to observe spread in dpa due to aleatoric uncertainty."
    )

    if st.button("Run irradiation model"):
        rng = np.random.default_rng(456)
        samples = irr_model.sample_model({"flux_avg": np.array([flux_avg])}, n_samples=n_samples2, rng=rng)[0]
        st.markdown("### 📈 Output: Peak dpa distribution")
        st.write(f"Mean dpa_peak: **{samples.mean():.4f}**, Std: **{samples.std():.4f}**")
        st.bar_chart(samples)

    ui_components.data_section("irradiation", "data/irradiation")

with tab3:
    ui_components.model_summary_box(
        "Model 3: Yield Strength",
        "Maps temperature and dpa to yield strength, representing degradation of mechanical properties "
        "under thermal and irradiation loading, with scatter representing material variability."
    )

    geom = ys_model.get_default_geometry()
    ui_components.geometry_expander("Fixed material / strain-rate parameters", geom)

    st.markdown("### 🔧 Inputs")
    T_in = st.slider(
        "Temperature (K)",
        min_value=300.0,
        max_value=1200.0,
        value=600.0,
        step=10.0,
        help="Representative operating temperature of the component."
    )
    dpa_in = st.slider(
        "Damage (dpa)",
        min_value=0.0,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Displacements per atom at the material location."
    )
    n_samples3 = st.slider(
        "Number of model samples (yield strength)",
        min_value=1,
        max_value=200,
        value=100,
        step=1,
        help="Increase to inspect the scatter in yield strength due to material variability."
    )

    if st.button("Run yield strength model"):
        rng = np.random.default_rng(789)
        samples = ys_model.sample_model({"temperature": np.array([T_in]), "dpa": np.array([dpa_in])}, n_samples=n_samples3, rng=rng)[0]
        st.markdown("### 📈 Output: Yield strength distribution")
        st.write(f"Mean yield strength: **{samples.mean():.2f} MPa**, Std: **{samples.std():.2f} MPa**")
        st.bar_chart(samples)

    ui_components.data_section("yield_strength", "data/yield_strength")
