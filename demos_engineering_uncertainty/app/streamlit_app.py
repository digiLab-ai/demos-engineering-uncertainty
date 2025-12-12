import sys
from pathlib import Path

# Make repo root importable no matter where Streamlit runs from
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from app.config import branding
from utils import ui_components
from models.temperature_model import model as temp_model
from models.irradiation_model import model as irr_model
from models.yield_strength_model import model as ys_model

TARGET_POINTS = 20
YIELD_POINTS = 20
TRAIN_FRACTION = 0.8


def _split_train_val(inputs_df, outputs_df, rng):
    """Random 80/20 split while keeping inputs and outputs aligned."""
    n_points = len(inputs_df)
    indices = rng.permutation(n_points)

    n_train = max(1, int(np.floor(TRAIN_FRACTION * n_points)))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    if len(val_idx) == 0:
        # Ensure we always have at least one validation point
        val_idx = train_idx[-1:]
        train_idx = train_idx[:-1]

    train_inputs = inputs_df.iloc[train_idx].reset_index(drop=True)
    val_inputs = inputs_df.iloc[val_idx].reset_index(drop=True)
    train_outputs = outputs_df.iloc[train_idx].reset_index(drop=True)
    val_outputs = outputs_df.iloc[val_idx].reset_index(drop=True)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_temperature_sweep():
    rng = np.random.default_rng()
    lower_count = TARGET_POINTS // 2
    upper_count = TARGET_POINTS - lower_count
    q_values = np.concatenate([
        np.linspace(0.0, 2.0, lower_count, endpoint=False),
        np.linspace(6.0, 10.0, upper_count)
    ])
    samples = temp_model.sample_model({"q_peak": q_values}, n_samples=1, rng=rng)[:, 0]

    inputs_df = pd.DataFrame({"q_peak": q_values})
    outputs_df = pd.DataFrame({"T_peak": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_irradiation_sweep():
    rng = np.random.default_rng()
    # flux values expressed in units of "x 1e18 n/m²/s" to keep magnitudes near unity
    flux_values = np.linspace(0.0, 5.0, TARGET_POINTS)
    samples = irr_model.sample_model({"flux_avg": flux_values}, n_samples=1, rng=rng)[:, 0]

    inputs_df = pd.DataFrame({"flux_avg": flux_values})
    outputs_df = pd.DataFrame({"dpa_peak": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_yield_strength_sweep():
    rng = np.random.default_rng()

    def sample_with_gap(n_points):
        temps = []
        dpas = []
        while len(temps) < n_points:
            T_batch = rng.uniform(300.0, 1200.0, size=n_points)
            dpa_batch = rng.uniform(0.0, 5.0, size=n_points)
            # Exclude a mid-range gap to represent epistemic uncertainty
            mask_gap = (T_batch >= 650.0) & (T_batch <= 900.0) & (dpa_batch >= 2.0) & (dpa_batch <= 3.0)
            keep_idx = np.where(~mask_gap)[0]
            for idx in keep_idx:
                temps.append(T_batch[idx])
                dpas.append(dpa_batch[idx])
                if len(temps) >= n_points:
                    break
        return np.array(temps[:n_points]), np.array(dpas[:n_points])

    T_flat, dpa_flat = sample_with_gap(YIELD_POINTS)

    samples = ys_model.sample_model(
        {"temperature": T_flat, "dpa": dpa_flat},
        n_samples=1,
        rng=rng
    )[:, 0]

    inputs_df = pd.DataFrame({"temperature": T_flat, "dpa": dpa_flat})
    outputs_df = pd.DataFrame({"failure": samples.astype(bool)})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    return train_inputs, val_inputs, train_outputs, val_outputs


def _store_sweep(key, sweep_tuple):
    train_inputs, val_inputs, train_outputs, val_outputs = sweep_tuple
    st.session_state[key] = {
        "train_inputs": train_inputs,
        "val_inputs": val_inputs,
        "train_outputs": train_outputs,
        "val_outputs": val_outputs,
    }


def _ensure_precomputed_sweeps():
    if "temp_sweep" not in st.session_state:
        _store_sweep("temp_sweep", run_temperature_sweep())
    if "irr_sweep" not in st.session_state:
        _store_sweep("irr_sweep", run_irradiation_sweep())
    if "ys_sweep" not in st.session_state:
        _store_sweep("ys_sweep", run_yield_strength_sweep())


def _render_sweep_results(label_prefix, state_key):
    sweep = st.session_state.get(state_key)
    if not sweep:
        return

    train_inputs = sweep["train_inputs"]
    val_inputs = sweep["val_inputs"]
    train_outputs = sweep["train_outputs"]
    val_outputs = sweep["val_outputs"]

    st.success(f"Generated {len(train_inputs)} train / {len(val_inputs)} validation samples.")
    train_df = pd.concat([train_inputs, train_outputs], axis=1)
    val_df = pd.concat([val_inputs, val_outputs], axis=1)

    with st.expander("Train preview"):
        st.dataframe(train_df, use_container_width=True)
    with st.expander("Validation preview"):
        st.dataframe(val_df, use_container_width=True)

    st.markdown("#### Download CSVs")
    downloads = {
        "Train data": train_df,
        "Validation data": val_df,
    }
    cols = st.columns(2)
    for col, (label, df) in zip(cols, downloads.items()):
        with col:
            st.download_button(
                label=f"{label}",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name=f"{label_prefix}_{label.lower().replace(' ', '_')}.csv",
                mime="text/csv",
                key=f"{state_key}_{label.replace(' ', '_').lower()}",
            )

    st.markdown("#### Scatter of generated data")
    input_cols = list(train_inputs.columns)
    output_col = list(train_outputs.columns)[0]
    train_df["split"] = "train"
    val_df["split"] = "validation"
    all_df = pd.concat([train_df, val_df], ignore_index=True)
    is_bool_output = pd.api.types.is_bool_dtype(all_df[output_col])
    is_yield_model = label_prefix == "yield_strength"
    shape_scale = alt.Scale(domain=["train", "validation"], range=["circle", "square"])

    if len(input_cols) == 1:
        x_col = input_cols[0]
        if is_yield_model:
            color_enc = alt.Color(
                output_col,
                title=output_col,
                scale=alt.Scale(scheme="viridis")
            )
        elif is_bool_output:
            color_enc = alt.value(branding.INDIGO)
        else:
            color_enc = alt.value(branding.INDIGO)

        chart = (
            alt.Chart(all_df)
            .mark_point(filled=True, size=80, opacity=0.85)
            .encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(output_col, title=output_col),
                color=color_enc,
                shape=alt.Shape("split", title="split", scale=shape_scale),
                tooltip=input_cols + [output_col, "split"],
            )
        )
    else:
        temp_col, dpa_col = input_cols[:2]
        if is_yield_model:
            color_enc = alt.Color(
                output_col,
                title=output_col,
                scale=alt.Scale(scheme="viridis")
            )
        else:
            color_enc = alt.value(branding.INDIGO)

        chart = (
            alt.Chart(all_df)
            .mark_point(filled=True, size=80, opacity=0.85)
            .encode(
                x=alt.X(temp_col, title=temp_col),
                y=alt.Y(dpa_col, title=dpa_col),
                color=color_enc,
                shape=alt.Shape("split", title="split", scale=shape_scale),
                tooltip=input_cols + [output_col, "split"],
            )
        )

    st.altair_chart(chart, use_container_width=True)

st.set_page_config(
    page_title="Fusion Materials Uncertainty Demo",
    layout="wide"
)

ui_components.app_header()
_ensure_precomputed_sweeps()

tab1, tab2, tab3 = st.tabs([
    "Thermal model",
    "Neutronics model",
    "Failure model"
])

with tab1:
    ui_components.model_summary_box(
        "Thermal model",
        "Maps peak heat flux (e.g. ELM peak in space and time) to peak temperature, "
        "with uncertainty driven by variation in thermal / geometric properties."
    )

    geom = temp_model.get_default_geometry()
    temp_unc = temp_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed thermal / geometric parameters", geom, temp_unc)
    st.caption("Parameter uncertainties are applied as independent Gaussian perturbations at each sweep point.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Run {TARGET_POINTS} evenly spaced peak heat-flux points from 0 to 10, sample T_peak with perturbed thermal parameters, "
        "and randomly hold out 20% for validation."
    )

    _render_sweep_results("temperature", "temp_sweep")

with tab2:
    ui_components.model_summary_box(
        "Neutronics model",
        "Maps average neutron flux (entered as multiples of 1e18 n/m²/s) to peak dpa at the component, incorporating shielding and geometric factors "
        "with stochastic variation to mimic transport and spectrum effects."
    )

    geom = irr_model.get_default_geometry()
    irr_unc = irr_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed shielding / exposure parameters", geom, irr_unc)
    st.caption("Enter flux in units of ×1e18 n/m²/s; parameter uncertainties are applied independently per point with an epistemic multiplier per sample.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Sweep {TARGET_POINTS} evenly spaced flux points from 0 to 5 (×1e18 n/m²/s), sample dpa_peak values "
        "with perturbed shielding/geometric factors, and randomly hold out 20% for validation."
    )

    _render_sweep_results("irradiation", "irr_sweep")

with tab3:
    ui_components.model_summary_box(
        "Failure model",
        "Classifies failure (yield strength below allowable) for EUROFER first-wall backing as a function of temperature and dpa, "
        "with variability from material uncertainties plus epistemic factors."
    )

    geom = ys_model.get_default_geometry()
    ys_unc = ys_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed material / strain-rate parameters", geom, ys_unc)
    st.caption("Material property uncertainties are applied as independent Gaussian perturbations at each sweep point with an epistemic multiplier per sample.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Create a meshgrid across temperature 300 to 1200 K and damage 0 to 5 dpa (~{TARGET_POINTS} points total), "
        "sample yield strength with noise, and randomly hold out 20% for validation."
    )

    _render_sweep_results("yield_strength", "ys_sweep")
