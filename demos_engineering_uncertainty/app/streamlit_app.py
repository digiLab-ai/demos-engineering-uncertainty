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
from models.irradiation_model import model as neutronics_model
from models.yield_strength_model import model as ys_model

TARGET_POINTS = 20
IRR_POINTS = 20
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
        np.linspace(7.0, 10.0, upper_count)
    ])
    samples = temp_model.sample_model({"q": q_values}, n_samples=1, rng=rng)[:, 0]

    inputs_df = pd.DataFrame({"q": q_values})
    outputs_df = pd.DataFrame({"T": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_neutronics_sweep():
    rng = np.random.default_rng()
    # flux values expressed in units of "x 1e18 n/m²/s" to keep magnitudes near unity
    flux_values = np.linspace(0.0, 5.0, IRR_POINTS)
    samples = neutronics_model.sample_model({"phi": flux_values}, n_samples=1, rng=rng)[:, 0]

    inputs_df = pd.DataFrame({"phi": flux_values})
    outputs_df = pd.DataFrame({"dpa": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_yield_strength_sweep():
    rng = np.random.default_rng()

    T_flat = rng.uniform(290.0, 320.0, size=YIELD_POINTS)
    dpa_flat = rng.uniform(0.0, 0.08, size=YIELD_POINTS)

    samples = ys_model.sample_model(
        {"T": T_flat, "dpa": dpa_flat},
        n_samples=1,
        rng=rng
    )[:, 0]

    inputs_df = pd.DataFrame({"T": T_flat, "dpa": dpa_flat})
    outputs_df = pd.DataFrame({"breaking_stress": samples})
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
    if "neutronics_sweep" not in st.session_state:
        _store_sweep("neutronics_sweep", run_neutronics_sweep())
    if "ys_sweep" not in st.session_state:
        _store_sweep("ys_sweep", run_yield_strength_sweep())


def _compute_domain(series: pd.Series, pad_frac: float = 0.05):
    vmin = float(series.min())
    vmax = float(series.max())
    if vmin == vmax:
        # Avoid zero-width domains
        return [vmin - 1.0, vmax + 1.0]
    span = vmax - vmin
    pad = span * pad_frac
    return [vmin - pad, vmax + pad]


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
    split_scale = alt.Scale(domain=["train", "validation"], range=["circle", "square"])
    shape_scale = alt.Scale(domain=["train", "validation"], range=["circle", "square"])

    if len(input_cols) == 1:
        x_col = input_cols[0]
        chart = (
            alt.Chart(all_df)
            .mark_point(filled=True, size=80, opacity=0.85)
            .encode(
                x=alt.X(x_col, title=x_col, scale=alt.Scale(domain=_compute_domain(all_df[x_col]))),
                y=alt.Y(output_col, title=output_col, scale=alt.Scale(domain=_compute_domain(all_df[output_col]))),
                color=(
                    alt.Color(
                        output_col,
                        title=output_col,
                        scale=alt.Scale(range=[branding.INDIGO, branding.KEPPEL, branding.KEY_LIME])
                    )
                    if (label_prefix == "breaking_stress" and not is_bool_output)
                    else alt.Color(f"{output_col}:N", title=output_col, scale=alt.Scale(scheme="set1"))
                    if is_bool_output
                    else alt.value(branding.INDIGO)
                ),
                shape=alt.Shape("split", title="split", scale=shape_scale),
                tooltip=input_cols + [output_col, "split"],
            )
        )
    else:
        temp_col, dpa_col = input_cols[:2]
        chart = (
            alt.Chart(all_df)
            .mark_point(filled=True, size=80, opacity=0.85)
            .encode(
                x=alt.X(temp_col, title=temp_col, scale=alt.Scale(domain=_compute_domain(all_df[temp_col]))),
                y=alt.Y(dpa_col, title=dpa_col, scale=alt.Scale(domain=_compute_domain(all_df[dpa_col]))),
                color=(
                    alt.Color(
                        output_col,
                        title=output_col,
                        scale=alt.Scale(range=[branding.INDIGO, branding.KEPPEL, branding.KEY_LIME])
                    )
                    if (label_prefix == "breaking_stress" and not is_bool_output)
                    else alt.Color(f"{output_col}:N", title=output_col, scale=alt.Scale(scheme="set1"))
                    if is_bool_output
                    else alt.value(branding.INDIGO)
                ),
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
    "Breaking stress model"
])

with tab1:
    ui_components.model_summary_box(
        "Thermal model",
        "Maps peak heat flux (e.g. ELM peak in space and time) to peak temperature, "
        "with uncertainty driven by variation in thermal / geometric properties."
    )

    ui_components.model_parameters_expander(
        "Model parameters",
        [
            {"Parameter": "q", "Unit": "arb.", "Description": "Peak heat flux input"},
            {"Parameter": "T", "Unit": "K", "Description": "Predicted peak temperature output"},
        ]
    )
    geom = temp_model.get_default_geometry()
    temp_unc = temp_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed thermal / geometric parameters", geom, temp_unc)
    st.caption("Parameter uncertainties are applied as independent Gaussian perturbations at each sweep point.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Run {TARGET_POINTS} evenly spaced peak heat-flux points from 0 to 10, sample T with perturbed thermal parameters, "
        "and randomly hold out 20% for validation."
    )

    _render_sweep_results("thermal", "temp_sweep")

with tab2:
    ui_components.model_summary_box(
        "Neutronics model",
        "Maps average neutron flux (entered as multiples of 1e18 n/m²/s) to peak dpa at the component using an exponential flux response, "
        "incorporating shielding and geometric factors with stochastic variation to mimic transport and spectrum effects."
    )

    ui_components.model_parameters_expander(
        "Model parameters",
        [
            {"Parameter": "phi", "Unit": "×1e18 n/m²/s", "Description": "Average neutron flux input"},
            {"Parameter": "dpa", "Unit": "dpa", "Description": "Predicted damage output"},
        ]
    )
    geom = neutronics_model.get_default_geometry()
    irr_unc = neutronics_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed shielding / exposure parameters", geom, irr_unc)
    st.caption("Enter flux in units of ×1e18 n/m²/s; parameter uncertainties are applied independently per point with an epistemic multiplier per sample.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Sweep {IRR_POINTS} evenly spaced flux points from 0 to 5 (×1e18 n/m²/s), sample dpa values "
        "with perturbed shielding/geometric factors, and randomly hold out 20% for validation."
    )

    _render_sweep_results("neutronics", "neutronics_sweep")

with tab3:
    ui_components.model_summary_box(
        "Breaking stress model",
        "Predicts breaking stress (MPa) for EUROFER first-wall backing, decreasing with temperature and neutron damage."
    )

    ui_components.model_parameters_expander(
        "Model parameters",
        [
            {"Parameter": "T", "Unit": "K", "Description": "Temperature input"},
            {"Parameter": "dpa", "Unit": "dpa", "Description": "Damage input"},
            {"Parameter": "breaking_stress", "Unit": "MPa", "Description": "Predicted breaking stress output"},
        ]
    )
    geom = ys_model.get_default_geometry()
    ys_unc = ys_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed material / strain-rate parameters", geom, ys_unc)
    st.caption("Material property uncertainties are applied as independent Gaussian perturbations at each sweep point with an epistemic multiplier per sample.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Randomly sample ~{YIELD_POINTS} points over temperature 290–320 K and damage 0–0.08 dpa, "
        "predict breaking stress, and randomly hold out 20% for validation."
    )

    _render_sweep_results("breaking_stress", "ys_sweep")
