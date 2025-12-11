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

from utils import ui_components
from models.temperature_model import model as temp_model
from models.irradiation_model import model as irr_model
from models.yield_strength_model import model as ys_model

TARGET_POINTS = 20
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


def _write_dataset(output_root, train_inputs, val_inputs, train_outputs, val_outputs):
    output_root.mkdir(parents=True, exist_ok=True)
    train_inputs.to_csv(output_root / "train_inputs.csv", index=False)
    val_inputs.to_csv(output_root / "val_inputs.csv", index=False)
    train_outputs.to_csv(output_root / "train_outputs.csv", index=False)
    val_outputs.to_csv(output_root / "val_outputs.csv", index=False)


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
    _write_dataset(ROOT / "data" / "temperature", train_inputs, val_inputs, train_outputs, val_outputs)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_irradiation_sweep():
    rng = np.random.default_rng()
    # flux values expressed in units of "x 1e18 n/m²/s" to keep magnitudes near unity
    flux_values = np.linspace(0.0, 5.0, TARGET_POINTS)
    samples = irr_model.sample_model({"flux_avg": flux_values}, n_samples=1, rng=rng)[:, 0]

    inputs_df = pd.DataFrame({"flux_avg": flux_values})
    outputs_df = pd.DataFrame({"dpa_peak": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    _write_dataset(ROOT / "data" / "irradiation", train_inputs, val_inputs, train_outputs, val_outputs)
    return train_inputs, val_inputs, train_outputs, val_outputs


def run_yield_strength_sweep():
    rng = np.random.default_rng()
    n_temp = int(np.ceil(np.sqrt(TARGET_POINTS)))
    n_dpa = int(np.ceil(TARGET_POINTS / n_temp))

    temp_values = np.linspace(300.0, 1200.0, n_temp)
    dpa_values = np.linspace(0.0, 5.0, n_dpa)
    T_grid, dpa_grid = np.meshgrid(temp_values, dpa_values, indexing="ij")

    T_flat = T_grid.ravel()
    dpa_flat = dpa_grid.ravel()
    samples = ys_model.sample_model(
        {"temperature": T_flat, "dpa": dpa_flat},
        n_samples=1,
        rng=rng
    )[:, 0]

    inputs_df = pd.DataFrame({"temperature": T_flat, "dpa": dpa_flat})
    outputs_df = pd.DataFrame({"yield_strength": samples})
    train_inputs, val_inputs, train_outputs, val_outputs = _split_train_val(inputs_df, outputs_df, rng)
    _write_dataset(ROOT / "data" / "yield_strength", train_inputs, val_inputs, train_outputs, val_outputs)
    return train_inputs, val_inputs, train_outputs, val_outputs


def _render_sweep_results(label_prefix, state_key):
    sweep = st.session_state.get(state_key)
    if not sweep:
        return

    train_inputs = sweep["train_inputs"]
    val_inputs = sweep["val_inputs"]
    train_outputs = sweep["train_outputs"]
    val_outputs = sweep["val_outputs"]

    st.success(f"Generated {len(train_inputs)} train / {len(val_inputs)} validation samples.")
    st.caption("Train preview")
    st.dataframe(pd.concat([train_inputs, train_outputs], axis=1), use_container_width=True)
    st.caption("Validation preview")
    st.dataframe(pd.concat([val_inputs, val_outputs], axis=1), use_container_width=True)

    st.markdown("#### Download CSVs")
    downloads = {
        "Train inputs": train_inputs,
        "Validation inputs": val_inputs,
        "Train outputs": train_outputs,
        "Validation outputs": val_outputs,
    }
    cols = st.columns(4)
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
    train_df = pd.concat([train_inputs, train_outputs], axis=1)
    train_df["split"] = "train"
    val_df = pd.concat([val_inputs, val_outputs], axis=1)
    val_df["split"] = "validation"
    all_df = pd.concat([train_df, val_df], ignore_index=True)

    if len(input_cols) == 1:
        x_col = input_cols[0]
        chart = (
            alt.Chart(all_df)
            .mark_circle(size=70, opacity=0.8)
            .encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(output_col, title=output_col),
                color=alt.Color("split", title="split"),
                tooltip=input_cols + [output_col, "split"],
            )
        )
    else:
        x_col = input_cols[0]
        chart = (
            alt.Chart(all_df)
            .mark_circle(size=70, opacity=0.8)
            .encode(
                x=alt.X(x_col, title=x_col),
                y=alt.Y(output_col, title=output_col),
                color=alt.Color("split", title="split"),
                tooltip=input_cols + [output_col, "split"],
            )
        )

    st.altair_chart(chart, use_container_width=True)

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

    if st.button("Run temperature sweep"):
        train_inputs, val_inputs, train_outputs, val_outputs = run_temperature_sweep()
        st.session_state["temp_sweep"] = {
            "train_inputs": train_inputs,
            "val_inputs": val_inputs,
            "train_outputs": train_outputs,
            "val_outputs": val_outputs,
        }

    _render_sweep_results("temperature", "temp_sweep")

with tab2:
    ui_components.model_summary_box(
        "Model 2: Irradiation",
        "Maps average neutron flux (entered as multiples of 1e18 n/m²/s) to peak dpa at the component, incorporating shielding and geometric factors "
        "with stochastic variation to mimic transport and spectrum effects."
    )

    geom = irr_model.get_default_geometry()
    irr_unc = irr_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed shielding / exposure parameters", geom, irr_unc)
    st.caption("Enter flux in units of ×1e18 n/m²/s; parameter uncertainties are applied independently per point.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Sweep {TARGET_POINTS} evenly spaced flux points from 0 to 5 (×1e18 n/m²/s), sample dpa_peak values "
        "with perturbed shielding/geometric factors, and randomly hold out 20% for validation."
    )

    if st.button("Run irradiation sweep"):
        train_inputs, val_inputs, train_outputs, val_outputs = run_irradiation_sweep()
        st.session_state["irr_sweep"] = {
            "train_inputs": train_inputs,
            "val_inputs": val_inputs,
            "train_outputs": train_outputs,
            "val_outputs": val_outputs,
        }

    _render_sweep_results("irradiation", "irr_sweep")

with tab3:
    ui_components.model_summary_box(
        "Model 3: Yield Strength",
        "Maps temperature and dpa to yield strength for EUROFER first-wall backing, representing degradation "
        "of mechanical properties under thermal and irradiation loading with variability from material uncertainties."
    )

    geom = ys_model.get_default_geometry()
    ys_unc = ys_model.get_geometry_uncertainties()
    ui_components.geometry_expander("Fixed material / strain-rate parameters", geom, ys_unc)
    st.caption("Material property uncertainties are applied as independent Gaussian perturbations at each sweep point.")

    st.markdown("### 🔧 Sweep setup")
    st.write(
        f"Create a meshgrid across temperature 300 to 1200 K and damage 0 to 5 dpa (~{TARGET_POINTS} points total), "
        "sample yield strength with noise, and randomly hold out 20% for validation."
    )

    if st.button("Run yield strength sweep"):
        train_inputs, val_inputs, train_outputs, val_outputs = run_yield_strength_sweep()
        st.session_state["ys_sweep"] = {
            "train_inputs": train_inputs,
            "val_inputs": val_inputs,
            "train_outputs": train_outputs,
            "val_outputs": val_outputs,
        }

    _render_sweep_results("yield_strength", "ys_sweep")
