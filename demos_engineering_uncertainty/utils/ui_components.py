import streamlit as st
from app.config import branding
import pandas as pd
from pathlib import Path
from typing import Optional, Dict


def _resolve_logo_path() -> Path:
    raw = Path(branding.LOGO_PATH)
    if raw.is_absolute():
        return raw

    base_repo = Path(__file__).resolve().parents[2]
    base_pkg = Path(__file__).resolve().parents[1]
    candidate_repo = base_repo / raw
    candidate_pkg = base_pkg / raw
    if candidate_repo.exists():
        return candidate_repo
    if candidate_pkg.exists():
        return candidate_pkg
    return candidate_repo  # fallback for error messaging

def app_header():
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        logo_path = _resolve_logo_path()
        if logo_path.exists():
            st.image(str(logo_path), use_container_width=True)
        else:
            st.warning(f"Logo not found at {logo_path}. Update branding.LOGO_PATH.")
    with col_title:
        st.markdown(f"### {branding.APP_TITLE}")
        st.caption(branding.APP_TAGLINE)
        st.markdown("---")

def model_summary_box(title: str, description: str, explainer_path: Optional[str | Path] = None):
    st.subheader(title)
    st.write(description)
    with st.expander("📊 Explanatory graphic"):
        if explainer_path:
            path = Path(explainer_path)
            if not path.is_absolute():
                path = Path(__file__).resolve().parents[2] / path
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.warning(f"Graphic not found at {path}.")
        else:
            st.info("Insert a diagram or figure here to explain where the data comes from (e.g. experiment or complex simulation).")

def geometry_expander(title: str, geom: dict, uncertainties: Optional[Dict] = None):
    with st.expander(title):
        st.markdown("**Fixed geometric / material factors**")
        data = {
            "Parameter": list(geom.keys()),
            "Value": [str(v) for v in geom.values()]
        }
        if uncertainties:
            data["Uncertainty (%)"] = [
                f"{uncertainties.get(k, 0.0) * 100:.1f}%" for k in geom.keys()
            ]
        df = pd.DataFrame(data)
        st.table(df)


def model_parameters_expander(title: str, parameters: list[dict]):
    """Render a parameter table with name, unit, and description."""
    with st.expander(title):
        df = pd.DataFrame(parameters)
        st.table(df)

def data_section(model_name: str, base_path: str):
    st.markdown("### 📥 Data (train / validation)")
    base = Path(base_path)

    files = {
        "Train inputs": base / "train_inputs.csv",
        "Validation inputs": base / "val_inputs.csv",
        "Train outputs": base / "train_outputs.csv",
        "Validation outputs": base / "val_outputs.csv",
    }

    for label, path in files.items():
        with st.expander(f"{label} ({model_name})"):
            if path.exists():
                df = pd.read_csv(path)
                st.dataframe(df, use_container_width=True)
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {label.lower()} CSV",
                    data=csv_bytes,
                    file_name=path.name,
                    mime="text/csv",
                    key=f"{model_name}_{path.stem}_download"
                )
            else:
                st.error(f"File not found: {path}")
