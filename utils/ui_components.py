import streamlit as st
from app.config import branding
import pandas as pd
from pathlib import Path

def app_header():
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.markdown("**Company Logo**")
        st.caption("Drop your logo file into `app/assets` and update `LOGO_PATH` in `branding.py`.")
    with col_title:
        st.markdown(f"### {branding.APP_TITLE}")
        st.caption(branding.APP_TAGLINE)
        st.markdown("---")

def model_summary_box(title: str, description: str):
    st.subheader(title)
    st.write(description)
    with st.expander("📊 Explanatory graphic placeholder"):
        st.info("Insert a diagram or figure here to explain where the data comes from (e.g. experiment or complex simulation).")

def geometry_expander(title: str, geom: dict):
    with st.expander(title):
        st.markdown("**Fixed geometric / material factors**")
        df = pd.DataFrame(
            {
                "Parameter": list(geom.keys()),
                "Value": [str(v) for v in geom.values()]
            }
        )
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
