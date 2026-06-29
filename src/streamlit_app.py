"""Minimal Streamlit demo shell for AdvectNet outputs."""

from __future__ import annotations

from pathlib import Path

import streamlit as st


st.set_page_config(page_title="AdvectNet Demo", layout="wide")
st.title("AdvectNet Satellite Temporal Super-Resolution")

st.write(
    "Use the Kaggle notebooks to generate validation figures and prediction PNGs, "
    "then place them in `results/` for lightweight demo viewing."
)

results = Path("results")
images = sorted(list(results.glob("*.png")) + list(results.glob("*.jpg")) + list(results.glob("*.gif")))

if not images:
    st.info("No result images found in `results/`.")
else:
    for path in images:
        st.subheader(path.name)
        st.image(str(path), use_container_width=True)
