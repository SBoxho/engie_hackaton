import streamlit as st

from app.components.layout import apply_theme

apply_theme()

st.title("Explainability — coming next")
st.write("SHAP feature contributions will explain why each forecast moved up or down.")
