import streamlit as st

st.set_page_config(page_title="PatrolIQ Crime Analytics", layout="wide")
st.title("PatrolIQ")

st.markdown("""
This platform analyzes **Chicago crime patterns** using
unsupervised machine learning.

Features:
- Crime hotspot clustering
- Temporal crime analysis
- PCA / t-SNE visualization
- Model comparison using MLflow
            
Use the sidebar to explore different analysis pages.
"""
)