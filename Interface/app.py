import streamlit as st
import time

# Konfigurasi halaman global
st.set_page_config(
    page_title="EPL Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide sidebar completely
hide_sidebar_style = """
    <style>
        [data-testid="collapsedControl"] {display: none;}
        section[data-testid="stSidebar"] {display: none;}
        div.block-container {padding-left: 2rem; padding-right: 2rem;}
    </style>
"""
st.markdown(hide_sidebar_style, unsafe_allow_html=True)

#Creating Navbar
col1, col2, col3, col4 = st.columns([12, 1, 1, 1])
with col1:
    st.subheader("⚽ EPL Prediction")
with col2:
    if st.button("Home"):
        st.switch_page("pages/home.py")
with col3:
    if st.button("Predict"):
        st.switch_page("pages/predict.py")
with col4:
    if st.button("About"):
        st.switch_page("pages/about.py")
st.markdown("####")

# Automatically redirect to home.py when app.py is run
st.switch_page("pages/home.py")

# Konten Halaman Home
# ... (sisa kode home.py tetap sama seperti sebelumnya) 