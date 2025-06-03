import streamlit as st

st.set_page_config(
    page_title="EPL Prediction - About",
    page_icon="ℹ️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

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

st.title("Tentang Aplikasi Ini")
st.write(
    """
    Aplikasi Prediksi Liga Premier Inggris ini dibuat untuk membantu Anda menganalisis
    dan memprediksi hasil pertandingan berdasarkan model XGBoost.

    **Fitur Utama:**
    - Prediksi hasil pertandingan.
    - Analisis statistik (akan datang).
    - Informasi tim (akan datang).

    Dibuat dengan ❤️ menggunakan Streamlit.
    """
)

st.subheader("Tim Pengembang")
st.write("Mahasiswa Luar Biasa (Isi dengan nama Anda/tim)")

st.subheader("Kontak")
st.write("Jika ada pertanyaan atau masukan, silakan hubungi kami di email@contoh.com.") 