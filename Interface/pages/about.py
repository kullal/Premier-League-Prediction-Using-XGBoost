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

# Main Project Description
st.write(
    """
    Aplikasi Prediksi Liga Premier Inggris ini dibuat untuk membantu Anda menganalisis
    dan memprediksi hasil pertandingan berdasarkan model XGBoost yang canggih.
    """
)

# Project Overview Section
st.subheader("Gambaran Proyek")
st.write(
    """
    Proyek Premier League Prediction menggunakan algoritma machine learning XGBoost untuk memprediksi hasil pertandingan
    Liga Premier Inggris dengan akurasi tinggi. Kami menganalisis data historis pertandingan,
    statistik tim, performa pemain, dan berbagai faktor lain untuk memberikan prediksi yang dapat diandalkan.
    
    Aplikasi ini merupakan hasil dari penelitian mendalam tentang faktor-faktor yang mempengaruhi hasil pertandingan
    sepak bola, dengan fokus khusus pada Liga Premier Inggris. Dengan menggunakan data dari beberapa musim terakhir,
    kami telah melatih model yang mampu memahami pola dan tren dalam pertandingan sepak bola.
    """
)

# Methodology Section
st.subheader("Metodologi")
st.write(
    """
    Kami menggunakan pendekatan berbasis data untuk membangun model prediksi kami:
    
    1. **Pengumpulan Data**: Data pertandingan dari beberapa musim Liga Premier Inggris dikumpulkan, termasuk statistik tim,
       hasil pertandingan, dan berbagai metrik performa.
       
    2. **Pra-pemrosesan Data**: Data mentah dibersihkan dan ditransformasi menjadi fitur-fitur yang dapat digunakan
       untuk melatih model machine learning.
       
    3. **Pemilihan Model**: Setelah mengevaluasi beberapa algoritma, XGBoost dipilih karena kemampuannya dalam menangani
       data kompleks dan memberikan hasil prediksi yang akurat.
       
    4. **Pelatihan dan Validasi**: Model dilatih menggunakan data historis dan divalidasi untuk memastikan akurasi prediksi.
       
    5. **Implementasi**: Model yang terlatih diimplementasikan dalam aplikasi web menggunakan Streamlit untuk memberikan
       antarmuka yang user-friendly.
    """
)

# Features Section
st.subheader("Fitur Utama")
st.write(
    """
    - **Prediksi Hasil Pertandingan**: Prediksi hasil pertandingan untuk musim mendatang dengan tingkat akurasi tinggi.
    
    - **Analisis Pertandingan Historis**: Analisis mendalam tentang pertandingan-pertandingan sebelumnya antara dua tim,
      membantu pengguna memahami pola dan tren historis.
      
    - **Model XGBoost yang Canggih**: Menggunakan algoritma XGBoost yang merupakan state-of-the-art dalam machine learning
      untuk memberikan prediksi yang lebih akurat dibandingkan metode statistik tradisional.
      
    - **Antarmuka yang User-Friendly**: Desain antarmuka yang intuitif dan mudah digunakan, memungkinkan pengguna
      untuk dengan cepat mendapatkan prediksi dan analisis yang mereka butuhkan.
      
    - **Analisis Mendalam**: Selain hasil prediksi, aplikasi juga menyediakan analisis mendalam tentang faktor-faktor
      yang mempengaruhi hasil pertandingan.
    """
)

# Technical Details
st.subheader("Detail Teknis")
st.write(
    """
    **Teknologi yang Digunakan**:
    - Python sebagai bahasa pemrograman utama
    - XGBoost untuk algoritma machine learning
    - Pandas dan NumPy untuk manipulasi data
    - Streamlit untuk pengembangan antarmuka web
    - Matplotlib dan Seaborn untuk visualisasi data
    
    **Sumber Data**:
    Data yang digunakan dalam proyek ini berasal dari berbagai sumber terpercaya yang menyediakan statistik
    Liga Premier Inggris, termasuk hasil pertandingan, statistik tim dan pemain, serta berbagai metrik performa lainnya.
    """
)

# Future Development
st.subheader("Pengembangan Masa Depan")
st.write(
    """
    Kami berencana untuk terus mengembangkan aplikasi ini dengan menambahkan fitur-fitur baru seperti:
    
    - Analisis pemain individual dan pengaruhnya terhadap hasil pertandingan
    - Integrasi data real-time untuk prediksi yang lebih akurat
    - Visualisasi data yang lebih komprehensif
    - Perluasan ke liga sepak bola lainnya
    """
)

st.subheader("Tim Pengembang")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Riski Yuniar Pratama**")
    st.write("Model Builder")

with col2:
    st.markdown("**Gangsar Reka Pambudi**")
    st.write("Project Manager & Frontend Developer")

with col3:
    st.markdown("**Yumna Salma Salsabilla**")
    st.write("UI Designer")

st.subheader("Kontak")
st.write("Jika ada pertanyaan atau masukan, silakan hubungi kami di instagram salah satu pengembang.") 