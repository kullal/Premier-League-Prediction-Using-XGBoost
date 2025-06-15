import streamlit as st
import os

st.set_page_config(
page_title="EPL Prediction",
page_icon="⚽",
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

# Hero Section Column
kolom_hero1, kolom_hero2, kolom_hero3 = st.columns([1.7, 0.1, 1.2]) # Proporsi lebar kolom

with kolom_hero1:
    st.image("Interface/assets/Hero.png") 
with kolom_hero3:
    st.markdown("####")
    st.header("Predict Match of EPL Next Season 2025/2026!! 🔥")
    st.write("")
    st.write("""
    Prediksi Match EPL Next Season 2025/2026 dengan akurasi tinggi dan analisis mendalam.
    Dapatkan prediksi pertandingan yang akurat dan analisis mendalam untuk setiap pertandingan.
    Prediksi pertandingan EPL Next Season 2025/2026 dengan akurasi tinggi dan analisis mendalam.
    Dapatkan prediksi pertandingan yang akurat dan analisis mendalam untuk setiap pertandingan.
    """)
    st.markdown("##")
    if st.button("Predict Your Match Now!", type="primary", help="Klik untuk memulai prediksi"):
        st.switch_page("pages/predict.py")
st.markdown("---")

# Creating Feature Section
st.markdown("<h1 style='text-align: center;'>Features</h1>", unsafe_allow_html=True)
st.markdown("")

# Define image paths - use default image as fallback
default_image = "Interface/assets/MatchPredict.png"
next_season_image = default_image
historical_analysis_image = default_image
xgboost_powered_image = default_image

# Check if feature images exist and use them if they do
if os.path.exists("Interface/assets/feature_images/next_season_prediction.jpg"):
    next_season_image = "Interface/assets/feature_images/next_season_prediction.jpg"
if os.path.exists("Interface/assets/feature_images/historical_analysis.jpg"):
    historical_analysis_image = "Interface/assets/feature_images/historical_analysis.jpg"
if os.path.exists("Interface/assets/feature_images/xgboost_powered.jpg"):
    xgboost_powered_image = "Interface/assets/feature_images/xgboost_powered.jpg"

# Feature 1: Next Season Prediction
with st.container(border=True):
    col_feature1, spacer, col_feature_text = st.columns([1, 0.1, 1.2])
    with col_feature1:
        # Create a container with fixed height to match the image
        with st.container():
            # Add some vertical space to center the image
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.image(next_season_image, width=600)
    
    with col_feature_text:
        subheader_text = "Next Season Prediction"
        write_text = "Prediksi pertandingan EPL Next Season 2025/2026 dengan akurasi tinggi dan analisis mendalam. Dapatkan prediksi pertandingan yang akurat dan analisis mendalam untuk setiap pertandingan."

        # Use CSS to vertically center the content
        centered_content_html = f'''
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; min-height: 300px;">
            <h3 style="margin-bottom: 20px;">{subheader_text}</h3>
            <p>{write_text}</p>
        </div>
        '''
        st.markdown(centered_content_html, unsafe_allow_html=True)

st.markdown("")

# Feature 2: Historical Match Analysis
with st.container(border=True):
    col_feature_text2, spacer, col_feature2 = st.columns([1, 0.1, 1.2])
    
    with col_feature_text2:
        subheader_text = "Historical Match Analysis"
        write_text = "Analisis pertandingan historis untuk mengetahui tren dan pola yang dapat membantu Anda membuat prediksi yang lebih akurat."

        # Use CSS to vertically center the content
        centered_content_html = f'''
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; min-height: 300px;">
            <h3 style="margin-bottom: 20px;">{subheader_text}</h3>
            <p>{write_text}</p>
        </div>
        '''
        st.markdown(centered_content_html, unsafe_allow_html=True)

    with col_feature2:
        # Create a container with fixed height to match the image
        with st.container():
            # Add some vertical space to center the image
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.image(historical_analysis_image, width=600)

st.markdown("")

# Feature 3: Powered by XGBoost
with st.container(border=True):
    col_feature1, spacer, col_feature_text = st.columns([1, 0.1, 1.2])
    
    with col_feature1:
        # Create a container with fixed height to match the image
        with st.container():
            # Add some vertical space to center the image
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
            st.image(xgboost_powered_image, width=600)
    
    with col_feature_text:
        subheader_text = "Powered by XGBoost"
        write_text = "Prediksi kami menggunakan XGBoost, algoritma machine learning canggih yang memberikan hasil lebih akurat dibandingkan metode statistik tradisional."

        # Use CSS to vertically center the content
        centered_content_html = f'''
        <div style="display: flex; flex-direction: column; justify-content: center; height: 100%; min-height: 300px;">
            <h3 style="margin-bottom: 20px;">{subheader_text}</h3>
            <p>{write_text}</p>
        </div>
        '''
        st.markdown(centered_content_html, unsafe_allow_html=True)

# Membuat Footer
st.markdown("---") # Garis pemisah sebelum footer
st.markdown(
    """
    <div style="text-align: center; padding: 20px; color: #FFFFFF;">
        <p>© 2024 EPL Prediction App | Dibuat dengan Streamlit</p>
        <p>Hubungi kami: eplprediction@contact.com</p>
    </div>
    """,
    unsafe_allow_html=True
)