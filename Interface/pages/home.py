import streamlit as st

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

with st.container(border=True):
    col_feature1, spacer, col_feature_text = st.columns([1, 0.1, 1.2])
    with col_feature1:
            st.image("Interface/assets/MatchPredict.png", width=600)
    with col_feature_text:
            subheader_text = "Next Season Prediction"
            write_text = "Prediksi pertandingan EPL Next Season 2025/2026 dengan akurasi tinggi dan analisis mendalam. Dapatkan prediksi pertandingan yang akurat dan analisis mendalam untuk setiap pertandingan."

            centered_content_html = f'''
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start; min-height: 220px; height: 100%; text-align: left;">
                <h3 style="margin-bottom: 10px;">{subheader_text} </h3>
                <p>{write_text}</p>
            </div>
            '''
            st.markdown(centered_content_html, unsafe_allow_html=True)
st.markdown("")

with st.container(border=True):
    col_feature_text2, spacer, col_feature2  = st.columns([1, 0.1, 1.2])
    with col_feature_text2:
        subheader_text = "Historical Match Analysis"
        write_text = "Analisis pertandingan historis untuk mengetahui tren dan pola yang dapat membantu Anda membuat prediksi yang lebih akurat."

        centered_content_html = f'''
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start; min-height: 220px; height: 100%; text-align: left;">
            <h3 style="margin-bottom: 10px;">{subheader_text} </h3>
            <p>{write_text}</p>
        </div>
        '''
        st.markdown(centered_content_html, unsafe_allow_html=True)

    with col_feature2:
        st.image("Interface/assets/MatchPredict.png", width=600)
st.markdown("")

with st.container(border=True):
    col_feature1, spacer, col_feature_text = st.columns([1, 0.1, 1.2])
    with col_feature1:
            st.image("Interface/assets/MatchPredict.png", width=600)
    with col_feature_text:
            subheader_text = "Powered by XGBoost"
            write_text = "Our predictions leverage XGBoost, a state-of-the-art machine learning algorithm that outperforms traditional statistical methods."

            centered_content_html = f'''
            <div style="display: flex; flex-direction: column; justify-content: center; align-items: flex-start; min-height: 220px; height: 100%; text-align: left;">
                <h3 style="margin-bottom: 10px;">{subheader_text} </h3>
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
        <p>Hubungi kami: contact@eplprediction.com</p>
    </div>
    """,
    unsafe_allow_html=True
)