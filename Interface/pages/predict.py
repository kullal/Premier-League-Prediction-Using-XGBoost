# Interface/pages/predict.py
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import pandas as pd
from PIL import Image

# Tambahkan path ke direktori models
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

# Import fungsi prediksi dari lokasi yang benar
from predict_future_modules import predict_future_match, display_result_chart, get_teams_and_referees, get_default_odds
from predict_history_modules import predict_history_matchup

st.set_page_config(
    page_title="EPL Prediction - Predict",
    page_icon="🔮", # Ganti ikon jika mau
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

st.title("Prediksi Pertandingan 🙏🏿")
st.write("Masukkan detail pertandingan di bawah ini untuk mendapatkan prediksi. Anda juga bisa memilih jenis prediksi yang ingin Anda lakukan.")
st.markdown("####")

# Fungsi untuk mendapatkan path logo tim
def get_team_logo_path(team_name):
    # Mapping untuk mencocokkan nama tim dengan nama file logo
    team_name_to_logo_map = {
        # Team names that differ between model and logo files
        "Tottenham": "Tottenham_Hotspur",
        "Leicester": "Leicester_City",
        "West Ham": "West_Ham_United",
        "Newcastle": "Newcastle_United",
        "Brighton": "Brighton_&_Hove_Albion",
        "Wolves": "Wolverhampton_Wanderers",
        "Leeds": "Leeds_United",
        "Norwich": "Norwich_City",
        "West Brom": "West_Bromwich_Albion",
        # Add any other mappings as needed
    }
    
    # Get the logo filename (either from mapping or directly)
    logo_filename = team_name_to_logo_map.get(team_name, team_name)
    
    # Cek apakah file logo tim ada
    logo_path = f"Interface/assets/team_logos/{logo_filename.replace(' ', '_')}.png"
    if os.path.exists(logo_path):
        return logo_path
    else:
        # Gunakan logo placeholder jika tidak ada logo tim
        return "Interface/assets/Hero.png"

# Pilih jenis prediksi di luar form
prediction_type = st.selectbox(
    "Pilih Jenis Prediksi:",
    ["Prediksi Berdasarkan Riwayat", "Prediksi Pertandingan Baru"]
)

# Dapatkan daftar tim
teams_and_refs = get_teams_and_referees()
if teams_and_refs[0] is not None:
    teams = ["Pilih Tim"] + teams_and_refs[0]
    referees = teams_and_refs[1]
else:
    # Fallback list with correct team names matching the model's expected format
    teams = ["Pilih Tim", "Arsenal", "Man City", "Liverpool", "Chelsea", "Man United", "Tottenham", 
             "Leicester", "West Ham", "Everton", "Aston Villa", "Newcastle", "Crystal Palace",
             "Brighton", "Southampton", "Burnley", "Wolves", "Leeds",
             "Watford", "Norwich", "Brentford", "Fulham", "West Brom", "Sheffield United", 
             "Bournemouth", "Ipswich", "Luton", "Nott'm Forest"]
    referees = ["Michael Oliver"]

# Container untuk Input Prediksi
with st.container(border=True):
    col1, col2, col3 = st.columns([5, 5, 5])
    
    with col2:
        st.title("Detail Pertandingan")

    # Membuat kolom untuk meletakkan elemen di tengah
    col1, col2, col3 = st.columns([3, 2, 3])
    
    with col1:
        home_team = st.selectbox("Pilih Tim Tuan Rumah:", teams, key=f"home_team_{prediction_type}")
        with st.container(border=True):
            # Tampilkan logo tim tuan rumah jika tim dipilih
            if home_team != "Pilih Tim":
                home_logo_path = get_team_logo_path(home_team)
                st.image(home_logo_path, caption=f"{home_team} (Tuan Rumah)")

    with col3:
        away_team = st.selectbox("Pilih Tim Tandang:", teams, key=f"away_team_{prediction_type}")
        with st.container(border=True):
        # Tampilkan logo tim tandang jika tim dipilih
            if away_team != "Pilih Tim":
                away_logo_path = get_team_logo_path(away_team)
                st.image(away_logo_path, caption=f"{away_team} (Tandang)")

    with col2:
        # Tampilkan VS hanya jika kedua tim sudah dipilih
        if home_team != "Pilih Tim" and away_team != "Pilih Tim":
            st.markdown("#")
            st.markdown("#")
            st.markdown("#")
            st.markdown(f"<h1 style='text-align: center;'>VS</h1>", unsafe_allow_html=True)
    
    st.markdown("####")
    # Tambahkan input tambahan untuk Prediksi Pertandingan Baru
    if prediction_type == "Prediksi Pertandingan Baru":
        st.subheader("Detail Tambahan")
        
        col1, col2 = st.columns([2, 2])
        with col1:
            match_date = st.date_input("Tanggal Pertandingan:", key=f"date_{prediction_type}")
        with col2:
            referee = st.selectbox("Wasit:", referees, index=0)
        
        # Tambahkan opsi untuk odds
        show_odds = st.checkbox("Kustomisasi Odds Pertandingan", value=False)
        
        if show_odds:
            st.subheader("Odds Pertandingan")
            
            # Dapatkan default odds
            default_odds = get_default_odds()
            
            # Buat 3 kolom untuk odds utama
            odds_col1, odds_col2, odds_col3 = st.columns(3)
            
            with odds_col1:
                b365h = st.number_input("Home Win Odds:", value=default_odds["B365H"], min_value=1.0, step=0.1, format="%.2f")
            
            with odds_col2:
                b365d = st.number_input("Draw Odds:", value=default_odds["B365D"], min_value=1.0, step=0.1, format="%.2f")
            
            with odds_col3:
                b365a = st.number_input("Away Win Odds:", value=default_odds["B365A"], min_value=1.0, step=0.1, format="%.2f")
            
            # Buat custom odds dictionary
            custom_odds = {
                "B365H": b365h, "B365D": b365d, "B365A": b365a,
                "BSH": b365h, "BSD": b365d, "BSA": b365a,
                "BWH": b365h, "BWD": b365d, "BWA": b365a,
                "PSH": b365h, "PSD": b365d, "PSA": b365a,
                "MaxH": b365h, "MaxD": b365d, "MaxA": b365a,
                "AvgH": b365h, "AvgD": b365d, "AvgA": b365a
            }
        else:
            custom_odds = None

    col1, col2, col3 = st.columns([3.5, 1, 3.5])

    with col2:
        submit_button = st.button(label="Dapatkan Prediksi!")

if submit_button:
    if home_team == "Pilih Tim" or away_team == "Pilih Tim":
        st.warning("Mohon pilih Tim Tuan Rumah dan Tim Tandang.")
    elif home_team == away_team:
        st.warning("Tim Tuan Rumah dan Tim Tandang tidak boleh sama.")
    else:
        with st.spinner("Sedang memproses prediksi..."):
            if prediction_type == "Prediksi Pertandingan Baru":
                # Menggunakan predict_future_match dari modul dengan odds kustom jika ada
                result = predict_future_match(home_team, away_team, match_date, referee, custom_odds)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("####")
                    st.success(f"Berhasil memprediksi pertandingan {home_team} vs {away_team}:")
                    st.markdown("####")
                    
                    with st.container(border=True):
                        st.markdown(f"<h2 style='text-align: center;'>Prediksi Pertandingan Baru</h2>", unsafe_allow_html=True)

                        col1, col2= st.columns([2, 1])

                        with col1:
                            with st.container(border=True):
                                st.subheader("Statistik Prediksi")

                                # Data untuk chart
                                win_pct = int(result['home_win_prob'] * 100)
                                draw_pct = int(result['draw_prob'] * 100)
                                loss_pct = int(result['away_win_prob'] * 100)
                                
                                # Membuat chart dengan matplotlib
                                fig = display_result_chart(result['home_win_prob'], result['draw_prob'], result['away_win_prob'])
                                
                                # Menampilkan chart di Streamlit
                                st.pyplot(fig)

                                column1, column2, column3 = st.columns([1, 1, 1])

                                with column1:
                                    st.markdown(f"<p style='text-align: center;'>Home<br>{win_pct}%</p>", unsafe_allow_html=True)

                                with column2:
                                    st.markdown(f"<p style='text-align: center;'>Draw<br>{draw_pct}%</p>", unsafe_allow_html=True)

                                with column3:
                                    st.markdown(f"<p style='text-align: center;'>Away<br>{loss_pct}%</p>", unsafe_allow_html=True)
                        
                            with st.container(border=True):
                                    st.subheader("Detail Pertandingan")
                                    # Tampilkan hasil prediksi
                                    st.markdown(f"**Tanggal Pertandingan:** `{result['match_date']}`")
                                    st.markdown(f"**Hasil yang Diprediksi:** `{result['predicted_outcome']}`")
                                    st.markdown(f"**Wasit:** `{result['referee']}`")
                                    
                                    # Tampilkan odds jika custom odds digunakan
                                    if custom_odds is not None:
                                        st.markdown("**Odds yang Digunakan:**")
                                        odds_col1, odds_col2, odds_col3 = st.columns(3)
                                        with odds_col1:
                                            st.markdown(f"Home Win: `{result['odds']['B365H']}`")
                                        with odds_col2:
                                            st.markdown(f"Draw: `{result['odds']['B365D']}`")
                                        with odds_col3:
                                            st.markdown(f"Away Win: `{result['odds']['B365A']}`")

                        with col2:
                            with st.container(border=True):
                                st.subheader("Tim Pemenang")
                                # Tampilkan logo tim pemenang berdasarkan prediksi
                                winner_team = result['predicted_outcome'].split(" ")[0]
                                if winner_team in ["Home", "Draw", "Away"]:
                                    if winner_team == "Home":
                                        winner_logo = get_team_logo_path(home_team)
                                        winner_name = home_team
                                    elif winner_team == "Away":
                                        winner_logo = get_team_logo_path(away_team)
                                        winner_name = away_team
                                    else:  # Draw
                                        st.image("Interface/assets/team_logos/DRAW.png")
                                        winner_name = "Draw"
                                        st.markdown(f"<p style='text-align: center;'>{winner_name}</p>", unsafe_allow_html=True)
                                else:
                                    st.image("Interface/assets/Hero.png")
                                    winner_name = result['predicted_outcome']
                                
                                if winner_team != "Draw":
                                    st.image(winner_logo)
                                    st.markdown(f"<p style='text-align: center;'>{winner_name}</p>", unsafe_allow_html=True)
            else:
                # Panggil fungsi predict_history_matchup
                result = predict_history_matchup(home_team, away_team)
                
                if "error" in result:
                    st.error(f"Error: {result['error']}")
                else:
                    st.markdown("####")
                    st.success(f"Berhasil menganalisis pertandingan {home_team} vs {away_team}:")
                    st.markdown("####")
                    
                    with st.container(border=True):
                        st.markdown(f"<h2 style='text-align: center;'>Analisis Riwayat Pertandingan</h2>", unsafe_allow_html=True)

                        col1, col2= st.columns([2, 1])

                        with col1:
                            with st.container(border=True):
                                st.subheader("Statistik Prediksi")

                                # Data untuk chart
                                win_pct = int(result['home_win_prob'] * 100)
                                draw_pct = int(result['draw_prob'] * 100)
                                loss_pct = int(result['away_win_prob'] * 100)
                                
                                # Membuat chart dengan matplotlib
                                fig, ax = plt.subplots(figsize=(10, 2))
                                
                                # Membuat segmented bar chart
                                segments = [win_pct, draw_pct, loss_pct]
                                colors = ['#5cb85c', '#f0ad4e', '#d9534f']  # hijau, kuning, merah
                                labels = ['Home', 'Draw', 'Away']
                                
                                # Plot segmented bar
                                left = 0
                                for i, (segment, color) in enumerate(zip(segments, colors)):
                                    ax.barh(0, segment, left=left, height=0.5, color=color)
                                    # Menambahkan label di tengah segmen
                                    ax.text(left + segment/2, 0, f"{labels[i]} {segment}%", 
                                            ha='center', va='center', color='black', fontweight='bold')
                                    left += segment
                                
                                # Menghilangkan sumbu dan border
                                ax.axis('off')
                                
                                # Menggunakan tema Streamlit untuk background
                                fig.patch.set_alpha(0.0)  # Transparan
                                ax.set_facecolor('none')  # Transparan
                                
                                plt.xlim(0, 100)  # Memastikan skala 0-100
                                plt.tight_layout()
                                
                                # Menampilkan chart di Streamlit
                                st.pyplot(fig)

                                column1, column2, column3 = st.columns([1, 1, 1])

                                with column1:
                                    st.markdown(f"<p style='text-align: center;'>Home<br>{win_pct}%</p>", unsafe_allow_html=True)

                                with column2:
                                    st.markdown(f"<p style='text-align: center;'>Draw<br>{draw_pct}%</p>", unsafe_allow_html=True)

                                with column3:
                                    st.markdown(f"<p style='text-align: center;'>Away<br>{loss_pct}%</p>", unsafe_allow_html=True)

                            with st.container(border=True):
                                st.subheader("Validation with real match")
                                # Tampilkan hasil prediksi
                                st.markdown(f"**Tanggal Pertandingan:** `{result['match_date']}`")
                                st.markdown(f"**Hasil yang Diprediksi:** `{result['predicted_outcome']}`")
                                
                                # Tampilkan hasil sebenarnya jika tersedia
                                if "actual_outcome" in result:
                                    st.markdown(f"**Hasil Sebenarnya:** `{result['actual_outcome']} ({result['actual_score']})`")
                                            

                        with col2:
                            with st.container(border=True):
                                st.subheader("Tim Pemenang")
                                # Tampilkan logo tim pemenang berdasarkan prediksi
                                winner_team = result['predicted_outcome'].split(" ")[0]
                                if winner_team in ["Home", "Draw", "Away"]:
                                    if winner_team == "Home":
                                        winner_logo = get_team_logo_path(home_team)
                                        winner_name = home_team
                                    elif winner_team == "Away":
                                        winner_logo = get_team_logo_path(away_team)
                                        winner_name = away_team
                                    else:  # Draw
                                        st.image("Interface/assets/team_logos/DRAW.png")
                                        winner_name = "Draw"
                                        st.markdown(f"<p style='text-align: center;'>{winner_name}</p>", unsafe_allow_html=True)
                                else:
                                    st.image("Interface/assets/Hero.png")
                                    winner_name = result['predicted_outcome']
                                
                                if winner_team != "Draw":
                                    st.image(winner_logo)
                                    st.markdown(f"<p style='text-align: center;'>{winner_name}</p>", unsafe_allow_html=True)