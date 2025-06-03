import pandas as pd
import xgboost as xgb
import joblib
import json
import os
import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt

MODEL_DIR = "models"
DATA_DIR = "Combined Dataset"
FUTURE_DATA_DIR = "Dataset EPL New" # Directory for 2024-2025 data

def display_result_chart(home_win, draw, away_win):
    # Konversi probabilitas ke persentase
    total = home_win + draw + away_win
    home_win_pct = int(home_win * 100 / total)
    draw_pct = int(draw * 100 / total)
    away_win_pct = int(away_win * 100 / total)
    
    # Buat figure
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # Buat bar horizontal berdasarkan persentase
    bar_height = 0.3
    bar_width = 1.0
    
    # Buat 3 bar dengan warna berbeda
    ax.barh(0, home_win_pct/100, height=bar_height, left=0, color='#4CAF50')
    ax.barh(0, draw_pct/100, height=bar_height, left=home_win_pct/100, color='#FFC107')
    ax.barh(0, away_win_pct/100, height=bar_height, left=(home_win_pct+draw_pct)/100, color='#F44336')
    
    # Tambahkan teks di atas bar
    ax.text(home_win_pct/200, 0.5, f"Win {int(home_win)}\n{home_win_pct}%", 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax.text(home_win_pct/100 + draw_pct/200, 0.5, f"Draw {int(draw)}\n{draw_pct}%", 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    ax.text((home_win_pct+draw_pct)/100 + away_win_pct/200, 0.5, f"Lost {int(away_win)}\n{away_win_pct}%", 
            ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Tambahkan garis berwarna di bawah teks
    ax.plot([0, home_win_pct/100], [-0.1, -0.1], color='#4CAF50', linewidth=3)
    ax.plot([home_win_pct/100, (home_win_pct+draw_pct)/100], [-0.1, -0.1], color='#FFC107', linewidth=3)
    ax.plot([(home_win_pct+draw_pct)/100, 1], [-0.1, -0.1], color='#F44336', linewidth=3)
    
    # Hapus sumbu dan border
    ax.axis('off')
    
    # Atur batas
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.2, 1)
    
    plt.tight_layout()
    plt.show()

# --- Feature Name Cleaning (MUST be identical to preprocess_data.py) ---
def clean_feature_name(col_name):
    """Cleans a column name to be XGBoost compatible."""
    import re
    new_name = str(col_name).replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
    new_name = new_name.replace('[', '_').replace(']', '_').replace(' ', '_')
    new_name = "".join(char for char in new_name if char.isalnum() or char == '_')
    new_name = new_name.strip('_') 
    if not new_name: new_name = 'unnamed_col_fallback'
    return new_name

# --- Load Model, Encoders, and Model Feature Names ---
def load_dependencies():
    """Loads model, encoders, and the list of feature names the model expects."""
    model_path = os.path.join(MODEL_DIR, "xgboost_epl_model.json")
    encoders_path = os.path.join(DATA_DIR, "label_encoders.joblib")
    xtrain_path = os.path.join(DATA_DIR, "X_train.csv")

    if not all(os.path.exists(p) for p in [model_path, encoders_path, xtrain_path]):
        print("Error: Model, encoders, or X_train.csv not found. Ensure all prerequisite files exist.")
        print(f"Missing: {'model' if not os.path.exists(model_path) else ''} {'encoders' if not os.path.exists(encoders_path) else ''} {'X_train' if not os.path.exists(xtrain_path) else ''}")
        return None, None, None

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    label_encoders = joblib.load(encoders_path)
    model_feature_names = pd.read_csv(xtrain_path, nrows=0).columns.tolist()
    
    print("Model, label encoders, and model feature names loaded successfully.")
    print(f"Model expects {len(model_feature_names)} features.")
    return model, label_encoders, model_feature_names

# --- Calculate Historical Features (adapted from predict_match.py) ---
def get_historical_features(home_team_name, away_team_name, date_of_match, historical_data_df):
    features = {}
    teams = {'HomeTeam': home_team_name, 'AwayTeam': away_team_name} # Match keys in historical_df
    windows = [3, 5, 10]

    if not pd.api.types.is_datetime64_any_dtype(historical_data_df['Date']):
        historical_data_df['Date'] = pd.to_datetime(historical_data_df['Date'], errors='coerce')
    
    # Ensure date_of_match is also datetime
    if not isinstance(date_of_match, pd.Timestamp):
        date_of_match = pd.to_datetime(date_of_match, errors='coerce')
    if pd.isna(date_of_match):
        print("Error: Invalid date_of_match provided for historical feature calculation.")
        return None

    for team_key_prefix, team_name in teams.items(): # team_key_prefix will be 'HomeTeam' or 'AwayTeam'
        team_matches = historical_data_df[
            ((historical_data_df['HomeTeam'] == team_name) | (historical_data_df['AwayTeam'] == team_name)) &
            (historical_data_df['Date'] < date_of_match)
        ].sort_values(by='Date', ascending=False)

        for N in windows:
            col_prefix = f"{team_key_prefix}" # e.g. HomeTeam_AvgGS_L3
            
            last_n_matches = team_matches.head(N)
            if len(last_n_matches) < N:
                features[clean_feature_name(f'{col_prefix}_AvgGS_L{N}')] = np.nan
                features[clean_feature_name(f'{col_prefix}_AvgGC_L{N}')] = np.nan
                features[clean_feature_name(f'{col_prefix}_Form_Points_L{N}')] = np.nan # Match naming from preprocess_data.py
                continue

            gs, gc, pts = 0, 0, 0
            for _, row in last_n_matches.iterrows():
                if row['HomeTeam'] == team_name:
                    gs += row['FTHG']
                    gc += row['FTAG']
                    if row['FTHG'] > row['FTAG']: pts += 3
                    elif row['FTHG'] == row['FTAG']: pts += 1
                else:
                    gs += row['FTAG']
                    gc += row['FTHG']
                    if row['FTAG'] > row['FTHG']: pts += 3
                    elif row['FTAG'] == row['FTHG']: pts += 1
            
            features[clean_feature_name(f'{col_prefix}_AvgGS_L{N}')] = gs / N
            features[clean_feature_name(f'{col_prefix}_AvgGC_L{N}')] = gc / N
            features[clean_feature_name(f'{col_prefix}_Form_Points_L{N}')] = pts # Match naming
    return features

# --- Prepare Data for a Single Future Match ---
def prepare_future_match_data(future_match_row, label_encoders, model_feature_names, historical_data_df):
    processed_data = {}
    
    # 1. Categorical Encoding (HomeTeam, AwayTeam, Referee)
    for col_to_encode in ['HomeTeam', 'AwayTeam', 'Referee']:
        original_val = future_match_row.get(col_to_encode)
        encoded_col_name = clean_feature_name(col_to_encode + '_encoded')
        
        if original_val is not None and col_to_encode in label_encoders:
            le = label_encoders[col_to_encode]
            if original_val not in le.classes_:
                print(f"Warning: Value '{original_val}' for '{col_to_encode}' not in training encoder. Using -1.")
                processed_data[encoded_col_name] = -1
            else:
                processed_data[encoded_col_name] = le.transform([original_val])[0]
        else:
            print(f"Warning: Missing '{col_to_encode}' or its encoder. Using -1 for {encoded_col_name}.")
            processed_data[encoded_col_name] = -1

    # 2. Numerical Features from future_match_row (e.g., Odds)
    for orig_col_name, value in future_match_row.items():
        if orig_col_name not in ['HomeTeam', 'AwayTeam', 'Referee', 'Date', 'FTR', 'FTHG', 'FTAG', 'Time', 'Div'] and not pd.isna(value):
            cleaned_key = clean_feature_name(orig_col_name)
            processed_data[cleaned_key] = value

    # 3. Historical Features
    home_team = future_match_row.get('HomeTeam')
    away_team = future_match_row.get('AwayTeam')
    match_date_str = future_match_row.get('Date')

    if home_team and away_team and match_date_str:
        match_date = pd.to_datetime(match_date_str, dayfirst=True, errors='coerce') # Assuming dd/mm/yyyy
        if pd.isna(match_date):
            print(f"Error: Could not parse match date '{match_date_str}' for historical features.")
            hist_features = None
        else:
            hist_features = get_historical_features(home_team, away_team, match_date, historical_data_df)
        
        if hist_features:
            processed_data.update(hist_features) # hist_features keys are already cleaned
        else: # Fill with NaN if hist_features couldn't be calculated
            for h_feat_template in ['{}_AvgGS_L{}', '{}_AvgGC_L{}', '{}_Form_Points_L{}']:
                for team_prefix in ['HomeTeam', 'AwayTeam']:
                    for N in [3,5,10]:
                        processed_data[clean_feature_name(h_feat_template.format(team_prefix, N))] = np.nan
    else:
        print("Warning: HomeTeam, AwayTeam, or Date missing in future_match_row. Cannot calculate historical features.")
        # Fill all historical features with NaN
        for h_feat_template in ['{}_AvgGS_L{}', '{}_AvgGC_L{}', '{}_Form_Points_L{}']:
            for team_prefix in ['HomeTeam', 'AwayTeam']:
                for N in [3,5,10]:
                    processed_data[clean_feature_name(h_feat_template.format(team_prefix, N))] = np.nan

    # 4. Create DataFrame and align with model_feature_names
    final_data_row = pd.DataFrame([processed_data])
    final_data_row = final_data_row.reindex(columns=model_feature_names, fill_value=np.nan)
    
    # 5. Imputation (simple fill with 0 for now)
    if final_data_row.isnull().any().any():
        final_data_row = final_data_row.fillna(0) 
        
    return final_data_row

# --- Main Prediction Logic ---
def make_prediction(data_row, model):
    try:
        prediction_encoded = model.predict(data_row)
        prediction_proba = model.predict_proba(data_row)
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        return outcome_map.get(prediction_encoded[0], "Unknown"), prediction_proba[0]
    except Exception as e:
        print(f"Error during XGBoost prediction: {e}")
        print(f"Data row causing error (first 5 features): {data_row.iloc[:, :5].to_dict()}")
        return f"Prediction Error: {e}", None

# --- Fungsi untuk predict_history_matchup yang akan dipanggil dari predict.py ---
def predict_history_matchup(home_team, away_team):
    """
    Memprediksi hasil pertandingan berdasarkan riwayat pertandingan sebelumnya
    
    Args:
        home_team (str): Nama tim tuan rumah
        away_team (str): Nama tim tandang
    
    Returns:
        dict: Hasil prediksi berisi predicted_outcome dan probabilities
    """
    model, label_encoders, model_feature_names = load_dependencies()
    
    if not all([model, label_encoders, model_feature_names]):
        return {"error": "Could not load model, encoders, or feature names"}
    
    try:
        hist_data_path = os.path.join(DATA_DIR, "combined_epl_data.csv")
        historical_df = pd.read_csv(hist_data_path, low_memory=False)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True, errors='coerce')
        historical_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], inplace=True)
        
        future_matches_path = os.path.join(FUTURE_DATA_DIR, "EPL 2024-2025.csv")
        future_df = pd.read_csv(future_matches_path, low_memory=False)
        future_df['HomeTeam'] = future_df['HomeTeam'].astype(str)
        future_df['AwayTeam'] = future_df['AwayTeam'].astype(str)
        future_df['Date_dt'] = pd.to_datetime(future_df['Date'], dayfirst=True, errors='coerce')
    except Exception as e:
        return {"error": f"Error loading data: {str(e)}"}
    
    # Find the match in future_df
    target_matches = future_df[
        future_df['HomeTeam'].str.contains(home_team, case=False, na=False) &
        future_df['AwayTeam'].str.contains(away_team, case=False, na=False)
    ]
    
    if target_matches.empty:
        return {"error": f"No match found for {home_team} vs {away_team}"}
    
    # Use the first match found
    future_match_details_row = target_matches.iloc[0]
    
    # Prepare data for prediction
    prepared_data_row = prepare_future_match_data(future_match_details_row, label_encoders, model_feature_names, historical_df)
    
    if prepared_data_row is None or prepared_data_row.empty:
        return {"error": "Could not prepare data for this match"}
    
    # Make prediction
    predicted_outcome, probabilities = make_prediction(prepared_data_row, model)
    
    if isinstance(predicted_outcome, str) and predicted_outcome.startswith("Prediction Error"):
        return {"error": predicted_outcome}
    
    result = {
        "predicted_outcome": predicted_outcome,
        "home_win_prob": float(probabilities[2]) if probabilities is not None else 0,
        "draw_prob": float(probabilities[1]) if probabilities is not None else 0,
        "away_win_prob": float(probabilities[0]) if probabilities is not None else 0,
        "match_date": future_match_details_row.get('Date', 'N/A')
    }
    
    # Add actual result if available
    actual_ftr = future_match_details_row.get('FTR', None)
    actual_fthg = future_match_details_row.get('FTHG', None)
    actual_ftag = future_match_details_row.get('FTAG', None)
    
    if pd.notna(actual_ftr) and pd.notna(actual_fthg) and pd.notna(actual_ftag):
        actual_outcome_map = {'H': "Home Win", 'D': "Draw", 'A': "Away Win"}
        result["actual_outcome"] = actual_outcome_map.get(actual_ftr, "Unknown")
        result["actual_score"] = f"{int(actual_fthg)}-{int(actual_ftag)}"
    
    return result 