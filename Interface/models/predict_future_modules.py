import pandas as pd
import xgboost as xgb
import joblib
import json
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

MODEL_DIR = "models"
DATA_DIR = "Combined Dataset"

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

# --- Load Model and Encoders ---
def load_model_and_encoders():
    """Loads the trained XGBoost model and label encoders."""
    model_path = os.path.join(MODEL_DIR, "xgboost_epl_model.json")
    encoders_path = os.path.join(DATA_DIR, "label_encoders.joblib")

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None, None
    if not os.path.exists(encoders_path):
        print(f"Error: Label encoders file not found at {encoders_path}")
        return None, None

    model = xgb.XGBClassifier()
    model.load_model(model_path)
    label_encoders = joblib.load(encoders_path)
    
    print("Model and label encoders loaded successfully.")
    return model, label_encoders

# --- Feature Name Cleaning (consistent with preprocess_data.py) ---
def clean_feature_name(col_name):
    """Cleans a column name to be XGBoost compatible. THIS MUST MATCH preprocess_data.py."""
    import re
    new_name = str(col_name).replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
    new_name = new_name.replace('[', '_').replace(']', '_').replace(' ', '_')
    new_name = "".join(char for char in new_name if char.isalnum() or char == '_')
    new_name = new_name.strip('_') # Remove leading/trailing underscores
    if not new_name: new_name = 'unnamed_col_fallback'
    return new_name

# --- Function to calculate historical features for a new match ---
def get_historical_features(home_team_name, away_team_name, date_of_match, historical_data_df):
    """Calculates historical features for the home and away teams up to the match date."""
    features = {}
    teams = {'Home': home_team_name, 'Away': away_team_name}
    windows = [3, 5, 10]

    # Ensure Date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(historical_data_df['Date']):
        historical_data_df['Date'] = pd.to_datetime(historical_data_df['Date'], errors='coerce')

    for team_type, team_name in teams.items():
        # Filter matches for the current team before the match_date
        team_matches = historical_data_df[
            ((historical_data_df['HomeTeam'] == team_name) | (historical_data_df['AwayTeam'] == team_name)) &
            (historical_data_df['Date'] < date_of_match)
        ].sort_values(by='Date', ascending=False)

        for N in windows:
            last_n_matches = team_matches.head(N)
            if len(last_n_matches) < N:
                # Not enough historical matches, fill with a default (e.g., mean or 0)
                features[f'{team_type}_AvgGS_L{N}'] = np.nan
                features[f'{team_type}_AvgGC_L{N}'] = np.nan
                features[f'{team_type}_FormPts_L{N}'] = np.nan
                continue

            goals_scored = 0
            goals_conceded = 0
            form_points = 0

            for _, row in last_n_matches.iterrows():
                if row['HomeTeam'] == team_name:
                    goals_scored += row['FTHG']
                    goals_conceded += row['FTAG']
                    if row['FTHG'] > row['FTAG']: form_points += 3
                    elif row['FTHG'] == row['FTAG']: form_points += 1
                else: # Away team
                    goals_scored += row['FTAG']
                    goals_conceded += row['FTHG']
                    if row['FTAG'] > row['FTHG']: form_points += 3
                    elif row['FTAG'] == row['FTHG']: form_points += 1
            
            features[f'{team_type}_AvgGS_L{N}'] = goals_scored / N if N > 0 else 0
            features[f'{team_type}_AvgGC_L{N}'] = goals_conceded / N if N > 0 else 0
            features[f'{team_type}_FormPts_L{N}'] = form_points
            
    return features

# --- Prepare a single match data for prediction ---
def prepare_match_data(match_input, label_encoders, all_features_columns, historical_data_df):
    """Prepares a single match data row for prediction, including historical features."""
    processed_data = {}

    # Basic categorical encoding
    for col, le in label_encoders.items():
        original_col_name = col # e.g., 'HomeTeam'
        encoded_col_name = clean_feature_name(col + '_encoded') # e.g., 'HomeTeam_encoded'
        if original_col_name in match_input:
            team_name = match_input[original_col_name]
            try:
                if team_name not in le.classes_:
                    print(f"Warning: Team '{team_name}' in column '{original_col_name}' not seen during training. Assigning -1.")
                    processed_data[encoded_col_name] = -1
                else:
                    processed_data[encoded_col_name] = le.transform([team_name])[0]
            except ValueError:
                print(f"Error encoding '{team_name}' for column '{original_col_name}'. It might be a new team/referee.")
                processed_data[encoded_col_name] = -1 
        else:
            print(f"Warning: Column '{original_col_name}' not found in match_input for encoding.")
            processed_data[encoded_col_name] = -1

    # Add other numerical features directly from input (ensure names are cleaned)
    for key, value in match_input.items():
        if key not in ['HomeTeam', 'AwayTeam', 'Referee', 'Date']:
            cleaned_key = clean_feature_name(key)
            processed_data[cleaned_key] = value

    # Calculate and add historical features
    if 'HomeTeam' in match_input and 'AwayTeam' in match_input and 'Date' in match_input:
        match_date = pd.to_datetime(match_input['Date'])
        hist_features = get_historical_features(match_input['HomeTeam'], match_input['AwayTeam'], match_date, historical_data_df)
        for key, value in hist_features.items():
            cleaned_key = clean_feature_name(key)
            processed_data[cleaned_key] = value
    else:
        print("Warning: HomeTeam, AwayTeam, or Date missing, cannot calculate historical features.")
        hist_feat_names = [col for col in all_features_columns if any(suff in col for suff in ['AvgGS','AvgGC','FormPts'])]
        for h_feat in hist_feat_names:
            processed_data[h_feat] = np.nan
            
    # Create DataFrame and ensure all model features are present
    final_data_row = pd.DataFrame([processed_data])
    
    # Reindex to ensure column order and presence, fill missing with NaN
    final_data_row = final_data_row.reindex(columns=all_features_columns, fill_value=np.nan)

    # Basic Imputation for any NaNs
    final_data_row = final_data_row.fillna(0)
    
    return final_data_row

# --- Main Prediction Function ---
def predict_single_match(match_input, model, label_encoders, train_columns, historical_data_df):
    """Makes a prediction for a single match."""
    processed_row = prepare_match_data(match_input, label_encoders, train_columns, historical_data_df)
    
    if processed_row is None:
        return "Error in preparing match data.", None
    if processed_row.isnull().any().any():
        print("Warning: Null values detected in the processed row before prediction:")
        print(processed_row[processed_row.isnull().any(axis=1)])

    try:
        prediction_encoded = model.predict(processed_row)
        prediction_proba = model.predict_proba(processed_row)
        
        # Assuming encoding: 0 for Away Win, 1 for Draw, 2 for Home Win
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        predicted_outcome = outcome_map.get(prediction_encoded[0], "Unknown Outcome")
        
        return predicted_outcome, prediction_proba[0]
    except Exception as e:
        return f"Error during prediction: {e}", None

# Fungsi untuk mendapatkan data historis
def get_historical_data():
    try:
        historical_data_path = os.path.join(DATA_DIR, "combined_epl_data.csv")
        historical_df = pd.read_csv(historical_data_path, low_memory=False)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True, errors='coerce')
        historical_df.dropna(subset=['Date'], inplace=True)
        historical_df = historical_df[historical_df['FTHG'].notna()]
        return historical_df
    except Exception as e:
        print(f"Error loading historical data: {str(e)}")
        return None

# Fungsi untuk mendapatkan nama fitur model
def get_model_feature_names():
    try:
        x_train_path = os.path.join(DATA_DIR, "X_train.csv")
        MODEL_FEATURE_NAMES = pd.read_csv(x_train_path, nrows=0).columns.tolist()
        return MODEL_FEATURE_NAMES
    except Exception as e:
        print(f"Error loading model feature names: {str(e)}")
        return None

# Jika file dijalankan langsung (bukan diimpor)
if __name__ == "__main__":
    print("Modul predict_future_modules.py dijalankan langsung.")
    print("Gunakan fungsi predict_single_match() untuk memprediksi pertandingan.")