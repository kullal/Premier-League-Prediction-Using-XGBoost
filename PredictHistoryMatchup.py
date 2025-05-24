import pandas as pd
import xgboost as xgb
import joblib
import json
import os
import numpy as np
from datetime import datetime

MODEL_DIR = "models"
DATA_DIR = "Combined Dataset"
FUTURE_DATA_DIR = "Dataset EPL New" # Directory for 2024-2025 data

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
    # Assume future_match_row is a Series or dict; its keys are original column names
    for orig_col_name, value in future_match_row.items():
        if orig_col_name not in ['HomeTeam', 'AwayTeam', 'Referee', 'Date', 'FTR', 'FTHG', 'FTAG', 'Time', 'Div'] and not pd.isna(value):
             # Exclude IDs, results, and already processed. Ensure value is not NaN.
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
    
    # 5. Imputation (simple fill with 0 for now, consistent with predict_match.py)
    # Check if any columns expected by the model are entirely missing and were not filled by reindex (should not happen if reindex works)
    # For NaNs that exist (e.g. missing odds for a specific bookie, or failed hist data), fill with 0.
    if final_data_row.isnull().any().any():
        # print("NaNs found before final fill, filling with 0:")
        # print(final_data_row.isnull().sum()[final_data_row.isnull().sum() > 0])
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
        # Check for non-numeric data if that's a common issue
        # for col in data_row.columns:
        #     if not pd.api.types.is_numeric_dtype(data_row[col]):
        #         print(f"Non-numeric column: {col}, type: {data_row[col].dtype}, value: {data_row[col].iloc[0]}")
        return f"Prediction Error: {e}", None

# --- Main Execution ---
if __name__ == "__main__":
    model, label_encoders, model_feature_names = load_dependencies()

    if not all([model, label_encoders, model_feature_names]):
        exit()

    # Load historical data (combined_epl_data.csv)
    try:
        hist_data_path = os.path.join(DATA_DIR, "combined_epl_data.csv")
        historical_df = pd.read_csv(hist_data_path, low_memory=False)
        historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True, errors='coerce') # Must match preprocess
        historical_df.dropna(subset=['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'], inplace=True) # Ensure key columns are good
    except Exception as e:
        print(f"Error loading historical_df from {hist_data_path}: {e}")
        exit()

    # Load future matches data (EPL 2024-2025.csv)
    try:
        future_matches_path = os.path.join(FUTURE_DATA_DIR, "EPL 2024-2025.csv")
        future_df = pd.read_csv(future_matches_path, low_memory=False)
        # Basic cleaning for future_df - ensure team names are strings
        future_df['HomeTeam'] = future_df['HomeTeam'].astype(str)
        future_df['AwayTeam'] = future_df['AwayTeam'].astype(str)
        if 'Date' in future_df.columns:
             future_df['Date_dt'] = pd.to_datetime(future_df['Date'], dayfirst=True, errors='coerce') # Keep original 'Date' string too
        else:
            print("Error: 'Date' column missing in future matches CSV.")
            exit()

    except Exception as e:
        print(f"Error loading future_matches_df from {future_matches_path}: {e}")
        exit()

    # Get a unique, sorted list of all teams available in the future dataset
    all_teams = sorted(list(pd.unique(future_df[['HomeTeam', 'AwayTeam']].values.ravel('K'))))

    print("\n--- Predict Future Matchup ---")
    print("Available teams:")
    for i, team_name in enumerate(all_teams):
        print(f"{i + 1}. {team_name}")
    print("0. Exit")

    home_team_idx_str = input(f"Enter number for Home Team (1-{len(all_teams)}, or 0 to exit): ").strip()
    if not home_team_idx_str.isdigit():
        print("Invalid input. Please enter a number.")
        # exit() # Exit if input is not a digit
        # Consider what to do here: exit or allow re-try. For one-shot, exiting is fine.
        # Let's make it exit cleanly.
        print("Exiting prediction script.")
        exit() 
    
    home_team_idx = int(home_team_idx_str)

    if home_team_idx == 0:
        # break # This was for the loop, now we just exit
        print("Exiting prediction script.")
        exit()

    if not (1 <= home_team_idx <= len(all_teams)):
        print("Invalid team number. Please try again.")
        # continue # This was for the loop
        print("Exiting prediction script.") # Exit if invalid
        exit()
    
    input_home_team = all_teams[home_team_idx - 1]

    away_team_idx_str = input(f"Enter number for Away Team (1-{len(all_teams)}, not {input_home_team}): ").strip()
    if not away_team_idx_str.isdigit():
        print("Invalid input. Please enter a number.")
        # continue
        print("Exiting prediction script.") # Exit if invalid
        exit()

    away_team_idx = int(away_team_idx_str)

    if not (1 <= away_team_idx <= len(all_teams)):
        print("Invalid team number. Please try again.")
        # continue
        print("Exiting prediction script.") # Exit if invalid
        exit()
    if away_team_idx == home_team_idx:
        print("Away team cannot be the same as home team. Please try again.")
        # continue
        print("Exiting prediction script.") # Exit if invalid
        exit()
        
    input_away_team = all_teams[away_team_idx - 1]
    
    print(f"\nSelected Matchup: {input_home_team} (Home) vs {input_away_team} (Away)")

    # Find the match(es) in future_df
    # Case-insensitive search is safer for team names if CSV has inconsistencies
    target_matches = future_df[
        future_df['HomeTeam'].str.contains(input_home_team, case=False, na=False) &
        future_df['AwayTeam'].str.contains(input_away_team, case=False, na=False)
    ]

    if target_matches.empty:
        print(f"No match found for {input_home_team} vs {input_away_team} in {future_matches_path}")
        # Try reversing if user might have mixed them up
        target_matches_reversed = future_df[
            future_df['HomeTeam'].str.contains(input_away_team, case=False, na=False) &
            future_df['AwayTeam'].str.contains(input_home_team, case=False, na=False)
        ]
        if not target_matches_reversed.empty:
            print(f"Found match(es) if teams are reversed: {input_away_team} (H) vs {input_home_team} (A). Please re-enter if this was intended.")
        print("\nExiting prediction script.") # Moved this to be the final print
        exit()

    print(f"Found {len(target_matches)} match(es) for {input_home_team} vs {input_away_team}:")
    
    for idx, future_match_details_row in target_matches.iterrows():
        print(f"Processing Match (Original Index: {idx}):")
        print(f"  Date: {future_match_details_row.get('Date', 'N/A')}, Home: {future_match_details_row.get('HomeTeam')}, Away: {future_match_details_row.get('AwayTeam')}")

        # Prepare data for this specific match
        # future_match_details_row is a Series
        prepared_data_row = prepare_future_match_data(future_match_details_row, label_encoders, model_feature_names, historical_df)
        
        if prepared_data_row is None or prepared_data_row.empty:
            print("  Could not prepare data for this match. Skipping.")
            continue
        
        if prepared_data_row.isnull().any().any():
            print("  Warning: Prepared data row contains NaNs even after fillna(0). This is unexpected.")
            print(prepared_data_row.isnull().sum()[prepared_data_row.isnull().sum() > 0])
            # Decide: skip, or try to predict anyway if XGBoost handles it (it might if only a few non-critical)

        # Make prediction
        predicted_outcome, probabilities = make_prediction(prepared_data_row, model)

        print(f"  Predicted Outcome: {predicted_outcome}")
        if probabilities is not None:
            prob_dict = {"Away Win": probabilities[0], "Draw": probabilities[1], "Home Win": probabilities[2]}
            print(f"  Probabilities: Away Win: {prob_dict['Away Win']:.3f}, Draw: {prob_dict['Draw']:.3f}, Home Win: {prob_dict['Home Win']:.3f}")

        # Cross-check with actual result if available in the 2024-2025 CSV
        actual_ftr = future_match_details_row.get('FTR', None) # Full Time Result (H, D, A)
        actual_fthg = future_match_details_row.get('FTHG', None)
        actual_ftag = future_match_details_row.get('FTAG', None)

        if pd.notna(actual_ftr) and pd.notna(actual_fthg) and pd.notna(actual_ftag):
            actual_outcome_map = {'H': "Home Win", 'D': "Draw", 'A': "Away Win"}
            actual_result_str = actual_outcome_map.get(actual_ftr, "Unknown Actual Result Code")
            print(f"  Actual Result (from CSV): {actual_result_str} ({actual_fthg}-{actual_ftag})")
            if predicted_outcome == actual_result_str:
                print("  Prediction was CORRECT!")
            else:
                print("  Prediction was INCORRECT.")
        else:
            print("  Actual result not yet available in the CSV or columns missing (FTR, FTHG, FTAG).")
    print("\nExiting prediction script.") # Moved this to be the final print 