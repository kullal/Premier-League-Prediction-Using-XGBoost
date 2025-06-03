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

# Pindahkan fungsi ini ke bagian atas file, setelah import
def display_result_chart(home_win, draw, away_win):
    # Konversi probabilitas ke persentase
    total = home_win + draw + away_win
    home_win_pct = int(home_win * 100 / total)
    draw_pct = int(draw * 100 / total)
    away_win_pct = int(away_win * 100 / total)
    
    # Buat figure
    fig, ax = plt.subplots(figsize=(6, 2))
    
    # Buat bar horizontal berdasarkan persentasi
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
# This is a simplified version. A more robust version would query combined_epl_data.csv
def get_historical_features(home_team_name, away_team_name, date_of_match, historical_data_df):
    """Calculates historical features for the home and away teams up to the match date.
    Args:
        home_team_name (str): Name of the home team.
        away_team_name (str): Name of the away team.
        date_of_match (pd.Timestamp): The date of the match to predict.
        historical_data_df (pd.DataFrame): DataFrame containing all historical matches with 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG'.
    Returns:
        dict: Dictionary containing historical features for home and away teams.
    """
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
                # Or, you could decide not to predict if history is insufficient
                features[f'{team_type}_AvgGS_L{N}'] = np.nan # Or some default like 1.0
                features[f'{team_type}_AvgGC_L{N}'] = np.nan # Or some default like 1.0
                features[f'{team_type}_FormPts_L{N}'] = np.nan # Or some default like N (e.g., 1 point per match average)
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
    """Prepares a single match data row for prediction, including historical features.
    Args:
        match_input (dict): Dictionary with raw match data (e.g., HomeTeam, AwayTeam, Referee, odds, etc.)
                            It must also include 'Date' for historical feature calculation.
        label_encoders (dict): Loaded label encoders.
        all_features_columns (list): List of all feature columns expected by the model (cleaned names).
        historical_data_df (pd.DataFrame): DataFrame of all historical matches for feature calculation.
    Returns:
        pd.DataFrame: A single-row DataFrame ready for prediction, or None if error.
    """
    processed_data = {}

    # Basic categorical encoding
    for col, le in label_encoders.items():
        original_col_name = col # e.g., 'HomeTeam'
        encoded_col_name = clean_feature_name(col + '_encoded') # e.g., 'HomeTeam_encoded'
        if original_col_name in match_input:
            team_name = match_input[original_col_name]
            try:
                # Handle unseen labels by assigning a default (e.g., -1 or a specific 'unknown' category if trained)
                if team_name not in le.classes_:
                    print(f"Warning: Team '{team_name}' in column '{original_col_name}' not seen during training. Assigning -1.")
                    # Option 1: Assign a specific value like -1 or len(le.classes_)
                    # This requires the model to have potentially seen this during robust CV
                    # For now, let's try to skip or raise error if critical like HomeTeam/AwayTeam
                    # return None # Or handle as per strategy
                    processed_data[encoded_col_name] = -1 # Or another placeholder
                else:
                    processed_data[encoded_col_name] = le.transform([team_name])[0]
            except ValueError:
                print(f"Error encoding '{team_name}' for column '{original_col_name}'. It might be a new team/referee.")
                # Decide how to handle: skip prediction, use a default, etc.
                # For now, setting to a placeholder like -1, but this needs careful consideration
                processed_data[encoded_col_name] = -1 
        else:
            print(f"Warning: Column '{original_col_name}' not found in match_input for encoding.")
            processed_data[encoded_col_name] = -1 # Or np.nan, then impute

    # Add other numerical features directly from input (ensure names are cleaned)
    # Example: match_input could have {'B365H': 2.5, 'B365D': 3.0, ...}
    for key, value in match_input.items():
        if key not in ['HomeTeam', 'AwayTeam', 'Referee', 'Date']: # Avoid re-processing already handled ones
            cleaned_key = clean_feature_name(key)
            processed_data[cleaned_key] = value

    # Calculate and add historical features
    if 'HomeTeam' in match_input and 'AwayTeam' in match_input and 'Date' in match_input:
        match_date = pd.to_datetime(match_input['Date'])
        hist_features = get_historical_features(match_input['HomeTeam'], match_input['AwayTeam'], match_date, historical_data_df)
        for key, value in hist_features.items():
            cleaned_key = clean_feature_name(key) # Ensure these are cleaned too
            processed_data[cleaned_key] = value
    else:
        print("Warning: HomeTeam, AwayTeam, or Date missing, cannot calculate historical features.")
        # Need to add placeholders for all historical feature columns if not calculated
        hist_feat_names = [col for col in all_features_columns if any(suff in col for suff in ['AvgGS','AvgGC','FormPts'])]
        for h_feat in hist_feat_names:
            processed_data[h_feat] = np.nan # Or a default value
            
    # Create DataFrame and ensure all model features are present
    # The `all_features_columns` should be derived from the X_train.columns used for training the saved model
    final_data_row = pd.DataFrame([processed_data])
    
    # Reindex to ensure column order and presence, fill missing with NaN (or a specific strategy)
    # This is crucial: the order and number of columns must match the training data
    final_data_row = final_data_row.reindex(columns=all_features_columns, fill_value=np.nan)

    # Basic Imputation for any NaNs that might have occurred (e.g. from missing hist data)
    # A more sophisticated imputation (like mean/median from X_train) could be used if necessary
    final_data_row = final_data_row.fillna(0) # Simple: fill with 0, adjust as needed
    
    return final_data_row

# --- Main Prediction Function ---
def predict_single_match(match_input, model, label_encoders, train_columns, historical_data_df):
    """Makes a prediction for a single match.
    Args:
        match_input (dict): Raw input data for the match. Must include 'Date'.
        model: Trained XGBoost model.
        label_encoders: Loaded label encoders.
        train_columns (list): Column names from the training data (X_train.columns), cleaned.
        historical_data_df (pd.DataFrame): DataFrame of all historical matches.
    Returns:
        str: Predicted outcome string (e.g., "Home Win", "Draw", "Away Win") or error message.
    """
    processed_row = prepare_match_data(match_input, label_encoders, train_columns, historical_data_df)
    
    if processed_row is None:
        return "Error in preparing match data."
    if processed_row.isnull().any().any():
        print("Warning: Null values detected in the processed row before prediction:")
        print(processed_row[processed_row.isnull().any(axis=1)])
        # Fallback or error, e.g., fill with 0 or mean from training, or return error
        # For now, let's assume fillna(0) in prepare_match_data handles it.
        # If not, an error in XGBoost might occur.

    try:
        prediction_encoded = model.predict(processed_row)
        prediction_proba = model.predict_proba(processed_row)
        
        # Assuming encoding: 0 for Away Win, 1 for Draw, 2 for Home Win (consistent with preprocess_data.py)
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        predicted_outcome = outcome_map.get(prediction_encoded[0], "Unknown Outcome")
        
        return predicted_outcome, prediction_proba[0]
    except Exception as e:
        return f"Error during prediction: {e}", None

# --- Example Usage --- 
if __name__ == "__main__":
    model, label_encoders = load_model_and_encoders()

    if model and label_encoders:
        try:
            historical_data_path = os.path.join(DATA_DIR, "combined_epl_data.csv")
            if not os.path.exists(historical_data_path):
                raise FileNotFoundError(f"Historical data file not found at {historical_data_path}")
            historical_df = pd.read_csv(historical_data_path, low_memory=False) 
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True, errors='coerce')
            historical_df.dropna(subset=['Date'], inplace=True)
            historical_df = historical_df[historical_df['FTHG'].notna()]
        except FileNotFoundError as e:
            print(e)
            print("Please ensure 'combined_epl_data.csv' is in the 'Combined Dataset' directory.")
            exit()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            exit()
            
        try:
            x_train_path = os.path.join(DATA_DIR, "X_train.csv")
            if not os.path.exists(x_train_path):
                 raise FileNotFoundError(f"X_train.csv not found at {x_train_path}. Needed for feature names.")
            MODEL_FEATURE_NAMES = pd.read_csv(x_train_path, nrows=0).columns.tolist()
            print(f"Model expects {len(MODEL_FEATURE_NAMES)} features. First 5: {MODEL_FEATURE_NAMES[:5]}")
        except FileNotFoundError as e:
            print(e)
            print("Please run preprocess_data.py to generate X_train.csv.")
            exit()
        except Exception as e:
            print(f"Error loading X_train column names: {e}")
            exit()

        all_teams = sorted(list(pd.unique(historical_df[['HomeTeam', 'AwayTeam']].values.ravel('K'))))
        all_referees = sorted(list(historical_df['Referee'].dropna().unique()))

        date_str = None
        home_team = None
        away_team = None
        referee = None
        b365h, b365d, b365a = 2.0, 3.0, 4.0 # Default odds

        if len(sys.argv) >= 5: # script_name, date, home, away, referee, [optional_odds_H D A]
            try:
                cli_date_input = sys.argv[1]
                cli_match_date = pd.to_datetime(cli_date_input, dayfirst=True)
                if pd.isna(cli_match_date):
                    print(f"Invalid date format from CLI: {cli_date_input}. Use DD/MM/YYYY. Falling back to interactive.")
                else:
                    date_str = cli_match_date.strftime('%Y-%m-%d')
                    
                    cli_home_team = sys.argv[2]
                    cli_away_team = sys.argv[3]
                    cli_referee = sys.argv[4]

                    if cli_home_team in all_teams and cli_away_team in all_teams and cli_referee in all_referees:
                        if cli_home_team != cli_away_team:
                            home_team = cli_home_team
                            away_team = cli_away_team
                            referee = cli_referee
                            print(f"Using CLI arguments - Date: {date_str}, Home: {home_team}, Away: {away_team}, Referee: {referee}")
                            # Optionally parse odds if provided
                            if len(sys.argv) == 8:
                                try:
                                    b365h = float(sys.argv[5])
                                    b365d = float(sys.argv[6])
                                    b365a = float(sys.argv[7])
                                    print(f"Using CLI odds: H={b365h}, D={b365d}, A={b365a}")
                                except ValueError:
                                    print("Invalid odds from CLI. Using default odds.")
                            elif len(sys.argv) > 5 and len(sys.argv) != 8:
                                print("Incorrect number of odds provided via CLI (expected 3: H D A). Using default odds.")
                        else:
                            print("CLI Error: Home and Away teams cannot be the same. Fallback to interactive.")
                            date_str = None # Reset to trigger interactive
                    else:
                        print("CLI Error: Invalid team or referee name. Fallback to interactive.")
                        if cli_home_team not in all_teams: print(f"Invalid Home Team: {cli_home_team}")
                        if cli_away_team not in all_teams: print(f"Invalid Away Team: {cli_away_team}")
                        if cli_referee not in all_referees: print(f"Invalid Referee: {cli_referee}")
                        date_str = None # Reset to trigger interactive
            except Exception as e:
                print(f"Error processing CLI arguments: {e}. Falling back to interactive input.")
                date_str = None # Ensure fallback

        if not all([date_str, home_team, away_team, referee]):
            print("\n--- Predict Future Match (Interactive) ---")
            while date_str is None:
                date_input = input("Enter match date (DD/MM/YYYY): ").strip()
                try:
                    match_date = pd.to_datetime(date_input, dayfirst=True)
                    if pd.isna(match_date):
                        print("Invalid date format. Please use DD/MM/YYYY.")
                        continue
                    date_str = match_date.strftime('%Y-%m-%d')
                except Exception:
                    print("Invalid date format. Please use DD/MM/YYYY.")
            
            print("\nSelect Home Team:")
            for i, team in enumerate(all_teams): print(f"{i+1}. {team}")
            while home_team is None:
                try:
                    home_team_idx = int(input("\nEnter number for Home Team: ").strip())
                    if 1 <= home_team_idx <= len(all_teams):
                        home_team = all_teams[home_team_idx-1]
                    else:
                        print(f"Please enter a number between 1 and {len(all_teams)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print("\nSelect Away Team:")
            for i, team in enumerate(all_teams): print(f"{i+1}. {team}")
            while away_team is None:
                try:
                    away_team_idx = int(input("\nEnter number for Away Team: ").strip())
                    if 1 <= away_team_idx <= len(all_teams):
                        if all_teams[away_team_idx-1] == home_team:
                            print("Away team cannot be the same as home team.")
                            continue
                        away_team = all_teams[away_team_idx-1]
                    else:
                        print(f"Please enter a number between 1 and {len(all_teams)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print("\nSelect Referee:")
            for i, ref in enumerate(all_referees): print(f"{i+1}. {ref}")
            while referee is None:
                try:
                    ref_idx = int(input("\nEnter number for Referee: ").strip())
                    if 1 <= ref_idx <= len(all_referees):
                        referee = all_referees[ref_idx-1]
                    else:
                        print(f"Please enter a number between 1 and {len(all_referees)}.")
                except ValueError:
                    print("Please enter a valid number.")
            
            print("\nEnter betting odds (decimal format, e.g., 2.50). Press Enter to use defaults (2.0, 3.0, 4.0):")
            try:
                b365h_in = input(f"Bet365 Home win odds (default {b365h}): ").strip()
                b365d_in = input(f"Bet365 Draw odds (default {b365d}): ").strip()
                b365a_in = input(f"Bet365 Away win odds (default {b365a}): ").strip()
                if b365h_in: b365h = float(b365h_in)
                if b365d_in: b365d = float(b365d_in)
                if b365a_in: b365a = float(b365a_in)
            except ValueError:
                print("Invalid odds format. Using default values.")
                # Defaults are already set

        # Create match data dictionary
        new_match_data = {
            'Date': date_str,
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Referee': referee,
            # Betting Odds
            'B365H': b365h, 
            'B365D': b365d, 
            'B365A': b365a,
            # Use the same odds for other bookmakers if not provided
            'BSH': b365h, 'BSD': b365d, 'BSA': b365a,
            'BWH': b365h, 'BWD': b365d, 'BWA': b365a,
            'PSH': b365h, 'PSD': b365d, 'PSA': b365a,
            'MaxH': b365h, 'MaxD': b365d, 'MaxA': b365a,
            'AvgH': b365h, 'AvgD': b365d, 'AvgA': b365a,
        }

        print(f"\nPredicting for match: {new_match_data['HomeTeam']} vs {new_match_data['AwayTeam']} on {new_match_data['Date']}")
        print(f"Referee: {new_match_data['Referee']}")
        print(f"Odds (Home/Draw/Away): {b365h:.2f} / {b365d:.2f} / {b365a:.2f}")
        
        predicted_result, probabilities = predict_single_match(new_match_data, model, label_encoders, MODEL_FEATURE_NAMES, historical_df)
        
        print(f"\nPredicted Outcome: {predicted_result}")
        if probabilities is not None:
            print(f"Probabilities:")
            print(f"  Home Win: {probabilities[2]:.3f}")
            print(f"  Draw: {probabilities[1]:.3f}")
            print(f"  Away Win: {probabilities[0]:.3f}")
            
            # Tambahkan kode ini untuk menampilkan chart
            display_result_chart(probabilities[2], probabilities[1], probabilities[0])

    else:
        print("Could not load model or encoders. Prediction cannot proceed.") 