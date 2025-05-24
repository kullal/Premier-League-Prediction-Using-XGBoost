import pandas as pd
import xgboost as xgb
import joblib
import json
import os
import numpy as np

MODEL_DIR = "models"
DATA_DIR = "Combined Dataset"

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
        # 1. Load historical data for feature calculation
        # This should be your combined_epl_data.csv or similar
        try:
            historical_data_path = os.path.join(DATA_DIR, "combined_epl_data.csv")
            if not os.path.exists(historical_data_path):
                raise FileNotFoundError(f"Historical data file not found at {historical_data_path}")
            
            # Specify dtypes to handle potential mixed type columns if any, though Date is main one
            # Let's assume other columns are mostly numeric or string as per original CSVs
            historical_df = pd.read_csv(historical_data_path, low_memory=False) 
            # Ensure Date column is parsed correctly; it's critical for historical feature logic
            historical_df['Date'] = pd.to_datetime(historical_df['Date'], dayfirst=True, errors='coerce')
            # Drop rows where Date could not be parsed if any, as they are unusable for time-series logic
            historical_df.dropna(subset=['Date'], inplace=True)

            # Filter out future data if the CSV contains it (e.g. placeholder fixtures)
            historical_df = historical_df[historical_df['FTHG'].notna()] 

        except FileNotFoundError as e:
            print(e)
            print("Please ensure 'combined_epl_data.csv' is in the 'Combined Dataset' directory.")
            exit()
        except Exception as e:
            print(f"Error loading historical data: {e}")
            exit()
            
        # 2. Get the feature names the model was trained on (from X_train.csv columns)
        try:
            x_train_path = os.path.join(DATA_DIR, "X_train.csv")
            if not os.path.exists(x_train_path):
                 raise FileNotFoundError(f"X_train.csv not found at {x_train_path}. Needed for feature names.")
            # X_train.csv columns are ALREADY CLEANED by preprocess_data.py
            MODEL_FEATURE_NAMES = pd.read_csv(x_train_path, nrows=0).columns.tolist()
            print(f"Model expects {len(MODEL_FEATURE_NAMES)} features. First 5: {MODEL_FEATURE_NAMES[:5]}")
        except FileNotFoundError as e:
            print(e)
            print("Please run preprocess_data.py to generate X_train.csv.")
            exit()
        except Exception as e:
            print(f"Error loading X_train column names: {e}")
            exit()

        # 3. Define a new match to predict (example)
        # You'll need to fill this with actual data for a new match
        # Ensure all features used by your model (from X_train.columns) are potentially covered here
        # or handled (e.g. set to 0 or NaN if unavailable) in prepare_match_data
        new_match_data = {
            'Date': '2024-08-20', # Important: Date for historical context
            'HomeTeam': 'Arsenal',
            'AwayTeam': 'Chelsea',
            'Referee': 'Michael Oliver', 
            # Betting Odds (example - use actual odds for the match)
            'B365H': 1.80, 'B365D': 3.50, 'B365A': 4.20,
            'BSH': 1.85, 'BSD': 3.40, 'BSA': 4.10,
            'BWH': 1.90, 'BWD': 3.60, 'BWA': 4.00,
            'PSH': 1.82, 'PSD': 3.65, 'PSA': 4.25, # Pinnacle odds
            'MaxH':1.95, 'MaxD':3.70, 'MaxA':4.35, # Max odds from various bookies
            'AvgH':1.88, 'AvgD':3.62, 'AvgA':4.15, # Avg odds
            # Add any other simple features your model was trained on
            # e.g., if you had 'StadiumCapacity', 'Month', 'DayOfWeek' as features directly from input,
            # they would need to be included here.
            # Historical features (AvgGS, AvgGC, FormPts) will be calculated by prepare_match_data
        }

        print(f"\nPredicting for match: {new_match_data['HomeTeam']} vs {new_match_data['AwayTeam']} on {new_match_data['Date']}")
        
        predicted_result, probabilities = predict_single_match(new_match_data, model, label_encoders, MODEL_FEATURE_NAMES, historical_df)
        
        print(f"\nPredicted Outcome: {predicted_result}")
        if probabilities is not None:
            print(f"Probabilities (A,D,H): {[f'{p:.3f}' for p in probabilities]}")

        # Example with a team potentially not in training data (to test error handling)
        # new_match_data_unknown_team = {
        #     'Date': '2024-08-21',
        #     'HomeTeam': 'Luton Town', # Assuming Luton might be newly promoted or not in original training's top teams
        #     'AwayTeam': 'New Team FC', # Definitely not in encoders
        #     'Referee': 'A. Taylor',
        #     'B365H': 2.5, 'B365D': 3.0, 'B365A': 2.8
        # }
        # print(f"\nPredicting for match: {new_match_data_unknown_team['HomeTeam']} vs {new_match_data_unknown_team['AwayTeam']}")
        # predicted_result_unknown, probabilities_unknown = predict_single_match(new_match_data_unknown_team, model, label_encoders, MODEL_FEATURE_NAMES, historical_df)
        # print(f"\nPredicted Outcome (unknown team): {predicted_result_unknown}")
        # if probabilities_unknown is not None:
        #      print(f"Probabilities (A,D,H): {[f'{p:.3f}' for p in probabilities_unknown]}")

    else:
        print("Could not load model or encoders. Prediction cannot proceed.") 