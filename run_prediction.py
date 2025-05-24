import os
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def load_models():
    """Load all available prediction models"""
    print("Loading models...")
    
    # Use relative path from script location
    models_dir = os.path.join(os.getcwd(), 'models')
    
    # Create dict to store loaded models
    models = {}
    
    # Try to load various models
    model_types = [
        ('odds', 'xgboost_with_odds.pkl', 'feature_names.pkl'), 
        ('realistis', 'xgboost_realistis.pkl', 'feature_names_realistis.pkl'),
        ('prematch', 'xgboost_prematch.pkl', 'feature_names_prematch.pkl')
    ]
    
    # Try loading each model
    for model_type, model_file, feature_file in model_types:
        model_path = os.path.join(models_dir, model_file)
        feature_path = os.path.join(models_dir, feature_file)
        
        if os.path.exists(model_path) and os.path.exists(feature_path):
            try:
                model = joblib.load(model_path)
                features = joblib.load(feature_path)
                
                models[model_type] = {
                    'model': model,
                    'features': features
                }
                
                print(f"✓ Loaded {model_type} model with {len(features)} features")
            except Exception as e:
                print(f"✗ Error loading {model_type} model: {e}")
    
    if not models:
        print("No models found. Please train models first.")
        sys.exit(1)
    
    return models

def get_team_list():
    """Get list of teams from prematch features or data files"""
    teams = []
    
    # Try to get team names from prematch features
    prematch_file = 'data/prematch_features.csv'
    if os.path.exists(prematch_file):
        try:
            df = pd.read_csv(prematch_file)
            home_teams = df['HomeTeam'].unique().tolist()
            away_teams = df['AwayTeam'].unique().tolist()
            teams = sorted(list(set(home_teams + away_teams)))
            print(f"Found {len(teams)} teams from prematch features")
            return teams
        except Exception as e:
            print(f"Error reading prematch features: {e}")
    
    # Fallback to data files
    data_dir = os.path.join(os.getcwd(), 'data', 'Dataset EPL New')
    if os.path.exists(data_dir):
        try:
            files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
            if files:
                df = pd.read_csv(files[-1])  # Use most recent file
                teams = sorted(df['HomeTeam'].unique().tolist())
                print(f"Found {len(teams)} teams from data files")
                return teams
        except Exception as e:
            print(f"Error reading data files: {e}")
    
    # Fallback to predefined team list
    teams = [
        'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 
        'Brighton', 'Chelsea', 'Crystal Palace', 'Everton',
        'Fulham', 'Liverpool', 'Luton', 'Man City',
        'Man United', 'Newcastle', 'Nottingham', 'Southampton',
        'Tottenham', 'West Ham', 'Wolves', 'Sheffield United'
    ]
    print(f"Using default list of {len(teams)} teams")
    return teams

def select_models():
    """Let user select which models to use"""
    available_models = ['odds', 'realistis', 'prematch']
    selected_models = []
    
    print("\n=== MODEL SELECTION ===")
    print("Available models:")
    print("1. Model dengan odds (akurasi ~97%)")
    print("2. Model realistis tanpa odds (akurasi ~52%)")
    print("3. Model prematch dengan fitur yang dihitung (akurasi ~62%)")
    print("4. Semua model di atas")
    
    choice = input("\nPilih model (1-4): ")
    
    if choice == '1':
        selected_models = ['odds']
    elif choice == '2':
        selected_models = ['realistis']
    elif choice == '3':
        selected_models = ['prematch']
    elif choice == '4':
        selected_models = available_models
    else:
        print("Pilihan tidak valid, menggunakan semua model")
        selected_models = available_models
    
    return selected_models

def select_teams(teams):
    """Let user select home and away teams"""
    print("\n=== TEAM SELECTION ===")
    
    # Display teams with numbers
    print("Daftar Tim:")
    for i, team in enumerate(teams, 1):
        print(f"{i}. {team}")
    
    # Select home team
    while True:
        try:
            home_idx = int(input("\nPilih tim tuan rumah (nomor): ")) - 1
            if 0 <= home_idx < len(teams):
                home_team = teams[home_idx]
                break
            else:
                print("Nomor tim tidak valid")
        except ValueError:
            print("Masukkan nomor tim yang valid")
    
    # Select away team
    while True:
        try:
            away_idx = int(input("Pilih tim tamu (nomor): ")) - 1
            if 0 <= away_idx < len(teams) and away_idx != home_idx:
                away_team = teams[away_idx]
                break
            elif away_idx == home_idx:
                print("Tim tamu tidak boleh sama dengan tim tuan rumah")
            else:
                print("Nomor tim tidak valid")
        except ValueError:
            print("Masukkan nomor tim yang valid")
    
    return home_team, away_team

def predict_match(models, selected_models, home_team, away_team):
    """Predict match outcome using selected models"""
    # Map for outcome codes
    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    
    # Initialize results dictionary
    result = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Date': datetime.now().strftime("%Y-%m-%d"),
        'models': {}
    }
    
    # Load prematch features if available
    prematch_file = 'data/prematch_features.csv'
    prematch_features = None
    
    if os.path.exists(prematch_file):
        try:
            prematch_features = pd.read_csv(prematch_file)
            print(f"Loaded prematch features from {prematch_file}")
        except Exception as e:
            print(f"Error loading prematch features: {e}")
    
    fixture = {'HomeTeam': home_team, 'AwayTeam': away_team}
    
    # Predict with each selected model
    for model_type in selected_models:
        if model_type in models:
            try:
                model_info = models[model_type]
                
                # Create prediction data
                pred_data = {}
                
                if prematch_features is not None and model_type == 'prematch':
                    # Find the most recent prematch features for both teams
                    home_features = prematch_features[prematch_features['HomeTeam'] == home_team]
                    away_features = prematch_features[prematch_features['AwayTeam'] == away_team]
                    
                    # Get team specific features
                    team_features = {}
                    
                    # Process home team features
                    if not home_features.empty:
                        latest_home = home_features.sort_values('Date').iloc[-1]
                        for col in latest_home.index:
                            if col.startswith('Home') and col not in ['HomeTeam']:
                                team_features[col] = latest_home[col]
                    
                    # Process away team features
                    if not away_features.empty:
                        latest_away = away_features.sort_values('Date').iloc[-1]
                        for col in latest_away.index:
                            if col.startswith('Away') and col not in ['AwayTeam']:
                                team_features[col] = latest_away[col]
                    
                    # Check for head-to-head features
                    h2h_features = prematch_features[(prematch_features['HomeTeam'] == home_team) & 
                                                   (prematch_features['AwayTeam'] == away_team)]
                    
                    if not h2h_features.empty:
                        latest_h2h = h2h_features.sort_values('Date').iloc[-1]
                        for col in latest_h2h.index:
                            if col.startswith('H2H_'):
                                team_features[col] = latest_h2h[col]
                    
                    # Add team features to prediction data
                    pred_data.update(team_features)
                
                # Basic feature encoding for team names
                from sklearn.preprocessing import LabelEncoder
                required_features = model_info['features']
                
                if isinstance(required_features, list) and 'HomeTeam' in required_features and 'AwayTeam' in required_features:
                    if prematch_features is not None:
                        team_encoder = LabelEncoder().fit(pd.concat([prematch_features['HomeTeam'], prematch_features['AwayTeam']]))
                        try:
                            pred_data['HomeTeam'] = team_encoder.transform([home_team])[0]
                            pred_data['AwayTeam'] = team_encoder.transform([away_team])[0]
                        except:
                            # Fallback if teams aren't in the encoder
                            pred_data['HomeTeam'] = 0
                            pred_data['AwayTeam'] = 1
                    else:
                        pred_data['HomeTeam'] = 0
                        pred_data['AwayTeam'] = 1
                
                # Make sure all required features are present
                for feature in required_features:
                    if feature not in pred_data:
                        # Set default values for missing features
                        if 'avg' in feature.lower() or 'mean' in feature.lower():
                            pred_data[feature] = 1.0  # Average value
                        elif 'form' in feature.lower():
                            pred_data[feature] = 0.5  # Medium form
                        elif 'win' in feature.lower() or 'pct' in feature.lower():
                            pred_data[feature] = 0.5  # Neutral win percentage
                        else:
                            pred_data[feature] = 0  # Default to zero for other features
                
                # Create DataFrame with a single row
                X_pred = pd.DataFrame([pred_data])
                
                # Select only required features in correct order
                X_pred = X_pred[required_features]
                
                # Make prediction
                model = model_info['model']
                prediction = model.predict(X_pred)[0]
                probabilities = model.predict_proba(X_pred)[0]
                
                # Store results
                result['models'][model_type] = {
                    'prediction': int(prediction),
                    'outcome': outcome_map[int(prediction)],
                    'probabilities': {
                        'Home Win': float(probabilities[0]),
                        'Draw': float(probabilities[1]),
                        'Away Win': float(probabilities[2])
                    }
                }
                
                # Print prediction
                print(f"\n{model_type.capitalize()} model predicts: {outcome_map[int(prediction)]}")
                print(f"  Home Win: {probabilities[0]:.2f}, Draw: {probabilities[1]:.2f}, Away Win: {probabilities[2]:.2f}")
                
            except Exception as e:
                print(f"Error predicting with {model_type} model: {e}")
    
    return result

def visualize_prediction(result):
    """Create visualization for the prediction"""
    print("\nGenerating visualization...")
    
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)
    
    home_team = result['HomeTeam']
    away_team = result['AwayTeam']
    
    # Process predictions from each model
    if result['models']:
        # Setup the visualization
        n_models = len(result['models'])
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 4))
        
        # Handle case with only one model
        if n_models == 1:
            axes = [axes]
        
        # Plot each model's prediction
        for i, (model_type, prediction) in enumerate(result['models'].items()):
            # Get prediction data
            probabilities = prediction['probabilities']
            outcome = prediction['outcome']
            
            # Plot probabilities
            ax = axes[i]
            outcomes = list(probabilities.keys())
            probs = list(probabilities.values())
            
            # Create bar chart
            colors = ['green', 'gray', 'red']
            bars = ax.bar(outcomes, probs, color=colors)
            
            # Add values on top of bars
            for bar, prob in zip(bars, probs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                       f'{prob:.2f}', ha='center', va='bottom')
            
            # Customize plot
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{model_type.capitalize()} Model: {outcome}")
            ax.set_ylabel('Probability')
            
            # Highlight predicted outcome
            highlight_idx = outcomes.index(outcome)
            bars[highlight_idx].set_alpha(1.0)
            for j, bar in enumerate(bars):
                if j != highlight_idx:
                    bar.set_alpha(0.6)
        
        # Set common title
        plt.suptitle(f"{home_team} vs {away_team}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save the figure
        safe_filename = f"{home_team}_vs_{away_team}".replace(' ', '_')
        output_path = f'predictions/{safe_filename}_manual.png'
        plt.savefig(output_path)
        plt.close()
        
        print(f"Visualization saved to {output_path}")
        
        # Save detailed results as JSON
        with open(f'predictions/{safe_filename}_result.json', 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Detailed results saved to predictions/{safe_filename}_result.json")

def main():
    """Main function to predict matches manually"""
    clear_screen()
    print("Premier League Match Predictor")
    print("=============================")
    
    # Load models
    models = load_models()
    
    # Get team list
    teams = get_team_list()
    
    # Let user select models
    selected_models = select_models()
    print(f"Selected models: {', '.join(selected_models)}")
    
    # Let user select teams
    home_team, away_team = select_teams(teams)
    print(f"\nPredicting: {home_team} vs {away_team}")
    
    # Make prediction
    result = predict_match(models, selected_models, home_team, away_team)
    
    # Visualize prediction
    visualize_prediction(result)
    
    print("\nPrediction complete!")

if __name__ == "__main__":
    main() 