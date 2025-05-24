import pandas as pd
import numpy as np
import joblib
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def load_models():
    """Load all available prediction models"""
    # Print current working directory for debugging
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")
    
    # Use relative path from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    models_dir = os.path.join(project_root, 'models')
    
    print(f"Looking for models in: {models_dir}")
    
    # Create dict to store loaded models
    models = {}
    
    # Try to load various models
    model_types = [
        ('odds', 'xgboost_with_odds.pkl', 'feature_names.pkl'), 
        ('realistis', 'xgboost_realistis.pkl', 'feature_names_realistis.pkl'),
        ('prematch', 'xgboost_prematch.pkl', 'feature_names_prematch.pkl')
    ]
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
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
                
                print(f"Loaded {model_type} model with {len(features)} features")
            except Exception as e:
                print(f"Error loading {model_type} model: {e}")
    
    if not models:
        print("No models found. Please train models first.")
    
    return models

def load_or_create_fixtures():
    """Load fixture data or create a sample fixture list"""
    fixtures_file = 'predictions/upcoming_fixtures.csv'
    
    # Check if fixtures file exists
    if os.path.exists(fixtures_file):
        print(f"Loading fixtures from {fixtures_file}")
        fixtures = pd.read_csv(fixtures_file)
        print(f"Loaded {len(fixtures)} fixtures")
        return fixtures
    else:
        print(f"Fixtures file not found, creating sample fixtures")
        
        # Try to get team names from reference data
        try:
            # Load reference data to get team names
            teams = []
            
            # Use relative path from script location
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, '..'))
            data_dir = os.path.join(project_root, 'data', 'Dataset EPL New')
            
            if os.path.exists(data_dir):
                files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
                if files:
                    df = pd.read_csv(files[-1])  # Use most recent file
                    teams = sorted(df['HomeTeam'].unique().tolist())
            
            if not teams:
                # Fallback to predefined team list
                teams = [
                    'Arsenal', 'Aston Villa', 'Bournemouth', 'Brentford', 
                    'Brighton', 'Chelsea', 'Crystal Palace', 'Everton',
                    'Fulham', 'Liverpool', 'Luton', 'Man City',
                    'Man United', 'Newcastle', 'Nottingham', 'Southampton',
                    'Tottenham', 'West Ham', 'Wolves', 'Sheffield United'
                ]
            
            # Create sample fixtures (10 matches)
            fixtures = []
            
            # Generate some derby matches for better examples
            fixtures.append({'Date': '2025-08-16', 'HomeTeam': 'Man United', 'AwayTeam': 'Man City'})
            fixtures.append({'Date': '2025-08-16', 'HomeTeam': 'Arsenal', 'AwayTeam': 'Tottenham'})
            fixtures.append({'Date': '2025-08-16', 'HomeTeam': 'Liverpool', 'AwayTeam': 'Everton'})
            
            # Add some random matches
            import random
            random.seed(42)  # For reproducibility
            
            for i in range(7):
                home = random.choice(teams)
                # Ensure away team is different from home team
                away = random.choice([t for t in teams if t != home])
                fixtures.append({'Date': '2025-08-16', 'HomeTeam': home, 'AwayTeam': away})
            
            # Convert to DataFrame
            fixtures_df = pd.DataFrame(fixtures)
            
            # Create predictions directory if it doesn't exist
            os.makedirs('predictions', exist_ok=True)
            
            # Save to file
            fixtures_df.to_csv(fixtures_file, index=False)
            print(f"Created and saved sample fixtures to {fixtures_file}")
            
            return fixtures_df
            
        except Exception as e:
            print(f"Error creating sample fixtures: {e}")
            # Return a simple DataFrame with a few matches
            return pd.DataFrame([
                {'Date': '2025-08-16', 'HomeTeam': 'Man United', 'AwayTeam': 'Man City'},
                {'Date': '2025-08-16', 'HomeTeam': 'Arsenal', 'AwayTeam': 'Tottenham'},
                {'Date': '2025-08-16', 'HomeTeam': 'Liverpool', 'AwayTeam': 'Everton'}
            ])

def predict_with_model(model_info, fixture, prematch_features=None):
    """Predict match outcome using the given model"""
    model = model_info['model']
    required_features = model_info['features']
    
    # Create prediction data
    pred_data = {}
    
    if prematch_features is not None and isinstance(required_features, list) and 'HomeTeam' in required_features and 'AwayTeam' in required_features:
        # Basic feature encoding for team names
        team_encoder = LabelEncoder().fit(pd.concat([prematch_features['HomeTeam'], prematch_features['AwayTeam']]))
        
        # Encode team names
        try:
            pred_data['HomeTeam'] = team_encoder.transform([fixture['HomeTeam']])[0]
            pred_data['AwayTeam'] = team_encoder.transform([fixture['AwayTeam']])[0]
        except:
            # Fallback if teams aren't in the encoder
            pred_data['HomeTeam'] = 0
            pred_data['AwayTeam'] = 1
            
        # Find the most recent prematch features for both teams
        home_features = prematch_features[prematch_features['HomeTeam'] == fixture['HomeTeam']]
        away_features = prematch_features[prematch_features['AwayTeam'] == fixture['AwayTeam']]
        
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
        h2h_features = prematch_features[(prematch_features['HomeTeam'] == fixture['HomeTeam']) & 
                                        (prematch_features['AwayTeam'] == fixture['AwayTeam'])]
        
        if not h2h_features.empty:
            latest_h2h = h2h_features.sort_values('Date').iloc[-1]
            for col in latest_h2h.index:
                if col.startswith('H2H_'):
                    team_features[col] = latest_h2h[col]
        
        # Add team features to prediction data
        pred_data.update(team_features)
    else:
        # Basic feature encoding for team names
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
    prediction = model.predict(X_pred)[0]
    probabilities = model.predict_proba(X_pred)[0]
    
    return prediction, probabilities

def predict_all_fixtures(fixtures, models):
    """Predict all fixtures using all available models"""
    # Map for outcome codes
    outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
    
    # Initialize results dictionary
    all_results = {}
    
    # Load prematch features if available
    prematch_file = 'data/prematch_features.csv'
    prematch_features = None
    
    if os.path.exists(prematch_file):
        try:
            prematch_features = pd.read_csv(prematch_file)
            print(f"Loaded prematch features from {prematch_file}")
        except Exception as e:
            print(f"Error loading prematch features: {e}")
    
    # Process each fixture
    for idx, fixture in fixtures.iterrows():
        home_team = fixture['HomeTeam']
        away_team = fixture['AwayTeam']
        fixture_key = f"{home_team} vs {away_team}"
        
        print(f"\nPredicting: {fixture_key}")
        
        # Initialize fixture results
        all_results[fixture_key] = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': fixture['Date'] if 'Date' in fixture else 'Unknown',
            'models': {}
        }
        
        # Predict with each model
        for model_type, model_info in models.items():
            try:
                prediction, probabilities = predict_with_model(
                    model_info, 
                    fixture, 
                    prematch_features
                )
                
                # Store results
                all_results[fixture_key]['models'][model_type] = {
                    'prediction': int(prediction),
                    'outcome': outcome_map[int(prediction)],
                    'probabilities': {
                        'Home Win': float(probabilities[0]),
                        'Draw': float(probabilities[1]),
                        'Away Win': float(probabilities[2])
                    }
                }
                
                # Print prediction
                print(f"  {model_type.capitalize()} model: {outcome_map[int(prediction)]}")
                print(f"    Home Win: {probabilities[0]:.2f}, Draw: {probabilities[1]:.2f}, Away Win: {probabilities[2]:.2f}")
                
            except Exception as e:
                print(f"  Error predicting with {model_type} model: {e}")
    
    return all_results

def visualize_predictions(all_results):
    """Create visualizations for the predictions"""
    print("\nGenerating visualizations...")
    
    # Create predictions directory if it doesn't exist
    os.makedirs('predictions', exist_ok=True)
    
    # Create a summary DataFrame for CSV export
    summary_data = []
    
    # Process each fixture
    for fixture_key, fixture_data in all_results.items():
        home_team = fixture_data['HomeTeam']
        away_team = fixture_data['AwayTeam']
        date = fixture_data['Date']
        
        # Create row for summary data
        summary_row = {
            'Date': date,
            'HomeTeam': home_team,
            'AwayTeam': away_team
        }
        
        # Process predictions from each model
        if fixture_data['models']:
            # Setup the visualization
            n_models = len(fixture_data['models'])
            fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 4))
            
            # Handle case with only one model
            if n_models == 1:
                axes = [axes]
            
            # Plot each model's prediction
            for i, (model_type, prediction) in enumerate(fixture_data['models'].items()):
                # Get prediction data
                probabilities = prediction['probabilities']
                outcome = prediction['outcome']
                
                # Update summary row
                summary_row[f'{model_type}_prediction'] = outcome
                for outcome_type, prob in probabilities.items():
                    summary_row[f'{model_type}_{outcome_type.replace(" ", "")}'] = prob
                
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
            plt.suptitle(f"{home_team} vs {away_team} ({date})", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save the figure
            safe_filename = fixture_key.replace(' ', '_').replace('/', '_')
            plt.savefig(f'predictions/{safe_filename}.png')
            plt.close()
            
        # Add summary row to data
        summary_data.append(summary_row)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('predictions/prediction_summary.csv', index=False)
    print(f"Summary saved to predictions/prediction_summary.csv")
    
    # Save detailed results as JSON
    with open('predictions/detailed_predictions.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Detailed results saved to predictions/detailed_predictions.json")

def main():
    """Main function to predict upcoming fixtures"""
    print("Premier League Match Predictor")
    print("===============================")
    
    # Current timestamp
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Run at: {now}")
    
    # Load models
    models = load_models()
    
    if not models:
        print("No models available. Please train models first.")
        return
    
    # Load or create fixtures
    fixtures = load_or_create_fixtures()
    
    # Make predictions
    all_results = predict_all_fixtures(fixtures, models)
    
    # Visualize predictions
    visualize_predictions(all_results)
    
    print("\nPrediction complete!")
    print(f"Check the 'predictions' folder for results.")

if __name__ == "__main__":
    main() 