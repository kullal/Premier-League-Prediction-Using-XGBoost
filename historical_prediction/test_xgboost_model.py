import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score

def load_model_and_metadata():
    """Load the XGBoost model and associated metadata"""
    model_path = 'models/xgboost_with_odds.pkl'
    feature_names_path = 'models/feature_names.pkl'
    encoders_path = 'models/label_encoders.pkl'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train_xgboost_with_odds.py first.")
    
    model = joblib.load(model_path)
    feature_names = joblib.load(feature_names_path)
    encoders = joblib.load(encoders_path)
    
    print(f"Loaded model from {model_path}")
    print(f"Model has {len(feature_names)} features")
    
    return model, feature_names, encoders

def load_and_preprocess_data(test_file=None):
    """Load and preprocess test data, similar to training pipeline"""
    # If specific test file is provided, use it
    # Otherwise, use the latest season as test data
    if test_file:
        print(f"Using specified test file: {test_file}")
        df = pd.read_csv(test_file)
    else:
        # Print current working directory for debugging
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")
        
        # Use relative path from script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, '..'))
        data_dir = os.path.join(project_root, 'data', 'Dataset EPL New')
        print(f"Looking for CSV files in: {os.path.abspath(data_dir)}")
        
        # Mendapatkan list file secara manual tanpa menggunakan glob karena terkendala spasi
        try:
            if os.path.exists(data_dir):
                files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
            else:
                files = []
        except Exception as e:
            print(f"Error listing directory: {e}")
            files = []
        
        print(f"Found {len(files)} dataset files: {files}")
        
        # If no files found, try potential alternative paths
        if not files:
            alternative_paths = [
                os.path.join(project_root, 'data', 'Dataset EPL New'),
                os.path.join(current_dir, 'data', 'Dataset EPL New'),
                os.path.join(os.path.dirname(current_dir), 'data', 'Dataset EPL New'),
                os.path.abspath(os.path.join('..', '..', 'data', 'Dataset EPL New')),
                os.path.abspath(os.path.join('..', 'data', 'Dataset EPL New')),
                os.path.abspath('data/Dataset EPL New')
            ]
            
            for alt_path in alternative_paths:
                print(f"Trying alternative path: {alt_path}")
                try:
                    if os.path.exists(alt_path):
                        files = [os.path.join(alt_path, f) for f in os.listdir(alt_path) if f.endswith('.csv')]
                        if files:
                            print(f"Found {len(files)} dataset files in: {alt_path}")
                            data_dir = alt_path  # Update data_dir to the successful path
                            break
                    else:
                        print(f"Path does not exist: {alt_path}")
                except Exception as e:
                    print(f"Error listing alternative directory {alt_path}: {e}")
        
        if not files:
            raise FileNotFoundError(f"No dataset files found. Please check the path and file names.")
        
        # Urutkan file berdasarkan nama dan ambil yang terbaru
        files = sorted(files)
        test_file = files[-1]  # Take the most recent season
        print(f"Using most recent season as test data: {test_file}")
        
        try:
        df = pd.read_csv(test_file)
        except Exception as e:
            raise ValueError(f"Error reading test file {test_file}: {e}")
    
    print(f"Test dataset shape: {df.shape}")
    
    # Process the data similar to training pipeline
    # Ensure 'FTR' is available for evaluation
    if 'FTR' not in df.columns:
        raise ValueError("Target column 'FTR' not found in test data")
    
    # Process categorical columns and target
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'FTR' in categorical_cols:
        categorical_cols.remove('FTR')
    
    # Basic encoding for now - note we're not using the saved encoders
    # since we might have new categories in test data
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    
    # Target encoding
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    df = df[df['FTR'].notnull()]
    
    # Prepare features and target
    X = df.drop(['FTR'], axis=1)
    y = df['FTR']
    
    # Clean column names
    X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]
    
    # Convert to numeric, handle missing values
    numeric_data = []
    for col in X.columns:
        try:
            converted_series = pd.to_numeric(X[col], errors='coerce')
            numeric_data.append(converted_series)
        except:
            pass
    
    X_clean = pd.concat(numeric_data, axis=1)
    
    # Fill missing values
    X_clean = X_clean.fillna(X_clean.median())
    
    # Align target with cleaned features
    y_clean = y.loc[X_clean.index]
    
    return X_clean, y_clean, df

def prepare_test_data_for_model(X_test, model):
    """Prepare test data to match the model's expected features"""
    try:
        # Get model's feature names
        if hasattr(model, 'feature_names_in_'):
            model_features = model.feature_names_in_
        else:
            try:
                model_features = model.get_booster().feature_names
            except:
                # If we can't get feature names from model, try to load them from file
                try:
                    feature_names_path = 'models/feature_names.pkl'
                    model_features = joblib.load(feature_names_path)
                except:
                    print("Could not determine model features. Using all available features.")
                    return X_test
        
        print(f"Model has {len(model_features)} features")
        
        # Find common features between test data and model
        common_features = [f for f in model_features if f in X_test.columns]
        print(f"Found {len(common_features)} common features between test data and model")
        
        # Check for missing features
        missing_features = [f for f in model_features if f not in X_test.columns]
        if missing_features:
            print(f"Warning: {len(missing_features)} features required by model not found in test data:")
            # Print only first few missing features to avoid cluttering output
            print(f"  Missing features (first 10): {missing_features[:10]}")
            
            # Create dummy columns for missing features with zeros
            for feature in missing_features:
                X_test[feature] = 0
            print("Added missing features to test data with default values (0)")
        
        # Ensure test data has all model features and in the right order
        X_test_prepared = X_test[model_features]
        print(f"Final test data shape: {X_test_prepared.shape}")
        
        return X_test_prepared
        
    except Exception as e:
        print(f"Error preparing test data: {e}")
        print("Using original test data without feature alignment")
        return X_test

def evaluate_model(model, X, y, feature_names=None):
    """Evaluate the model using various metrics and plots"""
    # Prepare test data to match model's expected features
    X_prepared = prepare_test_data_for_model(X, model)
    
    try:
        # Make predictions
        y_pred = model.predict(X_prepared)
        y_pred_proba = model.predict_proba(X_prepared)
        
        # Basic accuracy
        accuracy = accuracy_score(y, y_pred)
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        
        # Detailed classification report
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Home Win', 'Draw', 'Away Win'],
                    yticklabels=['Home Win', 'Draw', 'Away Win'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix visualization saved to 'confusion_matrix.png'")
        
        # Feature importance visualization
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(10, 8))
                feature_importance = model.feature_importances_
                
                # Get model feature names
                if hasattr(model, 'feature_names_in_'):
                    columns_for_importance = model.feature_names_in_
                else:
                    columns_for_importance = X_prepared.columns
                
                # Show top 20 features
                indices = np.argsort(feature_importance)[::-1]
                top_n = min(20, len(indices))
                plt.barh(range(top_n), feature_importance[indices[:top_n]])
                plt.yticks(range(top_n), [columns_for_importance[i] for i in indices[:top_n]])
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Important Features')
                plt.tight_layout()
                plt.savefig('feature_importance.png')
                print(f"Feature importance visualization saved to 'feature_importance.png'")
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
        
        return accuracy, y_pred, y_pred_proba
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None, None, None

def predict_match(model, feature_names, home_team, away_team, test_data):
    """Predict the outcome of a specific match"""
    # Find matches in the test set with these teams
    try:
        match = test_data[(test_data['HomeTeam'] == home_team) & 
                          (test_data['AwayTeam'] == away_team)]
        
        if len(match) == 0:
            print(f"No match found between {home_team} and {away_team} in the test data")
            # Try to find any match with home team
            match = test_data[test_data['HomeTeam'] == home_team].iloc[0:1]
            if len(match) == 0:
                # If still not found, use any match as a template
                match = test_data.iloc[0:1]
                print(f"Using generic match data as template")
            else:
                print(f"Using a match with {home_team} as home team as template")
                # Update away team for the prediction
                match['AwayTeam'] = away_team
        
        # Make a copy for prediction (to avoid modifying original)
        match_copy = match.copy()
        
        # Use this match for prediction
        match_features = match_copy.drop(['FTR'], axis=1)
        
        # Clean column names
        match_features.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') 
                                for col in match_features.columns]
        
        # Convert to numeric, handle missing values
        numeric_data = []
        for col in match_features.columns:
            try:
                converted_series = pd.to_numeric(match_features[col], errors='coerce')
                numeric_data.append(converted_series)
            except:
                pass
        
        match_features_clean = pd.concat(numeric_data, axis=1)
        match_features_clean = match_features_clean.fillna(match_features_clean.median())
        
        # Prepare data to match model's expected features
        match_features_prepared = prepare_test_data_for_model(match_features_clean, model)
        
        # Make prediction
        pred = model.predict(match_features_prepared)[0]
        pred_proba = model.predict_proba(match_features_prepared)[0]
        
        # Map prediction to human-readable outcome
        outcome_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
        
        print(f"\nPrediction for {home_team} vs {away_team}:")
        print(f"Predicted outcome: {outcome_map[pred]}")
        
        # If we have the actual result
        if 'FTR' in match.columns:
            try:
                actual = match['FTR'].values[0]
                print(f"Actual outcome: {outcome_map[actual]}")
            except:
                print("Actual outcome not available")
        
        print(f"Prediction probabilities:")
        print(f"  Home Win: {pred_proba[0]:.2f} ({pred_proba[0]*100:.1f}%)")
        print(f"  Draw: {pred_proba[1]:.2f} ({pred_proba[1]*100:.1f}%)")
        print(f"  Away Win: {pred_proba[2]:.2f} ({pred_proba[2]*100:.1f}%)")
        
        # Create a visualization of the prediction
        plt.figure(figsize=(8, 5))
        sns.barplot(x=['Home Win', 'Draw', 'Away Win'], y=pred_proba)
        plt.title(f'Prediction: {home_team} vs {away_team}')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        for i, p in enumerate(pred_proba):
            plt.text(i, p + 0.02, f'{p:.2f}', ha='center')
        plt.tight_layout()
        
        # Create predictions directory if it doesn't exist
        os.makedirs('predictions', exist_ok=True)
        plt.savefig(f'predictions/{home_team}_vs_{away_team}.png')
        print(f"Prediction visualization saved to 'predictions/{home_team}_vs_{away_team}.png'")
        
        return pred, pred_proba, match['FTR'].values[0] if 'FTR' in match.columns else None
        
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def main():
    # Load model and metadata
    try:
        model, feature_names, encoders = load_model_and_metadata()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please run the training script first to generate the model.")
        return
    
    # Load and preprocess test data
    try:
        X_test, y_test, test_data = load_and_preprocess_data()
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    # Evaluate the model
    try:
        accuracy, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, feature_names)
    except Exception as e:
        print(f"Error during model evaluation: {e}")
        return
    
    # Test with specific match examples
    # Get team names (will depend on your dataset structure)
    try:
        team_names = test_data['HomeTeam'].unique()
        if len(team_names) >= 2:
            # Test a few specific matches
            # Pick a few teams - adjust these to teams in your dataset
            test_matches = [
                (team_names[0], team_names[1]),  # e.g., Arsenal vs Chelsea
                (team_names[2], team_names[3]) if len(team_names) > 3 else (team_names[1], team_names[0])
            ]
            
            for home, away in test_matches:
                predict_match(model, feature_names, home, away, test_data)
    except Exception as e:
        print(f"Error testing specific matches: {e}")
    
    print("\nModel testing completed.")

if __name__ == "__main__":
    main() 