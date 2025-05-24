import joblib
import os
import numpy as np
import pandas as pd

def inspect_model():
    """Inspect the saved XGBoost model and its features"""
    model_path = 'models/xgboost_with_odds.pkl'
    feature_names_path = 'models/feature_names.pkl'
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Try different ways to get feature names
    print("\nAttempting to access model features...")
    
    # Method 1: From saved feature names file
    if os.path.exists(feature_names_path):
        feature_names = joblib.load(feature_names_path)
        print(f"\nFeature names from saved file (first 10): {feature_names[:10]}")
        print(f"Total feature count: {len(feature_names)}")
    
    # Method 2: From model's booster
    try:
        booster = model.get_booster()
        model_features = booster.feature_names
        if model_features:
            print(f"\nFeature names from model booster (first 10): {model_features[:10]}")
            print(f"Total booster feature count: {len(model_features)}")
        else:
            print("\nNo feature names found in model booster")
    except Exception as e:
        print(f"\nError accessing booster features: {e}")
    
    # Method 3: From feature_importances_
    try:
        importances = model.feature_importances_
        print(f"\nFeature importances shape: {importances.shape}")
        print(f"Top 5 importance values: {importances[:5]}")
        
        # Get top features
        if os.path.exists(feature_names_path):
            feature_names = joblib.load(feature_names_path)
            if len(importances) == len(feature_names):
                indices = np.argsort(importances)[::-1]
                print("\nTop 10 most important features:")
                for i in range(min(10, len(indices))):
                    print(f"  {i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    except Exception as e:
        print(f"\nError accessing feature importances: {e}")
    
    # Examine other model properties
    print(f"\nModel type: {type(model)}")
    print(f"Model parameters: {model.get_params()}")
    
    # Check if model was trained with feature names
    if hasattr(model, 'feature_names_in_'):
        print(f"\nFeature names used during training (first 10): {model.feature_names_in_[:10]}")
        print(f"Total training features: {len(model.feature_names_in_)}")

if __name__ == "__main__":
    inspect_model() 