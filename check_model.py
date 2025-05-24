import joblib
import os
import numpy as np
import pickle

# Define ModelWrapper class to handle models with feature names
class ModelWrapper:
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
    
    def predict(self, X):
        return self.model.predict(X)
    
    def predict_proba(self, X):
        return self.model.predict_proba(X)

# Load the model and feature names
print("Loading model and feature names...")
model_path = os.path.join('models', 'xgboost_prematch.pkl')
features_path = os.path.join('models', 'feature_names_prematch.pkl')

try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    
    print(f"Model type: {type(model)}")
    print(f"Has feature_names attribute: {hasattr(model, 'feature_names')}")
    print(f"Has model attribute: {hasattr(model, 'model')}")
    
    if hasattr(model, 'model'):
        inner_model = model.model
        print(f"Inner model type: {type(inner_model)}")
        
        # For CalibratedClassifierCV
        if hasattr(inner_model, 'calibrated_classifiers_'):
            print("Model has calibrated_classifiers_")
            calibrated_classifiers = inner_model.calibrated_classifiers_
            print(f"Number of calibrated classifiers: {len(calibrated_classifiers)}")
            
            for i, calibrated_clf in enumerate(calibrated_classifiers):
                print(f"\nCalibrated classifier {i+1}:")
                print(f"Type: {type(calibrated_clf)}")
                
                # Check for estimator attribute
                if hasattr(calibrated_clf, 'estimator'):
                    print(f"Estimator type: {type(calibrated_clf.estimator)}")
                    estimator = calibrated_clf.estimator
                    
                    # For XGBoost models
                    if hasattr(estimator, 'get_booster'):
                        print("Estimator has get_booster method")
                        try:
                            booster = estimator.get_booster()
                            print(f"Booster type: {type(booster)}")
                            
                            # Try to get feature count from n_features_in_
                            if hasattr(estimator, 'n_features_in_'):
                                print(f"Estimator n_features_in_: {estimator.n_features_in_}")
                        except Exception as e:
                            print(f"Error getting booster: {e}")
    
    print(f"\nNumber of features in feature_names_prematch.pkl: {len(features)}")
    
    # Check if there's a mismatch
    if hasattr(model, 'feature_names'):
        model_features = model.feature_names
        print(f"Number of features in model.feature_names: {len(model_features)}")
        
        if len(model_features) != len(features):
            print("MISMATCH DETECTED between model.feature_names and feature_names_prematch.pkl")
            
            # Find missing features
            missing_in_file = set(model_features) - set(features)
            missing_in_model = set(features) - set(model_features)
            
            if missing_in_file:
                print(f"Features in model but not in file: {missing_in_file}")
            if missing_in_model:
                print(f"Features in file but not in model: {missing_in_model}")
    
    # Try to create a dummy input with the right number of features
    print("\nTrying to create a dummy prediction...")
    if hasattr(model, 'feature_names'):
        dummy_input = np.zeros((1, len(model.feature_names)))
        try:
            result = model.model.predict(dummy_input)
            print(f"Prediction succeeded with {len(model.feature_names)} features")
        except Exception as e:
            print(f"Prediction failed: {e}")
            
            # Try with different feature counts
            for n_features in [73, 112]:
                print(f"\nTrying with {n_features} features...")
                try:
                    dummy_input = np.zeros((1, n_features))
                    result = model.model.predict(dummy_input)
                    print(f"Prediction succeeded with {n_features} features")
                except Exception as e:
                    print(f"Prediction failed: {e}")
    
    # Check if there are improved models available
    improved_model_path = os.path.join('models', 'xgboost_prematch_improved.pkl')
    if os.path.exists(improved_model_path):
        print("\nFound improved model. Checking compatibility...")
        try:
            improved_model = joblib.load(improved_model_path)
            print(f"Improved model type: {type(improved_model)}")
            
            if hasattr(improved_model, 'feature_names'):
                print(f"Improved model has {len(improved_model.feature_names)} features")
                
                # Try prediction with improved model
                dummy_input = np.zeros((1, len(improved_model.feature_names)))
                try:
                    result = improved_model.model.predict(dummy_input)
                    print(f"Prediction with improved model succeeded")
                except Exception as e:
                    print(f"Prediction with improved model failed: {e}")
        except Exception as e:
            print(f"Error loading improved model: {e}")
    
except Exception as e:
    print(f"Error loading model: {e}") 