import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
import json
import os

# Load the preprocessed training data
try:
    X_train = pd.read_csv("Combined Dataset/X_train.csv")
    # y_train dibaca dan diratakan menjadi array 1D
    y_train = pd.read_csv("Combined Dataset/y_train.csv").iloc[:, 0].values.ravel()
except FileNotFoundError:
    print("Error: X_train.csv or y_train.csv not found. Please run preprocess_data.py first.")
    exit()

print("Data loaded successfully.")
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"First 5 columns of X_train: {X_train.columns.tolist()[:5]}") # Untuk verifikasi nama kolom bersih

# Define the parameter grid for RandomizedSearchCV
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 10),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'gamma': uniform(0, 0.5),
    'min_child_weight': randint(1, 5)
}

print("\nStarting Hyperparameter Tuning with RandomizedSearchCV...")
# Pastikan eval_metric dan use_label_encoder sudah sesuai
# eval_metric='mlogloss' untuk multiclass classification
# use_label_encoder=False karena y_train sudah di-encode
xgb_model = XGBClassifier(objective='multi:softmax', num_class=3, use_label_encoder=False, random_state=42, eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=param_dist,
    n_iter=50, # Sesuai histori, 50 iterasi
    cv=3,      # Sesuai histori, 3-fold CV
    scoring='accuracy',
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\nHyperparameter Tuning Finished.")
print(f"Best Parameters found: {random_search.best_params_}")
print(f"Best Cross-validation Accuracy: {random_search.best_score_:.4f}")

# --- Save Best Parameters ---
output_dir_models = "models"
if not os.path.exists(output_dir_models):
    os.makedirs(output_dir_models)

best_params_path = os.path.join(output_dir_models, "best_params.json")
with open(best_params_path, 'w') as f:
    json.dump(random_search.best_params_, f, indent=4)

print(f"\nBest parameters saved to {best_params_path}")
print("\nTuning script finished.") 