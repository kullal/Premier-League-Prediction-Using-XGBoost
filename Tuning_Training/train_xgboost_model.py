import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns # Untuk confusion matrix heatmap

# Fungsi untuk memuat data
def load_data(X_path, y_path):
    try:
        X = pd.read_csv(X_path)
        y = pd.read_csv(y_path).iloc[:, 0].values.ravel()
        print(f"Data loaded: {X_path}, {y_path}")
        print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")
        if X.empty or len(y) == 0:
            print("Error: Loaded data is empty.")
            return None, None
        print(f"First 5 columns of X (in load_data): {X.columns.tolist()[:5]}") # Verifikasi
        return X, y
    except FileNotFoundError:
        print(f"Error: File not found. Please ensure {X_path} and {y_path} exist.")
        print("Try running preprocess_data.py first.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"Error: No data or empty file at {X_path} or {y_path}.")
        return None, None

# Muat data latih dan uji
print("Loading training data...")
X_train, y_train = load_data("Combined Dataset/X_train.csv", "Combined Dataset/y_train.csv")
print("\nLoading testing data...")
X_test, y_test = load_data("Combined Dataset/X_test.csv", "Combined Dataset/y_test.csv")

if X_train is None or y_train is None or X_test is None or y_test is None:
    print("\nExiting due to data loading errors.")
    exit()

# Muat parameter terbaik
params_path = "models/best_params.json"
if not os.path.exists(params_path):
    print(f"Error: Best parameters file not found at {params_path}.")
    print("Please run tune_xgboost_model.py first to generate the parameters.")
    print("Using default XGBoost parameters as a fallback.")
    best_params = {
        'objective': 'multi:softmax',
        'num_class': 3,
        'eval_metric': 'mlogloss',
        'use_label_encoder': False,
        'random_state': 42
    }
else:
    with open(params_path, 'r') as f:
        best_params = json.load(f)
    print(f"\nLoaded best parameters from {params_path}: {best_params}")
    # Pastikan parameter inti ada dan konsisten
    best_params.setdefault('objective', 'multi:softmax')
    best_params.setdefault('num_class', 3)
    best_params.setdefault('eval_metric', 'mlogloss')
    best_params.setdefault('use_label_encoder', False)
    best_params.setdefault('random_state', 42)

# Inisialisasi dan latih model XGBoost dengan parameter terbaik
print("\nInitializing and training XGBoost model with best parameters...")
final_model = xgb.XGBClassifier(**best_params)

final_model.fit(X_train, y_train)
print("Model training finished.")

# Simpan model yang telah dilatih
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, "xgboost_epl_model.json")
final_model.save_model(model_path)
print(f"\nTrained model saved to {model_path}")

# Evaluasi model pada data uji
print("\nEvaluating model on test data...")
y_pred_test = final_model.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f"\nTest Accuracy: {accuracy_test * 100:.2f}%")
print("\nClassification Report on Test Data:")
# Tentukan target names sesuai dengan encoding di preprocess_data.py (0:A, 1:D, 2:H)
target_names_report = ['Away Win (0)', 'Draw (1)', 'Home Win (2)']
print(classification_report(y_test, y_pred_test, target_names=target_names_report, labels=[0,1,2], zero_division=0))

print("\nConfusion Matrix on Test Data:")
cm = confusion_matrix(y_test, y_pred_test, labels=[0,1,2]) # Pastikan labels konsisten
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names_report, yticklabels=target_names_report)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(model_dir, "confusion_matrix_test.png"))
print(f"Confusion matrix plot saved to {os.path.join(model_dir, 'confusion_matrix_test.png')}")
# plt.show() # Komentari jika berjalan di environment non-GUI

# Tampilkan feature importance
print("\nTop 20 Feature Importances:")
if hasattr(final_model, 'feature_importances_') and final_model.feature_importances_ is not None:
    feature_importances = pd.Series(final_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print(feature_importances.nlargest(20))
    
    plt.figure(figsize=(10, 12)) # Disesuaikan agar muat
    feature_importances.nlargest(20).plot(kind='barh')
    plt.title('Top 20 Feature Importances (Tuned XGBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, "feature_importances.png"))
    print(f"Feature importance plot saved to {os.path.join(model_dir, 'feature_importances.png')}")
    # plt.show() # Komentari jika berjalan di environment non-GUI
else:
    print("Could not retrieve feature importances.")

print("\nTraining and evaluation script finished.") 