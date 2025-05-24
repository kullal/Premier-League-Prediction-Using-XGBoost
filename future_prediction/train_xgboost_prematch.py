import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import joblib

print("Loading enhanced prematch features...")
# Load data
prematch_df = pd.read_csv('data/prematch_features.csv')
print(f"Loaded dataset with {prematch_df.shape[0]} matches and {prematch_df.shape[1]} features")

# Label encoding untuk tim
print("Encoding categorical features...")
for col in ['HomeTeam', 'AwayTeam']:
    prematch_df[col] = LabelEncoder().fit_transform(prematch_df[col].astype(str))

# Target encoding
prematch_df = prematch_df[prematch_df['FTR'].notnull()]
prematch_df['FTR'] = prematch_df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Drop baris yang ada NaN di fitur (sudah diisi dengan nilai default oleh generate_prematch_features.py)
prematch_df = prematch_df.dropna()

# Memisahkan fitur dan target
print("Preparing features and target...")
feature_cols = [col for col in prematch_df.columns if col not in ['FTR', 'Date']]
X = prematch_df[feature_cols]
y = prematch_df['FTR']

print(f"Feature set: {X.shape}")
print(f"Target distribution:\n{y.value_counts(normalize=True).map(lambda x: f'{x:.1%}')}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Hyperparameter tuning (opsional)
print("Training XGBoost model...")
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2],
    'n_estimators': [100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', use_label_encoder=False)

# Uncomment jika ingin melakukan grid search (akan memakan waktu lama)
# grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', verbose=1)
# grid_search.fit(X_train, y_train)
# best_params = grid_search.best_params_
# print(f"Best parameters: {best_params}")
# model = grid_search.best_estimator_

# Gunakan parameter default atau parameter terbaik dari grid search
model.fit(X_train, y_train)

# Evaluasi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAkurasi (prematch features): {accuracy*100:.2f}%")

# Tampilkan classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
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
os.makedirs('predictions', exist_ok=True)
plt.savefig('predictions/confusion_matrix_prematch.png')
print("Confusion matrix visualization saved to 'predictions/confusion_matrix_prematch.png'")

# Feature importance
plt.figure(figsize=(12, 8))
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1]
top_n = min(20, len(indices))  # Show top 20 features
plt.barh(range(top_n), feature_importance[indices[:top_n]])
plt.yticks(range(top_n), [X.columns[i] for i in indices[:top_n]])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig('predictions/feature_importance_prematch.png')
print("Feature importance visualization saved to 'predictions/feature_importance_prematch.png'")

# Simpan model untuk prediksi di masa depan
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_prematch.pkl')
print(f"Model saved to models/xgboost_prematch.pkl")

# Simpan feature names juga
joblib.dump(list(X.columns), 'models/feature_names_prematch.pkl')
print(f"Feature names saved to models/feature_names_prematch.pkl") 

# Print model summary
print("\nModel Summary:")
print(f"- Model type: XGBoost Classifier")
print(f"- Number of features: {X.shape[1]}")
print(f"- Number of estimators: {model.get_params()['n_estimators']}")
print(f"- Learning rate: {model.get_params()['learning_rate']}")
print(f"- Max depth: {model.get_params()['max_depth']}")
print(f"- Accuracy: {accuracy*100:.2f}%")

# Print prediction probabilities example
print("\nExample prediction (first 3 test samples):")
probas = model.predict_proba(X_test.iloc[:3])
for i, (pred, proba) in enumerate(zip(y_pred[:3], probas[:3])):
    outcome = ["Home Win", "Draw", "Away Win"][int(pred)]
    actual = ["Home Win", "Draw", "Away Win"][int(y_test.iloc[i])]
    print(f"Sample {i+1}: Predicted {outcome} (Actual: {actual})")
    print(f"  Home Win: {proba[0]:.2f}, Draw: {proba[1]:.2f}, Away Win: {proba[2]:.2f}") 