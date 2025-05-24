import pandas as pd
import glob
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Print current working directory for debugging
current_dir = os.getcwd()
print(f"Current working directory: {current_dir}")

# Use relative path from script location
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
data_dir = os.path.join(project_root, 'data', 'No Odds Dataset')
print(f"Looking for CSV files in: {os.path.abspath(data_dir)}")

# Mendapatkan list file secara manual tanpa menggunakan glob karena terkendala spasi
try:
    if os.path.exists(data_dir):
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('_no_odds.csv')]
    else:
        files = []
except Exception as e:
    print(f"Error listing directory: {e}")
    files = []

print(f"Found {len(files)} dataset files: {files}")

# If no files found, try potential alternative paths
if not files:
    alternative_paths = [
        os.path.join(project_root, 'data', 'No Odds Dataset'),
        os.path.join(current_dir, 'data', 'No Odds Dataset'),
        os.path.join(os.path.dirname(current_dir), 'data', 'No Odds Dataset'),
        os.path.abspath(os.path.join('..', '..', 'data', 'No Odds Dataset')),
        os.path.abspath(os.path.join('..', 'data', 'No Odds Dataset')),
        os.path.abspath('data/No Odds Dataset')
    ]
    
    for alt_path in alternative_paths:
        print(f"Trying alternative path: {alt_path}")
        try:
            if os.path.exists(alt_path):
                files = [os.path.join(alt_path, f) for f in os.listdir(alt_path) if f.endswith('_no_odds.csv')]
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

# Baca semua file CSV dan gabungkan
dataframes = []
for file in files:
    try:
        print(f"Reading {file}...")
        df_temp = pd.read_csv(file)
        dataframes.append(df_temp)
        print(f"Successfully read {file}, shape: {df_temp.shape}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dataframes:
    raise ValueError("No data could be loaded from the CSV files.")

df = pd.concat(dataframes, ignore_index=True)
print(f"Combined dataset shape: {df.shape}")

# Urutkan data berdasarkan tanggal agar rolling statistic benar
# Asumsi format tanggal: dd/mm/yyyy
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values('Date').reset_index(drop=True)

# Label encoding untuk kolom kategori
for col in ['HomeTeam', 'AwayTeam', 'Referee', 'Div']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Target encoding
if 'FTR' in df.columns:
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})

# Fungsi untuk menghitung statistik dengan window tertentu
def rolling_avg(df, team_col, goal_col, new_col, window=5, min_periods=1):
    """Calculate rolling average stats for teams"""
    df[new_col] = np.nan
    for team in df[team_col].unique():
        idx = df[df[team_col] == team].index
        df.loc[idx, new_col] = df.loc[idx, goal_col].shift(1).rolling(window=window, min_periods=min_periods).mean()
    return df

def rolling_form(df, window=5):
    """Calculate recent form (points from last matches) for teams"""
    # Create form columns
    df['HomeTeam_form'] = np.nan
    df['AwayTeam_form'] = np.nan
    
    # Points for each result: win=3, draw=1, loss=0
    for team in df['HomeTeam'].unique():
        # Home form when playing at home
        home_matches = df[(df['HomeTeam'] == team)]
        
        if len(home_matches) > 0:
            home_points = home_matches['FTR'].map({0: 3, 1: 1, 2: 0})
            for i, idx in enumerate(home_matches.index):
                if i == 0:
                    df.loc[idx, 'HomeTeam_form'] = 0
                else:
                    # Calculate form using last n matches
                    prev_idx = home_matches.index[:i]
                    if len(prev_idx) > window:
                        prev_idx = prev_idx[-window:]
                    df.loc[idx, 'HomeTeam_form'] = home_points.loc[prev_idx].mean()
        
        # Away form when playing away
        away_matches = df[(df['AwayTeam'] == team)]
        
        if len(away_matches) > 0:
            away_points = away_matches['FTR'].map({0: 0, 1: 1, 2: 3})
            for i, idx in enumerate(away_matches.index):
                if i == 0:
                    df.loc[idx, 'AwayTeam_form'] = 0
                else:
                    # Calculate form using last n matches
                    prev_idx = away_matches.index[:i]
                    if len(prev_idx) > window:
                        prev_idx = prev_idx[-window:]
                    df.loc[idx, 'AwayTeam_form'] = away_points.loc[prev_idx].mean()
                    
    return df

def head_to_head(df):
    """Calculate head-to-head statistics between teams"""
    # Create H2H columns
    df['H2H_HomeWin'] = np.nan
    df['H2H_Draw'] = np.nan
    df['H2H_AwayWin'] = np.nan
    
    # Process each match
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        match_date = match['Date']
        
        # Get previous meetings between these teams
        prev_h2h = df[((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) | 
                       (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) & 
                      (df['Date'] < match_date)]
        
        if len(prev_h2h) > 0:
            # Count results from home team perspective
            home_win = 0
            draw = 0
            away_win = 0
            
            for _, h2h in prev_h2h.iterrows():
                if h2h['HomeTeam'] == home_team:
                    # Home team was at home in previous meeting
                    if h2h['FTR'] == 0:
                        home_win += 1
                    elif h2h['FTR'] == 1:
                        draw += 1
                    else:
                        away_win += 1
                else:
                    # Home team was away in previous meeting
                    if h2h['FTR'] == 2:
                        home_win += 1
                    elif h2h['FTR'] == 1:
                        draw += 1
                    else:
                        away_win += 1
            
            total_matches = len(prev_h2h)
            df.loc[idx, 'H2H_HomeWin'] = home_win / total_matches
            df.loc[idx, 'H2H_Draw'] = draw / total_matches
            df.loc[idx, 'H2H_AwayWin'] = away_win / total_matches
        else:
            # No previous meetings, use default
            df.loc[idx, 'H2H_HomeWin'] = 0.5
            df.loc[idx, 'H2H_Draw'] = 0.25
            df.loc[idx, 'H2H_AwayWin'] = 0.25
    
    return df

# Fitur berbasis statistik tim
print("Menghitung statistik rata-rata tim...")

# Rata-rata gol home (FTHG) untuk HomeTeam
if 'HomeTeam' in df.columns and 'FTHG' in df.columns:
    df = rolling_avg(df, 'HomeTeam', 'FTHG', 'HomeTeam_avg_FTHG')
    df = rolling_avg(df, 'HomeTeam', 'FTAG', 'HomeTeam_avg_goals_conceded')

# Rata-rata gol away (FTAG) untuk AwayTeam
if 'AwayTeam' in df.columns and 'FTAG' in df.columns:
    df = rolling_avg(df, 'AwayTeam', 'FTAG', 'AwayTeam_avg_FTAG')
    df = rolling_avg(df, 'AwayTeam', 'FTHG', 'AwayTeam_avg_goals_conceded')

# Tembakan dan tembakan tepat sasaran
if 'HomeTeam' in df.columns and 'HS' in df.columns:
    df = rolling_avg(df, 'HomeTeam', 'HS', 'HomeTeam_avg_shots')
    df = rolling_avg(df, 'AwayTeam', 'AS', 'AwayTeam_avg_shots')
    
    if 'HST' in df.columns:
        df = rolling_avg(df, 'HomeTeam', 'HST', 'HomeTeam_avg_shots_target')
        df = rolling_avg(df, 'AwayTeam', 'AST', 'AwayTeam_avg_shots_target')
        
        # Shot accuracy (shots on target / total shots)
        df['HomeTeam_shot_accuracy'] = df['HomeTeam_avg_shots_target'] / df['HomeTeam_avg_shots']
        df['AwayTeam_shot_accuracy'] = df['AwayTeam_avg_shots_target'] / df['AwayTeam_avg_shots']
        
        # Replace infinite values with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Form calculation
print("Menghitung form tim...")
df = rolling_form(df)

# Head-to-head statistics
print("Menghitung statistik head-to-head...")
df = head_to_head(df)

# Goal difference
if 'FTHG' in df.columns and 'FTAG' in df.columns:
    df['GD'] = df['FTHG'] - df['FTAG']
    df = rolling_avg(df, 'HomeTeam', 'GD', 'HomeTeam_avg_GD')
    df = rolling_avg(df, 'AwayTeam', 'GD', 'AwayTeam_avg_GD', window=5, min_periods=1)

# Hanya gunakan fitur yang diketahui sebelum pertandingan
fitur_prediksi = [
    'HomeTeam', 'AwayTeam', 'Referee', 'Div',
    'HomeTeam_avg_FTHG', 'HomeTeam_avg_goals_conceded',
    'AwayTeam_avg_FTAG', 'AwayTeam_avg_goals_conceded',
    'HomeTeam_form', 'AwayTeam_form',
    'H2H_HomeWin', 'H2H_Draw', 'H2H_AwayWin',
]

# Tambahkan fitur tambahan jika tersedia
optional_features = [
    'HomeTeam_avg_shots', 'AwayTeam_avg_shots',
    'HomeTeam_avg_shots_target', 'AwayTeam_avg_shots_target',
    'HomeTeam_shot_accuracy', 'AwayTeam_shot_accuracy',
    'HomeTeam_avg_GD', 'AwayTeam_avg_GD'
]

for feature in optional_features:
    if feature in df.columns:
        fitur_prediksi.append(feature)

# Buat dataset fitur dan target
X = df[fitur_prediksi]
y = df['FTR']

# Drop baris yang masih ada NaN (karena rolling mean awal)
X = X.dropna()
y = y.loc[X.index]

print(f"Dataset siap untuk training: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
print("Melakukan hyperparameter tuning...")
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
accuracy = model.score(X_test, y_test)
print(f'Akurasi (realistis): {accuracy:.2%}')

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
plt.savefig('predictions/confusion_matrix_realistis.png')
print("Confusion matrix visualization saved to 'predictions/confusion_matrix_realistis.png'")

# Feature importance
plt.figure(figsize=(10, 8))
feature_importance = model.feature_importances_
indices = np.argsort(feature_importance)[::-1]
plt.barh(range(len(indices)), feature_importance[indices])
plt.yticks(range(len(indices)), [X.columns[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('XGBoost Feature Importance')
plt.tight_layout()
plt.savefig('predictions/feature_importance_realistis.png')
print("Feature importance visualization saved to 'predictions/feature_importance_realistis.png'")

# Simpan model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_realistis.pkl')
print(f'Model saved to models/xgboost_realistis.pkl')

# Simpan feature names juga
joblib.dump(list(X.columns), 'models/feature_names_realistis.pkl')
print(f'Feature names saved to models/feature_names_realistis.pkl') 