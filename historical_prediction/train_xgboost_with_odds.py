import pandas as pd
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import numpy as np
import joblib

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

# Baca semua file CSV dan gabungkan
dataframes = []
for file in files:
    try:
        print(f"Reading {file}...")
        df = pd.read_csv(file)
        dataframes.append(df)
        print(f"Successfully read {file}, shape: {df.shape}")
    except Exception as e:
        print(f"Error reading {file}: {e}")

if not dataframes:
    raise ValueError("No data could be loaded from the CSV files.")

df = pd.concat(dataframes, ignore_index=True)

print(f"Dataset shape after loading: {df.shape}")
print(f"Columns: {list(df.columns)}")

# Label encoding untuk semua kolom kategori (object), kecuali target
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'FTR' in categorical_cols:
    categorical_cols.remove('FTR')

for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Target encoding
if 'FTR' in df.columns:
    df['FTR'] = df['FTR'].map({'H': 0, 'D': 1, 'A': 2})
    
    # Check for any unmapped values
    unmapped_count = df['FTR'].isnull().sum()
    if unmapped_count > 0:
        print(f"Warning: {unmapped_count} rows have unmapped FTR values")
        print("Unique FTR values before mapping:", df['FTR'].unique())

# Drop baris dengan target yang tidak valid (jika ada)
df_before = len(df)
df = df[df['FTR'].notnull()]
df_after = len(df)
print(f"Rows removed due to null FTR: {df_before - df_after}")

if len(df) == 0:
    raise ValueError("No valid data remaining after cleaning!")

# Pisahkan fitur dan target
X = df.drop(['FTR'], axis=1)
y = df['FTR']

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Bersihkan nama kolom agar valid untuk XGBoost
X.columns = [str(col).replace('[', '_').replace(']', '_').replace('<', '_').replace('>', '_') for col in X.columns]

# Identifikasi kolom numerik dan non-numerik
numeric_cols = []
non_numeric_cols = []

for col in X.columns:
    try:
        # Try to convert to numeric
        pd.to_numeric(X[col], errors='raise')
        numeric_cols.append(col)
    except (ValueError, TypeError):
        non_numeric_cols.append(col)

print(f"Numeric columns: {len(numeric_cols)}")
print(f"Non-numeric columns: {len(non_numeric_cols)}")
if non_numeric_cols:
    print(f"Non-numeric columns list: {non_numeric_cols[:10]}...")  # Show first 10

# Proses kolom numerik secara batch untuk menghindari fragmentasi
if numeric_cols:
    # Convert all numeric columns at once using pd.concat
    numeric_data = []
    conversion_stats = {}
    
    for col in numeric_cols:
        original_series = X[col]
        converted_series = pd.to_numeric(original_series, errors='coerce')
        nan_count = converted_series.isnull().sum()
        conversion_stats[col] = {
            'original_nulls': original_series.isnull().sum(),
            'after_conversion_nulls': nan_count,
            'conversion_failed': nan_count - original_series.isnull().sum()
        }
        numeric_data.append(converted_series)
    
    # Combine all numeric columns at once
    X_clean = pd.concat(numeric_data, axis=1, keys=numeric_cols)
    
    # Show conversion statistics
    print("Conversion Statistics (top 10 problematic columns):")
    problem_cols = sorted(conversion_stats.items(), 
                         key=lambda x: x[1]['conversion_failed'], 
                         reverse=True)[:10]
    
    for col, stats in problem_cols:
        if stats['conversion_failed'] > 0:
            print(f"  {col}: {stats['conversion_failed']} failed conversions, "
                  f"{stats['after_conversion_nulls']} total NaNs")
            # Show sample values that failed conversion
            failed_mask = (pd.to_numeric(X[col], errors='coerce').isnull()) & (X[col].notnull())
            if failed_mask.any():
                failed_values = X[col][failed_mask].unique()[:5]
                print(f"    Sample failed values: {failed_values}")
else:
    X_clean = pd.DataFrame(index=X.index)

print(f"X_clean shape after numeric conversion: {X_clean.shape}")

# Check NaN distribution across columns
nan_counts = X_clean.isnull().sum()
print(f"Columns with NaN values: {(nan_counts > 0).sum()}")
print(f"Max NaN count in any column: {nan_counts.max()}")

# Remove columns that are all NaN
X_clean = X_clean.dropna(axis=1, how='all')
print(f"X_clean shape after removing all-NaN columns: {X_clean.shape}")

# Check if we have any features left
if X_clean.shape[1] == 0:
    print("No numeric features found! Checking data types...")
    print(X.dtypes.value_counts())
    print("\nSample of non-numeric data:")
    for col in X.columns[:5]:  # Check first 5 columns
        print(f"{col}: {X[col].head().tolist()}")
    raise ValueError("No numeric features available for training!")

# Check how many rows have ANY NaN values
rows_with_nan = X_clean.isnull().any(axis=1).sum()
print(f"Rows with ANY NaN values: {rows_with_nan} out of {len(X_clean)}")

# Check how many rows would remain with different NaN thresholds
for threshold in [0, 0.1, 0.2, 0.5]:
    max_nan_per_row = int(threshold * X_clean.shape[1])
    remaining_rows = (X_clean.isnull().sum(axis=1) <= max_nan_per_row).sum()
    print(f"Rows remaining with ≤{threshold*100}% NaN values: {remaining_rows}")

# Instead of dropping all rows with any NaN, let's be more strategic
print("\nTrying different cleaning strategies:")

# Strategy 1: Remove columns with too many NaN values first
nan_percentage = X_clean.isnull().sum() / len(X_clean)
good_columns = nan_percentage[nan_percentage < 0.5].index  # Keep columns with <50% NaN
X_strategy1 = X_clean[good_columns].copy()
X_strategy1_clean = X_strategy1.dropna(axis=0)
print(f"Strategy 1 (remove columns >50% NaN, then drop NaN rows): {X_strategy1_clean.shape}")

# Strategy 2: Fill NaN values with median
X_strategy2 = X_clean.fillna(X_clean.median())
print(f"Strategy 2 (fill NaN with median): {X_strategy2.shape}")

# Strategy 3: Keep rows with at most 20% NaN values, fill rest with median
max_nan_per_row = int(0.2 * X_clean.shape[1])
good_rows = X_clean.isnull().sum(axis=1) <= max_nan_per_row
X_strategy3 = X_clean[good_rows].fillna(X_clean[good_rows].median())
print(f"Strategy 3 (keep rows ≤20% NaN, fill with median): {X_strategy3.shape}")

# Choose the best strategy
if len(X_strategy1_clean) >= 100:
    X_clean = X_strategy1_clean
    print("Using Strategy 1: Remove high-NaN columns, then drop NaN rows")
elif len(X_strategy3) >= 100:
    X_clean = X_strategy3
    print("Using Strategy 3: Keep good rows, fill NaN with median")
else:
    X_clean = X_strategy2
    print("Using Strategy 2: Fill all NaN with median")

print(f"Final X_clean shape: {X_clean.shape}")

rows_before = len(X_clean)
rows_after = len(X_clean)

# Align target with cleaned features
y_clean = y.loc[X_clean.index]

print(f"Final dataset shape: {X_clean.shape}")
print(f"Final target shape: {y_clean.shape}")

# Check if we have enough data
if len(X_clean) == 0:
    raise ValueError("No data remaining after cleaning!")

if len(X_clean) < 10:
    print(f"Warning: Very small dataset ({len(X_clean)} samples)")

# Check target distribution
print(f"Target distribution:\n{y_clean.value_counts()}")

# Bagi data
# Check if we have enough samples for stratification
min_class_count = y_clean.value_counts().min()
if min_class_count < 2:
    print("Warning: Some classes have less than 2 samples. Disabling stratification.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Training XGBoost
model = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=3, 
    eval_metric='mlogloss', 
    use_label_encoder=False
)

model.fit(X_train, y_train)

# Evaluasi
acc = model.score(X_test, y_test)
print(f"\nAkurasi (dengan odds): {acc*100:.2f}%")

# Additional evaluation metrics
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Home Win', 'Draw', 'Away Win']))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained model for later use
# Create a models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the model
model_path = 'models/xgboost_with_odds.pkl'
joblib.dump(model, model_path)
print(f"\nModel saved to {model_path}")

# Save feature names for later use
feature_names_path = 'models/feature_names.pkl'
joblib.dump(list(X_clean.columns), feature_names_path)
print(f"Feature names saved to {feature_names_path}")

# Save label encoder for future predictions if needed
encoders = {}
for col in categorical_cols:
    # Create a new LabelEncoder and fit it on the data
    le = LabelEncoder()
    le.fit(df[col].astype(str).unique())
    encoders[col] = le

encoders_path = 'models/label_encoders.pkl'
joblib.dump(encoders, encoders_path)
print(f"Label encoders saved to {encoders_path}")

print("\nTraining and evaluation completed successfully.")
