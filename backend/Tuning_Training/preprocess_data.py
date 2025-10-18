import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib # Tambahkan import joblib
import os # Untuk membuat direktori jika belum ada

# Load the dataset
file_path = "Combined Dataset/combined_epl_data.csv"
df = pd.read_csv(file_path)

print("Original Dataset Shape:")
print(df.shape)

# Convert 'Date' column to datetime and sort df
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
df = df.sort_values(by=['Date', 'Time']).reset_index(drop=True) # Sort early by Date and Time

# Define a more manageable subset of columns for initial analysis
selected_columns = [
    'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', # Match Info & Result
    'HTHG', 'HTAG', 'HTR', # Halftime Info
    'Referee', # Referee Info
    'HS', 'AS', 'HST', 'AST', # Shots
    'HF', 'AF', # Fouls
    'HC', 'AC', # Corners
    'HY', 'AY', 'HR', 'AR', # Cards
    'B365H', 'B365D', 'B365A' # Basic Odds from Bet365
]

# df_selected = df[selected_columns].copy()
# print("\nSelected DataFrame Shape:")
# print(df_selected.shape)
# print("\nMissing Values in Selected DataFrame:")
# print(df_selected.isnull().sum().sort_values(ascending=False).head())

# --- Basic Feature Engineering (Encoders, simple result-based features) ---
# These are created first, some might be excluded later if they are direct leakage for the final model

# 1. Encode FTR (Full Time Result) and HTR (Half Time Result)
result_mapping = {'H': 2, 'D': 1, 'A': 0}
df['FTR_encoded'] = df['FTR'].map(result_mapping)
df['HTR_encoded'] = df['HTR'].map(result_mapping) # HTR_encoded will be excluded later as per user preference

# 2. Goal Difference (GD) and Total Goals (TG) - potential leakage, will be excluded
df['GD'] = df['FTHG'] - df['FTAG']
df['TG'] = df['FTHG'] + df['FTAG']

# 3. Label Encoding for categorical features
categorical_cols = ['HomeTeam', 'AwayTeam', 'Referee']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le # Store the encoder

# Pastikan direktori untuk menyimpan encoders ada
output_dir_for_encoders = "Combined Dataset"
if not os.path.exists(output_dir_for_encoders):
    os.makedirs(output_dir_for_encoders)

# Simpan label encoders
joblib.dump(label_encoders, os.path.join(output_dir_for_encoders, 'label_encoders.joblib'))
print(f"\nLabel encoders saved to {os.path.join(output_dir_for_encoders, 'label_encoders.joblib')}")

# --- Advanced Feature Engineering: Historical Team Stats ---
print("\nStarting historical feature engineering...")

# Calculate points for each game (used for form calculation)
df['HomePoints'] = df['FTR'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
df['AwayPoints'] = df['FTR'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

# Prepare a long-form DataFrame for team-specific rolling calculations
home_stats_base = df[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'HomePoints']].copy()
home_stats_base.rename(columns={'HomeTeam': 'Team', 'FTHG': 'GS', 'FTAG': 'GC', 'HomePoints': 'Points'}, inplace=True)

away_stats_base = df[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'AwayPoints']].copy()
away_stats_base.rename(columns={'AwayTeam': 'Team', 'FTAG': 'GS', 'FTHG': 'GC', 'AwayPoints': 'Points'}, inplace=True)

team_game_stats_base = pd.concat([home_stats_base, away_stats_base]).sort_values(by=['Team', 'Date']).reset_index(drop=True)

# Calculate rolling stats (Average Goals Scored, Goals Conceded, Form Points over last 3, 5, and 10 games)
window_sizes = [3, 5, 10] # Define multiple window sizes

for window_size in window_sizes:
    print(f"Calculating rolling stats for window size: {window_size}")
    team_game_stats_base[f'AvgGS_L{window_size}'] = team_game_stats_base.groupby('Team')['GS'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    team_game_stats_base[f'AvgGC_L{window_size}'] = team_game_stats_base.groupby('Team')['GC'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    team_game_stats_base[f'Form_Points_L{window_size}'] = team_game_stats_base.groupby('Team')['Points'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).sum()
    )

# Merge these rolling stats back to the main df
# Need a unique game identifier in df if not already present (reset_index earlier helps)
df['game_id_temp'] = df.index

# Columns to merge from team_game_stats_base
stats_cols_to_merge = []
for ws in window_sizes:
    stats_cols_to_merge.extend([f'AvgGS_L{ws}', f'AvgGC_L{ws}', f'Form_Points_L{ws}'])

# For HomeTeam stats
home_merge_df_cols = ['Date', 'Team'] + stats_cols_to_merge
home_merge_df = team_game_stats_base[home_merge_df_cols].copy()
home_merge_df.rename(columns={'Team': 'HomeTeam'}, inplace=True)
for col_name in stats_cols_to_merge:
    home_merge_df.rename(columns={col_name: f'HomeTeam_{col_name}'}, inplace=True)
home_merge_df = home_merge_df.drop_duplicates(subset=['Date', 'HomeTeam'])
df = pd.merge(df, home_merge_df, on=['Date', 'HomeTeam'], how='left')

# For AwayTeam stats
away_merge_df_cols = ['Date', 'Team'] + stats_cols_to_merge
away_merge_df = team_game_stats_base[away_merge_df_cols].copy()
away_merge_df.rename(columns={'Team': 'AwayTeam'}, inplace=True)
for col_name in stats_cols_to_merge:
    away_merge_df.rename(columns={col_name: f'AwayTeam_{col_name}'}, inplace=True)
away_merge_df = away_merge_df.drop_duplicates(subset=['Date', 'AwayTeam'])
df = pd.merge(df, away_merge_df, on=['Date', 'AwayTeam'], how='left')

# Fill NaNs for all historical stat columns
all_historical_stat_cols = []
for ws in window_sizes:
    all_historical_stat_cols.extend([
        f'HomeTeam_AvgGS_L{ws}', f'HomeTeam_AvgGC_L{ws}', f'HomeTeam_Form_Points_L{ws}',
        f'AwayTeam_AvgGS_L{ws}', f'AwayTeam_AvgGC_L{ws}', f'AwayTeam_Form_Points_L{ws}'
    ])
for col in all_historical_stat_cols:
    if col in df.columns: # Check if column exists after merge
        df[col].fillna(0, inplace=True)
    else:
        print(f"Warning: Expected historical column {col} not found after merge.")

print("Historical feature engineering complete for multiple window sizes.")

# --- Clean ALL DataFrame column names for XGBoost compatibility ---
def clean_feature_name(col_name):
    new_name = str(col_name).replace('<', '_lt_').replace('>', '_gt_').replace('=', '_eq_')
    new_name = new_name.replace('[', '_').replace(']', '_').replace(' ', '_')
    # Basic regex to remove other special chars, leaving alphanumeric and underscore
    new_name = "".join(char for char in new_name if char.isalnum() or char == '_')
    new_name = new_name.strip('_') # Remove leading/trailing underscores
    return new_name

df.columns = [clean_feature_name(col) for col in df.columns]
print("\nAll df column names cleaned (first 15):")
print(df.columns.tolist()[:15])


# --- Feature Selection and Target Definition ---
target = 'FTR_encoded' # This name is already clean

features_to_exclude = [
    'FTR', 'HTR', # Original result strings
    'FTHG', 'FTAG', 'GD', 'TG', # Leakage / Direct full-time result indicators
    'HTHG', 'HTAG', 'HTR_encoded',# Halftime scores (user preference to exclude)
    'HomeTeam', 'AwayTeam', 'Referee', # Original categorical columns (using encoded versions)
    'Date', # Date is used for sorting/context but not as a direct model feature here
    'Div', 'Time', # Columns previously identified as problematic or unneeded
    'HomePoints', 'AwayPoints', # Intermediate point calculation columns, direct leakage if used
    'game_id_temp' # Temporary ID
]
# Ensure all names in features_to_exclude are also in their "cleaned" form if they were complex
# For this list, they are simple and clean_feature_name wouldn't change them.

# Select all columns that are NOT in features_to_exclude and also NOT the target
# All df.columns are now cleaned, so 'features' list will contain cleaned names.
features = [col for col in df.columns if col not in features_to_exclude and col != target and target in df.columns]

print(f"\nTarget variable: {target}")
print(f"\nSelected features for the model ({len(features)}):")
# print(features) # This list can be very long

X = df[features].copy()
y = df[target].copy()

# Handle potential NaN values in final features (e.g., from betting odds if not covered by historical fillna)
# The main NaN filling for odds columns should happen here.
print("\nFilling NaNs in final feature set X...")
for col in X.columns:
    if X[col].isnull().any():
        # print(f"Filling NaNs in feature column: {col} with 0") # Can be verbose
        X[col] = X[col].fillna(0)

if y.isnull().any():
    print("Warning: NaN values found in target variable! Check data integrity.")

# --- Data Splitting ---
# df is already sorted by Date. We use the 'features' list which has cleaned names.
# X and y are already derived from the (potentially re-indexed) df.
# Re-derive X_sorted and y_sorted from the main df to ensure correct chronological order from the start.
df_sorted_for_split = df.sort_values(by='Date') # Re-sort just to be absolutely sure before split
X_final_sorted = df_sorted_for_split[features]
y_final_sorted = df_sorted_for_split[target]


X_train, X_test, y_train, y_test = train_test_split(
    X_final_sorted, y_final_sorted, test_size=0.2, shuffle=False
)

print("\nShape of Training data (X_train, y_train):")
print(X_train.shape, y_train.shape)
print("Shape of Testing data (X_test, y_test):")
print(X_test.shape, y_test.shape)
print("\nFirst 10 X_train columns (post multiple historical FE & cleaning):")
print(X_train.columns.tolist()[:10])


# --- Save Processed Data ---
output_dir = "Combined Dataset"
X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
y_train.to_csv(f"{output_dir}/y_train.csv", index=False, header=True)
X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
y_test.to_csv(f"{output_dir}/y_test.csv", index=False, header=True)

print(f"\nProcessed data (X_train, y_train, X_test, y_test) saved to '{output_dir}' directory.")
print("\nPreprocessing script finished.") 