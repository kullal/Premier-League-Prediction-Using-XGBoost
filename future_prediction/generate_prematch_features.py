import pandas as pd
import glob
import numpy as np
import os
from tqdm import tqdm
import time

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
        files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv') and 'EPL 202' in f]
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
                files = [os.path.join(alt_path, f) for f in os.listdir(alt_path) if f.endswith('.csv') and 'EPL 202' in f]
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

# Pastikan urut berdasarkan tanggal
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.sort_values('Date').reset_index(drop=True)
    print(f"Data range: {df['Date'].min()} to {df['Date'].max()}")

# Fungsi untuk menghitung statistik per tim
def calculate_team_stats(df, window=5):
    """Calculate comprehensive team statistics with a sliding window"""
    print(f"Calculating team statistics with {window}-match window...")
    
    # Mendapatkan list semua tim
    teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
    print(f"Processing {len(teams)} teams...")
    
    # Dictionary untuk menyimpan statistik semua tim
    team_stats = {}
    
    # Inisialisasi dictionary statistik untuk setiap tim
    for team in teams:
        team_stats[team] = {
            'matches': [],
            'dates': [],
            'GF': [],
            'GA': [],
            'result': [],
            'shots': [],
            'shots_target': [],
            'corners': [],
            'fouls': [],
            'yellows': [],
            'reds': []
        }
    
    # Populate statistics for each team from all matches
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing matches"):
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        match_date = row['Date']
        
        # Get match data
        home_goals = row['FTHG'] if 'FTHG' in row else 0
        away_goals = row['FTAG'] if 'FTAG' in row else 0
        result = row['FTR'] if 'FTR' in row else 'N/A'
        
        # Home team stats
        team_stats[home_team]['matches'].append('H')
        team_stats[home_team]['dates'].append(match_date)
        team_stats[home_team]['GF'].append(home_goals)
        team_stats[home_team]['GA'].append(away_goals)
        team_stats[home_team]['result'].append(result)
        
        # Additional stats if available
        if 'HS' in row: team_stats[home_team]['shots'].append(row['HS'])
        if 'HST' in row: team_stats[home_team]['shots_target'].append(row['HST'])
        if 'HC' in row: team_stats[home_team]['corners'].append(row['HC'])
        if 'HF' in row: team_stats[home_team]['fouls'].append(row['HF'])
        if 'HY' in row: team_stats[home_team]['yellows'].append(row['HY'])
        if 'HR' in row: team_stats[home_team]['reds'].append(row['HR'])
        
        # Away team stats
        team_stats[away_team]['matches'].append('A')
        team_stats[away_team]['dates'].append(match_date)
        team_stats[away_team]['GF'].append(away_goals)
        team_stats[away_team]['GA'].append(home_goals)
        team_stats[away_team]['result'].append(result)
        
        # Additional stats if available
        if 'AS' in row: team_stats[away_team]['shots'].append(row['AS'])
        if 'AST' in row: team_stats[away_team]['shots_target'].append(row['AST'])
        if 'AC' in row: team_stats[away_team]['corners'].append(row['AC'])
        if 'AF' in row: team_stats[away_team]['fouls'].append(row['AF'])
        if 'AY' in row: team_stats[away_team]['yellows'].append(row['AY'])
        if 'AR' in row: team_stats[away_team]['reds'].append(row['AR'])
    
    return team_stats, teams

# Calculate head-to-head statistics
def calculate_h2h_stats(df, max_matches=10):
    """Calculate head-to-head statistics between teams"""
    print("Calculating head-to-head statistics...")
    
    # Create a dictionary to store H2H data
    h2h_stats = {}
    
    # Process each team pair
    team_pairs = set()
    for _, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']
        pair = tuple(sorted([home, away]))
        team_pairs.add(pair)
    
    print(f"Processing {len(team_pairs)} team pairs...")
    
    # Calculate H2H for each pair
    for team1, team2 in tqdm(team_pairs, desc="Processing team pairs"):
        h2h_matches = df[((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | 
                         ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))]
        
        h2h_matches = h2h_matches.sort_values('Date')
        
        if len(h2h_matches) > 0:
            h2h_stats[(team1, team2)] = {
                'matches': len(h2h_matches),
                'team1_wins': sum((h2h_matches['HomeTeam'] == team1) & (h2h_matches['FTR'] == 'H')) + 
                             sum((h2h_matches['AwayTeam'] == team1) & (h2h_matches['FTR'] == 'A')),
                'team2_wins': sum((h2h_matches['HomeTeam'] == team2) & (h2h_matches['FTR'] == 'H')) + 
                             sum((h2h_matches['AwayTeam'] == team2) & (h2h_matches['FTR'] == 'A')),
                'draws': sum(h2h_matches['FTR'] == 'D'),
                'last_matches': h2h_matches.tail(max_matches),
                'avg_goals': h2h_matches['FTHG'].mean() + h2h_matches['FTAG'].mean()
            }
    
    return h2h_stats

# Calculate team statistics and H2H data
team_stats, teams = calculate_team_stats(df)
h2h_stats = calculate_h2h_stats(df)

# Buat fitur pre-match yang akan digunakan untuk prediksi
window = 5  # rolling window 5 match terakhir
features = []

print("Generating pre-match features for each match...")
for idx, row in tqdm(df.iterrows(), total=len(df), desc="Creating features"):
    home = row['HomeTeam']
    away = row['AwayTeam']
    date = row['Date']
    
    # Get team indices in their respective stats lists
    home_indices = [i for i, d in enumerate(team_stats[home]['dates']) if d < date]
    away_indices = [i for i, d in enumerate(team_stats[away]['dates']) if d < date]
    
    # Take only last n (window) indices
    if home_indices:
        home_indices = home_indices[-window:] if len(home_indices) > window else home_indices
    if away_indices:
        away_indices = away_indices[-window:] if len(away_indices) > window else away_indices
    
    # Calculate home team stats
    home_gf = [team_stats[home]['GF'][i] for i in home_indices] if home_indices else []
    home_ga = [team_stats[home]['GA'][i] for i in home_indices] if home_indices else []
    home_shots = [team_stats[home]['shots'][i] for i in home_indices] if home_indices and team_stats[home]['shots'] else []
    home_shots_target = [team_stats[home]['shots_target'][i] for i in home_indices] if home_indices and team_stats[home]['shots_target'] else []
    
    # Calculate form based on results
    home_w = home_d = home_l = 0
    for i in home_indices:
        loc = team_stats[home]['matches'][i]
        res = team_stats[home]['result'][i]
        if loc == 'H':  # Home game
            if res == 'H': home_w += 1
            elif res == 'D': home_d += 1
            elif res == 'A': home_l += 1
        else:  # Away game
            if res == 'A': home_w += 1
            elif res == 'D': home_d += 1
            elif res == 'H': home_l += 1
    
    # Calculate away team stats
    away_gf = [team_stats[away]['GF'][i] for i in away_indices] if away_indices else []
    away_ga = [team_stats[away]['GA'][i] for i in away_indices] if away_indices else []
    away_shots = [team_stats[away]['shots'][i] for i in away_indices] if away_indices and team_stats[away]['shots'] else []
    away_shots_target = [team_stats[away]['shots_target'][i] for i in away_indices] if away_indices and team_stats[away]['shots_target'] else []
    
    # Calculate form based on results
    away_w = away_d = away_l = 0
    for i in away_indices:
        loc = team_stats[away]['matches'][i]
        res = team_stats[away]['result'][i]
        if loc == 'A':  # Away game
            if res == 'A': away_w += 1
            elif res == 'D': away_d += 1
            elif res == 'H': away_l += 1
        else:  # Home game
            if res == 'H': away_w += 1
            elif res == 'D': away_d += 1
            elif res == 'A': away_l += 1
    
    # Get H2H data
    pair = tuple(sorted([home, away]))
    h2h_data = h2h_stats.get(pair, {
        'matches': 0,
        'team1_wins': 0,
        'team2_wins': 0,
        'draws': 0,
        'avg_goals': 0
    })
    
    # Calculate H2H stats from home team perspective
    if pair[0] == home:
        h2h_home_wins = h2h_data['team1_wins']
        h2h_away_wins = h2h_data['team2_wins']
    else:
        h2h_home_wins = h2h_data['team2_wins']
        h2h_away_wins = h2h_data['team1_wins']
    
    h2h_draws = h2h_data['draws']
    h2h_matches = h2h_data['matches']
    
    # Create feature dictionary
    feature = {
        'Date': date,
        'HomeTeam': home,
        'AwayTeam': away,
        'HomeAvgGF': np.mean(home_gf) if home_gf else np.nan,
        'HomeAvgGA': np.mean(home_ga) if home_ga else np.nan,
        'HomeAvgShots': np.mean(home_shots) if home_shots else np.nan,
        'HomeAvgShotsTarget': np.mean(home_shots_target) if home_shots_target else np.nan,
        'HomeW': home_w,
        'HomeD': home_d,
        'HomeL': home_l,
        'HomePoints': home_w * 3 + home_d,
        'HomeForm': (home_w * 3 + home_d) / (len(home_indices) * 3) if home_indices else np.nan,
        'AwayAvgGF': np.mean(away_gf) if away_gf else np.nan,
        'AwayAvgGA': np.mean(away_ga) if away_ga else np.nan,
        'AwayAvgShots': np.mean(away_shots) if away_shots else np.nan,
        'AwayAvgShotsTarget': np.mean(away_shots_target) if away_shots_target else np.nan,
        'AwayW': away_w,
        'AwayD': away_d,
        'AwayL': away_l,
        'AwayPoints': away_w * 3 + away_d,
        'AwayForm': (away_w * 3 + away_d) / (len(away_indices) * 3) if away_indices else np.nan,
        'H2H_Matches': h2h_matches,
        'H2H_HomeWinPct': h2h_home_wins / h2h_matches if h2h_matches > 0 else np.nan,
        'H2H_DrawPct': h2h_draws / h2h_matches if h2h_matches > 0 else np.nan,
        'H2H_AwayWinPct': h2h_away_wins / h2h_matches if h2h_matches > 0 else np.nan,
        'H2H_AvgGoals': h2h_data['avg_goals'] if h2h_matches > 0 else np.nan,
        'FTR': row['FTR']
    }
    
    features.append(feature)

# Simpan ke CSV
print("Creating DataFrame and saving to CSV...")
prematch_df = pd.DataFrame(features)

# Fill NaN values with appropriate defaults
print("Filling missing values...")
for col in prematch_df.columns:
    if col not in ['Date', 'HomeTeam', 'AwayTeam', 'FTR']:
        if 'Pct' in col or 'Form' in col:
            # Default to 0.5 for percentages and form
            prematch_df[col] = prematch_df[col].fillna(0.5)
        else:
            # Use column median for other stats
            prematch_df[col] = prematch_df[col].fillna(prematch_df[col].median())

# Pastikan folder data ada
os.makedirs('data', exist_ok=True)
prematch_df.to_csv('data/prematch_features.csv', index=False)
print('Enhanced prematch features saved to data/prematch_features.csv')

# Generate some basic statistics about the features
print("\nFeature Statistics:")
print(f"Total matches: {len(prematch_df)}")
print(f"Unique teams: {len(teams)}")
print(f"Date range: {prematch_df['Date'].min()} to {prematch_df['Date'].max()}")
print(f"Features generated: {len(prematch_df.columns) - 4}")  # Exclude Date, HomeTeam, AwayTeam, FTR

# Show features distribution by outcome
outcome_distribution = prematch_df['FTR'].value_counts()
print("\nOutcome Distribution:")
for outcome, count in outcome_distribution.items():
    pct = count / len(prematch_df) * 100
    print(f"{outcome}: {count} ({pct:.1f}%)") 