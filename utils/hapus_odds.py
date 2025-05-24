import pandas as pd
import os

# Folder dataset
folder = '../data/Dataset EPL New'

# Daftar file dataset EPL
files = [
    'EPL 2022-2023.csv',
    'EPL 2023-2024.csv',
    'EPL 2024-2025.csv',
]

# Kolom yang ingin dipertahankan (dari notes.txt)
keep_columns = [
    'Div', 'Date', 'Time', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
    'Attendance', 'Referee', 'HS', 'AS', 'HST', 'AST', 'HHW', 'AHW', 'HC', 'AC', 'HF', 'AF',
    'HFKC', 'AFKC', 'HO', 'AO', 'HY', 'AY', 'HR', 'AR', 'HBP', 'ABP'
]

for file in files:
    path = os.path.join(folder, file)
    df = pd.read_csv(path)
    # Hanya ambil kolom yang ada di keep_columns dan memang ada di file
    keep = [col for col in keep_columns if col in df.columns]
    df = df[keep]
    out_path = os.path.join('../data/No Odds Dataset', file.replace('.csv', '_no_odds.csv'))
    df.to_csv(out_path, index=False)
    print(f"Selesai: {file} -> {out_path}")