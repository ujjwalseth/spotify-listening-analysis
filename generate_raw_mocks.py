import pandas as pd
import glob
import os

# Create a folder for the raw mocks
os.makedirs('data/raw_exports', exist_ok=True)

# Columns that represent the "Raw" data you get from Spotify Privacy Export
raw_columns = [
    'member_id', 'timestamp', 'track_name', 'artist', 'album', 
    'platform', 'ms_played', 'skipped', 'shuffle', 'offline', 
    'private_session', 'start_reason', 'end_reason'
]

file_paths = glob.glob('data/spotify-history-*.csv')

for file in file_paths:
    df = pd.read_csv(file)
    # Keep only the raw columns
    df_raw = df[raw_columns].copy()
    
    # Save as "raw"
    filename = os.path.basename(file)
    df_raw.to_csv(f'data/raw_exports/raw_{filename}', index=False)
    
print("Successfully generated 'raw' backup files in data/raw_exports/")
