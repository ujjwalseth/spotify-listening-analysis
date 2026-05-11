import pandas as pd
import glob

# Get all CSV files in both the main data folder and raw_exports
all_files = glob.glob('data/*.csv') + glob.glob('data/raw_exports/*.csv')

for file in all_files:
    try:
        df = pd.read_csv(file)
        if 'platform' in df.columns:
            df = df.drop(columns=['platform'])
            df.to_csv(file, index=False)
            print(f"Removed 'platform' from {file}")
    except Exception as e:
        print(f"Error processing {file}: {e}")

print("Successfully cleaned platform data from all files.")
