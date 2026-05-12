"""
Data Collection Pipeline (Spotify API)

As per the guidelines, this script demonstrates the pipeline used to enrich the 
raw Spotify listening history with audio features via the Spotify Web API.

Note: To use this live, you need to install spotipy (`pip install spotipy`)
and set your SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.
"""

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import time
import os

from dotenv import load_dotenv

# Load credentials from .env file
load_dotenv()

try:
    # Spotipy automatically looks for SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET in environment variables
    sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials())
except Exception as e:
    print("Spotify credentials not found. Ensure you have a .env file with SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.")

def fetch_audio_features(track_uris):
    """
    Fetches audio features for a list of track URIs in batches of 100.
    Spotify API limits audio-features endpoint to 100 tracks per call.
    """
    print(f"Fetching audio features for {len(track_uris)} unique tracks...")
    features = []
    
    for i in range(0, len(track_uris), 100):
        batch = track_uris[i:i+100]
        try:
            # Call Spotify API
            batch_features = sp.audio_features(batch)
            # Remove empty responses
            features.extend([f for f in batch_features if f is not None])
            # Respect API rate limits
            time.sleep(1.5) 
        except Exception as e:
            print(f"Error fetching batch {i}-{i+100}: {e}")
            
    return pd.DataFrame(features)

def build_dataset(raw_history_path, output_path):
    """
    Complete raw -> processed ingestion pipeline.
    """
    print(f"Loading raw listening history from {raw_history_path}")
    df_raw = pd.read_csv(raw_history_path)
    
    # Extract unique URIs
    # Note: raw spotify exports provide 'spotify_track_uri'
    if 'spotify_track_uri' in df_raw.columns:
        unique_uris = df_raw['spotify_track_uri'].dropna().unique().tolist()
        
        # Fetch features
        audio_features_df = fetch_audio_features(unique_uris)
        
        if not audio_features_df.empty:
            # Merge features back into the main dataset
            print("Merging audio features into main dataset...")
            df_final = df_raw.merge(audio_features_df, left_on='spotify_track_uri', right_on='uri', how='left')
            
            # Save the enriched dataset
            df_final.to_csv(output_path, index=False)
            print(f"Dataset successfully built and saved to {output_path}")
        else:
            print("Failed to fetch audio features.")
    else:
        print("Required column 'spotify_track_uri' not found in raw data.")

if __name__ == "__main__":
    # Example execution (assuming 'raw_export.csv' is the file downloaded from Spotify Privacy settings)
    # build_dataset('raw_export.csv', 'data/spotify-history-enriched.csv')
    pass
