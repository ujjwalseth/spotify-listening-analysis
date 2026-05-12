"""
Data Collection Pipeline (Spotify API)
=======================================
This script takes the RAW Spotify listening history (which only contains
track_name, artist, album, timestamps, and behavior fields) and enriches it
with audio features fetched from the Spotify Web API using a search-based lookup.

Pipeline:
    raw_exports/*.csv  -->  Search API (track + artist)  -->  Audio Features API  -->  data/*.csv

Requirements:
    pip install spotipy python-dotenv pandas

Credentials:
    Copy .env.example to .env and fill in your Spotify API credentials.
    Get credentials at: https://developer.spotify.com/dashboard
"""

import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv

# --- 1. Load Credentials from .env file ---
load_dotenv()

try:
    sp = spotipy.Spotify(
        client_credentials_manager=SpotifyClientCredentials(),
        requests_timeout=10
    )
    print("✅ Spotify API connection successful.")
except Exception as e:
    print("❌ Spotify credentials not found.")
    print("   Ensure you have a .env file with SPOTIPY_CLIENT_ID and SPOTIPY_CLIENT_SECRET.")
    raise


# --- 2. Search for a Track URI using Track Name + Artist ---
def search_track_uri(track_name, artist_name):
    """
    Searches the Spotify catalogue for a track by name and artist.
    Returns the track URI (e.g., 'spotify:track:3Gm5...') or None if not found.

    We use name + artist to narrow the search because track names alone
    can match hundreds of unrelated songs.
    """
    query = f"track:{track_name} artist:{artist_name}"
    try:
        results = sp.search(q=query, type='track', limit=1)
        items = results['tracks']['items']
        if items:
            return items[0]['uri']
        return None
    except Exception as e:
        print(f"   ⚠️  Search failed for '{track_name}' by '{artist_name}': {e}")
        return None


# --- 3. Fetch Audio Features in Batches of 100 ---
def fetch_audio_features(uris):
    """
    Fetches audio features for a list of Spotify track URIs.
    The Spotify API limits this endpoint to 100 tracks per call,
    so we loop in batches and respect rate limits with a sleep.

    Returns a DataFrame with one row per URI.
    """
    print(f"   Fetching audio features for {len(uris)} tracks in batches of 100...")
    features = []

    for i in range(0, len(uris), 100):
        batch = uris[i:i + 100]
        try:
            batch_features = sp.audio_features(batch)
            # Filter out None responses (API couldn't find the track)
            features.extend([f for f in batch_features if f is not None])
            time.sleep(1.0)  # Respect API rate limits
        except Exception as e:
            print(f"   ⚠️  Error fetching batch {i}–{i+100}: {e}")
            time.sleep(3.0)  # Back off longer on error

    return pd.DataFrame(features)


# --- 4. Main Pipeline ---
def build_enriched_dataset(raw_file, output_file):
    """
    Full pipeline: raw CSV --> Spotify Search --> Audio Features --> enriched CSV.

    Args:
        raw_file   : Path to the raw listening history CSV (from data/raw_exports/).
        output_file: Path to save the enriched CSV (to data/).
    """
    print(f"\n📂 Loading raw file: {raw_file}")
    df = pd.read_csv(raw_file)

    # Drop the ghost empty column from export if present
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # --- Step A: Find unique track + artist pairs ---
    unique_tracks = df[['track_name', 'artist']].drop_duplicates().reset_index(drop=True)
    print(f"   Found {len(unique_tracks)} unique track+artist combinations.")

    # --- Step B: Search for each unique track's URI ---
    print("   Searching Spotify for track URIs...")
    uris = []
    for _, row in unique_tracks.iterrows():
        uri = search_track_uri(row['track_name'], row['artist'])
        uris.append(uri)
        time.sleep(0.3)  # Small delay to avoid hitting rate limits

    unique_tracks['uri'] = uris
    found = unique_tracks['uri'].notna().sum()
    print(f"   Found URIs for {found}/{len(unique_tracks)} tracks.")

    # --- Step C: Fetch audio features for all found URIs ---
    valid_uris = unique_tracks['uri'].dropna().tolist()
    if not valid_uris:
        print("❌ No valid URIs found. Cannot fetch audio features.")
        return

    audio_features_df = fetch_audio_features(valid_uris)

    # Define which audio feature columns we want to keep
    feature_cols = [
        'uri', 'danceability', 'energy', 'valence', 'tempo',
        'acousticness', 'speechiness', 'instrumentalness', 'loudness', 'popularity'
    ]
    # Popularity isn't in audio_features endpoint — fetch separately if needed
    audio_features_df = audio_features_df[[c for c in feature_cols if c in audio_features_df.columns]]

    # --- Step D: Merge URIs + features back into the original listening history ---
    df = df.merge(unique_tracks[['track_name', 'artist', 'uri']], on=['track_name', 'artist'], how='left')
    df = df.merge(audio_features_df, on='uri', how='left')

    # Drop the URI column — it's not needed for analysis
    df = df.drop(columns=['uri'], errors='ignore')

    # --- Step E: Save the enriched dataset ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"✅ Enriched dataset saved to: {output_file}")


# --- 5. Entry Point ---
if __name__ == "__main__":
    import glob

    raw_files = sorted(glob.glob('data/raw_exports/raw_spotify-history-*.csv'))

    if not raw_files:
        print("❌ No raw files found in data/raw_exports/. Run the Spotify data export first.")
    else:
        for raw_file in raw_files:
            # e.g., raw_spotify-history-1.csv  -->  spotify-history-1.csv
            filename = os.path.basename(raw_file).replace('raw_', '')
            output_file = os.path.join('data', filename)
            build_enriched_dataset(raw_file, output_file)
