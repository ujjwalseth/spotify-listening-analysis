# 🎵 Spotify Listening Analysis — Unsupervised Machine Learning

> **UGDSAI 29 | End-Term Group Project**  
> Applying Unsupervised Machine Learning to real, self-collected Spotify listening data.

---

## 📌 Project Overview

This project investigates whether distinct **"listening modes"** exist in the music consumption behavior of college students — purely from audio features and behavioral patterns — without using any labels.

**Core Research Question:**  
> *Do different group members cluster separately, or does college life converge our music tastes?*

---

## 📂 Repository Structure

```
├── data/
│   ├── spotify-history-1.csv       # Member 1 listening history (enriched)
│   ├── spotify-history-2.csv       # Member 2 listening history (enriched)
│   ├── spotify-history-3.csv       # Member 3 listening history (enriched)
│   ├── spotify-history-4.csv       # Member 4 listening history (enriched)
│   └── raw_exports/                # Raw Spotify exports (before API enrichment)
│       ├── raw_spotify-history-1.csv
│       ├── raw_spotify-history-2.csv
│       ├── raw_spotify-history-3.csv
│       └── raw_spotify-history-4.csv
│
├── output/
│   └── spotify_clustered_cleaned.csv  # Final dataset with cluster assignments
│
├── spotify_pipeline.ipynb          # ⭐ Main Analysis Notebook (6 Phases)
├── spotify_analysis.py             # Source code for the notebook
├── data_collection_pipeline.py     # Spotify API data collection pipeline
├── methodology_and_decisions.md    # Every decision made + why
├── spotify_proposal.docx           # Approved project proposal
└── README.md
```

---

## 🔬 Methodology (6 Phases)

| Phase | Description | Key Tools |
|---|---|---|
| **1. Data Preparation** | Cleaning, log-transforms, scaling, EDA & correlation heatmap | `pandas`, `StandardScaler` |
| **2. Dimensionality Reduction** | PCA (variance analysis), t-SNE (perplexity 15/30/50), UMAP | `sklearn`, `umap-learn` |
| **3. Clustering** | K-Means (with & without PCA), Agglomerative Clustering + Dendrogram | `sklearn` |
| **4. Member Analysis** | Cross-member cluster composition, UMAP colored by member | `seaborn`, `matplotlib` |
| **5. Association Rules** | Tertile discretization, 1-hour baskets, Apriori (`support=0.08`, `lift>1.2`) | `mlxtend` |
| **6. Synthesis** | Export final clustered dataset | `pandas` |

---

## 📊 Dataset

- **Size:** 1,195 listening sessions across 4 group members  
- **Source:** Personal Spotify data export + Spotify Web API (audio features)  
- **Features:** `danceability`, `energy`, `valence`, `tempo`, `acousticness`, `speechiness`, `instrumentalness`, `loudness`, `popularity`, `ms_played`, `skipped`, `shuffle`, timestamps

---

## ⚙️ Setup & Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn umap-learn mlxtend
```

Then open `spotify_pipeline.ipynb` in Jupyter and run all cells top-to-bottom.

---

## 📋 Key Design Decisions

See [`methodology_and_decisions.md`](./methodology_and_decisions.md) for a full breakdown of every analytical choice made and why. Highlights:

- **Filtered plays < 30s** — Short skips are noise, not true listening modes.
- **Log-transformed `ms_played`** — Prevents skewed duration from dominating distance metrics.
- **Agglomerative over DBSCAN** — Dendrogram makes cluster count selection visually defensible.
- **Tertile binning for Apriori** — Data-driven discretization, not arbitrary thresholds.
- **t-SNE at 3 perplexity values** — Proves structural stability, not lucky hyperparameter choice.

---

## 📁 Data Collection Pipeline

See [`data_collection_pipeline.py`](./data_collection_pipeline.py) for the full raw → processed pipeline using the Spotify Web API (`spotipy` library).
