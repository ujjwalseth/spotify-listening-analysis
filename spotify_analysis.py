# %%
"""
# Unsupervised Machine Learning: Spotify Listening Patterns
## Phase 1: Data Preparation & Understanding
"""

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import glob
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# %%
"""
### 1.1 Data Ingestion
"""

# %%
# Load all CSV files in the data directory
file_paths = glob.glob('data/spotify-history-*.csv')
dfs = [pd.read_csv(f) for f in file_paths]
df_raw = pd.concat(dfs, ignore_index=True)

print(f"Initial raw dataset shape: {df_raw.shape}")
df_raw.head()

# %%
"""
### 1.2 Data Cleaning & Missing Values
"""

# %%
# Drop rows with missing audio features (we cannot safely impute them)
audio_features = ['danceability', 'energy', 'valence', 'tempo', 'acousticness', 
                  'speechiness', 'instrumentalness', 'loudness', 'popularity']
df_clean = df_raw.dropna(subset=audio_features).copy()

print(f"Shape after dropping missing audio features: {df_clean.shape}")

# Filter out extreme short plays (< 30 seconds / 30,000 ms) as they are likely skips and not true listening modes
df_clean = df_clean[df_clean['ms_played'] >= 30000].copy()
print(f"Shape after filtering < 30s skips: {df_clean.shape}")

# Convert timestamp to datetime
df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'])

# %%
"""
### 1.3 Feature Engineering & Transformations
"""

# %%
# 1. Log transform ms_played to handle skewness
df_clean['log_ms_played'] = np.log1p(df_clean['ms_played'])

# 2. Bin hour_of_day for Apriori (Association Rules)
def bin_hour(hour):
    if 5 <= hour < 12: return 'Morning'
    elif 12 <= hour < 17: return 'Afternoon'
    elif 17 <= hour < 21: return 'Evening'
    else: return 'Late Night'

df_clean['hour_binned'] = df_clean['hour_of_day'].apply(bin_hour)

# 3. Create cyclic features for hour_of_day for clustering
df_clean['hour_sin'] = np.sin(2 * np.pi * df_clean['hour_of_day']/24.0)
df_clean['hour_cos'] = np.cos(2 * np.pi * df_clean['hour_of_day']/24.0)

# %%
"""
### 1.4 Scaling & Correlation Heatmap (EDA)
"""

# %%
# Define features for clustering
continuous_features = audio_features + ['log_ms_played']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[continuous_features])
df_scaled = pd.DataFrame(X_scaled, columns=continuous_features, index=df_clean.index)

# Plot Correlation Heatmap
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[continuous_features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlation Heatmap of Audio Features", fontsize=14, pad=20)
plt.tight_layout()
plt.show()

# %%
"""
## Phase 2: Dimensionality Reduction & Visualization
"""

# %%
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

# %%
"""
### 2.1 Principal Component Analysis (PCA)
"""

# %%
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Plot explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
         np.cumsum(pca.explained_variance_ratio_), 'bo-')
plt.axhline(y=0.85, color='r', linestyle='--', label='85% Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA: Cumulative Explained Variance', fontsize=14)
plt.legend()
plt.grid(True)
plt.show()

# Extract top components (e.g., explaining ~85% variance, usually 5-6 components)
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.85) + 1
print(f"Optimal number of components for 85% variance: {n_components}")

pca_optimal = PCA(n_components=n_components)
X_pca_optimal = pca_optimal.fit_transform(X_scaled)

# Component Loadings (Top 2 components)
loadings = pd.DataFrame(pca_optimal.components_.T[:, :2], 
                        columns=['PC1', 'PC2'], 
                        index=continuous_features)
print("\nPCA Loadings (Top 2 PCs):")
print(loadings.sort_values('PC1', ascending=False))

# %%
"""
### 2.2 Manifold Learning (t-SNE & UMAP)
"""

# %%
# t-SNE with varying perplexities to prove structural stability
perplexities = [15, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i, p in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=p, random_state=42)
    X_tsne = tsne.fit_transform(X_scaled)
    axes[i].scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=15)
    axes[i].set_title(f"t-SNE (Perplexity = {p})")

plt.tight_layout()
plt.show()

# Run UMAP for global structure preservation
reducer = umap.UMAP(random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(X_umap[:, 0], X_umap[:, 1], alpha=0.6, s=15, c='teal')
plt.title('UMAP Projection', fontsize=14)
plt.show()

# %%
"""
## Phase 3: Clustering Analysis
"""

# %%
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch

# %%
"""
### 3.1 Agglomerative Clustering (Dendrogram)
"""

# %%
# Plot Dendrogram for Agglomerative Clustering
plt.figure(figsize=(12, 6))
dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'), truncate_mode='level', p=5)
plt.title('Dendrogram (Ward Linkage)', fontsize=14)
plt.xlabel('Data Points')
plt.ylabel('Euclidean Distance')
plt.show()

# %%
"""
### 3.2 K-Means Clustering (Without PCA)
"""

# %%
inertias = []
sil_scores = []
k_range = range(2, 9)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(k_range, inertias, 'bo-', label='Inertia')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (Without PCA)', color='b')

ax2 = ax1.twinx()
ax2.plot(k_range, sil_scores, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score (Without PCA)', color='r')

plt.title('K-Means (Without PCA): Elbow Method & Silhouette Score', fontsize=14)
plt.show()

# %%
"""
### 3.3 K-Means Clustering (With PCA)
"""

# %%
inertias_pca = []
sil_scores_pca = []

for k in k_range:
    kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_pca.fit(X_pca_optimal)
    inertias_pca.append(kmeans_pca.inertia_)
    sil_scores_pca.append(silhouette_score(X_pca_optimal, kmeans_pca.labels_))

fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(k_range, inertias_pca, 'bo-', label='Inertia')
ax1.set_xlabel('Number of Clusters (k)')
ax1.set_ylabel('Inertia (With PCA)', color='b')

ax2 = ax1.twinx()
ax2.plot(k_range, sil_scores_pca, 'ro-', label='Silhouette Score')
ax2.set_ylabel('Silhouette Score (With PCA)', color='r')

plt.title('K-Means (With PCA): Elbow Method & Silhouette Score', fontsize=14)
plt.show()

print(f"Max Silhouette Score without PCA: {max(sil_scores):.3f}")
print(f"Max Silhouette Score with PCA: {max(sil_scores_pca):.3f}")

# %%
"""
### 3.4 Final Clustering & Profiling
"""

# %%
# Based on the dendrogram and silhouette score, let's select an optimal K (e.g., K=4)
OPTIMAL_K = 4
final_model = AgglomerativeClustering(n_clusters=OPTIMAL_K, linkage='ward')
df_clean['cluster'] = final_model.fit_predict(X_scaled)

# Profiling the clusters (calculating median of original unscaled features)
cluster_profiles = df_clean.groupby('cluster')[continuous_features].median()

# -----------------------------------------------------------------------
# CLUSTER NAMING — Update these names after inspecting the heatmap below.
# Look at each cluster's median values to assign a human-readable label.
# Example logic:
#   High energy + high tempo + low acousticness  → "High Energy / Workout"
#   Low energy + high acousticness + late hour   → "Night Chill / Study"
#   High valence + high popularity               → "Upbeat / Mainstream"
#   Low ms_played + high skipped                 → "Casual Browse"
# -----------------------------------------------------------------------
cluster_names = {
    0: "Cluster 0 — [Name after viewing heatmap]",
    1: "Cluster 1 — [Name after viewing heatmap]",
    2: "Cluster 2 — [Name after viewing heatmap]",
    3: "Cluster 3 — [Name after viewing heatmap]",
}
df_clean['cluster_label'] = df_clean['cluster'].map(cluster_names)

# Heatmap of cluster medians for profiling
plt.figure(figsize=(10, 6))
sns.heatmap(StandardScaler().fit_transform(cluster_profiles), 
            annot=True, cmap='viridis', fmt=".2f", 
            xticklabels=continuous_features, 
            yticklabels=[cluster_names[i] for i in range(OPTIMAL_K)])
plt.title("Cluster Profiles (Standardized Medians)", fontsize=14)
plt.tight_layout()
plt.show()

print("\nCluster size distribution:")
print(df_clean['cluster_label'].value_counts())

# %%
"""
## Phase 4: Member-Level Analysis (Core Question)
"""

# %%
"""
*Research Question: Do different group members cluster separately, or do shared genres create cross-member clusters?*
"""

# %%
# 1. Visualizing Member Overlap on UMAP
plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=df_clean['member_id'].astype(str), 
                palette='Set1', alpha=0.7, s=30)
plt.title('UMAP Projection Colored by Member ID', fontsize=14)
plt.legend(title='Member ID')
plt.show()

# 2. Cross-tabulation: Cluster Composition by Member
cross_tab = pd.crosstab(df_clean['cluster'], df_clean['member_id'], normalize='index') * 100

plt.figure(figsize=(8, 5))
cross_tab.plot(kind='bar', stacked=True, colormap='Set2', ax=plt.gca())
plt.title('Cluster Composition by Member ID (%)', fontsize=14)
plt.ylabel('Percentage')
plt.xlabel('Cluster')
plt.legend(title='Member ID', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# %%
"""
## Phase 5: Association Rule Mining (Apriori)
"""

# %%
from mlxtend.frequent_patterns import apriori, association_rules

# %%
"""
### 5.1 Tertile Discretization & Basket Formatting
"""

# %%
df_apriori = df_clean.copy()

# Discretize continuous variables using Tertiles (Low, Medium, High)
cols_to_bin = ['energy', 'valence', 'danceability', 'popularity']
for col in cols_to_bin:
    df_apriori[f'{col}_cat'] = pd.qcut(df_apriori[col], q=3, labels=['Low', 'Medium', 'High'])

# Prepare items
df_apriori['item_genre'] = 'Genre_' + df_apriori['genre'].astype(str)
df_apriori['item_hour'] = 'Hour_' + df_apriori['hour_binned'].astype(str)
df_apriori['item_energy'] = 'Energy_' + df_apriori['energy_cat'].astype(str)
df_apriori['item_popularity'] = 'Pop_' + df_apriori['popularity_cat'].astype(str)

# Sort by timestamp to prepare for rolling 1-hour basket creation
df_apriori = df_apriori.sort_values(['member_id', 'timestamp'])

# Group tracks into baskets (1-hour window per member)
df_apriori['date'] = df_apriori['timestamp'].dt.date
df_apriori['hour'] = df_apriori['timestamp'].dt.hour
df_apriori['basket_id'] = df_apriori['member_id'].astype(str) + "_" + df_apriori['date'].astype(str) + "_" + df_apriori['hour'].astype(str)

items = ['item_genre', 'item_hour', 'item_energy', 'item_popularity']

# Create list of transactions
transactions = []
for basket, group in df_apriori.groupby('basket_id'):
    basket_items = set()
    for item_col in items:
        basket_items.update(group[item_col].dropna().unique())
    if len(basket_items) > 0:
        transactions.append(list(basket_items))

print(f"Total valid 1-hour baskets (transactions): {len(transactions)}")

# %%
"""
### 5.2 Apriori Algorithm execution
"""

# %%
from mlxtend.preprocessing import TransactionEncoder

te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Apriori with min_support = 0.08
frequent_itemsets = apriori(df_trans, min_support=0.08, use_colnames=True)

# Generate rules with lift > 1.2
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)

print(f"Number of rules found: {len(rules)}")
rules.sort_values('lift', ascending=False).head(10)[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

# %%
"""
## Phase 6: Synthesis & Presentation Prep
"""

# %%
import os
os.makedirs('output', exist_ok=True)

# Exporting the cleaned and clustered dataset to the output folder
df_clean.to_csv('output/spotify_clustered_cleaned.csv', index=False)
print("Saved final dataset with cluster assignments to 'output/spotify_clustered_cleaned.csv'.")
print("Analysis Pipeline Complete.")
