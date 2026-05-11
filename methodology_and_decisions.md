# Spotify Unsupervised ML: Methodology & Decision Log

This document tracks every analytical step taken in the project, specifically focusing on **what** was done and **why** it was done. This is designed to serve as the script and defense for the final presentation.

## 1. Data Preparation & Cleaning

| Action Taken | Reason / Justification |
| :--- | :--- |
| **Dropped rows with missing audio features** | The Spotify API occasionally fails to return features for certain tracks. We cannot safely impute (e.g., using mean or median) specific musical characteristics like `danceability` or `tempo` without artificially distorting the clustering space. |
| **Filtered out extreme short plays (< 30 seconds)** | Plays under 30 seconds are typically immediate skips. If our goal is to find "listening modes" (e.g., focused studying, working out), a 10-second skip does not represent a true mode. Including them adds noise to the clusters. |
| **Applied Log Transformation to `ms_played`** | The duration of plays is highly skewed (ranging from 30 seconds to 5+ minute tracks). Standard scaling a highly skewed distribution will cause the outliers to dominate the distance metrics in PCA and K-Means. A log transform normalizes this spread. |
| **Maintained two versions of `hour_of_day`** | **For Clustering:** Kept as a continuous numeric variable (or applied cyclical sine/cosine transforms) because distance-based algorithms rely on continuous space. <br><br>**For Apriori:** Binned into categories (Morning, Afternoon, Evening, Late Night) because association rule mining requires discrete itemsets. |
| **Applied `StandardScaler` to all continuous features** | Distance-based algorithms (PCA, K-Means, Agglomerative, t-SNE) are highly sensitive to feature magnitudes. Without scaling, a feature like `tempo` (range 40-200) would completely overpower `valence` (range 0-1). |
| **Generated a Correlation Heatmap (EDA)** | Audio features are inherently correlated (e.g., `energy` vs. `loudness`, `danceability` vs. `valence`). Identifying these collinearities upfront is essential for interpreting the principal components in PCA. A heatmap also serves as a strong visual indicator of statistical rigour for the presentation. |
| **Analyzed Track `popularity`** | The Spotify dataset includes a 0-100 `popularity` score. We chose to include this in our analysis to explore whether members listen to mainstream hits or niche tracks, and to see if popularity drives distinct clustering behavior. |

---

## 2. Dimensionality Reduction

| Action Taken | Reason / Justification |
| :--- | :--- |
| **Ran PCA prior to visualization** | PCA helps identify which original audio features are driving the most variance in our dataset, giving us a statistical baseline before applying non-linear manifold learning. |
| **Ran t-SNE at multiple perplexity values (15, 30, 50)** | t-SNE visualizations can drastically change based on the perplexity hyperparameter. By testing multiple values, we prove to the evaluation panel that our visual structures are stable and not just an artifact of a lucky parameter choice. |
| **Added UMAP alongside t-SNE** | While t-SNE is good at local clustering, UMAP is statistically stronger at preserving *global* topology (how clusters relate to one another). Running both demonstrates analytical rigor. |

---

## 3. Clustering

| Action Taken | Reason / Justification |
| :--- | :--- |
| **Selected Agglomerative (Hierarchical) Clustering over DBSCAN** | DBSCAN requires precise tuning of `eps` and `min_samples` (usually via a k-distance graph), otherwise it outputs one giant cluster and noise. Agglomerative allows us to plot a **Dendrogram**, visually determining the exact number of clusters without arbitrary parameter guessing. This provides a highly defensible presentation slide. |
| **Ran K-Means alongside Agglomerative** | We used the Elbow Method (Inertia) and Silhouette Scores to find optimal $k$ for K-Means. Comparing these results against the Dendrogram provides a cross-validation of our identified "listening modes." |
| **Cluster Profiling via Centroids** | To assign human-readable names to our clusters (e.g., "Late Night Acoustic"), we calculated the median/mean of the original features for each identified cluster. |

---

## 4. Association Rule Mining (Apriori)

| Action Taken | Reason / Justification |
| :--- | :--- |
| **Discretized continuous variables using Tertiles** | Instead of picking arbitrary thresholds (e.g., "High energy is > 0.6"), we used data-driven tertiles (bottom 33% = Low, middle 33% = Medium, top 33% = High). This is statistically robust. |
| **Defined a "Basket" as a 1-hour window per member** | Music is consumed sequentially. Grouping tracks played by the same person within a 1-hour rolling window accurately represents a single "listening session." |
| **Set `min_support = 0.08` and filtered by `lift > 1.2`** | After filtering out skips (< 30s), our dataset will drop from ~1,200 rows to roughly 800-900 valid plays. A support of 0.08 requires a rule to appear in about 65-75 true listening sessions. This prevents generating hundreds of trivial rules while remaining mathematically sensitive to the cleaned dataset size. |

---

## 5. Member-Level Analysis (Answering the Proposal)

| Action Taken | Reason / Justification |
| :--- | :--- |
| **Calculated Cluster Composition by `member_id`** | Our proposal explicitly asked: *"Do different group members cluster separately, or do shared genres create cross-member clusters?"* By cross-tabulating the final clusters against member IDs, we directly answer our core research question. |
| **Color-coded UMAP/t-SNE plots by Member** | Provides immediate, intuitive visual proof during the presentation of whether our music tastes converge or remain highly distinct. |
