# Temporal-HGNN: Cross-Region Trending Prediction

## Overview

This project investigates cross-country trending video prediction using a **Temporal Heterogeneous Hypergraph Neural Network (HGNN)**.

Nodes:(video_id, country, window)
|_____Each node contains:
      aggregated statistics (views, rank, comments)
      temporal features (prev, delta, streak)
      
I represent:
- Videos (video_id = same, different countries)
- Countries (all videos in same country)
- Categories (videos with same category)

as **hyperedges**, and extend this structure across time to capture **temporal dynamics**.

The goal is to predict **which countries a video will trend in during the next time window**.

---

## Pipeline

1. **Data Sharding**
   - `1_shard_windows.py`
   - Split raw dataset into fixed time windows

2. **Window-Level Aggregation**
   - `2_window_tables.py`
   - Aggregate per (video, country, window)

3. **Temporal Feature Engineering**
   - `3_temp_feature.py`
   - Add previous-window features, deltas, and streaks

4. **Label Construction**
   - `4_build_labels.py`
   - Multi-label targets: countries in next window

5. **Static Hypergraph Construction**
   - `5_build_hyper_snapshot.py`
   - Build per-window heterogeneous hypergraph

6. **Temporal Hypergraph Construction**
   - `6_build_temp_hypergraph.py`
   - Add temporal edges between windows

7. **Model Definitions**
   - `models_temp_7.py`
   - Contains all model architectures:
     - MLP Only: Baseline using tabular features only.
     - Static Category HGNN: Hypergraph using the category relation only.
     - Static Heterogeneous HGNN: Multi-relation (video + country + category).
     - Temporal Category HGNN: Category + within-region temporal edges.
     - Temporal Heterogeneous HGNN: Multi-relation + within-region temporal edges.
   -`9_cross_region.py`
     - Heterogeneous + Cross-Region: Temporal heterogeneous backbone + directed cross-region propagation edges with gated attention.

8. **Model Training**
   - `8_train_models.py`
   - Train and evaluate multiple architectures

---

## Models

We evaluate five model variants:

| Model | Description |
|-------|-------------|
| **MLP Only** | Baseline using tabular features only |
| **Static Category HGNN** | Hypergraph using category relation only |
| **Static Heterogeneous HGNN** | Multi-relation (video + country + category) |
| **Temporal Category HGNN** | Category + temporal edges |
| **Temporal Heterogeneous HGNN** | Full model (multi-relation + temporal) |

---

## Results

Evaluation metrics: **NDCG@5 (primary)**, Hit@1, Hit@5, NDCG@1, NDCG@10, F1, Loss  
All experiments run on N = 109,033 samples.

| Model | Hit@1 | Hit@5 | NDCG@1 | NDCG@5 | NDCG@10 | F1 | Loss |
|-------|-------|-------|--------|--------|---------|-----|------|
| Static Heterogeneous HGNN | 0.0999 | 0.2853 | 0.0999 | 0.1619 | 0.1852 | 0.0612 | 1.0927 |
|**Temporal Heterogeneous HGNN** | **0.1003** | **0.2848** | **0.1003** | **0.1773** | **0.2219** | **0.0665** | **0.9756** |
| MLP Baseline | 0.0620 | 0.2147 | 0.0620 | 0.1116 | 0.1505 | 0.0519 | 1.2873 |
| Temporal Category HGNN | 0.0446 | 0.1959 | 0.0446 | 0.1014 | 0.1436 | 0.0461 | 1.7888 |
| Static Category HGNN | 0.0366 | 0.1286 | 0.0366 | 0.0503 | 0.0629 | 0.0308 | 1.4307 |
| **Heterogeneous + Cross-Region** | **0.1521** | **0.4374** | **0.1521** | **0.2893** | **0.3215** | **0.0967** | **0.6002** |
---

## Key Findings
Cross-Region Structure is the Missing Signal: The newly implemented Heterogeneous + Cross-Region model drastically outperforms all other variants, achieving an NDCG@5 of 0.2893 (an approximate +78% relative improvement over the Temporal Heterogeneous HGNN). This validates the core hypothesis: popularity is not a purely local quantity, and cross-border spillover drives trend prediction.

Massive Gains in Hit Rate: By adding directed cross-region edges to the static heterogeneous backbone, the model significantly improves its ability to place a true country in the top 5, boosting the Hit@5 metric from 0.2853 to 0.4374.

Temporal Signal Confirmed: The Temporal Heterogeneous HGNN (NDCG@5 of 0.1773) successfully beats the Static Heterogeneous HGNN (0.1619), resolving prior aggregation artifacts and demonstrating that localized temporal trajectories contribute meaningful predictive power.

## How to Run

```bash
python 1_shard_windows.py
python 2_window_tables.py
python 3_temp_feature.py
python 4_build_labels.py
python 5_build_hyper_snapshot.py
python 6_build_temp_hypergraph.py
python models_temp_7.py
python 8_train_models.py
python 9_cross_region.py
```
