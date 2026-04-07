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
     - MLP baseline
     - Static category hypergraph encoder
     - Static heterogeneous hypergraph encoder
     - Temporal category hypergraph encoder
     - Temporal heterogeneous hypergraph encoder

8. **Model Training**
   - `8_train_models.py`
   - Train and evaluate multiple architectures

---

## Models

We evaluate five model variants:

| Model | Description |
|------|--------|
| **MLP Only** | Baseline using tabular features only |
| **Static Category HGNN** | Hypergraph using category relation only |
| **Static Heterogeneous HGNN** | Multi-relation (video + country + category) |
| **Temporal Category HGNN** | Category + temporal edges |
| **Temporal Heterogeneous HGNN** | Full model (multi-relation + temporal) |

---

## Results

Evaluation metric: **Recall@3 (primary)**

|        Model                    | Recall@3   | Hit@3      | F1_micro   | Loss   |
|---------------------------------|------------|------------|------------|--------|
| **Temporal Heterogeneous HGNN** | **0.1386** | **0.2099** | **0.0821** | 1.1864 |
| Static Heterogeneous HGNN       | 0.1024     | 0.1823     | 0.0461     | 1.3754 |
| Temporal Category HGNN          | 0.0994     | 0.1255     | 0.0452     | 2.0037 |
| MLP Baseline                    | 0.0955     | 0.1493     | 0.0508     | 1.3474 |
| Static Category HGNN            | 0.0376     | 0.0868     | 0.0309     | 1.4307 |

---

## Key Findings

- **Temporal modeling improves performance**
  - Temporal HGNN > Static HGNN

- **Heterogeneous relations are critical**
  - Multi-relation (video + country + category) significantly outperforms single-relation models

- **Best model: Temporal Heterogeneous HGNN**
  - Achieves highest Recall@3 and Hit@3

- **MLP baseline is competitive but limited**
  - Lacks relational and temporal structure

---

## How to Run

```bash
python 1_shard_windows.py
python 2_window_tables.py
python 3_temp_feature.py
python 4_build_labels.py
python 5_build_hyper_snapshot.py
python 6_build_temp_hypergraph.py
python 8_train_models.py
