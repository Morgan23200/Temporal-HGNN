"""
9_cross_region_experiment.py

Cross-region propagation edge experiment.

This script adds the missing piece of the original proposal: directed
video-level edges that connect (video v, region r_j, week t-1) to
(video v, region r_k, week t) with r_j != r_k, so the graph can express
"trending in r_j this week predicts r_k next week".

It reuses the EXISTING temporal snapshots produced by
6_build_temp_hypergraph.py. No re-run of the parquet pipeline is needed:

  - node_video_idx_current  identifies the same video across time layers
  - node_country_idx        identifies each node's region
  - time_layer              separates layer 0 (t-1) from layer 1 (t)

so the cross-region edge index can be derived directly from each saved
snapshot.

Two stages (mirrors the numbered-pipeline style of the repo):

  python 9_cross_region_experiment.py build
      Reads ./temporal_hetero_hypergraph_snapshots, adds
      cross_region_edge_index (+ per-edge src/dst country ids in the
      GLOBAL country vocab), writes augmented snapshots to
      ./xregion_hetero_hypergraph_snapshots.

  python 9_cross_region_experiment.py train
      Trains three new variants under the exact same protocol as
      8_train_models.py (same split, loss, pos_weight, optimizer,
      metrics, checkpoint-on-val-NDCG@5):

        hetero_xregion            hetero relations + gated cross-region
                                  attention block (no same-region temporal)
        hetero_xregion_temporal   hetero relations + gated same-region
                                  temporal block + gated cross-region block
        temporal_hetero_gated     control: hetero + same-region temporal
                                  only, but GATED (tests whether gating
                                  alone fixes the static-beats-temporal
                                  result, independent of cross-region)

      After test evaluation, the cross-region variants also export a
      directed country -> country influence matrix aggregated from the
      attention coefficients on cross-region edges over the test split.

Design notes
------------
* The cross-region block uses per-destination softmax ATTENTION over
  incoming cross-region messages (instead of uniform averaging), so the
  attention coefficients are interpretable as directional region-to-region
  influence, and a GATED RESIDUAL:

      h_out = h + gate * (attention-aggregated messages)

  with the gate bias initialized negative, so every new model starts out
  ~equivalent to static_hetero and only takes on cross-region /
  temporal signal where it helps. This addresses the observed
  static_hetero > temporal_hetero result, which is consistent with
  uniform averaging forcing uninformative messages into every node.

* Edge blow-up control: a video trending in many regions in both layers
  creates up to |src regions| x |dst regions| edges. --max-src-per-video
  caps the number of layer-0 source regions per video (kept by highest
  log_mean_views), default 20.

* With the current two-layer snapshots the delay is always exactly one
  window, so no delay embedding is needed yet. When snapshots are
  extended to K weeks, add an nn.Embedding(K, emb_dim) consumed by
  CrossRegionAttentionBlock and concatenated to the message - the hook
  point is marked with "DELAY-EMBEDDING HOOK" below.

Usage:
  python 9_cross_region_experiment.py build [--max-src-per-video 20]
  python 9_cross_region_experiment.py train [--epochs 30]
  python 9_cross_region_experiment.py all     # build then train
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Reuse the existing training utilities and model components so the protocol
# is IDENTICAL to 8_train_models.py (module name starts with a digit, so use
# importlib).
# ---------------------------------------------------------------------------
train_mod = importlib.import_module("8_train_models")
models_mod = importlib.import_module("models_temp_7")

SnapshotDataset = train_mod.SnapshotDataset
chronological_split = train_mod.chronological_split
snapshot_identity_collate = train_mod.snapshot_identity_collate
run_epoch = train_mod.run_epoch
print_metrics = train_mod.print_metrics
save_json = train_mod.save_json
compute_pos_weight = train_mod.compute_pos_weight
move_snapshot_to_device = train_mod.move_snapshot_to_device
set_seed = train_mod.set_seed

EmbeddingSelector = models_mod.EmbeddingSelector
HypergraphConvLite = models_mod.HypergraphConvLite
RelationFusion = models_mod.RelationFusion
pool_nodes_to_videos = models_mod.pool_nodes_to_videos
CountryMLPPredictor = models_mod.CountryMLPPredictor
VideoCountryModel = models_mod.VideoCountryModel

# ---------------------------------------------------------------------------
# Config (mirrors 8_train_models.py)
# ---------------------------------------------------------------------------
TEMPORAL_SNAPSHOT_DIR = "./temporal_hetero_hypergraph_snapshots"
XREGION_SNAPSHOT_DIR = "./xregion_hetero_hypergraph_snapshots"
XREGION_PREFIX = "xregion_hetero_snapshot"
OUTPUT_DIR = "./training_runs_xregion"

SEED = train_mod.SEED
DEVICE = train_mod.DEVICE
TRAIN_RATIO = train_mod.TRAIN_RATIO
VAL_RATIO = train_mod.VAL_RATIO
TEST_RATIO = train_mod.TEST_RATIO
HIDDEN_CHANNELS = train_mod.HIDDEN_CHANNELS
EMB_DIM = train_mod.EMB_DIM
DROPOUT = train_mod.DROPOUT
LEARNING_RATE = train_mod.LEARNING_RATE
WEIGHT_DECAY = train_mod.WEIGHT_DECAY
USE_POS_WEIGHT = train_mod.USE_POS_WEIGHT
USE_AMP = train_mod.USE_AMP
NUM_WORKERS = train_mod.NUM_WORKERS
BATCH_SIZE = train_mod.BATCH_SIZE
PRIMARY_METRIC = train_mod.PRIMARY_METRIC  # "ndcg@5"

DEFAULT_MAX_SRC_PER_VIDEO = 20


# ===========================================================================
# STAGE 1: BUILD - derive cross-region edges from existing temporal snapshots
# ===========================================================================

def build_cross_region_edges(
    snapshot: dict,
    max_src_per_video: int = DEFAULT_MAX_SRC_PER_VIDEO,
) -> dict:
    """Derive (v, r_j, t-1) -> (v, r_k, t), r_j != r_k edges from a snapshot.

    Returns dict with:
      cross_region_edge_index  LongTensor [2, E]  (src node idx, dst node idx)
      cross_edge_src_country   LongTensor [E]     global-vocab country id of src
      cross_edge_dst_country   LongTensor [E]     global-vocab country id of dst
    """
    time_layer = snapshot["time_layer"]
    vid_idx = snapshot["node_video_idx_current"]
    local_country_idx = snapshot["node_country_idx"]

    # Map LOCAL (per-snapshot factorized) country ids -> GLOBAL vocab ids so
    # the exported influence matrix is consistent across snapshots.
    local_vocab: List[str] = snapshot["local_country_vocab"]
    country_to_idx: Dict[str, int] = snapshot["country_to_idx"]
    local_to_global = torch.tensor(
        [country_to_idx.get(c, -1) for c in local_vocab], dtype=torch.long
    )
    global_country_idx = local_to_global[local_country_idx]

    # Source strength for the top-K cap: log_mean_views if present, else the
    # first feature column.
    feat_cols: List[str] = snapshot["node_feature_cols"]
    views_col = feat_cols.index("log_mean_views") if "log_mean_views" in feat_cols else 0
    src_strength = snapshot["x_node"][:, views_col]

    layer0 = (time_layer == 0) & (vid_idx >= 0)
    layer1 = (time_layer == 1) & (vid_idx >= 0)

    src_nodes = torch.nonzero(layer0, as_tuple=False).flatten()
    dst_nodes = torch.nonzero(layer1, as_tuple=False).flatten()

    empty = {
        "cross_region_edge_index": torch.empty((2, 0), dtype=torch.long),
        "cross_edge_src_country": torch.empty((0,), dtype=torch.long),
        "cross_edge_dst_country": torch.empty((0,), dtype=torch.long),
    }
    if src_nodes.numel() == 0 or dst_nodes.numel() == 0:
        return empty

    # Group node indices by video on each layer.
    def group_by_video(nodes: torch.Tensor) -> Dict[int, np.ndarray]:
        vids = vid_idx[nodes].numpy()
        nodes_np = nodes.numpy()
        order = np.argsort(vids, kind="stable")
        vids_sorted = vids[order]
        nodes_sorted = nodes_np[order]
        uniq, starts = np.unique(vids_sorted, return_index=True)
        splits = np.split(nodes_sorted, starts[1:])
        return dict(zip(uniq.tolist(), splits))

    src_groups = group_by_video(src_nodes)
    dst_groups = group_by_video(dst_nodes)

    strength_np = src_strength.numpy()
    gci_np = global_country_idx.numpy()

    e_src: List[np.ndarray] = []
    e_dst: List[np.ndarray] = []
    for vid, s_nodes in src_groups.items():
        d_nodes = dst_groups.get(vid)
        if d_nodes is None:
            continue
        # Cap source regions per video by strength (keeps the biggest
        # candidate "origin" regions, bounds edge count).
        if len(s_nodes) > max_src_per_video:
            keep = np.argsort(-strength_np[s_nodes], kind="stable")[:max_src_per_video]
            s_nodes = s_nodes[keep]
        # Cartesian product, then drop same-country pairs (those are the
        # within-region temporal edges, already present separately).
        ss = np.repeat(s_nodes, len(d_nodes))
        dd = np.tile(d_nodes, len(s_nodes))
        mask = gci_np[ss] != gci_np[dd]
        if mask.any():
            e_src.append(ss[mask])
            e_dst.append(dd[mask])

    if not e_src:
        return empty

    src_cat = np.concatenate(e_src)
    dst_cat = np.concatenate(e_dst)
    edge_index = torch.tensor(np.vstack([src_cat, dst_cat]), dtype=torch.long)
    return {
        "cross_region_edge_index": edge_index,
        "cross_edge_src_country": torch.tensor(gci_np[src_cat], dtype=torch.long),
        "cross_edge_dst_country": torch.tensor(gci_np[dst_cat], dtype=torch.long),
    }


def build_all(
    temporal_dir: str = TEMPORAL_SNAPSHOT_DIR,
    out_dir: str = XREGION_SNAPSHOT_DIR,
    max_src_per_video: int = DEFAULT_MAX_SRC_PER_VIDEO,
) -> None:
    in_dir = Path(temporal_dir)
    files = sorted(in_dir.glob("temporal_hetero_snapshot_*.pt"))
    if not files:
        raise ValueError(f"No temporal snapshots found in {temporal_dir}. Run 6_build_temp_hypergraph.py first.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    summaries = []
    for i, p in enumerate(files, start=1):
        snap = torch.load(p, map_location="cpu")
        extras = build_cross_region_edges(snap, max_src_per_video=max_src_per_video)
        snap.update(extras)
        w = int(p.stem.split("_")[-1])
        out_path = out / f"{XREGION_PREFIX}_{w:05d}.pt"
        torch.save(snap, out_path)
        n_x = int(extras["cross_region_edge_index"].size(1))
        n_t = int(snap["temporal_edge_index"].size(1))
        summaries.append({"window_idx": w, "num_cross_region_edges": n_x, "num_temporal_edges": n_t, "num_nodes": int(snap["x_node"].size(0))})
        print(f"[INFO] [{i}/{len(files)}] window {w:05d}: {n_x} cross-region edges (vs {n_t} within-region temporal)")

    # Copy meta if present, add build info.
    meta_src = in_dir / "meta_temporal_hetero_snapshots.json"
    meta = {}
    if meta_src.exists():
        meta = json.loads(meta_src.read_text(encoding="utf-8"))
    meta.update({
        "snapshot_prefix": XREGION_PREFIX,
        "max_src_per_video": max_src_per_video,
        "xregion_summaries": summaries,
    })
    (out / "meta_xregion_snapshots.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    total_edges = sum(s["num_cross_region_edges"] for s in summaries)
    print(f"[INFO] Done. {len(summaries)} snapshots -> {out_dir}. Total cross-region edges: {total_edges}")


# ===========================================================================
# STAGE 2: MODELS - gated attention blocks + encoders
# ===========================================================================

class GatedAttentionEdgeBlock(nn.Module):
    """Directed edge message passing with per-destination softmax attention
    and a gated residual update.

        m_e     = W_src h_src                       (message per edge)
        alpha_e = softmax_dst( a([h_dst ; m_e]) )   (attention over incoming)
        agg_d   = sum_e alpha_e m_e
        gate_d  = sigmoid(W_g [h_d ; agg_d])        (bias init < 0)
        h_d'    = h_d + gate_d * W_upd agg_d        (residual; nodes with no
                                                     incoming edges unchanged)

    The gate starts near zero (bias=-2 => gate~0.12), so training begins
    close to the static model and only opens the gate where the edge type
    carries signal. `last_alpha` is retained for interpretability export.

    DELAY-EMBEDDING HOOK: for K-week snapshots, add
        self.delay_emb = nn.Embedding(K, channels)
    and compute messages as W_src h_src + delay_emb(delta_e).
    """

    def __init__(self, channels: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.src_lin = nn.Linear(channels, channels)
        self.upd_lin = nn.Linear(channels, channels)
        self.att = nn.Linear(2 * channels, 1)
        self.gate = nn.Linear(2 * channels, 1)
        nn.init.constant_(self.gate.bias, -2.0)
        self.norm = nn.LayerNorm(channels)
        self.dropout = dropout
        self.last_alpha: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        self.last_alpha = None
        if edge_index.numel() == 0:
            return x

        src, dst = edge_index[0].long(), edge_index[1].long()
        messages = self.src_lin(x[src])                              # [E, C]

        # --- per-destination softmax attention (numerically stable) ---
        scores = self.att(torch.cat([x[dst], messages], dim=-1)).squeeze(-1)  # [E]
        n = x.size(0)
        max_per_dst = torch.full((n,), float("-inf"), device=x.device)
        max_per_dst.scatter_reduce_(0, dst, scores, reduce="amax", include_self=True)
        exp_scores = torch.exp(scores - max_per_dst[dst])
        denom = x.new_zeros(n).index_add_(0, dst, exp_scores)
        alpha = exp_scores / denom[dst].clamp_min(1e-12)             # [E]
        self.last_alpha = alpha.detach()

        agg = x.new_zeros(x.size()).index_add_(0, dst, alpha.unsqueeze(-1) * messages)

        # --- gated residual ---
        gate = torch.sigmoid(self.gate(torch.cat([x, agg], dim=-1)))  # [N, 1]
        has_incoming = x.new_zeros(n).index_add_(0, dst, torch.ones_like(exp_scores)).unsqueeze(-1) > 0
        update = self.norm(self.upd_lin(agg))
        update = F.dropout(update, p=self.dropout, training=self.training)
        return x + gate * update * has_incoming.float()


class XRegionHeteroEncoder(nn.Module):
    """Static hetero backbone (2 rounds of video/country/category conv +
    fusion, same shape as StaticHeteroHypergraphEncoder) followed by optional
    gated same-region temporal block and/or gated cross-region block."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        emb_dim: int,
        dropout: float = 0.2,
        pooling: str = "mean",
        use_temporal: bool = False,
        use_cross_region: bool = True,
    ) -> None:
        super().__init__()
        self.selector = EmbeddingSelector(in_channels, hidden_channels, emb_dim, dropout)

        self.video_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv1 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse1 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.video_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.country_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.category_conv2 = HypergraphConvLite(emb_dim, emb_dim, dropout=dropout)
        self.fuse2 = RelationFusion(emb_dim, num_relations=3, dropout=dropout)

        self.use_temporal = use_temporal
        self.use_cross_region = use_cross_region
        self.temporal_block = GatedAttentionEdgeBlock(emb_dim, dropout=dropout) if use_temporal else None
        self.cross_region_block = GatedAttentionEdgeBlock(emb_dim, dropout=dropout) if use_cross_region else None

        self.pooling = pooling
        self.output_dim = emb_dim if pooling in {"mean", "max"} else emb_dim * 2

    def forward(self, snapshot: dict) -> torch.Tensor:
        z = self.selector(snapshot["x_node"])

        h_v = self.video_conv1(z, snapshot["hyperedge_index_video"])
        h_c = self.country_conv1(z, snapshot["hyperedge_index_country"])
        h_k = self.category_conv1(z, snapshot["hyperedge_index_category"])
        h = self.fuse1([h_v, h_c, h_k])

        h_v = self.video_conv2(h, snapshot["hyperedge_index_video"])
        h_c = self.country_conv2(h, snapshot["hyperedge_index_country"])
        h_k = self.category_conv2(h, snapshot["hyperedge_index_category"])
        h = self.fuse2([h_v, h_c, h_k])

        if self.temporal_block is not None:
            h = self.temporal_block(h, snapshot["temporal_edge_index"])
        if self.cross_region_block is not None:
            h = self.cross_region_block(h, snapshot["cross_region_edge_index"])

        current_mask = snapshot["time_layer"] == 1
        return pool_nodes_to_videos(
            node_embeddings=h,
            node_video_idx=snapshot["node_video_idx_current"],
            num_videos=len(snapshot["video_ids"]),
            pooling=self.pooling,
            valid_mask=current_mask,
        )


EXPERIMENTS_XR: Dict[str, Dict] = {
    # Cross-region only: isolates the contribution of the new edge type.
    "hetero_xregion": {"use_temporal": False, "use_cross_region": True},
    # Full: hetero + gated same-region temporal + gated cross-region.
    "hetero_xregion_temporal": {"use_temporal": True, "use_cross_region": True},
    # Control: does gating alone fix static > temporal? (no cross-region)
    "temporal_hetero_gated": {"use_temporal": True, "use_cross_region": False},
}


def build_model_xr(exp_cfg: Dict, in_channels: int, num_countries: int) -> nn.Module:
    encoder = XRegionHeteroEncoder(
        in_channels=in_channels,
        hidden_channels=HIDDEN_CHANNELS,
        emb_dim=EMB_DIM,
        dropout=DROPOUT,
        pooling="mean",
        use_temporal=exp_cfg["use_temporal"],
        use_cross_region=exp_cfg["use_cross_region"],
    )
    predictor = CountryMLPPredictor(
        in_dim=encoder.output_dim,
        hidden_channels=HIDDEN_CHANNELS,
        num_countries=num_countries,
        dropout=DROPOUT,
    )
    return VideoCountryModel(encoder=encoder, predictor=predictor)


# ===========================================================================
# Interpretability: country -> country influence matrix from attention
# ===========================================================================

@torch.no_grad()
def export_influence_matrix(model: nn.Module, dataset: SnapshotDataset, device: torch.device, num_countries: int) -> np.ndarray:
    """Aggregate cross-region attention coefficients into a directed
    [num_countries, num_countries] matrix M where M[j, k] is the mean
    attention weight on edges from country j (week t-1) to country k
    (week t) over the given split."""
    block = getattr(model.encoder, "cross_region_block", None)
    if block is None:
        raise ValueError("Model has no cross_region_block.")
    model.eval()
    m_sum = np.zeros((num_countries, num_countries), dtype=np.float64)
    m_cnt = np.zeros((num_countries, num_countries), dtype=np.float64)
    for i in range(len(dataset)):
        snapshot = move_snapshot_to_device(dataset[i], device)
        if snapshot["cross_region_edge_index"].numel() == 0:
            continue
        model(snapshot)  # populates block.last_alpha
        alpha = block.last_alpha
        if alpha is None:
            continue
        src_c = snapshot["cross_edge_src_country"].cpu().numpy()
        dst_c = snapshot["cross_edge_dst_country"].cpu().numpy()
        a = alpha.cpu().numpy()
        valid = (src_c >= 0) & (dst_c >= 0)
        np.add.at(m_sum, (src_c[valid], dst_c[valid]), a[valid])
        np.add.at(m_cnt, (src_c[valid], dst_c[valid]), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        m_mean = np.where(m_cnt > 0, m_sum / m_cnt, 0.0)
    return m_mean


# ===========================================================================
# STAGE 2: TRAIN (mirrors train_one_experiment in 8_train_models.py)
# ===========================================================================

def train_one_experiment_xr(exp_name: str, exp_cfg: Dict, epochs: int) -> Dict[str, float]:
    print("=" * 80)
    print(f"[INFO] Running experiment: {exp_name}")
    print("=" * 80)

    split = chronological_split(
        snapshot_dir=XREGION_SNAPSHOT_DIR, prefix=XREGION_PREFIX,
        train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
    )
    print(f"[INFO] Split: {split}")

    train_ds = SnapshotDataset(XREGION_SNAPSHOT_DIR, XREGION_PREFIX, split["train"])
    val_ds = SnapshotDataset(XREGION_SNAPSHOT_DIR, XREGION_PREFIX, split["val"])
    test_ds = SnapshotDataset(XREGION_SNAPSHOT_DIR, XREGION_PREFIX, split["test"])

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)

    first = train_ds[0]
    in_channels = first["x_node"].size(1)
    num_countries = len(first["country_vocab"])

    model = build_model_xr(exp_cfg, in_channels, num_countries).to(DEVICE)
    pos_weight = compute_pos_weight(train_ds).to(DEVICE) if USE_POS_WEIGHT else None
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device="cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    history = []
    best_state = None
    best_val_metric = -float("inf")
    best_epoch = -1

    exp_dir = Path(OUTPUT_DIR) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        {
            "experiment": exp_name,
            "encoder": "XRegionHeteroEncoder",
            "use_temporal": exp_cfg["use_temporal"],
            "use_cross_region": exp_cfg["use_cross_region"],
            "snapshot_dir": XREGION_SNAPSHOT_DIR,
            "prefix": XREGION_PREFIX,
            "hidden_channels": HIDDEN_CHANNELS,
            "emb_dim": EMB_DIM,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "epochs": epochs,
            "primary_metric": PRIMARY_METRIC,
            "use_pos_weight": USE_POS_WEIGHT,
            "gated_residual": True,
            "attention_aggregation": True,
        },
        exp_dir / "config.json",
    )
    save_json(split, exp_dir / "split.json")

    for epoch in range(1, epochs + 1):
        print(f"\n[INFO] {exp_name} | Epoch {epoch}/{epochs}")
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_metrics = run_epoch(model, val_loader, None, criterion, DEVICE, None)
        print_metrics("TRAIN", train_metrics)
        print_metrics("VAL  ", val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        val_score = val_metrics.get(PRIMARY_METRIC, 0.0)
        if val_score > best_val_metric:
            best_val_metric = val_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), exp_dir / "last_model.pt")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = epochs

    torch.save(best_state, exp_dir / "best_model.pt")
    save_json({"history": history}, exp_dir / "history.json")
    save_json({"best_epoch": best_epoch, f"best_val_{PRIMARY_METRIC}": best_val_metric}, exp_dir / "best_summary.json")

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, criterion, DEVICE, None)
    print_metrics("TEST ", test_metrics)
    save_json(test_metrics, exp_dir / "test_metrics.json")

    # -- influence matrix export (the interpretability payoff) --------------
    if exp_cfg["use_cross_region"]:
        print("[INFO] Exporting country->country influence matrix from test-split attention...")
        M = export_influence_matrix(model, test_ds, DEVICE, num_countries)
        vocab = first["country_vocab"]
        np.save(exp_dir / "influence_matrix.npy", M)
        save_json({"country_vocab": vocab}, exp_dir / "influence_matrix_vocab.json")
        # Top directed pairs, human-readable.
        flat = [(float(M[j, k]), vocab[j], vocab[k]) for j in range(len(vocab)) for k in range(len(vocab)) if j != k and M[j, k] > 0]
        flat.sort(reverse=True)
        top = [{"src": s, "dst": d, "mean_attention": round(a, 6)} for a, s, d in flat[:30]]
        save_json({"top_directed_pairs": top}, exp_dir / "influence_top_pairs.json")
        print(f"[INFO] Influence matrix saved to {exp_dir}/influence_matrix.npy (top pairs in influence_top_pairs.json)")

    return {"best_epoch": best_epoch, f"best_val_{PRIMARY_METRIC}": best_val_metric, **test_metrics}


def train_all(epochs: int) -> None:
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Device:         {DEVICE}")
    print(f"[INFO] Epochs:         {epochs}")
    print(f"[INFO] Primary metric: {PRIMARY_METRIC}")

    leaderboard: Dict[str, Dict[str, float]] = {}
    for exp_name, exp_cfg in EXPERIMENTS_XR.items():
        leaderboard[exp_name] = train_one_experiment_xr(exp_name, exp_cfg, epochs)

    sorted_lb = dict(sorted(leaderboard.items(), key=lambda kv: kv[1].get(PRIMARY_METRIC, 0.0), reverse=True))
    save_json(sorted_lb, Path(OUTPUT_DIR) / "leaderboard.json")

    print("\n" + "=" * 80)
    print(f"[INFO] Cross-region leaderboard (sorted by {PRIMARY_METRIC})")
    print("[INFO] Compare against 8_train_models.py: static_hetero NDCG@5 = 0.1773, temporal_hetero = 0.1619")
    print("=" * 80)
    header = f"{'Experiment':<26} {'N':>6}  {'Hit@1':>6}  {'Hit@5':>6}  {'NDCG@1':>7}  {'NDCG@5':>7}  {'NDCG@10':>8}  {'F1':>6}  {'Loss':>7}"
    print(header)
    print("-" * len(header))
    for name, m in sorted_lb.items():
        print(
            f"{name:<26} "
            f"{int(m.get('N', 0)):>6}  "
            f"{m.get('hit@1', 0.0):>6.4f}  "
            f"{m.get('hit@5', 0.0):>6.4f}  "
            f"{m.get('ndcg@1', 0.0):>7.4f}  "
            f"{m.get('ndcg@5', 0.0):>7.4f}  "
            f"{m.get('ndcg@10', 0.0):>8.4f}  "
            f"{m.get('f1_micro', 0.0):>6.4f}  "
            f"{m.get('loss', 0.0):>7.4f}"
        )


# ===========================================================================
# CLI
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-region propagation edge experiment")
    parser.add_argument("stage", choices=["build", "train", "all"], help="build: derive cross-region edges; train: run experiments; all: both")
    parser.add_argument("--max-src-per-video", type=int, default=DEFAULT_MAX_SRC_PER_VIDEO, help="Cap on layer-0 source regions per video (by log_mean_views)")
    parser.add_argument("--epochs", type=int, default=train_mod.EPOCHS, help="Training epochs (default: same as 8_train_models.py)")
    args = parser.parse_args()

    if args.stage in {"build", "all"}:
        build_all(max_src_per_video=args.max_src_per_video)
    if args.stage in {"train", "all"}:
        train_all(epochs=args.epochs)


if __name__ == "__main__":
    main()
