from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from typing import Dict, List, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models_temp_7 import build_model

STATIC_SNAPSHOT_DIR = "./static_hetero_hypergraph_snapshots"
TEMPORAL_SNAPSHOT_DIR = "./temporal_hetero_hypergraph_snapshots"
OUTPUT_DIR = "./training_runs_no_earlystop"

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

HIDDEN_CHANNELS = 128
EMB_DIM = 128
DROPOUT = 0.2

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
EPOCHS = 30
DECISION_THRESHOLD = 0.5

USE_POS_WEIGHT = True
USE_AMP = True
PRELOAD_SNAPSHOTS = False
NUM_WORKERS = 0
BATCH_SIZE = 1

# Primary metric used for best-model selection and leaderboard ranking.
# Options: "hit@1", "hit@5", "ndcg@1", "ndcg@5", "ndcg@10", "recall@5", etc.
PRIMARY_METRIC = "ndcg@5"

EXPERIMENTS = {
    "mlp_only": {"snapshot_dir": STATIC_SNAPSHOT_DIR, "prefix": "static_hetero_snapshot", "model_name": "mlp_only"},
    "static_category": {"snapshot_dir": STATIC_SNAPSHOT_DIR, "prefix": "static_hetero_snapshot", "model_name": "static_category"},
    "static_hetero": {"snapshot_dir": STATIC_SNAPSHOT_DIR, "prefix": "static_hetero_snapshot", "model_name": "static_hetero"},
    "temporal_category": {"snapshot_dir": TEMPORAL_SNAPSHOT_DIR, "prefix": "temporal_hetero_snapshot", "model_name": "temporal_category"},
    "temporal_hetero": {"snapshot_dir": TEMPORAL_SNAPSHOT_DIR, "prefix": "temporal_hetero_snapshot", "model_name": "temporal_hetero"},
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SnapshotDataset(Dataset):
    def __init__(self, snapshot_dir: str, prefix: str, window_indices: Optional[List[int]] = None, map_location: str = "cpu", preload: bool = False) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.prefix = prefix
        self.map_location = map_location
        self.preload = preload
        self.files = sorted(self.snapshot_dir.glob(f"{prefix}_*.pt"))
        if window_indices is not None:
            wanted = set(window_indices)
            self.files = [p for p in self.files if int(p.stem.split("_")[-1]) in wanted]
        if not self.files:
            raise ValueError(f"No snapshot files found in {snapshot_dir} with prefix '{prefix}'.")
        self.cache: Optional[List[Dict]] = None
        if self.preload:
            print(f"[INFO] Preloading {len(self.files)} snapshots from {snapshot_dir} ...")
            start = time.time()
            self.cache = [torch.load(p, map_location=map_location) for p in self.files]
            print(f"[INFO] Preload finished in {time.time() - start:.1f}s")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict:
        if self.cache is not None:
            return self.cache[idx]
        return torch.load(self.files[idx], map_location=self.map_location)


def chronological_split(snapshot_dir: str, prefix: str, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[int]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    files = sorted(Path(snapshot_dir).glob(f"{prefix}_*.pt"))
    windows = [int(p.stem.split("_")[-1]) for p in files]
    if len(windows) < 3:
        raise ValueError(f"Need at least 3 snapshots for train/val/test split. Found {len(windows)} in {snapshot_dir} with prefix '{prefix}'.")
    n = len(windows)
    train_end = max(1, int(n * train_ratio))
    val_end = max(train_end + 1, int(n * (train_ratio + val_ratio)))
    if val_end >= n:
        val_end = n - 1
    split = {"train": windows[:train_end], "val": windows[train_end:val_end], "test": windows[val_end:]}
    if not split["train"] or not split["val"] or not split["test"]:
        raise ValueError(f"Invalid split sizes: train={len(split['train'])}, val={len(split['val'])}, test={len(split['test'])}")
    return split


def snapshot_identity_collate(batch: List[Dict]) -> List[Dict]:
    return batch


def move_snapshot_to_device(snapshot: Dict, device: torch.device) -> Dict:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in snapshot.items()}


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:

    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    intersection = (preds * targets).sum().item()
    union = ((preds + targets) > 0).float().sum().item()

    jaccard_micro = safe_div(intersection, union)

    # N = number of prediction rows (video nodes across all snapshots in the split)
    N = int(targets.size(0))

    def ranking_metrics(k: int):
        k = min(k, probs.size(1))

        topk_idx = torch.topk(
            probs,
            k=k,
            dim=1
        ).indices

        hits = 0
        recall_sum = 0.0
        ndcg_sum = 0.0
        valid = 0

        for i in range(probs.size(0)):

            true_idx = torch.nonzero(
                targets[i] > 0,
                as_tuple=False
            ).flatten().tolist()

            if len(true_idx) == 0:
                continue

            valid += 1

            true_set = set(true_idx)
            pred_list = topk_idx[i].tolist()

            overlap = len(
                true_set.intersection(pred_list)
            )

            if overlap > 0:
                hits += 1

            recall_sum += overlap / len(true_set)

            dcg = 0.0

            for rank, pred in enumerate(pred_list, start=1):
                if pred in true_set:
                    dcg += 1.0 / math.log2(rank + 1)

            ideal_hits = min(len(true_set), k)

            idcg = sum(
                1.0 / math.log2(r + 1)
                for r in range(1, ideal_hits + 1)
            )

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0

        denom = max(valid, 1)

        return {
            "hit": hits / denom,
            "recall": recall_sum / denom,
            "ndcg": ndcg_sum / denom,
        }

    m1  = ranking_metrics(1)
    m3  = ranking_metrics(3)
    m5  = ranking_metrics(5)
    m10 = ranking_metrics(10)

    return {
        # ── sample size ──────────────────────────────────────────────────────
        "N": N,

        # ── threshold-based classification ───────────────────────────────────
        "precision_micro": precision,
        "recall_micro":    recall,
        "f1_micro":        f1,
        "jaccard_micro":   jaccard_micro,

        # ── Hit@k  (fraction of queries with ≥1 relevant item in top-k) ─────
        "hit@1":  m1["hit"],
        "hit@3":  m3["hit"],
        "hit@5":  m5["hit"],
        "hit@10": m10["hit"],

        # ── Recall@k ─────────────────────────────────────────────────────────
        "recall@1":  m1["recall"],
        "recall@3":  m3["recall"],
        "recall@5":  m5["recall"],
        "recall@10": m10["recall"],

        # ── NDCG@k ───────────────────────────────────────────────────────────
        "ndcg@1":  m1["ndcg"],
        "ndcg@3":  m3["ndcg"],
        "ndcg@5":  m5["ndcg"],
        "ndcg@10": m10["ndcg"],
    }


def _empty_metrics() -> Dict[str, float]:
    """Return a zero-filled metrics dict (used when a loader yields no data)."""
    keys = [
        "N", "loss", "epoch_time_sec",
        "precision_micro", "recall_micro", "f1_micro", "jaccard_micro",
        "hit@1",  "hit@3",  "hit@5",  "hit@10",
        "recall@1", "recall@3", "recall@5", "recall@10",
        "ndcg@1", "ndcg@3", "ndcg@5", "ndcg@10",
    ]
    return {k: 0.0 for k in keys}


def compute_pos_weight(dataset: SnapshotDataset) -> torch.Tensor:
    print("[INFO] Computing pos_weight over training snapshots...")
    pos = None
    total_rows = 0
    for i in range(len(dataset)):
        y = dataset[i]["y_video"].float()
        if y.numel() == 0:
            continue
        pos = y.sum(dim=0) if pos is None else pos + y.sum(dim=0)
        total_rows += y.size(0)
    if pos is None:
        raise ValueError("Training dataset has no valid labels for pos_weight computation.")
    neg = total_rows - pos
    return (neg / pos.clamp_min(1.0)).float()


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], criterion: nn.Module, device: torch.device, scaler: Optional[torch.amp.GradScaler] = None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    all_logits, all_targets = [], []
    total_loss = 0.0
    total_rows = 0
    start_time = time.time()
    amp_enabled = USE_AMP and device.type == "cuda"

    for batch in loader:
        for snapshot in batch:
            snapshot = move_snapshot_to_device(snapshot, device)
            y = snapshot["y_video"].float()
            if y.numel() == 0:
                continue
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(snapshot)
                loss = criterion(logits, y)
            if training:
                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * y.size(0)
            total_rows += y.size(0)
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    if not all_logits:
        m = _empty_metrics()
        m["epoch_time_sec"] = time.time() - start_time
        return m

    logits_full  = torch.cat(all_logits,  dim=0)
    targets_full = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_full, targets_full, threshold=DECISION_THRESHOLD)
    metrics["loss"]           = total_loss / max(1, total_rows)
    metrics["epoch_time_sec"] = time.time() - start_time
    return metrics


def print_metrics(prefix: str, metrics: Dict[str, float]) -> None:
    """One-line summary surfacing the key metrics requested: N, Hit@1, Hit@5, NDCG@5."""
    print(
        f"{prefix} | "
        f"N={int(metrics['N']):>6} | "
        f"Loss={metrics['loss']:.4f} | "
        f"F1={metrics['f1_micro']:.4f} | "
        f"Jac={metrics['jaccard_micro']:.4f} | "
        f"Hit@1={metrics['hit@1']:.4f} | "
        f"Hit@5={metrics['hit@5']:.4f} | "
        f"NDCG@1={metrics['ndcg@1']:.4f} | "
        f"NDCG@5={metrics['ndcg@5']:.4f} | "
        f"NDCG@10={metrics['ndcg@10']:.4f} | "
        f"R@5={metrics['recall@5']:.4f} | "
        f"Time={metrics['epoch_time_sec']:.1f}s"
    )


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def train_one_experiment(exp_name: str, exp_cfg: Dict[str, str]) -> Dict[str, float]:
    print("=" * 80)
    print(f"[INFO] Running experiment: {exp_name}")
    print("=" * 80)

    snapshot_dir = exp_cfg["snapshot_dir"]
    prefix       = exp_cfg["prefix"]
    model_name   = exp_cfg["model_name"]

    split = chronological_split(snapshot_dir=snapshot_dir, prefix=prefix, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO)
    print(f"[INFO] Split: {split}")

    train_ds = SnapshotDataset(snapshot_dir, prefix, split["train"], preload=PRELOAD_SNAPSHOTS)
    val_ds   = SnapshotDataset(snapshot_dir, prefix, split["val"],   preload=PRELOAD_SNAPSHOTS)
    test_ds  = SnapshotDataset(snapshot_dir, prefix, split["test"],  preload=PRELOAD_SNAPSHOTS)

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=snapshot_identity_collate, num_workers=NUM_WORKERS, pin_memory=pin)

    first        = train_ds[0]
    in_channels  = first["x_node"].size(1)
    num_countries = len(first["country_vocab"])

    model     = build_model(model_name=model_name, in_channels=in_channels, num_countries=num_countries, hidden_channels=HIDDEN_CHANNELS, emb_dim=EMB_DIM, dropout=DROPOUT).to(DEVICE)
    pos_weight = compute_pos_weight(train_ds).to(DEVICE) if USE_POS_WEIGHT else None
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler     = torch.amp.GradScaler(device="cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

    history          = []
    best_state       = None
    best_val_metric  = -float("inf")
    best_epoch       = -1

    exp_dir = Path(OUTPUT_DIR) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            "experiment":      exp_name,
            "model_name":      model_name,
            "snapshot_dir":    snapshot_dir,
            "prefix":          prefix,
            "hidden_channels": HIDDEN_CHANNELS,
            "emb_dim":         EMB_DIM,
            "dropout":         DROPOUT,
            "learning_rate":   LEARNING_RATE,
            "weight_decay":    WEIGHT_DECAY,
            "epochs":          EPOCHS,
            "primary_metric":  PRIMARY_METRIC,
            "use_pos_weight":  USE_POS_WEIGHT,
            "use_amp":         USE_AMP,
            "device":          str(DEVICE),
            "no_early_stopping": True,
        },
        exp_dir / "config.json",
    )
    save_json(split, exp_dir / "split.json")

    for epoch in range(1, EPOCHS + 1):
        print(f"\n[INFO] {exp_name} | Epoch {epoch}/{EPOCHS}")
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, DEVICE, scaler)
        val_metrics   = run_epoch(model, val_loader,   None,      criterion, DEVICE, None)
        print_metrics("TRAIN", train_metrics)
        print_metrics("VAL  ", val_metrics)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

        # ── best-model selection on PRIMARY_METRIC ────────────────────────
        val_score = val_metrics.get(PRIMARY_METRIC, 0.0)
        if val_score > best_val_metric:
            best_val_metric = val_score
            best_epoch      = epoch
            best_state      = copy.deepcopy(model.state_dict())

        torch.save(model.state_dict(), exp_dir / "last_model.pt")

    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
        best_epoch = EPOCHS

    torch.save(best_state, exp_dir / "best_model.pt")
    save_json({"history": history}, exp_dir / "history.json")
    save_json(
        {
            "best_epoch":               best_epoch,
            f"best_val_{PRIMARY_METRIC}": best_val_metric,
        },
        exp_dir / "best_summary.json",
    )

    # ── final evaluation on test set ─────────────────────────────────────
    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_loader, None, criterion, DEVICE, None)
    print_metrics("TEST ", test_metrics)
    save_json(test_metrics, exp_dir / "test_metrics.json")

    return {
        "best_epoch":               best_epoch,
        f"best_val_{PRIMARY_METRIC}": best_val_metric,
        **test_metrics,
    }


def main() -> None:
    set_seed(SEED)
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if DEVICE.type == "cuda":
        print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Device:          {DEVICE}")
    print(f"[INFO] Fixed epochs:    {EPOCHS}")
    print(f"[INFO] Primary metric:  {PRIMARY_METRIC}")
    print("[INFO] Early stopping:  DISABLED")

    leaderboard: Dict[str, Dict[str, float]] = {}
    for exp_name, exp_cfg in EXPERIMENTS.items():
        leaderboard[exp_name] = train_one_experiment(exp_name, exp_cfg)

    # Sort leaderboard by PRIMARY_METRIC descending
    sorted_leaderboard = dict(
        sorted(leaderboard.items(), key=lambda kv: kv[1].get(PRIMARY_METRIC, 0.0), reverse=True)
    )
    save_json(sorted_leaderboard, Path(OUTPUT_DIR) / "leaderboard.json")

    print("\n" + "=" * 80)
    print(f"[INFO] Final leaderboard (sorted by {PRIMARY_METRIC})")
    print("=" * 80)
    header = f"{'Experiment':<22} {'N':>6}  {'Hit@1':>6}  {'Hit@5':>6}  {'NDCG@1':>7}  {'NDCG@5':>7}  {'NDCG@10':>8}  {'F1':>6}  {'Loss':>7}"
    print(header)
    print("-" * len(header))
    for name, m in sorted_leaderboard.items():
        print(
            f"{name:<22} "
            f"{int(m.get('N', 0)):>6}  "
            f"{m.get('hit@1',  0.0):>6.4f}  "
            f"{m.get('hit@5',  0.0):>6.4f}  "
            f"{m.get('ndcg@1', 0.0):>7.4f}  "
            f"{m.get('ndcg@5', 0.0):>7.4f}  "
            f"{m.get('ndcg@10',0.0):>8.4f}  "
            f"{m.get('f1_micro', 0.0):>6.4f}  "
            f"{m.get('loss', 0.0):>7.4f}"
        )


if __name__ == "__main__":
    main()
