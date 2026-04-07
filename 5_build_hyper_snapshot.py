from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

DEFAULT_NODE_FEATURE_COLS = [
    "log_mean_views",
    "log_max_views",
    "log_mean_comments",
    "log_max_comments",
    "mean_rank",
    "best_rank",
    "last_rank",
    "rank_pct_in_country_window",
    "views_pct_in_country_window",
    "comments_pct_in_country_window",
    "num_countries_for_video_in_window",
    "is_us_present_for_video_in_window",
    "prev_exists_same_country",
    "prev_mean_view_count",
    "prev_max_view_count",
    "prev_mean_comment_count",
    "prev_max_comment_count",
    "prev_mean_rank",
    "prev_best_rank",
    "delta_mean_view_count",
    "delta_max_view_count",
    "delta_mean_comment_count",
    "delta_max_comment_count",
    "delta_mean_rank",
    "delta_best_rank",
    "streak_same_country",
]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file format: {path}")


def _discover_temporal_files(temporal_dir: str) -> List[Path]:
    base = Path(temporal_dir)
    parquet_files = sorted(base.glob("temporal_*.parquet"))
    if parquet_files:
        return parquet_files
    csv_files = sorted(base.glob("temporal_*.csv"))
    if csv_files:
        return csv_files
    raise ValueError(f"No temporal_*.parquet or temporal_*.csv files found in {temporal_dir}")


def _discover_label_files(label_dir: str) -> Dict[int, Path]:
    base = Path(label_dir)
    parquet_files = sorted(base.glob("labels_*.parquet"))
    if parquet_files:
        return {int(p.stem.split("_")[-1]): p for p in parquet_files}
    csv_files = sorted(base.glob("labels_*.csv"))
    if csv_files:
        return {int(p.stem.split("_")[-1]): p for p in csv_files}
    raise ValueError(f"No labels_*.parquet or labels_*.csv files found in {label_dir}")


def _extract_window_idx(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _safe_country_col(country: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(country))
    return f"label_country_{cleaned}"


def _ensure_temporal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["video_id"] = df["video_id"].astype("string")
    df["country"] = df["country"].astype("string")
    df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce").fillna(-1).astype("int32")
    df["category_id"] = pd.to_numeric(df["category_id"], errors="coerce").fillna(-1).astype("int32")
    for col in df.columns:
        if col in {"video_id", "country", "first_day_in_window", "last_day_in_window"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
            if str(df[col].dtype).startswith("int"):
                df[col] = df[col].fillna(0).astype("int32")
            else:
                df[col] = df[col].fillna(0).astype("float32")
    return df


def _ensure_label_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty:
        return df
    df["video_id"] = df["video_id"].astype("string")
    df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce").fillna(-1).astype("int32")
    return df


def _resolve_feature_cols(df: pd.DataFrame, requested_feature_cols: List[str]) -> List[str]:
    existing = [c for c in requested_feature_cols if c in df.columns]
    missing = [c for c in requested_feature_cols if c not in df.columns]
    if missing:
        print(f"[WARN] Missing feature columns skipped: {missing}")
    if not existing:
        raise ValueError("No requested feature columns found in temporal table.")
    return existing


def _build_node_index(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    df = df.copy().reset_index(drop=True)
    df["node_idx"] = np.arange(len(df), dtype=np.int64)
    video_codes, video_uniques = pd.factorize(df["video_id"], sort=False)
    country_codes, country_uniques = pd.factorize(df["country"], sort=False)
    df["node_video_idx_current"] = video_codes.astype(np.int64)
    df["node_country_idx"] = country_codes.astype(np.int64)
    return df, [str(v) for v in video_uniques.tolist()], [str(c) for c in country_uniques.tolist()]


def _build_relation_hyperedge_index(df: pd.DataFrame, group_col: str) -> torch.Tensor:
    node_ids_all: List[int] = []
    edge_ids_all: List[int] = []
    edge_id = 0
    for _, g in df.groupby(group_col, sort=False):
        node_ids = g["node_idx"].to_numpy(dtype=np.int64)
        if len(node_ids) == 0:
            continue
        node_ids_all.extend(node_ids.tolist())
        edge_ids_all.extend([edge_id] * len(node_ids))
        edge_id += 1
    if not node_ids_all:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([node_ids_all, edge_ids_all]), dtype=torch.long)


def _build_y_video(label_df: pd.DataFrame, video_ids_in_window: List[str], country_vocab: List[str]) -> torch.Tensor:
    label_df = _ensure_label_dtypes(label_df)
    label_cols = [_safe_country_col(c) for c in country_vocab]
    missing = [c for c in label_cols if c not in label_df.columns]
    if missing:
        raise ValueError(f"Missing expected label columns: {missing[:10]}")
    label_map: Dict[str, np.ndarray] = {}
    for _, row in label_df.iterrows():
        label_map[str(row["video_id"])] = row[label_cols].to_numpy(dtype=np.float32)
    zero = np.zeros(len(country_vocab), dtype=np.float32)
    rows = [label_map.get(vid, zero) for vid in video_ids_in_window]
    return torch.tensor(np.vstack(rows), dtype=torch.float32)


def _extract_country_vocab(label_files: Dict[int, Path]) -> List[str]:
    first_label_df = _ensure_label_dtypes(_read_table(label_files[min(label_files)]))
    first_cols = [c for c in first_label_df.columns if c.startswith("label_country_")]
    vocab = [c.replace("label_country_", "") for c in first_cols]
    if not vocab:
        raise ValueError("No label_country_* columns found in label files.")
    return vocab


def build_one_static_hetero_snapshot(
    temporal_path: Path,
    label_path: Path,
    country_vocab: List[str],
    out_path: Path,
    node_feature_cols: List[str],
) -> Dict[str, int]:
    temporal_df = _ensure_temporal_dtypes(_read_table(temporal_path))
    label_df = _ensure_label_dtypes(_read_table(label_path))
    window_idx = _extract_window_idx(temporal_path)

    if temporal_df.empty:
        snapshot = {
            "window_idx": window_idx,
            "x_node": torch.empty((0, 0), dtype=torch.float32),
            "hyperedge_index_video": torch.empty((2, 0), dtype=torch.long),
            "hyperedge_index_country": torch.empty((2, 0), dtype=torch.long),
            "hyperedge_index_category": torch.empty((2, 0), dtype=torch.long),
            "node_video_idx_current": torch.empty((0,), dtype=torch.long),
            "node_country_idx": torch.empty((0,), dtype=torch.long),
            "video_ids": [],
            "country_vocab": country_vocab,
            "country_to_idx": {c: i for i, c in enumerate(country_vocab)},
            "local_country_vocab": [],
            "node_feature_cols": [],
            "y_video": torch.empty((0, len(country_vocab)), dtype=torch.float32),
        }
        torch.save(snapshot, out_path)
        return {"window_idx": window_idx, "num_nodes": 0, "num_videos": 0, "num_video_hyperedges": 0, "num_country_hyperedges": 0, "num_category_hyperedges": 0}

    feature_cols = _resolve_feature_cols(temporal_df, node_feature_cols)
    temporal_df, video_ids, local_country_vocab = _build_node_index(temporal_df)

    x_node = torch.tensor(temporal_df[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    hyperedge_index_video = _build_relation_hyperedge_index(temporal_df, "video_id")
    hyperedge_index_country = _build_relation_hyperedge_index(temporal_df, "country")
    hyperedge_index_category = _build_relation_hyperedge_index(temporal_df, "category_id")
    y_video = _build_y_video(label_df, video_ids, country_vocab)

    snapshot = {
        "window_idx": window_idx,
        "x_node": x_node,
        "hyperedge_index_video": hyperedge_index_video,
        "hyperedge_index_country": hyperedge_index_country,
        "hyperedge_index_category": hyperedge_index_category,
        "node_video_idx_current": torch.tensor(temporal_df["node_video_idx_current"].to_numpy(dtype=np.int64), dtype=torch.long),
        "node_country_idx": torch.tensor(temporal_df["node_country_idx"].to_numpy(dtype=np.int64), dtype=torch.long),
        "video_ids": video_ids,
        "country_vocab": country_vocab,
        "country_to_idx": {c: i for i, c in enumerate(country_vocab)},
        "local_country_vocab": local_country_vocab,
        "node_feature_cols": feature_cols,
        "y_video": y_video,
    }
    torch.save(snapshot, out_path)

    return {
        "window_idx": window_idx,
        "num_nodes": int(x_node.size(0)),
        "num_videos": int(len(video_ids)),
        "num_video_hyperedges": int(hyperedge_index_video[1].max().item() + 1) if hyperedge_index_video.numel() else 0,
        "num_country_hyperedges": int(hyperedge_index_country[1].max().item() + 1) if hyperedge_index_country.numel() else 0,
        "num_category_hyperedges": int(hyperedge_index_category[1].max().item() + 1) if hyperedge_index_category.numel() else 0,
    }


def build_all_static_hetero_snapshots(
    temporal_dir: str,
    label_dir: str,
    out_dir: str,
    node_feature_cols: List[str] | None = None,
) -> None:
    node_feature_cols = node_feature_cols or DEFAULT_NODE_FEATURE_COLS
    temporal_files = _discover_temporal_files(temporal_dir)
    label_files = _discover_label_files(label_dir)
    usable_temporal_files = [p for p in temporal_files if _extract_window_idx(p) in label_files]
    if not usable_temporal_files:
        raise ValueError("No usable temporal files matched with label files.")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    country_vocab = _extract_country_vocab(label_files)

    summaries = []
    for i, temporal_path in enumerate(usable_temporal_files, start=1):
        window_idx = _extract_window_idx(temporal_path)
        out_path = out_dir_path / f"static_hetero_snapshot_{window_idx:05d}.pt"
        print(f"[INFO] [{i}/{len(usable_temporal_files)}] Building static snapshot for window {window_idx:05d}")
        summaries.append(
            build_one_static_hetero_snapshot(
                temporal_path=temporal_path,
                label_path=label_files[window_idx],
                country_vocab=country_vocab,
                out_path=out_path,
                node_feature_cols=node_feature_cols,
            )
        )

    meta = {
        "num_snapshots": len(summaries),
        "country_vocab": country_vocab,
        "country_to_idx": {c: i for i, c in enumerate(country_vocab)},
        "node_feature_cols_requested": node_feature_cols,
        "snapshot_prefix": "static_hetero_snapshot",
        "summaries": summaries,
    }
    with open(out_dir_path / "meta_static_hetero_snapshots.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Done. Wrote {len(summaries)} static snapshots to {out_dir}")


if __name__ == "__main__":
    build_all_static_hetero_snapshots(
        temporal_dir="./window_temporal",
        label_dir="./window_labels",
        out_dir="./static_hetero_hypergraph_snapshots",
    )
