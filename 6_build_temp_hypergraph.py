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


def _discover_temporal_files(temporal_dir: str) -> Dict[int, Path]:
    base = Path(temporal_dir)
    parquet_files = sorted(base.glob("temporal_*.parquet"))
    if parquet_files:
        return {int(p.stem.split("_")[-1]): p for p in parquet_files}
    csv_files = sorted(base.glob("temporal_*.csv"))
    if csv_files:
        return {int(p.stem.split("_")[-1]): p for p in csv_files}
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
        raise ValueError("No requested feature columns found.")
    return existing


def _prepare_two_window_nodes(prev_df: pd.DataFrame, cur_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    prev_df = prev_df.copy()
    cur_df = cur_df.copy()
    prev_df["time_layer"] = 0
    cur_df["time_layer"] = 1
    merged = pd.concat([prev_df, cur_df], axis=0, ignore_index=True).reset_index(drop=True)
    merged["node_idx"] = np.arange(len(merged), dtype=np.int64)

    country_codes, country_uniques = pd.factorize(merged["country"], sort=False)
    merged["node_country_idx"] = country_codes.astype(np.int64)

    cur_video_codes, cur_video_uniques = pd.factorize(cur_df["video_id"], sort=False)
    cur_video_map = {str(v): i for i, v in enumerate(cur_video_uniques.tolist())}
    merged["node_video_idx_current"] = merged["video_id"].map(lambda v: cur_video_map.get(str(v), -1)).astype(np.int64)
    return merged, [str(v) for v in cur_video_uniques.tolist()], [str(c) for c in country_uniques.tolist()]


def _build_layered_relation_hyperedge_index(df: pd.DataFrame, group_col: str) -> torch.Tensor:
    node_ids_all: List[int] = []
    edge_ids_all: List[int] = []
    edge_id = 0
    for layer in [0, 1]:
        sub = df[df["time_layer"] == layer]
        for _, g in sub.groupby(group_col, sort=False):
            node_ids = g["node_idx"].to_numpy(dtype=np.int64)
            if len(node_ids) == 0:
                continue
            node_ids_all.extend(node_ids.tolist())
            edge_ids_all.extend([edge_id] * len(node_ids))
            edge_id += 1
    if not node_ids_all:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(np.vstack([node_ids_all, edge_ids_all]), dtype=torch.long)


def _build_temporal_edge_index(df: pd.DataFrame) -> torch.Tensor:
    prev_nodes = df[df["time_layer"] == 0][["video_id", "country", "node_idx"]].copy()
    cur_nodes = df[df["time_layer"] == 1][["video_id", "country", "node_idx"]].copy()
    joined = prev_nodes.merge(cur_nodes, on=["video_id", "country"], how="inner", suffixes=("_prev", "_cur"), validate="one_to_one")
    if joined.empty:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(joined[["node_idx_prev", "node_idx_cur"]].to_numpy(dtype=np.int64).T, dtype=torch.long)


def _build_y_video(label_df: pd.DataFrame, current_video_ids: List[str], country_vocab: List[str]) -> torch.Tensor:
    label_df = _ensure_label_dtypes(label_df)
    label_cols = [_safe_country_col(c) for c in country_vocab]
    missing = [c for c in label_cols if c not in label_df.columns]
    if missing:
        raise ValueError(f"Missing expected label columns: {missing[:10]}")
    label_map = {str(row['video_id']): row[label_cols].to_numpy(dtype=np.float32) for _, row in label_df.iterrows()}
    zero = np.zeros(len(country_vocab), dtype=np.float32)
    rows = [label_map.get(vid, zero) for vid in current_video_ids]
    return torch.tensor(np.vstack(rows), dtype=torch.float32)


def _extract_country_vocab(label_files: Dict[int, Path]) -> List[str]:
    first_label_df = _ensure_label_dtypes(_read_table(label_files[min(label_files)]))
    first_cols = [c for c in first_label_df.columns if c.startswith("label_country_")]
    vocab = [c.replace("label_country_", "") for c in first_cols]
    if not vocab:
        raise ValueError("No label_country_* columns found in label files.")
    return vocab


def build_one_temporal_hetero_snapshot(
    prev_temporal_path: Path,
    cur_temporal_path: Path,
    label_path: Path,
    out_path: Path,
    country_vocab: List[str],
    feature_cols_requested: List[str],
) -> Dict[str, int]:
    prev_df = _ensure_temporal_dtypes(_read_table(prev_temporal_path))
    cur_df = _ensure_temporal_dtypes(_read_table(cur_temporal_path))
    label_df = _ensure_label_dtypes(_read_table(label_path))

    feature_cols = _resolve_feature_cols(cur_df, feature_cols_requested)
    merged_nodes, current_video_ids, local_country_vocab = _prepare_two_window_nodes(prev_df, cur_df)

    x_node = torch.tensor(merged_nodes[feature_cols].to_numpy(dtype=np.float32), dtype=torch.float32)
    hyperedge_index_video = _build_layered_relation_hyperedge_index(merged_nodes, "video_id")
    hyperedge_index_country = _build_layered_relation_hyperedge_index(merged_nodes, "country")
    hyperedge_index_category = _build_layered_relation_hyperedge_index(merged_nodes, "category_id")
    temporal_edge_index = _build_temporal_edge_index(merged_nodes)
    y_video = _build_y_video(label_df, current_video_ids, country_vocab)

    snapshot = {
        "prev_window_idx": int(prev_df["window_idx"].iloc[0]) if not prev_df.empty else -1,
        "window_idx": int(cur_df["window_idx"].iloc[0]) if not cur_df.empty else -1,
        "x_node": x_node,
        "time_layer": torch.tensor(merged_nodes["time_layer"].to_numpy(dtype=np.int64), dtype=torch.long),
        "hyperedge_index_video": hyperedge_index_video,
        "hyperedge_index_country": hyperedge_index_country,
        "hyperedge_index_category": hyperedge_index_category,
        "temporal_edge_index": temporal_edge_index,
        "node_video_idx_current": torch.tensor(merged_nodes["node_video_idx_current"].to_numpy(dtype=np.int64), dtype=torch.long),
        "node_country_idx": torch.tensor(merged_nodes["node_country_idx"].to_numpy(dtype=np.int64), dtype=torch.long),
        "video_ids": current_video_ids,
        "country_vocab": country_vocab,
        "country_to_idx": {c: i for i, c in enumerate(country_vocab)},
        "local_country_vocab": local_country_vocab,
        "node_feature_cols": feature_cols,
        "y_video": y_video,
    }
    torch.save(snapshot, out_path)

    return {
        "prev_window_idx": snapshot["prev_window_idx"],
        "window_idx": snapshot["window_idx"],
        "num_nodes": int(x_node.size(0)),
        "num_current_videos": int(len(current_video_ids)),
        "num_video_hyperedges": int(hyperedge_index_video[1].max().item() + 1) if hyperedge_index_video.numel() else 0,
        "num_country_hyperedges": int(hyperedge_index_country[1].max().item() + 1) if hyperedge_index_country.numel() else 0,
        "num_category_hyperedges": int(hyperedge_index_category[1].max().item() + 1) if hyperedge_index_category.numel() else 0,
        "num_temporal_edges": int(temporal_edge_index.size(1)),
    }


def build_all_temporal_hetero_snapshots(
    temporal_dir: str,
    label_dir: str,
    out_dir: str,
    feature_cols_requested: List[str] | None = None,
) -> None:
    feature_cols_requested = feature_cols_requested or DEFAULT_NODE_FEATURE_COLS
    temporal_files = _discover_temporal_files(temporal_dir)
    label_files = _discover_label_files(label_dir)
    common_windows = sorted(w for w in temporal_files if w in label_files and (w - 1) in temporal_files)
    if not common_windows:
        raise ValueError("No usable consecutive temporal windows matched with label files.")

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    country_vocab = _extract_country_vocab(label_files)

    summaries = []
    for i, w in enumerate(common_windows, start=1):
        print(f"[INFO] [{i}/{len(common_windows)}] Building temporal snapshot for window {w:05d}")
        out_path = out_dir_path / f"temporal_hetero_snapshot_{w:05d}.pt"
        summaries.append(
            build_one_temporal_hetero_snapshot(
                prev_temporal_path=temporal_files[w - 1],
                cur_temporal_path=temporal_files[w],
                label_path=label_files[w],
                out_path=out_path,
                country_vocab=country_vocab,
                feature_cols_requested=feature_cols_requested,
            )
        )

    meta = {
        "num_snapshots": len(summaries),
        "country_vocab": country_vocab,
        "country_to_idx": {c: i for i, c in enumerate(country_vocab)},
        "node_feature_cols_requested": feature_cols_requested,
        "snapshot_prefix": "temporal_hetero_snapshot",
        "summaries": summaries,
    }
    with open(out_dir_path / "meta_temporal_hetero_snapshots.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Done. Wrote {len(summaries)} temporal snapshots to {out_dir}")


if __name__ == "__main__":
    build_all_temporal_hetero_snapshots(
        temporal_dir="./window_temporal",
        label_dir="./window_labels",
        out_dir="./temporal_hetero_hypergraph_snapshots",
    )