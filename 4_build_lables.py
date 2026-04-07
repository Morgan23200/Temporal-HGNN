import json
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd


# ============================================================
# IO HELPERS
# ============================================================

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported file format: {path}")


def _write_table(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


def _discover_temporal_files(temporal_dir: str) -> List[Path]:
    temporal_dir_path = Path(temporal_dir)

    parquet_files = sorted(temporal_dir_path.glob("temporal_*.parquet"))
    if parquet_files:
        return parquet_files

    csv_files = sorted(temporal_dir_path.glob("temporal_*.csv"))
    if csv_files:
        return csv_files

    raise ValueError(f"No temporal_*.parquet or temporal_*.csv files found in {temporal_dir}")


def _extract_window_idx(path: Path) -> int:
    return int(path.stem.split("_")[-1])


def _safe_country_col(country: str) -> str:
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(country))
    return f"label_country_{cleaned}"


# ============================================================
# LABEL BUILDING
# ============================================================

def _group_video_current_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse row-level (video_id, country, window) temporal table into
    one row per (video_id, window), keeping useful current-window summaries.
    """
    df = df.copy()
    df["video_id"] = df["video_id"].astype("string")
    df["country"] = df["country"].astype("string")
    df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce").astype("int32")

    grouped = (
        df.groupby(["video_id", "window_idx"], sort=False)
        .agg(
            num_current_countries=("country", "nunique"),
            current_countries_serialized=("country", lambda s: "|".join(sorted(set(map(str, s))))),
            current_country_list=("country", lambda s: sorted(set(map(str, s)))),
            best_rank_current=("best_rank", "min"),
            mean_rank_current=("mean_rank", "mean"),
            max_view_count_current=("max_view_count", "max"),
            mean_view_count_current=("mean_view_count", "mean"),
            max_comment_count_current=("max_comment_count", "max"),
            mean_comment_count_current=("mean_comment_count", "mean"),
            category_mode_current=("category_id", lambda s: int(s.mode().iloc[0]) if len(s.mode()) > 0 else int(s.iloc[0])),
            num_prev_exists_rows=("prev_exists_same_country", "sum"),
        )
        .reset_index()
    )

    return grouped


def build_next_window_labels(
    temporal_dir: str,
    out_dir: str,
    prefer_parquet: bool = True,
) -> None:
    """
    For each input window t, build one row per (video_id, t) with a multi-hot label
    over countries in window t+1.

    Output excludes the last window because it has no next-window target.
    """
    temporal_files = _discover_temporal_files(temporal_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(temporal_files)} temporal tables.")

    # Read all windows first
    per_window_raw: Dict[int, pd.DataFrame] = {}
    all_countries: Set[str] = set()

    for path in temporal_files:
        window_idx = _extract_window_idx(path)
        df = _read_table(path)

        if df.empty:
            df = df.copy()
            if "video_id" not in df.columns:
                df["video_id"] = pd.Series(dtype="string")
            if "country" not in df.columns:
                df["country"] = pd.Series(dtype="string")
            if "window_idx" not in df.columns:
                df["window_idx"] = pd.Series(dtype="int32")

        df["video_id"] = df["video_id"].astype("string")
        df["country"] = df["country"].astype("string")
        df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce").fillna(window_idx).astype("int32")

        per_window_raw[window_idx] = df
        all_countries.update(set(map(str, df["country"].dropna().unique().tolist())))

    all_window_indices = sorted(per_window_raw.keys())
    country_vocab = sorted(all_countries)
    country_to_idx = {country: idx for idx, country in enumerate(country_vocab)}

    summaries = []

    # Build labels for windows that have t+1
    usable_windows = [w for w in all_window_indices if (w + 1) in per_window_raw]

    for i, w in enumerate(usable_windows):
        cur_df = per_window_raw[w]
        nxt_df = per_window_raw[w + 1]

        cur_video = _group_video_current_state(cur_df)

        next_country_sets = (
            nxt_df.groupby("video_id", sort=False)["country"]
            .agg(lambda s: sorted(set(map(str, s))))
            .reset_index()
            .rename(columns={"country": "next_country_list"})
        )

        labeled = cur_video.merge(
            next_country_sets,
            on="video_id",
            how="left",
            validate="one_to_one",
        )

        labeled["next_country_list"] = labeled["next_country_list"].apply(
            lambda x: x if isinstance(x, list) else []
        )
        labeled["next_countries_serialized"] = labeled["next_country_list"].apply(lambda x: "|".join(x))

        # Multi-hot label columns
        for country in country_vocab:
            col = _safe_country_col(country)
            labeled[col] = labeled["next_country_list"].apply(lambda lst, c=country: int(c in lst)).astype("int8")

        labeled["num_next_countries"] = labeled["next_country_list"].apply(len).astype("int32")
        labeled["has_any_next_country"] = (labeled["num_next_countries"] > 0).astype("int8")

        # Reorder
        label_cols = [_safe_country_col(c) for c in country_vocab]
        ordered_cols = [
            "video_id",
            "window_idx",
            "num_current_countries",
            "current_countries_serialized",
            "next_countries_serialized",
            "num_next_countries",
            "has_any_next_country",
            "best_rank_current",
            "mean_rank_current",
            "max_view_count_current",
            "mean_view_count_current",
            "max_comment_count_current",
            "mean_comment_count_current",
            "category_mode_current",
            "num_prev_exists_rows",
            *label_cols,
        ]
        labeled = labeled[ordered_cols].copy()

        suffix = ".parquet" if prefer_parquet else ".csv"
        out_path = out_dir_path / f"labels_{w:05d}{suffix}"
        _write_table(labeled, out_path)

        summaries.append(
            {
                "window_idx": int(w),
                "num_video_samples": int(len(labeled)),
                "num_positive_samples": int(labeled["has_any_next_country"].sum()),
                "avg_num_next_countries": float(labeled["num_next_countries"].mean()) if len(labeled) > 0 else 0.0,
            }
        )

        print(f"[INFO] [{i+1}/{len(usable_windows)}] Wrote {out_path.name}")

    meta = {
        "task_definition": "for each (video_id, window_idx=t), predict countries where the video appears at t+1",
        "input_dir": str(Path(temporal_dir).resolve()),
        "num_input_windows": len(all_window_indices),
        "num_labeled_windows": len(usable_windows),
        "country_vocab": country_vocab,
        "country_to_idx": country_to_idx,
        "label_column_prefix": "label_country_",
        "file_format": "parquet" if prefer_parquet else "csv",
        "window_summaries": summaries,
    }

    with open(out_dir_path / "meta_labels.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Label build complete. Output dir: {out_dir_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    temporal_dir = "./window_temporal"
    out_dir = "./window_labels"
    prefer_parquet = True

    build_next_window_labels(
        temporal_dir=temporal_dir,
        out_dir=out_dir,
        prefer_parquet=prefer_parquet,
    )