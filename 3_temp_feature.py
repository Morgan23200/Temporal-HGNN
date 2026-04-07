import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

KEY_COLS = ["video_id", "country", "window_idx"]

BASE_LOOKBACK_COLS = [
    "mean_view_count",
    "max_view_count",
    "last_view_count",
    "mean_comment_count",
    "max_comment_count",
    "last_comment_count",
    "mean_rank",
    "best_rank",
    "last_rank",
]

TEMPORAL_OUTPUT_EXTRA_COLS = [
    "prev_exists_same_country",
    "prev_mean_view_count",
    "prev_max_view_count",
    "prev_last_view_count",
    "prev_mean_comment_count",
    "prev_max_comment_count",
    "prev_last_comment_count",
    "prev_mean_rank",
    "prev_best_rank",
    "prev_last_rank",
    "delta_mean_view_count",
    "delta_max_view_count",
    "delta_last_view_count",
    "delta_mean_comment_count",
    "delta_max_comment_count",
    "delta_last_comment_count",
    "delta_mean_rank",
    "delta_best_rank",
    "delta_last_rank",
    "streak_same_country",
]


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


def _discover_table_files(table_dir: str) -> List[Path]:
    table_dir_path = Path(table_dir)

    parquet_files = sorted(table_dir_path.glob("table_*.parquet"))
    if parquet_files:
        return parquet_files

    csv_files = sorted(table_dir_path.glob("table_*.csv"))
    if csv_files:
        return csv_files

    raise ValueError(f"No table_*.parquet or table_*.csv files found in {table_dir}")


def _extract_window_idx(path: Path) -> int:
    return int(path.stem.split("_")[-1])


# ============================================================
# DTYPE / CLEANUP HELPERS
# ============================================================

def _ensure_base_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["video_id"] = df["video_id"].astype("string")
    df["country"] = df["country"].astype("string")
    df["window_idx"] = pd.to_numeric(df["window_idx"], errors="coerce").fillna(-1).astype("int32")

    int_cols = [
        "category_id",
        "days_present_in_window",
        "num_countries_for_video_in_window",
        "num_rows_for_video_in_window",
        "is_us_present_for_video_in_window",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int32")

    float_cols = [
        "mean_view_count",
        "max_view_count",
        "last_view_count",
        "mean_comment_count",
        "max_comment_count",
        "last_comment_count",
        "mean_rank",
        "best_rank",
        "last_rank",
        "log_mean_views",
        "log_max_views",
        "log_mean_comments",
        "log_max_comments",
        "rank_pct_in_country_window",
        "views_pct_in_country_window",
        "comments_pct_in_country_window",
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    return df


def _add_empty_prev_features(cur_df: pd.DataFrame) -> pd.DataFrame:
    cur_df = cur_df.copy()
    cur_df["prev_exists_same_country"] = np.int32(0)

    for col in BASE_LOOKBACK_COLS:
        cur_df[f"prev_{col}"] = np.float32(0.0)
        cur_df[f"delta_{col}"] = cur_df[col].astype("float32")

    cur_df["streak_same_country"] = np.int32(1)
    return cur_df


def _build_prev_lookup(prev_temporal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only the columns needed for looking up window t from window t-1.
    Shift previous window_idx forward by 1 so it can merge directly onto current window.
    """
    keep_cols = ["video_id", "country", "window_idx", "streak_same_country", *BASE_LOOKBACK_COLS]
    prev_lookup = prev_temporal_df[keep_cols].copy()
    prev_lookup["window_idx"] = prev_lookup["window_idx"].astype("int32") + 1

    rename_map = {col: f"prev_{col}" for col in BASE_LOOKBACK_COLS}
    rename_map["streak_same_country"] = "prev_streak_same_country"

    prev_lookup = prev_lookup.rename(columns=rename_map)
    return prev_lookup


def _merge_prev_window(cur_df: pd.DataFrame, prev_temporal_df: pd.DataFrame) -> pd.DataFrame:
    cur_df = cur_df.copy()
    prev_lookup = _build_prev_lookup(prev_temporal_df)

    merged = cur_df.merge(
        prev_lookup,
        on=["video_id", "country", "window_idx"],
        how="left",
        validate="one_to_one",
    )

    merged["prev_exists_same_country"] = merged["prev_streak_same_country"].notna().astype("int32")

    for col in BASE_LOOKBACK_COLS:
        prev_col = f"prev_{col}"
        merged[prev_col] = pd.to_numeric(merged[prev_col], errors="coerce").fillna(0).astype("float32")
        merged[f"delta_{col}"] = (
            merged[col].astype("float32") - merged[prev_col]
        ).astype("float32")

    merged["prev_streak_same_country"] = (
        pd.to_numeric(merged["prev_streak_same_country"], errors="coerce")
        .fillna(0)
        .astype("int32")
    )

    merged["streak_same_country"] = np.where(
        merged["prev_exists_same_country"].eq(1),
        merged["prev_streak_same_country"] + 1,
        1,
    ).astype("int32")

    merged = merged.drop(columns=["prev_streak_same_country"])
    return merged


def _finalize_temporal_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in TEMPORAL_OUTPUT_EXTRA_COLS:
        if col.startswith("prev_") and col != "prev_exists_same_country":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")
        elif col.startswith("delta_"):
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    df["prev_exists_same_country"] = pd.to_numeric(
        df["prev_exists_same_country"], errors="coerce"
    ).fillna(0).astype("int32")

    df["streak_same_country"] = pd.to_numeric(
        df["streak_same_country"], errors="coerce"
    ).fillna(1).astype("int32")

    return df


# ============================================================
# MAIN STREAMING LOGIC
# ============================================================

def build_temporal_features_for_all_tables(
    table_dir: str,
    out_dir: str,
    prefer_parquet: bool = True,
) -> None:
    """
    Streaming / window-by-window temporal feature builder.

    Memory usage stays bounded because it only keeps:
    - current window table
    - previous temporal window table

    Input:
        table_XXXXX.parquet or table_XXXXX.csv
        one row per (video_id, country, window)

    Output:
        temporal_XXXXX.parquet or temporal_XXXXX.csv
        same rows + previous-window features + deltas + streak
    """
    table_files = _discover_table_files(table_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(table_files)} window tables.")

    summaries = []
    prev_temporal_df: Optional[pd.DataFrame] = None

    for i, path in enumerate(table_files):
        window_idx = _extract_window_idx(path)
        print(f"[INFO] [{i+1}/{len(table_files)}] Processing window {window_idx:05d}")

        cur_df = _read_table(path)
        cur_df = _ensure_base_dtypes(cur_df)

        if cur_df.empty:
            temporal_df = cur_df.copy()
            for col in TEMPORAL_OUTPUT_EXTRA_COLS:
                if col == "prev_exists_same_country" or col == "streak_same_country":
                    temporal_df[col] = pd.Series(dtype="int32")
                else:
                    temporal_df[col] = pd.Series(dtype="float32")
        elif prev_temporal_df is None:
            temporal_df = _add_empty_prev_features(cur_df)
            temporal_df = _finalize_temporal_dtypes(temporal_df)
        else:
            temporal_df = _merge_prev_window(cur_df, prev_temporal_df)
            temporal_df = _finalize_temporal_dtypes(temporal_df)

        suffix = ".parquet" if prefer_parquet else ".csv"
        out_path = out_dir_path / f"temporal_{window_idx:05d}{suffix}"
        _write_table(temporal_df, out_path)

        summaries.append(
            {
                "window_idx": int(window_idx),
                "num_rows": int(len(temporal_df)),
                "num_prev_exists": int(temporal_df["prev_exists_same_country"].sum()) if not temporal_df.empty else 0,
                "num_unique_videos": int(temporal_df["video_id"].nunique()) if not temporal_df.empty else 0,
                "num_unique_countries": int(temporal_df["country"].nunique()) if not temporal_df.empty else 0,
            }
        )

        # Only keep what is needed for the next iteration.
        prev_temporal_df = temporal_df[
            ["video_id", "country", "window_idx", "streak_same_country", *BASE_LOOKBACK_COLS]
        ].copy()

        print(f"[INFO] Wrote {out_path.name}")

    meta = {
        "row_definition": "one row per (video_id, country, window_idx) with temporal lookback to t-1",
        "base_input_dir": str(Path(table_dir).resolve()),
        "num_windows_written": len(table_files),
        "file_format": "parquet" if prefer_parquet else "csv",
        "implementation": "streaming_window_by_window",
        "lookback_definition": "previous row for same (video_id, country) at window_idx - 1",
        "base_lookback_cols": BASE_LOOKBACK_COLS,
        "added_temporal_columns": TEMPORAL_OUTPUT_EXTRA_COLS,
        "window_summaries": summaries,
    }

    with open(out_dir_path / "meta_temporal.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[INFO] Temporal feature build complete. Output dir: {out_dir_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    table_dir = "./window_tables"
    out_dir = "./window_temporal"
    prefer_parquet = True

    build_temporal_features_for_all_tables(
        table_dir=table_dir,
        out_dir=out_dir,
        prefer_parquet=prefer_parquet,
    )