import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATE_COL = "collection_date"
REGION_COL = "region_code"
VIDEO_COL = "video_id"
CATEGORY_COL = "category_id"

RAW_FEATURE_COLS = ["view_count", "comment_count", "rank"]

GROUP_KEY_COLS = ["video_id", "region_code", "window_idx"]

# Output table will have one row per (video_id, country, window)
# These are the core engineered columns for the redesigned pipeline.
OUTPUT_COLUMNS = [
    "video_id",
    "country",
    "window_idx",
    "category_id",
    "days_present_in_window",
    "first_day_in_window",
    "last_day_in_window",
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
    "num_countries_for_video_in_window",
    "num_rows_for_video_in_window",
    "is_us_present_for_video_in_window",
]


# ============================================================
# HELPERS
# ============================================================

def load_meta(shard_dir: str) -> Dict[str, object]:
    meta_path = Path(shard_dir) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta.json in {shard_dir}")

    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _clean_window_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL])
    if df.empty:
        return df

    df[DATE_COL] = df[DATE_COL].dt.floor("D")
    df[VIDEO_COL] = df[VIDEO_COL].astype("string").fillna("")
    df[REGION_COL] = df[REGION_COL].astype("string").fillna("UNK")
    df[CATEGORY_COL] = pd.to_numeric(df[CATEGORY_COL], errors="coerce").fillna(-1).astype("int32")

    for col in RAW_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("float32")

    df = df[df[VIDEO_COL].str.len() > 0]
    return df


def _mode_or_first(series: pd.Series) -> int:
    if series.empty:
        return -1
    mode_vals = series.mode(dropna=True)
    if len(mode_vals) > 0:
        return int(mode_vals.iloc[0])
    return int(series.iloc[0])


def _add_percentile_features(table: pd.DataFrame) -> pd.DataFrame:
    """
    Percentiles are computed within each (country, window).
    Better rank means smaller rank number, so we percentile over -best_rank.
    """
    table = table.copy()

    group_keys = ["country", "window_idx"]

    table["rank_pct_in_country_window"] = (
        table.groupby(group_keys)["best_rank"]
        .transform(lambda s: (-s).rank(method="average", pct=True))
        .astype("float32")
    )

    table["views_pct_in_country_window"] = (
        table.groupby(group_keys)["max_view_count"]
        .transform(lambda s: s.rank(method="average", pct=True))
        .astype("float32")
    )

    table["comments_pct_in_country_window"] = (
        table.groupby(group_keys)["max_comment_count"]
        .transform(lambda s: s.rank(method="average", pct=True))
        .astype("float32")
    )

    return table


def _build_video_window_level_features(table: pd.DataFrame) -> pd.DataFrame:
    """
    Add cross-country current-window features at the (video_id, window) level,
    then broadcast them back to each country-row for that video in that window.
    """
    table = table.copy()

    video_window_stats = (
        table.groupby(["video_id", "window_idx"], sort=False)
        .agg(
            num_countries_for_video_in_window=("country", "nunique"),
            num_rows_for_video_in_window=("country", "size"),
            is_us_present_for_video_in_window=("country", lambda s: int((s == "US").any())),
        )
        .reset_index()
    )

    table = table.merge(
        video_window_stats,
        on=["video_id", "window_idx"],
        how="left",
        validate="many_to_one",
    )

    return table


# ============================================================
# CORE TABLE BUILDING
# ============================================================

def build_window_table(window_csv_path: str, window_idx: int) -> pd.DataFrame:
    """
    Build one row per (video_id, country, window_idx).

    Aggregation strategy:
    - Keep country-specific identity
    - Aggregate all daily rows inside the window
    - Preserve last-day and best-rank information
    """
    df = pd.read_csv(window_csv_path, low_memory=False)
    df = _clean_window_df(df)

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df["window_idx"] = int(window_idx)
    df["country"] = df[REGION_COL]

    # Sort so "last_*" is meaningful
    df = df.sort_values([VIDEO_COL, REGION_COL, DATE_COL], kind="stable").reset_index(drop=True)

    grouped = (
        df.groupby(GROUP_KEY_COLS, sort=False)
        .agg(
            category_id=(CATEGORY_COL, _mode_or_first),
            days_present_in_window=(DATE_COL, "nunique"),
            first_day_in_window=(DATE_COL, "min"),
            last_day_in_window=(DATE_COL, "max"),
            mean_view_count=("view_count", "mean"),
            max_view_count=("view_count", "max"),
            last_view_count=("view_count", "last"),
            mean_comment_count=("comment_count", "mean"),
            max_comment_count=("comment_count", "max"),
            last_comment_count=("comment_count", "last"),
            mean_rank=("rank", "mean"),
            best_rank=("rank", "min"),
            last_rank=("rank", "last"),
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={REGION_COL: "country"})

    # Transform features
    grouped["log_mean_views"] = np.log1p(grouped["mean_view_count"].clip(lower=0)).astype("float32")
    grouped["log_max_views"] = np.log1p(grouped["max_view_count"].clip(lower=0)).astype("float32")
    grouped["log_mean_comments"] = np.log1p(grouped["mean_comment_count"].clip(lower=0)).astype("float32")
    grouped["log_max_comments"] = np.log1p(grouped["max_comment_count"].clip(lower=0)).astype("float32")

    grouped = _add_percentile_features(grouped)
    grouped = _build_video_window_level_features(grouped)

    # Cast and reorder
    numeric_float_cols = [
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
    numeric_int_cols = [
        "window_idx",
        "category_id",
        "days_present_in_window",
        "num_countries_for_video_in_window",
        "num_rows_for_video_in_window",
        "is_us_present_for_video_in_window",
    ]

    for col in numeric_float_cols:
        grouped[col] = grouped[col].astype("float32")
    for col in numeric_int_cols:
        grouped[col] = grouped[col].astype("int32")

    grouped["video_id"] = grouped["video_id"].astype("string")
    grouped["country"] = grouped["country"].astype("string")

    return grouped[OUTPUT_COLUMNS].copy()


def build_all_window_tables(
    shard_dir: str,
    out_dir: str,
    prefer_parquet: bool = True,
) -> None:
    """
    Read all window_XXXXX.csv files and write one engineered table per window.

    Output:
      table_00000.parquet / table_00000.csv
      ...
      meta_tables.json
    """
    shard_dir_path = Path(shard_dir)
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    _ = load_meta(shard_dir)  # validates presence of meta.json

    window_files: List[Path] = sorted(shard_dir_path.glob("window_*.csv"))
    if not window_files:
        raise ValueError(f"No window shard files found in {shard_dir}")

    all_countries = set()
    all_categories = set()
    table_summaries = []

    for i, window_csv in enumerate(window_files):
        window_idx = int(window_csv.stem.split("_")[-1])
        print(f"[INFO] [{i+1}/{len(window_files)}] Building table for window {window_idx:05d}")

        table = build_window_table(str(window_csv), window_idx)

        all_countries.update(table["country"].dropna().astype(str).unique().tolist())
        all_categories.update(table["category_id"].dropna().astype(int).unique().tolist())

        if prefer_parquet:
            out_path = out_dir_path / f"table_{window_idx:05d}.parquet"
            table.to_parquet(out_path, index=False)
        else:
            out_path = out_dir_path / f"table_{window_idx:05d}.csv"
            table.to_csv(out_path, index=False)

        table_summaries.append(
            {
                "window_idx": window_idx,
                "num_rows": int(len(table)),
                "num_unique_videos": int(table["video_id"].nunique()) if not table.empty else 0,
                "num_unique_countries": int(table["country"].nunique()) if not table.empty else 0,
            }
        )

    meta_tables = {
        "row_definition": "one row per (video_id, country, window_idx)",
        "output_columns": OUTPUT_COLUMNS,
        "countries": sorted(all_countries),
        "categories": sorted(all_categories),
        "num_windows_written": len(window_files),
        "file_format": "parquet" if prefer_parquet else "csv",
        "window_summaries": table_summaries,
    }

    with open(out_dir_path / "meta_tables.json", "w", encoding="utf-8") as f:
        json.dump(meta_tables, f, indent=2)

    print(f"[INFO] Window tables complete. Output dir: {out_dir_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # Example usage:
    #   python window_tables.py
    shard_dir = "./window_shards(1)"
    out_dir = "./window_tables"
    prefer_parquet = True

    build_all_window_tables(
        shard_dir=shard_dir,
        out_dir=out_dir,
        prefer_parquet=prefer_parquet,
    )