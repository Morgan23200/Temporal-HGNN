import math
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

DATE_COL = "collection_date"
REGION_COL = "region_code"
VIDEO_COL = "video_id"
CATEGORY_COL = "category_id"

FEATURE_COLS = ["view_count", "comment_count", "rank"]

USECOLS = [
    VIDEO_COL,
    DATE_COL,
    CATEGORY_COL,
    REGION_COL,
    *FEATURE_COLS,
]

DTYPES = {
    VIDEO_COL: "string",
    CATEGORY_COL: "Int32",
    REGION_COL: "string",
    "view_count": "float32",
    "comment_count": "float32",
    "rank": "float32",
}


# ============================================================
# HELPERS
# ============================================================

def _clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Standardize types and remove unusable rows."""
    chunk = chunk.copy()

    chunk[DATE_COL] = pd.to_datetime(chunk[DATE_COL], errors="coerce")
    chunk = chunk.dropna(subset=[DATE_COL])
    if chunk.empty:
        return chunk

    chunk[DATE_COL] = chunk[DATE_COL].dt.floor("D")

    chunk[VIDEO_COL] = chunk[VIDEO_COL].astype("string").fillna("")
    chunk[REGION_COL] = chunk[REGION_COL].astype("string").fillna("UNK")
    chunk[CATEGORY_COL] = pd.to_numeric(
        chunk[CATEGORY_COL], errors="coerce"
    ).fillna(-1).astype("int32")

    for col in FEATURE_COLS:
        chunk[col] = pd.to_numeric(chunk[col], errors="coerce").fillna(0).astype("float32")

    # Drop blank video ids; they are not useful for downstream tracking
    chunk = chunk[chunk[VIDEO_COL].str.len() > 0]
    return chunk


def find_date_range(
    csv_path: str,
    date_col: str = DATE_COL,
    chunksize: int = 1_000_000,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Scan the raw CSV once to find min/max valid date."""
    min_date: Optional[pd.Timestamp] = None
    max_date: Optional[pd.Timestamp] = None

    for chunk in pd.read_csv(
        csv_path,
        usecols=[date_col],
        chunksize=chunksize,
        low_memory=False,
    ):
        dates = pd.to_datetime(chunk[date_col], errors="coerce").dropna()
        if dates.empty:
            continue

        cmin = dates.min().floor("D")
        cmax = dates.max().floor("D")

        min_date = cmin if min_date is None else min(min_date, cmin)
        max_date = cmax if max_date is None else max(max_date, cmax)

    if min_date is None or max_date is None:
        raise ValueError(f"No valid dates found in column '{date_col}'")

    return min_date, max_date


def _build_meta(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    window_size_days: int,
    num_windows: int,
) -> Dict[str, object]:
    return {
        "start_date": str(start_date.date()),
        "end_date": str(end_date.date()),
        "window_size_days": int(window_size_days),
        "num_windows": int(num_windows),
        "date_col": DATE_COL,
        "region_col": REGION_COL,
        "video_col": VIDEO_COL,
        "category_col": CATEGORY_COL,
        "feature_cols": FEATURE_COLS,
        "window_definition": "floor((collection_date - start_date).days / window_size_days)",
        "row_definition": "raw trending row before window-table aggregation",
    }


# ============================================================
# MAIN SHARDING LOGIC
# ============================================================

def shard_csv_by_time_window(
    csv_path: str,
    shard_dir: str,
    window_size_days: int = 7,
    chunksize: int = 1_000_000,
) -> None:
    """
    Split the raw CSV into window_XXXXX.csv files.

    Each output shard contains raw rows whose collection_date falls inside
    that window. No graph objects are created here.
    """
    if window_size_days <= 0:
        raise ValueError("window_size_days must be a positive integer")

    shard_dir_path = Path(shard_dir)
    shard_dir_path.mkdir(parents=True, exist_ok=True)

    start_date, end_date = find_date_range(
        csv_path=csv_path,
        date_col=DATE_COL,
        chunksize=chunksize,
    )

    total_days = (end_date - start_date).days + 1
    num_windows = math.ceil(total_days / window_size_days)

    print(f"[INFO] Global date range: {start_date.date()} -> {end_date.date()}")
    print(f"[INFO] Window size (days): {window_size_days}")
    print(f"[INFO] Number of windows: {num_windows}")

    header_written = set()

    reader = pd.read_csv(
        csv_path,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=chunksize,
        low_memory=False,
    )

    for chunk_idx, chunk in enumerate(reader):
        print(f"[INFO] Processing chunk {chunk_idx}")

        chunk = _clean_chunk(chunk)
        if chunk.empty:
            continue

        day_offset = (chunk[DATE_COL] - start_date).dt.days.astype("int32")
        chunk["window_idx"] = (day_offset // window_size_days).astype("int32")

        for w in chunk["window_idx"].unique():
            sub = chunk.loc[chunk["window_idx"] == w, USECOLS]

            out_path = shard_dir_path / f"window_{int(w):05d}.csv"
            write_header = out_path not in header_written and not out_path.exists()

            sub.to_csv(
                out_path,
                mode="a",
                header=write_header,
                index=False,
            )
            header_written.add(out_path)

    meta = _build_meta(
        start_date=start_date,
        end_date=end_date,
        window_size_days=window_size_days,
        num_windows=num_windows,
    )
    pd.Series(meta).to_json(shard_dir_path / "meta.json", indent=2)
    print(f"[INFO] Sharding complete. Output dir: {shard_dir_path}")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    # Example usage:
    #   python window_shards.py
    csv_path = "most_popular.csv"
    shard_dir = "./window_shards(1)"
    window_size_days = 7
    chunksize = 1_000_000

    shard_csv_by_time_window(
        csv_path=csv_path,
        shard_dir=shard_dir,
        window_size_days=window_size_days,
        chunksize=chunksize,
    )