#!/usr/bin/env python3
"""
prepare_data.py – Load NER‑annotated parquet file(s) from a folder,
validate, normalise offset_mapping, split 80/10/10, save as HuggingFace
Arrow datasets.

Usage
-----
    python prepare_data.py \
        --input ./parquet_folder \
        --output-dir ./prepared_data \
        --seed 42 \
        --subsample-frac 1.0
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl
from datasets import Dataset, DatasetDict

REQUIRED_COLUMNS = {"texts", "inputs", "outputs", "attention_mask", "offset_mapping"}
SEQ_COLUMNS = ["inputs", "outputs", "attention_mask", "offset_mapping"]
MAX_SEQ_LEN = 512

ID2LABEL = {0: "O", 1: "B-ENT", 2: "I-ENT"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
NUM_LABELS = len(ID2LABEL)

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Parquet discovery & loading
# ──────────────────────────────────────────────────────────────────────
def discover_parquets(input_path: str) -> List[Path]:
    """Return sorted list of *.parquet files.

    ``input_path`` may be a single file *or* a directory.
    """
    p = Path(input_path)
    if p.is_file() and p.suffix == ".parquet":
        return [p]
    if p.is_dir():
        files = sorted(p.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No *.parquet files found in {p}")
        return files
    raise FileNotFoundError(
        f"{p} is neither a parquet file nor a directory containing parquet files."
    )


def load_parquets(paths: List[Path]) -> pl.DataFrame:
    """Read and concatenate multiple parquet files with Polars."""
    frames: List[pl.DataFrame] = []
    for path in paths:
        log.info("  Reading %s …", path.name)
        frames.append(pl.read_parquet(path))
    df = pl.concat(frames, how="vertical_relaxed")
    log.info("Loaded %d rows from %d file(s).", len(df), len(paths))
    return df


# ──────────────────────────────────────────────────────────────────────
# Offset‑mapping normalisation
# ──────────────────────────────────────────────────────────────────────
def _normalise_single_offset(offset: Any) -> List[int]:
    """Convert a single offset element to ``[start, end]``.

    Handles:
      - dict  ``{"start": s, "end": e}``
      - struct  (Polars returns these as dicts)
      - tuple / list  ``(s, e)`` or ``[s, e]``
      - numpy structured array
    """
    if isinstance(offset, dict):
        return [int(offset["start"]), int(offset["end"])]
    if isinstance(offset, (list, tuple)):
        return [int(offset[0]), int(offset[1])]
    if hasattr(offset, "item"):
        arr = offset.tolist()
        if isinstance(arr, dict):
            return [int(arr["start"]), int(arr["end"])]
        return [int(arr[0]), int(arr[1])]
    raise TypeError(f"Unsupported offset type: {type(offset)} – value: {offset!r}")


def normalise_offset_mapping(offsets: Any) -> List[List[int]]:
    """Normalise an entire row's ``offset_mapping`` to ``list[list[int]]``."""
    return [_normalise_single_offset(o) for o in offsets]


# ──────────────────────────────────────────────────────────────────────
# Schema validation & preparation  (Polars)
# ──────────────────────────────────────────────────────────────────────
def validate_and_prepare(df: pl.DataFrame) -> pl.DataFrame:
    """Validate schema, normalise offsets, enforce length constraints.

    Returns a Polars DataFrame with canonical column names:
      text, input_ids, labels, attention_mask, offset_mapping
    """
    # ── 1. Check required columns ───────────────────────────────────
    present = set(df.columns)
    missing = REQUIRED_COLUMNS - present
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    log.info("All required columns present.")

    # Drop spurious index columns
    drop_cols = [c for c in df.columns if c.startswith("Unnamed") or c.startswith("__index")]
    if drop_cols:
        log.info("Dropping columns: %s", drop_cols)
        df = df.drop(drop_cols)

    # ── 2. Rename to canonical names ────────────────────────────────
    df = df.rename(
        {"texts": "text", "inputs": "input_ids", "outputs": "labels"}
    )

    # ── 3. Normalise offset_mapping ─────────────────────────────────
    #    Polars map_elements on a struct/list column; returns list[list[int]]
    log.info("Normalising offset_mapping to list[list[int]] …")
    t0 = time.perf_counter()
    df = df.with_columns(
        pl.col("offset_mapping")
        .map_elements(normalise_offset_mapping, return_dtype=pl.List(pl.List(pl.Int64)))
        .alias("offset_mapping")
    )
    log.info("  Normalisation done in %.2f s", time.perf_counter() - t0)

    # ── 4. Ensure list columns are plain List[Int64] ────────────────
    #    Polars parquet reader usually infers correctly; cast if needed.
    for col in ["input_ids", "labels", "attention_mask"]:
        if df[col].dtype != pl.List(pl.Int64):
            df = df.with_columns(pl.col(col).cast(pl.List(pl.Int64)))

    # ── 5. Per‑row length consistency ───────────────────────────────
    df = df.with_columns(
        pl.col("input_ids").list.len().alias("_len_ids"),
        pl.col("labels").list.len().alias("_len_lab"),
        pl.col("attention_mask").list.len().alias("_len_att"),
        pl.col("offset_mapping").list.len().alias("_len_off"),
    )
    df = df.with_columns(
        pl.min_horizontal("_len_ids", "_len_lab", "_len_att", "_len_off").alias("_min_len")
    )

    n_mismatch = df.filter(
        (pl.col("_len_ids") != pl.col("_min_len"))
        | (pl.col("_len_lab") != pl.col("_min_len"))
        | (pl.col("_len_att") != pl.col("_min_len"))
        | (pl.col("_len_off") != pl.col("_min_len"))
    ).height
    if n_mismatch > 0:
        log.warning(
            "%d rows have mismatched sequence lengths – truncating to min.", n_mismatch
        )

    # Truncate all sequence columns to _min_len (also caps at MAX_SEQ_LEN)
    cap = pl.min_horizontal(pl.col("_min_len"), pl.lit(MAX_SEQ_LEN))
    df = df.with_columns(
        pl.col("input_ids").list.head(cap).alias("input_ids"),
        pl.col("labels").list.head(cap).alias("labels"),
        pl.col("attention_mask").list.head(cap).alias("attention_mask"),
        pl.col("offset_mapping").list.head(cap).alias("offset_mapping"),
    )

    n_long = df.filter(pl.col("_min_len") > MAX_SEQ_LEN).height
    if n_long > 0:
        log.warning("%d rows exceeded %d tokens – truncated.", n_long, MAX_SEQ_LEN)

    # Drop helper columns
    df = df.drop(["_len_ids", "_len_lab", "_len_att", "_len_off", "_min_len"])

    # ── 6. Log statistics ───────────────────────────────────────────
    final_lens = df.select(pl.col("input_ids").list.len().alias("slen"))["slen"]
    log.info(
        "Sequence lengths — min: %d, max: %d, mean: %.1f, median: %.1f",
        final_lens.min(),
        final_lens.max(),
        final_lens.mean(),
        final_lens.median(),
    )

    # Label distribution  — explode labels column in Polars
    label_counts = (
        df.select(pl.col("labels").explode())
        .group_by("labels")
        .len()
        .sort("labels")
    )
    total_tokens = label_counts["len"].sum()
    log.info("Label distribution (%d total tokens):", total_tokens)
    for row in label_counts.iter_rows(named=True):
        lid = row["labels"]
        cnt = row["len"]
        pct = 100.0 * cnt / total_tokens
        label_name = ID2LABEL.get(lid, f"UNK-{lid}")
        log.info("  %d (%s): %d  (%.2f%%)", lid, label_name, cnt, pct)

    return df


# ──────────────────────────────────────────────────────────────────────
# Splitting  (Polars)
# ──────────────────────────────────────────────────────────────────────
def split_dataframe(
    df: pl.DataFrame, seed: int
) -> Dict[str, pl.DataFrame]:
    """Deterministic 80 / 10 / 10 split."""
    n = len(df)
    indices = np.arange(n)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])

    splits: Dict[str, pl.DataFrame] = {
        "train": df[indices[:n_train].tolist()],
        "val": df[indices[n_train : n_train + n_val].tolist()],
        "test": df[indices[n_train + n_val :].tolist()],
    }
    for name, sdf in splits.items():
        log.info("Split %-5s : %d rows", name, len(sdf))
    return splits


# ──────────────────────────────────────────────────────────────────────
# Polars DataFrame → HuggingFace Dataset
# ──────────────────────────────────────────────────────────────────────
def df_to_hf_dataset(df: pl.DataFrame) -> Dataset:
    """Convert a prepared Polars DataFrame to a HuggingFace ``Dataset``.

    Polars ``.to_list()`` on list columns yields native Python lists,
    exactly what ``Dataset.from_dict`` needs.
    """
    return Dataset.from_dict(
        {
            "text": df["text"].to_list(),
            "input_ids": df["input_ids"].to_list(),
            "labels": df["labels"].to_list(),
            "attention_mask": df["attention_mask"].to_list(),
            "offset_mapping": df["offset_mapping"].to_list(),
        }
    )


# ──────────────────────────────────────────────────────────────────────
# Smoke test
# ──────────────────────────────────────────────────────────────────────
def smoke_test(output_dir: str) -> None:
    """Reload saved dataset and verify integrity."""
    log.info("Running smoke test on saved dataset …")
    ds = DatasetDict.load_from_disk(output_dir)

    expected_splits = {"train", "val", "test"}
    assert set(ds.keys()) == expected_splits, (
        f"Expected splits {expected_splits}, got {set(ds.keys())}"
    )

    required_cols = {"text", "input_ids", "labels", "attention_mask", "offset_mapping"}
    for split_name, split_ds in ds.items():
        assert required_cols.issubset(set(split_ds.column_names)), (
            f"[{split_name}] missing columns: "
            f"{required_cols - set(split_ds.column_names)}"
        )
        sample = split_ds[0]
        seq_len = len(sample["input_ids"])
        assert seq_len <= MAX_SEQ_LEN, (
            f"[{split_name}] first sample has {seq_len} tokens (> {MAX_SEQ_LEN})"
        )
        assert len(sample["labels"]) == seq_len
        assert len(sample["attention_mask"]) == seq_len
        assert len(sample["offset_mapping"]) == seq_len

        om0 = sample["offset_mapping"][0]
        assert isinstance(om0, list) and len(om0) == 2, (
            f"[{split_name}] offset_mapping[0] has unexpected format: {om0!r}"
        )

        log.info(
            "  ✓ %-5s — %d rows, first sample %d tokens",
            split_name,
            len(split_ds),
            seq_len,
        )

    log.info("Smoke test passed ✓")


# ──────────────────────────────────────────────────────────────────────
# CLI & main
# ──────────────────────────────────────────────────────────────────────
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare NER dataset from parquet(s) for CamemBERT fine‑tuning."
    )
    p.add_argument(
        "--input",
        type=str,
        default="./parquets",
        help="Path to a single .parquet file or a directory containing *.parquet files.",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="./prepared_data",
        help="Directory to save the HF Arrow dataset.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splitting.",
    )
    p.add_argument(
        "--subsample-frac",
        type=float,
        default=1.0,
        help="Fraction of data to keep (for quick pipeline testing). "
        "Applied before splitting.",
    )
    return p.parse_args(argv)


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv)
    log.info("Arguments: %s", vars(args))

    # ── Discover & load parquets ────────────────────────────────────
    parquet_files = discover_parquets(args.input)
    log.info("Found %d parquet file(s) in %s:", len(parquet_files), args.input)
    for pf in parquet_files:
        log.info("  • %s", pf.name)

    t0 = time.perf_counter()
    df = load_parquets(parquet_files)
    log.info("Parquet loading took %.2f s", time.perf_counter() - t0)

    # ── Optional subsample ──────────────────────────────────────────
    if 0.0 < args.subsample_frac < 1.0:
        n_sample = max(1, int(len(df) * args.subsample_frac))
        df = df.sample(n=n_sample, seed=args.seed, shuffle=True)
        log.info("Subsampled to %d rows (frac=%.4f)", len(df), args.subsample_frac)

    # ── Validate & prepare ──────────────────────────────────────────
    t0 = time.perf_counter()
    df = validate_and_prepare(df)
    log.info("Validation & preparation took %.2f s", time.perf_counter() - t0)

    # ── Split ───────────────────────────────────────────────────────
    splits = split_dataframe(df, seed=args.seed)

    # ── Convert & save ──────────────────────────────────────────────
    t0 = time.perf_counter()
    hf_splits = {name: df_to_hf_dataset(sdf) for name, sdf in splits.items()}
    ds_dict = DatasetDict(hf_splits)

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    log.info("Saving DatasetDict to %s …", out_path)
    ds_dict.save_to_disk(str(out_path))
    log.info("Conversion & save took %.2f s", time.perf_counter() - t0)

    # ── Smoke test ──────────────────────────────────────────────────
    smoke_test(str(out_path))

    log.info("Done. Dataset ready at: %s", out_path.resolve())


if __name__ == "__main__":
    main()
