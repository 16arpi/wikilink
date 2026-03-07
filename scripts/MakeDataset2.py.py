#!/usr/bin/env python3
"""
Produce a CamemBERT-compatible NER-style dataset from a parquet Wikipedia corpus

Output schema (per row in each Parquet file):
  texts             : cleaned text string (link display kept, links removed)
  inputs            : list[int] (tokenizer input_ids, incl. special tokens)
  outputs           : list[int] (token labels: 0 outside, 1 begin, 2 inside; specials=0)
  attention_masks   : list[int]
  offset_mapping    : list[tuple[int,int]] (token → character spans in cleaned text)
"""

from __future__ import annotations

import argparse
import math
import random
import re
import time
from typing import Dict, List, Tuple
from pathlib import Path
import polars as pl
import torch
from transformers import AutoTokenizer, set_seed


LINK_REGEX = re.compile(r"\[\[(.*?)\]\]")


def process_text(original_text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Convert original_text into (cleaned_text, spans) where spans is a list of
    (start_char, end_char, display_text) anchored to cleaned_text positions.

    Rationale:
    - keeps the display text (parts[-1]) from [[inner]] or [[inner|display]]
    - removes the entire [[…]] construct and stitches the rest, producing a
      cleaned string that serves as the coordinate space for token offset_mapping.
    """
    matches = list(LINK_REGEX.finditer(original_text))
    cleaned_text = ""
    spans: List[Tuple[int, int, str]] = []
    last_idx = 0

    for match in matches:
        inner = match.group(1)          # e.g. "Paris" or "Paris|The city"
        parts = inner.split("|")
        display_text = parts[-1]        # keep the display text when present

        # append chunk before the link
        cleaned_text += original_text[last_idx : match.start()]

        # record span in cleaned_text for this display_text
        start_span = len(cleaned_text)
        end_span = start_span + len(display_text)
        spans.append((start_span, end_span, display_text))

        # add the display text instead of the link
        cleaned_text += display_text
        last_idx = match.end()

    # append the rest of the text after the last link
    cleaned_text += original_text[last_idx:]
    return cleaned_text, spans


def label_tokens(offsets: List[Tuple[int, int]], spans: List[Tuple[int, int, str]]) -> List[int]:
    """
    Produce token labels aligned to tokenizer offsets relative to cleaned text.
    Labels:
      0 = Outside any link span (including special tokens where start=end=0)
      1 = Begin (token starts at/inside span and is the first overlap)
      2 = Inside (token fully/partly inside span but not starting it)

    Rationale:
    - Begin/Inside logic helps a token-classification head identify exact span
      boundaries, consistent with hyperlink entities and typical span tagging
      formulations (and it generalizes cleanly even if spans overlap, though
      Wikipedia hyperlink spans are usually non-overlapping in display text).
    """
    labels: List[int] = []
    for (tstart, tend) in offsets:
        if tstart == tend == 0:  # special tokens (CLS/SEP, and potentially padding handled outside)
            labels.append(0)
            continue

        current_label = 0  # default: Outside

        for sstart, send, _ in spans:
            # Check intersection using standard overlap test.
            if max(tstart, sstart) < min(tend, send):
                if tstart <= sstart or (current_label == 0 and tstart == sstart):
                    current_label = 1  # Begin
                else:
                    current_label = 2  # Inside
                break  # stop after first intersecting span (avoids ambiguous multiple-spans cases)

        labels.append(current_label)
    return labels


def process_file_slice(
    tokenizer: AutoTokenizer,
    polars_df: pl.DataFrame,
    batch_size: int,
    log_every: int,
    processed_total: int,
    file_idx: int,
    start_idx: int,
) -> Tuple[pl.DataFrame, int]:
    """
    Take a bounded slice of rows, clean them, tokenize in batches, label tokens,
    and return a single DataFrame for writing + the number of rows processed during
    this call (used for progress accounting).

    Memory-safety choices:
    - polars_df is assumed to be the result of a lazy scan + slice/limit operation
      (see below), so its footprint is bounded to this slice.
    - We accumulate Python lists per column during tokenization batches rather than
      building many Polars DataFrames inside the loop; this reduces per-iteration
      memory churn while keeping the final conversion bounded.
    - We flush only once per output file by returning a single DataFrame of size
      at most per_file rows.
    """

    n_slice = polars_df.height
    if n_slice == 0:
        print(f"[file {file_idx:03d}] slice [{start_idx}:{start_idx+n_slice}] is empty; skipping.")
        # Return an empty DataFrame with the correct schema to keep downstream logic uniform.
        return pl.DataFrame({
            "texts": [],
            "inputs": [],
            "outputs": [],
            "attention_mask": [],
            "offset_mapping": [],
        }, schema={
            "texts": pl.Utf8,
            "inputs": pl.List(pl.Int64),
            "outputs": pl.List(pl.Int64),
            "attention_mask": pl.List(pl.Int64),
            "offset_mapping": pl.List(
                pl.Struct({"start": pl.Int64, "end": pl.Int64})
            ),
        }), 0

    texts_raw = polars_df["text"].to_list()  # list[str], bounded to slice
    texts_clean: List[str] = []
    spans_per_example: List[List[Tuple[int, int, str]]] = []

    # Step 1: clean text and compute spans in cleaned coordinates
    # Deterministic behavior: no random operation here; order follows input order.
    for t in texts_raw:
        cleaned, spans = process_text(t)
        texts_clean.append(cleaned)
        spans_per_example.append(spans)

    # Step 2: tokenize + label in batches (batched for throughput on large slices)
    # Accumulators follow the requested output schema.
    inputs: List[List[int]] = []
    outputs: List[List[int]] = []
    attention_masks: List[List[int]] = []
    offset_mapping: List[List[Tuple[int, int]]] = []

    n_batches = math.ceil(n_slice / batch_size)
    processed_in_slice = 0

    t0 = time.perf_counter()

    for b in range(n_batches):
        b_start = b * batch_size
        b_end = min(b_start + batch_size, n_slice)

        batch_texts = texts_clean[b_start:b_end]
        batch_spans = spans_per_example[b_start:b_end]

        # Tokenization with offset_mapping (required for labeling)
        tokens: Dict[str, List[List]] = tokenizer(
            batch_texts,
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        batch_input_ids = tokens["input_ids"]
        batch_attention = tokens["attention_mask"]
        batch_offsets = tokens["offset_mapping"]

        # Step 3: label each example in this batch
        for offsets, spans in zip(batch_offsets, batch_spans):
            labels = label_tokens(offsets, spans)
            outputs.append(labels)
            offset_mapping.append(offsets)

        inputs.extend(batch_input_ids)
        attention_masks.extend(batch_attention)

        processed_in_slice += (b_end - b_start)
        processed_total += (b_end - b_start)

        # Progress logging every log_every rows (and at batch boundaries for clarity)
        if processed_total % log_every == 0 or b == n_batches - 1:
            rate = (processed_total - processed_total + b_end - b_start) / (time.perf_counter() - t0)
            print(
                f"[file {file_idx:03d}] batch {b+1}/{n_batches} | "
                f"processed {processed_total:12,} rows | "
                f"slice-rate {rate:6.0f} rows/s"
            )
            t0 = time.perf_counter()

    # Step 4: build final DataFrame for this file (single materialization)
    # Convert offset mapping tuples to a Polars struct to keep schema consistent
    out_df = pl.DataFrame({
        "texts": texts_clean,
        "inputs": inputs,
        "outputs": outputs,
        "attention_mask": attention_masks,
        "offset_mapping": [
            [{"start": s, "end": e} for s, e in offs]
            for offs in offset_mapping
        ],
    })

    return out_df, processed_in_slice


def main(args: argparse.Namespace) -> None:
    # Determinism: set global seeds and PyTorch reproducibility
    random.seed(args.seed)
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Keep determinism practical for transformers/tokenizers (no forced deterministic kernels here,
    # but seeding and order-preserving partitioning should ensure reproducible runs per config and input).

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    # Using fast tokenizer is preferred for large-scale offset mapping.

    # Discover total rows via lazy scan to avoid loading everything into memory.
    print(f"Counting total rows in input path: {args.input}")
    total_rows = pl.scan_parquet(args.input, low_memory=True).select(pl.count()).collect()[0, 0]
    print(f"Total input rows available: {total_rows:14,}")

    # Compute how many rows we will actually write
    total_to_write = total_rows if args.max_entries is None else min(args.max_entries, total_rows)
    num_files = math.ceil(total_to_write / args.per_file)
    print(f"Planned output: {total_to_write:14,} rows in {num_files:3d} files "
          f"(≈ {args.per_file} rows/file)")

    processed_total = 0

    # File-by-file loop: deterministic partitioning via index arithmetic using lazy slicing
    # Rationale: each iteration materializes only [start_idx:end_idx], processes it in batches,
    # then writes the bounded DataFrame immediately, controlling peak memory.
    for file_idx in range(num_files):
        start_idx = file_idx * args.per_file
        end_idx = min(start_idx + args.per_file, total_to_write)
        slice_len = end_idx - start_idx

        print(f"\n[phase] Creating file {file_idx:03d} | slice [{start_idx}:{end_idx}] ({slice_len:,} rows)")

        # Use lazy scanning with a slice to bound memory: scan_parquet + slice + collect once per file
        batch_df = pl.scan_parquet(args.input, low_memory=True).slice(start_idx, slice_len).collect()

        # Process that slice in tokenization batches and produce the output DataFrame
        out_df, processed_in_slice = process_file_slice(
            tokenizer=tokenizer,
            polars_df=batch_df,
            batch_size=args.batch_size,
            log_every=args.log_every,
            processed_total=processed_total,
            file_idx=file_idx,
            start_idx=start_idx,
        )

        out_path = out_dir / f"dataset_{file_idx:03d}.parquet"
        out_df.write_parquet(out_path)
        processed_total += processed_in_slice

        print(f"[file {file_idx:03d}] wrote {processed_in_slice:8,} rows → {out_path}")

    print(f"\nDone. Produced {processed_total:12,} rows across {num_files:3d} files in {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Format large-scale corpus into token-classification records "
            "for hyperlink span detection (CamemBERT lineage)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Input parquet file or directory (Polars-readable); e.g. ./final_corpus.")
    parser.add_argument("--output_dir", "-o", required=True, type=Path,
                        help="Directory to write output parquet files.")
    parser.add_argument("--tokenizer", "-t", default="almanach/camembertv2-base",
                        help="HuggingFace tokenizer name.")
    parser.add_argument("--max_entries", "-n", type=int, default=100000,
                        help="Total number of output rows to produce.")
    parser.add_argument("--per_file", type=int, default=100000,
                        help="Number of rows per output parquet file.")
    parser.add_argument("--batch_size", type=int, default=512,
                        help="Processing batch size for tokenization.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic behavior.")
    parser.add_argument("--log_every", type=int, default=10000,
                        help="Log progress every X processed rows.")

    args = parser.parse_args()
    main(args)

    args = parser.parse_args()
    main(args)