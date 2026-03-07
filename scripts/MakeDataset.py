#!/usr/bin/env python3
"""Produce CamemBERT-compatible NER-style records from a Parquet Wikipedia corpus.

Output schema (one row per document in every output Parquet file):
    texts           – str: cleaned text (link display kept, markup removed)
    inputs          – list[int]: tokeniser input_ids (incl. special tokens)
    outputs         – list[int]: BIO labels (0=O, 1=B, 2=I; specials → 0)
    attention_mask  – list[int]
    offset_mapping  – list[struct{start, end}]: token → character spans
"""

from __future__ import annotations

import argparse
import gc
import logging
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer, PreTrainedTokenizerFast, set_seed

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_LINK_RE: re.Pattern[str] = re.compile(r"\[\[(.*?)\]\]")

LABEL_OUTSIDE: int = 0
LABEL_BEGIN: int = 1
LABEL_INSIDE: int = 2

_OUTPUT_SCHEMA: dict[str, pl.DataType] = {
    "texts": pl.Utf8,
    "inputs": pl.List(pl.Int64),
    "outputs": pl.List(pl.Int64),
    "attention_mask": pl.List(pl.Int64),
    "offset_mapping": pl.List(
        pl.Struct({"start": pl.Int64, "end": pl.Int64})
    ),
}

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

logger = logging.getLogger(__name__)


def _configure_logging(verbosity: int = logging.INFO) -> None:
    """Attach a timestamped stream handler to the root logger.

    Guards against duplicate handler registration when called more than
    once (e.g. in tests).
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(verbosity)


# --------------------------------------------------------------------------- #
# Text cleaning – O(n) per document
# --------------------------------------------------------------------------- #

def process_text(text: str) -> tuple[str, list[tuple[int, int, str]]]:
    """Strip ``[[…]]`` wiki-link markup while keeping display text.

    Returns
    -------
    cleaned : str
        Plain text with every ``[[target|display]]`` (or ``[[target]]``)
        replaced by its display portion.
    spans : list[tuple[int, int, str]]
        ``(start, end, display)`` anchored to *cleaned*-text positions.

    Complexity
    ----------
    **O(n)** – one regex scan plus list-based string assembly joined once.
    The original ``+=`` concatenation on immutable Python strings was
    O(n²) in the worst case.
    """
    matches = list(_LINK_RE.finditer(text))
    if not matches:
        return text, []

    chunks: list[str] = []
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    clean_len = 0

    for m in matches:
        # Text between the previous match end and this match start.
        prefix = text[cursor:m.start()]
        chunks.append(prefix)
        clean_len += len(prefix)

        # Resolve display: [[target|display]] → display ; [[target]] → target.
        # rfind avoids allocating a throw-away list (cf. str.split).
        inner = m.group(1)
        pipe = inner.rfind("|")
        display = inner[pipe + 1:] if pipe != -1 else inner

        spans.append((clean_len, clean_len + len(display), display))
        chunks.append(display)
        clean_len += len(display)
        cursor = m.end()

    chunks.append(text[cursor:])
    return "".join(chunks), spans


# --------------------------------------------------------------------------- #
# BIO labelling – O(T + S) two-pointer sweep
# --------------------------------------------------------------------------- #

def label_tokens(
    offsets: list[tuple[int, int]],
    spans: list[tuple[int, int, str]],
) -> list[int]:
    """Map tokeniser character *offsets* to BIO labels given entity *spans*.

    Both sequences **must** be sorted by start position (guaranteed by
    HuggingFace fast tokenisers and by :func:`process_text`).

    A **two-pointer sweep** produces correct labels in **O(T + S)**
    instead of the naïve O(T × S) nested loop.

    Labels
    ------
    * ``0`` – Outside / special token (offset ``(0, 0)``).
    * ``1`` – Begin: token whose start ≤ span start (first overlap).
    * ``2`` – Inside: token whose start > span start (later overlap).
    """
    n_spans = len(spans)
    if n_spans == 0:
        return [LABEL_OUTSIDE] * len(offsets)

    labels: list[int] = []
    si = 0  # span pointer – only advances forward

    for tok_start, tok_end in offsets:
        # Special tokens (CLS / SEP) carry zero-width offsets.
        if tok_start == tok_end == 0:
            labels.append(LABEL_OUTSIDE)
            continue

        # Advance past spans that end at or before the token start –
        # they can never overlap with this or any later token.
        while si < n_spans and spans[si][1] <= tok_start:
            si += 1

        if si < n_spans:
            sp_start, sp_end, _ = spans[si]
            # Standard half-open interval overlap test.
            if tok_start < sp_end and tok_end > sp_start:
                labels.append(
                    LABEL_BEGIN if tok_start <= sp_start else LABEL_INSIDE
                )
                continue

        labels.append(LABEL_OUTSIDE)

    return labels


# --------------------------------------------------------------------------- #
# Per-file slice processor
# --------------------------------------------------------------------------- #

def _empty_output_df() -> pl.DataFrame:
    """Return a zero-row DataFrame matching :data:`_OUTPUT_SCHEMA`."""
    return pl.DataFrame(
        {col: [] for col in _OUTPUT_SCHEMA},
        schema=_OUTPUT_SCHEMA,
    )


def process_file_slice(
    tokenizer: PreTrainedTokenizerFast,
    df: pl.DataFrame,
    *,
    batch_size: int,
    log_every: int,
    global_offset: int,
    file_idx: int,
) -> tuple[pl.DataFrame, int]:
    """Clean, tokenise, and BIO-label a bounded row slice.

    Parameters
    ----------
    tokenizer :
        HuggingFace **fast** tokenizer (needed for ``offset_mapping``).
    df :
        Already-collected input slice whose size bounds peak memory.
    batch_size :
        Micro-batch size fed to the tokeniser for throughput.
    log_every :
        Emit a progress line every *log_every* rows within the slice.
    global_offset :
        Rows fully processed before this slice (for display only).
    file_idx :
        Zero-based output-file index (for display only).

    Returns
    -------
    tuple[pl.DataFrame, int]
        The labelled output DataFrame and the number of rows produced.
    """
    n_rows = df.height
    if n_rows == 0:
        logger.warning("[file %03d] empty slice – skipping.", file_idx)
        return _empty_output_df(), 0

    # ------------------------------------------------------------------ #
    # 1. Text cleaning                                                    #
    # ------------------------------------------------------------------ #
    raw_texts: list[str | None] = df["text"].to_list()

    texts_clean: list[str] = []
    spans_all: list[list[tuple[int, int, str]]] = []

    for raw in raw_texts:
        if raw is None:
            texts_clean.append("")
            spans_all.append([])
        else:
            cln, sp = process_text(raw)
            texts_clean.append(cln)
            spans_all.append(sp)

    del raw_texts  # free original strings early

    # ------------------------------------------------------------------ #
    # 2. Tokenise in micro-batches + assign BIO labels                    #
    # ------------------------------------------------------------------ #
    acc_ids: list[list[int]] = []
    acc_labels: list[list[int]] = []
    acc_attn: list[list[int]] = []
    acc_offs: list[list[tuple[int, int]]] = []

    n_batches = math.ceil(n_rows / batch_size)
    done = 0
    t0 = time.perf_counter()
    next_log = log_every

    for b_idx in range(n_batches):
        lo = b_idx * batch_size
        hi = min(lo + batch_size, n_rows)

        enc = tokenizer(
            texts_clean[lo:hi],
            return_offsets_mapping=True,
            add_special_tokens=True,
            padding=False,
            truncation=False,
        )

        # Label + accumulate offsets per example in this micro-batch.
        for offs, sp in zip(enc["offset_mapping"], spans_all[lo:hi]):
            acc_labels.append(label_tokens(offs, sp))
            acc_offs.append(offs)

        acc_ids.extend(enc["input_ids"])
        acc_attn.extend(enc["attention_mask"])

        done += hi - lo

        if done >= next_log or b_idx == n_batches - 1:
            elapsed = time.perf_counter() - t0
            rate = done / elapsed if elapsed > 0.0 else float("inf")
            logger.info(
                "[file %03d] batch %d/%d | %s/%s rows | "
                "%.0f rows/s | global %s",
                file_idx,
                b_idx + 1,
                n_batches,
                f"{done:,}",
                f"{n_rows:,}",
                rate,
                f"{global_offset + done:,}",
            )
            next_log += log_every

    del spans_all  # no longer needed after labelling

    # ------------------------------------------------------------------ #
    # 3. Assemble output DataFrame (single materialisation)               #
    # ------------------------------------------------------------------ #
    out_df = pl.DataFrame(
        {
            "texts": texts_clean,
            "inputs": acc_ids,
            "outputs": acc_labels,
            "attention_mask": acc_attn,
            "offset_mapping": [
                [{"start": s, "end": e} for s, e in offs]
                for offs in acc_offs
            ],
        },
    )
    return out_df, done


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser."""
    p = argparse.ArgumentParser(
        description=(
            "Convert a large-scale Parquet corpus into token-classification "
            "records for hyperlink-span detection (CamemBERT lineage)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input", "-i", required=True, type=Path,
        help="Input Parquet file or directory (Polars-readable).",
    )
    p.add_argument(
        "--output_dir", "-o", required=True, type=Path,
        help="Directory for output Parquet files.",
    )
    p.add_argument(
        "--tokenizer", "-t", default="almanach/camembertv2-base",
        help="HuggingFace tokenizer name or repository path.",
    )
    p.add_argument(
        "--max_entries", "-n", type=int, default=100_000,
        help="Max output rows.",
    )
    p.add_argument(
        "--per_file", type=int, default=100_000,
        help="Rows per output Parquet file.",
    )
    p.add_argument(
        "--batch_size", type=int, default=512,
        help="Tokenisation micro-batch size.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility.",
    )
    p.add_argument(
        "--log_every", type=int, default=10_000,
        help="Progress log interval (rows).",
    )
    return p


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _validate_args(args: argparse.Namespace) -> None:
    """Fail fast on obviously invalid CLI parameters."""
    if not args.input.exists():
        raise FileNotFoundError(f"Input path not found: {args.input}")
    for name in ("per_file", "batch_size", "log_every"):
        val = getattr(args, name)
        if val <= 0:
            raise ValueError(f"--{name} must be > 0 (got {val})")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def main(args: argparse.Namespace) -> None:
    """Run the end-to-end dataset-creation pipeline."""
    _configure_logging()
    _validate_args(args)

    # -- Determinism ------------------------------------------------------ #
    random.seed(args.seed)
    set_seed(args.seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # HuggingFace Rust tokenizer parallelism (safe in single-process mode).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # -- Tokenizer -------------------------------------------------------- #
    logger.info("Loading tokenizer: %s", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(
            f"A fast tokenizer is required for offset-mapping support; "
            f"'{args.tokenizer}' resolved to a slow tokenizer."
        )

    # -- Corpus discovery ------------------------------------------------- #
    input_path: Path = args.input
    logger.info("Scanning corpus: %s", input_path)

    lazy_frame = pl.scan_parquet(input_path, low_memory=True)
    schema = lazy_frame.collect_schema()
    if "text" not in schema:
        raise KeyError(
            f"Column 'text' missing from input schema {sorted(schema)}."
        )

    total_rows: int = lazy_frame.select(pl.len()).collect().item()
    logger.info("Total input rows: %s", f"{total_rows:,}")

    # -- Plan ------------------------------------------------------------- #
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    total_to_write = (
        total_rows
        if args.max_entries is None
        else min(args.max_entries, total_rows)
    )
    if total_to_write == 0:
        logger.warning("Nothing to write (0 rows selected). Exiting.")
        return

    num_files = math.ceil(total_to_write / args.per_file)
    logger.info(
        "Plan: %s rows -> %d file(s) (~%s rows/file) -> %s",
        f"{total_to_write:,}",
        num_files,
        f"{args.per_file:,}",
        out_dir,
    )

    # -- Main loop -------------------------------------------------------- #
    processed_total = 0
    t_pipeline = time.perf_counter()

    for fid in range(num_files):
        start = fid * args.per_file
        length = min(args.per_file, total_to_write - start)

        logger.info(
            "[file %03d] reading rows [%s..%s) (%s rows)",
            fid,
            f"{start:,}",
            f"{start + length:,}",
            f"{length:,}",
        )

        slice_df = (
            pl.scan_parquet(input_path, low_memory=True)
            .slice(start, length)
            .collect()
        )

        try:
            out_df, n_produced = process_file_slice(
                tokenizer,
                slice_df,
                batch_size=args.batch_size,
                log_every=args.log_every,
                global_offset=processed_total,
                file_idx=fid,
            )
        except Exception:
            logger.exception("[file %03d] processing failed – skipping.", fid)
            continue
        finally:
            del slice_df

        out_path = out_dir / f"dataset_{fid:03d}.parquet"
        out_df.write_parquet(out_path)
        processed_total += n_produced

        logger.info(
            "[file %03d] wrote %s rows -> %s",
            fid,
            f"{n_produced:,}",
            out_path,
        )

        del out_df
        gc.collect()  # reclaim Rust-backed Polars memory between files

    elapsed = time.perf_counter() - t_pipeline
    rate = processed_total / elapsed if elapsed > 0.0 else 0.0
    logger.info(
        "Done. %s rows in %d file(s), %.1f s (%.0f rows/s) -> %s",
        f"{processed_total:,}",
        num_files,
        elapsed,
        rate,
        out_dir,
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    try:
        main(_build_parser().parse_args())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

