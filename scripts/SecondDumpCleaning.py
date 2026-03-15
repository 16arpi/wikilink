#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Wikification corpus cleaner/augmenter for FR Wikipedia parquet shards.


Pipeline:
PASS A1 (parallel, map->disk, reduce): count freq_link(anchor) on full corpus.
PASS A2 (parallel, map->disk, reduce): for whitelist candidates only, count (anchor,target) to pick canonical target + share.
PASS B  (parallel, map->disk, reduce): count freq_text(anchor) OUTSIDE links for candidates using Aho-Corasick.
PASS C  (parallel): rewrite corpus (strip BLACKLIST links, add WHITELIST links in plain text),
                    split paragraphs, compute link_ratio, filter, write parquet streaming.
                    ONE OUTPUT ROW PER INPUT ARTICLE (paragraphs with poor ratio are
                    removed, the rest are re-joined; articles left empty are dropped).


Outputs (work_dir):
- link_counts.parquet
- anchor_target_counts.parquet
- stats_surface_forms.parquet
- whitelist.parquet
- blacklist.parquet
Outputs (out_dir):
- <shard>.cleaned.parquet with columns: text, link_ratio
"""


import os
import re
import time
import math
import argparse
import unicodedata
import multiprocessing as mp
from pathlib import Path
from glob import glob
from collections import defaultdict
from bisect import bisect_right


import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq


import ahocorasick  # pyahocorasick




# ----------------------------
# Normalization
# ----------------------------


APOS_TRANSLATION = str.maketrans({
    "\u2019": "'",  # '
    "\u2018": "'",  # '
    "\u02BC": "'",  # ʼ
    "\u0060": "'",  # `
    "\u00B4": "'",  # ´
})


# Optional: normalize curly double quotes to "
QUOTE_TRANSLATION = str.maketrans({
    "\u201C": '"',  # "
    "\u201D": '"',  # "
    "\u201E": '"',  # „
})


NBSP = "\u00A0"




def norm_text(t: str, normalize_quotes: bool) -> str:
    if not t:
        return ""
    t = t.replace(NBSP, " ")
    t = t.translate(APOS_TRANSLATION)
    if normalize_quotes:
        t = t.translate(QUOTE_TRANSLATION)
    # Wikipedia is mostly NFC already, but we enforce it for consistency.
    return unicodedata.normalize("NFC", t)




def norm_surface(s: str, normalize_quotes: bool) -> str:
    # Normalize anchor/target surface forms for stable counting.
    if not s:
        return ""
    s = s.replace(NBSP, " ").strip()
    s = s.translate(APOS_TRANSLATION)
    if normalize_quotes:
        s = s.translate(QUOTE_TRANSLATION)
    s = re.sub(r"\s+", " ", s)
    s = s.strip("[]").strip()
    return unicodedata.normalize("NFC", s)




# ----------------------------
# Boundaries / word counting
# ----------------------------


WORD_RE = re.compile(r"\w+(?:[''\-]\w+)*", flags=re.UNICODE)




def is_word_char(c: str) -> bool:
    return c.isalnum() or c == "_"




def boundary_ok(text: str, start: int, end_incl: int) -> bool:
    # Reject if match is embedded in a larger word
    if start > 0 and is_word_char(text[start - 1]):
        return False
    if end_incl + 1 < len(text) and is_word_char(text[end_incl + 1]):
        return False
    return True




def count_words(s: str) -> int:
    if not s:
        return 0
    return sum(1 for _ in WORD_RE.finditer(s))




# ----------------------------
# Wiki link parsing (fast, non-regex)
# ----------------------------


NS_IGNORE = {
    "catégorie", "category",
    "fichier", "file",
    "wikipédia", "wikipedia",
    "aide", "help",
    "portail", "portal",
    "modèle", "template",
    "special", "spécial",
    "mediawiki",
    "projet", "project",
    "discussion", "talk",
    "utilisateur", "user",
    "module",
}




def is_content_target(target: str) -> bool:
    # Filter out non-article namespaces like "Catégorie:...", "Fichier:..."
    if not target:
        return False
    i = target.find(":")
    if i <= 0:
        return True
    prefix = target[:i].strip().lower()
    return prefix not in NS_IGNORE




def iter_link_spans(text: str):
    """
    Yield (span_start, span_end_excl, inner) for each [[...]] block (best-effort).
    """
    i = 0
    while True:
        a = text.find("[[", i)
        if a < 0:
            return
        b = text.find("]]", a + 2)
        if b < 0:
            return
        inner = text[a + 2: b]
        yield a, b + 2, inner
        i = b + 2




def parse_inner_link(inner: str):
    """
    For [[target|anchor]] => (target, anchor)
    For [[target]] => (target, target)
    """
    inner = inner.strip()
    if not inner:
        return None, None
    target, sep, anchor = inner.partition("|")
    target = target.strip()
    if not target:
        return None, None
    if sep:
        anchor = anchor.strip()
        if not anchor:
            anchor = target
    else:
        anchor = target
    return target, anchor




def strip_links(text: str) -> str:
    """
    Replace [[...]] with spaces; used for PASS B plain-text counting.
    """
    if "[[" not in text:
        return text
    out = []
    i = 0
    for a, b, _inner in iter_link_spans(text):
        if a > i:
            out.append(text[i:a])
        out.append(" ")
        i = b
    if i < len(text):
        out.append(text[i:])
    return "".join(out)




# ----------------------------
# Overlap check (bisect) for augmentation
# ----------------------------


def overlaps_any_span(start: int, end_excl: int, starts: list[int], ends: list[int]) -> bool:
    """
    True if [start, end_excl) overlaps any [span_start, span_end_excl).
    Spans are sorted by start.
    """
    if not starts:
        return False


    # Candidate span at or before start
    i = bisect_right(starts, start) - 1
    if i >= 0 and ends[i] > start:
        return True


    # Next span might start before our end
    j = i + 1
    if j < len(starts) and starts[j] < end_excl:
        return True


    return False




# ----------------------------
# PASS A1: map counts of anchors in links -> parquet partials
# ----------------------------


def map_count_links_one_file(args):
    in_file, out_tmp_dir, batch_size, min_anchor_len, normalize_quotes = args
    out_tmp_dir = Path(out_tmp_dir)
    out_path = out_tmp_dir / (Path(in_file).stem + ".link_counts.parquet")


    counts = defaultdict(int)


    pf = pq.ParquetFile(in_file)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for t in texts:
            if t is None or t == "":
                continue
            t = norm_text(t, normalize_quotes=normalize_quotes)


            for _a, _b, inner in iter_link_spans(t):
                target, anchor = parse_inner_link(inner)
                if not anchor or not target:
                    continue
                if not is_content_target(target):
                    continue


                anchor_n = norm_surface(anchor, normalize_quotes=normalize_quotes)
                if len(anchor_n) < min_anchor_len:
                    continue
                if not any(ch.isalnum() for ch in anchor_n):
                    continue
                counts[anchor_n] += 1




    table = pa.Table.from_arrays(
        [pa.array(list(counts.keys()), pa.large_utf8()),
         pa.array(list(counts.values()), pa.int64())],
        names=["anchor", "freq_link"]
    )
    pq.write_table(table, out_path, compression="zstd")
    return str(out_path)




# ----------------------------
# PASS A2: map counts of (anchor,target) for whitelist candidates only
# ----------------------------


def map_count_anchor_target_one_file(args):
    in_file, out_tmp_dir, batch_size, wl_anchor_set, normalize_quotes = args
    out_tmp_dir = Path(out_tmp_dir)
    out_path = out_tmp_dir / (Path(in_file).stem + ".anchor_target.parquet")


    pair_counts = defaultdict(int)


    pf = pq.ParquetFile(in_file)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for t in texts:
            if t is None or t == "":
                continue
            t = norm_text(t, normalize_quotes=normalize_quotes)


            for _a, _b, inner in iter_link_spans(t):
                target, anchor = parse_inner_link(inner)
                if not anchor or not target:
                    continue
                if not is_content_target(target):
                    continue


                anchor_n = norm_surface(anchor, normalize_quotes=normalize_quotes)
                if anchor_n not in wl_anchor_set:
                    continue
                target_n = norm_surface(target, normalize_quotes=normalize_quotes)
                if not target_n:
                    continue
                pair_counts[(anchor_n, target_n)] += 1




    anchors = []
    targets = []
    counts = []
    for (a, tg), c in pair_counts.items():
        anchors.append(a)
        targets.append(tg)
        counts.append(c)


    table = pa.Table.from_arrays(
        [pa.array(anchors, pa.large_utf8()),
         pa.array(targets, pa.large_utf8()),
         pa.array(counts, pa.int64())],
        names=["anchor", "target", "pair_count"]
    )
    pq.write_table(table, out_path, compression="zstd")
    return str(out_path)




# ----------------------------
# Reduce helpers (Polars)
# ----------------------------


def reduce_sum_counts(parquet_files: list[str], key: str, val: str) -> pl.DataFrame:
    lf = pl.scan_parquet(parquet_files)
    return (
        lf.group_by(key)
          .agg(pl.col(val).sum().alias(val))
          .collect()
    )




def reduce_anchor_target(parquet_files: list[str]) -> pl.DataFrame:
    lf = pl.scan_parquet(parquet_files)
    return (
        lf.group_by(["anchor", "target"])
          .agg(pl.col("pair_count").sum().alias("pair_count"))
          .collect()
    )




# ----------------------------
# PASS B: plain counts using Aho-Corasick on candidate anchors
#   - workers write .npy partial arrays to disk (no huge pickling)
# ----------------------------


PLAIN_AUT = None
PLAIN_LENS = None
PLAIN_NEEDS_BOUNDARY = None
PLAIN_NORMALIZE_QUOTES = False




def build_plain_automaton(patterns: list[str]):
    A = ahocorasick.Automaton()
    lens = np.empty(len(patterns), dtype=np.int32)
    needs = np.empty(len(patterns), dtype=np.bool_)
    for i, p in enumerate(patterns):
        A.add_word(p, i)
        lens[i] = len(p)
        needs[i] = (len(p) > 0 and is_word_char(p[0]) and is_word_char(p[-1]))
    A.make_automaton()
    return A, lens, needs




def map_count_plain_one_file(args):
    in_file, out_tmp_dir, batch_size = args
    out_tmp_dir = Path(out_tmp_dir)
    out_path = out_tmp_dir / (Path(in_file).stem + ".plain_counts.npy")


    A = PLAIN_AUT
    lens = PLAIN_LENS
    needs = PLAIN_NEEDS_BOUNDARY
    normalize_quotes = PLAIN_NORMALIZE_QUOTES


    counts = np.zeros(len(lens), dtype=np.int64)


    pf = pq.ParquetFile(in_file)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for t in texts:
            if t is None or t == "":
                continue
            t = norm_text(t, normalize_quotes=normalize_quotes)
            plain = strip_links(t)
            if not plain:
                continue


            for end_idx, idx in A.iter(plain):
                L = lens[idx]
                start = end_idx - L + 1
                if start < 0:
                    continue
                if needs[idx] and not boundary_ok(plain, start, end_idx):
                    continue
                counts[idx] += 1




    np.save(out_path, counts)
    return str(out_path)




# ----------------------------
# PASS C: rewriting
# ----------------------------


WL_AUT = None
WL_LENS = None
WL_NEEDS_BOUNDARY = None
WL_TARGETS = None
BLACKLIST = None
REWRITE_NORMALIZE_QUOTES = False




def build_whitelist_automaton(wl_patterns: list[str]):
    A = ahocorasick.Automaton()
    lens = np.empty(len(wl_patterns), dtype=np.int32)
    needs = np.empty(len(wl_patterns), dtype=np.bool_)
    for i, p in enumerate(wl_patterns):
        A.add_word(p, i)
        lens[i] = len(p)
        needs[i] = (len(p) > 0 and is_word_char(p[0]) and is_word_char(p[-1]))
    A.make_automaton()
    return A, lens, needs




def rewrite_blacklisted_links(text: str) -> str:
    if "[[" not in text:
        return text


    out = []
    i = 0
    for a, b, inner in iter_link_spans(text):
        if a > i:
            out.append(text[i:a])


        target, anchor = parse_inner_link(inner)
        if not anchor:
            out.append(text[a:b])
        else:
            anchor_n = norm_surface(anchor, normalize_quotes=REWRITE_NORMALIZE_QUOTES)
            if anchor_n in BLACKLIST:
                out.append(anchor)  # keep visible
            else:
                out.append(text[a:b])


        i = b


    if i < len(text):
        out.append(text[i:])
    return "".join(out)




def augment_whitelist_outside_links(text: str) -> str:
    """
    Scan full text with AC, but skip matches overlapping existing link spans using bisect.
    Resolve overlaps among candidate matches greedily (left-to-right, prefer longer).
    """
    A = WL_AUT
    if A is None or not text:
        return text


    # Existing link spans (sorted)
    starts = []
    ends = []
    for a, b, _inner in iter_link_spans(text):
        starts.append(a)
        ends.append(b)


    # Collect matches outside existing spans
    matches = []
    lens = WL_LENS
    needs = WL_NEEDS_BOUNDARY


    for end_idx, idx in A.iter(text):
        L = lens[idx]
        start = end_idx - L + 1
        if start < 0:
            continue
        end_excl = end_idx + 1


        if overlaps_any_span(start, end_excl, starts, ends):
            continue
        if needs[idx] and not boundary_ok(text, start, end_idx):
            continue
        matches.append((start, end_excl, idx))


    if not matches:
        return text


    # Resolve overlaps among matches: sort by start asc, length desc
    matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))


    chosen = []
    cur = 0
    for s, e, idx in matches:
        if s < cur:
            continue
        chosen.append((s, e, idx))
        cur = e


    # Rebuild with insertions
    out = []
    pos = 0
    for s, e, idx in chosen:
        out.append(text[pos:s])
        anchor_txt = text[s:e]
        target = WL_TARGETS[idx]
        out.append(f"[[{target}|{anchor_txt}]]")
        pos = e
    out.append(text[pos:])
    return "".join(out)




def split_paragraphs(text: str) -> list[str]:
    if not text:
        return []
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    parts = t.split("\n\n")
    return [p.strip() for p in parts if p and p.strip()]




def compute_link_ratio(text: str) -> float:
    if not text:
        return 0.0


    total_words = 0
    linked_words = 0


    if "[[" not in text:
        total_words = count_words(text)
        return 0.0 if total_words == 0 else 0.0


    i = 0
    for a, b, inner in iter_link_spans(text):
        if a > i:
            total_words += count_words(text[i:a])


        target, anchor = parse_inner_link(inner)
        visible = anchor if anchor else ""
        w = count_words(visible)
        total_words += w
        linked_words += w
        i = b


    if i < len(text):
        total_words += count_words(text[i:])


    if total_words == 0:
        return 0.0
    return linked_words / total_words




# ──────────────────────────────────────────────────────────────────────
# FIX: rewrite_one_file now emits exactly ONE output row per input row
#      (= one Wikipedia article).  Paragraphs whose link_ratio is below
#      the threshold are dropped, and the survivors are re-joined with
#      "\n\n".  If every paragraph of an article is filtered out the
#      article is skipped entirely.
#      This guarantees that two distinct articles can never end up
#      concatenated in the same Parquet cell.
# ──────────────────────────────────────────────────────────────────────


def rewrite_one_file(args):
    in_file, out_dir, batch_size, ratio_threshold, normalize_quotes = args
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(in_file).stem + ".cleaned.parquet")


    schema = pa.schema([
        ("text", pa.large_utf8()),
        ("link_ratio", pa.float32()),
    ])
    writer = pq.ParquetWriter(out_path, schema=schema, compression="zstd")


    buf_text = []
    buf_ratio = []


    pf = pq.ParquetFile(in_file)
    for batch in pf.iter_batches(batch_size=batch_size, columns=["text"]):
        texts = batch.column(0).to_pylist()
        for t in texts:
            if t is None or t == "":
                continue


            t = norm_text(t, normalize_quotes=normalize_quotes)


            t1 = rewrite_blacklisted_links(t)
            t2 = augment_whitelist_outside_links(t1)


            # --- per-paragraph ratio filter (kept as requested) ---
            kept_paragraphs = []
            for p in split_paragraphs(t2):
                r = compute_link_ratio(p)
                if r > ratio_threshold:
                    kept_paragraphs.append(p)


            # If nothing survived, drop the whole article
            if not kept_paragraphs:
                continue


            # Re-join the surviving paragraphs into ONE cell
            article_text = "\n\n".join(kept_paragraphs)
            article_ratio = compute_link_ratio(article_text)


            buf_text.append(article_text)
            buf_ratio.append(float(article_ratio))


            if len(buf_text) >= 5000:
                table = pa.Table.from_arrays(
                    [pa.array(buf_text, pa.large_utf8()),
                    pa.array(buf_ratio, pa.float32())],
                    names=["text", "link_ratio"]
                )
                writer.write_table(table)
                buf_text.clear()
                buf_ratio.clear()


    if buf_text:
        table = pa.Table.from_arrays(
            [pa.array(buf_text, pa.large_utf8()),
             pa.array(buf_ratio, pa.float32())],
            names=["text", "link_ratio"]
        )
        writer.write_table(table)


    writer.close()
    return str(out_path)




# ----------------------------
# Main
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", type=str, required=True)
    ap.add_argument("--glob", type=str, default="wiki_shard_*.parquet")
    ap.add_argument("--work_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)


    ap.add_argument("--processes", type=int, default=max(1, os.cpu_count() or 8))
    ap.add_argument("--batch_size", type=int, default=4096)


    ap.add_argument("--normalize_quotes", action="store_true",
                    help="Normalize curly double quotes to straight quotes (keeps guillemets).")


    ap.add_argument("--min_anchor_len", type=int, default=2)
    ap.add_argument("--min_total", type=int, default=10)


    ap.add_argument("--min_link_for_plain_count", type=int, default=10)


    ap.add_argument("--whitelist_p", type=float, default=0.95)
    ap.add_argument("--whitelist_min_total", type=int, default=50)
    ap.add_argument("--whitelist_target_share", type=float, default=0.98)


    ap.add_argument("--blacklist_p", type=float, default=0.05)


    ap.add_argument("--ratio_threshold", type=float, default=0.03)


    args = ap.parse_args()


    input_dir = Path(args.input_dir)
    work_dir = Path(args.work_dir)
    out_dir = Path(args.out_dir)


    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = work_dir / "tmp_partials"
    tmp_dir.mkdir(parents=True, exist_ok=True)


    files = sorted(glob(str(input_dir / args.glob)))
    if not files:
        raise FileNotFoundError(f"No parquet found with {input_dir / args.glob}")
    print(f"Found {len(files)} shards.")


    # On AWS Linux, fork gives COW sharing for big automata
    ctx = mp.get_context("fork")


    # ------------------------
    # PASS A1
    # ------------------------
    t0 = time.time()
    print("\nPASS A1: Counting freq_link(anchor) (parallel map->disk, reduce)...")
    link_tmp = tmp_dir / "passA1_link"
    link_tmp.mkdir(exist_ok=True)


    map_args = [(f, str(link_tmp), args.batch_size, args.min_anchor_len, args.normalize_quotes) for f in files]
    with ctx.Pool(processes=args.processes) as pool:
        partials = list(pool.imap_unordered(map_count_links_one_file, map_args, chunksize=1))


    df_link = reduce_sum_counts(partials, key="anchor", val="freq_link")
    link_counts_path = work_dir / "link_counts.parquet"
    df_link.write_parquet(link_counts_path, compression="zstd")
    print(f"Saved: {link_counts_path} (rows={df_link.height:,})  time={time.time()-t0:.1f}s")


    # Candidates for PASS B
    df_candidates = df_link.filter(pl.col("freq_link") >= args.min_link_for_plain_count)
    candidates = df_candidates["anchor"].to_list()
    print(f"Candidates for plain counting: {len(candidates):,} (freq_link >= {args.min_link_for_plain_count})")


    # Candidates for PASS A2 (targets) = whitelist_min_total
    df_wl_target_cand = df_link.filter(pl.col("freq_link") >= args.whitelist_min_total)
    wl_target_candidates = set(df_wl_target_cand["anchor"].to_list())
    print(f"Candidates for target mapping: {len(wl_target_candidates):,} (freq_link >= {args.whitelist_min_total})")


    # ------------------------
    # PASS A2
    # ------------------------
    t1 = time.time()
    print("\nPASS A2: Counting (anchor,target) for whitelist candidates (parallel map->disk, reduce)...")
    at_tmp = tmp_dir / "passA2_anchor_target"
    at_tmp.mkdir(exist_ok=True)


    map_args2 = [(f, str(at_tmp), args.batch_size, wl_target_candidates, args.normalize_quotes) for f in files]
    with ctx.Pool(processes=args.processes) as pool:
        partials2 = list(pool.imap_unordered(map_count_anchor_target_one_file, map_args2, chunksize=1))


    df_at = reduce_anchor_target(partials2)
    anchor_target_path = work_dir / "anchor_target_counts.parquet"
    df_at.write_parquet(anchor_target_path, compression="zstd")
    print(f"Saved: {anchor_target_path} (rows={df_at.height:,})  time={time.time()-t1:.1f}s")


    df_at2 = (
        df_at.sort(["anchor", "pair_count"], descending=[False, True])
             .with_columns(pl.col("pair_count").sum().over("anchor").alias("anchor_link_total"))
             .group_by("anchor")
             .agg([
                 pl.first("target").alias("target"),
                 pl.first("pair_count").alias("target_count"),
                 pl.first("anchor_link_total").alias("anchor_link_total"),
             ])
             .with_columns((pl.col("target_count") / pl.col("anchor_link_total")).alias("target_share"))
    )


    # ------------------------
    # PASS B (build automaton once, then fork)
    # ------------------------
    t2 = time.time()
    print("\nPASS B: Counting freq_text(anchor) outside links (parallel map->disk .npy, reduce)...")
    global PLAIN_AUT, PLAIN_LENS, PLAIN_NEEDS_BOUNDARY, PLAIN_NORMALIZE_QUOTES
    PLAIN_NORMALIZE_QUOTES = args.normalize_quotes


    print("Building Aho-Corasick automaton for candidates (built once, shared via fork)...")
    PLAIN_AUT, PLAIN_LENS, PLAIN_NEEDS_BOUNDARY = build_plain_automaton(candidates)
    print("Automaton built.")


    plain_tmp = tmp_dir / "passB_plain"
    plain_tmp.mkdir(exist_ok=True)


    map_argsB = [(f, str(plain_tmp), args.batch_size) for f in files]
    with ctx.Pool(processes=args.processes) as pool:
        npy_paths = list(pool.imap_unordered(map_count_plain_one_file, map_argsB, chunksize=1))


    # Reduce: sum .npy arrays sequentially (no huge memory spike)
    plain_counts = np.zeros(len(candidates), dtype=np.int64)
    for p in npy_paths:
        arr = np.load(p, mmap_mode="r")
        plain_counts += arr


    df_plain = pl.DataFrame({"anchor": candidates, "freq_text": plain_counts})


    df_stats = (
        df_link.join(df_plain, on="anchor", how="left")
              .with_columns(pl.col("freq_text").fill_null(0))
              .with_columns((pl.col("freq_link") + pl.col("freq_text")).alias("total"))
              .filter(pl.col("total") >= args.min_total)
              .with_columns((pl.col("freq_link") / pl.col("total")).alias("p_link"))
    )


    stats_path = work_dir / "stats_surface_forms.parquet"
    df_stats.write_parquet(stats_path, compression="zstd")
    print(f"Saved: {stats_path} (rows={df_stats.height:,})  time={time.time()-t2:.1f}s")


    # ------------------------
    # BLACKLIST / WHITELIST
    # ------------------------
    df_black = df_stats.filter(pl.col("p_link") <= args.blacklist_p).select(
        "anchor", "freq_link", "freq_text", "total", "p_link"
    )
    blacklist_path = work_dir / "blacklist.parquet"
    df_black.write_parquet(blacklist_path, compression="zstd")
    blacklist_set = set(df_black["anchor"].to_list())
    print(f"Saved: {blacklist_path} (rows={df_black.height:,})")


    df_wl = (
        df_stats.filter((pl.col("p_link") >= args.whitelist_p) & (pl.col("total") >= args.whitelist_min_total))
               .join(df_at2, on="anchor", how="left")
               .filter(pl.col("target").is_not_null() & (pl.col("target_share") >= args.whitelist_target_share))
               .select("anchor", "target", "freq_link", "freq_text", "total", "p_link", "target_share")
    )
    whitelist_path = work_dir / "whitelist.parquet"
    df_wl.write_parquet(whitelist_path, compression="zstd")
    print(f"Saved: {whitelist_path} (rows={df_wl.height:,})")


    # ------------------------
    # PASS C (build WL automaton once, then fork)
    # ------------------------
    t3 = time.time()
    print("\nPASS C: Rewriting corpus (parallel, streaming ParquetWriter)...")
    global WL_AUT, WL_LENS, WL_NEEDS_BOUNDARY, WL_TARGETS, BLACKLIST, REWRITE_NORMALIZE_QUOTES
    REWRITE_NORMALIZE_QUOTES = args.normalize_quotes


    wl_anchor = df_wl["anchor"].to_list()
    wl_target = df_wl["target"].to_list()


    # We normalize text to ASCII apostrophe anyway, so patterns are stored normalized.
    # If you want to match both ' and ' without normalizing full text, expand patterns here.
    wl_patterns = wl_anchor
    WL_TARGETS = wl_target
    BLACKLIST = blacklist_set


    print("Building whitelist Aho-Corasick automaton (built once, shared via fork)...")
    WL_AUT, WL_LENS, WL_NEEDS_BOUNDARY = build_whitelist_automaton(wl_patterns)
    print("Whitelist automaton built.")


    rewrite_args = [(f, str(out_dir), args.batch_size, args.ratio_threshold, args.normalize_quotes) for f in files]
    with ctx.Pool(processes=args.processes) as pool:
        out_files = list(pool.imap_unordered(rewrite_one_file, rewrite_args, chunksize=1))


    print(f"\nDone. Wrote {len(out_files)} cleaned shards to: {out_dir}")
    print(f"PASS C time={time.time()-t3:.1f}s")




if __name__ == "__main__":
    # Strongly recommended on AWS Linux
    # (if you run this inside certain orchestrators, start_method might already be set)
    try:
        mp.set_start_method("fork")
    except RuntimeError:
        pass


    # Avoid thread oversubscription if you later add polars inside workers
    # os.environ.setdefault("POLARS_MAX_THREADS", "1")


    main()
