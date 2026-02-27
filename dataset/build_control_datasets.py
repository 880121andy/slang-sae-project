"""
build_control_datasets.py
=========================
Generates two control/baseline datasets for the Layer 25 "general hotspot" test:

1. literal_paraphrase_baseline.csv
   Source: PAWS labeled_final (label=1 paraphrase pairs), both sentences are fully literal.
   Purpose: Do Layer 25 SAE features still peak when BOTH sides are literal but lexically
            different? If yes, Layer 25 is a general "word-change" detector, not a
            non-literal language detector.

2. identical_baseline.csv
   Source: PAWS sentence1 column (literal English sentences, duplicated).
   Purpose: Sanity-check floor — comparing a sentence to itself should produce
            cosine_dist ≈ 0 and slang_only_features = 0 at every layer.

Both files share the same column schema as metaphor_baseline.csv so they slot directly
into the existing run.py loaders (load_literal_paraphrase_pairs / load_identical_pairs).

Column schema (both files):
  normal             — the "baseline" sentence (sentence B)
  [type]             — the paired sentence (sentence A):
                       'paraphrase'  for literal_paraphrase_baseline.csv
                       'identical'   for identical_baseline.csv
  literal_segments   — differing words/phrases in 'normal'
  [type]_segments    — differing words/phrases in the paired sentence
  literal_positions  — word-level (start, end) tuples for 'normal' differences
  [type]_positions   — word-level (start, end) tuples for paired sentence differences
  num_differences    — number of differing spans

Position format: List[Tuple[int, int]] where (start, end) is 0-indexed, end exclusive,
matching the convention used by word_to_token_position() in run.py.
"""

from __future__ import annotations

import ast
import os
import sys
from difflib import SequenceMatcher
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_SIZE = 1000          # rows per output file
MAX_SCAN    = 50_000        # max PAWS rows to scan before giving up
SEED        = 42

OUT_DIR = os.path.dirname(os.path.abspath(__file__))  # dataset/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_differing_positions(
    sent_a: str,
    sent_b: str,
) -> tuple[
    list[tuple[int, int]],   # a_positions
    list[tuple[int, int]],   # b_positions
    str,                     # a_segments  (pipe-separated)
    str,                     # b_segments
]:
    """
    Use difflib to find word-level spans that differ between two sentences.

    Returns positions as (start_word_idx, end_word_idx) tuples where the range is
    [start, end) — matching the convention in run.py word_to_token_position().
    """
    words_a = sent_a.split()
    words_b = sent_b.split()

    matcher = SequenceMatcher(None, words_a, words_b, autojunk=False)

    a_positions: list[tuple[int, int]] = []
    b_positions: list[tuple[int, int]] = []
    a_segs: list[str] = []
    b_segs: list[str] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        if i1 < i2:
            a_positions.append((i1, i2))
            a_segs.append(" ".join(words_a[i1:i2]))
        if j1 < j2:
            b_positions.append((j1, j2))
            b_segs.append(" ".join(words_b[j1:j2]))

    return a_positions, b_positions, " | ".join(a_segs), " | ".join(b_segs)


def mid_word_position(sentence: str) -> tuple[int, int]:
    """Return the position tuple for the middle word of a sentence."""
    n = len(sentence.split())
    mid = max(0, n // 2)
    return (mid, mid + 1)


# ---------------------------------------------------------------------------
# Dataset 1 — Literal-Literal Paraphrase (PAWS label=1)
# ---------------------------------------------------------------------------

def build_literal_paraphrase(out_path: str) -> None:
    """
    Pull PAWS labeled_final train split, keep label=1 paraphrase pairs,
    compute word-level differing positions, write TARGET_SIZE rows.
    """
    print("=" * 60)
    print("Building literal_paraphrase_baseline.csv")
    print("=" * 60)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading PAWS labeled_final train split...")
    ds = load_dataset("paws", "labeled_final", split="train")
    print(f"  Total rows: {len(ds)}, paraphrases (label=1): "
          f"{sum(1 for x in ds['label'] if x == 1)}")

    rows = []
    scanned = 0

    for item in ds:
        if scanned >= MAX_SCAN:
            break
        scanned += 1

        if item["label"] != 1:
            continue

        s1: str = item["sentence1"].strip()
        s2: str = item["sentence2"].strip()

        # Skip trivially identical pairs
        if s1 == s2:
            continue

        a_pos, b_pos, a_seg, b_seg = find_differing_positions(s1, s2)

        # Need at least one differing span on each side
        if not a_pos or not b_pos:
            continue

        # s1 → 'paraphrase' role (mapped to slang_text in the loader)
        # s2 → 'normal' role    (mapped to literal_text in the loader)
        rows.append({
            "normal":               s2,
            "paraphrase":           s1,
            "literal_segments":     b_seg,
            "paraphrase_segments":  a_seg,
            "literal_positions":    str(b_pos),
            "paraphrase_positions": str(a_pos),
            "num_differences":      len(a_pos),
        })

        if len(rows) >= TARGET_SIZE:
            break

    if len(rows) < TARGET_SIZE:
        print(f"WARNING: Only collected {len(rows)} rows (target {TARGET_SIZE}). "
              "Consider increasing MAX_SCAN.")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample row:\n{df.iloc[0].to_dict()}\n")


# ---------------------------------------------------------------------------
# Dataset 2 — Identical (sentence compared to itself)
# ---------------------------------------------------------------------------

def build_identical(out_path: str) -> None:
    """
    Pull literal sentences from PAWS (sentence1 of any pair) and duplicate them.
    The pipeline will see zero distance at every layer — sanity-check floor.
    """
    print("=" * 60)
    print("Building identical_baseline.csv")
    print("=" * 60)

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("ERROR: 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)

    print("Loading PAWS labeled_final train split for sentence pool...")
    ds = load_dataset("paws", "labeled_final", split="train")

    rows = []
    seen: set[str] = set()

    for item in ds:
        sent = item["sentence1"].strip()

        # Deduplicate, require non-trivial length
        if sent in seen or len(sent.split()) < 5:
            continue
        seen.add(sent)

        pos = mid_word_position(sent)

        rows.append({
            "normal":               sent,
            "identical":            sent,
            "literal_segments":     sent.split()[pos[0]],
            "identical_segments":   sent.split()[pos[0]],
            "literal_positions":    str([pos]),
            "identical_positions":  str([pos]),
            "num_differences":      0,
        })

        if len(rows) >= TARGET_SIZE:
            break

    if len(rows) < TARGET_SIZE:
        print(f"WARNING: Only collected {len(rows)} rows (target {TARGET_SIZE}).")

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df)} rows → {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Sample row:\n{df.iloc[0].to_dict()}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lit_par_path = os.path.join(OUT_DIR, "literal_paraphrase_baseline.csv")
    identical_path = os.path.join(OUT_DIR, "identical_baseline.csv")

    build_literal_paraphrase(lit_par_path)
    build_identical(identical_path)

    print("Done. Files written:")
    print(f"  {lit_par_path}")
    print(f"  {identical_path}")
