#!/usr/bin/env python3
"""Parquet-cached TSV ingestion with strict string type-safety.

PURPOSE
-------
This module solves two problems with naively calling ``pd.read_csv`` on the
raw TSV files every run:

1. **NaN coercion**: by default pandas silently converts "NA", "NULL",
   "NaN" literals and genuinely empty cells into ``float NaN`` objects,
   which then leak through the pipeline as ``NaN`` instead of empty strings.
   We prevent this with ``keep_default_na=False, na_values=[]``.

2. **Repeat-load cost**: reading and parsing large TSV files on every
   preprocessing run wastes time and I/O.  We compress each raw TSV into
   a Parquet file (Snappy, all columns stored as dictionary-encoded strings)
   on the first load and serve subsequent reads directly from the cache.

CACHE INVALIDATION
------------------
The cache filename embeds the SHA-256 hash of the raw file's bytes.  If the
raw TSV is ever updated (even a single byte), the hash changes, a new
Parquet file is written, and the stale one is cleaned up automatically.  No
manual cache-busting is required.

COLUMN TYPE CONTRACT
--------------------
Every column is stored in Parquet as ``pa.dictionary(pa.int32(), pa.string())``
(dictionary-encoded UTF-8) for space efficiency.  On load the Parquet columns
are cast back to plain Python ``object`` (i.e. ``str``) dtype so that the
returned DataFrame is byte-for-byte identical to what ``pd.read_csv``
produces — downstream code never sees any difference.

PIPELINE COMPATIBILITY
----------------------
``load_tsv_cached()`` returns the *exact same DataFrame* as ``read_tsv()``
in ``src/preprocess.py``.  It is a drop-in replacement; existing validation,
normalization, and output-column checks are unaffected.

STANDALONE CLI
--------------
Run this script directly to pre-warm the cache for an entire dataset tree
before a full preprocessing run::

    python scripts/prepare_data.py \\
        --input-dir test_data \\
        --cache-dir cache \\
        --splits train test

Or import the function in any other module::

    from scripts.prepare_data import load_tsv_cached
    df = load_tsv_cached(Path("test_data/train/train_source1.tsv"),
                         cache_dir=Path("cache"))
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# pyarrow is an optional import so that this module can be imported even in
# environments where pyarrow is not yet installed — the function will fall
# back to plain read_csv and emit a warning.
# ---------------------------------------------------------------------------
try:
    import pyarrow as pa
    import pyarrow.parquet as pq

    _PYARROW_AVAILABLE = True
except ImportError:  # pragma: no cover
    _PYARROW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Compression codec written into every Parquet file.
# Snappy: lossless, very fast decompression, ~2-4× smaller than raw TSV.
_PARQUET_COMPRESSION = "snappy"

# Cache subdirectory name used when the caller does not provide --cache-dir.
_DEFAULT_CACHE_SUBDIR = "cache"

# Block size used by hashlib when streaming the file for its SHA-256.
_HASH_BLOCK = 1 << 20  # 1 MiB


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*'s raw bytes.

    Streams the file in 1 MiB blocks so even multi-GB TSVs are handled
    without loading the whole file into memory.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(_HASH_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _cache_path(tsv_path: Path, cache_dir: Path, file_hash: str) -> Path:
    """Return the Parquet cache path for *tsv_path* with the given hash.

    Layout: ``<cache_dir>/<stem>.<hash[:12]>.snappy.parquet``

    The first 12 hex characters of the SHA-256 (48 bits of entropy) are
    sufficient to distinguish any realistic set of TSV file versions while
    keeping filenames human-readable.
    """
    return cache_dir / f"{tsv_path.stem}.{file_hash[:12]}.snappy.parquet"


def _purge_stale_cache(cache_dir: Path, tsv_stem: str, current_hash_prefix: str) -> None:
    """Delete Parquet cache entries for *tsv_stem* whose hash prefix does
    not match *current_hash_prefix*.  Silently ignores missing files.
    """
    pattern = f"{tsv_stem}.*.snappy.parquet"
    for old in cache_dir.glob(pattern):
        # Keep files whose embedded hash prefix still matches.
        parts = old.stem.split(".")  # ["<stem>", "<hash>", "snappy"]
        if len(parts) >= 2 and parts[-2] == current_hash_prefix:
            continue
        try:
            old.unlink()
        except OSError:
            pass  # best-effort; leave it if we can't remove it


def _read_tsv_raw(path: Path) -> pd.DataFrame:
    """Read *path* as a TSV with strict NaN prevention.

    Uses ``keep_default_na=False`` and ``na_values=[]`` so that every cell —
    including literals like "NA", "NULL", "NaN", and truly empty cells — is
    loaded as a Python ``str``, never a ``float NaN``.  Every column gets
    ``dtype=str`` explicitly.

    This is the *only* place in this module where a raw TSV is read.
    """
    return pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_values=[],           # empty string list: no value is treated as NA
    )


def _coerce_all_columns_to_str(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every column in *df* has Python ``object`` (str) dtype.

    When a Parquet file is read back with ``pandas_metadata=False`` or with
    dictionary columns, some columns may come back as ``pd.CategoricalDtype``
    or ``pd.ArrowDtype``.  This function normalises all of them back to plain
    ``object`` so the returned DataFrame is identical in dtype to what
    ``_read_tsv_raw`` produces.

    Also replaces any genuine ``NaN``/``None`` that could have slipped in
    with empty string ``""``, guarding the strict-string contract.
    """
    for col in df.columns:
        if df[col].dtype != object:
            df[col] = df[col].astype(object)
        # Replace NaN/None → "" so downstream code never sees a float NaN.
        df[col] = df[col].fillna("")
    return df


def _write_parquet(df: pd.DataFrame, dest: Path) -> None:
    """Write *df* to *dest* as a Snappy-compressed Parquet file.

    All columns are dictionary-encoded (``pa.dictionary(pa.int32(),
    pa.string())``) which gives 3-10× compression over storing raw UTF-8
    strings, and is especially efficient for high-cardinality string columns
    (entity_id) and low-cardinality ones (country).

    The parent directory is created if it does not exist.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Build a PyArrow schema that stores every column as dict-encoded string.
    fields = [
        pa.field(col, pa.dictionary(pa.int32(), pa.string()))
        for col in df.columns
    ]
    schema = pa.schema(fields)

    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)
    pq.write_table(table, dest, compression=_PARQUET_COMPRESSION)


def _read_parquet(path: Path) -> pd.DataFrame:
    """Read a Parquet file and return a DataFrame with all-``object`` dtypes.

    ``read_pandas=False`` skips pandas-specific metadata so we always get a
    clean, predictable result regardless of which pandas version wrote the
    file.
    """
    table = pq.read_table(path)
    df = table.to_pandas(types_mapper=lambda _: pd.StringDtype())  # uniform dtype
    return _coerce_all_columns_to_str(df)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tsv_cached(
    tsv_path: Path,
    cache_dir: Optional[Path] = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load a raw TSV file, using a Parquet cache for fast subsequent reads.

    Parameters
    ----------
    tsv_path:
        Absolute or relative path to the raw ``.tsv`` file.
    cache_dir:
        Directory where Parquet cache files are stored.  Defaults to a
        ``cache/`` subdirectory next to *tsv_path*'s **parent's parent**
        (i.e. if *tsv_path* is ``data/train/train_source1.tsv``, the cache
        lives in ``data/cache/``).  Created automatically if it does not exist.
    verbose:
        Print a one-line status message (cache hit / cache miss + timing).

    Returns
    -------
    pd.DataFrame
        All columns have ``object`` dtype (plain Python ``str``).  Empty
        cells are ``""``; no column ever contains a ``float NaN``.
        The column set and row order are identical to calling
        ``pd.read_csv(tsv_path, sep="\\t", dtype=str,
        keep_default_na=False, na_values=[])``.

    Raises
    ------
    FileNotFoundError
        If *tsv_path* does not exist.
    RuntimeError
        If ``pyarrow`` is not installed *and* the cache Parquet file also
        does not exist — i.e. we can neither write nor read the cache.
        In practice this is unreachable if pyarrow is installed (see
        requirements.txt).
    """
    tsv_path = Path(tsv_path)
    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV file not found: {tsv_path}")

    # Resolve the cache directory.
    if cache_dir is None:
        cache_dir = tsv_path.parent.parent / _DEFAULT_CACHE_SUBDIR
    cache_dir = Path(cache_dir)

    t0 = time.perf_counter()

    # Compute the hash of the source file.
    file_hash = _sha256_file(tsv_path)
    parquet_file = _cache_path(tsv_path, cache_dir, file_hash)

    if parquet_file.exists():
        # ── Cache HIT ────────────────────────────────────────────────────
        df = _read_parquet(parquet_file)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"[prepare_data] cache HIT  {tsv_path.name}"
                f"  ({len(df):,} rows, {elapsed:.3f}s)"
            )
        return df

    # ── Cache MISS ───────────────────────────────────────────────────────
    if not _PYARROW_AVAILABLE:
        # pyarrow not installed: fall back to raw TSV read, warn the user.
        import warnings
        warnings.warn(
            "pyarrow is not installed; Parquet caching is disabled. "
            "Install it with: pip install 'pyarrow>=14.0,<20.0'",
            RuntimeWarning,
            stacklevel=2,
        )
        df = _read_tsv_raw(tsv_path)
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"[prepare_data] fallback  {tsv_path.name}"
                f"  ({len(df):,} rows, {elapsed:.3f}s, no cache written)"
            )
        return df

    # Read the raw TSV.
    df = _read_tsv_raw(tsv_path)

    # Write to Parquet cache (atomic: write to a temp name, then rename).
    tmp = parquet_file.with_suffix(".tmp")
    try:
        _write_parquet(df, tmp)
        tmp.rename(parquet_file)
    except Exception:
        # If the write fails for any reason (disk full, permissions, …),
        # remove the incomplete temp file and return the in-memory DataFrame
        # rather than raising — the pipeline can still proceed.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        if verbose:
            elapsed = time.perf_counter() - t0
            print(
                f"[prepare_data] cache WRITE FAILED {tsv_path.name}"
                f"  ({len(df):,} rows, {elapsed:.3f}s)"
            )
        return df

    # Clean up any stale cache entries for this TSV stem.
    _purge_stale_cache(cache_dir, tsv_path.stem, file_hash[:12])

    if verbose:
        elapsed = time.perf_counter() - t0
        print(
            f"[prepare_data] cache MISS {tsv_path.name}"
            f"  ({len(df):,} rows, {elapsed:.3f}s, cache written)"
        )
    return df


# ---------------------------------------------------------------------------
# CLI — pre-warm cache for a full dataset tree
# ---------------------------------------------------------------------------

_SPLIT_FILES = {
    "train": [
        "train_source1.tsv",
        "train_source2.tsv",
        "train_source3.tsv",
        "train_ground_truth.tsv",
    ],
    "test": [
        "test_source1.tsv",
        "test_source2.tsv",
        "test_source3.tsv",
    ],
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help=(
            "Dataset root directory containing train/ and/or test/ subdirectories "
            "with raw TSV files (same layout expected by src/preprocess.py)."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Directory where Parquet cache files are written.  "
            "Defaults to <input-dir>/cache/."
        ),
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        choices=["train", "test"],
        help="Which splits to cache (default: both train and test).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-file status lines.",
    )
    return parser


def main(argv=None) -> int:
    args = _build_arg_parser().parse_args(argv)
    input_dir = Path(args.input_dir)
    cache_dir = Path(args.cache_dir) if args.cache_dir else input_dir / _DEFAULT_CACHE_SUBDIR
    verbose = not args.quiet

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    if not _PYARROW_AVAILABLE:
        print(
            "ERROR: pyarrow is not installed.  "
            "Run: pip install 'pyarrow>=14.0,<20.0'",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []
    total_rows = 0
    t_start = time.perf_counter()

    for split in args.splits:
        split_dir = input_dir / split
        if not split_dir.exists():
            print(
                f"  WARNING: split directory not found, skipping: {split_dir}",
                file=sys.stderr,
            )
            continue

        for filename in _SPLIT_FILES[split]:
            tsv_path = split_dir / filename
            if not tsv_path.exists():
                # Ground truth is optional for the test split in some setups.
                if verbose:
                    print(f"  skip  {tsv_path.relative_to(input_dir)}  (not found)")
                continue
            try:
                df = load_tsv_cached(tsv_path, cache_dir=cache_dir, verbose=verbose)
                total_rows += len(df)
            except Exception as exc:
                msg = f"ERROR loading {tsv_path}: {exc}"
                print(msg, file=sys.stderr)
                errors.append(msg)

    elapsed = time.perf_counter() - t_start
    print(
        f"\nDone. {total_rows:,} total rows cached in {elapsed:.2f}s.  "
        f"Cache: {cache_dir}"
    )

    if errors:
        print(f"\n{len(errors)} error(s) encountered:", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
