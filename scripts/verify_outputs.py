#!/usr/bin/env python3
"""Independently verify a completed preprocessing run's outputs.

Re-checks, from the files on disk (raw + processed + metadata), the
invariants the pipeline itself enforces during the run:
  * row counts preserved for every source/split
  * entity_id set equality for every source/split
  * required output columns present
  * ground truth referential integrity (train only)
  * preprocessing_report.json exists and reports status "completed"

This is meant to be run AFTER downloading S3 outputs locally (see
AWS_SAGEMAKER_GUIDE.md section J/K), or against a local run's output-dir,
as an independent sanity check separate from the pipeline's own internal
validation.

Usage:
    python scripts/verify_outputs.py \\
        --raw-dir ./data_local \\
        --processed-dir ./processed_local/processed \\
        --metadata-dir ./processed_local/metadata \\
        [--splits train test]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = [
    "entity_id",
    "business_name", "business_name_normalized", "business_name_canonical",
    "business_address", "business_address_normalized", "business_address_canonical",
    "address_landmark",
    "country", "country_normalized",
    "business_name_is_missing", "business_address_is_missing",
]

RAW_FILE_MAP = {
    "train": {
        "source1": "train_source1.tsv",
        "source2": "train_source2.tsv",
        "source3": "train_source3.tsv",
    },
    "test": {
        "source1": "test_source1.tsv",
        "source2": "test_source2.tsv",
        "source3": "test_source3.tsv",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", required=True, help="Directory containing the raw TSVs (flat, standard names).")
    parser.add_argument("--processed-dir", required=True, help="Directory containing processed/<split>/*_clean.tsv.")
    parser.add_argument("--metadata-dir", required=True, help="Directory containing preprocessing_report.json.")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    metadata_dir = Path(args.metadata_dir)

    errors = []
    warnings = []

    report_path = metadata_dir / "preprocessing_report.json"
    if not report_path.exists():
        errors.append(f"Missing report: {report_path}")
    else:
        report = json.loads(report_path.read_text())
        if report.get("status") != "completed":
            errors.append(f"Report status is '{report.get('status')}', expected 'completed'")

    for split in args.splits:
        for source_key, raw_name in RAW_FILE_MAP[split].items():
            raw_path = raw_dir / raw_name
            clean_path = processed_dir / split / f"{source_key}_clean.tsv"

            if not raw_path.exists():
                errors.append(f"Missing raw file: {raw_path}")
                continue
            if not clean_path.exists():
                errors.append(f"Missing cleaned file: {clean_path}")
                continue

            raw_df = pd.read_csv(raw_path, sep="\t", dtype=str, keep_default_na=False)
            clean_df = pd.read_csv(clean_path, sep="\t", dtype=str, keep_default_na=False)

            missing_cols = [c for c in REQUIRED_COLUMNS if c not in clean_df.columns]
            if missing_cols:
                errors.append(f"{clean_path}: missing required columns {missing_cols}")

            if len(raw_df) != len(clean_df):
                errors.append(
                    f"{clean_path}: row count mismatch (raw={len(raw_df)}, clean={len(clean_df)})"
                )

            raw_ids = set(raw_df["entity_id"])
            clean_ids = set(clean_df["entity_id"])
            if raw_ids != clean_ids:
                missing = raw_ids - clean_ids
                added = clean_ids - raw_ids
                errors.append(
                    f"{clean_path}: entity_id set mismatch "
                    f"(missing={len(missing)}, added={len(added)})"
                )

            dup_count = int(clean_df["entity_id"].duplicated().sum())
            if dup_count > 0:
                errors.append(f"{clean_path}: {dup_count} duplicate entity_id values found")

        if split == "train":
            gt_raw_path = raw_dir / "train_ground_truth.tsv"
            gt_clean_path = processed_dir / "train" / "ground_truth_clean.tsv"
            s1_clean_path = processed_dir / "train" / "source1_clean.tsv"
            s2_clean_path = processed_dir / "train" / "source2_clean.tsv"
            s3_clean_path = processed_dir / "train" / "source3_clean.tsv"

            if not gt_clean_path.exists():
                errors.append(f"Missing cleaned ground truth: {gt_clean_path}")
            elif all(p.exists() for p in (s1_clean_path, s2_clean_path, s3_clean_path)):
                gt = pd.read_csv(gt_clean_path, sep="\t", dtype=str, keep_default_na=False)
                s1_ids = set(pd.read_csv(s1_clean_path, sep="\t", dtype=str, keep_default_na=False)["entity_id"])
                s2_ids = set(pd.read_csv(s2_clean_path, sep="\t", dtype=str, keep_default_na=False)["entity_id"])
                s3_ids = set(pd.read_csv(s3_clean_path, sep="\t", dtype=str, keep_default_na=False)["entity_id"])

                unknown_s1 = set(gt["source1_entity_id"]) - s1_ids
                if unknown_s1:
                    errors.append(f"{gt_clean_path}: {len(unknown_s1)} source1_entity_id not found in source1_clean.tsv")

                all_tokens = set()
                for cell in gt["matched_entity_ids"]:
                    if cell:
                        all_tokens.update(cell.split(","))
                s2_tokens = {t for t in all_tokens if t.startswith("S2-")}
                s3_tokens = {t for t in all_tokens if t.startswith("S3-")}
                unknown_s2 = s2_tokens - s2_ids
                unknown_s3 = s3_tokens - s3_ids
                if unknown_s2:
                    errors.append(f"{gt_clean_path}: {len(unknown_s2)} referenced S2 ids not found in source2_clean.tsv")
                if unknown_s3:
                    errors.append(f"{gt_clean_path}: {len(unknown_s3)} referenced S3 ids not found in source3_clean.tsv")

    print("=" * 70)
    print("Verification summary")
    print("=" * 70)
    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")
        print("\nRESULT: FAILED")
        return 1

    print("\nAll checks passed: row counts preserved, entity_id sets equal, "
          "required columns present, ground truth referentially valid.")
    print("\nRESULT: PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
