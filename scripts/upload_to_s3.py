#!/usr/bin/env python3
"""Upload the seven raw TSV files to S3, unmodified.

Raw immutability (implementation.md Section 5): by default this script
refuses to overwrite an existing key. Pass --force to intentionally
replace an already-uploaded raw file.

Credentials: this script uses whatever AWS credentials boto3 finds via the
standard credential chain (environment variables, `~/.aws/credentials`,
an assumed role, SSO, or an EC2/SageMaker instance role). No access keys
are hardcoded or requested by this script. See AWS_SAGEMAKER_GUIDE.md for
setup instructions.

Usage:
    python scripts/upload_to_s3.py \\
        --bucket my-org-entity-resolution-dev \\
        --local-dir ./data_local \\
        [--prefix entity-resolution/raw] \\
        [--region us-east-1] \\
        [--force]

`--local-dir` must contain the seven raw files, flat, with their standard
names (train_source1.tsv, train_source2.tsv, train_source3.tsv,
train_ground_truth.tsv, test_source1.tsv, test_source2.tsv,
test_source3.tsv).
"""
from __future__ import annotations

import argparse
import sys

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
except ImportError:  # pragma: no cover
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

# Maps: destination key suffix (relative to --prefix) -> expected local file name.
FILES = {
    "train/train_source1.tsv": "train_source1.tsv",
    "train/train_source2.tsv": "train_source2.tsv",
    "train/train_source3.tsv": "train_source3.tsv",
    "train/train_ground_truth.tsv": "train_ground_truth.tsv",
    "test/test_source1.tsv": "test_source1.tsv",
    "test/test_source2.tsv": "test_source2.tsv",
    "test/test_source3.tsv": "test_source3.tsv",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket", required=True, help="Destination S3 bucket name.")
    parser.add_argument("--local-dir", required=True, help="Local directory containing the 7 raw TSV files.")
    parser.add_argument("--prefix", default="entity-resolution/raw",
                         help="S3 key prefix for raw data (default: entity-resolution/raw).")
    parser.add_argument("--region", default=None,
                         help="AWS region (default: whatever the boto3/AWS CLI config resolves to).")
    parser.add_argument("--force", action="store_true",
                         help="Overwrite an existing object instead of skipping it.")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if boto3 is None:
        print("ERROR: boto3 is not installed. Run: pip install boto3", file=sys.stderr)
        return 1

    from pathlib import Path
    local_dir = Path(args.local_dir)

    missing_local = [name for name in FILES.values() if not (local_dir / name).exists()]
    if missing_local:
        print(f"ERROR: missing local file(s) in {local_dir}: {missing_local}", file=sys.stderr)
        return 1

    session_kwargs = {"region_name": args.region} if args.region else {}
    try:
        s3 = boto3.client("s3", **session_kwargs)
    except NoCredentialsError:
        print("ERROR: no AWS credentials found. Configure the AWS CLI (`aws configure`) "
              "or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_SESSION_TOKEN.",
              file=sys.stderr)
        return 1

    uploaded, skipped = 0, 0
    for key_suffix, local_name in FILES.items():
        key = f"{args.prefix}/{key_suffix}"
        local_path = local_dir / local_name

        if not args.force:
            try:
                s3.head_object(Bucket=args.bucket, Key=key)
                print(f"SKIP (already exists): s3://{args.bucket}/{key}")
                skipped += 1
                continue
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                if code not in ("404", "NoSuchKey", "NotFound"):
                    print(f"ERROR checking s3://{args.bucket}/{key}: {e}", file=sys.stderr)
                    return 1

        try:
            s3.upload_file(
                str(local_path), args.bucket, key,
                ExtraArgs={"ServerSideEncryption": "AES256"},
            )
        except ClientError as e:
            print(f"ERROR uploading {local_path} -> s3://{args.bucket}/{key}: {e}", file=sys.stderr)
            return 1

        print(f"Uploaded: {local_path} -> s3://{args.bucket}/{key}")
        uploaded += 1

    print(f"\nDone. Uploaded: {uploaded}, Skipped (already present): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
