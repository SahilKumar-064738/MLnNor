#!/usr/bin/env python3
"""Launch the SageMaker Processing job that runs src/preprocess.py against
the raw data already uploaded to S3 (see scripts/upload_to_s3.py).

This script only ORCHESTRATES the job (uploads the code, configures the
container, sets input/output S3 locations) — it does not implement any
cleaning logic itself; that all lives in src/preprocess.py, which is the
exact same module used for local execution (implementation.md Section 26,
item 9: no divergent local/SageMaker implementations).

Credentials/role: uses an IAM role ARN you provide (--role-arn), assumed by
the SageMaker Processing job itself — no AWS access keys are used or
requested by this script directly. The caller (you, running this script)
still needs your own AWS CLI/SSO credentials configured with permission to
call `sagemaker:CreateProcessingJob` and `iam:PassRole` for --role-arn. See
AWS_SAGEMAKER_GUIDE.md.

Region: never hardcoded. Pass --region explicitly, or rely on the AWS
CLI/SDK default region resolution (AWS_REGION / AWS_DEFAULT_REGION /
`aws configure`).

SKLearnProcessor framework_version: the value below is a CLI default you
can override with --framework-version. AWS periodically retires old
container versions; verify the version you use is still listed as
supported before relying on it (see AWS_SAGEMAKER_GUIDE.md, section F,
and the SageMaker "Available Deep Learning Containers"/"scikit-learn
container" documentation for the current list) — do not assume any
version baked into this script remains valid indefinitely.

Usage:
    python scripts/run_processing.py \\
        --role-arn arn:aws:iam::123456789012:role/EntityResolutionProcessingRole \\
        --bucket my-org-entity-resolution-dev \\
        --region us-east-1 \\
        [--raw-prefix entity-resolution/raw] \\
        [--processed-prefix entity-resolution/processed] \\
        [--metadata-prefix entity-resolution/metadata] \\
        [--instance-type ml.m5.4xlarge] \\
        [--instance-count 1] \\
        [--framework-version 1.2-1] \\
        [--splits train test] \\
        [--job-name entity-resolution-preprocess-<git-sha>-<timestamp>]
"""
from __future__ import annotations

import argparse
import sys
import time


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--role-arn", required=True, help="SageMaker execution IAM role ARN (least-privilege; see AWS_SAGEMAKER_GUIDE.md).")
    parser.add_argument("--bucket", required=True, help="S3 bucket holding raw/ and receiving processed/+metadata/.")
    parser.add_argument("--region", default=None, help="AWS region. No default — must be set here, via AWS_REGION, or via AWS CLI config.")
    parser.add_argument("--raw-prefix", default="entity-resolution/raw", help="S3 prefix for raw input (default: entity-resolution/raw).")
    parser.add_argument("--processed-prefix", default="entity-resolution/processed", help="S3 prefix for cleaned output (default: entity-resolution/processed).")
    parser.add_argument("--metadata-prefix", default="entity-resolution/metadata", help="S3 prefix for reports/diagnostics (default: entity-resolution/metadata).")
    parser.add_argument("--instance-type", default="ml.m5.4xlarge", help="Processing instance type (default: ml.m5.4xlarge, per implementation.md Section 14).")
    parser.add_argument("--instance-count", type=int, default=1)
    parser.add_argument("--framework-version", default="1.2-1",
                         help="SKLearnProcessor framework_version. VERIFY this is still a supported "
                              "version in your region before running (see module docstring).")
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    parser.add_argument("--job-name", default=None, help="Defaults to an auto-generated timestamped name.")
    parser.add_argument("--wait", action="store_true", help="Block until the job completes (default: submit and return).")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    try:
        from sagemaker.sklearn.processing import SKLearnProcessor
        from sagemaker.processing import ProcessingInput, ProcessingOutput
        import sagemaker
    except ImportError:
        print(
            "ERROR: the 'sagemaker' package is not installed. Run: pip install sagemaker\n"
            "(This script only needs to run on your local/orchestrating machine, not inside "
            "the Processing container itself.)",
            file=sys.stderr,
        )
        return 1

    boto_session = None
    if args.region:
        import boto3
        boto_session = boto3.Session(region_name=args.region)

    sagemaker_session = sagemaker.Session(boto_session=boto_session) if boto_session else sagemaker.Session()
    resolved_region = sagemaker_session.boto_region_name
    if not resolved_region:
        print(
            "ERROR: no AWS region resolved. Pass --region explicitly or set "
            "AWS_REGION / configure it via `aws configure`.",
            file=sys.stderr,
        )
        return 1

    job_name = args.job_name or f"entity-resolution-preprocess-{int(time.time())}"

    processor = SKLearnProcessor(
        framework_version=args.framework_version,
        role=args.role_arn,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        base_job_name="entity-resolution-preprocess",
        sagemaker_session=sagemaker_session,
    )

    inputs = [
        ProcessingInput(
            source=f"s3://{args.bucket}/{args.raw_prefix}/train",
            destination="/opt/ml/processing/input/train",
        ),
        ProcessingInput(
            source=f"s3://{args.bucket}/{args.raw_prefix}/test",
            destination="/opt/ml/processing/input/test",
        ),
    ]
    outputs = [
        ProcessingOutput(
            source="/opt/ml/processing/output/processed",
            destination=f"s3://{args.bucket}/{args.processed_prefix}",
        ),
        ProcessingOutput(
            source="/opt/ml/processing/output/metadata",
            destination=f"s3://{args.bucket}/{args.metadata_prefix}",
        ),
    ]

    print(f"Submitting SageMaker Processing job '{job_name}' in region {resolved_region}...")
    print(f"  Instance:  {args.instance_count}x {args.instance_type}")
    print(f"  Input:     s3://{args.bucket}/{args.raw_prefix}/{{train,test}}")
    print(f"  Output:    s3://{args.bucket}/{args.processed_prefix} , s3://{args.bucket}/{args.metadata_prefix}")

    try:
        processor.run(
            code="src/preprocess.py",
            inputs=inputs,
            outputs=outputs,
            arguments=[
                "--input-dir", "/opt/ml/processing/input",
                "--output-processed-dir", "/opt/ml/processing/output/processed",
                "--output-metadata-dir", "/opt/ml/processing/output/metadata",
                "--splits", *args.splits,
            ],
            job_name=job_name,
            logs=args.wait,
            wait=args.wait,
        )
    except Exception as e:  # noqa: BLE001 - surface any SageMaker/boto error clearly
        print(f"ERROR: failed to submit/run Processing job: {e}", file=sys.stderr)
        return 1

    print(f"\nJob '{job_name}' submitted.")
    print("Monitor it in the SageMaker console under Processing jobs, or CloudWatch Logs "
          "group /aws/sagemaker/ProcessingJobs. See AWS_SAGEMAKER_GUIDE.md section H.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
