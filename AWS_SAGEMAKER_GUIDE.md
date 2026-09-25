# AWS & SageMaker Guide — Business Entity Resolution Preprocessing

This is a detailed, beginner-friendly walkthrough for running the
preprocessing pipeline on AWS S3 + SageMaker Processing. It assumes you
have already run the pipeline successfully **locally** (see the main
`README.md`, sections 6–11) before spending any AWS money.

> **Note on execution status:** the steps and commands below were not run
> against a live AWS account as part of producing this repository (see
> "What was verified locally vs. not verified" in `README.md`). Follow
> them carefully and validate each step's actual output before trusting
> the pipeline at full scale.

---

## A. AWS account prerequisites

You need:
- An AWS account with billing enabled (S3 storage + SageMaker Processing
  both incur charges — see Section L).
- An IAM identity (user or role) for **you** (the human running these
  scripts) with permission to: create S3 buckets and objects, create/pass
  an IAM role, and call `sagemaker:CreateProcessingJob`. If you're on a
  shared/organization AWS account, ask an administrator for these
  permissions rather than requesting `AdministratorAccess`.
- The AWS CLI installed and configured (`aws configure`), or equivalent
  credentials available via environment variables / SSO / an assumed
  role. This project's scripts never accept or hardcode access keys —
  they use whatever credentials boto3/the SageMaker SDK resolve from your
  environment.
- **Region selection:** pick one region for this whole project (e.g.
  `us-east-1`) and use it consistently for the bucket and the SageMaker
  Processing job — S3 buckets and SageMaker jobs are region-scoped, and
  cross-region access adds latency/cost with no benefit here. Never
  hardcode a region in code; always pass `--region` or set
  `AWS_REGION`/`AWS_DEFAULT_REGION`.

---

## B. Create the S3 bucket

**Bucket naming:** use a globally unique, lowercase, dot-free name, e.g.
`<your-org>-entity-resolution-dev`. Dots are avoided so virtual-hosted-style
HTTPS URLs for the bucket don't run into TLS certificate wildcard issues.

**Console:**
1. S3 → *Create bucket*.
2. Bucket name: `<your-org>-entity-resolution-dev`. Region: your chosen
   region from Section A.
3. **Block Public Access:** leave all four boxes checked (the default) —
   this project's data should never be public.
4. **Bucket Versioning:** enable it. This is cheap protection against an
   accidental overwrite of `processed/` on a bad re-run.
5. **Default encryption:** enable SSE-S3 (`Amazon S3-managed keys
   (SSE-S3)`), or SSE-KMS if your org requires a customer-managed key.
6. Create the bucket.

**CLI equivalent:**
```bash
aws s3api create-bucket \
  --bucket <your-org>-entity-resolution-dev \
  --region <your-region> \
  --create-bucket-configuration LocationConstraint=<your-region>   # omit this flag if region is us-east-1

aws s3api put-bucket-versioning \
  --bucket <your-org>-entity-resolution-dev \
  --versioning-configuration Status=Enabled

aws s3api put-bucket-encryption \
  --bucket <your-org>-entity-resolution-dev \
  --server-side-encryption-configuration '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'

aws s3api put-public-access-block \
  --bucket <your-org>-entity-resolution-dev \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
```

---

## C. Create the folder/prefix structure

S3 doesn't require you to pre-create "folders" (they're just key
prefixes), but it's useful to know the layout `scripts/upload_to_s3.py`
and `scripts/run_processing.py` expect:

```
entity-resolution/
├── raw/
│   ├── train/   (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/    (test_source1.tsv, test_source2.tsv, test_source3.tsv)
├── processed/
│   ├── train/   (source1_clean.tsv, source2_clean.tsv, source3_clean.tsv, ground_truth_clean.tsv)
│   └── test/    (source1_clean.tsv, source2_clean.tsv, source3_clean.tsv)
└── metadata/
    ├── preprocessing_report.json
    ├── preprocessing_report.txt
    └── canonicalization_collisions_*.tsv
```

These prefixes get created automatically the first time you upload/write
to them — no explicit action needed here.

---

## D. Upload the raw TSVs

Raw files must be uploaded **untouched** (see `implementation.md` Section
5 on raw immutability).

**Python/boto3 method (recommended — this is what `scripts/upload_to_s3.py` does):**
```bash
python scripts/upload_to_s3.py \
  --bucket <your-org>-entity-resolution-dev \
  --local-dir /path/to/flat/dir/with/7/files \
  --prefix entity-resolution/raw \
  --region <your-region>
```
By default this **skips** any file that already exists at its destination
key (never silently overwrites raw data). Pass `--force` if you
deliberately want to replace an already-uploaded raw file.

**AWS CLI method (equivalent, manual):**
```bash
aws s3 cp train_source1.tsv       s3://<bucket>/entity-resolution/raw/train/train_source1.tsv       --sse AES256
aws s3 cp train_source2.tsv       s3://<bucket>/entity-resolution/raw/train/train_source2.tsv       --sse AES256
aws s3 cp train_source3.tsv       s3://<bucket>/entity-resolution/raw/train/train_source3.tsv       --sse AES256
aws s3 cp train_ground_truth.tsv  s3://<bucket>/entity-resolution/raw/train/train_ground_truth.tsv  --sse AES256
aws s3 cp test_source1.tsv        s3://<bucket>/entity-resolution/raw/test/test_source1.tsv          --sse AES256
aws s3 cp test_source2.tsv        s3://<bucket>/entity-resolution/raw/test/test_source2.tsv          --sse AES256
aws s3 cp test_source3.tsv        s3://<bucket>/entity-resolution/raw/test/test_source3.tsv          --sse AES256
```

**Verify the objects exist:**
```bash
aws s3 ls s3://<bucket>/entity-resolution/raw/train/
aws s3 ls s3://<bucket>/entity-resolution/raw/test/
```
You should see all 4 train files and all 3 test files listed with
non-zero sizes.

---

## E. Create the SageMaker execution IAM role

The Processing job runs *as* an IAM role, not as you. Create a
**least-privilege** role — do not use `AdministratorAccess`.

**Trust policy** (who can assume this role — SageMaker's service):
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Service": "sagemaker.amazonaws.com"},
    "Action": "sts:AssumeRole"
  }]
}
```

**Permissions policy** (minimum practical permissions — scoped to only
this project's prefix, not the whole bucket):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadRaw",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<bucket>",
        "arn:aws:s3:::<bucket>/entity-resolution/raw/*"
      ]
    },
    {
      "Sid": "ReadWriteProcessedAndMetadata",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::<bucket>",
        "arn:aws:s3:::<bucket>/entity-resolution/processed/*",
        "arn:aws:s3:::<bucket>/entity-resolution/metadata/*"
      ]
    },
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents",
        "logs:DescribeLogStreams"
      ],
      "Resource": "arn:aws:logs:*:*:log-group:/aws/sagemaker/*"
    },
    {
      "Sid": "ECRPullForProcessingContainer",
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage"
      ],
      "Resource": "*"
    }
  ]
}
```

**Console steps:**
1. IAM → Roles → *Create role*.
2. Trusted entity type: *AWS service* → Use case: *SageMaker* → *SageMaker - Execution*.
3. Attach a new customer-managed policy with the JSON above (replace
   `<bucket>` with your actual bucket name in both `Resource` entries).
4. Name it e.g. `EntityResolutionProcessingRole`, create it, and copy its
   ARN (`arn:aws:iam::<account-id>:role/EntityResolutionProcessingRole`)
   — you'll pass this to `scripts/run_processing.py --role-arn`.

---

## F. SageMaker setup

1. Open the SageMaker console, confirm you're in the same region as your
   bucket (Section A/B).
2. You do not need to pre-create anything else in the console —
   `scripts/run_processing.py` creates the Processing job via the
   SageMaker Python SDK.
3. **Instance type:** the default in this project is `ml.m5.4xlarge` (16
   vCPU / 64 GB RAM), matching `implementation.md` Section 14's sizing
   analysis for the ~2.5 GB / ~20M-row dataset. Override with
   `--instance-type` if you've benchmarked a different size as
   sufficient/insufficient.
4. **Framework version:** `scripts/run_processing.py --framework-version`
   defaults to `1.2-1` (an SKLearn container version). SageMaker
   periodically retires old container versions — before running, check
   the current list of supported `SKLearnProcessor` framework versions in
   the [SageMaker Python SDK documentation](https://sagemaker.readthedocs.io/)
   or the AWS Deep Learning Containers release notes for your region, and
   pass `--framework-version` explicitly if `1.2-1` is no longer listed.

---

## G. Run processing

```bash
python scripts/run_processing.py \
  --role-arn arn:aws:iam::<account-id>:role/EntityResolutionProcessingRole \
  --bucket <your-org>-entity-resolution-dev \
  --region <your-region> \
  --instance-type ml.m5.4xlarge \
  --instance-count 1 \
  --splits train test
```

Argument meanings:
- `--role-arn` — the IAM role from Section E (the job runs as this role).
- `--bucket` — where `raw/` lives and where `processed/`+`metadata/` will
  be written.
- `--region` — must match the bucket's region.
- `--raw-prefix` / `--processed-prefix` / `--metadata-prefix` — override
  if you used non-default prefixes in Section D (defaults:
  `entity-resolution/raw`, `entity-resolution/processed`,
  `entity-resolution/metadata`).
- `--instance-type` / `--instance-count` — compute sizing (Section F).
- `--framework-version` — the SKLearn container version (Section F).
- `--splits` — which splits to process (default: both).
- `--job-name` — defaults to an auto-generated, timestamped name; pass
  your own (ideally including a git commit SHA) for easier tracking.
- `--wait` — if passed, the script blocks and streams logs until the job
  finishes; otherwise it submits the job and returns immediately (check
  status via the console or Section H below).

---

## H. Monitor the job

**Console:** SageMaker → *Processing* → *Processing jobs* → click your job
name. The **Status** field shows `InProgress`, `Completed`, `Failed`, or
`Stopped`.

**Logs:** the same job page has a *Monitor* section linking to CloudWatch
Logs, or go directly to CloudWatch → Log groups →
`/aws/sagemaker/ProcessingJobs` → find the log stream matching your job
name.

**CLI:**
```bash
aws sagemaker describe-processing-job --processing-job-name <job-name> --query ProcessingJobStatus
aws logs tail /aws/sagemaker/ProcessingJobs --log-stream-name-prefix <job-name> --follow
```

**Failure diagnosis:** if status is `Failed`, check
`FailureReason` in `describe-processing-job` output and the CloudWatch
logs — a `FatalValidationError` from `src/preprocess.py` will appear
clearly in the log with an `ERROR:` prefix (see `README.md` section 15,
Troubleshooting, for what common ones mean).

---

## I. Inspect S3 output

```bash
aws s3 ls s3://<bucket>/entity-resolution/processed/train/
aws s3 ls s3://<bucket>/entity-resolution/processed/test/
aws s3 ls s3://<bucket>/entity-resolution/metadata/
```

Expect: `source1_clean.tsv`, `source2_clean.tsv`, `source3_clean.tsv` (and
`ground_truth_clean.tsv` for `train/` only) under each split, plus
`preprocessing_report.json`, `preprocessing_report.txt`, and several
`canonicalization_collisions_*.tsv` files under `metadata/`.

---

## J. Validate the output

Download first (see Section K), then either read
`preprocessing_report.json` directly, or run the independent checker:

```bash
python scripts/verify_outputs.py \
  --raw-dir ./raw_local \
  --processed-dir ./processed_local/processed \
  --metadata-dir ./processed_local/metadata \
  --splits train test
```

This independently re-checks (from the files themselves, not just trusting
the report): row counts, entity-ID set equality, required output columns,
and ground-truth referential integrity. It also reports the country
distribution and missing-value counts are present in
`preprocessing_report.json` if you want to eyeball those (e.g. confirm
`France` appears in the test country distribution and was not dropped or
rejected).

---

## K. Download results

```bash
mkdir -p ./processed_local
aws s3 cp s3://<bucket>/entity-resolution/processed/ ./processed_local/processed/ --recursive
aws s3 cp s3://<bucket>/entity-resolution/metadata/  ./processed_local/metadata/  --recursive

mkdir -p ./raw_local
aws s3 cp s3://<bucket>/entity-resolution/raw/train/ ./raw_local/ --recursive
aws s3 cp s3://<bucket>/entity-resolution/raw/test/  ./raw_local/ --recursive
```

---

## L. Cost / safety

- **SageMaker Processing** is billed per instance-second while the job
  runs. A single-instance `ml.m5.4xlarge` job over ~2.5 GB should complete
  in low single-digit minutes (not benchmarked on real hardware as part of
  this project — see `implementation.md` Section 14), so per-run compute
  cost should be modest, but **check current on-demand pricing on the AWS
  Pricing Calculator / SageMaker pricing page** before committing to a
  budget — exact figures are not reproduced here because they change.
- **EC2/SageMaker instances are not free** — a Processing job that fails
  to terminate correctly (rare, but possible with SDK bugs or console
  issues) keeps billing until it's stopped. If you ever see a job stuck
  in `InProgress` far longer than expected, stop it manually:
  ```bash
  aws sagemaker stop-processing-job --processing-job-name <job-name>
  ```
- **S3 storage/requests** — raw + processed + metadata for this dataset
  total a few GB; storage cost is minor. Requests (a handful of
  `PutObject`/`GetObject` calls per run) are negligible.
- **CloudWatch Logs** — default retention is indefinite unless you set a
  retention policy; for a project like this, consider setting the
  `/aws/sagemaker/ProcessingJobs` log group's retention to 30–90 days to
  avoid unbounded accumulation across many runs:
  ```bash
  aws logs put-retention-policy --log-group-name /aws/sagemaker/ProcessingJobs --retention-in-days 90
  ```
- **Cleanup:** SageMaker Processing jobs don't leave any persistent
  compute running after they finish (unlike, say, a SageMaker Notebook
  instance or an Endpoint) — there is no "stop the instance" step needed
  post-completion. If you also spun up anything else while testing (a
  Notebook instance, a persistent Studio app), stop/delete those
  separately, as those *do* bill continuously while running.

---

## Summary command reference

```bash
# 1. Upload raw data
python scripts/upload_to_s3.py --bucket <bucket> --local-dir <flat-dir> --region <region>

# 2. Run processing
python scripts/run_processing.py --role-arn <role-arn> --bucket <bucket> --region <region>

# 3. Monitor
aws sagemaker describe-processing-job --processing-job-name <job-name> --query ProcessingJobStatus

# 4. Download + verify
aws s3 cp s3://<bucket>/entity-resolution/processed/ ./processed_local/processed/ --recursive
aws s3 cp s3://<bucket>/entity-resolution/metadata/  ./processed_local/metadata/  --recursive
python scripts/verify_outputs.py --raw-dir <flat-raw-dir> --processed-dir ./processed_local/processed --metadata-dir ./processed_local/metadata
```
