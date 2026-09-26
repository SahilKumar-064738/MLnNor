# Business Entity Resolution — Data Ingestion & Preprocessing

> **This repository stops at cleaned/canonicalized datasets and diagnostics.
> Entity resolution/matching (embeddings, candidate generation, similarity
> scoring, matching) is NOT implemented here.** See "Next-stage handoff"
> at the bottom.

## 1. Project purpose

This project implements the raw-data ingestion and preprocessing stage of
a Business Entity Resolution pipeline: seven TSV files (three "sources" of
business records plus a training ground-truth file) are validated,
cleaned, normalized, and canonicalized, with every step designed to be
deterministic, auditable, and safe on millions of rows. The output is a
set of cleaned TSVs plus machine-readable data-quality reports — ready to
hand off to a later embeddings/matching stage, which this repository does
not implement.

Full design rationale, measured dataset statistics, and before/after
examples live in [`implementation.md`](implementation.md) — read that
first; this README is the "how to run it" companion.

## 2. Architecture

```
Local 7 TSVs
   → Local schema/structural validation (fail fast, before any S3 cost)
   → Upload raw, untouched, to S3 raw/
   → SageMaker Processing Job  (src/preprocess.py — same code as local)
        → Read TSV (sep="\t", dtype=str, keep_default_na=False)
        → Schema + entity-ID validation
        → Ground-truth referential validation (train only)
        → Missing-value detection & explicit-null encoding
        → Unicode NFKC + case + whitespace normalization
        → Business-name canonical suffix mapping
        → Address canonical abbreviation mapping + landmark extraction
        → Country normalization (open-set, no hardcoded universe)
        → Canonicalization collision diagnostics (report only, no deletion)
        → Exact-duplicate-row diagnostics (report only, no deletion)
        → Row-count + entity-ID-set invariant checks (fail job on violation)
        → Write cleaned TSVs + preprocessing_report.json/.txt
   → S3 processed/ + metadata/
```

One Python module, `src/preprocess.py`, implements the whole pipeline and
is used **identically** for local runs and inside the SageMaker Processing
container — only the `--input-dir` / `--output-*-dir` paths differ between
the two.

## 3. Repository structure

```
business-entity-resolution/
├── src/
│   ├── __init__.py
│   ├── config.py            # schemas, mappings, output columns, diagnostic limits
│   ├── normalization.py     # single-value + vectorized normalization functions
│   ├── validation.py        # schema/ID/invariant/ground-truth validation
│   ├── diagnostics.py       # memory-safe collision & exact-duplicate reports
│   └── preprocess.py        # CLI entrypoint (local AND SageMaker)
├── scripts/
│   ├── upload_to_s3.py      # upload the 7 raw files, unmodified, to S3
│   ├── run_processing.py    # launch the SageMaker Processing job
│   ├── verify_outputs.py    # independent post-hoc invariant check
│   └── create_sample.py     # build a small sample from the full files
├── tests/
│   ├── test_normalization.py
│   ├── test_validation.py
│   ├── test_diagnostics.py
│   └── test_preprocess_integration.py   # end-to-end run via the real CLI
├── data/
│   └── README.md            # where to place the 7 TSVs (no data committed)
├── requirements.txt
├── pytest.ini
├── implementation.md        # the detailed design spec/source of truth
├── AWS_SAGEMAKER_GUIDE.md   # beginner-friendly, step-by-step AWS walkthrough
├── .gitignore
└── README.md                # this file
```

## 4. Dataset placement

Put the seven raw TSV files under `data/full/train/` and `data/full/test/`
using their standard names. See [`data/README.md`](data/README.md) for the
exact expected layout. **No data is committed to this repository.**

## 5. Installation

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires Python 3.9+ (uses `from __future__ import annotations` and
modern type hints; tested with pandas 2.x/3.x — see `requirements.txt`).

## 6. Running unit tests

```bash
python -m pytest
```

This runs `tests/test_normalization.py`, `tests/test_validation.py`,
`tests/test_diagnostics.py`, and an end-to-end integration test
(`tests/test_preprocess_integration.py`) that writes a tiny 7-file sample
to a temp directory and runs the real `src.preprocess.main()` CLI entrypoint
against it — the same code path used for the full local and SageMaker runs.

## 7. Creating sample data

Once the full files are in `data/full/`:

```bash
python scripts/create_sample.py \
  --input-dir data/full \
  --output-dir data/sample \
  --rows 500 \
  --keep-referenced
```

This reads each file in chunks (never loading a full 5M-row file into
memory just to take its first 500 rows), keeps the first `--rows` rows of
each Source 1/2/3 file, and keeps only the training ground-truth rows
whose `source1_entity_id` is in the sampled Source 1 file.
`--keep-referenced` additionally pulls in any Source 2/3 rows the sampled
ground truth references but that fell outside the row-prefix sample, so
the sample passes ground-truth referential validation unmodified.

## 8. Running sample preprocessing

```bash
python -m src.preprocess \
  --input-dir data/sample \
  --output-dir outputs/sample
```

This creates `outputs/sample/processed/{train,test}/*.tsv` and
`outputs/sample/metadata/preprocessing_report.{json,txt}` plus
`canonicalization_collisions_*.tsv` diagnostic files.

Exit code is `0` on success, `1` on any fatal validation error (with a
clear `ERROR: ...` message on stderr and no report file written).

## 9. Inspecting sample results

```bash
# Row counts / stats
python -c "
import json
r = json.load(open('outputs/sample/metadata/preprocessing_report.json'))
print(r['splits']['train']['source1'])
print(r['splits']['train']['ground_truth'])
"

# Spot-check cleaned columns
head -5 outputs/sample/processed/train/source1_clean.tsv
```

Expected columns in every `*_clean.tsv`: `entity_id, business_name,
business_name_normalized, business_name_canonical, business_address,
business_address_normalized, business_address_canonical,
address_landmark, country, country_normalized, business_name_is_missing,
business_address_is_missing`. Original `business_name`/`business_address`/
`country` are always preserved verbatim for auditability.
`processed/train/ground_truth_clean.tsv` keeps `source1_entity_id,
matched_entity_ids` unchanged (validated, never transformed).

## 10. Running full local preprocessing

```bash
python -m src.preprocess \
  --input-dir data/full \
  --output-dir outputs/full
```

The pipeline processes one source file at a time and frees each raw/clean
DataFrame before moving to the next (see `src/preprocess.py::run`), and
all normalization is vectorized (pandas `.str` operations / column-wise
list comprehensions) rather than `df.apply(axis=1)`, so it comfortably
handles the ~2.5 GB / ~20M-row full dataset on a single machine with
enough RAM (see Section 14 of `implementation.md` for the recommended
SageMaker instance size; a modern laptop with 16+ GB free RAM should also
manage a full local run, though this has not been benchmarked on real
hardware as part of this project — see the note in `implementation.md`
Section 14).

## 11. Verifying invariants

The pipeline enforces these invariants itself during the run (fatal error,
non-zero exit, no output written, if violated):

- `input_rows == output_rows` for every one of the 7 files
- the input and output `entity_id` sets are exactly equal (not just equal
  in count — a swap would still be caught)
- no duplicate `entity_id` in any source file
- every ground-truth reference resolves against the cleaned Source 2/3 id
  sets, with no malformed or duplicate-within-cell ids

You can additionally re-check a completed run independently:

```bash
python scripts/verify_outputs.py \
  --raw-dir data/full/train \
  --processed-dir outputs/full/processed \
  --metadata-dir outputs/full/metadata \
  --splits train test
```

(Point `--raw-dir` at a flat directory containing the raw files with their
standard names — e.g. copy or symlink `train_*.tsv`/`test_*.tsv` into one
directory, since `verify_outputs.py` checks both splits' files by name.)

## 12. Uploading raw data to S3

```bash
python scripts/upload_to_s3.py \
  --bucket <your-bucket> \
  --local-dir data/full/train    # or wherever the flat 7 files are staged
  --prefix entity-resolution/raw
```

Raw files are uploaded **unmodified**, with server-side encryption
(`AES256`), and are never overwritten unless you pass `--force` (raw
immutability — see `implementation.md` Section 5). Full step-by-step AWS
setup (bucket, IAM, encryption, Block Public Access) is in
[`AWS_SAGEMAKER_GUIDE.md`](AWS_SAGEMAKER_GUIDE.md).

## 13. Running SageMaker Processing

```bash
python scripts/run_processing.py \
  --role-arn arn:aws:iam::<account-id>:role/EntityResolutionProcessingRole \
  --bucket <your-bucket> \
  --region <your-region> \
  --instance-type ml.m5.4xlarge
```

Full walkthrough (IAM role creation, launching, monitoring, troubleshooting)
in [`AWS_SAGEMAKER_GUIDE.md`](AWS_SAGEMAKER_GUIDE.md).

## 14. Inspecting S3 outputs

```bash
aws s3 ls s3://<your-bucket>/entity-resolution/processed/train/
aws s3 ls s3://<your-bucket>/entity-resolution/metadata/
aws s3 cp s3://<your-bucket>/entity-resolution/metadata/preprocessing_report.json .
```

## 15. Troubleshooting

| Symptom                                                                        | Likely cause                                                                                             | Fix                                                                                                                                 |
| ------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `ERROR: schema mismatch for ...`                                               | Wrong file in the wrong split/source slot, or upstream export changed columns                            | Re-check the file's header row against `src/config.py: SOURCE_COLUMNS` / `GT_COLUMNS`                                               |
| `ERROR: malformed entity_id in ...`                                            | A source file has ids from the wrong source (e.g. `S2-` ids in a `source1` file) or an unexpected format | Confirm you passed the right file to the right source slot                                                                          |
| `ERROR: duplicate entity_id in ...`                                            | A data refresh introduced duplicate ids                                                                  | Fatal by design — investigate upstream, do not silently dedupe                                                                      |
| `ERROR: row-count invariant failed` / `ERROR: ID-set invariant failed`         | A bug in cleaning logic dropped/added/changed rows                                                       | Treat as a code regression — these invariants exist to catch exactly this, not because the shipped dataset is expected to fail them |
| `ERROR: ground truth references unknown source1_entity_id` / unknown S2/S3 ids | Mismatched file versions (ground truth built against a different Source 2/3 export)                      | Confirm all 4 training files are the same version/export                                                                            |
| Job fails with "missing expected input file(s)"                                | `--input-dir` doesn't contain the expected `train/`/`test/` subfolders with standard file names          | Check the layout in `data/README.md`                                                                                                |
| Local run runs out of memory                                                   | Machine has far less RAM than recommended                                                                | Test with `scripts/create_sample.py` first; run the full dataset on SageMaker (`ml.m5.4xlarge`+) instead of a laptop                |
| SageMaker job stuck `InProgress`                                               | Instance under-provisioned for actual data volume                                                        | Check CloudWatch Logs for the current stage; scale up instance type before assuming a code bug                                      |

## 16. Next-stage handoff

```
THIS REPOSITORY (implemented)
──────────────────────────────
Raw 7 TSVs → S3 raw/ → SageMaker Processing → cleaned + normalized +
canonical fields → S3 processed/ + metadata/preprocessing_report.*

NEXT STAGE (NOT implemented here)
──────────────────────────────
S3 processed/ → feature engineering → multilingual embeddings (needed
specifically because normalization here cannot equate e.g. "Raj
Investments LLP" with its Tamil-script transliteration of the same
business — see implementation.md Section 22) → candidate generation →
similarity scoring → entity matching → evaluation against
ground_truth_clean.tsv → final submission
```

The next stage should consume `processed/{train,test}/source{1,2,3}_clean.tsv`
and `processed/train/ground_truth_clean.tsv` — never the `raw/` files
directly, and never any test ground truth (none exists/is fabricated).

---

## Design decisions / resolved inconsistencies

Per the task brief's instruction to resolve spec inconsistencies "in the
safest implementation-oriented way" and document them here:

1. **CLI output-directory flags.** The outer task brief suggests a single
   `--output-dir` flag; `implementation.md` Section 15.4 uses separate
   `--output-processed-dir` / `--output-metadata-dir` (matching SageMaker's
   two independent `ProcessingOutput` destinations). `src/preprocess.py`
   supports both: pass `--output-dir` for the common local case (it
   derives `<output-dir>/processed` and `<output-dir>/metadata`), or pass
   the two `--output-*-dir` flags directly for full control (as
   `scripts/run_processing.py` does when configuring the SageMaker job).
   If both are given, the explicit `--output-*-dir` flags win.
2. **Landmark word-boundary matching (Section 26, item 3).** The original
   design sketch in `implementation.md` used a plain substring search for
   landmark prefixes (`"near"`, `"opp"`, etc.), which would incorrectly
   match inside words like "Nearby" or "Shearer Road". This implementation
   uses a word-boundary-aware regex (`(?<![a-z0-9])(?:near|opp|...)(?![a-z0-9])`)
   instead — see `src/normalization.py::extract_landmark` and its tests.
3. **`&`/legal-suffix consistency (Section 26, item 1).** `canonicalize_name`
   expands `&` to `and` _before_ tokenizing and applying the legal-suffix
   map, so `"Smith & Jones Inc"` and `"Smith and Jones Incorporated"`
   canonicalize to the exact same string — verified by
   `test_ampersand_and_suffix_mapping_are_consistent`.
4. **Legal-suffix mapping scope.** `implementation.md`'s Section 15.1 code
   listing included `.`-suffixed duplicate keys (e.g. both `"ltd"` and
   `"ltd."`) in `LEGAL_SUFFIX_MAP`; this implementation instead strips
   trailing `.`/`,` from a token before the lookup (one canonical key per
   suffix), which is equivalent but avoids a duplicated, drift-prone map —
   see `_TOKEN_TRIM_RE` usage in `src/normalization.py`.
5. **Collision diagnostics scope (Section 10 of the task brief).** In
   addition to the `business_name_canonical`-only collision report from
   `implementation.md` Section 3/9, this implementation also reports
   `business_name_canonical + country_normalized` and
   `business_name_canonical + business_address_canonical +
country_normalized` collisions, all computed with bounded, capped
   memory usage (`MAX_COLLISION_GROUPS_REPORTED` / `MAX_SAMPLE_IDS_PER_GROUP`
   in `src/config.py`) — see `src/diagnostics.py`.
6. **SageMaker `framework_version`.** `implementation.md` Section 15.6
   hardcodes `framework_version="1.2-1"`. `scripts/run_processing.py`
   exposes this as a `--framework-version` CLI flag (still defaulting to
   `"1.2-1"`) with an explicit docstring warning to verify current
   supported versions before relying on it, per Section 26 item 7 — this
   implementation was not run against live SageMaker, so the default is
   not guaranteed current at the time you read this.

## What was verified locally vs. not verified

**LOCAL TESTS VERIFIED:**

- `python -m pytest` — all unit + integration tests pass.
- `python -m src.preprocess --input-dir ... --output-dir ...` run
  end-to-end against a synthetic sample dataset (including an ALL-CAPS
  source, embedded `NULL` address components, a franchise-style name
  collision, a word-embedded "near"/"opp" false-positive check, and an
  unseen `France` country value) — produced correct cleaned TSVs and a
  passing `preprocessing_report.json`.
- `scripts/verify_outputs.py` run against that same output — passed.
- `--help` verified to work (no crash, no missing dependency needed) for
  every script: `src/preprocess.py`, `scripts/upload_to_s3.py`,
  `scripts/run_processing.py`, `scripts/verify_outputs.py`,
  `scripts/create_sample.py`.
- All modules confirmed importable (`import src.config`,
  `src.normalization`, `src.validation`, `src.diagnostics`,
  `src.preprocess`).

**AWS EXECUTION NOT VERIFIED:** `scripts/upload_to_s3.py` and
`scripts/run_processing.py` were not run against a real AWS account as
part of producing this repository — no S3 upload or SageMaker Processing
job was actually executed. Their logic follows the documented boto3/
SageMaker SDK APIs, but you should run them against a real (ideally
non-production) AWS account and confirm the job reaches `Completed`
before trusting this pipeline at full scale (see
`AWS_SAGEMAKER_GUIDE.md`).
