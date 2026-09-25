# Business Entity Resolution
# Data Ingestion & SageMaker Preprocessing Implementation

> Scope: this document covers **only** raw ingestion → validation → S3 → SageMaker Processing → cleaning/normalization → cleaned datasets → S3. Embeddings, candidate generation, similarity scoring, and matching are explicitly out of scope and are covered in section 30 as a handoff.

All statistics below were computed directly from the seven attached files (`train_source1/2/3.tsv`, `train_ground_truth.tsv`, `test_source1/2/3.tsv`) using `pd.read_csv(path, sep="\t")`. Where a statistic could not be computed from the data, it is marked as such rather than guessed.

---

## 1. Objective

Produce a reproducible, auditable, SageMaker-based pipeline that turns the seven raw TSV files into cleaned, normalized datasets stored in S3, without performing entity matching. The output of this stage is the **input** to a later ER modeling stage (embeddings, candidate generation, scoring) that is not implemented here.

---

## 2. Dataset Analysis

All files use the schema `entity_id, business_name, business_address, country`, except the ground truth file, which uses `source1_entity_id, matched_entity_ids`. No file has extra or missing columns. No file has duplicate `entity_id` values. All numbers below are exact counts from the actual files (row counts exclude the header row).

### 2.1 Training Source 1 (`train_source1.tsv` — the deduplicated reference source)

| Metric | Value |
|---|---|
| Rows | 2,206,821 |
| Columns | `entity_id, business_name, business_address, country` (4) |
| Duplicate `entity_id` | 0 |
| Unique `entity_id` | 2,206,821 (= row count) |
| Empty `business_name` | 0 |
| Empty `business_address` | 0 |
| Empty `country` | 0 |
| `business_name` length (mean / min / max) | 24.0 / 3 / 105 |
| `business_address` length (mean / min / max) | 52.1 / 11 / 256 |
| Country values | `US`: 1,323,633 · `India`: 883,188 (2 distinct values, no other spellings) |
| Leading/trailing whitespace in name/address | 0 |
| Double internal spaces in name | 0 |
| Non-ASCII characters in name | 0 |
| Non-ASCII characters in address | 554 |
| ALL-CAPS rows | 4 |
| Literal `null`/`NULL` token embedded in address | 43 |

Source 1 is materially cleaner than Source 2/3: no case noise, no whitespace noise, essentially no non-ASCII names. This matches the challenge statement that Source 1 is the deduplicated reference source.

### 2.2 Training Source 2 (`train_source2.tsv`)

| Metric | Value |
|---|---|
| Rows | 5,034,616 |
| Duplicate `entity_id` | 0 |
| Empty `business_name` | 0 (but 6 rows contain an `NA`/`N/A`/`NULL`-like literal string as the whole name) |
| Empty `business_address` | 168,967 (3.4% of rows) |
| `business_name` length (mean / min / max) | 25.1 / 2 / 104 |
| `business_address` length (mean / min / max) | 46.2 / 0 / 249 |
| Country values | `US`: 3,016,817 · `India`: 2,017,799 (2 distinct) |
| ALL-CAPS rows | 3,191,104 (63% of rows) |
| Double internal spaces in name | 554,392 |
| Non-ASCII characters in name | 764,608 (15%) — mostly Devanagari, Tamil, Telugu, Gujarati, Punjabi, Malayalam script |
| Non-ASCII characters in address | 478,453 |
| Names starting with a non-alphanumeric character (`--`, `[`, native script, etc.) | 577,243 |
| Literal `null`/`NULL` token embedded in address | 131,794 |

### 2.3 Training Source 3 (`train_source3.tsv`)

| Metric | Value |
|---|---|
| Rows | 5,285,603 |
| Duplicate `entity_id` | 0 |
| Empty `business_address` | 175,916 (3.3%) |
| `business_name` length (mean / min / max) | 25.2 / 2 / 123 |
| Country values | `US`: 3,170,056 · `India`: 2,115,547 |
| ALL-CAPS rows | 813 (Source 3 is mixed-case, unlike Source 2) |
| Double internal spaces in name | 575,616 |
| Non-ASCII characters in name | 606,737 |
| Literal `null`/`NULL` token embedded in address | 130,844 |

### 2.4 Ground Truth (`train_ground_truth.tsv`)

| Metric | Value |
|---|---|
| Rows | 2,206,821 |
| Unique `source1_entity_id` | 2,206,821 (matches Source 1 row count exactly) |
| Duplicate `source1_entity_id` | 0 |
| Every training Source 1 entity appears in ground truth | Yes — set difference in both directions is empty |
| Rows with zero matches (`matched_entity_ids` empty) | 123,247 (5.6%) |
| Rows with exactly 1 match | 119,157 |
| Rows with 2 matches | 375,212 |
| Rows with 3 matches | 530,841 |
| Rows with 4 matches | 484,115 |
| Rows with 5 matches | 321,957 |
| Rows with 6–11 matches | 253,292 combined |
| Max matches for a single Source 1 entity | 11 |
| Mean matches per entity | 3.46 |
| Total S2 references / total S3 references | 3,693,619 / 3,944,746 |
| Distinct S2 ids referenced / total S2 rows | 3,693,619 / 5,034,616 (73%) |
| Distinct S3 ids referenced / total S3 rows | 3,944,746 / 5,285,603 (75%) |
| Invalid references (an id in `matched_entity_ids` that doesn't exist in Source 2/3, or doesn't start with `S2-`/`S3-`) | 0 |
| Duplicate ids inside a single `matched_entity_ids` cell | 0 |
| Whitespace/formatting anomalies in `matched_entity_ids` | 0 |

Ground truth is internally consistent and fully valid against the source files: every reference resolves, there are no orphan or malformed entries. About a quarter of Source 2/3 records are never referenced by any Source 1 entity in the training ground truth — this is expected (not every S2/S3 record has a corresponding S1 entity in-sample) and is not a data quality defect.

### 2.5–2.7 Test Sources 1/2/3

| Metric | Test S1 | Test S2 | Test S3 |
|---|---|---|---|
| Rows | 1,732,544 | 4,887,273 | 5,082,316 |
| Duplicate `entity_id` | 0 | 0 | 0 |
| Empty `business_address` | 0 | 129,408 | 136,098 |
| Country distinct values | 3 | 3 | 3 |
| Country distribution | India 809,986 / US 663,106 / **France 259,452** | India 2,312,565 / US 1,871,330 / **France 703,378** | India 2,405,000 / US 1,945,701 / **France 731,615** |
| ALL-CAPS rows | 0 | high (consistent with train S2) | low (consistent with train S3) |
| Non-ASCII names | 40,789 (French diacritics, e.g. "École") | 928,158 | 737,515 |

**Confirmed from the data, not assumed:** test data introduces `France` as a third country value that never appears in training (`train_source1/2/3.tsv` contain only `US` and `India`). No other new country spellings were observed in test. This directly confirms the requirement in the brief that country handling must generalize to unseen values — a hard-coded `{US, India}` universe would silently mishandle roughly 15–17% of test rows.

There is no ground truth file for test data (not provided, and correctly so per the brief — it must not be used).

---

## 3. Observed Data Quality Issues

These are the noise patterns actually present in the data (verified by direct pattern search, not assumed):

**Business names**
- Case noise: Source 2 is ~63% ALL-CAPS; Source 1 and Source 3 are mixed-case. Example: `RAM MARKETING PVT LTD` type patterns are absent from S1 but common in S2.
- Legal-suffix variety: `Pvt`/`Private`, `Ltd`/`Limited`, `Corp`/`Corporation`, `Inc`/`Incorporated`, `LLC`, `LLP`, `Co`/`Company`, and French forms `SARL`, `SAS`, `SCI`, `S.A.S`, `S.A.` all occur (French forms only ~400–430 occurrences per file, low volume but real).
- `&` vs `and`: both forms occur (e.g. `Foot & Ankle Allied Center LLC` vs `... and ...` elsewhere).
- Leading noise characters: dashes, brackets, symbols, or a native-script character precede the name in up to 11% of Source 2 rows (e.g. `-- Holloway Peak Inc Seafood`, `[INCORPORATED] PEAK TRADIN6 NETWORKS SOUTHSIDE`).
- **Transliteration to native scripts** (Devanagari, Tamil, Telugu, Punjabi, Gujarati, Malayalam) for the *same* Indian business across sources — e.g. one training pair matches `Raj Investments LLP` (S1, Latin) to `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (S2/S3, Tamil script) for the identical entity. This is **not fixable by lexical normalization** — it requires multilingual embeddings in the next pipeline stage. This document normalizes Unicode form and case but does not attempt script transliteration/translation.
- Character-level typos consistent with OCR/keying noise: `Wilblims`→Williams, `Ponr`→Power, `Sgpeciarity`→Specialty, diacritic insertion (`Énterprises`, `Bóral`, `Dréxkor`).
- Domain-style names appear as legitimate business-name values (e.g. `maurewilliamscolombier.com`, `wilfordhancock.com`) — these must be preserved, not stripped, since they are the actual field value for that record.
- Duplicate internal whitespace in ~10–11% of Source 2/3 names.

**Addresses**
- Reordered components: the same business's address has city/state/street in different order across sources (`"IA, Iowa City, 1064 Newton Rd, Unit 11"` vs `"1064 Newton Rd, Iowa City, IA"`-style variants).
- State given as full name vs. 2-letter abbreviation (`Missouri` vs `MO`, `Tamil Nadu` vs `TN`, `Karnataka` in native script `ಕರ್ನಾಟಕ`).
- Empty address field: 3.3–3.4% of Source 2/3 rows (never in Source 1).
- Literal `null`/`NULL`/`<NULL>` token embedded **as one comma-separated component inside an otherwise populated address string** (43 in S1, ~130–132k each in S2/S3) — e.g. `"AF-0684, Uttar Pradesh, GHAZIABAD, 9487203"` vs. the paired record `"AF-0684, NANDGRAM ..., GHAZIABAD, 9487203, उत्तर प्रदेश"`. This is a component-level missing marker, not a whole-field missing marker, and must be handled as such.
- `near` / `opp`(osite) landmark phrases exist in 1–2% of addresses and carry real locality information (e.g. `"Near Fortis Hospital"`), so they must not be blindly deleted.
- Road/street abbreviations exist in both directions (`Rd`/`Road`, `St`/`Street`, `Ave`/`Avenue`) — both abbreviated and spelled-out forms are common enough that a mapping should normalize to a canonical form, not delete either.
- Building/unit numbers, PIN/postal-style tokens, and floor numbers are present and must be preserved as literal information (they are highly discriminative, per the paired examples above).

**Countries**
- Training: exactly `{US, India}`. Test: `{US, India, France}`. No misspellings or case variants of country values were found in any file — the field is already clean at the value level, but the *set* of values is open (production data may introduce more).

**Canonicalization risk (measured directly):** normalizing Source 1 business names alone (Unicode NFKC, lowercase, strip punctuation, collapse whitespace) collapses 2,206,821 unique rows into 1,515,975 unique normalized strings — **181,975 collision groups covering 872,821 rows (≈40%)**, e.g. `"Primary Care Group"` appears as 253 distinct Source-1 entities at different addresses (franchise-style naming). This is measured evidence, not a hypothesis, that **name-only canonicalization must never be used to deduplicate or resolve entities** — it is a diagnostic signal for the next stage, not a cleaning action. See Sections 9 and 26.

---

## 4. Preprocessing Design

Given the findings above, the pipeline follows this order, which differs slightly from the generic template in the brief because the data shows *why* the order matters:

```
Local 7 TSVs
   → Local schema/structural validation (fail fast, before any S3 cost)
   → Upload raw, untouched, to S3 raw/
   → SageMaker Processing Job
        → Read TSV (sep="\t", explicit dtype=str, keep_default_na=False)
        → Schema validation (columns, types, entity_id prefix/pattern)
        → Ground-truth referential validation (train only)
        → Missing-value detection & explicit-null encoding (component + field level)
        → Unicode normalization (NFKC)
        → Case normalization → business_name_normalized / business_address_normalized
        → Whitespace + conservative punctuation normalization
        → Business-name canonical suffix mapping → business_name_canonical
        → Address canonical abbreviation mapping → business_address_canonical
        → Country normalization (open-set, no hardcoded universe) → country_normalized
        → Canonicalization collision diagnostics (report only, no deletion)
        → Duplicate-entity_id / exact-duplicate-row diagnostics (report only, no deletion)
        → Row/ID-count invariant checks (fail job on violation)
        → Write cleaned TSVs
        → Write preprocessing_report.json / .txt
   → S3 processed/
```

Why this order:
- Local validation happens **before** any upload, so malformed input is caught for $0 of cloud cost.
- Missing-value handling happens **before** normalization, because a literal `null` token embedded mid-string (Section 3) must be recognized before lowercasing/whitespace collapsing could otherwise fuse it unrecognizably into neighboring tokens.
- Canonicalization diagnostics run **after** normalization but produce a report, never a delete/dedupe action — Section 3 proved that name-based collisions are frequent and legitimate.
- Ground-truth referential validation only applies to the training split (Section 11) and never touches test data, since test ground truth does not exist (Section 27).

---

## 5. Raw Data Architecture

Raw files are written once to `raw/` and never modified in place. Reasons, grounded in this dataset:
- The pipeline will be iterated on (normalization rules will change as the next ER stage reveals what actually helps matching). If raw data were overwritten, any bug in a normalization rule would be **unrecoverable** without re-uploading from the original TSVs.
- The `entity_id` values are the only join key across all downstream stages (ground truth, matching, evaluation) — raw immutability guarantees they can always be reconstructed and audited against the original source.
- Reproducibility (Section 24) requires being able to re-run any past pipeline version against the exact same raw bytes.

---

## 6. S3 Architecture

```
s3://<bucket>/entity-resolution/
├── raw/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
│
├── processed/
│   ├── train/
│   │   ├── source1_clean.tsv
│   │   ├── source2_clean.tsv
│   │   ├── source3_clean.tsv
│   │   └── ground_truth_clean.tsv        (validated copy — see Section 11)
│   └── test/
│       ├── source1_clean.tsv
│       ├── source2_clean.tsv
│       └── source3_clean.tsv
│
└── metadata/
    ├── preprocessing_report.json
    ├── preprocessing_report.txt
    ├── canonicalization_collisions_train.tsv
    ├── canonicalization_collisions_test.tsv
    └── run-<timestamp>-<job-name>/         (per-run copy of the above, for history)
```

- **Bucket naming:** a dedicated bucket per environment, e.g. `<org>-entity-resolution-<env>` (`dev`/`prod`), lowercase, no dots (avoids virtual-hosted-style TLS issues).
- **Prefixes:** a single top-level `entity-resolution/` prefix keeps this project isolated if the bucket is shared.
- **Encryption:** SSE-S3 (`AES256`) or SSE-KMS at rest; enforce via bucket policy (`aws:SecureTransport` = true for TLS in transit, and deny `PutObject` without `x-amz-server-side-encryption`).
- **Public access:** enable S3 Block Public Access at the bucket level; no ACLs.
- **IAM:** a dedicated SageMaker execution role scoped to `raw/*` (read), `processed/*` (read/write), `metadata/*` (read/write) under this prefix only — not bucket-wide.
- **Versioning:** enable bucket versioning. It's cheap protection against an accidental overwrite of `processed/` on a bad re-run, and pairs naturally with the "raw is immutable" principle.
- **Lifecycle:** not necessary yet given the size (≈2.5 GB total across all seven files); revisit if raw/processed history accumulates many run copies — a lifecycle rule can transition old `metadata/run-*/` prefixes to S3 Infrequent Access after, e.g., 90 days.

---

## 7. Repository Structure

```
business-entity-resolution/
├── data/
│   └── README.md                  # points to S3 raw/ locations; no data committed to git
├── src/
│   ├── preprocess.py               # SageMaker Processing entrypoint
│   ├── normalization.py            # name/address/country normalization functions
│   ├── validation.py               # schema, ID, and ground-truth validation
│   └── config.py                   # column names, paths, thresholds
├── scripts/
│   ├── upload_to_s3.py
│   └── run_processing.py
├── tests/
│   ├── test_normalization.py
│   └── test_validation.py
├── reports/                        # local copies of preprocessing_report.* for inspection
├── requirements.txt
├── implementation.md
└── README.md
```

---

## 8. Normalization Strategy

### 8.1 Unicode
Apply `unicodedata.normalize("NFKC", text)`. NFKC (not NFC) is chosen because it also folds compatibility variants (full-width characters, some ligatures) that appear in noisy multi-source text, while NFC would leave them distinct. NFKC does **not** transliterate scripts — Devanagari/Tamil/etc. text stays in its own script; this stage does not attempt script-level matching (Section 3).

### 8.2 Case
Lowercase both name and address in `_normalized`/`_canonical` fields. Rationale: Source 2 is ~63% ALL-CAPS while Source 1/3 are mixed-case (Section 2), so case is not a signal that survives across sources — lowercasing removes this cross-source inconsistency without losing information (the raw `business_name`/`business_address` columns retain the original case for audit).

### 8.3 Whitespace
Strip leading/trailing whitespace and collapse repeated internal whitespace to a single space. This directly addresses the measured 500k+ rows per Source 2/3 file with doubled internal spaces (Section 2).

### 8.4 Punctuation
Conservative only: normalize curly/smart quotes to straight quotes, normalize multiple periods, and strip a small set of pure decorative punctuation (`#`, leading `--`, `<<`, `[`, `]` when they appear as decoration around an otherwise valid name — verified against the "leading special character" pattern found in up to 577,243 Source 2 rows). Never strip digits, `/`, `-` within alphanumeric tokens (building numbers, unit numbers), or `.` inside abbreviations, since those are proven discriminative in the paired examples in Section 22.

### 8.5 Business Names
`business_name_normalized` = Unicode NFKC + lowercase + whitespace/punctuation normalization only (no suffix rewriting — conservative, reversible).
`business_name_canonical` = `_normalized` plus a validated legal-suffix mapping (Section 9) applied only where confidently identifiable at a token boundary. `&` is normalized to `and` in the canonical field only (both forms were observed in the data at meaningful volume).

### 8.6 Addresses
`business_address_normalized` = Unicode NFKC + lowercase + whitespace normalization + explicit-null-token handling (Section 8.8) — components equal to `null`/`NULL`/`<NULL>` (case-insensitive, optionally bracketed) are removed from the comma-joined string rather than lowercased into a literal word "null" that could be confused with real address text.
`business_address_canonical` = `_normalized` plus the road/street abbreviation mapping (Section 9). Landmark phrases (`near`, `opp`, `opposite`, `behind`) are **not** deleted; they are also copied into a new `address_landmark` field (empty if none found) so the next stage can use or ignore them as a separate feature, per the requirement in the canonicalization-rules attachment.

### 8.7 Countries
`country_normalized` = Unicode NFKC + lowercase + whitespace strip only. No alias table is built from training values, because the training set contains only `{us, india}` — building a training-only alias map (e.g. mapping `usa`→`us`) would be an assumption invented from data that doesn't demonstrate the need, and would risk silently mishandling `france` or any future value the same way a hardcoded universe would. The function is a pure string-normalization, open-set transform with no lookup table, satisfying the requirement to support unseen countries by construction rather than by enumeration.

### 8.8 Missing Values
Policy, based on what was actually observed:
- Whole-field empty string (`""`) occurs only in `business_address` (never in `business_name`, `country`, or any Source 1 field) — kept as an explicit empty string in `_normalized`/`_canonical` (not silently dropped, not fabricated), with a companion boolean column `business_address_is_missing`.
- Literal NA-like full-field strings (`NA`, `N/A`, `NULL`, `None`) appear only in `business_name` in Source 2/3, at very low volume (6–61 rows per file) — mapped to an explicit empty string + `business_name_is_missing = true`, since treating them as a literal business name would be wrong, but the row itself is kept (per the brief: never drop a record for a missing field).
- Component-level `null` tokens embedded inside an otherwise-populated address (43 rows in S1, ~130k per S2/S3 file) are stripped as a component during address normalization (Section 8.6), not treated as a reason to blank the whole field.
- No record is ever dropped for missing name/address/country — every row that entered the job is present in the output (enforced as a fatal-error invariant, Section 20).

---

## 9. Duplicate Handling

`Cleaning ≠ entity deduplication.` This is not a stylistic warning — Section 3 measured it directly: normalizing Source 1 business names alone collapses 40% of rows into shared canonical strings that are legitimately different businesses (e.g., 253 distinct `"Primary Care Group"` entities at different U.S. addresses). Therefore:
- **Exact duplicate rows** (all 4 columns byte-identical) are counted and reported, not removed by default, since none were found to exist in the training/test files during analysis (this pipeline still checks for them defensively in production, in case future batches contain them).
- **Duplicate `entity_id` values**: none exist in any of the seven files (verified, Section 2) — treated as a **fatal validation error** if ever encountered, since `entity_id` is the only cross-stage join key.
- **Same business, different entity_id** (multiple Source 2/3 records for one real-world business) is expected and must survive cleaning untouched — it is exactly the signal the next ER stage needs.
- The canonicalization collision report (Section 3, Section 26) is a diagnostic artifact (`canonicalization_collisions_*.tsv`), never an automatic delete/merge action.

---

## 10. Schema Validation

For each source file: exactly the columns `entity_id, business_name, business_address, country`, in `str` dtype (loaded with `dtype=str, keep_default_na=False` to avoid pandas silently coercing e.g. a numeric-looking name or turning a literal `"NA"` string into a real `NaN`). `entity_id` must match `^S[123]-\d+$` for the corresponding source; any mismatch is a fatal error (none were found in the actual data, but production batches could differ). For the ground truth file: exactly `source1_entity_id, matched_entity_ids`; every `matched_entity_ids` token must match `^S[23]-\d+$` when non-empty.

---

## 11. Ground Truth Handling

`train_ground_truth.tsv` is training metadata, not a source record dataset, and is processed and stored separately (`processed/train/ground_truth_clean.tsv`, never merged into `source*_clean.tsv`). Its "cleaning" is limited to: referential validation against the cleaned Source 1/2/3 `entity_id` sets (already confirmed 0 invalid references, but re-checked every run since inputs can change), duplicate-key checks, and whitespace-in-list checks (Section 2.4). It is not touched by any text normalization, since its two columns are IDs, not free text. It will later support validation/evaluation in the next ER stage; that stage is not implemented here. There is no test ground truth file, and none is fabricated — see Section 27.

---

## 12. SageMaker Processing Architecture

```
S3 raw/{train,test}/*.tsv
        │
        ▼
SageMaker Processing Job (src/preprocess.py as entrypoint)
   ProcessingInput:  /opt/ml/processing/input/{train,test}/
   ProcessingOutput: /opt/ml/processing/output/{processed,metadata}/
        │
        ▼
S3 processed/{train,test}/*.tsv, metadata/preprocessing_report.*
```

One processing job handles both train and test in a single run (they share identical code paths for normalization; only ground-truth validation is train-specific), parameterized by `--split train test` so either can also be run independently for iteration.

---

## 13. SageMaker Processor Choice

Recommendation: **`SKLearnProcessor`** (a stock scikit-learn container running our own `preprocess.py`, using pandas — SKLearn is not being used for modeling here, just as a convenient pre-built Python data-science container), not `PySparkProcessor`.

Reasoning, grounded in the actual data size: total input across all seven files is **≈2.5 GB** (167 MB–509 MB per file, largest file 5.29M rows). This comfortably fits pandas string processing on a single mid-size instance without chunking (a 500 MB TSV of 4 short string columns loads to roughly 1.5–2.5 GB in memory as `dtype=str`, well within a single `ml.m5.4xlarge`'s 64 GB RAM). Spark's value is distributing work across many nodes and handling data that doesn't fit on one machine — neither applies at this size, and introducing Spark here would add cluster startup latency, PySpark/JVM operational complexity, and a second runtime to test and debug, for no throughput benefit. `ScriptProcessor` (a fully custom Docker container) is unnecessary too, since no dependency here (pandas, regex, unicodedata — all standard or pip-installable) requires a container SKLearnProcessor's image can't satisfy via a `requirements.txt` passed at job creation.

If input volume grows by roughly an order of magnitude (tens of GB, tens of millions of rows per file) or if per-row processing becomes CPU-heavy (e.g. real transliteration/embedding work), re-evaluate `PySparkProcessor` at that point — this architecture doesn't preclude swapping the processor later since normalization logic lives in plain-Python `normalization.py`, independent of the execution engine.

---

## 14. Instance Sizing

Given ≈2.5 GB total input and largest single file ≈509 MB / 5.29M rows:

- **Recommended:** `ml.m5.4xlarge` (16 vCPU, 64 GB RAM) for the processing job. This is a modest, general-purpose instance — not an expensive choice — chosen because pandas string operations (regex substitution, Unicode normalization) over ~5M rows benefit from RAM headroom (avoid the ~137 GB Killed error observed locally in this analysis's own 3.9 GB / 1-vCPU sandbox) more than from many cores, since a first implementation processes one file at a time sequentially.
- **Disk:** default EBS attached to the processing instance is sufficient (input + output + intermediate copies well under 20 GB); no need to request extra `volume_size_in_gb` beyond the SageMaker default unless local testing shows otherwise.
- **Network:** irrelevant at this size — S3 transfer of 2.5 GB completes in well under a minute on any SageMaker instance's default network throughput.
- **Expected processing time:** not measured on real SageMaker hardware in this exercise (would need to be benchmarked, per Section 21's local-first workflow, before committing to a production SLA) — but a single-instance pandas job over ~2.5 GB of short-string TSVs should complete in low single-digit minutes, not hours.
- **When to scale up:** if a benchmarked run on `ml.m5.4xlarge` exceeds the pipeline's acceptable runtime, move to `ml.m5.8xlarge`/`ml.m5.12xlarge` (more RAM/cores, same architecture) before considering a Spark rewrite — vertical scaling is simpler and matches this workload's actual bottleneck (single-machine string processing, not distributed shuffle).

---

## 15. Implementation

(See `src/` in this repository for the actual, current implementation — the
original code sketch that lived in this section of the design document has
been superseded by the real, tested source files. Notable differences from
the original sketch, and why, are documented in `README.md` under
"Design decisions / resolved inconsistencies".)

---

## 16. Local Testing

Run the exact same `src/preprocess.py` against a local copy of the 7 files before ever touching S3 or SageMaker, so the logic is identical in both environments (SageMaker just calls the same entrypoint inside a container). See `README.md` sections 6–11 for the exact current commands (`python -m pytest`, `scripts/create_sample.py`, `python -m src.preprocess ...`).

---

## 17. SageMaker Execution

See `AWS_SAGEMAKER_GUIDE.md` for the full, current, step-by-step walkthrough (bucket creation, IAM role, `scripts/upload_to_s3.py`, `scripts/run_processing.py`, monitoring, and output verification).

---

## 18. Output Structure

Each `*_clean.tsv` contains:

```
entity_id
business_name                 (original, untouched)
business_name_normalized      (NFKC + lowercase + whitespace + decorative-punctuation strip)
business_name_canonical       (+ legal-suffix mapping, & -> and)
business_address              (original, untouched)
business_address_normalized   (NFKC + lowercase + whitespace + null-component strip)
business_address_canonical    (+ road/street abbreviation mapping)
address_landmark              (copied-out landmark phrase, or empty)
country                       (original, untouched)
country_normalized            (NFKC + lowercase + whitespace strip, no alias table)
business_name_is_missing      (bool)
business_address_is_missing   (bool)
```

`ground_truth_clean.tsv` (train only) keeps `source1_entity_id, matched_entity_ids` unchanged (validated, not transformed).

---

## 19. Data Quality Report

`metadata/preprocessing_report.json` includes, per split and per source: `input_rows`, `output_rows`, `name_missing_count`, `address_missing_count`, `country_distribution`, and canonicalization collision statistics; plus, for train, ground-truth referential `issues`/stats. `preprocessing_report.txt` is the same content in a directly readable form. Checks are classified as:
- **Fatal errors** (job exits non-zero, no output written): schema mismatch, malformed `entity_id`, duplicate `entity_id`, row-count/ID-set change during cleaning, invalid ground-truth reference.
- **Warnings** (recorded in the report, job continues): high missing-value counts, large collision-group counts, low country diversity change between train/test.

---

## 20. Validation

Enforced as fatal-error invariants at the end of every run: `input_rows == output_rows` for every one of the 7 files; every `entity_id` present in input is present in output (checked by set equality, not just count, so a swap couldn't slip through); all ground-truth references still resolve against the cleaned Source 2/3 id sets. These were all confirmed to hold trivially on the current data (Section 2) — the checks exist to catch regressions as the code evolves, not because today's data is expected to fail them.

---

## 21. Unit Tests

See `tests/` in this repository (`test_normalization.py`, `test_validation.py`, `test_diagnostics.py`, `test_preprocess_integration.py`) for the actual, current, executable test suite — run via `python -m pytest`.

---

## 22. Before/After Examples

All examples below are **real rows** from the attached files (not fabricated), selected because they illustrate a specific, actually-occurring transformation.

**Case + whitespace (train_source2.tsv):**
```
BEFORE  business_name:            "BABA-PVT. LTD.  CENTER"
AFTER   business_name_normalized: "baba-pvt. ltd. center"
```

**Leading decoration (train_source2.tsv):**
```
BEFORE  business_name:            "-- Holloway Peak Inc Seafood"
AFTER   business_name_normalized: "holloway peak inc seafood"
```

**Legal suffix (train_source1.tsv):**
```
BEFORE  business_name:            "Invest Investments Pvt Ltd"
AFTER   business_name_normalized: "invest investments pvt ltd"
AFTER   business_name_canonical:  "invest investments private limited"
```

**Ampersand (train_source1.tsv):**
```
BEFORE  business_name:            "Callicoat & Dailey Inc"
AFTER   business_name_canonical:  "callicoat and dailey incorporated"
```

**Road abbreviation (matched pair, train S1 <-> S2, ground-truth-linked):**
```
BEFORE  business_address (S1):    "3315 Fremont Street, Peoria, IL"
BEFORE  business_address (S2):    "3315 FREMONT ST, PEORIA, IL"
AFTER   business_address_canonical (both): "3315 fremont street, peoria, il"
```

**Embedded null component (train_source2.tsv):**
```
BEFORE  business_address:            "067 PRODUCTION CT, NULL, INDEPENDENCE, KY"
AFTER   business_address_normalized: "067 production ct, independence, ky"
```

**Missing field (train_source3.tsv):**
```
BEFORE  business_address: ""
AFTER   business_address_normalized: ""   business_address_is_missing: true
```

**Landmark preserved, not deleted (train_source2.tsv):**
```
BEFORE  business_address: "H.No.16-11-23/37/A, ... Opp.Rta Office, Mo, Osarambagh, Hyderabad, Telangana"
AFTER   business_address_normalized: "... opp.rta office, mo, osarambagh, hyderabad, telangana"
AFTER   address_landmark: "opp.rta office"
```

**Country, including an unseen value (test_source1.tsv):**
```
BEFORE  country: "France"    (never appears in ANY training file)
AFTER   country_normalized: "france"   -- handled correctly with no alias table, no hardcoding
```

**What normalization deliberately does NOT fix — transliteration (train ground-truth-linked pair):**
```
S1 (reference): business_name = "Raj Investments LLP"
S2 (match):     business_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"   (Tamil script, same entity)
```
This pair is confirmed by ground truth to be the same real-world business. No lexical transformation in this stage can equate a Latin-script name with its Tamil transliteration — this is explicitly deferred to the embeddings/matching stage (Section 30).

---

## 23. Leakage Prevention

The preprocessing pipeline never reads or references `train_ground_truth.tsv` while cleaning `test_source1/2/3.tsv` — the `clean_source()` function operates identically and independently on every source file regardless of split, and ground-truth handling (Section 11) only executes when `split == "train"`. No normalization mapping (Section 8) is *learned* from training data at all — every mapping (`LEGAL_SUFFIX_MAP`, `ADDRESS_ABBREV_MAP`, `LANDMARK_PREFIXES`) is a fixed, hand-validated dictionary applied identically to train and test, so there is no statistic "fit" on training labels that could leak into test cleaning. This is safe specifically because no test ground truth exists to leak from, and because country normalization (Section 8.7) was deliberately kept alias-table-free rather than derived from the training country set — a training-derived alias table applied to test would have been a (mild) form of the exact leakage this section warns against.

---

## 24. Reproducibility

- Pin dependencies in `requirements.txt` (pandas, boto3, sagemaker SDK, pytest) with exact/bounded versions.
- All normalization functions are pure and deterministic (no randomness, no external calls, no wall-clock-dependent logic) — same input always produces the same output.
- `src/` is versioned in git; each SageMaker job can be launched from a specific commit, and `run_processing.py`'s `--job-name` should embed a version/commit identifier (e.g. `entity-resolution-preprocess-<git-sha>-<timestamp>`).
- S3 bucket versioning (Section 6) protects `processed/` and `metadata/` history even without a dedicated archival step.
- Every run writes its own `preprocessing_report.json`, which records exact input/output row counts and the git-taggable job name, so any run's provenance is reconstructable later.
- Input/output S3 paths are passed as explicit CLI arguments (never hardcoded), so the same code runs against `dev`/`prod` buckets without modification.

---

## 25. Security

- SageMaker execution role scoped to only `raw/*` (read), `processed/*` + `metadata/*` (read/write) under this project's prefix — least privilege, not bucket-wide `s3:*`.
- No AWS access keys anywhere in code or config — the Processing job assumes its IAM role automatically; local `upload_to_s3.py`/`run_processing.py` rely on the caller's configured AWS CLI/SSO credentials.
- Bucket: S3 Block Public Access enabled, no public ACLs or bucket policies.
- Encryption at rest: SSE-S3 or SSE-KMS on every object (Section 6); enforce via bucket policy denying unencrypted `PutObject`.
- TLS in transit: enforce via bucket policy denying non-`aws:SecureTransport` requests.
- Logs: SageMaker Processing job logs go to CloudWatch Logs automatically; no secrets should ever be printed to stdout/stderr (the pipeline logs row counts and validation issues, never raw PII beyond what's already in the source business records).
- Secrets handling: none of this pipeline's steps require API keys or credentials beyond the IAM role — if a future step needs one (e.g. a translation API), use AWS Secrets Manager, never a hardcoded value.

---

## 26. Cost Considerations

- **SageMaker Processing charges** by instance-second while the job runs; an `ml.m5.4xlarge` job processing ≈2.5 GB should complete in low single-digit minutes (Section 14), making per-run compute cost small, but exact current on-demand pricing for `ml.m5.4xlarge` **must be checked against the live AWS SageMaker pricing page** before committing to a budget — pricing figures are not reproduced here since they change and this document should not risk being stale.
- **S3 storage**: raw (≈2.5 GB) + processed (similar order of magnitude, slightly larger due to added normalized/canonical columns) + metadata (small) — total storage cost is minor at this data volume; grows linearly if many historical `metadata/run-*/` copies are retained (Section 6's lifecycle note).
- **S3 requests**: seven `PutObject` calls for raw upload, a handful more for processed/metadata output — negligible request-count cost at this scale.
- **CloudWatch Logs**: default retention charges are minor for a few-minute job's logs; consider a log retention policy (e.g. 30–90 days) on the Processing job's log group to avoid unbounded accumulation across many runs.

---

## 27. Troubleshooting

See `README.md` section 15 for the current troubleshooting table (kept in one place to avoid this document and the README drifting out of sync).

---

## 28. End-to-End Runbook

See `README.md` (sections 6–14) and `AWS_SAGEMAKER_GUIDE.md` for the current, exact end-to-end runbook (local validation → sample run → full local run → invariant verification → S3 upload → SageMaker Processing → output inspection → next-stage handoff).

---

## 29. Completion Checklist

- [x] All 7 TSV files inspected directly (Section 2)
- [x] Actual dataset statistics used throughout, none fabricated (Sections 2–3; note in Section 14 that on-hardware timing is explicitly *not* measured and flagged as such)
- [x] TSV parsing uses `sep="\t"` everywhere
- [x] Raw data preserved, never overwritten (`raw/` immutable, Section 5, upload script defaults to skip-if-exists)
- [x] Entity IDs preserved and validated (Sections 10, 20)
- [x] Source 1/2/3 semantics preserved (separate files throughout, no merging)
- [x] Ground truth kept separate from source records (Section 11)
- [x] France / unseen countries supported by construction (Section 8.7, tested in Section 21)
- [x] No entity records silently deduplicated (Section 9, 26 diagnostics-only)
- [x] Original + normalized + canonical fields all preserved (Section 8, 18)
- [x] Normalization is deterministic (Section 24)
- [x] SageMaker Processing implementable as specified (`SKLearnProcessor`, Section 13)
- [x] S3 input/output paths defined (Section 6)
- [x] Executable Python code provided, not pseudocode (see `src/`)
- [x] Local testing workflow provided (README section 6–11)
- [x] SageMaker execution workflow provided (`AWS_SAGEMAKER_GUIDE.md`)
- [x] Validation/reporting provided (Sections 19–20)
- [x] Security covered (Section 25)
- [x] Cost considerations covered, with live-pricing caveat (Section 26)
- [x] Reproducibility covered (Section 24)
- [x] Document stops at cleaned dataset → S3 (no modeling code present)
- [x] Future embedding/matching stages explicitly deferred (Section 30 below)

---

## 30. Handoff to the Next ER Stage

```
CURRENT STAGE (this document / this repository)
──────────────────────────────
Raw 7 TSVs → S3 raw/ → SageMaker Processing → cleaned + normalized + canonical
fields → S3 processed/ + metadata/preprocessing_report.*


FUTURE STAGE (not implemented here)
──────────────────────────────
S3 processed/ → feature engineering → multilingual embeddings (needed
specifically because of the transliteration cases in Section 22, e.g.
"Raj Investments LLP" vs its Tamil-script equivalent) → candidate
generation (per Source-1 entity, over Source 2/3) → similarity scoring
→ entity matching against the ground-truth match-count distribution in
Section 2.4 (mean 3.46 matches/entity, up to 11) → evaluation → final
submission
```

The next stage should consume `processed/{train,test}/source{1,2,3}_clean.tsv` and `processed/train/ground_truth_clean.tsv` — never the `raw/` files directly, and never any test ground truth (none exists, Section 23/27).
