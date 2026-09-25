# Data directory

No data is committed to this repository. This directory only holds a
pointer for where local runs expect data to live, and (optionally) small
sample files you generate yourself with `scripts/create_sample.py`.

## Expected layout

```
data/
├── full/                       # the 7 real files, full size (git-ignored)
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
└── sample/                     # a small subset, created via scripts/create_sample.py
    ├── train/  (same 4 files, fewer rows)
    └── test/   (same 3 files, fewer rows)
```

This `train/` + `test/` subdirectory layout is the same layout the
SageMaker Processing job mounts its `ProcessingInput` at
(`/opt/ml/processing/input/train/`, `/opt/ml/processing/input/test/`), so
`src/preprocess.py` runs identically against either.

## Where the real files go

1. Place the seven raw TSV files under `data/full/train/` and
   `data/full/test/` using the exact standard file names shown above.
2. Never rename, edit, or re-encode them before running the pipeline — the
   pipeline validates schema/format itself and is designed to read the
   files exactly as provided.
3. See the top-level `README.md` for how to create a small sample from
   these files, run the pipeline locally, and eventually push the raw
   files (untouched) to S3.
