"""Shared configuration for the preprocessing pipeline.

All mappings here were validated against the seven attached dataset files
before inclusion (see implementation.md, Sections 2-3, 8-9). Nothing here
is applied "because it's a common ER trick" without evidence — and nothing
here performs entity matching or deduplication.
"""
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

SOURCE_ID_PATTERNS = {
    "source1": r"^S1-\d+$",
    "source2": r"^S2-\d+$",
    "source3": r"^S3-\d+$",
}

# Which source key is expected for each canonical source file name.
SOURCE_KEYS = ("source1", "source2", "source3")

GT_SOURCE1_PATTERN = r"^S1-\d+$"
GT_MATCH_PATTERN = r"^S[23]-\d+$"

# ---------------------------------------------------------------------------
# Missing-value handling
# ---------------------------------------------------------------------------
# Whole-field NA-like tokens (business_name). Matched case-insensitively
# after stripping whitespace.
NA_LIKE_TOKENS = {"na", "n/a", "null", "none", "nan", "<null>", "<na>"}

# ---------------------------------------------------------------------------
# Business-name legal-suffix canonicalization
# ---------------------------------------------------------------------------
# Token-boundary-only mapping applied to business_name_canonical. Keys are
# matched against a token with trailing punctuation stripped (e.g. "ltd."
# and "ltd" both match the "ltd" key). This map is intentionally small and
# conservative — it only contains variants actually observed in the dataset
# (Section 3 / Section 9 of implementation.md).
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "inc": "incorporated",
    "co": "company",
}

# ---------------------------------------------------------------------------
# Address abbreviation canonicalization
# ---------------------------------------------------------------------------
ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "ste": "suite",
    "fl": "floor",
}

# Landmark prefixes (word-boundary matched — see normalization.py). Order
# matters only in that "opp." is checked before "opp" would be redundant
# since both are listed explicitly; longer/more-specific tokens are checked
# first to avoid a short prefix masking a more specific one.
LANDMARK_PREFIXES = ["opposite", "opp.", "opp", "behind", "near"]

# Decorative leading characters stripped from the START of a name/address
# only (never from the middle) — verified against the "leading special
# character" pattern observed in up to ~577k Source 2 rows.
DECORATIVE_LEADING_CHARS = r"\-\#\[\]<>@\.\*\s"

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
REQUIRED_OUTPUT_COLUMNS = [
    "entity_id",
    "business_name", "business_name_normalized", "business_name_canonical",
    "business_address", "business_address_normalized", "business_address_canonical",
    "address_landmark",
    "country", "country_normalized",
    "business_name_is_missing", "business_address_is_missing",
]

GT_OUTPUT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]

# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
# Bound how much detail the collision report keeps in memory / on disk.
MAX_COLLISION_GROUPS_REPORTED = 200
MAX_SAMPLE_IDS_PER_GROUP = 10

# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"
ALL_SPLITS = (TRAIN_SPLIT, TEST_SPLIT)


@dataclass
class Paths:
    input_dir: str
    output_processed_dir: str
    output_metadata_dir: str
