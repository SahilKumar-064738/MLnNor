"""Unit tests for src/validation.py."""
import pandas as pd
import pytest

from src.config import GT_COLUMNS, SOURCE_COLUMNS
from src.validation import (
    FatalValidationError,
    validate_entity_ids,
    validate_ground_truth_duplicates,
    validate_ground_truth_referential,
    validate_ground_truth_schema,
    validate_id_set_preserved,
    validate_row_count_preserved,
    validate_schema,
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def test_schema_ok_does_not_raise():
    df = pd.DataFrame(columns=SOURCE_COLUMNS)
    validate_schema(df, SOURCE_COLUMNS, "t")  # should not raise


def test_schema_mismatch_raises():
    df = pd.DataFrame({"entity_id": ["S1-1"], "name": ["x"]})
    with pytest.raises(FatalValidationError):
        validate_schema(df, SOURCE_COLUMNS, "t")


# ---------------------------------------------------------------------------
# Entity IDs
# ---------------------------------------------------------------------------

def test_valid_entity_ids_pass():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["a", "b"],
        "business_address": ["x", "y"],
        "country": ["US", "US"],
    })
    validate_entity_ids(df, "source1", "t")  # should not raise


def test_malformed_entity_id_raises():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "BAD-ID"],
        "business_name": ["a", "b"],
        "business_address": ["x", "y"],
        "country": ["US", "US"],
    })
    with pytest.raises(FatalValidationError):
        validate_entity_ids(df, "source1", "t")


def test_wrong_source_prefix_raises():
    # S2 id in a source1 file must be fatal.
    df = pd.DataFrame({
        "entity_id": ["S2-1"],
        "business_name": ["a"],
        "business_address": ["x"],
        "country": ["US"],
    })
    with pytest.raises(FatalValidationError):
        validate_entity_ids(df, "source1", "t")


def test_duplicate_entity_id_raises():
    df = pd.DataFrame({"entity_id": ["S1-1", "S1-1"],
                        "business_name": ["a", "b"],
                        "business_address": ["x", "y"],
                        "country": ["US", "US"]})
    with pytest.raises(FatalValidationError):
        validate_entity_ids(df, "source1", "t")


# ---------------------------------------------------------------------------
# Row-count and ID-set invariants
# ---------------------------------------------------------------------------

def test_row_count_match_passes():
    validate_row_count_preserved(100, 100, "t")  # should not raise


def test_row_count_mismatch_raises():
    with pytest.raises(FatalValidationError):
        validate_row_count_preserved(100, 99, "t")


def test_id_set_equal_passes():
    validate_id_set_preserved({"S1-1", "S1-2"}, {"S1-1", "S1-2"}, "t")  # should not raise


def test_id_set_mismatch_raises_even_with_same_count():
    # Same COUNT but different actual IDs (a "swap") must still be caught --
    # this is exactly the invariant a row-count-only check would miss
    # (Section 26 item 5).
    with pytest.raises(FatalValidationError):
        validate_id_set_preserved({"S1-1", "S1-2"}, {"S1-1", "S1-3"}, "t")


def test_id_set_missing_id_raises():
    with pytest.raises(FatalValidationError):
        validate_id_set_preserved({"S1-1", "S1-2"}, {"S1-1"}, "t")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def test_ground_truth_schema_ok():
    gt = pd.DataFrame(columns=GT_COLUMNS)
    validate_ground_truth_schema(gt, GT_COLUMNS, "t")  # should not raise


def test_ground_truth_duplicate_key_raises():
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1"],
        "matched_entity_ids": ["S2-1", "S2-2"],
    })
    with pytest.raises(FatalValidationError):
        validate_ground_truth_duplicates(gt, "t")


def test_ground_truth_valid_reference_passes():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S3-1"]})
    stats = validate_ground_truth_referential(gt, {"S1-1"}, {"S2-1"}, {"S3-1"}, "t")
    assert stats["rows"] == 1
    assert stats["zero_match_rows"] == 0


def test_ground_truth_empty_match_list_is_valid():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": [""]})
    stats = validate_ground_truth_referential(gt, {"S1-1"}, set(), set(), "t")
    assert stats["zero_match_rows"] == 1


def test_ground_truth_invalid_reference_raises():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-999"]})
    with pytest.raises(FatalValidationError):
        validate_ground_truth_referential(gt, {"S1-1"}, {"S2-1"}, {"S3-1"}, "t")


def test_ground_truth_unknown_source1_id_raises():
    gt = pd.DataFrame({"source1_entity_id": ["S1-999"], "matched_entity_ids": [""]})
    with pytest.raises(FatalValidationError):
        validate_ground_truth_referential(gt, {"S1-1"}, set(), set(), "t")


def test_ground_truth_malformed_match_token_raises():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["NOT-AN-ID"]})
    with pytest.raises(FatalValidationError):
        validate_ground_truth_referential(gt, {"S1-1"}, {"S2-1"}, {"S3-1"}, "t")


def test_ground_truth_duplicate_id_within_cell_raises():
    gt = pd.DataFrame({"source1_entity_id": ["S1-1"], "matched_entity_ids": ["S2-1,S2-1"]})
    with pytest.raises(FatalValidationError):
        validate_ground_truth_referential(gt, {"S1-1"}, {"S2-1"}, set(), "t")


def test_ground_truth_stats_match_counts():
    gt = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-2", "S1-3"],
        "matched_entity_ids": ["S2-1,S3-1", "S2-2", ""],
    })
    stats = validate_ground_truth_referential(
        gt, {"S1-1", "S1-2", "S1-3"}, {"S2-1", "S2-2"}, {"S3-1"}, "t"
    )
    assert stats["zero_match_rows"] == 1
    assert stats["max_matches_for_one_entity"] == 2
    assert stats["total_s2_references"] == 2
    assert stats["total_s3_references"] == 1
