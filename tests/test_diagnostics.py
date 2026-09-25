"""Unit tests for src/diagnostics.py — collision & exact-duplicate reporting."""
import pandas as pd

from src.diagnostics import (
    exact_duplicate_rows,
    name_canonical_collisions,
    name_address_country_collisions,
    collision_groups_to_frame,
)


def _df():
    return pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3", "S1-4"],
        "business_name_canonical": ["primary care group", "primary care group", "primary care group", "acme inc"],
        "business_address_canonical": ["1 main st", "2 oak ave", "3 pine rd", "9 elm st"],
        "country_normalized": ["us", "us", "us", "us"],
    })


def test_name_canonical_collision_detected():
    result = name_canonical_collisions(_df())
    assert result["group_count"] == 1
    assert result["rows_involved"] == 3
    assert result["max_group_size"] == 3
    assert len(result["top_groups"]) == 1
    assert result["top_groups"][0]["group_size"] == 3


def test_no_collisions_when_all_unique():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name_canonical": ["a co", "b co"],
    })
    result = name_canonical_collisions(df)
    assert result["group_count"] == 0
    assert result["rows_involved"] == 0
    assert result["top_groups"] == []


def test_name_address_country_collision_splits_franchise_case():
    # Same canonical name but different addresses -> NOT a collision under
    # the tightest grouping (this is the "253 distinct Primary Care Group
    # entities" franchise case from implementation.md Section 3).
    result = name_address_country_collisions(_df())
    assert result["group_count"] == 0


def test_collision_groups_to_frame_bounded_and_shaped():
    result = name_canonical_collisions(_df())
    frame = collision_groups_to_frame(result)
    assert "group_size" in frame.columns
    assert "sample_entity_ids" in frame.columns
    assert len(frame) == 1
    assert frame.iloc[0]["group_size"] == 3


def test_collision_groups_to_frame_empty_result():
    result = {"top_groups": []}
    frame = collision_groups_to_frame(result)
    assert len(frame) == 0


def test_exact_duplicate_rows_none_found():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2"],
        "business_name": ["a", "b"],
        "business_address": ["x", "y"],
        "country": ["US", "US"],
    })
    result = exact_duplicate_rows(df, ["business_name", "business_address", "country"])
    assert result["exact_duplicate_row_count"] == 0


def test_exact_duplicate_rows_detected():
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["a", "a", "b"],
        "business_address": ["x", "x", "y"],
        "country": ["US", "US", "US"],
    })
    result = exact_duplicate_rows(df, ["business_name", "business_address", "country"])
    assert result["exact_duplicate_row_count"] == 2
    assert result["exact_duplicate_group_count"] == 1
