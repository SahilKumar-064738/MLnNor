"""Unit tests for src/normalization.py.

Examples are drawn from patterns actually present in the dataset described
in implementation.md (Section 22), plus targeted edge cases for the fixes
called out in Section 26 (word-boundary landmark detection, token-boundary
suffix/abbreviation mapping, &/legal-suffix consistency).
"""
import pandas as pd
import pytest

from src.normalization import (
    canonicalize_address,
    canonicalize_address_series,
    canonicalize_name,
    canonicalize_name_series,
    collapse_whitespace,
    extract_landmark,
    extract_landmark_series,
    is_na_like,
    nfkc,
    normalize_address_basic,
    normalize_address_series,
    normalize_country,
    normalize_country_series,
    normalize_name_basic,
    normalize_name_series,
)


# ---------------------------------------------------------------------------
# Unicode NFKC
# ---------------------------------------------------------------------------

def test_nfkc_folds_fullwidth_characters():
    # U+FF21 FULLWIDTH LATIN CAPITAL LETTER A -> "A"
    assert nfkc("\uFF21\uFF22\uFF23") == "ABC"


def test_nfkc_does_not_transliterate_scripts():
    # Devanagari text must remain Devanagari -- NFKC folds compatibility
    # variants, it does not translate/transliterate between scripts.
    text = "नमस्ते"
    assert nfkc(text) == text


# ---------------------------------------------------------------------------
# Lowercasing + whitespace
# ---------------------------------------------------------------------------

def test_collapse_whitespace():
    assert collapse_whitespace("  a   b\t\tc  ") == "a b c"


def test_name_case_and_whitespace():
    # from train_source2.tsv: ALL-CAPS + double space pattern
    norm, missing = normalize_name_basic("BABA-PVT. LTD.  CENTER")
    assert norm == "baba-pvt. ltd. center"
    assert missing is False


# ---------------------------------------------------------------------------
# Decorative leading characters
# ---------------------------------------------------------------------------

def test_name_leading_decoration_stripped():
    # from train_source2.tsv
    norm, _ = normalize_name_basic("-- Holloway Peak Inc Seafood")
    assert norm == "holloway peak inc seafood"


def test_name_leading_bracket_decoration_stripped():
    # The decorative strip only removes a LEADING run of decorative chars;
    # once a non-decorative character (the letter 'i') is hit, stripping
    # stops -- so the "]" after "incorporated" is untouched (internal char).
    norm, _ = normalize_name_basic("[INCORPORATED] PEAK TRADIN6 NETWORKS SOUTHSIDE")
    assert norm == "incorporated] peak tradin6 networks southside"


def test_internal_characters_not_stripped():
    # Decorative-char stripping must only ever touch the START of the
    # string, never characters in the middle/end.
    norm, _ = normalize_name_basic("Acme - Best [Deals] Inc.")
    assert "acme - best [deals] inc." == norm


# ---------------------------------------------------------------------------
# Ampersand + legal suffix canonicalization (Section 26 item 1)
# ---------------------------------------------------------------------------

def test_legal_suffix_canonicalization():
    norm, _ = normalize_name_basic("Invest Investments Pvt Ltd")
    canon = canonicalize_name(norm)
    assert "private" in canon and "limited" in canon


def test_ampersand_canonicalization():
    norm, _ = normalize_name_basic("Callicoat & Dailey Inc")
    canon = canonicalize_name(norm)
    assert " and " in canon
    assert "incorporated" in canon


def test_ampersand_and_suffix_mapping_are_consistent():
    # "X & Y Inc" and "X and Y Incorporated" must canonicalize identically --
    # this is the internal-consistency issue flagged in Section 26 item 1.
    norm1, _ = normalize_name_basic("Smith & Jones Inc")
    norm2, _ = normalize_name_basic("Smith and Jones Incorporated")
    assert canonicalize_name(norm1) == canonicalize_name(norm2)


def test_suffix_mapping_is_token_boundary_only():
    # "co" must not match inside "coffee" or "company" itself.
    norm, _ = normalize_name_basic("Coffee Co")
    canon = canonicalize_name(norm)
    assert canon == "coffee company"
    assert "companyffee" not in canon


def test_suffix_mapping_handles_trailing_period():
    norm, _ = normalize_name_basic("Acme Corp.")
    canon = canonicalize_name(norm)
    assert canon == "acme corporation"


def test_empty_name_canonicalizes_to_empty():
    assert canonicalize_name("") == ""


# ---------------------------------------------------------------------------
# NA-like / missing names
# ---------------------------------------------------------------------------

def test_is_na_like():
    assert is_na_like("NA")
    assert is_na_like(" n/a ")
    assert is_na_like("NULL")
    assert not is_na_like("National Bank")


def test_na_like_name_is_flagged_missing():
    norm, missing = normalize_name_basic("N/A")
    assert norm == ""
    assert missing is True


def test_empty_name_is_flagged_missing():
    norm, missing = normalize_name_basic("")
    assert norm == ""
    assert missing is True


# ---------------------------------------------------------------------------
# Address: null components, abbreviations, missing
# ---------------------------------------------------------------------------

def test_address_null_component_stripped():
    # from train_source2.tsv real example
    norm, missing = normalize_address_basic("067 PRODUCTION CT, NULL, INDEPENDENCE, KY")
    assert "null" not in norm
    assert missing is False


def test_address_bracketed_null_component_stripped():
    norm, missing = normalize_address_basic("12 Main St, <NULL>, Springfield")
    assert "null" not in norm
    assert missing is False


def test_address_empty_field_flagged_missing():
    norm, missing = normalize_address_basic("")
    assert norm == "" and missing is True


def test_address_all_null_components_flagged_missing():
    norm, missing = normalize_address_basic("NULL, NULL")
    assert norm == "" and missing is True


def test_road_abbreviation_canonicalization():
    norm, _ = normalize_address_basic("3315 Fremont St, Peoria, IL")
    canon = canonicalize_address(norm)
    assert "street" in canon


def test_address_abbreviation_is_token_boundary_only():
    # "st" must not match inside "street" (already spelled out) or "first".
    norm, _ = normalize_address_basic("First Street, Boston")
    canon = canonicalize_address(norm)
    assert canon == "first street, boston"


def test_address_matched_pair_normalizes_identically():
    # ground-truth-linked pair from train S1 <-> S2
    norm1, _ = normalize_address_basic("3315 Fremont Street, Peoria, IL")
    norm2, _ = normalize_address_basic("3315 FREMONT ST, PEORIA, IL")
    assert canonicalize_address(norm1) == canonicalize_address(norm2)


def test_building_and_unit_numbers_preserved():
    norm, _ = normalize_address_basic("Suite 400, 1064 Newton Rd, Unit 11")
    canon = canonicalize_address(norm)
    assert "400" in canon
    assert "1064" in canon
    assert "11" in canon


# ---------------------------------------------------------------------------
# Landmark extraction (Section 26 item 3: word-boundary aware)
# ---------------------------------------------------------------------------

def test_landmark_preserved_not_deleted():
    original = "H.No.16-11-23/37/A, Opp.Rta Office, Hyderabad, Telangana"
    landmark = extract_landmark(original)
    assert landmark != ""
    norm, _ = normalize_address_basic(original)
    assert "opp" in norm  # landmark text still present in the address itself


def test_landmark_near():
    landmark = extract_landmark("12 Main St, Near Fortis Hospital, Delhi")
    assert landmark.lower().startswith("near")
    assert "fortis" in landmark.lower()


def test_landmark_word_boundary_not_matched_inside_word():
    # "near" must NOT match inside "nearby" -- this is the exact bug
    # flagged in Section 26 item 3.
    landmark = extract_landmark("123 Nearby Lane, Chicago")
    assert landmark == ""


def test_landmark_word_boundary_not_matched_inside_shearer():
    landmark = extract_landmark("45 Shearer Road, Melbourne")
    assert landmark == ""


def test_landmark_opp_not_matched_inside_opposition():
    landmark = extract_landmark("10 Opposition Ave, Springfield")
    assert landmark == ""


def test_landmark_none_found():
    assert extract_landmark("100 Main St, Anytown, CA") == ""


def test_landmark_behind():
    landmark = extract_landmark("5 Elm St, Behind City Mall, Pune")
    assert landmark.lower().startswith("behind")


# ---------------------------------------------------------------------------
# Country: open-set, must handle unseen values (e.g. France, test-only)
# ---------------------------------------------------------------------------

def test_country_normalization_handles_unseen_value():
    # France only appears in TEST data, never in training -- must still work
    assert normalize_country("France") == "france"
    assert normalize_country(" FRANCE ") == "france"
    assert normalize_country("India") == "india"


def test_country_normalization_arbitrary_unseen_value():
    # No hardcoded universe -- any string normalizes safely.
    assert normalize_country("Elbonia") == "elbonia"


def test_country_normalization_empty():
    assert normalize_country("") == ""
    assert normalize_country(None) == ""


# ---------------------------------------------------------------------------
# Vectorized (Series) wrappers must match single-value behavior exactly
# ---------------------------------------------------------------------------

def test_name_series_matches_single_value_function():
    names = pd.Series(["BABA-PVT. LTD.  CENTER", "-- Holloway Peak Inc Seafood", "N/A", ""])
    norm_series, missing_series = normalize_name_series(names)
    for i, n in enumerate(names):
        expected_norm, expected_missing = normalize_name_basic(n)
        assert norm_series.iloc[i] == expected_norm
        assert bool(missing_series.iloc[i]) == expected_missing


def test_address_series_matches_single_value_function():
    addrs = pd.Series([
        "067 PRODUCTION CT, NULL, INDEPENDENCE, KY",
        "",
        "3315 Fremont St, Peoria, IL",
    ])
    norm_series, missing_series = normalize_address_series(addrs)
    for i, a in enumerate(addrs):
        expected_norm, expected_missing = normalize_address_basic(a)
        assert norm_series.iloc[i] == expected_norm
        assert bool(missing_series.iloc[i]) == expected_missing


def test_country_series_matches_single_value_function():
    countries = pd.Series(["France", " India ", "US", ""])
    result = normalize_country_series(countries)
    for i, c in enumerate(countries):
        assert result.iloc[i] == normalize_country(c)


def test_canonicalize_name_series_matches_single_value():
    normalized = pd.Series(["invest investments pvt ltd", "callicoat & dailey inc", ""])
    result = canonicalize_name_series(normalized)
    for i, n in enumerate(normalized):
        assert result.iloc[i] == canonicalize_name(n)


def test_canonicalize_address_series_matches_single_value():
    normalized = pd.Series(["3315 fremont st, peoria, il", ""])
    result = canonicalize_address_series(normalized)
    for i, n in enumerate(normalized):
        assert result.iloc[i] == canonicalize_address(n)


def test_extract_landmark_series_matches_single_value():
    addrs = pd.Series(["123 Nearby Lane, Chicago", "5 Elm St, Behind City Mall, Pune", ""])
    result = extract_landmark_series(addrs)
    for i, a in enumerate(addrs):
        assert result.iloc[i] == extract_landmark(a)
