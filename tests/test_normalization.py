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


# ===========================================================================
# P1 NEW TESTS — added during Pipeline A finalization
# ===========================================================================

from src.normalization import (
    extract_address_numbers,
    extract_address_numbers_series,
    extract_address_postal_code,
    extract_address_postal_code_series,
    make_address_sorted_tokens,
    make_address_sorted_tokens_series,
    classify_name_script,
    classify_name_script_series,
)


# ---------------------------------------------------------------------------
# P1a — Legal suffix map extensions
# ---------------------------------------------------------------------------

def test_llc_canonicalized():
    n, _ = normalize_name_basic("Meridian LLC")
    assert canonicalize_name(n) == "meridian llc"


def test_llp_canonicalized():
    n, _ = normalize_name_basic("Meridian LLP")
    assert canonicalize_name(n) == "meridian llp"


def test_pllc_canonicalized():
    n, _ = normalize_name_basic("Meridian PLLC")
    assert canonicalize_name(n) == "meridian pllc"


def test_pc_canonicalized():
    n, _ = normalize_name_basic("Meridian PC")
    assert canonicalize_name(n) == "meridian pc"


def test_sarl_canonicalized():
    n, _ = normalize_name_basic("Meridian SARL")
    assert canonicalize_name(n) == "meridian sarl"


def test_sas_canonicalized():
    n, _ = normalize_name_basic("Meridian SAS")
    assert canonicalize_name(n) == "meridian sas"


def test_sci_canonicalized():
    n, _ = normalize_name_basic("Meridian SCI")
    assert canonicalize_name(n) == "meridian sci"


def test_sa_canonicalized():
    n, _ = normalize_name_basic("Ecole Francaise SA")
    assert canonicalize_name(n) == "ecole francaise sa"


def test_dot_separated_llc_canonicalized():
    # L.L.C. -> llc (dots stripped at token level)
    n, _ = normalize_name_basic("Meridian L.L.C.")
    assert canonicalize_name(n) == "meridian llc"


def test_dot_separated_sarl_canonicalized():
    n, _ = normalize_name_basic("Meridian S.A.R.L")
    assert canonicalize_name(n) == "meridian sarl"


def test_new_suffixes_token_boundary_only():
    # "sa" must NOT fire inside "savings"
    n, _ = normalize_name_basic("Santa Savings Corp")
    canon = canonicalize_name(n)
    assert "savings" in canon          # not replaced
    assert canon == "santa savings corporation"


def test_pc_token_boundary_safe():
    # "pc" must NOT fire inside "pacific"
    n, _ = normalize_name_basic("Pacific Coast Inc")
    canon = canonicalize_name(n)
    assert "pacific" in canon
    assert canon == "pacific coast incorporated"


def test_existing_suffixes_still_work_after_map_extension():
    # Regression: pvt/ltd/corp/inc/co must still canonicalize as before
    n, _ = normalize_name_basic("Invest Investments Pvt Ltd")
    assert canonicalize_name(n) == "invest investments private limited"


# ---------------------------------------------------------------------------
# P1b — Address abbreviation map extensions
# ---------------------------------------------------------------------------

def test_dr_abbreviation():
    n, _ = normalize_address_basic("123 Main Dr, Austin, TX")
    assert "drive" in canonicalize_address(n)


def test_ct_abbreviation():
    n, _ = normalize_address_basic("45 Oak Ct, Denver, CO")
    assert "court" in canonicalize_address(n)


def test_ln_abbreviation():
    n, _ = normalize_address_basic("7 Elm Ln, Nashville, TN")
    assert "lane" in canonicalize_address(n)


def test_blvd_abbreviation():
    n, _ = normalize_address_basic("500 Sunset Blvd, Hollywood, CA")
    assert "boulevard" in canonicalize_address(n)


def test_pl_abbreviation():
    n, _ = normalize_address_basic("10 Park Pl, New York, NY")
    assert "place" in canonicalize_address(n)


def test_cir_abbreviation():
    n, _ = normalize_address_basic("3 Pine Cir, Portland, OR")
    assert "circle" in canonicalize_address(n)


def test_ter_abbreviation():
    n, _ = normalize_address_basic("22 Oak Ter, Boston, MA")
    assert "terrace" in canonicalize_address(n)


def test_hwy_abbreviation():
    n, _ = normalize_address_basic("900 Old Hwy, Memphis, TN")
    assert "highway" in canonicalize_address(n)


def test_pkwy_abbreviation():
    n, _ = normalize_address_basic("1 River Pkwy, Minneapolis, MN")
    assert "parkway" in canonicalize_address(n)


def test_sq_abbreviation():
    n, _ = normalize_address_basic("5 Town Sq, London")
    assert "square" in canonicalize_address(n)


def test_dr_token_boundary_safe():
    # "dr" inside "drexel" must NOT expand
    n, _ = normalize_address_basic("Drexel Ave, Chicago")
    canon = canonicalize_address(n)
    assert "drexel" in canon
    assert "drive" not in canon


def test_pl_token_boundary_safe():
    n, _ = normalize_address_basic("Plymouth St, Boston")
    canon = canonicalize_address(n)
    assert "plymouth" in canon
    assert "place" not in canon


def test_existing_abbreviations_still_work_after_extension():
    # Regression
    n, _ = normalize_address_basic("3315 Fremont St, Peoria, IL")
    assert canonicalize_address(n) == "3315 fremont street, peoria, il"


# ---------------------------------------------------------------------------
# P1c — address_numbers
# ---------------------------------------------------------------------------

def test_address_numbers_shop_12_mg_road():
    assert extract_address_numbers("shop 12, mg road") == "12"


def test_address_numbers_12_mg_road():
    assert extract_address_numbers("12 mg road") == "12"


def test_address_numbers_sector_17():
    assert extract_address_numbers("sector 17, chandigarh") == "17"


def test_address_numbers_plot_42_sector_17():
    assert extract_address_numbers("plot 42, sector 17") == "42 17"


def test_address_numbers_pin_only():
    assert extract_address_numbers("160017") == "160017"


def test_address_numbers_suite_400_1064():
    assert extract_address_numbers("suite 400, 1064 newton road, unit 11") == "400 1064 11"


def test_address_numbers_no_digits():
    assert extract_address_numbers("mg road, chandigarh") == ""


def test_address_numbers_empty():
    assert extract_address_numbers("") == ""


def test_address_numbers_raw_preserved():
    # numbers must not alter the normalized address itself
    norm, _ = normalize_address_basic("Shop 12, MG Road, Chandigarh, 160017")
    nums = extract_address_numbers(norm)
    assert "12" in nums
    assert "160017" in nums
    assert "shop 12" in norm          # original tokens still in norm


def test_address_numbers_series_parity():
    s = pd.Series(["shop 12, mg road", "sector 17", "", "160017"])
    result = extract_address_numbers_series(s)
    for i, v in enumerate(s):
        assert result.iloc[i] == extract_address_numbers(v)


# ---------------------------------------------------------------------------
# P1c — address_postal_code
# ---------------------------------------------------------------------------

def test_postal_code_6digit_india():
    assert extract_address_postal_code("chandigarh 160017") == "160017"


def test_postal_code_6digit_leading():
    assert extract_address_postal_code("560001 bangalore") == "560001"


def test_postal_code_5digit_us():
    assert extract_address_postal_code("peoria, il 61602") == "61602"


def test_postal_code_5digit_france():
    assert extract_address_postal_code("paris 75001") == "75001"


def test_postal_code_leading_zero_preserved():
    # Must return string "02101", not integer 2101
    result = extract_address_postal_code("boston ma 02101")
    assert result == "02101"
    assert result[0] == "0"


def test_postal_code_4digit_no_match():
    assert extract_address_postal_code("1064 newton road, iowa city") == ""


def test_postal_code_empty():
    assert extract_address_postal_code("") == ""


def test_postal_code_10digit_no_match():
    # 10-digit phone number must not be returned
    assert extract_address_postal_code("phone 9876543210") == ""


def test_postal_code_series_parity():
    s = pd.Series(["chandigarh 160017", "peoria, il 61602", "", "mg road"])
    result = extract_address_postal_code_series(s)
    for i, v in enumerate(s):
        assert result.iloc[i] == extract_address_postal_code(v)


# ---------------------------------------------------------------------------
# P1c — address_sorted_tokens
# ---------------------------------------------------------------------------

def test_sorted_tokens_order_invariance():
    # Two addresses with the same tokens in different order → same output
    a = make_address_sorted_tokens("mg road sector 17")
    b = make_address_sorted_tokens("sector 17 mg road")
    assert a == b


def test_sorted_tokens_deterministic():
    addr = "sector 17 mg road chandigarh"
    assert make_address_sorted_tokens(addr) == make_address_sorted_tokens(addr)


def test_sorted_tokens_empty():
    assert make_address_sorted_tokens("") == ""


def test_sorted_tokens_single_token():
    assert make_address_sorted_tokens("chandigarh") == "chandigarh"


def test_sorted_tokens_does_not_replace_canonical():
    # sorted view is ADDITIONAL — canonical must be unchanged
    norm, _ = normalize_address_basic("3315 Fremont St, Peoria, IL")
    canon = canonicalize_address(norm)
    sorted_view = make_address_sorted_tokens(canon)
    assert canon == "3315 fremont street, peoria, il"   # unchanged
    assert sorted_view != canon                          # different view


def test_sorted_tokens_series_parity():
    s = pd.Series(["mg road sector 17", "sector 17 mg road", ""])
    result = make_address_sorted_tokens_series(s)
    for i, v in enumerate(s):
        assert result.iloc[i] == make_address_sorted_tokens(v)


# ---------------------------------------------------------------------------
# P1c — name_script_class
# ---------------------------------------------------------------------------

def test_script_latin_ascii():
    assert classify_name_script("primary care group") == "latin"


def test_script_latin_all_caps():
    assert classify_name_script("ABC TRADING") == "latin"


def test_script_latin_accented():
    assert classify_name_script("écoles françaises") == "latin"


def test_script_empty_string():
    assert classify_name_script("") == "empty"


def test_script_whitespace_only():
    assert classify_name_script("   ") == "empty"


def test_script_digits_only():
    # No letter characters → empty
    assert classify_name_script("123456") == "empty"


def test_script_devanagari():
    assert classify_name_script("नमस्ते") == "devanagari"


def test_script_tamil():
    assert classify_name_script("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்") == "tamil"


def test_script_gujarati():
    assert classify_name_script("ગુજરાત") == "gujarati"


def test_script_gurmukhi():
    assert classify_name_script("ਪੰਜਾਬ") == "gurmukhi"


def test_script_malayalam():
    assert classify_name_script("കേരളം") == "malayalam"


def test_script_telugu():
    assert classify_name_script("తెలుగు") == "telugu"


def test_script_mixed_latin_devanagari():
    assert classify_name_script("Raj नमस्ते Investments") == "latin_and_non_latin"


def test_script_mixed_latin_tamil():
    assert classify_name_script("Raj ராஜ் LLP") == "latin_and_non_latin"


def test_script_does_not_modify_name():
    name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
    classify_name_script(name)       # must not raise or mutate
    assert name == "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"


def test_script_series_parity():
    names = pd.Series(["primary care group", "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்",
                        "", "Raj नमस्ते LLP", "123"])
    result = classify_name_script_series(names)
    for i, n in enumerate(names):
        assert result.iloc[i] == classify_name_script(n)


# ---------------------------------------------------------------------------
# Missing-value safety for new functions
# ---------------------------------------------------------------------------

def test_address_numbers_none_input():
    # Series wrapper fills None with ""
    s = pd.Series([None, "shop 12"])
    result = extract_address_numbers_series(s)
    assert result.iloc[0] == ""
    assert result.iloc[1] == "12"


def test_postal_code_none_input():
    s = pd.Series([None, "160017"])
    result = extract_address_postal_code_series(s)
    assert result.iloc[0] == ""
    assert result.iloc[1] == "160017"


def test_sorted_tokens_none_input():
    s = pd.Series([None, "mg road"])
    result = make_address_sorted_tokens_series(s)
    assert result.iloc[0] == ""
    assert result.iloc[1] == "mg road"


def test_script_none_input():
    s = pd.Series([None, "abc"])
    result = classify_name_script_series(s)
    assert result.iloc[0] == "empty"
    assert result.iloc[1] == "latin"


# ---------------------------------------------------------------------------
# Open-set country: unseen country still survives
# ---------------------------------------------------------------------------

def test_open_set_country_france_survives():
    from src.normalization import normalize_country
    assert normalize_country("France") == "france"
    assert normalize_country("FRANCE") == "france"


# ---------------------------------------------------------------------------
# Raw preservation invariant
# ---------------------------------------------------------------------------

def test_raw_values_preserved_in_clean_source():
    from src.preprocess import clean_source
    import pandas as pd
    rows = [
        ["S1-1", "Meridian LLC", "123 Main Dr, Austin, TX", "US"],
        ["S1-2", "ராஜ் எல்எல்பி", "560001 Bangalore", "India"],
        ["S1-3", "N/A", "", "France"],
    ]
    df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                      "business_address", "country"])
    out, stats = clean_source(df, "source1", "test")
    # row count preserved
    assert len(out) == 3
    # raw fields untouched
    assert out.loc[out["entity_id"] == "S1-1", "business_name"].iloc[0] == "Meridian LLC"
    assert out.loc[out["entity_id"] == "S1-2", "business_address"].iloc[0] == "560001 Bangalore"
    assert out.loc[out["entity_id"] == "S1-3", "country"].iloc[0] == "France"
    # new derived fields present
    assert "address_numbers" in out.columns
    assert "address_postal_code" in out.columns
    assert "address_sorted_tokens" in out.columns
    assert "name_script_class" in out.columns
    # spot-check values
    assert out.loc[out["entity_id"] == "S1-1", "business_name_canonical"].iloc[0] == "meridian llc"
    assert out.loc[out["entity_id"] == "S1-1", "address_numbers"].iloc[0] == "123"
    assert out.loc[out["entity_id"] == "S1-2", "address_postal_code"].iloc[0] == "560001"
    assert out.loc[out["entity_id"] == "S1-2", "name_script_class"].iloc[0] == "tamil"
    assert out.loc[out["entity_id"] == "S1-3", "business_name_is_missing"].iloc[0] == True
    assert out.loc[out["entity_id"] == "S1-3", "country_normalized"].iloc[0] == "france"


# ---------------------------------------------------------------------------
# No Matcher / downstream logic in Pipeline A
# ---------------------------------------------------------------------------

def test_no_matcher_in_preprocess_run():
    import inspect
    from src import preprocess
    src_text = inspect.getsource(preprocess.run)
    assert "Matcher" not in src_text
    assert "cross_validate" not in src_text
    assert "candidate_pairs" not in src_text
    assert "score_pairs" not in src_text
    assert "fit_transform" not in src_text
