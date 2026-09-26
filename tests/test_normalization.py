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


# ===========================================================================
# P2 NEW TESTS — production-grade cleaning engine upgrade
# Five feature areas: accent folding, URL/email strip, leet repair,
# DBA/TA strip, config map extensions (suffixes + address abbrevs).
# All existing 114 tests above must continue to pass unchanged.
# ===========================================================================

from src.normalization import fold_accents


# ---------------------------------------------------------------------------
# Accent folding — fold_accents() helper
# ---------------------------------------------------------------------------

class TestFoldAccents:
    def test_cafe(self):
        assert fold_accents("café") == "cafe"

    def test_naive(self):
        assert fold_accents("naïve") == "naive"

    def test_resume(self):
        assert fold_accents("résumé") == "resume"

    def test_angstrom(self):
        assert fold_accents("Ångström") == "Angstrom"

    def test_ecole_francaise(self):
        assert fold_accents("École Française") == "Ecole Francaise"

    def test_pure_ascii_unchanged(self):
        assert fold_accents("hello world") == "hello world"

    def test_empty_string(self):
        assert fold_accents("") == ""

    def test_devanagari_untouched(self):
        # Combining marks in Devanagari are phonemic — must NOT be stripped.
        text = "नमस्ते"
        assert fold_accents(text) == text

    def test_tamil_untouched(self):
        text = "ராஜ்"
        assert fold_accents(text) == text

    def test_gujarati_untouched(self):
        text = "ગુજરાત"
        assert fold_accents(text) == text

    def test_mixed_latin_non_latin_untouched(self):
        # If ANY non-Latin letter is present the whole string is left alone.
        text = "Café नमस्ते"
        assert fold_accents(text) == text


# ---------------------------------------------------------------------------
# Accent folding — integrated into normalize_name_basic
# ---------------------------------------------------------------------------

class TestAccentFoldingInNames:
    def test_cafe_variants_normalise_identically(self):
        n1, _ = normalize_name_basic("Café du Monde")
        n2, _ = normalize_name_basic("Cafe du Monde")
        assert n1 == n2

    def test_ecole_francaise_folded(self):
        norm, missing = normalize_name_basic("École Française")
        assert norm == "ecole francaise"
        assert missing is False

    def test_resume_in_name(self):
        norm, _ = normalize_name_basic("Résumé Writers LLC")
        assert "resume" in norm
        assert "é" not in norm

    def test_accented_name_not_missing(self):
        _, missing = normalize_name_basic("Société Générale")
        assert missing is False

    def test_devanagari_name_combining_marks_preserved(self):
        # normalize_name_basic must not corrupt Devanagari vowel signs.
        norm, missing = normalize_name_basic("नमस्ते")
        assert norm == "नमस्ते"
        assert missing is False

    def test_series_accent_parity(self):
        names = pd.Series(["Café du Monde", "Cafe du Monde", "École", "नमस्ते"])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ---------------------------------------------------------------------------
# Accent folding — integrated into normalize_address_basic
# ---------------------------------------------------------------------------

class TestAccentFoldingInAddresses:
    def test_residence_accent_folded(self):
        norm, _ = normalize_address_basic("Résidence du Parc, Paris")
        assert "residence" in norm
        assert "é" not in norm

    def test_accented_and_unaccented_address_canonicalise_identically(self):
        n1, _ = normalize_address_basic("Résidence du Parc, Paris")
        n2, _ = normalize_address_basic("Residence du Parc, Paris")
        assert n1 == n2

    def test_french_address_allee(self):
        norm, _ = normalize_address_basic("Allée des Roses, Lyon")
        canon = canonicalize_address(norm)
        # "allee" is what fold_accents gives us, and ADDRESS_ABBREV_MAP maps
        # "all" → "allee"; the word "allee" is already the canonical form.
        assert "allee" in canon

    def test_devanagari_address_untouched(self):
        text = "नमस्ते, दिल्ली"
        norm, missing = normalize_address_basic(text)
        # All Devanagari combining marks must survive normalisation.
        assert "ते" in norm
        assert missing is False

    def test_series_address_accent_parity(self):
        addrs = pd.Series(["Résidence du Parc, Paris", "Residence du Parc, Paris", ""])
        result, _ = normalize_address_series(addrs)
        for i, a in enumerate(addrs):
            expected, _ = normalize_address_basic(a)
            assert result.iloc[i] == expected


# ---------------------------------------------------------------------------
# URL / email stripping
# ---------------------------------------------------------------------------

class TestURLEmailStripping:
    # --- email ---
    def test_email_stripped_from_name(self):
        norm, _ = normalize_name_basic("Acme Corp info@acme.com")
        assert "@" not in norm
        assert "acme" in norm   # the non-email portion survives

    def test_email_only_name_becomes_missing(self):
        norm, missing = normalize_name_basic("info@acme.com")
        assert missing is True

    def test_email_mixed_case(self):
        norm, _ = normalize_name_basic("Best Buy Support@BestBuy.COM")
        assert "@" not in norm

    # --- full URL ---
    def test_http_url_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp http://www.acme.com")
        assert "http" not in norm
        assert "acme" in norm

    def test_https_url_stripped(self):
        norm, _ = normalize_name_basic("Visit us at https://shop.acme.in/sale")
        assert "https" not in norm
        assert "visit" in norm

    # --- www. domain ---
    def test_www_domain_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp www.acme.com")
        assert "www" not in norm
        assert "acme" in norm

    # --- bare domain token ---
    def test_bare_dotcom_stripped(self):
        norm, _ = normalize_name_basic("Acme Corp acme.com")
        assert ".com" not in norm

    def test_bare_dotin_stripped(self):
        norm, _ = normalize_name_basic("Flipkart flipkart.in")
        assert ".in" not in norm
        assert "flipkart" in norm

    def test_bare_dotorg_stripped(self):
        norm, _ = normalize_name_basic("Wikipedia wikipedia.org")
        assert ".org" not in norm

    # --- safety: legitimate names must not be mangled ---
    def test_legal_suffix_dot_not_stripped(self):
        # "S.A.R.L" must survive — it's a legal suffix, not a domain.
        norm, _ = normalize_name_basic("Meridian S.A.R.L")
        assert "s.a.r.l" in norm

    def test_address_dotcom_not_stripped(self):
        # The URL stripper must NOT be applied to addresses.
        norm, _ = normalize_address_basic("123 Commerce.com Drive, Austin")
        # Address normalisation leaves the text intact (no URL stripping).
        assert "commerce.com" in norm

    def test_name_without_url_unchanged(self):
        norm, _ = normalize_name_basic("Acme Corporation")
        assert norm == "acme corporation"

    def test_series_url_parity(self):
        names = pd.Series([
            "Acme Corp www.acme.com",
            "info@acme.com",
            "Just A Name",
        ])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ---------------------------------------------------------------------------
# Leet-speak repair
# ---------------------------------------------------------------------------

class TestLeetRepair:
    def test_preparatory_0_to_o(self):
        norm, _ = normalize_name_basic("Preparat0ry School")
        assert "preparatory" in norm

    def test_4cme_to_acme(self):
        norm, _ = normalize_name_basic("4cme Corp")
        assert "acme" in norm

    def test_3lite_to_elite(self):
        norm, _ = normalize_name_basic("3lite Networks")
        assert "elite" in norm

    def test_5ecure_to_secure(self):
        norm, _ = normalize_name_basic("5ecure Systems")
        assert "secure" in norm

    def test_7rading_to_trading(self):
        norm, _ = normalize_name_basic("7rading Co")
        assert "trading" in norm

    def test_1nvestments_to_investments(self):
        norm, _ = normalize_name_basic("1nvestments Ltd")
        assert "investments" in norm

    def test_multi_leet_token(self):
        norm, _ = normalize_name_basic("Pr3parat0ry Ac4d3my")
        assert "preparatory" in norm
        assert "academy" in norm

    # --- safety: pure-digit tokens must never be touched ---
    def test_pure_digit_token_untouched(self):
        # "7" alone is a pure-digit token — must stay "7", not become "t".
        norm, _ = normalize_name_basic("Studio 7")
        assert "7" in norm
        assert "t" not in norm.replace("studio", "")  # only the '7' check

    def test_building_number_untouched(self):
        # This is a NAME test — building numbers in names must survive.
        norm, _ = normalize_name_basic("Section 5 Investments")
        # "5" in "Section 5" is a pure-digit token (surrounded by spaces)
        assert "5" in norm

    def test_year_in_name_untouched(self):
        norm, _ = normalize_name_basic("Est. 1975 Bakeries")
        assert "1975" in norm

    def test_no_leet_digits_name_unchanged(self):
        norm, _ = normalize_name_basic("Blue Ocean Ltd")
        # No leet digits (0,1,3,4,5,7) present — leet repair does nothing.
        # normalize_name_basic produces the normalized form; canonicalize_name
        # separately maps "ltd" → "limited".
        assert norm == "blue ocean ltd"
        assert canonicalize_name(norm) == "blue ocean limited"

    def test_series_leet_parity(self):
        names = pd.Series(["Preparat0ry School", "4cme Corp", "Blue Ocean Ltd"])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ---------------------------------------------------------------------------
# DBA / T/A / AKA stripping
# ---------------------------------------------------------------------------

class TestDBAStripping:
    def test_dba_lowercase(self):
        norm, _ = normalize_name_basic("Acme Corp dba The Widget Store")
        # normalize_name_basic strips the DBA alias; "corp" is still the
        # abbreviated form at this stage — canonicalize_name turns it to
        # "corporation", but that is a separate step.
        assert norm == "acme corp"
        assert canonicalize_name(norm) == "acme corporation"

    def test_dba_uppercase(self):
        norm, _ = normalize_name_basic("Acme Corp DBA The Widget Store")
        assert norm == "acme corp"
        assert canonicalize_name(norm) == "acme corporation"

    def test_d_slash_b_slash_a(self):
        norm, _ = normalize_name_basic("Acme Corp d/b/a The Widget Store")
        assert norm == "acme corp"
        assert canonicalize_name(norm) == "acme corporation"

    def test_d_dot_b_dot_a(self):
        norm, _ = normalize_name_basic("Acme Corp d.b.a. The Widget Store")
        assert norm == "acme corp"
        assert canonicalize_name(norm) == "acme corporation"

    def test_doing_business_as(self):
        norm, _ = normalize_name_basic("Smith Plumbing doing business as Smith & Sons")
        assert "smith plumbing" in norm
        assert "sons" not in norm

    def test_trading_as(self):
        norm, _ = normalize_name_basic("Global Ventures trading as GV Express")
        assert "global ventures" in norm
        assert "express" not in norm

    def test_t_slash_a(self):
        norm, _ = normalize_name_basic("Jones Bakery t/a The Cake Shop")
        assert "jones bakery" in norm
        assert "cake" not in norm

    def test_t_dot_a_dot(self):
        norm, _ = normalize_name_basic("Jones Bakery t.a. The Cake Shop")
        assert "jones bakery" in norm
        assert "cake" not in norm

    def test_also_known_as(self):
        norm, _ = normalize_name_basic("First National Bank also known as FNB")
        assert "first national bank" in norm
        assert "fnb" not in norm

    def test_aka(self):
        norm, _ = normalize_name_basic("First National Bank aka FNB")
        assert "first national bank" in norm
        assert "fnb" not in norm

    def test_a_dot_k_dot_a_dot(self):
        norm, _ = normalize_name_basic("Raj Enterprises a.k.a. Raj & Co")
        assert "raj enterprises" in norm

    def test_name_without_dba_unchanged(self):
        norm, _ = normalize_name_basic("Acme Corporation")
        assert norm == "acme corporation"

    def test_comma_separated_dba(self):
        norm, _ = normalize_name_basic("City Lights, dba Night Owls")
        assert "city lights" in norm
        assert "night" not in norm

    def test_series_dba_parity(self):
        names = pd.Series([
            "Acme Corp dba The Widget Store",
            "Jones Bakery t/a The Cake Shop",
            "No Alias Here Inc",
        ])
        result, _ = normalize_name_series(names)
        for i, n in enumerate(names):
            expected, _ = normalize_name_basic(n)
            assert result.iloc[i] == expected


# ---------------------------------------------------------------------------
# Extended NA_LIKE_TOKENS
# ---------------------------------------------------------------------------

class TestExtendedNALike:
    def test_n_dot_a_dot_is_missing(self):
        norm, missing = normalize_name_basic("n.a.")
        assert missing is True

    def test_nil_is_missing(self):
        norm, missing = normalize_name_basic("nil")
        assert missing is True

    def test_nil_mixed_case(self):
        _, missing = normalize_name_basic("NIL")
        assert missing is True

    def test_not_available_is_missing(self):
        _, missing = normalize_name_basic("not available")
        assert missing is True

    def test_not_applicable_is_missing(self):
        _, missing = normalize_name_basic("not applicable")
        assert missing is True

    def test_unknown_is_missing(self):
        _, missing = normalize_name_basic("unknown")
        assert missing is True

    def test_lone_hyphen_is_missing(self):
        _, missing = normalize_name_basic("-")
        assert missing is True

    def test_en_dash_is_missing(self):
        _, missing = normalize_name_basic("\u2013")
        assert missing is True

    def test_em_dash_is_missing(self):
        _, missing = normalize_name_basic("\u2014")
        assert missing is True

    def test_legitimate_name_with_hyphen_not_missing(self):
        # A hyphen INSIDE a name is not a lone-hyphen placeholder.
        _, missing = normalize_name_basic("Coca-Cola Ltd")
        assert missing is False

    def test_is_na_like_helper_nil(self):
        from src.normalization import is_na_like
        assert is_na_like("nil")
        assert is_na_like("NIL")
        assert is_na_like("not available")
        assert is_na_like("unknown")
        assert is_na_like("n.a.")
        assert not is_na_like("National Bank")


# ---------------------------------------------------------------------------
# Extended LEGAL_SUFFIX_MAP — US additions
# ---------------------------------------------------------------------------

class TestExtendedLegalSuffixUS:
    def test_lp_canonicalized(self):
        n, _ = normalize_name_basic("Meridian LP")
        assert canonicalize_name(n) == "meridian lp"

    def test_l_dot_p_canonicalized(self):
        n, _ = normalize_name_basic("Meridian L.P.")
        assert canonicalize_name(n) == "meridian lp"

    def test_lc_canonicalized(self):
        n, _ = normalize_name_basic("Meridian LC")
        assert canonicalize_name(n) == "meridian lc"

    def test_pa_canonicalized(self):
        n, _ = normalize_name_basic("Smith Dental PA")
        assert canonicalize_name(n) == "smith dental pa"

    def test_p_dot_a_canonicalized(self):
        n, _ = normalize_name_basic("Smith Dental P.A.")
        assert canonicalize_name(n) == "smith dental pa"

    def test_lp_token_boundary_safe(self):
        # "lp" must NOT fire inside "help" or "tulip".
        n, _ = normalize_name_basic("Tulip Help Group")
        canon = canonicalize_name(n)
        assert "tulip" in canon
        assert "help" in canon

    def test_pa_token_boundary_safe(self):
        # "pa" must NOT fire inside "pasta" or "spain".
        n, _ = normalize_name_basic("Pasta Palace Inc")
        canon = canonicalize_name(n)
        assert "pasta" in canon


# ---------------------------------------------------------------------------
# Extended LEGAL_SUFFIX_MAP — French additions
# ---------------------------------------------------------------------------

class TestExtendedLegalSuffixFrance:
    def test_eurl_canonicalized(self):
        n, _ = normalize_name_basic("Dupont EURL")
        assert canonicalize_name(n) == "dupont eurl"

    def test_e_dot_u_dot_r_dot_l_canonicalized(self):
        n, _ = normalize_name_basic("Dupont E.U.R.L")
        assert canonicalize_name(n) == "dupont eurl"

    def test_snc_canonicalized(self):
        n, _ = normalize_name_basic("Dupont SNC")
        assert canonicalize_name(n) == "dupont snc"

    def test_s_dot_n_dot_c_canonicalized(self):
        n, _ = normalize_name_basic("Dupont S.N.C")
        assert canonicalize_name(n) == "dupont snc"

    def test_gie_canonicalized(self):
        n, _ = normalize_name_basic("Dupont GIE")
        assert canonicalize_name(n) == "dupont gie"

    def test_g_dot_i_dot_e_canonicalized(self):
        n, _ = normalize_name_basic("Dupont G.I.E")
        assert canonicalize_name(n) == "dupont gie"

    def test_existing_sarl_still_works(self):
        n, _ = normalize_name_basic("Dupont SARL")
        assert canonicalize_name(n) == "dupont sarl"

    def test_existing_sas_still_works(self):
        n, _ = normalize_name_basic("Dupont SAS")
        assert canonicalize_name(n) == "dupont sas"

    def test_eurl_token_boundary_safe(self):
        # "eurl" is unusual inside other words, but confirm boundary logic.
        n, _ = normalize_name_basic("Beurling Acoustics")
        canon = canonicalize_name(n)
        assert "beurling" in canon


# ---------------------------------------------------------------------------
# Extended ADDRESS_ABBREV_MAP — US additions
# ---------------------------------------------------------------------------

class TestExtendedAddressAbbrevUS:
    def test_expy_to_expressway(self):
        n, _ = normalize_address_basic("100 Northern Expy, Atlanta, GA")
        assert "expressway" in canonicalize_address(n)

    def test_fwy_to_freeway(self):
        n, _ = normalize_address_basic("500 Harbor Fwy, Los Angeles, CA")
        assert "freeway" in canonicalize_address(n)

    def test_tpke_to_turnpike(self):
        n, _ = normalize_address_basic("1 Garden State Tpke, NJ")
        assert "turnpike" in canonicalize_address(n)

    def test_tpk_to_turnpike(self):
        n, _ = normalize_address_basic("1 Garden State Tpk, NJ")
        assert "turnpike" in canonicalize_address(n)

    def test_xing_to_crossing(self):
        n, _ = normalize_address_basic("10 River Xing, Portland, OR")
        assert "crossing" in canonicalize_address(n)

    def test_expy_token_boundary_safe(self):
        # "expy" is unlikely inside a real word, but confirm no substring fire.
        n, _ = normalize_address_basic("Expressway Business Park, Dallas")
        canon = canonicalize_address(n)
        assert "expressway" in canon   # already spelled out — unchanged


# ---------------------------------------------------------------------------
# Extended ADDRESS_ABBREV_MAP — French additions
# ---------------------------------------------------------------------------

class TestExtendedAddressAbbrevFrance:
    def test_bd_to_boulevard(self):
        n, _ = normalize_address_basic("12 Bd Haussmann, Paris")
        assert "boulevard" in canonicalize_address(n)

    def test_imp_to_impasse(self):
        n, _ = normalize_address_basic("3 Imp des Lilas, Lyon")
        assert "impasse" in canonicalize_address(n)

    def test_res_to_residence(self):
        n, _ = normalize_address_basic("5 Res du Parc, Marseille")
        assert "residence" in canonicalize_address(n)

    def test_bd_token_boundary_safe(self):
        # "bd" must NOT fire inside "abduct" etc. — test a safe word.
        n, _ = normalize_address_basic("abdallah street, paris")
        canon = canonicalize_address(n)
        assert "abdallah" in canon   # not mangled

    def test_res_token_boundary_safe(self):
        # "res" must NOT fire inside "restaurant" — it is only a standalone token.
        n, _ = normalize_address_basic("Restaurant Row, Nice")
        canon = canonicalize_address(n)
        assert "restaurant" in canon


# ---------------------------------------------------------------------------
# Extended ADDRESS_ABBREV_MAP — India additions
# ---------------------------------------------------------------------------

class TestExtendedAddressAbbrevIndia:
    def test_sec_to_sector(self):
        n, _ = normalize_address_basic("Plot 12, Sec 17, Chandigarh")
        assert "sector" in canonicalize_address(n)

    def test_sec_token_boundary_safe(self):
        # "sec" must NOT fire inside "second" or "section".
        n, _ = normalize_address_basic("Second Floor, Sector 21, Noida")
        canon = canonicalize_address(n)
        assert "second" in canon     # "second" left intact
        assert "sector" in canon     # "sec" → "sector" only for the abbreviation

    def test_sector_sec_pair_canonicalise_identically(self):
        n1, _ = normalize_address_basic("Plot 12, Sector 17, Chandigarh")
        n2, _ = normalize_address_basic("Plot 12, Sec 17, Chandigarh")
        assert canonicalize_address(n1) == canonicalize_address(n2)


# ---------------------------------------------------------------------------
# Pipeline-level regression: full clean_source still works end-to-end
# with all new cleaning steps active
# ---------------------------------------------------------------------------

class TestCleanSourceRegression:
    def test_accented_french_name_and_address(self):
        from src.preprocess import clean_source
        rows = [
            ["S1-10", "Société Café SARL", "12 Bd Haussmann, Paris 75009", "France"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        # Raw fields preserved
        assert out.loc[0, "business_name"] == "Société Café SARL"
        assert out.loc[0, "business_address"] == "12 Bd Haussmann, Paris 75009"
        # Normalised: accents folded, lowercase
        assert out.loc[0, "business_name_normalized"] == "societe cafe sarl"
        # Canonical: suffix mapped
        assert out.loc[0, "business_name_canonical"] == "societe cafe sarl"
        # Address canonical: bd → boulevard
        assert "boulevard" in out.loc[0, "business_address_canonical"]

    def test_dba_and_leet_name(self):
        from src.preprocess import clean_source
        rows = [
            ["S1-20", "Preparat0ry Academy dba Prep Co", "500 Harbor Fwy, LA, CA", "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, _ = clean_source(df, "source1", "test")
        norm = out.loc[0, "business_name_normalized"]
        assert "preparatory" in norm          # leet repaired
        assert "dba" not in norm              # DBA tag itself removed
        # The alias "Prep Co" is stripped; only the primary name remains.
        # "academy" is part of the primary name so it must survive.
        assert "academy" in norm
        assert "fwy" not in out.loc[0, "business_address_canonical"]
        assert "freeway" in out.loc[0, "business_address_canonical"]

    def test_url_in_name_stripped(self):
        from src.preprocess import clean_source
        rows = [
            ["S1-30", "Acme Corp www.acme.com", "100 Main St, Austin, TX", "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, _ = clean_source(df, "source1", "test")
        assert "www" not in out.loc[0, "business_name_normalized"]
        assert "acme" in out.loc[0, "business_name_normalized"]

    def test_extended_na_tokens(self):
        from src.preprocess import clean_source
        rows = [
            ["S1-40", "nil",            "123 Main St", "US"],
            ["S1-41", "not available",  "123 Main St", "US"],
            ["S1-42", "unknown",        "123 Main St", "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        assert stats["name_missing_count"] == 3

    def test_row_count_preserved_after_all_new_steps(self):
        from src.preprocess import clean_source
        rows = [
            ["S1-50", "Café EURL",            "12 Allée des Roses, Paris", "France"],
            ["S1-51", "4cme Corp",             "Sec 17, Chandigarh",       "India"],
            ["S1-52", "Global dba Local Inc",  "500 Tpke, NJ",             "US"],
            ["S1-53", "nil",                   "",                          "US"],
        ]
        df = pd.DataFrame(rows, columns=["entity_id", "business_name",
                                          "business_address", "country"])
        out, stats = clean_source(df, "source1", "test")
        assert len(out) == 4                      # no rows dropped
        assert stats["output_rows"] == 4
        assert set(out["entity_id"]) == {"S1-50", "S1-51", "S1-52", "S1-53"}
