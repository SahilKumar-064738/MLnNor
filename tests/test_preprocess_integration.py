"""End-to-end integration test: writes a tiny 7-file sample dataset to disk,
runs the real CLI entrypoint (src.preprocess.main), and checks the outputs
and invariants -- exercising the same code path used for local and
SageMaker execution (implementation.md Section 26 item 9).
"""
import json
from pathlib import Path

import pandas as pd
import pytest

from src.preprocess import main


def _write_tsv(path: Path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, sep="\t", index=False)


@pytest.fixture()
def sample_dataset(tmp_path):
    input_dir = tmp_path / "input"

    # --- train ---
    _write_tsv(
        input_dir / "train" / "train_source1.tsv",
        [
            ["S1-1", "Callicoat & Dailey Inc", "3315 Fremont Street, Peoria, IL", "US"],
            ["S1-2", "Raj Investments LLP", "12 MG Road, Chennai", "India"],
            ["S1-3", "Primary Care Group", "1 Main St, Boston, MA", "US"],
            ["S1-4", "Primary Care Group", "9 Oak Ave, Denver, CO", "US"],
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(
        input_dir / "train" / "train_source2.tsv",
        [
            ["S2-1", "CALLICOAT & DAILEY INC", "3315 FREMONT ST, PEORIA, IL", "US"],
            ["S2-2", "-- Holloway Peak Inc Seafood", "067 PRODUCTION CT, NULL, INDEPENDENCE, KY", "US"],
            ["S2-3", "N/A", "", "US"],
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(
        input_dir / "train" / "train_source3.tsv",
        [
            ["S3-1", "Raj Investments LLP Tamil Variant", "12 MG Rd, Chennai", "India"],
            ["S3-2", "H.No.16, Opp.Rta Office", "H.No.16-11-23/37/A, Opp.Rta Office, Hyderabad", "India"],
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(
        input_dir / "train" / "train_ground_truth.tsv",
        [
            ["S1-1", "S2-1"],
            ["S1-2", "S3-1"],
            ["S1-3", ""],
            ["S1-4", ""],
        ],
        ["source1_entity_id", "matched_entity_ids"],
    )

    # --- test (includes an unseen country: France) ---
    _write_tsv(
        input_dir / "test" / "test_source1.tsv",
        [["S1-101", "Ecole Française SARL", "1 Rue de Paris", "France"]],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(
        input_dir / "test" / "test_source2.tsv",
        [["S2-101", "Ecole Francaise SARL", "1 Rue de Paris", "France"]],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(
        input_dir / "test" / "test_source3.tsv",
        [["S3-101", "Some Other Biz", "5 Elm St, Behind City Mall, Pune", "India"]],
        ["entity_id", "business_name", "business_address", "country"],
    )

    return input_dir


def test_end_to_end_pipeline_success(sample_dataset, tmp_path):
    output_dir = tmp_path / "output"
    exit_code = main([
        "--input-dir", str(sample_dataset),
        "--output-dir", str(output_dir),
        "--splits", "train", "test",
    ])
    assert exit_code == 0

    processed = output_dir / "processed"
    metadata = output_dir / "metadata"

    # Cleaned files exist for both splits.
    for split in ("train", "test"):
        for fname in ("source1_clean.tsv", "source2_clean.tsv", "source3_clean.tsv"):
            assert (processed / split / fname).exists()
    assert (processed / "train" / "ground_truth_clean.tsv").exists()
    assert not (processed / "test" / "ground_truth_clean.tsv").exists()

    # Report exists and is well-formed JSON.
    report_path = metadata / "preprocessing_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["status"] == "completed"

    # Row counts preserved exactly.
    s1_raw = pd.read_csv(sample_dataset / "train" / "train_source1.tsv", sep="\t", dtype=str)
    s1_clean = pd.read_csv(processed / "train" / "source1_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert len(s1_raw) == len(s1_clean)
    assert set(s1_raw["entity_id"]) == set(s1_clean["entity_id"])

    # Original fields preserved verbatim (auditability).
    assert s1_clean.loc[s1_clean["entity_id"] == "S1-1", "business_name"].iloc[0] == "Callicoat & Dailey Inc"

    # Canonicalization applied correctly.
    canon = s1_clean.loc[s1_clean["entity_id"] == "S1-1", "business_name_canonical"].iloc[0]
    assert canon == "callicoat and dailey incorporated"

    # Matched-pair address canonicalizes identically across S1/S2.
    s2_clean = pd.read_csv(processed / "train" / "source2_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
    addr1 = s1_clean.loc[s1_clean["entity_id"] == "S1-1", "business_address_canonical"].iloc[0]
    addr2 = s2_clean.loc[s2_clean["entity_id"] == "S2-1", "business_address_canonical"].iloc[0]
    assert addr1 == addr2

  
    # Null component removed, row not dropped, missing flags correct.
    assert (s2_clean.loc[s2_clean["entity_id"] == "S2-2", "business_address_is_missing"].iloc[0]
            in ("False", False))
    assert "null" not in s2_clean.loc[s2_clean["entity_id"] == "S2-2", "business_address_normalized"].iloc[0]
    assert (s2_clean.loc[s2_clean["entity_id"] == "S2-3", "business_name_is_missing"].iloc[0]
            in ("True", True))
    assert (s2_clean.loc[s2_clean["entity_id"] == "S2-3", "business_address_is_missing"].iloc[0]
            in ("True", True))

    # No row was dropped for source2 despite missing name/address.
    assert len(s2_clean) == 3

    # Landmark preserved and extracted.
    s3_clean = pd.read_csv(processed / "train" / "source3_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
    landmark = s3_clean.loc[s3_clean["entity_id"] == "S3-2", "address_landmark"].iloc[0]
    assert landmark != ""
    assert "opp" in landmark.lower()

    # Unseen country (France, test-only) handled correctly, no crash, no
    # hardcoded rejection.
    test_s1_clean = pd.read_csv(processed / "test" / "source1_clean.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert test_s1_clean.loc[test_s1_clean["entity_id"] == "S1-101", "country_normalized"].iloc[0] == "france"

    # Franchise-style collision: same canonical name, different address ->
    # reported as a name-only collision, but NOT collapsed/removed (both
    # rows still present).
    assert (s1_clean["entity_id"].isin(["S1-3", "S1-4"])).sum() == 2
    collision_file = metadata / "canonicalization_collisions_train_business_name_canonical.tsv"
    assert collision_file.exists()
    collisions = pd.read_csv(collision_file, sep="\t")
    assert len(collisions) >= 1  # "primary care group" collision group present


def test_pipeline_fails_loudly_on_duplicate_entity_id(tmp_path):
    input_dir = tmp_path / "input"
    _write_tsv(
        input_dir / "train" / "train_source1.tsv",
        [
            ["S1-1", "A", "1 Main St", "US"],
            ["S1-1", "B", "2 Main St", "US"],  # duplicate id
        ],
        ["entity_id", "business_name", "business_address", "country"],
    )
    _write_tsv(input_dir / "train" / "train_source2.tsv", [], ["entity_id", "business_name", "business_address", "country"])
    _write_tsv(input_dir / "train" / "train_source3.tsv", [], ["entity_id", "business_name", "business_address", "country"])
    _write_tsv(input_dir / "train" / "train_ground_truth.tsv", [], ["source1_entity_id", "matched_entity_ids"])

    output_dir = tmp_path / "output"
    exit_code = main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--splits", "train",
    ])
    assert exit_code == 1
    # No processed output should be left for a failed run's report file.
    assert not (output_dir / "metadata" / "preprocessing_report.json").exists()


def test_pipeline_fails_loudly_on_missing_input_file(tmp_path):
    input_dir = tmp_path / "input"
    (input_dir / "train").mkdir(parents=True)
    # Deliberately do not write any of the required files.
    output_dir = tmp_path / "output"
    exit_code = main([
        "--input-dir", str(input_dir),
        "--output-dir", str(output_dir),
        "--splits", "train",
    ])
    assert exit_code == 1
