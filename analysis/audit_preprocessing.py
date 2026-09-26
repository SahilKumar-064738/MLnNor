#!/usr/bin/env python3
"""Read-only forensic audit of the current preprocessing/normalization pipeline.

Uses ONLY training data (train_source1/2/3 + train_ground_truth). Never writes
to data/, never modifies the pipeline. Produces:
  - stdout progress + report
  - analysis/report/audit_log.txt   (progress log)
  - analysis/report/audit_results.json

Runtime target: a few minutes. All expensive transforms are computed once and
reused; per-row Python work is minimized.
"""
from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.normalization import (  # noqa: E402
    canonicalize_address_series,
    canonicalize_name_series,
    normalize_address_series,
    normalize_name_series,
)
from rapidfuzz import fuzz  # noqa: E402

SEED = 42
TRAIN = ROOT / "data" / "full" / "train"
OUT_DIR = ROOT / "analysis" / "report"
OUT_JSON = OUT_DIR / "audit_results.json"
OUT_LOG = OUT_DIR / "audit_log.txt"

REPORT: dict = {}
_log_fh = None


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _log_fh:
        _log_fh.write(line + "\n")
        _log_fh.flush()


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, na_values=[])


# ---------------------------------------------------------------------------
# Variant transforms (operate on pipeline "normalized" lowercase strings).
# Implemented as module-level regexes applied via .map (C-level re).
# ---------------------------------------------------------------------------
_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_WS2_RE = re.compile(r"\s+")
_NUM_TOKEN_RE = re.compile(r"^\d+([./-]\d+)*$")
_POSTAL_RE = re.compile(r"^\d{5,6}$")

SUFFIX_TOKENS = {
    "pvt", "ltd", "corp", "inc", "co", "llc", "llp", "pllc", "pc", "sa", "sas", "sci", "sarl",
    "company", "limited", "private", "corporation", "incorporated",
}


def aggressive(norm: str) -> str:
    if not norm:
        return ""
    return _WS2_RE.sub(" ", _PUNCT_RE.sub(" ", norm)).strip()


def no_digits(norm: str) -> str:
    if not norm:
        return ""
    return _WS2_RE.sub(" ", re.sub(r"\d", " ", norm)).strip()


def no_numeric_tokens(norm: str) -> str:
    if not norm:
        return ""
    return " ".join(t for t in norm.split() if not _NUM_TOKEN_RE.match(t.strip(".,")))


def no_suffix(norm: str) -> str:
    if not norm:
        return ""
    return " ".join(t for t in norm.split() if t.strip(".,") not in SUFFIX_TOKENS)


def sorted_tokens(norm: str) -> str:
    if not norm:
        return ""
    return " ".join(sorted(norm.split()))


def digit_tokens(text: str) -> list:
    return re.findall(r"\d+", text or "")


def postal_tokens(text: str) -> set:
    return {t for t in re.split(r"[^0-9]+", text or "") if _POSTAL_RE.match(t)}


# ---------------------------------------------------------------------------
# Collision analysis
# ---------------------------------------------------------------------------
def collision_stats(raw: pd.Series, variants: dict, top_n: int = 12) -> dict:
    """variants: name -> pd.Series aligned with `raw`. Non-empty values only."""
    out = {}
    for vname, s in variants.items():
        mask = s != ""
        s_ne = s[mask]
        raw_ne = raw[mask]
        n_nonempty = len(s_ne)
        sizes = s_ne.value_counts()
        coll = sizes[sizes > 1]
        top = []
        for key, size in coll.head(top_n).items():
            sample_raw = raw_ne[s_ne == key].drop_duplicates().head(5).tolist()
            top.append({"value": key, "size": int(size), "sample_raw": sample_raw})
        out[vname] = {
            "rows_nonempty": int(n_nonempty),
            "unique_values": int(sizes.shape[0]),
            "collapse_pct": round(100.0 * (1 - sizes.shape[0] / max(n_nonempty, 1)), 2),
            "collision_groups": int(coll.shape[0]),
            "rows_in_collisions": int(coll.sum()),
            "rows_in_collisions_pct": round(100.0 * coll.sum() / max(n_nonempty, 1), 2),
            "max_group_size": int(sizes.max()) if len(sizes) else 0,
            "top_groups": top,
        }
    return out


# ---------------------------------------------------------------------------
# Similarity machinery
# ---------------------------------------------------------------------------
def token_stats(a: str, b: str) -> tuple:
    ta, tb = set((a or "").split()), set((b or "").split())
    if not ta or not tb:
        return (0.0, 0.0)
    inter = len(ta & tb)
    return (inter / len(ta | tb), inter / min(len(ta), len(tb)))


def sim_record(a: dict, b: dict) -> dict:
    r = {}
    for side in ("name", "addr"):
        va_map, vb_map = a[f"{side}_variants"], b[f"{side}_variants"]
        for vname in va_map:
            va, vb = va_map[vname], vb_map[vname]
            key = f"{side}.{vname}"
            r[f"{key}.exact"] = 1.0 if (va == vb and va != "") else 0.0
            r[f"{key}.ratio"] = fuzz.ratio(va, vb) / 100.0 if (va or vb) else 0.0
            jac, cont = token_stats(va, vb)
            r[f"{key}.jaccard"] = jac
            r[f"{key}.contain"] = cont
        # token-set (order-insensitive) on normalized only
        va, vb = va_map["norm"], vb_map["norm"]
        r[f"{side}.norm.token_set"] = fuzz.token_set_ratio(va, vb) / 100.0 if (va or vb) else 0.0
    # address structural signals on RAW strings
    da, db = digit_tokens(a["addr_raw"]), digit_tokens(b["addr_raw"])
    if da or db:
        ca, cb = Counter(da), Counter(db)
        inter = sum((ca & cb).values())
        r["addr.digits.agreement"] = inter / max(sum(ca.values()), sum(cb.values()))
        r["addr.digits.any"] = 1.0
    else:
        r["addr.digits.agreement"] = np.nan
        r["addr.digits.any"] = 0.0
    pa, pb = postal_tokens(a["addr_raw"]), postal_tokens(b["addr_raw"])
    r["addr.postal.both_present"] = 1.0 if (pa and pb) else 0.0
    r["addr.postal.equal"] = (1.0 if pa & pb else 0.0) if (pa and pb) else np.nan
    r["country.equal"] = 1.0 if (a["country"] == b["country"] and a["country"] != "") else 0.0

    def nonascii(s):
        return any(ord(c) > 127 for c in s)
    r["name.script_mismatch"] = 1.0 if (nonascii(a["name_raw"]) != nonascii(b["name_raw"])) else 0.0
    r["addr.empty_either"] = 1.0 if (a["addr_raw"] == "" or b["addr_raw"] == "") else 0.0
    return r


def summarize_sims(records: list) -> dict:
    keys = sorted({k for rec in records for k in rec})
    out = {}
    for k in keys:
        vals = np.array([rec[k] for rec in records if k in rec], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            out[k] = {"n": 0}
            continue
        out[k] = {
            "n": int(len(vals)),
            "mean": round(float(vals.mean()), 4),
            "p05": round(float(np.percentile(vals, 5)), 4),
            "p25": round(float(np.percentile(vals, 25)), 4),
            "p50": round(float(np.percentile(vals, 50)), 4),
            "p75": round(float(np.percentile(vals, 75)), 4),
            "p95": round(float(np.percentile(vals, 95)), 4),
            "frac_ge_0.90": round(float((vals >= 0.90).mean()), 4),
            "frac_ge_0.95": round(float((vals >= 0.95).mean()), 4),
            "frac_eq_1.0": round(float((vals >= 0.999999).mean()), 4),
        }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    global _log_fh
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_fh = open(OUT_LOG, "w", encoding="utf-8")
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    log("Loading train sources ...")
    s1 = read_tsv(TRAIN / "train_source1.tsv")
    s2 = read_tsv(TRAIN / "train_source2.tsv")
    s3 = read_tsv(TRAIN / "train_source3.tsv")
    gt = read_tsv(TRAIN / "train_ground_truth.tsv")
    log(f"  loaded in {time.time()-t0:.1f}s: S1={len(s1)} S2={len(s2)} S3={len(s3)} GT={len(gt)}")

    # ---------------- 1. FULL-S1 name+address normalization + collisions ----
    log("Pipeline normalization on full S1 (name) ...")
    name_norm, name_missing = normalize_name_series(s1["business_name"])
    name_canon = canonicalize_name_series(name_norm)
    log(f"  name norm+canon done {time.time()-t0:.1f}s")
    log("Pipeline normalization on full S1 (address) ...")
    addr_norm, addr_missing = normalize_address_series(s1["business_address"])
    addr_canon = canonicalize_address_series(addr_norm)
    log(f"  address norm+canon done {time.time()-t0:.1f}s")

    log("Computing name variants (full S1) ...")
    name_variants = {
        "raw": s1["business_name"].str.strip(),
        "norm_current": name_norm,
        "canon_current": name_canon,
        "no_suffix": name_norm.map(no_suffix),
        "sorted_tokens": name_norm.map(sorted_tokens),
    }
    log("Computing address variants (full S1) ...")
    addr_variants = {
        "raw": s1["business_address"].str.strip(),
        "norm_current": addr_norm,
        "canon_current": addr_canon,
        "no_digits": addr_norm.map(no_digits),
        "no_numeric_tokens": addr_norm.map(no_numeric_tokens),
        "sorted_tokens": addr_norm.map(sorted_tokens),
    }
    log("Collision analysis on full S1 names ...")
    REPORT["s1_name_collisions"] = collision_stats(s1["business_name"], name_variants)
    log("Collision analysis on full S1 addresses ...")
    REPORT["s1_address_collisions"] = collision_stats(s1["business_address"], addr_variants)

    REPORT["s1_misc"] = {
        "rows": int(len(s1)),
        "name_missing": int(name_missing.sum()),
        "addr_missing": int(addr_missing.sum()),
        "country_distribution": s1["country"].value_counts().to_dict(),
    }

    # cache variant arrays for pair scoring (full S1)
    s1_feat_cache = {
        "name_raw": s1["business_name"], "addr_raw": s1["business_address"],
        "country": s1["country"],
        "name_norm": name_norm, "name_canon": name_canon,
        "name_nosuf": name_variants["no_suffix"], "name_sorted": name_variants["sorted_tokens"],
        "addr_norm": addr_norm, "addr_canon": addr_canon,
        "addr_nodig": addr_variants["no_digits"], "addr_sorted": addr_variants["sorted_tokens"],
    }
    s1_feat = pd.DataFrame(s1_feat_cache).set_index(s1["entity_id"])
    del name_variants, addr_variants, s1_feat_cache

    # ---------------- 2. Sampled S2/S3 collisions --------------------------
    S23_SAMPLE = 500_000
    for tag, df in (("s2", s2), ("s3", s3)):
        log(f"Sampling {S23_SAMPLE} rows from {tag} ...")
        sub = df.sample(n=min(S23_SAMPLE, len(df)), random_state=SEED)
        n_norm, _ = normalize_name_series(sub["business_name"])
        n_canon = canonicalize_name_series(n_norm)
        a_norm, _ = normalize_address_series(sub["business_address"])
        a_canon = canonicalize_address_series(a_norm)
        REPORT[f"{tag}_sample_name_collisions"] = collision_stats(sub["business_name"], {
            "norm_current": n_norm,
            "canon_current": n_canon,
            "no_suffix": n_norm.map(no_suffix),
            "sorted_tokens": n_norm.map(sorted_tokens),
        })
        REPORT[f"{tag}_sample_address_collisions"] = collision_stats(sub["business_address"], {
            "norm_current": a_norm,
            "canon_current": a_canon,
            "no_digits": a_norm.map(no_digits),
            "sorted_tokens": a_norm.map(sorted_tokens),
        })
        REPORT[f"{tag}_misc"] = {
            "sample_rows": int(len(sub)),
            "addr_empty_pct": round(100.0 * (sub["business_address"] == "").mean(), 2),
            "country_distribution": sub["country"].value_counts().to_dict(),
        }
        log(f"  {tag} done {time.time()-t0:.1f}s")
        del sub, n_norm, n_canon, a_norm, a_canon

    # ---------------- 3. Positive pairs from ground truth ------------------
    log("Exploding ground truth into pairs ...")
    gtx = gt.assign(m=gt["matched_entity_ids"].str.split(",")).explode("m")
    gtx = gtx[gtx["m"] != ""]
    n_pairs_total = len(gtx)
    N_POS = 100_000
    pos = gtx.sample(n=min(N_POS, n_pairs_total), random_state=SEED)
    log(f"  GT pairs total={n_pairs_total}, sampled={len(pos)}")

    need_s1 = set(pos["source1_entity_id"])
    need_s2 = set(pos.loc[pos["m"].str.startswith("S2-"), "m"])
    need_s3 = set(pos.loc[pos["m"].str.startswith("S3-"), "m"])

    s2f = s2[s2["entity_id"].isin(need_s2)].set_index("entity_id")
    s3f = s3[s3["entity_id"].isin(need_s3)].set_index("entity_id")
    log(f"  pair-touching rows: S1={len(need_s1)} S2={len(s2f)} S3={len(s3f)}")

    def build_variants(frame: pd.DataFrame) -> pd.DataFrame:
        nn, _ = normalize_name_series(frame["business_name"])
        nc = canonicalize_name_series(nn)
        an, _ = normalize_address_series(frame["business_address"])
        ac = canonicalize_address_series(an)
        out = pd.DataFrame(index=frame.index)
        out["name_raw"] = frame["business_name"]
        out["addr_raw"] = frame["business_address"]
        out["country"] = frame["country"]
        out["name_norm"] = nn
        out["name_canon"] = nc
        out["name_nosuf"] = nn.map(no_suffix)
        out["name_sorted"] = nn.map(sorted_tokens)
        out["addr_norm"] = an
        out["addr_canon"] = ac
        out["addr_nodig"] = an.map(no_digits)
        out["addr_sorted"] = an.map(sorted_tokens)
        return out

    log("Building variants for pair-touching S2/S3 rows ...")
    v2 = build_variants(s2f)
    v3 = build_variants(s3f)
    v1 = s1_feat.loc[s1_feat.index.intersection(list(need_s1))]
    log(f"  variants built {time.time()-t0:.1f}s")

    NAME_KEYS = ("name_raw", "name_norm", "name_canon", "name_nosuf", "name_sorted")
    ADDR_KEYS = ("addr_raw", "addr_norm", "addr_canon", "addr_nodig", "addr_sorted")
    NAME_VNAMES = ("raw", "norm", "canon", "nosuf", "sorted")
    ADDR_VNAMES = ("raw", "norm", "canon", "nodig", "sorted")

    def feat(vframe: pd.DataFrame, eid: str):
        if eid not in vframe.index:
            return None
        row = vframe.loc[eid]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return {
            "name_raw": row["name_raw"], "addr_raw": row["addr_raw"], "country": row["country"],
            "name_variants": {vn: row[k] for vn, k in zip(NAME_VNAMES, NAME_KEYS)},
            "addr_variants": {vn: row[k] for vn, k in zip(ADDR_VNAMES, ADDR_KEYS)},
        }

    log("Scoring positive pairs ...")
    pos_records = []
    for s1_id, m_id in zip(pos["source1_entity_id"], pos["m"]):
        vframe = v2 if m_id.startswith("S2-") else v3
        a = feat(v1, s1_id)
        b = feat(vframe, m_id)
        if a is None or b is None:
            continue
        pos_records.append(sim_record(a, b))
    log(f"  scored {len(pos_records)} positive pairs {time.time()-t0:.1f}s")

    # ---------------- 4. Negative pairs ------------------------------------
    log("Building random negative pairs (S1 x S2) ...")
    positive_set = set(zip(pos["source1_entity_id"], pos["m"]))
    s1_ids_arr = v1.index.to_numpy()
    s2_ids_arr = v2.index.to_numpy()
    neg_records = []
    target = 50_000
    attempts = 0
    while len(neg_records) < target and attempts < target * 4:
        batch = 40_000
        attempts += batch
        i1 = rng.integers(0, len(s1_ids_arr), batch)
        i2 = rng.integers(0, len(s2_ids_arr), batch)
        for x, y in zip(i1, i2):
            sid, tid = s1_ids_arr[x], s2_ids_arr[y]
            if (sid, tid) in positive_set:
                continue
            a = feat(v1, sid)
            b = feat(v2, tid)
            if a is None or b is None:
                continue
            neg_records.append(sim_record(a, b))
            if len(neg_records) >= target:
                break
    log(f"  scored {len(neg_records)} random negative pairs {time.time()-t0:.1f}s")

    log("Building hard negative pairs (distinct S1 ids sharing canonical name) ...")
    canon = v1["name_canon"]
    canon = canon[canon != ""]
    sizes = canon.value_counts()
    coll_names = sizes[sizes >= 2].index
    hard_records = []
    if len(coll_names) > 0:
        grouped = v1[v1["name_canon"].isin(coll_names)].groupby("name_canon")
        groups = [g for _, g in grouped if len(g) >= 2]
        if groups:
            gi = rng.integers(0, len(groups), min(30_000, len(groups) * 3))
            for k in gi:
                g = groups[k]
                x, y = rng.choice(len(g), size=2, replace=False)
                a = feat(v1, g.index[x])
                b = feat(v1, g.index[y])
                if a is None or b is None:
                    continue
                hard_records.append(sim_record(a, b))
                if len(hard_records) >= 30_000:
                    break
    log(f"  scored {len(hard_records)} hard negative pairs {time.time()-t0:.1f}s")

    # ---------------- 5. Summaries -----------------------------------------
    log("Summarizing similarity distributions ...")
    REPORT["positive_pair_similarity"] = summarize_sims(pos_records)
    REPORT["negative_random_similarity"] = summarize_sims(neg_records)
    REPORT["negative_hard_similarity"] = summarize_sims(hard_records)

    REPORT["meta"] = {
        "seed": SEED,
        "gt_total_pairs": int(n_pairs_total),
        "pos_pairs_scored": len(pos_records),
        "neg_random_scored": len(neg_records),
        "neg_hard_scored": len(hard_records),
        "s23_sample": S23_SAMPLE,
        "duration_seconds": round(time.time() - t0, 1),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(REPORT, f, indent=2, ensure_ascii=False, default=str)
    log(f"Wrote {OUT_JSON}")

    def brief(table: dict, keys: list) -> None:
        for k in keys:
            if k in table:
                v = table[k]
                print(f"    {k:32s} mean={v.get('mean')}  p50={v.get('p50')}  "
                      f">=0.95={v.get('frac_ge_0.95')}  ==1.0={v.get('frac_eq_1.0')}")

    print("\n=== POSITIVE PAIRS ===")
    brief(REPORT["positive_pair_similarity"], [
        "name.raw.ratio", "name.norm.ratio", "name.canon.ratio", "name.canon.exact",
        "name.norm.exact", "name.sorted.exact", "addr.raw.ratio", "addr.norm.ratio",
        "addr.canon.ratio", "addr.canon.exact", "addr.nodig.ratio",
        "addr.digits.agreement", "country.equal", "name.script_mismatch",
        "addr.empty_either",
    ])
    print("\n=== RANDOM NEGATIVES ===")
    brief(REPORT["negative_random_similarity"], [
        "name.canon.ratio", "name.canon.exact", "name.sorted.exact",
        "addr.canon.ratio", "addr.canon.exact", "country.equal",
    ])
    print("\n=== HARD NEGATIVES (same canonical name, distinct S1 entities) ===")
    brief(REPORT["negative_hard_similarity"], [
        "name.canon.exact", "addr.canon.ratio", "addr.canon.exact",
        "addr.norm.ratio", "addr.digits.agreement", "country.equal",
    ])

    print("\n=== S1 NAME COLLISIONS (full 2.2M rows) ===")
    for vname, st in REPORT["s1_name_collisions"].items():
        print(f"  {vname:22s} unique={st['unique_values']:>9}  collapse={st['collapse_pct']:>6}%  "
              f"groups={st['collision_groups']:>8}  rows_in_coll={st['rows_in_collisions_pct']}%")
    print("\n=== S1 ADDRESS COLLISIONS (full 2.2M rows) ===")
    for vname, st in REPORT["s1_address_collisions"].items():
        print(f"  {vname:22s} unique={st['unique_values']:>9}  collapse={st['collapse_pct']:>6}%  "
              f"groups={st['collision_groups']:>8}  rows_in_coll={st['rows_in_collisions_pct']}%")

    print(f"\nTotal audit duration: {time.time()-t0:.1f}s")
    _log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
