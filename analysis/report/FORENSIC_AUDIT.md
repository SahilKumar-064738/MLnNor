# Forensic Audit — Preprocessing / Normalization Pipeline
Business Entity Resolution challenge · Training-data-only evidence · 2026-09-25

All numbers below were computed by `analysis/audit_preprocessing.py` from the
training files only (`train_source1/2/3.tsv`, `train_ground_truth.tsv`).
Raw results: `analysis/report/audit_results.json`. Seed=42.
Positive pairs: 100,000 sampled from 7,638,365 ground-truth pairs.
Random negatives: 50,000 (S1×S2). Hard negatives: 17,664 (distinct S1 entities
sharing the same `business_name_canonical`).

---

## 0. Executive diagnosis

**The pipeline is conservative and safe — and that is both its strength and its
limitation.** It destroys almost no information (good), but it produces only
**two derived views per field** (`_normalized`, `_canonical`). The measured
similarity distributions show that no single lexical view gets close to
separating positives from negatives by itself; the separation lives in
*combinations* of views (token sets, sorted components, digit agreement,
postal codes) that the pipeline does not yet expose.

Headline evidence:

| Signal | Positives (n=100k) | Hard negatives (n=17.7k) | Random neg (n=50k) |
|---|---|---|---|
| `name.canon` exact | 19.3% | **100%** (by construction) | 0% |
| `name.canon` ratio | 0.780 | **1.000** | 0.311 |
| `addr.canon` ratio | 0.763 | 0.381 | 0.320 |
| `addr.norm` token-set | 0.854 | 0.386 | 0.322 |
| `addr.sorted` ratio | **0.806** | 0.380 | 0.319 |
| `addr.digits` agreement | **0.620** | **0.015** | 0.006 |
| postal code equal (when both present) | 93.7% (n=5,454) | 0% (n=93) | 0% (n=254) |
| country equal | 100% | 99.8% | 52.0% |

**The single most important structural fact:** for hard negatives — the pairs
that will decide your F_0.5, because F_0.5 punishes false merges — name
similarity is *saturated at 1.0* and is therefore worthless. The entire
discriminative burden falls on **address tokens, address component order,
and digits**. The current pipeline preserves those (good) but does not expose
them as features (the gap).

Three further hard facts:
1. **13.9% of true positive pairs have a script mismatch** (Latin ↔ Devanagari/
   Tamil/Telugu/etc.). No lexical normalization can ever close this. It must be
   handled by multilingual embeddings in the second pipeline; preprocessing's
   job is to *preserve both scripts and flag the mismatch*, not to transliterate.
2. **Only 5.45% of positive pairs have a postal code on both sides**, but when
   both are present, equality is 93.7%-precise. Postal codes are rare but
   near-golden — they must be extracted, never deleted, never required.
3. **Address digits agree in 62% of positives but only 1.5% of hard negatives.**
   Digits are the best single separator in the dataset. Any transform that
   removes digits is catastrophic (measured: 20.4% address collapse, largest
   collision group 1,390 rows — `"street, brooklyn, ny"`).

**On the 99.2% F_0.5 target:** nothing in this audit supports or refutes it.
Given 13.9% cross-script positives, 4.35% of positives with an empty address on
one side, and only 16–19% exact-name-match rate after normalization, the
ceiling is set by the second pipeline (candidate generation + multilingual
scoring + a precision-biased decision rule), not by preprocessing.
Preprocessing's contribution is: (a) do not destroy the address/digit signal,
(b) expose multi-view features so the scorer can combine them, (c) keep
collision-prone views out of any exact-match or blocking-without-address role.

---

## PART 1 — The current transformation graph (reconstructed)

```
raw TSV (dtype=str, keep_default_na=False)                     [preprocess.py:113-120]
 → schema + entity-id validation                                [validation.py]
 → business_name:
     NA-like whole-field token → "" + is_missing                [normalization.py:80-94]
     NFKC → lowercase → strip DECORATIVE_LEADING_CHARS → collapse WS
        → business_name_normalized
     "&"→" and " + LEGAL_SUFFIX_MAP (token-boundary)
        → business_name_canonical                               [normalization.py:97-113]
 → business_address:
     split on "," → drop empty + null-like components
     NFKC → lowercase → collapse WS
        → business_address_normalized                           [normalization.py:136-156]
     ADDRESS_ABBREV_MAP (token-boundary per component)
        → business_address_canonical                            [normalization.py:159-177]
     LANDMARK_PREFIXES regex on RAW address (word-boundary)
        → address_landmark (copied, never removed)              [normalization.py:116-133]
 → country: NFKC → lowercase → collapse WS (open-set, no alias table)
        → country_normalized                                    [normalization.py:180-192]
 → diagnostics (collisions report only, never deletes)          [diagnostics.py]
 → row-count + id-set invariants → cleaned TSV
```

### Transformation-by-transformation table

| TRANSFORMATION | CURRENT BEHAVIOR | BENEFIT | RISK | RECOMMENDATION | PRIORITY |
|---|---|---|---|---|---|
| TSV read `keep_default_na=False` | Literal "NA"/"NULL" never becomes NaN | Correct NA handling downstream | None | Retain | — |
| NFKC | Folds compatibility chars; does NOT transliterate | Correct scope | None measurable | Retain | — |
| Lowercase | name/address/country | Removes S2 ALL-CAPS noise (63% of S2 rows) | None (raw kept) | Retain | — |
| Decorative **leading**-char strip (`-#[]<>@.*\s`) | Names only, leading only | Measured: merges "-- Primary Care"→"primary care" dupes (S2 sample top collision groups) | Trailing decoration NOT stripped (unmeasured residual noise) | Keep; add symmetric trailing strip after verifying volume | P2 |
| Whitespace collapse | All fields | Fixes ~550k double-space rows/file | None | Retain | — |
| NA-like → missing flag | Whole-field names only | 6–61 rows/file; correct | Address component "NA"/"na" kept as literal token (see `"979, na, vikas puri"` collision group) — harmless but noisy | Keep; optionally flag component-level NA | P2 |
| Null-component strip (address) | Removes comma components matching `null`/`<null>` | ~130k rows/file; prevents literal "null" tokens | None measured; note `"redwood rd, null, ..."` merges with `"redwood rd, ..."` (observed, group of 4) — correct merge | Retain | — |
| LEGAL_SUFFIX_MAP (pvt→private, ltd→limited, corp→corporation, inc→incorporated, co→company) | Canonical only, token-boundary | Positive exact-match 16.0%→19.3% (+3.3pp) | Incomplete map: `llc/llp/pllc/pc/sarl/sas/sci` untouched, so "Meridian LLC" vs "Meridian Limited" diverge; `co→company` could touch non-suffix "Co" tokens (rare) | Retain, **extend map symmetrically** (map all forms to one canonical per family, incl. French forms), never delete suffixes | P1 |
| `&`→`and` (canonical) | Pre-tokenization replace | Merges "Ear Nose & Throat" variants (observed group 251→same) | A business literally named "A & B" vs "A and B" — desired merge | Retain | — |
| ADDRESS_ABBREV_MAP (rd, st, ave, ste, fl) | Canonical only, token-boundary | Positive addr exact 7.6%→9.5% (+1.9pp); observed RD/ROAD, ST/STREET, AVE/AVENUE merges | Map too small (dr/ct/ln/blvd/pl/cir/ter/hwy/pkwy absent); `st`↔"saint" ambiguity unaddressed (verify city names first) | Extend map after data check; keep token-boundary | P1 |
| Landmark extraction | Word-boundary regex on raw; copied to `address_landmark`, never deleted | Preserves locality signal ("near fortis hospital") | Regex finds first match only; phrase cut at next comma; output not lowercased/normalized | Retain; normalize the extracted phrase too; keep raw in address | P2 |
| Country normalization (open-set, no alias table) | NFKC+lower only | France in test handled by construction | None — correct decision | Retain exactly as-is | — |
| Collision diagnostics | Report-only, bounded memory | Evidence base for this audit | None | Retain; extend to S2/S3 | P2 |

---

## PART 2 — Information-loss audit

Measured collapse of S1 (2,206,821 rows) under various transforms:

### Names
| Variant | Unique | Collapse | Collision groups | Rows in collisions | Max group |
|---|---|---|---|---|---|
| raw | 1,539,229 | 30.25% | 177,793 | 38.31% | 253 |
| norm_current | 1,538,800 | 30.27% | 178,082 | 38.34% | 253 |
| canon_current | 1,515,051 | 31.35% | 179,963 | 39.50% | 253 |
| **no_suffix** | **1,349,975** | **38.83%** | **200,215** | **47.90%** | **499** |
| sorted_tokens | 1,536,893 | 30.36% | 179,266 | 38.48% | 253 |

### Addresses
| Variant | Unique | Collapse | Groups | Rows in coll. | Max group |
|---|---|---|---|---|---|
| raw | 2,130,606 | 3.45% | 40,089 | 5.27% | 14 |
| norm_current | 2,130,606 | 3.45% | 40,089 | 5.27% | 14 |
| canon_current | 2,130,528 | 3.46% | 40,163 | 5.28% | 14 |
| **no_digits** | **1,756,008** | **20.43%** | **193,512** | **29.20%** | **1,390** |
| **no_numeric_tokens** | **1,795,186** | **18.65%** | **195,791** | **27.52%** | **1,392** |
| sorted_tokens | 2,124,818 | 3.72% | 42,902 | 5.66% | 14 |

### Dangerous-transform examples (constructed from measured groups)

1. **Digit removal (addresses).** Real measured group: removing digits turns
   `"1447 59 Street, Brooklyn, NY"`, `"1035 85 Street, Brooklyn, NY"`,
   `"1721 86 Street, Brooklyn, NY"` … into the single key
   `"street, brooklyn, ny"` — **1,390 distinct businesses**. Your synthetic
   example (`12 MG Road` vs `120 MG Road`) is exactly this failure at unit
   scale. **Never remove digits.** Expose them as a separate multiset feature.

2. **Suffix removal (names).** Measured: `"Meridian"`, `"Meridian Inc"`,
   `"Meridian LLC"`, `"Meridian Inc."`, `"Meridian PLLC"` are 5 of **499**
   distinct S1 entities collapsing to `"meridian"`. Suffix removal takes
   rows-in-collision from 39.5% to 47.9%. **Never delete suffixes; split them
   off as a separate field** (`name_core` + `name_legal_suffix`).

3. **Sorted name tokens.** Only +0.11pp collapse vs norm (30.36% vs 30.27%) —
   cheap, and positive sorted-exact is 20.3% vs canon-exact 19.3%, so token
   reordering in names is real but small. Safe as a *secondary scoring view*,
   never as a blocking key alone.

4. **Sorted address tokens.** +0.27pp collapse (3.72%) but positive ratio jumps
   0.763→0.806 (frac ≥0.95: 0.186→0.208) because component order varies across
   sources (`"Tucson, 6249 Eagles Roost Drive, AZ"` vs `"6249 Eagles Roost
   Drive, Tucson, AZ"` — measured group). Safe and useful as a secondary view.

5. **Stopword removal.** Not in the pipeline — keep it out. "Department of",
   "Office of", "Commission on" are real (truncated) names in the data;
   removing function words would merge them further.

6. **Punctuation stripping inside tokens.** Not in the pipeline (only
   token-boundary trim of `.`/`,` for map lookup). Correct: `"204-C"`,
   `"E.C.S - 1634/11"`, `"19 1/2"` carry identity. Keep.

7. **Transliteration.** Not attempted — correct. 13.9% of positives need it;
   that is an embeddings-stage job.

**Conclusion:** the current pipeline introduces essentially no new collisions
(canon adds only +1.1pp name collapse, and that extra collapse is *desired*
legal-form merging, evidenced by +3.3pp positive exact matches). The risk
table is about what you must **not add**, plus one thing you must **add
carefully** (suffix splitting).

---

## PART 3 — Name normalization recommendations

Measured positive-pair performance of each name view (n=100k GT pairs):

| View | ratio mean | exact | ≥0.95 | Notes |
|---|---|---|---|---|
| raw | 0.698 | 4.6% | 16.7% | baseline; case/punct noise |
| norm (current) | 0.788 | 16.0% | 25.5% | big win from case+WS+decoration |
| canon (current) | 0.780 | 19.3% | 28.3% | suffix/`&` merge adds +3.3pp exact |
| sorted tokens | 0.779 | 20.3% | 27.7% | small reorder win |
| **no_suffix** | **0.809** | **35.1%** | **40.9%** | biggest recall lever, but collision-prone |
| token-set (norm) | 0.848 mean | — | 54.5% ≥0.95 | best single lexical signal |

Recommended name representations (each justified by a measured effect):

| Field | Definition | Use downstream | Justification |
|---|---|---|---|
| `business_name` (raw) | untouched | audit, embeddings input, tie-breaks | never destroy |
| `business_name_normalized` | current (NFKC/lower/leading-decor/WS) | exact-match pass, blocking with address | +11pp exact vs raw |
| `business_name_canonical` | current + **extended, symmetric suffix map** + `&`→`and` | exact-match pass | +3.3pp exact; extend to llc/llp/pllc/pc/sarl/sas/sci families |
| `business_name_core` | canonical minus trailing legal-suffix tokens (suffixes **recorded**, not dropped) | recall scoring, embedding text | nosuf exact = 35.1% on positives |
| `business_name_legal_suffix` | canonical suffix token(s), canonicalized per family ("" if none) | feature; also a weak country/legal-system signal | preserves the info nosuf destroys |
| `business_name_tokens` | token list of normalized | token-set/Jaccard scoring | token-set is best lexical signal (0.848 mean) |
| `business_name_sorted_tokens` | sorted token list | reorder-robust scoring/blocking-secondary | +1pp exact vs canon |
| `name_script_class` | latin / devanagari / tamil / … / mixed | routing to multilingual embeddings; flag cross-script pairs | 13.9% of positives are cross-script |
| `name_is_missing` | current | decision rule | already exists |

**Do not add:** phonetic codes (Soundex/Metaphone) — no evidence they beat
token-set + ratio here, and they are script- and language-specific (useless on
Devanagari/Tamil); stemming/lemmatization (would merge "Pediatric Dentistry"
into "Pediatric Dental"-style collisions, which are already distinct
businesses); char-ngram columns in TSV (compute n-gram similarity at scoring
time from the normalized string — RapidFuzz token_set/WRatio already captures
most of it without storing the grams).

---

## PART 4 — Address normalization recommendations

Measured positive-pair performance (n=100k):

| View | ratio mean | exact | ≥0.95 |
|---|---|---|---|
| raw | 0.562 | 2.2% | 3.0% |
| norm (current) | 0.756 | 7.6% | 16.5% |
| canon (current) | 0.763 | 9.5% | 18.6% |
| **sorted tokens** | **0.806** | 8.4% | **20.8%** |
| token-set (norm) | 0.854 mean | — | 33.4% ≥0.95 |
| digits agreement | 0.620 (=1.0) | — | 62.0% |
| postal equal (both present) | 93.7% precision | — | n=5,454 |

Component signal classification:

| Component | Signal class | Evidence |
|---|---|---|
| Digits (house/plot/ward numbers) | **HIGH — never strip** | 62% pos agreement vs 1.5% hard-neg; digit removal ⇒ 20% collapse |
| Postal/PIN code | **HIGH when present, rare** | both-present 5.45%; equality 93.7% precise on positives, 0% on negatives |
| Street name tokens | HIGH | token-set 0.854 mean on positives |
| City/locality | HIGH | present in ~all addresses; part of every top collision group's identity |
| State (full vs abbrev: `Texas`/`TX`, `Karnataka`/`KA`) | MEDIUM — needs a normalization table | cross-source state-name variation observed in sample groups |
| Street type (Rd/Road…) | MEDIUM | canon map works (+1.9pp exact); extend map |
| Unit/floor/suite | MEDIUM — preserve as tokens, never drop | `"th street, unit , washington, dc"` shows unit numbers carry identity |
| Landmark phrases | MEDIUM (India), LOW (US) | preserved today; keep as separate feature |
| Component order | LOW — must be tolerated | sorted view +0.043 ratio on positives for +0.27pp collapse |
| `NA`/`null` components | NOISE — strip/flag | already handled for null; "NA" components kept (harmless) |

Recommended address representations:

| Field | Definition | Use |
|---|---|---|
| `business_address` (raw) | untouched | audit, embeddings |
| `business_address_normalized` | current | baseline view |
| `business_address_canonical` | current + extended abbrev map (dr, ct, ln, blvd, pl, cir, ter, hwy, pkwy, sq — after verifying no city-name conflicts; check "st" vs "saint" against city tokens before mapping) | exact-match pass |
| `address_tokens` | token list of canonical | token-set/Jaccard scoring |
| `address_sorted_tokens` | sorted token list | order-robust scoring (+0.043 ratio, +2.2pp ≥0.95) |
| `address_numbers` | **ordered multiset of all digit-runs** (e.g. `["204","c"]`-aware: extract maximal digit sequences) | digit-agreement feature — best hard-negative separator measured |
| `address_postal_code` | extracted candidate postal/PIN tokens (5-6 digit runs, position-agnostic, country-aware length: US 5, IN 6, FR 5) | high-precision feature; equality ≠ required |
| `address_components` | comma-split component list (normalized) | component-level Jaccard/containment |
| `address_landmark` | current, plus lowercase the extracted phrase | feature |
| `address_is_missing` | current | decision rule |

**Weak parsing, not full parsing:** do not attempt a full address parser.
The measured structure is too inconsistent (reordered components, missing
commas, `"IA, Iowa City, 1064 Newton Rd, Unit 11"`). Component-list + digit
multiset + postal extraction give 90% of the value of parsing with none of
the parser failure modes.

---

## PART 5 — Country recommendations

Current open-set handling is correct and should not change:
- Positives: country equal 100% (training).
- Random negatives: equal 52% (reflects class mix) — country alone separates
  only half of random pairs.
- Hard negatives: equal 99.8% — **country cannot separate the pairs that
  matter most.**

Rules for the second pipeline:
1. Keep `country` raw + `country_normalized` (NFKC/lower) — already done.
2. Never hard-code {US, India}; France proves the set is open. No alias table
   (correctly avoided today).
3. Use country as a **blocking partition and a scoring feature**, never as an
   absolute veto: a same-business pair with conflicting country values is
   possible via data error, but treat country mismatch as a strong negative
   feature (weight learned in stage 2), and never generate cross-country
   candidates in blocking (recall cost is ~0 in training; revisit only if
   stage-2 validation shows cross-country positives).
4. Optionally derive `country_inferred_from_address` later in stage 2 (e.g.
   state abbrev heuristics) — out of scope for preprocessing.

---

## PART 6/7 — What the pair analysis says about each transformation

| Transformation | Effect on positives | Effect on negatives | Net verdict |
|---|---|---|---|
| Case+WS+decor (current `_normalized`) | name exact 4.6%→16.0%; addr exact 2.2%→7.6% | random neg unchanged (~0 exact) | **Strongly keep** |
| Suffix canonicalization (`_canonical`) | +3.3pp name exact | negligible new neg collisions | **Keep + extend map** |
| `&`→`and` | merges observed variants | none measured | Keep |
| Abbrev map | +1.9pp addr exact | merges only true abbrev pairs | **Keep + extend** |
| Null-component strip | removes noise token | enables `"redwood rd"` merge (correct) | Keep |
| Sorted address tokens | ratio 0.763→0.806 | +0.27pp collapse only | **Add as view** |
| Token-set similarity inputs | best single lexical signal (0.848) | 0.322 on random neg | **Expose token views** |
| Digit removal | would lose 62%-agreement signal | creates 1,390-row collision groups | **Forbidden** |
| Suffix deletion | +16pp exact on positives | 47.9% of rows in collisions; "meridian" ×499 | **Forbidden as deletion; allowed as split** |
| Stopword/punct stripping | no measured benefit | collision risk | Do not add |
| Transliteration | would fix 13.9% of positives | n/a | Embeddings stage, not preprocessing |

---

## PART 8 — Collision analysis summary

- Current canonical views are safe: S1 name canon collapse 31.35% (39.5% rows
  in collision groups) — but these collisions are *legitimate same-name
  different businesses* (253× "Primary Care Group"), which is why name must
  never be a standalone match key. This is a property of the data, not of your
  normalization: **raw** names already collide at 30.25%.
- Hard-negative construction confirms: 17,664 S1-pair comparisons share an
  identical canonical name and are distinct entities. 84% of them also share
  an identical `business_name_normalized`. Your canonical field is not
  creating this problem; it inherits it.
- Address canonical collisions are tiny (max group 14) and mostly real
  duplicates (same address, different businesses in the same building) —
  fine for blocking, never sufficient alone for merging.
- Diagnostics currently run on S1 only; extend the collision report to S2/S3
  (cheap, already sampled evidence: S2 sample canon name groups up to 46).

---

## PART 9 — Recommended output schema

```
entity_id
country                      (raw)
country_normalized           (open-set, as today)

business_name                (raw)
business_name_normalized     (as today)
business_name_canonical      (extended symmetric suffix map, & → and)
business_name_core           (canonical minus trailing legal suffixes)
business_name_legal_suffix   (canonicalized suffix family or "")
business_name_tokens         (" "-joined, dedup-NO — keep duplicates)
business_name_sorted_tokens
name_script_class            (latin|devanagari|tamil|telugu|gujarati|gurmukhi|malayalam|mixed|other)
business_name_is_missing

business_address             (raw)
business_address_normalized  (as today)
business_address_canonical   (extended abbrev map)
address_tokens               (" "-joined canonical tokens)
address_sorted_tokens
address_components           ("|"-joined normalized comma components)
address_numbers              (","-joined digit runs in original order)
address_postal_code          (best candidate or "")
address_landmark             (as today, lowercased)
business_address_is_missing
```

Per-field downstream role:

| Field | Blocking | Pair scoring | Final decision |
|---|---|---|---|
| country_normalized | partition key | feature (strong negative if mismatch) | veto-level feature (learned weight) |
| name_canonical / name_core | candidate gen (with address co-key) | exact-match shortcut | exact canonical+address ⇒ high-confidence accept |
| name_tokens / sorted | — | token-set, Jaccard, containment | input to threshold |
| name_script_class | route cross-script pairs to embedding scorer | — | — |
| addr_canonical | co-key with name | exact-match shortcut | high-confidence accept |
| address_tokens / sorted / components | — | token-set, component Jaccard | main scoring mass |
| address_numbers | — | digit-agreement feature | **key hard-negative separator** |
| address_postal_code | optional tight block when present | near-golden feature when both present | strong accept evidence, never veto on absence |
| *_is_missing | — | missingness features | down-weight address rules when missing |

---

## PART 10 — Ablation plan (for the second pipeline; reproducible)

Evaluation harness first: macro-F_0.5 on training via a held-out split of S1
entities (e.g. hash `entity_id` → 80/20), ground truth restricted to the holdout.
Same negative sampling as this audit. Fixed seed. Every row below changes exactly one thing.

| # | Change | Expected effect | Metric to watch | Failure mode |
|---|---|---|---|---|
| A0 | Baseline: current cleaned output + stage-2 scorer (whatever it is) | reference | macro-F_0.5, P, R | — |
| A1 | + `address_numbers` + digit-agreement feature | ↑ precision on hard negatives | F_0.5, false-merge count on hard-neg set | digit OCR noise lowers recall — check pos digit-agreement stays weighted softly |
| A2 | + `address_sorted_tokens` / component Jaccard | ↑ recall on reordered addresses (~+2pp ≥0.95 positives) | recall at fixed precision | over-merging same-street different-number — guard with digit feature |
| A3 | + `business_name_core`/`_legal_suffix` split (replace any suffix deletion) | ↑ recall on suffix variants (+16pp exact in core) without collision blowup | collisions in blocking; F_0.5 | core used alone as match key ⇒ meridian-style merges |
| A4 | + extended suffix + abbrev maps | +1–3pp exact matches | exact-match rate on positives | map error (st↔saint) — validate against city tokens first |
| A5 | + `address_postal_code` feature | ↑ precision where present (5.4% of pairs) | precision; coverage | treating absence as mismatch ⇒ recall loss |
| A6 | + name token-set & containment features | ↑ recall on partial names | recall | generic-name containment (e.g. "care" ⊂ "primary care") — require address support |
| A7 | + script_class routing + multilingual embedding for cross-script pairs | addresses the 13.9% unreachable today | recall on cross-script subset | embedding false positives — precision-biased threshold |
| A8 | + country-mismatch penalty | ↑ precision | F_0.5 | rare true cross-country records — inspect before hard-veto |

---

## PART 11 — Computational efficiency

Measured timings (this machine, pandas 3.0.6, single process):
- Loading 4 train files: 155s.
- `normalize_name_series` + `canonicalize_name_series` on 2.2M rows: 254s.
- Address norm+canon on 2.2M rows: 414s (per-row Python in
  `normalize_address_series` — the bottleneck, `normalization.py:228-240`).
- Per-pair scoring at ~4.5k pairs/s (RapidFuzz) — 100k pairs = 22s; pair
  scoring will dominate stage 2, not preprocessing.

Recommendations (quality-preserving):
1. **Unique-value caching:** apply every per-string transform to
   `Series.unique()` then map back. S1 names: 2.21M rows → 1.54M unique
   (−30% work); addresses less (−3.5%); S2/S3 names gain more via ALL-CAPS
   dedup. Trivial, safe, ~20–40% speedup on name stages.
2. **Pre-compile + vectorize:** the per-row list comprehension in
   `normalize_address_series` can be replaced by vectorized comma-split via
   `str.split(",")` + explode/filter/group — or keep the comprehension but run
   it over uniques (item 1).
3. **NFKC once:** NFKC is applied per-value in a Python loop; after
   unique-caching this is acceptable. Do not drop NFKC.
4. **Multiprocessing:** `ProcessPoolExecutor` over row-chunks for the address
   pipeline (embarrassingly parallel); expected ~N_cores× on the 414s step.
5. **DuckDB/Polars:** optional; the workload fits in pandas+pyarrow
   (installed). If stage 2 does large joins (S1×candidates), DuckDB is the
   right tool there, not here.
6. **Precompute once, reuse:** all PART-9 schema columns are deterministic
   pure functions — compute once per split, persist as Parquet/TSV, and let
   stage 2 read them. Never recompute normalization inside the scoring loop.
7. Keep collision diagnostics bounded (already are).

---

## PART 12 — Deliverables summary

### Exact weaknesses in the current pipeline
1. Only 2 views per field; no token/sorted/digit/postal/suffix-split/script
   views (the features the measured separation actually lives in).
2. `LEGAL_SUFFIX_MAP` incomplete and asymmetric (llc/llp/pllc/pc/sarl/sas/sci
   untouched) — leaves recall on the table for suffix-variant pairs.
3. `ADDRESS_ABBREV_MAP` only 5 entries; dr/ct/ln/blvd/pl/cir/ter/hwy/pkwy
   missing (each is an observed cross-source variation pattern).
4. Trailing decorative characters not stripped (leading only).
5. `address_landmark` emitted un-normalized (raw case/punct).
6. Collision diagnostics on S1 only.
7. Speed: per-row Python in address normalization (414s/2.2M rows); no
   unique-caching.
8. Nothing harmful: no destructive transform exists in the pipeline today.
   The audit found no operation that should be *removed*.

### Priority list
- **P0 (major impact, enabling stage-2 quality):**
  1. Add `address_numbers` + digit-agreement feature (best measured
     hard-negative separator: 0.62 vs 0.015).
  2. Add token views (`name_tokens`, `address_tokens`, `address_components`)
     — token-set is the best lexical signal (0.848 vs 0.322).
  3. Add `address_sorted_tokens` (order robustness: +0.043 ratio positives,
     +2.2pp ≥0.95, for +0.27pp collapse).
  4. Add `name_script_class` so stage 2 can route the 13.9% cross-script
     positives to multilingual scoring.
- **P1 (meaningful improvement):**
  5. Suffix split: `business_name_core` + `business_name_legal_suffix`
     (never delete); extend suffix map symmetrically.
  6. Extend address abbrev map (after st/saint verification).
  7. `address_postal_code` extraction (US 5 / IN 6 / FR 5 digit patterns).
- **P2 (optimization):**
  8. Trailing decorative strip (measure volume first).
  9. Lowercase `address_landmark`.
  10. Unique-value caching + multiprocessing (runtime ~2–4×).
  11. Extend collision diagnostics to S2/S3.

### Invariants any rewrite must pass
1. Row count and entity_id set preserved exactly (already enforced — keep).
2. Raw columns byte-identical to input (keep).
3. No digits ever removed from any stored view.
4. No legal suffix ever deleted without being recorded in
   `business_name_legal_suffix`.
5. Null-component stripping must not blank a field that has real components.
6. Country transform must work on strings never seen in training.
7. Determinism: same input bytes → same output bytes (seed-free transforms).
8. `name_core` + `name_legal_suffix` must reconstruct `name_canonical`
   (round-trip test).
9. `address_numbers` order must match digit-run order in the raw string.
10. Missing flags set iff normalized value is empty.

### Final recommended architecture
Keep the existing pipeline as the conservative spine (it passed every safety
check in this audit) and **extend the output schema** to the PART-9 multi-view
design. Preprocessing stays matching-free; the second pipeline receives, for
every record, raw + normalized + canonical + token + digit + postal + suffix +
script views, and does blocking/scoring/decision on those. Expected effect:
the measured separation gaps (digits 0.62/0.015, token-set 0.85/0.32,
sorted-address ratio 0.81/0.38) become directly usable features — which is
the largest available lever on F_0.5, since name similarity alone provably
cannot separate the hard cases under F_0.5's precision weighting.

*No claim is made that these changes reach 99.2% F_0.5. The audit shows
preprocessing enables it; the second pipeline decides it.*