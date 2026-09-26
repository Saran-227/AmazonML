# Phase 4 & Sprint Step 1: Detailed Blocking Error Analysis (213 Missed Matches)

- **Benchmark Sample**: 2,000 Source-1 entities
- **Total True Matches**: 6865
- **Matches Captured by Baseline 6-Rule Blocker**: 6652 (96.90% recall)
- **Total Missed Matches**: 213 (3.10% missed)

## 1. Missed Match Taxonomy & Distribution

| Category | Count | % of Misses | Recoverable? | Additional Cands/S1 | Downstream Precision Risk |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **cross-script** | 139 | 65.26% | Yes (partially via address matching) | +15.3 cands/S1 (+30,643 total on 2k sample) | High: Pairwise models have 0.0 text similarit... |
| **domain concatenation** | 24 | 11.27% | Yes (fully) | +0.4 cands/S1 (+800 total) | Very Low: Clean token expansion without combi... |
| **address formatting** | 21 | 9.86% | Yes | +8.5 cands/S1 (+17,000 total) | Moderate: High candidate volume in dense metr... |
| **spelling variation** | 13 | 6.1% | Partially | +18.0 cands/S1 (+36,000 total) | High: Substantial precision loss on short gen... |
| **numeric variation** | 12 | 5.63% | Yes | +0.1 cands/S1 (+200 total) | Negligible: Addressed in preprocessing stage.... |
| **other** | 1 | 0.47% | No | N/A | N/A... |
| **token reordering** | 1 | 0.47% | Yes (already mostly covered) | +0.0 cands/S1 | Zero.... |
| **abbreviation** | 1 | 0.47% | Partially | +2.1 cands/S1 (+4,200 total) | Moderate: 2-letter and 3-letter acronyms have... |
| **missing address** | 1 | 0.47% | No (without excessive false positives) | +45.0 cands/S1 (+90,000 total) | Extreme: Without an address to verify, cross-... |

## 2. Key Questions Answered

### Q1: What causes most missed matches?
- **Primary Driver**: **cross-script** accounts for **139 out of 213 misses (65.26%)**.
- **Root Cause**: The Source-1 record business name is written in Latin English characters (e.g. `Shakti Agro Limited`), whereas the candidate record in Source-2 or Source-3 is written in regional Indian scripts (e.g. Odia `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍`, Devanagari `शक्ति एग्रो`, Bengali, Tamil, etc.). Because the name tokens share zero orthographic intersection and no transliteration map exists in deterministic ASCII processing, standard name-based blocking rules (`exact_name`, `name_token`, `compressed_name`) cannot intersect.
- **Secondary Drivers**: Address formatting variations (24 misses, 11.27%) and domain concatenation (21 misses, 9.86%).

### Q2: Which causes can realistically be recovered?
1. **Domain Concatenation (11.27%)**: High recoverability. Splitting domains (`technologiesmarketing.com` -> `[technologies, marketing]`) cleanly resolves URL squashing without candidate overhead.
2. **Address Formatting (11.74%)**: High recoverability. Relaxing the requirement of a street number to match on two rare address locality tokens captures these pairs.
3. **Cross-Script (65.26%)**: Mechanically recoverable at blocking stage (by matching exclusively on shared address tokens when scripts differ). However, **downstream model recovery is compromised**: pairwise text similarity on cross-script names evaluates to 0.0. Unless the address match is near-perfect, the ML model assigns low probability or generates false positives on other tenants at that address.
4. **Unrecoverable without high precision penalty**: Disjoint aliases (e.g. trade name vs legal entity) and records with missing addresses cannot be matched without unconstrained fuzzy matching that floods the candidate pool with false positives.

### Q3: What is the cheapest blocking rule capable of recovering them?
- **For Cross-Script**: `cross_script_address` — Generate candidate pairs where `detect_script(s1) != detect_script(cand)` AND both share at least two address tokens of length >= 4 with country agreement.
- **For Domain Concatenation**: Sub-token splitting in preprocessing or `domain_subtoken` blocking key that strips `.com`, `.in`, etc.

### Q4: How many additional candidates would that rule create?
- In `EXP_BLOCKING_001`, activating `cross_script_address` added **+25,314 candidate pairs** across the 2,000 S1 benchmark (+12.66 cands/S1 overhead).
- While this successfully recovered **108 previously missed true matches** (increasing blocking recall from 96.90% to 98.47%), the classifier evaluated on this larger pool suffered a **precision drop from 0.9918 to 0.9779**.
- Under Entity-Level Macro $F_{0.5}$ (where precision is weighted 4× more heavily than recall), the end-to-end score dropped from **0.9572 to 0.9499**.

## 3. Representative Error Examples

### Category: cross-script (139 instances, 65.26%)
- **Source-1**: `Shakti Agro Limited [c o gurnav singh saluja beside]`
- **Candidate**: `ଶକ୍ତି ଆଗ୍ରୋ ଲିମିଟେଡ୍ [c o gurnav singh saluja sadar ]`
- **Cheapest Rule**: `Cross-script dual address token blocking (restricted to records where at least one entity is non-Latin)`
- **Overhead**: `+15.3 cands/S1 (+30,643 total on 2k sample)`

### Category: domain concatenation (24 instances, 11.27%)
- **Source-1**: `Secure Marketing Technologies LLC [982 willman pike hartford city]`
- **Candidate**: `technologiesmarketing.com [hartford city willman pike in]`
- **Cheapest Rule**: `Sub-token domain splitting (strip .com/.in/.org suffixes and split on embedded capital/numeric boundaries)`
- **Overhead**: `+0.4 cands/S1 (+800 total)`

### Category: address formatting (21 instances, 9.86%)
- **Source-1**: `Shakti Agro Limited [c o gurnav singh saluja beside]`
- **Candidate**: `Wexveo [block f 264 c o gurnav singh s]`
- **Cheapest Rule**: `Shared street number + 1 rare address token, or distinctive locality pair`
- **Overhead**: `+8.5 cands/S1 (+17,000 total)`

### Category: spelling variation (13 instances, 6.1%)
- **Source-1**: `Bharani Trust [first floor c 23 fateh nagar j]`
- **Candidate**: `8harani Trust [first floor new delhi west del]`
- **Cheapest Rule**: `3-gram char blocking for distinctive name tokens (length >= 5, max postings <= 200)`
- **Overhead**: `+18.0 cands/S1 (+36,000 total)`

### Category: numeric variation (12 instances, 5.63%)
- **Source-1**: `One Infra Private Limited [b 803 safal solitaire corporat]`
- **Candidate**: `ECTOZEPH [b 8 03 safal solitaire corpora]`
- **Cheapest Rule**: `Address range normalization (split '1056-1060' into individual numeric tokens)`
- **Overhead**: `+0.1 cands/S1 (+200 total)`

### Category: other (1 instances, 0.47%)
- **Source-1**: `KB Arcelormittal LLC [202 bluebird court tx van]`
- **Candidate**: `KB LLC Services [bluebird ct van texas]`
- **Cheapest Rule**: `N/A`
- **Overhead**: `N/A`

