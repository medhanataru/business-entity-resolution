# Business Entity Resolution Pipeline (ML Challenge 2026)

An end-to-end Entity Resolution (ER) pipeline designed to link noisy, unstructured business records across three independent data sources (`Source 1`, `Source 2`, and `Source 3`) without shared primary keys.

## Problem Overview
- **Task:** Given deduplicated reference entities (`S1`) and noisy records from two external sources (`S2`, `S3`), identify all matching records per reference entity.
- **Key Challenges:**
  - **Severe Text Noise:** Legal suffix variations (`Corp` vs. `Corporation`, `Pvt Ltd`), typos, word-order transpositions, and landmark-based address formats.
  - **Zero-Shot Country Generalization:** Trained on `US` and `India` records while ensuring country-agnostic inference on unseen `France` test entities.
  - **Scale & Evaluation:** Pruning $O(N \times M)$ pairwise comparisons via blocking and optimizing for precision-weighted $F_{0.5}$ scoring.

## Repository Structure
- `inspect_data.py` — Exploratory data analysis, missing value profiling, and noise pattern inspection across `.tsv` sources.
- `blocking_name_test.py` & `blocking_address_test.py` — Multi-attribute blocking experiments using normalized business names and address tokens.
- `generate_candidates.py` — Generates filtered candidate pairs (`candidate_pairs.tsv`) prior to pairwise scoring.
- `blocking_recall_test.py` & `blocking_validation.py` — Evaluates candidate pair recall ceiling, reduction ratio, and output schema compliance.

## Pipeline Workflow
1. **Data Cleaning & Normalization:** Standardizes business names, punctuation, and address abbreviations while keeping features country-agnostic.
2. **Blocking & Candidate Generation:** Reduces the search space by grouping plausible entity pairs and outputting `candidate_pairs.tsv`.
3. **Validation & Verification:** Benchmarks blocking recall on held-out validation splits and verifies `.tsv` formatting rules for `matching_results.tsv`.
