import os
import re
import unicodedata
from collections import defaultdict

import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE = "student_resource"

TEST_DIR = os.path.join(BASE, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE, "output")

S1_FILE = os.path.join(TEST_DIR, "test_source1.tsv")
S2_FILE = os.path.join(TEST_DIR, "test_source2.tsv")
S3_FILE = os.path.join(TEST_DIR, "test_source3.tsv")

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

# We don't want extremely common keys creating millions
# of useless candidates.
MAX_KEY_FREQUENCY = 300

# Maximum candidates retained from one blocking key.
MAX_CANDIDATES_PER_KEY = 100


# Common business/legal words.
STOPWORDS = {
    "inc", "incorporated",
    "llc",
    "ltd", "limited",
    "pvt", "private",
    "corp", "corporation",
    "co", "company",
    "plc",
    "llp",
    "lp",
    "llc",
    "group",
    "holdings",
    "holding",
    "enterprises",
    "enterprise",
    "services",
    "service",
    "solutions",
    "solution",
    "industries",
    "industry",
    "international",
    "india",
    "usa",
    "us",
}


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Unicode-safe normalization.

    Keeps non-Latin scripts such as:
    Hindi, Tamil, etc.
    """
    if pd.isna(value):
        return ""

    text = str(value).lower().strip()

    # Normalize Unicode characters.
    text = unicodedata.normalize("NFKC", text)

    # Replace punctuation/symbols with spaces.
    chars = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            chars.append(ch)
        else:
            chars.append(" ")

    text = "".join(chars)

    # Collapse spaces.
    text = re.sub(r"\s+", " ", text).strip()

    return text


def get_tokens(text):
    return [x for x in text.split() if x]


def important_token(text):
    """
    Pick the longest informative token.

    Example:
    'maure williams colombier inc'
        -> 'colombier'

    'payne enterprises'
        -> 'payne'
    """
    tokens = get_tokens(text)

    useful = [
        t for t in tokens
        if len(t) >= 3 and t not in STOPWORDS
    ]

    if not useful:
        useful = [t for t in tokens if len(t) >= 2]

    if not useful:
        return ""

    return max(useful, key=len)


def token_prefix(text, n=6):
    token = important_token(text)

    if not token:
        return ""

    # For both Latin and non-Latin scripts.
    return token[:n]


def address_key(text):
    """
    Create a compact address blocking key.

    Uses:
      - first house/building number
      - first part of the longest informative address token

    This helps when business names are badly corrupted
    but addresses are similar.
    """
    if not text:
        return ""

    tokens = get_tokens(text)

    number = ""
    for token in tokens:
        if any(ch.isdigit() for ch in token):
            match = re.search(r"\d+", token)
            if match:
                number = match.group()
                break

    useful = [
        t for t in tokens
        if len(t) >= 4 and not t.isdigit()
    ]

    if useful:
        long_token = max(useful, key=len)
        token_part = long_token[:6]
    else:
        token_part = ""

    if number and token_part:
        return number + "_" + token_part

    if number:
        return number

    if token_part:
        return token_part

    return ""


# ============================================================
# LOAD SOURCE 1
# ============================================================

print("Loading Source 1...")

s1 = pd.read_csv(
    S1_FILE,
    sep="\t",
    dtype=str,
    keep_default_na=False
)

print("Source 1 records:", len(s1))

s1["name_norm"] = s1["business_name"].map(normalize_text)
s1["address_norm"] = s1["business_address"].map(normalize_text)

s1["name_prefix"] = s1["name_norm"].map(token_prefix)
s1["address_block"] = s1["address_norm"].map(address_key)


# ============================================================
# CANDIDATE STORAGE
# ============================================================

# source1 ID -> set of candidate IDs
candidate_map = defaultdict(set)


def add_candidates_from_merge(
    s1_part,
    source_part,
    key,
    source_label
):
    """
    Merge Source 1 and Source 2/3 on a blocking key.
    """
    if key not in s1_part.columns or key not in source_part.columns:
        return

    left = s1_part[
        ["entity_id", key]
    ].copy()

    right = source_part[
        ["entity_id", key]
    ].copy()

    left = left[left[key] != ""]
    right = right[right[key] != ""]

    if left.empty or right.empty:
        return

    # Remove very common keys.
    freq = right[key].value_counts()

    valid_keys = freq[
        freq <= MAX_KEY_FREQUENCY
    ].index

    right = right[
        right[key].isin(valid_keys)
    ]

    if right.empty:
        return

    merged = left.merge(
        right,
        on=key,
        how="inner",
        suffixes=("_s1", "_candidate")
    )

    if merged.empty:
        return

    # Limit candidates per S1/key.
    merged = merged[
        ["entity_id_s1", "entity_id_candidate"]
    ]

    for s1_id, group in merged.groupby("entity_id_s1"):
        ids = group["entity_id_candidate"].tolist()

        # Prevent one noisy key from exploding candidate count.
        ids = ids[:MAX_CANDIDATES_PER_KEY]

        candidate_map[s1_id].update(ids)


# ============================================================
# PROCESS SOURCE 2 AND SOURCE 3
# ============================================================

for source_number, source_file in [
    ("Source 2", S2_FILE),
    ("Source 3", S3_FILE)
]:

    print()
    print("=" * 60)
    print("Processing", source_number)
    print("=" * 60)

    source = pd.read_csv(
        source_file,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(source_number, "records:", len(source))

    source["name_norm"] = source["business_name"].map(
        normalize_text
    )

    source["address_norm"] = source["business_address"].map(
        normalize_text
    )

    source["name_prefix"] = source["name_norm"].map(
        token_prefix
    )

    source["address_block"] = source["address_norm"].map(
        address_key
    )

    # --------------------------------------------------------
    # BLOCK 1: exact normalized business name
    # --------------------------------------------------------

    print("Blocking by normalized business name...")

    add_candidates_from_merge(
        s1,
        source,
        "name_norm",
        source_number
    )

    print(
        "Candidate S1 entities so far:",
        len(candidate_map)
    )

    # --------------------------------------------------------
    # BLOCK 2: important-name-token prefix
    # --------------------------------------------------------

    print("Blocking by important name token...")

    add_candidates_from_merge(
        s1,
        source,
        "name_prefix",
        source_number
    )

    print(
        "Candidate S1 entities so far:",
        len(candidate_map)
    )

    # --------------------------------------------------------
    # BLOCK 3: address key
    # --------------------------------------------------------

    print("Blocking by address key...")

    add_candidates_from_merge(
        s1,
        source,
        "address_block",
        source_number
    )

    print(
        "Candidate S1 entities so far:",
        len(candidate_map)
    )

    del source


# ============================================================
# WRITE FINAL CANDIDATE FILE
# ============================================================

print()
print("=" * 60)
print("Writing candidate_pairs.tsv")
print("=" * 60)

rows = []

for s1_id in s1["entity_id"]:

    candidates = candidate_map.get(s1_id, set())

    # Sort for deterministic output.
    candidates = sorted(candidates)

    rows.append({
        "source1_entity_id": s1_id,
        "candidate_entity_ids": ",".join(candidates)
    })


candidate_df = pd.DataFrame(rows)

candidate_df.to_csv(
    OUTPUT_FILE,
    sep="\t",
    index=False,
    encoding="utf-8"
)

print()
print("DONE!")
print("Output:", OUTPUT_FILE)
print("Source 1 entities:", len(candidate_df))

total_candidates = candidate_df[
    "candidate_entity_ids"
].map(
    lambda x: 0 if not x else len(x.split(","))
).sum()

print("Total candidate links:", total_candidates)

print()
print("First 5 rows:")
print(candidate_df.head().to_string(index=False))