import pandas as pd
from rapidfuzz import process, fuzz
import re
import unicodedata


# ==========================================
# TEXT NORMALIZATION
# ==========================================
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove accents
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    # Replace punctuation with spaces
    text = re.sub(r"[^a-z0-9]+", " ", text)

    # Remove extra spaces
    text = " ".join(text.split())

    return text


# ==========================================
# LOAD DATA
# ==========================================
print("Loading Source 1...")

source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t"
)

print("Loading Source 2...")

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

print("Loading Ground Truth...")

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# ==========================================
# NORMALIZE SOURCE 1
# ==========================================
print("Normalizing Source 1...")

source1["name_norm"] = source1["business_name"].apply(
    normalize_text
)

source1["address_norm"] = source1["business_address"].apply(
    normalize_text
)


# ==========================================
# NORMALIZE SOURCE 2
# ==========================================
print("Normalizing Source 2...")

source2["name_norm"] = source2["business_name"].apply(
    normalize_text
)

source2["address_norm"] = source2["business_address"].apply(
    normalize_text
)


# ==========================================
# TEST FIRST 10 SOURCE 1 ENTITIES
# ==========================================
ground_truth_sample = ground_truth.head(10)

total_true_matches = 0
captured_true_matches = 0
total_candidates = 0


for _, gt in ground_truth_sample.iterrows():

    s1_id = gt["source1_entity_id"]

    # Find Source 1 record
    s1_rows = source1[
        source1["entity_id"] == s1_id
    ]

    if s1_rows.empty:
        continue

    s1 = s1_rows.iloc[0]

    country = s1["country"]

    # ==========================================
    # COUNTRY BLOCK
    # ==========================================
    country_candidates = source2[
        source2["country"] == country
    ].copy()

    # Reset index so RapidFuzz indexes match
    # dataframe positions safely
    country_candidates = country_candidates.reset_index(
        drop=True
    )

    # Remove records where BOTH name and address
    # are missing
    country_candidates = country_candidates[
        (country_candidates["name_norm"] != "") |
        (country_candidates["address_norm"] != "")
    ].reset_index(drop=True)


    # ==========================================
    # TOP 100 NAME CANDIDATES
    # ==========================================
    name_results = process.extract(
        s1["name_norm"],
        country_candidates["name_norm"],
        scorer=fuzz.token_set_ratio,
        limit=100
    )


    # ==========================================
    # TOP 100 ADDRESS CANDIDATES
    # ==========================================
    address_results = process.extract(
        s1["address_norm"],
        country_candidates["address_norm"],
        scorer=fuzz.token_set_ratio,
        limit=100
    )


    # ==========================================
    # COMBINE CANDIDATES
    # ==========================================
    candidate_ids = set()

    for _, score, index in name_results:

        candidate_ids.add(
            country_candidates.iloc[index]["entity_id"]
        )

    for _, score, index in address_results:

        candidate_ids.add(
            country_candidates.iloc[index]["entity_id"]
        )


    total_candidates += len(candidate_ids)


    # ==========================================
    # GET TRUE S2 MATCHES
    # ==========================================
    true_match_ids = [
        x.strip()
        for x in str(
            gt["matched_entity_ids"]
        ).split(",")
        if x.strip().startswith("S2-")
    ]

    total_true_matches += len(true_match_ids)


    # ==========================================
    # CHECK HOW MANY TRUE MATCHES WERE CAPTURED
    # ==========================================
    captured = (
        set(true_match_ids)
        & candidate_ids
    )

    captured_true_matches += len(captured)


    # ==========================================
    # DISPLAY RESULT FOR THIS S1
    # ==========================================
    print("\n--------------------------------------")

    print(
        "S1:",
        s1_id
    )

    print(
        "Business:",
        s1["business_name"]
    )

    print(
        "Candidates generated:",
        len(candidate_ids)
    )

    print(
        "True S2 matches:",
        len(true_match_ids)
    )

    print(
        "True matches captured:",
        len(captured)
    )


# ==========================================
# FINAL RESULTS
# ==========================================
print("\n======================================")
print("REAL CANDIDATE GENERATION TEST")
print("======================================")


print(
    "True S2 matches checked:",
    total_true_matches
)


print(
    "True matches captured:",
    captured_true_matches
)


print(
    "Total candidates generated:",
    total_candidates
)


print(
    "Average candidates per S1:",
    round(
        total_candidates /
        len(ground_truth_sample),
        2
    )
)


# ==========================================
# BLOCKING RECALL
# ==========================================
if total_true_matches > 0:

    recall = (
        captured_true_matches /
        total_true_matches
    ) * 100

    print(
        "Candidate blocking recall:",
        round(recall, 2),
        "%"
    )