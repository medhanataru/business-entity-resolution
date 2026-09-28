import pandas as pd
import re
import unicodedata
from rapidfuzz import fuzz


# -----------------------------
# Text normalization
# -----------------------------
def normalize_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove accents where possible
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


# -----------------------------
# Load data
# -----------------------------
source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t"
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)


# -----------------------------
# Normalize Source 1 and Source 2
# -----------------------------
source1["name_norm"] = source1["business_name"].apply(
    normalize_text
)

source1["address_norm"] = source1["business_address"].apply(
    normalize_text
)

source2["name_norm"] = source2["business_name"].apply(
    normalize_text
)

source2["address_norm"] = source2["business_address"].apply(
    normalize_text
)


# -----------------------------
# Test first 100 Source 1 entities
# -----------------------------
ground_truth_sample = ground_truth.head(100)

total_true_matches = 0
captured_true_matches = 0


print("Testing improved blocking recall on first 100 Source 1 entities...\n")


for _, gt in ground_truth_sample.iterrows():

    s1_id = gt["source1_entity_id"]

    s1_rows = source1[
        source1["entity_id"] == s1_id
    ]

    if s1_rows.empty:
        continue

    s1 = s1_rows.iloc[0]

    s1_name = s1["name_norm"]
    s1_address = s1["address_norm"]

    match_ids = str(
        gt["matched_entity_ids"]
    ).split(",")


    for match_id in match_ids:

        match_id = match_id.strip()

        # For now, validate S2 matches
        if not match_id.startswith("S2-"):
            continue

        candidate_rows = source2[
            source2["entity_id"] == match_id
        ]

        if candidate_rows.empty:
            continue

        candidate = candidate_rows.iloc[0]

        total_true_matches += 1

        candidate_name = candidate["name_norm"]
        candidate_address = candidate["address_norm"]


        # -----------------------------
        # Name similarity
        # -----------------------------
        name_score_1 = fuzz.token_set_ratio(
            s1_name,
            candidate_name
        )

        name_score_2 = fuzz.WRatio(
            s1_name,
            candidate_name
        )

        name_score = max(
            name_score_1,
            name_score_2
        )


        # -----------------------------
        # Address similarity
        # -----------------------------
        address_score = fuzz.token_set_ratio(
            s1_address,
            candidate_address
        )


        # -----------------------------
        # Multi-pass blocking
        # -----------------------------
        captured = (
            name_score >= 80
            or address_score >= 50
        )


        if captured:
            captured_true_matches += 1


# -----------------------------
# Results
# -----------------------------
print("======================================")
print("IMPROVED BLOCKING RECALL RESULT")
print("======================================")

print(
    "True S2 matches checked:",
    total_true_matches
)

print(
    "True matches captured:",
    captured_true_matches
)

if total_true_matches > 0:

    recall = (
        captured_true_matches /
        total_true_matches
    ) * 100

    print(
        "Blocking recall:",
        round(recall, 2),
        "%"
    )