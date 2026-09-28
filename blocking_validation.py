import pandas as pd
from rapidfuzz import fuzz

# Load data
source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t",
   
)

source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

ground_truth = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t",
    
)

# Test the first 5 Source 1 businesses
for _, gt in ground_truth.head(5).iterrows():

    s1_id = gt["source1_entity_id"]
    match_ids = str(gt["matched_entity_ids"]).split(",")

    s1 = source1[source1["entity_id"] == s1_id]

    if s1.empty:
        continue

    s1 = s1.iloc[0]

    print("\n======================================")
    print("SOURCE 1:", s1_id)
    print("NAME:", s1["business_name"])
    print("ADDRESS:", s1["business_address"])
    print("COUNTRY:", s1["country"])

    print("\nTRUE MATCHES:")
    
    for match_id in match_ids:

        match_id = match_id.strip()

        candidate = source2[
            source2["entity_id"] == match_id
        ]

        if candidate.empty:
            print(match_id, "NOT FOUND IN SOURCE 2")
            continue

        candidate = candidate.iloc[0]

        name_score = fuzz.token_set_ratio(
            str(s1["business_name"]),
            str(candidate["business_name"])
        )

        address_score = fuzz.token_set_ratio(
            str(s1["business_address"]),
            str(candidate["business_address"])
        )

        print("\n", match_id)
        print("Name:", candidate["business_name"])
        print("Address:", candidate["business_address"])
        print("Name similarity:", round(name_score, 1))
        print("Address similarity:", round(address_score, 1))