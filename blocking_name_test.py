import pandas as pd
from rapidfuzz import process, fuzz

# Load Source 1 sample
source1 = pd.read_csv(
    "student_resource/dataset/train/train_source1.tsv",
    sep="\t",
    nrows=100
)

# Load Source 2
source2 = pd.read_csv(
    "student_resource/dataset/train/train_source2.tsv",
    sep="\t"
)

# Take the first Source 1 business
s1 = source1.iloc[0]

name = s1["business_name"]
country = s1["country"]

print("SOURCE 1 BUSINESS")
print("------------------")
print("ID:", s1["entity_id"])
print("Name:", name)
print("Address:", s1["business_address"])
print("Country:", country)

# First apply country blocking
country_candidates = source2[
    source2["country"] == country
].copy()

print("\nAfter country blocking:", len(country_candidates))

# Remove missing names
country_candidates = country_candidates.dropna(
    subset=["business_name"]
)

# Find names most similar to the Source 1 name
results = process.extract(
    name,
    country_candidates["business_name"],
    scorer=fuzz.token_set_ratio,
    limit=20
)

print("\nTOP 20 NAME-BASED CANDIDATES")
print("-----------------------------")

for match_name, score, index in results:
    row = country_candidates.loc[index]

    print(
        f"\nScore: {score:.1f}"
        f"\nID: {row['entity_id']}"
        f"\nName: {row['business_name']}"
        f"\nAddress: {row['business_address']}"
    )