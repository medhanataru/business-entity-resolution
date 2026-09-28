import pandas as pd

# Read ground truth
gt = pd.read_csv(
    "student_resource/dataset/train/train_ground_truth.tsv",
    sep="\t"
)

# Take first 5 Source 1 entities
sample_gt = gt.head(5)

print("\nGROUND TRUTH EXAMPLES")
print("=" * 80)

for _, row in sample_gt.iterrows():
    print("\nSource 1:", row["source1_entity_id"])
    print("Matches:", row["matched_entity_ids"])

# Collect IDs we need to inspect
needed_s2 = set()
needed_s3 = set()

for ids in sample_gt["matched_entity_ids"].fillna(""):
    for entity_id in ids.split(","):
        entity_id = entity_id.strip()

        if entity_id.startswith("S2-"):
            needed_s2.add(entity_id)

        elif entity_id.startswith("S3-"):
            needed_s3.add(entity_id)

print("\nNumber of S2 records to find:", len(needed_s2))
print("Number of S3 records to find:", len(needed_s3))


# Function to find specific IDs without loading everything at once
def find_records(file_path, wanted_ids):
    found = []

    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=500_000
    ):
        matches = chunk[chunk["entity_id"].isin(wanted_ids)]

        if not matches.empty:
            found.append(matches)

        if sum(len(x) for x in found) >= len(wanted_ids):
            break

    if found:
        return pd.concat(found, ignore_index=True)

    return pd.DataFrame()


source2_matches = find_records(
    "student_resource/dataset/train/train_source2.tsv",
    needed_s2
)

source3_matches = find_records(
    "student_resource/dataset/train/train_source3.tsv",
    needed_s3
)


print("\n")
print("=" * 80)
print("MATCHING SOURCE 2 RECORDS")
print("=" * 80)

print(source2_matches.to_string(index=False))


print("\n")
print("=" * 80)
print("MATCHING SOURCE 3 RECORDS")
print("=" * 80)

print(source3_matches.to_string(index=False))