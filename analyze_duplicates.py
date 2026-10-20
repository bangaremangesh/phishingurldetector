from ucimlrepo import fetch_ucirepo

# Download the dataset
dataset = fetch_ucirepo(id=327)

X = dataset.data.features
y = dataset.data.targets

# Combine features and target for analysis
data = X.copy()
data["result"] = y.iloc[:, 0].values

print("Total records:", len(data))

# Count duplicate feature rows
duplicate_features = X.duplicated(keep=False)

print(
    "Records belonging to repeated feature groups:",
    duplicate_features.sum()
)

# Check for repeated complete records
duplicate_records = data.duplicated(keep=False)

print(
    "Records belonging to repeated complete-record groups:",
    duplicate_records.sum()
)

# Check whether identical features have conflicting labels
label_counts = data.groupby(
    list(X.columns),
    dropna=False
)["result"].nunique()

conflicting_groups = label_counts[label_counts > 1]

print(
    "Feature groups with conflicting labels:",
    len(conflicting_groups)
)

print("\nTarget distribution:")
print(data["result"].value_counts())

print("\nDuplicate analysis completed.")