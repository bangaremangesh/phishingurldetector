import pandas as pd
from pathlib import Path

# Locate the CSV file
data_folder = Path("data")
csv_files = list(data_folder.glob("*.csv"))

if not csv_files:
    print("ERROR: No CSV file found inside the data folder.")
    print("Please place the downloaded dataset inside data/.")
    raise SystemExit

if len(csv_files) > 1:
    print("Multiple CSV files found:")
    for file in csv_files:
        print("-", file.name)

    print("\nKeep only the PhiUSIIL CSV in data/ for this inspection.")
    raise SystemExit

csv_path = csv_files[0]

print("Dataset file:", csv_path.name)
print("\nLoading dataset...")

# Load the dataset
df = pd.read_csv(csv_path)

# Inspect the dataset
print("\nDataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nDuplicate complete rows:")
print(df.duplicated().sum())

print("\nDataset inspection completed.")


# Inspect target label distribution
print("\nTarget label distribution:")
print(df["label"].value_counts())

print("\nTarget label percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100).round(2)
)

# Inspect the URL column
print("\nExample URLs:")
print(df["URL"].head(10).to_string(index=False))

# Inspect feature ranges
print("\nURL-related feature statistics:")
url_features = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "NoOfSubDomain",
    "IsHTTPS"
]

print(df[url_features].describe())