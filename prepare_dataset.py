import pandas as pd
from pathlib import Path

# Load the PhiUSIIL dataset
data_path = Path("data/PhiUSIIL_Phishing_URL_Dataset.csv")

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

# Display target distribution
print("\nTarget label distribution:")
print(df["label"].value_counts())

# Display target percentages
print("\nTarget label percentages:")
print(
    (df["label"].value_counts(normalize=True) * 100).round(2)
)

# Inspect column names
print("\nDataset columns:")
print(df.columns.tolist())

# Inspect the target labels
print("\nUnique target labels:")
print(sorted(df["label"].dropna().unique()))

print("\nDataset inspection completed.")



print("\nSample URLs and their labels:")

print(
    df[["URL", "label"]].sample(10, random_state=42)
)

print("\nLabel distribution:")
print(df["label"].value_counts().sort_index())