from ucimlrepo import fetch_ucirepo

# Download the Phishing Websites Dataset from UCI

print("Downloading dataset...")

phishing_dataset = fetch_ucirepo(id=327)

# Separate the input features and target labels

X = phishing_dataset.data.features
y = phishing_dataset.data.targets

# Display basic information

print("\nDataset downloaded successfully!")

print("\nFeature dataset shape:")
print(X.shape)

print("\nTarget dataset shape:")
print(y.shape)

print("\nFirst five rows of the features:")
print(X.head())

print("\nFeature column names:")
print(X.columns.tolist())

print("\nTarget column names:")
print(y.columns.tolist())

print("\nTarget label distribution:")
print(y.iloc[:, 0].value_counts(dropna=False))

print("\nMissing values in features:")
print(X.isnull().sum().sum())

print("\nDuplicate feature rows:")
print(X.duplicated().sum())
