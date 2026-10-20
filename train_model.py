import os
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from url_features import extract_features


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATASET_PATH = "data/PhiUSIIL_Phishing_URL_Dataset.csv"
MODEL_PATH = "models/phishing_url_model.joblib"

# PhiUSIIL label mapping:
# 0 = Phishing
# 1 = Legitimate


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

print("Loading PhiUSIIL dataset...")

df = pd.read_csv(DATASET_PATH)

required_columns = {"URL", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain these columns: {required_columns}. "
        f"Found: {df.columns.tolist()}"
    )

df = df[["URL", "label"]].copy()

df = df.dropna(subset=["URL", "label"])

df["URL"] = df["URL"].astype(str).str.strip()

df = df[df["URL"] != ""]

df["label"] = pd.to_numeric(df["label"], errors="coerce")

df = df.dropna(subset=["label"])

df = df[df["label"].isin([0, 1])]

df["label"] = df["label"].astype(int)

# Remove exact duplicate URL strings.
df = df.drop_duplicates(subset=["URL"])

print(f"Usable URLs: {len(df):,}")
print("\nClass distribution:")
print(df["label"].value_counts().sort_index())


# --------------------------------------------------
# 3. Extract URL features
# --------------------------------------------------

print("\nExtracting URL features...")


def normalize_url(url):
    """Strip a bare 'www.' prefix so bare-domain and www-prefixed
    variants of the same host produce identical subdomain counts."""
    import re
    # Replace ://www. with :// to remove the www subdomain label.
    return re.sub(r"(?i)^(https?://)www\.", r"\1", str(url).strip())


feature_records = [
    extract_features(normalize_url(url))
    for url in df["URL"]
]

X = pd.DataFrame(feature_records)
y = df["label"].reset_index(drop=True)

print(f"Feature matrix shape: {X.shape}")
print(f"Number of features: {len(X.columns)}")


# --------------------------------------------------
# 4. Create a hostname-based split
# --------------------------------------------------

print("\nSplitting dataset by hostname...")

def get_hostname(url):
    from urllib.parse import urlparse
    import re

    url = str(url).strip()

    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url

    return (urlparse(url).hostname or "").lower()


groups = df["URL"].apply(get_hostname).reset_index(drop=True)

valid_groups = groups != ""

X = X.loc[valid_groups].reset_index(drop=True)
y = y.loc[valid_groups].reset_index(drop=True)
groups = groups.loc[valid_groups].reset_index(drop=True)

if y.nunique() < 2:
    raise ValueError("The dataset must contain both label classes.")

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print(f"Training samples: {len(X_train):,}")
print(f"Testing samples: {len(X_test):,}")
print(f"Training hostnames: {groups.iloc[train_indices].nunique():,}")
print(f"Testing hostnames: {groups.iloc[test_indices].nunique():,}")


# --------------------------------------------------
# 5. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

print("\nEvaluating model...")

y_pred = model.predict(X_test)

print(f"\nAccuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(
    "Precision (phishing): "
    f"{precision_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
)
print(
    "Recall (phishing):    "
    f"{recall_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
)
print(
    "F1-score (phishing):   "
    f"{f1_score(y_test, y_pred, pos_label=0, zero_division=0):.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Phishing", "Legitimate"],
        zero_division=0,
    )
)

print("Confusion Matrix (rows = actual, columns = predicted):")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1],
    )
)


# --------------------------------------------------
# 7. Save model and feature information
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

artifact = {
    "model": model,
    "feature_columns": X.columns.tolist(),
    "dataset": "PhiUSIIL",
    "label_mapping": {
        0: "Phishing",
        1: "Legitimate",
    },
}

joblib.dump(artifact, MODEL_PATH)

print(f"\nModel saved successfully to: {MODEL_PATH}")
print("Training completed.")