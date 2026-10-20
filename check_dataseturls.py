from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parent
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def is_google_host(url: str) -> bool:
    """Return whether a URL's hostname is google.com or its subdomain."""
    url = url.strip()
    if not url:
        return False

    if "://" not in url:
        url = f"http://{url}"

    try:
        hostname = (urlparse(url).hostname or "").lower()
    except ValueError:
        return False

    return hostname == "google.com" or hostname.endswith(".google.com")


def main() -> None:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required_columns = {"URL", "label"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing_columns)}"
        )

    df = df.loc[:, ["URL", "label"]].dropna().copy()
    df["URL"] = df["URL"].astype(str).str.strip()
    df = df[df["URL"].ne("")]

    labels = pd.to_numeric(df["label"], errors="coerce")
    if labels.isna().any():
        raise ValueError("Dataset contains non-numeric labels.")

    unexpected_labels = sorted(set(labels.unique()).difference({0, 1}))
    if unexpected_labels:
        raise ValueError(
            f"Unexpected labels {unexpected_labels}; expected 0 and 1."
        )
    df["label"] = labels.astype(int)

    print("\nLABEL MAPPING")
    print("0 = Phishing")
    print("1 = Legitimate")

    print("\nOVERALL LABEL DISTRIBUTION")
    print(df["label"].value_counts().sort_index())

    google_rows = df.loc[df["URL"].map(is_google_host)]

    print("\nGOOGLE-HOSTED URL ANALYSIS")
    print("Matching rows:", len(google_rows))

    print("\nLabel distribution:")
    print(google_rows["label"].value_counts().sort_index())

    print("\nSample URLs:")
    for row in google_rows.head(30).itertuples(index=False):
        label_name = "PHISHING" if row.label == 0 else "LEGITIMATE"
        print(f"{label_name:10} | {row.URL}")


if __name__ == "__main__":
    main()