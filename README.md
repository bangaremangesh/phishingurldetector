# Phishing URL Detector

An academic project for an Artificial Intelligence and Machine Learning course. It extracts features from a submitted URL and uses a Random Forest classifier to estimate whether the URL is phishing or legitimate.

> **Important:** This is an educational demonstration, not a security product. Predictions may be wrong. The app analyzes URL text only; it does not visit the website, inspect its content, or check live threat-intelligence services. Do not rely on it to decide whether a link is safe.

## Features

- Streamlit interface for entering and analyzing a URL.
- URL validation and feature extraction, including URL length, hostname characteristics, HTTPS usage, IP addresses, suspicious terms, and URL structure.
- Random Forest predictions with estimated phishing and legitimate class probabilities.
- A panel showing the extracted features used for a prediction.
- A training script that evaluates a hostname-grouped train/test split and saves the trained model.

## Project structure

```text
.
├── app.py                             # Streamlit application
├── url_features.py                    # Shared URL feature extraction
├── train_model.py                     # Train, evaluate, and save the Random Forest
├── test_predictions.py                 # Run sample predictions from the saved model
├── debug_model.py                     # Print model diagnostics and sample predictions
├── check_dataseturls.py                # Inspect Google-hosted URLs in the dataset
├── inspect_dataset.py                  # Inspect the local CSV dataset
├── inspect_phiusill.py                 # Print dataset details and statistics
├── prepare_dataset.py                  # Print target and dataset summaries
├── compare_models.py                   # Compare models using the UCI dataset
├── analyze_duplicates.py               # Analyze duplicate rows in the UCI dataset
├── data/
│   └── PhiUSIIL_Phishing_URL_Dataset.csv
└── models/
    ├── phishing_url_model.joblib       # Model used by the app
    └── decision_tree.joblib            # Additional saved model
```

The Streamlit app loads `models/phishing_url_model.joblib`. The CSV at `data/PhiUSIIL_Phishing_URL_Dataset.csv` is needed to train a new model and run the dataset-inspection scripts. The optional UCI-based comparison and duplicate-analysis scripts fetch data from UCI and therefore require an internet connection. The separate `phishing+websites (1).zip` archive is not included because its redistribution terms have not been verified.

## Requirements

- Python installed on your computer.
- Dependencies listed in [`requirements.txt`](requirements.txt).
- The trained model file at `models/phishing_url_model.joblib` to run the app.
- The PhiUSIIL CSV in `data/` if you want to retrain the model or inspect the dataset.

## Setup

From the project directory, create and activate a virtual environment, then install the dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, you can use the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the application

From the project directory:

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit in your browser (usually `http://localhost:8501`). Enter a URL and choose **Analyze URL**. URLs without a scheme are treated as HTTPS URLs.

## Train the model

Ensure the PhiUSIIL CSV is at `data/PhiUSIIL_Phishing_URL_Dataset.csv`, then run:

```bash
python train_model.py
```

The script cleans the URL and label columns, extracts features using `url_features.py`, splits data by hostname to reduce hostname overlap between the training and test sets, reports classification metrics, and saves the model artifact to `models/phishing_url_model.joblib`.

Labels used by this project:

| Label | Meaning    |
| ----- | ---------- |
| `0`   | Phishing   |
| `1`   | Legitimate |

Retraining overwrites the model artifact used by the app. Training on a different dataset or changing feature extraction may affect compatibility with the saved model.

## Run sample predictions

With dependencies and the saved model installed:

```bash
python test_predictions.py
```

For model details and additional sample outputs:

```bash
python debug_model.py
```

## Dataset and model notes

- The main application uses URL-derived numerical features; it does not use page HTML, browser behavior, domain registration data, or external reputation checks.
- The displayed scores are classifier probability estimates, not calibrated or verified probabilities of real-world safety.
- `joblib` model files are serialized Python objects. Only load model files from sources you trust.
- The included PhiUSIIL dataset is available under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) license. Give credit to the dataset authors and cite its source:
  - Arvind Prasad and Shalini Chandra, *PhiUSIIL: A diverse security profile empowered phishing URL detection framework based on similarity index and incremental learning*, Computers & Security (2024). [https://doi.org/10.1016/j.cose.2023.103545](https://doi.org/10.1016/j.cose.2023.103545)
  - [UCI Machine Learning Repository: PhiUSIIL Phishing URL (Website)](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- The separate `phishing+websites (1).zip` archive is intentionally excluded from GitHub until its redistribution rights are established.
