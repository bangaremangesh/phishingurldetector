import joblib
import pandas as pd

from url_features import extract_features


MODEL_PATH = "models/phishing_url_model.joblib"

artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_columns = artifact["feature_columns"]

test_urls = [
    "https://google.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "https://example.com",
    "http://google.com/login/verify/account",
    "http://192.168.1.1/login",
]

print("\nPHISHING URL DETECTOR - TEST RESULTS")
print("=" * 65)

for url in test_urls:

    features = extract_features(url)

    X = pd.DataFrame(
        [features],
        columns=feature_columns
    )

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)

    phishing_probability = probabilities[
        classes.index(0)
    ]

    result = (
        "POTENTIALLY PHISHING"
        if prediction == 0
        else "LIKELY LEGITIMATE"
    )

    print(f"\nURL: {url}")
    print(f"Prediction: {result}")
    print(
        f"Phishing score: "
        f"{phishing_probability * 100:.2f}%"
    )

print("\n" + "=" * 65)