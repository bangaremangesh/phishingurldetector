import joblib
import pandas as pd

from url_features import extract_features

# Load the trained model
artifact = joblib.load("models/phishing_url_model.joblib")

model = artifact["model"]
feature_columns = artifact["feature_columns"]

print("=" * 65)
print("PHISHING URL DETECTOR - MODEL DIAGNOSTICS")
print("=" * 65)

print("\nModel classes:", model.classes_)
print("Number of features expected:", len(feature_columns))
print("Features expected by model:", feature_columns)

# Test legitimate and suspicious URLs
urls = [
    "https://google.com",
    "https://www.google.com",
    "https://youtube.com",
    "https://www.microsoft.com",
    "https://www.wikipedia.org",
    "https://example.com",
    "http://google.com/login/verify/account",
    "http://192.168.1.1/login",
]

importance = pd.Series(
    model.feature_importances_,
    index=feature_columns,
).sort_values(ascending=False)

print("\nTop 10 important model features:")
print(importance.head(10).to_string())

for url in urls:
    features = extract_features(url)

    # Ensure the input columns match the training columns
    X = pd.DataFrame([features]).reindex(
        columns=feature_columns,
        fill_value=0,
    )

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    classes = list(model.classes_)
    phishing_score = probabilities[classes.index(0)] * 100
    legitimate_score = probabilities[classes.index(1)] * 100

    print("\n" + "-" * 65)
    print("URL:", url)
    print("Predicted class:", prediction)
    print("Prediction:", "PHISHING" if prediction == 0 else "LEGITIMATE")
    print(f"Phishing score:   {phishing_score:.2f}%")
    print(f"Legitimate score: {legitimate_score:.2f}%")

    print("\nExtracted features:")
    print(X.to_string(index=False))

print("\n" + "=" * 65)
print("DIAGNOSTICS COMPLETED")