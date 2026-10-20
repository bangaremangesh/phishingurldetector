import streamlit as st
import pandas as pd
import joblib

from urllib.parse import urlparse
from url_features import extract_features


# -----------------------------------------
# 1. Page configuration
# -----------------------------------------

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)


# -----------------------------------------
# 2. Load trained model
# -----------------------------------------

@st.cache_resource
def load_model():
    artifact = joblib.load("models/phishing_url_model.joblib")

    model = artifact["model"]
    feature_columns = artifact["feature_columns"]

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The trained model does not support probability estimates."
        )

    if 0 not in model.classes_ or 1 not in model.classes_:
        raise ValueError(
            f"Unexpected model classes: {model.classes_}"
        )

    return model, feature_columns


# -----------------------------------------
# 3. URL validation
# -----------------------------------------

def validate_url(raw_url):
    raw_url = raw_url.strip()

    if not raw_url:
        return False, "", "Please enter a URL."

    # Add a scheme if the user omitted it.
    if "://" not in raw_url:
        raw_url = "https://" + raw_url

    try:
        parsed = urlparse(raw_url)

        # Accessing .port can raise ValueError for an invalid port.
        _ = parsed.port

        if parsed.scheme.lower() not in ("http", "https"):
            return False, "", "Only HTTP and HTTPS URLs are supported."

        if not parsed.hostname:
            return False, "", "The URL must contain a valid hostname."

        if any(char.isspace() for char in raw_url):
            return False, "", "The URL must not contain spaces."

        hostname = parsed.hostname

        if "." not in hostname and hostname.lower() != "localhost":
            return (
                False,
                "",
                "Please enter a hostname such as example.com."
            )

        return True, raw_url, ""

    except ValueError:
        return False, "", "The URL or port is invalid."


# -----------------------------------------
# 4. Make a prediction
# -----------------------------------------

def predict_url(url, model, feature_columns):
    features = extract_features(url)

    missing_features = [
        column
        for column in feature_columns
        if column not in features
    ]

    if missing_features:
        raise ValueError(
            "The feature extractor does not provide all features "
            f"required by the model: {missing_features}"
        )

    # Match the exact feature order used during training.
    X = pd.DataFrame(
        [[features[column] for column in feature_columns]],
        columns=feature_columns
    )

    prediction = int(model.predict(X)[0])

    probabilities = model.predict_proba(X)[0]
    classes = list(model.classes_)

    phishing_idx = classes.index(0)
    legitimate_idx = classes.index(1)

    phishing_score = float(probabilities[phishing_idx] * 100)
    legitimate_score = float(probabilities[legitimate_idx] * 100)

    return (
        prediction,
        phishing_score,
        legitimate_score,
        features
    )


# -----------------------------------------
# 5. Website interface
# -----------------------------------------

st.title("🛡️ AI-Powered Phishing URL Detector")

st.write(
    "Enter a URL to analyze its characteristics "
    "and receive a machine-learning prediction."
)

st.info(
    "Educational project: predictions can be incorrect. "
    "Do not use this tool as the sole basis for deciding "
    "whether a website is safe."
)

url = st.text_input(
    "Enter website URL",
    placeholder="https://example.com"
)


# -----------------------------------------
# 6. Analyze URL
# -----------------------------------------

if st.button("Analyze URL", type="primary"):

    is_valid, normalized_url, error_message = validate_url(url)

    if not is_valid:
        st.warning(error_message)

    else:
        try:
            model, feature_columns = load_model()

            (
                prediction,
                phishing_score,
                legitimate_score,
                features
            ) = predict_url(
                normalized_url,
                model,
                feature_columns
            )

            st.subheader("Analysis Result")

            st.code(normalized_url, language=None)

            if prediction == 0:
                st.error("⚠️ Model prediction: Potentially Phishing")
            else:
                st.success("✅ Model prediction: Likely Legitimate")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Phishing score",
                    f"{phishing_score:.2f}%"
                )

            with col2:
                st.metric(
                    "Legitimate score",
                    f"{legitimate_score:.2f}%"
                )

            st.caption(
                "These are the Random Forest model's estimated class "
                "probabilities. They are not verified measurements "
                "of the website's actual safety."
            )

            with st.expander("View extracted URL features"):
                feature_table = pd.DataFrame(
                    list(features.items()),
                    columns=["Feature", "Value"]
                )

                st.dataframe(
                    feature_table,
                    use_container_width=True,
                    hide_index=True
                )

        except Exception as error:
            st.error("Unable to analyze this URL.")
            st.exception(error)


# -----------------------------------------
# 7. Footer
# -----------------------------------------

st.divider()

st.caption(
    "Built with Python, Streamlit, pandas, and a Random Forest "
    "classifier trained on URL-derived features from PhiUSIIL."
)