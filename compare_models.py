from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import pandas as pd


# Step 1: Load the dataset
print("Loading dataset...")

dataset = fetch_ucirepo(id=327)

X = dataset.data.features.copy()
y = dataset.data.targets.iloc[:, 0].copy()


# Step 2: Group identical feature rows
groups = pd.util.hash_pandas_object(
    X,
    index=False
).to_numpy()


# Step 3: Create one shared train-test split
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_indices, test_indices = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]


# Step 4: Define both algorithms
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}


# Step 5: Train and evaluate both models
results = []

for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results.append({
        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Phishing Precision": precision_score(
            y_test,
            predictions,
            pos_label=-1,
            zero_division=0
        ),

        "Phishing Recall": recall_score(
            y_test,
            predictions,
            pos_label=-1,
            zero_division=0
        ),

        "Phishing F1-score": f1_score(
            y_test,
            predictions,
            pos_label=-1,
            zero_division=0
        )
    })


# Step 6: Display the comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Phishing Precision": "{:.4f}".format,
            "Phishing Recall": "{:.4f}".format,
            "Phishing F1-score": "{:.4f}".format
        }
    )
)