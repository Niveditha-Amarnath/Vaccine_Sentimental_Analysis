import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

df = pd.read_csv("enhanced_dataset.csv")

TEXT_COL = "tweet"

tasks = {
    "sentiment": "label",
    "emotion": "emotion",
    "misinformation": "misinformation",
    "topic": "topic"
}

results = {}

for task_name, label_col in tasks.items():
    print("\n" + "=" * 60)
    print(f"Training {task_name.upper()} model")
    print("=" * 60)

    X = df[TEXT_COL].astype(str)
    y = df[label_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() > 1 else None
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=5000
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    results[task_name] = acc

    print(f"{task_name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    with open(f"models/{task_name}_model.pkl", "wb") as f:
        pickle.dump(model, f)

print("\nTraining completed successfully!")
print("Saved models inside models/ folder")

with open("outputs/results.txt", "w") as f:
    for task, acc in results.items():
        f.write(f"{task}: {acc:.4f}\n")

print("Results saved to outputs/results.txt")