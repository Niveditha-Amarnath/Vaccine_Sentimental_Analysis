import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("enhanced_dataset.csv")

tasks = {
    "sentiment": "label",
    "emotion": "emotion",
    "misinformation": "misinformation",
    "topic": "topic"
}

results = []

for task, label_col in tasks.items():
    with open(f"models/{task}_model.pkl", "rb") as f:
        model = pickle.load(f)

    X = df["tweet"].astype(str)
    y = df[label_col].astype(str)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)

    results.append({
        "Task": task,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, average="weighted", zero_division=0)
    })

results_df = pd.DataFrame(results)
results_df.to_csv("outputs/evaluation_metrics.csv", index=False)

print(results_df)
print("Evaluation saved to outputs/evaluation_metrics.csv")