import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

os.makedirs("outputs/confusion_matrices", exist_ok=True)

df = pd.read_csv("enhanced_dataset.csv")

tasks = {
    "sentiment": "label",
    "emotion": "emotion",
    "misinformation": "misinformation",
    "topic": "topic"
}

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

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted(y.unique()))

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45)
    plt.title(f"{task.capitalize()} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_matrices/{task}_confusion_matrix.png")
    plt.close()

print("Confusion matrices saved.")