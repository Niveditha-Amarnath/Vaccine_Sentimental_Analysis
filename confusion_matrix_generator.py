import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
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

# Custom pastel theme colormap matching website
pastel_cmap = LinearSegmentedColormap.from_list(
    "pastel_theme",
    [
        "#11131B",   # dark background
        "#1A1630",   # deep violet
        "#2B1F3F",   # muted purple
        "#B8C0FF",   # pastel blue
        "#E0B3FF",   # pastel violet
        "#FFB6D9"    # pastel pink
    ]
)

for task, label_col in tasks.items():

    with open(f"models/{task}_model.pkl", "rb") as f:
        model = pickle.load(f)

    X = df["tweet"].astype(str)
    y = df[label_col].astype(str)

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.1,
        random_state=42,
        stratify=y
    )

    y_pred = model.predict(X_test)

    labels = sorted(y.unique())

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=labels
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    disp.plot(
        ax=ax,
        xticks_rotation=45,
        cmap=pastel_cmap,
        colorbar=False,
        values_format="d"
    )

    # Background styling
    fig.patch.set_facecolor("#11131B")
    ax.set_facecolor("#1A1630")

    # Title
    ax.set_title(
        f"{task.capitalize()} Confusion Matrix",
        fontsize=8,
        color="#F5D0FE",
        pad=13,
        fontweight="bold"
    )

    # Axis labels
    ax.set_xlabel(
        "Predicted Label",
        fontsize=7,
        color="#E0B3FF",
        fontweight="bold"
    )

    ax.set_ylabel(
        "True Label",
        fontsize=7,
        color="#E0B3FF",
        fontweight="bold"
    )

    # Tick labels
    ax.tick_params(
        axis="x",
        colors="#F3F4FF",
        labelsize=7
    )

    ax.tick_params(
        axis="y",
        colors="#F3F4FF",
        labelsize=7
    )

    # Border styling
    for spine in ax.spines.values():
        spine.set_edgecolor("#B8C0FF")
        spine.set_linewidth(1.0)

    # Cell number styling
    for text in disp.text_.ravel():
        text.set_color("#FFFFFF")
        text.set_fontsize(7)
        text.set_fontweight("bold")

    plt.tight_layout()

    output_path = f"outputs/confusion_matrices/{task}_confusion_matrix.png"

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor()
    )

    plt.close()

print("Confusion matrices saved with pastel theme.")