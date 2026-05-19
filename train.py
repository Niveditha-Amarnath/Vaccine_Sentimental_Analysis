"""
train.py
--------
Trains a Logistic Regression model on vaccine-related tweets.

Pipeline:
  1. Load dataset (dataset.csv)
  2. Preprocess text using preprocess.py
  3. TF-IDF vectorisation
  4. Train / test split  (80 / 20)
  5. Train LogisticRegression
  6. Evaluate: accuracy, classification report, confusion matrix
  7. Save model artefacts to  models/
  8. Generate and save visualisations to  data/

Run:
    python train.py
"""

import os
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")           # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

from preprocess import clean_text

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────
DATA_PATH   = "dataset.csv"
MODEL_DIR   = "models"
OUTPUT_DIR  = "data"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── 1. Load dataset ──────────────────────────────────────────────
print("\n📂  Loading dataset …")
df = pd.read_csv(DATA_PATH)
print(f"    Total rows : {len(df)}")
print(f"    Columns    : {list(df.columns)}")
print(df["label"].value_counts())

# Drop rows with missing values
df.dropna(subset=["tweet", "label"], inplace=True)

# ─── 2. Preprocess ────────────────────────────────────────────────
print("\n🧹  Preprocessing tweets …")
df["clean_tweet"] = df["tweet"].apply(clean_text)

# Remove empty rows after cleaning
df = df[df["clean_tweet"].str.strip() != ""]
print(f"    Rows after cleaning : {len(df)}")

# ─── 3. Encode labels ─────────────────────────────────────────────
label_map = {"positive": 2, "neutral": 1, "negative": 0}
df["label_enc"] = df["label"].map(label_map)

X = df["clean_tweet"]
y = df["label_enc"]

# ─── 4. Train-test split ──────────────────────────────────────────
print("\n✂️   Splitting data  (80% train / 20% test) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"    Train : {len(X_train)}   Test : {len(X_test)}")

# ─── 5. TF-IDF Vectorisation ──────────────────────────────────────
print("\n🔢  Applying TF-IDF vectorisation …")
tfidf = TfidfVectorizer(
    max_features=5000,    # top 5 000 words
    ngram_range=(1, 2),   # unigrams + bigrams
    sublinear_tf=True,    # log(1 + tf) scaling
)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec  = tfidf.transform(X_test)
print(f"    Vocabulary size : {len(tfidf.vocabulary_)}")

# ─── 6. Train LogisticRegression ──────────────────────────────────
print("\n🤖  Training Logistic Regression …")
model = LogisticRegression(
    max_iter=1000,
    C=1.0,
    solver="lbfgs",
    random_state=42,
)
model.fit(X_train_vec, y_train)
print("    Training complete ✅")

# ─── 7. Evaluate ──────────────────────────────────────────────────
y_pred   = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n📊  Accuracy : {accuracy * 100:.2f}%")
print("\n📋  Classification Report:")
print(
    classification_report(
        y_test, y_pred, target_names=["Negative", "Neutral", "Positive"]
    )
)

# ─── 8. Save model artefacts ──────────────────────────────────────
print("\n💾  Saving model artefacts …")
with open(os.path.join(MODEL_DIR, "model.pkl"),  "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(MODEL_DIR, "tfidf.pkl"), "wb") as f:
    pickle.dump(tfidf, f)
with open(os.path.join(MODEL_DIR, "label_map.pkl"), "wb") as f:
    pickle.dump(label_map, f)
print("    Saved: models/model.pkl, models/tfidf.pkl, models/label_map.pkl")

# ─── 9. Visualisations ────────────────────────────────────────────

# Colour palette (Positive=green, Neutral=steelblue, Negative=crimson)
COLORS = {
    "Positive" : "#2ecc71",
    "Neutral"  : "#3498db",
    "Negative" : "#e74c3c",
}

label_counts = df["label"].value_counts()
label_counts.index = label_counts.index.str.capitalize()

# ── 9a. Pie chart ────────────────────────────────────────────────
print("\n🎨  Generating visualisations …")
fig, ax = plt.subplots(figsize=(6, 6))
ax.pie(
    label_counts,
    labels=label_counts.index,
    colors=[COLORS[l] for l in label_counts.index],
    autopct="%1.1f%%",
    startangle=140,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
ax.set_title("Sentiment Distribution of Vaccine Tweets", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pie_chart.png"), dpi=150)
plt.close()
print("    Saved: data/pie_chart.png")

# ── 9b. Bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(
    label_counts.index,
    label_counts.values,
    color=[COLORS[l] for l in label_counts.index],
    edgecolor="white",
    linewidth=1.5,
    width=0.5,
)
for bar, val in zip(bars, label_counts.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 5,
        str(val),
        ha="center", va="bottom", fontsize=12, fontweight="bold",
    )
ax.set_title("Tweet Count by Sentiment", fontsize=14, fontweight="bold")
ax.set_xlabel("Sentiment", fontsize=12)
ax.set_ylabel("Number of Tweets", fontsize=12)
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#ffffff")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "bar_chart.png"), dpi=150)
plt.close()
print("    Saved: data/bar_chart.png")

# ── 9c. Confusion matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Neutral", "Positive"],
    yticklabels=["Negative", "Neutral", "Positive"],
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label", fontsize=12)
ax.set_ylabel("True Label", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150)
plt.close()
print("    Saved: data/confusion_matrix.png")

# ── 9d. Word clouds (one per sentiment) ─────────────────────────
for sentiment in ["positive", "negative", "neutral"]:
    text_blob = " ".join(df[df["label"] == sentiment]["clean_tweet"])
    if not text_blob.strip():
        continue
    wc = WordCloud(
        width=800, height=400,
        background_color="white",
        colormap="RdYlGn" if sentiment != "negative" else "Reds",
        max_words=150,
        contour_width=1,
        contour_color="steelblue",
    ).generate(text_blob)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title(
        f"Word Cloud — {sentiment.capitalize()} Tweets",
        fontsize=14, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"wordcloud_{sentiment}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"    Saved: data/wordcloud_{sentiment}.png")

# ── 9e. Accuracy bar (train vs test) ─────────────────────────────
train_acc = accuracy_score(y_train, model.predict(X_train_vec))
test_acc  = accuracy

fig, ax = plt.subplots(figsize=(6, 5))
bars = ax.bar(
    ["Train Accuracy", "Test Accuracy"],
    [train_acc * 100, test_acc * 100],
    color=["#3498db", "#2ecc71"],
    width=0.4,
    edgecolor="white",
    linewidth=1.5,
)
for bar, val in zip(bars, [train_acc * 100, test_acc * 100]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}%",
        ha="center", va="bottom", fontsize=13, fontweight="bold",
    )
ax.set_ylim(0, 110)
ax.set_title("Model Accuracy", fontsize=14, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#ffffff")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_graph.png"), dpi=150)
plt.close()
print("    Saved: data/accuracy_graph.png")

# ─── Done ─────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print(f"  ✅  Training complete!  Accuracy = {accuracy * 100:.2f}%")
print("=" * 55)
print("\nNext step → run:  streamlit run app.py\n")
