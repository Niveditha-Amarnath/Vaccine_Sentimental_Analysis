import pandas as pd
import random

# Load dataset
df = pd.read_csv("dataset.csv")

# Emotion mapping
emotion_map = {
    "positive": ["Trust", "Happiness"],
    "negative": ["Fear", "Anger", "Anxiety"],
    "neutral": ["Neutral"]
}

# Topic keywords
topic_keywords = {
    "Side effects": ["side effects", "fever", "pain", "reaction"],
    "Booster dose": ["booster", "third dose"],
    "Awareness campaigns": ["awareness", "campaign"],
    "Government policy": ["government", "policy", "guidelines"],
    "Healthcare trust": ["doctor", "hospital", "healthcare"],
    "Conspiracy theories": ["chips", "fake", "dangerous", "autism"]
}

# Add emotion
df["emotion"] = df["label"].apply(
    lambda x: random.choice(emotion_map[x.lower()])
)

# Add misinformation
def misinformation_detector(text):
    fake_keywords = [
        "chips", "fake", "autism", "dangerous",
        "blood clots", "tracking"
    ]

    for word in fake_keywords:
        if word in text.lower():
            return "Fake"

    if "worried" in text.lower() or "fear" in text.lower():
        return "Misleading"

    return "Genuine"

df["misinformation"] = df["tweet"].apply(misinformation_detector)

# Add topic
def detect_topic(text):
    text = text.lower()

    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            if keyword in text:
                return topic

    return "General Vaccination"

df["topic"] = df["tweet"].apply(detect_topic)

# Add language
def detect_language(text):
    hindi_words = ["hai", "acha", "bahut"]

    for word in hindi_words:
        if word in text.lower():
            return "Code-mixed"

    return "English"

df["language"] = df["tweet"].apply(detect_language)

# Save upgraded dataset
df.to_csv("enhanced_dataset.csv", index=False)

print("Enhanced dataset created successfully!")