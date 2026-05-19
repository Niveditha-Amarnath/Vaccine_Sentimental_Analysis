import re
import emoji

SLANG_MAP = {
    "vax": "vaccine",
    "vacc": "vaccine",
    "govt": "government",
    "covid": "covid",
    "jab": "vaccine",
    "hai": "is",
    "nahi": "not",
    "bahut": "very",
    "acha": "good",
    "scary": "fearful",
    "sideeffect": "side effect",
    "sideeffects": "side effects"
}

def detect_language(text):
    text_lower = text.lower()

    code_mixed_words = [
        "hai", "nahi", "bahut", "acha",
        "lagta", "mujhe", "romba",
        "illa", "da", "maga"
    ]

    for word in code_mixed_words:
        if word in text_lower.split():
            return "Code-mixed"

    return "English"

def preprocess_text(text):
    text = str(text)

    # Convert emojis
    text = emoji.demojize(text)

    # Remove URLs
    text = re.sub(r"http\\S+|www\\S+", "", text)

    # Remove mentions
    text = re.sub(r"@\\w+", "", text)

    # Remove hashtags symbol
    text = re.sub(r"#", " ", text)

    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\\s]", " ", text)

    # Normalize slang
    words = text.split()
    words = [SLANG_MAP.get(word, word) for word in words]

    return " ".join(words).strip()