"""
preprocess.py
-------------
All text preprocessing functions used in both train.py and app.py.

Steps performed:
  1. Lowercase conversion
  2. URL removal
  3. Mention / hashtag removal
  4. Punctuation removal
  5. Emoji / non-ASCII removal
  6. Digit removal
  7. Extra whitespace removal
  8. Stopword removal (NLTK English + common Hindi stopwords)
  9. Stemming (PorterStemmer)
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Download required NLTK data (runs once) ───────────────────────
def download_nltk_resources():
    """Download NLTK resources if not already present."""
    resources = ["stopwords", "punkt", "wordnet"]
    for resource in resources:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass  # Offline environment — use built-in fallback

download_nltk_resources()

# ── Initialise stemmer and stopword list ──────────────────────────
stemmer = PorterStemmer()

english_stopwords = set(stopwords.words("english"))

# Common Hindi words that appear transliterated in Indian social media
hindi_stopwords = {
    "hai", "hain", "ka", "ke", "ki", "ko", "se", "me", "mein",
    "aur", "bhi", "nahi", "nahin", "kya", "tha", "the", "thi",
    "ho", "hoga", "hogi", "ek", "yeh", "woh", "jo", "koi",
    "sab", "apna", "apni", "apne", "mere", "mera", "meri",
    "kuch", "bahut", "sirf", "par", "pe", "ab", "toh", "phir",
    "agar", "lekin", "isliye", "kyunki", "matlab", "matlab",
}

ALL_STOPWORDS = english_stopwords | hindi_stopwords


def clean_text(text: str) -> str:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    text : str
        Raw tweet / post text.

    Returns
    -------
    str
        Cleaned, stemmed text ready for vectorisation.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs  (http / https / www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove @mentions and #hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # 4. Remove emojis and non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # 5. Remove punctuation and special characters (keep only letters/spaces)
    text = re.sub(r"[^a-z\s]", "", text)

    # 6. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 7. Tokenise (simple split — no punkt needed for basic stemming)
    tokens = text.split()

    # 8. Remove stopwords
    tokens = [t for t in tokens if t not in ALL_STOPWORDS]

    # 9. Stem each token
    tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Vaccines save lives! #VaccinateIndia @MoHFW_INDIA 💉",
        "I am scared of vaccine side effects http://fake.com",
        "Vaccination drive started today in our district",
        "मुझे टीका नहीं लगवाना hai mujhe dar lag raha hai",
    ]
    print("=" * 55)
    print("  Preprocessing Self-Test")
    print("=" * 55)
    for s in samples:
        print(f"\n  Raw   : {s}")
        print(f"  Clean : {clean_text(s)}")
