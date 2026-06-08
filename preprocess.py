"""
preprocess.py
-------------
All text preprocessing functions used in both train.py and app.py.

Steps performed:
  1. Lowercase conversion
  2. URL removal
  3. Mention / hashtag removal
  4. Punctuation removal
  5. Emoji / non-ASCII removal (optional)
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

# Words to ALWAYS preserve (negations and important sentiment words)
PRESERVE_WORDS = {
    # English negations
    'no', 'not', 'never', 'none', 'nobody', 'nothing', 'nowhere',
    'nor', 'cannot', "can't", "don't", "doesn't", "didn't", "won't",
    "wouldn't", "shouldn't", "couldn't", "isn't", "aren't", "wasn't",
    "weren't", "hasn't", "haven't", "hadn't",
    
    # Hindi negations
    'nahi', 'nhi', 'nahee', 'nahin',
    
    # Tamil negations
    'illai', 'koodathu', 'mudiyathu', 'pogathu',
    
    # Sentiment indicators
    'acha', 'accha', 'good', 'bad', 'bura', 'kharab', 'bekar', 'galat',
    'safe', 'unsafe', 'effective', 'useless', 'dangerous',
    'nalla', 'nallathu', 'iruku', 'illa',
    
    # Vaccine-related terms
    'vaccine', 'vaccines', 'vaccination', 'booster', 'dose', 'side', 'effects'
}

# Common Hindi stopwords (but exclude negations that we want to preserve)
hindi_stopwords_base = {
    "hai", "hain", "ka", "ke", "ki", "ko", "se", "me", "mein",
    "aur", "bhi", "kya", "tha", "the", "thi", "ho", "hoga", "hogi",
    "ek", "yeh", "woh", "jo", "koi", "sab", "apna", "apni", "apne",
    "mere", "mera", "meri", "kuch", "bahut", "sirf", "par", "pe",
    "ab", "toh", "phir", "agar", "lekin", "isliye", "kyunki", "matlab"
}

# Remove any preserve words from stopwords
hindi_stopwords = {word for word in hindi_stopwords_base if word not in PRESERVE_WORDS}
ALL_STOPWORDS = english_stopwords | hindi_stopwords


def detect_script(text: str) -> str:
    """
    Detect the primary script of the text.
    Returns: 'latin', 'devanagari', 'tamil', or 'mixed'
    """
    # Check for Devanagari (Hindi, Marathi, etc.)
    if re.search(r'[\u0900-\u097F]', text):
        return 'devanagari'
    
    # Check for Tamil
    if re.search(r'[\u0B80-\u0BFF]', text):
        return 'tamil'
    
    # Check for Latin/English
    if re.search(r'[a-zA-Z]', text):
        return 'latin'
    
    return 'unknown'


def preprocess_text(text: str, preserve_indic: bool = True) -> str:
    """
    Full preprocessing pipeline with Indic language support.

    Parameters
    ----------
    text : str
        Raw tweet / post text.
    preserve_indic : bool
        If True, preserve Indic scripts (Hindi, Tamil, etc.)
        If False, strip to ASCII only (legacy behavior)

    Returns
    -------
    str
        Cleaned text ready for analysis.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase (but careful with Indic scripts - they don't have case)
    text = text.lower()

    # 2. Remove URLs (http / https / www)
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # 3. Remove @mentions but KEEP #hashtags content
    text = re.sub(r"@\w+", "", text)
    # Keep hashtags but remove the # symbol
    text = re.sub(r"#(\w+)", r"\1", text)

    # 4. Handle Indic scripts
    if preserve_indic:
        # Keep Indic characters, but remove other emojis and special chars
        # This keeps Devanagari (\u0900-\u097F), Tamil (\u0B80-\u0BFF), and Latin
        text = re.sub(r'[^\u0900-\u097F\u0B80-\u0BFFa-zA-Z\s]', ' ', text)
    else:
        # Legacy: Remove emojis and non-ASCII characters
        text = text.encode("ascii", "ignore").decode("ascii")
        # Remove punctuation
        text = re.sub(r"[^a-z\s]", "", text)

    # 5. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 6. Tokenise
    tokens = text.split()

    # 7. Remove stopwords (but preserve negation words and important terms)
    tokens = [t for t in tokens if t not in ALL_STOPWORDS or t in PRESERVE_WORDS]

    # 8. Stem ONLY English/Latin words (preserve Indic words as-is)
    script = detect_script(text)
    if script == 'latin' or script == 'mixed':
        # Only stem words that appear to be English/Latin
        stemmed_tokens = []
        for token in tokens:
            # If token contains only Latin characters, stem it
            if re.match(r'^[a-z]+$', token):
                stemmed_tokens.append(stemmer.stem(token))
            else:
                # Preserve Indic words and mixed tokens as-is
                stemmed_tokens.append(token)
        tokens = stemmed_tokens

    return " ".join(tokens)


# This function is kept for backward compatibility
def clean_text(text: str) -> str:
    """Legacy function - calls preprocess_text with default settings"""
    return preprocess_text(text, preserve_indic=True)


def detect_language(text: str) -> str:
    """
    Detect language of input text with support for Indian languages.
    """
    text_lower = text.lower()
    
    # Check for Tamil script
    if re.search(r'[\u0B80-\u0BFF]', text):
        return "Tamil"
    
    # Check for Devanagari (Hindi, Marathi, etc.)
    if re.search(r'[\u0900-\u097F]', text):
        return "Hindi"
    
    # Check for Kannada script (Unicode range: 0C80-0CFF)
    if re.search(r'[\u0C80-\u0CFF]', text):
        return "Kannada"
    
    # Kannada words in Latin script
    kannada_latin_words = [
        'chennag', 'idhe', 'illa', 'alla', 'olleyadu', 'bedha', 
        'kettadu', 'nambike', 'bhaya', 'anjike', 'sandega',
        'chennagi', 'ide', 'iruthe', 'macha', 'appa', 'anna'
    ]
    
    # Tamil words in Latin script (common in social media)
    tamil_latin_words = [
        'nalla', 'iruku', 'illai', 'koodathu', 'mudiyathu', 'pogathu', 
        'nallathu', 'semma', 'thani', 'macha', 'unga', 'enna', 'epdi',
        'ivlo', 'romba', 'konjam', 'theriyum', 'pannu', 'vaanga'
    ]
    
    # Hindi words in Latin script
    hindi_latin_words = [
        'hai', 'nahi', 'acha', 'accha', 'bhai', 'kya', 'woh', 'yeh', 
        'sahi', 'theek', 'tum', 'main', 'apna', 'tera', 'mera', 'kya',
        'kahan', 'kaise', 'kitna', 'bahut', 'thoda'
    ]
    
    # Count matches
    kannada_count = sum(1 for word in kannada_latin_words if word in text_lower)
    tamil_count = sum(1 for word in tamil_latin_words if word in text_lower)
    hindi_count = sum(1 for word in hindi_latin_words if word in text_lower)
    
    # Check for English words (3+ letters)
    english_words = re.findall(r'\b[a-z]{3,}\b', text_lower)
    english_count = len(english_words)
    
    # Decision logic - check Kannada first
    if kannada_count > 0:
        if english_count > kannada_count:
            return "Code-mixed (Kannada-English)"
        elif kannada_count > 1:
            return "Kannada"
        else:
            return "Kannada"
    
    if tamil_count > 0:
        if english_count > tamil_count:
            return "Code-mixed (Tamil-English)"
        elif tamil_count > 2:
            return "Tamil"
        else:
            return "Tamil"
    
    if hindi_count > 0:
        if english_count > hindi_count:
            return "Code-mixed (Hindi-English)"
        elif hindi_count > 2:
            return "Hindi"
        else:
            return "Hindi"
    
    # Check for code-mixed with known patterns
    if (kannada_count > 0 or tamil_count > 0 or hindi_count > 0) and english_count > 0:
        return "Code-mixed"
    
    return "English"


# ── Quick self-test ───────────────────────────────────────────────
if __name__ == "__main__":
    samples = [
        "Vaccines save lives! #VaccinateIndia @MoHFW_INDIA 💉",
        "I am scared of vaccine side effects http://fake.com",
        "Vaccination drive started today in our district",
        "vaccines acha nahi hai",
        "vaccines nalla iruku",
        "मुझे टीका नहीं लगवाना hai mujhe dar lag raha hai",
        "vaccines safe nahi hai",
        "This vaccine is not effective at all",
    ]
    
    print("=" * 70)
    print("  Preprocessing Self-Test (with Indic language support)")
    print("=" * 70)
    
    for s in samples:
        print(f"\n  Raw      : {s}")
        print(f"  Clean    : {preprocess_text(s)}")
        print(f"  Language : {detect_language(s)}")
        print("-" * 70)