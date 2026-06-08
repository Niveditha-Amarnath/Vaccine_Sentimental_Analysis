import os
import pickle
import re

try:
    from utils.text_preprocessor import preprocess_text, detect_language
except ImportError:
    from text_preprocessor import preprocess_text, detect_language

try:
    from utils.sarcasm_detector import detect_sarcasm
except ImportError:
    def detect_sarcasm(text):
        return {"sarcasm": "Not Detected", "confidence": 0.60}


MODEL_PATHS = {
    "sentiment": "models/sentiment_model.pkl",
    "emotion": "models/emotion_model.pkl",
    "misinformation": "models/misinformation_model.pkl",
    "topic": "models/topic_model.pkl"
}

models = {}

for task, path in MODEL_PATHS.items():
    if os.path.exists(path):
        with open(path, "rb") as f:
            models[task] = pickle.load(f)
    else:
        models[task] = None


POSITIVE_PATTERNS = [
    "safe",
    "very safe",
    "effective",
    "very effective",
    "good",
    "very good",
    "helpful",
    "successful",
    "protected",
    "protection",
    "trust",
    "trusted",
    "recommend",
    "recommended",
    "doctors recommend",
    "awareness helped",
    "feeling safe",
    "feel safe",
    # Hindi positive patterns
    "acha hai",
    "sahi hai",
    "theek hai",
    "safe hai",
    "effective hai",
    # Tamil positive patterns
    "nalla iruku",     # "is good"
    "nallathu",        # "good"
    "super",           # (common in Tamil-English)
    "semma",           # "awesome" (colloquial Tamil)
    # Kannada positive patterns
    "chennag idhe",    # "is good"
    "chennagi ide",    # "is good" (alternative spelling)
    "olleyadu",        # "good"
    "good idhe",       # "is good"
    "safe idhe",       # "is safe"
    "effective idhe",  # "is effective"
    "chennagide",      # "is good" (without space)

]

NEGATIVE_PATTERNS = [
    "unsafe",
    "dangerous",
    "useless",
    "fake",
    "not safe",
    "safe not",
    "not effective",
    "effective not",
    "vaccine not safe",
    "vaccine unsafe",
    "safe nahi",
    "nahi safe",
    "dont trust",
    "do not trust",
    "scared",
    "fear",
    "afraid",
    "side effects",
    "tracking chips",
    "chips",
    "autism",
    "blood clots",
    # Hindi negative patterns
    "acha nahi",
    "good nahi",
    "sahi nahi",
    "theek nahi",
    "bura",
    "kharab",
    "bekar",
    "galat",
    "dangerous hai",
    "problem hai",
    "nahi lagta",     # "does not seem"
    "nahi hona",      # "should not happen"
    "kabhi nahi",     # "never"
    "bilkul nahi",    # "absolutely not"
    "thoda bhi nahi", # "not at all"
    # Tamil negations
    "nalla illa",      # "not good"
    "illai",           # "no/not"
    "koodathu",        # "should not"
    "mudiyathu",       # "cannot"
    "pogathu",         # "will not"
    "seri illa",       # "not okay"
    "safe illa",       # "not safe"
    "effective illa",  # "not effective"
    # Kannada negations
    "illa",            # "no/not"
    "allla",           # "not" (emphatic)
    "chennag illa",    # "not good"
    "olleyadu alla",   # "not good"
    "safe illa",       # "not safe"
    "effective illa",  # "not effective"
    "bedha",           # "bad"
    "kettadu",         # "bad"
    
]

MISINFO_FAKE_PATTERNS = [
    "tracking chips",
    "chips",
    "autism",
    "fake vaccine",
    "doctors are hiding",
    "government is hiding"
]

MISINFO_MISLEADING_PATTERNS = [
    "side effects",
    "worried",
    "fear",
    "scared",
    "afraid",
    "dangerous"
]

# Expanded negation patterns for Hindi/Urdu and code-mixed text
NEGATION_PATTERNS = [
    # English negations
    "not safe", "safe not", "not effective", "effective not",
    "dont trust", "do not trust", "not good",
    # Hindi/Urdu negations
    "nahi", "nhi", "nahee",  # Core negation words
    "acha nahi", "good nahi",  # "not good"
    "safe nahi", "nahi safe",
    "theek nahi",  # "not okay"
    "sahi nahi",   # "not correct"
    "bura",        # "bad"
    "kharab",      # "bad/spoiled"
    "bekar",       # "useless"
    "galat",       # "wrong"
    "nahi hai",    # "is not"
    "nhi hai",     # "is not" (short form)
    "nahi lagta",  # "does not seem"
    "nahi hona",   # "should not happen"
    # Tamil negations
    "nalla illa",
    "illai",
    "koodathu",
]

def handle_tamil_sentiment(text):
    """
    Simple rule-based sentiment for Tamil and Tamil-English text
    """
    text_lower = text.lower()
    
    # Tamil negation patterns
    tamil_negations = [
        "nalla illa", "illai", "koodathu", 
        "safe illa", "effective illa", "seri illa"
    ]
    
    for pattern in tamil_negations:
        if pattern in text_lower:
            return "negative", 0.90
    
    # Check for negative implication in "nalla iruku" (context-dependent)
    if "nalla iruku" in text_lower:
        # Look for additional negative context
        negative_context = ["illai", "koodathu", "problem", "issue", "side effect"]
        if any(ctx in text_lower for ctx in negative_context):
            return "negative", 0.85
        else:
            # "nalla iruku" alone usually means "is good" in positive context
            return "positive", 0.80
    
    return None, None
def handle_kannada_sentiment(text):
    """
    Simple rule-based sentiment for Kannada and Kannada-English text
    """
    text_lower = text.lower()
    
    # Kannada negation patterns
    kannada_negations = [
        "illa", "allla", "chennag illa", "olleyadu alla",
        "safe illa", "effective illa", "bedha", "kettadu"
    ]
    
    for pattern in kannada_negations:
        if pattern in text_lower:
            return "negative", 0.90
    
    # Kannada positive patterns
    kannada_positive = [
        "chennag idhe", "chennagi ide", "olleyadu",
        "good idhe", "safe idhe", "effective idhe", "chennagide"
    ]
    
    for pattern in kannada_positive:
        if pattern in text_lower:
            return "positive", 0.88
    
    # Check for "chennag idhe" with positive context
    if "chennag" in text_lower and "idhe" in text_lower:
        # Check if negation is present
        if "illa" not in text_lower and "alla" not in text_lower:
            return "positive", 0.85
    
    return None, None


def handle_kannada_emotion(text, sentiment):
    """
    Determine emotion for Kannada/Kannada-English text
    """
    text_lower = text.lower()
    
    if sentiment == "negative":
        # Fear/anxiety indicators in Kannada
        fear_keywords = ['bhaya', 'pedi', 'worried', 'scared', 'fear', 'anjike']
        if any(keyword in text_lower for keyword in fear_keywords):
            return "Fear", 0.88
        
        # Distrust indicators
        distrust_keywords = ['nambike illa', 'trust illa', 'doubt', 'suspect', 'sandega']
        if any(keyword in text_lower for keyword in distrust_keywords):
            return "Distrust", 0.85
        
        return "Fear", 0.80
    
    elif sentiment == "positive":
        # Trust indicators in Kannada
        trust_keywords = ['nambike', 'trust', 'chennag idhe', 'safe', 'effective']
        if any(keyword in text_lower for keyword in trust_keywords):
            return "Trust", 0.85
        
        # Hope/optimism
        hope_keywords = ['olleyadu', 'good', 'hope', 'chennag iruthe']
        if any(keyword in text_lower for keyword in hope_keywords):
            return "Hope", 0.82
        
        return "Trust", 0.80
    
    return None, None

def handle_tamil_emotion(text, sentiment):
    """
    Determine emotion for Tamil/Tamil-English text based on sentiment and keywords
    """
    text_lower = text.lower()
    
    # If sentiment is negative, determine specific emotion
    if sentiment == "negative":
        # Fear/anxiety indicators
        fear_keywords = ['bayam', 'bayama', 'pedi', 'pedika', 'worried', 'scared', 'fear']
        if any(keyword in text_lower for keyword in fear_keywords):
            return "Fear", 0.88
        
        # Distrust indicators  
        distrust_keywords = ['nambikkai illa', 'trust illa', 'doubt', 'suspect', 'nambave mudiyathu']
        if any(keyword in text_lower for keyword in distrust_keywords):
            return "Distrust", 0.85
        
        # Concern/worry indicators
        concern_keywords = ['problem', 'issue', 'side effect', 'prabhavam', 'danger', 'risky']
        if any(keyword in text_lower for keyword in concern_keywords):
            return "Fear", 0.82
        
        # General negative emotion
        return "Fear", 0.80
    
    elif sentiment == "positive":
        # Trust indicators
        trust_keywords = ['nambikkai', 'trust', 'nalla iruku', 'safe', 'effective', 'super']
        if any(keyword in text_lower for keyword in trust_keywords):
            return "Trust", 0.85
        
        # Hope/optimism
        hope_keywords = ['nalla', 'good', 'hope', 'nambaren', 'nalla irukum']
        if any(keyword in text_lower for keyword in hope_keywords):
            return "Hope", 0.82
        
        return "Trust", 0.80
    
    return None, None


def handle_code_mixed_emotion(text, sentiment):
    """
    Determine emotion for code-mixed text
    """
    text_lower = text.lower()
    
    if sentiment == "negative":
        # Hindi emotion indicators
        if 'dar' in text_lower or 'dar lag' in text_lower or 'scared' in text_lower:
            return "Fear", 0.85
        if 'bharosa nahi' in text_lower or 'trust nahi' in text_lower:
            return "Distrust", 0.85
        if 'side effect' in text_lower or 'problem' in text_lower:
            return "Fear", 0.82
            
        # Tamil emotion indicators (Latin script)
        if 'bayam' in text_lower or 'pedi' in text_lower:
            return "Fear", 0.85
        if 'nambikkai illa' in text_lower:
            return "Distrust", 0.85
            
        return "Fear", 0.80
        
    elif sentiment == "positive":
        if 'bharosa' in text_lower or 'trust' in text_lower:
            return "Trust", 0.85
        if 'asha' in text_lower or 'hope' in text_lower:
            return "Hope", 0.82
        return "Trust", 0.80
    
    return None, None

def _safe_confidence(confidence, fallback):
    if confidence is None:
        return fallback
    return max(float(confidence), fallback)


def _predict_with_model(task, cleaned_text):
    model = models.get(task)

    if model is None:
        return "unknown", 0.0

    prediction = model.predict([cleaned_text])[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([cleaned_text])[0]
        confidence = float(max(probabilities))

    return prediction, confidence


def handle_code_mixed_sentiment(text):
    """
    Simple rule-based sentiment for code-mixed Hindi-English text
    """
    text_lower = text.lower()
    
    # Check for negation patterns first
    for pattern in NEGATION_PATTERNS:
        if pattern in text_lower:
            return "negative", 0.92
    
    # Hindi negation phrases
    negation_phrases = [
        "acha nahi", "good nahi",
        "sahi nahi", "theek nahi",
        "nahi hai", "nhi hai",
        "nahi", "nhi"
    ]
    
    # If negation pattern exists, return negative
    for phrase in negation_phrases:
        if phrase in text_lower:
            return "negative", 0.92
    
    # Positive Hindi patterns
    positive_patterns = ["acha hai", "good hai", "sahi hai", "theek hai", "safe hai"]
    for phrase in positive_patterns:
        if phrase in text_lower:
            return "positive", 0.88
    
    return None, None


def improved_preprocess_preserve_negations(text):
    """
    Preprocess text while preserving negation words
    """
    # Convert to lowercase
    text = text.lower()
    
    # Define words to preserve (negations and important modifiers)
    preserve_words = {'nahi', 'nhi', 'not', 'no', 'never', 'none', "n't", 'bura', 'kharab', 'bekar', 'galat'}
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split into words
    words = text.split()
    
    # Filter out very short words (length 1) except 'a', 'i'
    # But preserve negation words even if they're short
    filtered_words = []
    for word in words:
        if len(word) == 1 and word not in ['a', 'i'] and word not in preserve_words:
            continue
        filtered_words.append(word)
    
    # Join back
    processed = ' '.join(filtered_words)
    
    # Remove extra spaces
    processed = re.sub(r'\s+', ' ', processed).strip()
    
    return processed


def apply_rule_corrections(text, cleaned_text, results):
    print("RULE ENGINE RUNNING")
    # Use original text for rule matching to preserve negations
    combined_text = f"{text} {cleaned_text}".lower()
    print("Combined Text:", combined_text)
    
    # Check for ANY negation in the text
    text_has_negation = False
    detected_pattern = None
    for pattern in NEGATION_PATTERNS:
        if pattern in combined_text:
            text_has_negation = True
            detected_pattern = pattern
            print(f"Negation detected: '{pattern}'")
            break
    
    if text_has_negation:
        results["sentiment"]["prediction"] = "negative"
        results["sentiment"]["confidence"] = 0.95
        results["emotion"]["prediction"] = "Fear"
        results["emotion"]["confidence"] = 0.90
        return results
    
    # Sentiment correction: negative phrases first
    if any(pattern in combined_text for pattern in NEGATIVE_PATTERNS):
        results["sentiment"]["prediction"] = "negative"
        results["sentiment"]["confidence"] = _safe_confidence(
            results["sentiment"]["confidence"],
            0.88
        )

    # Positive correction overrides only when no strong negative pattern exists
    if any(pattern in combined_text for pattern in POSITIVE_PATTERNS):
        if not any(pattern in combined_text for pattern in NEGATIVE_PATTERNS):
            results["sentiment"]["prediction"] = "positive"
            results["sentiment"]["confidence"] = _safe_confidence(
                results["sentiment"]["confidence"],
                0.92
            )

    # Specific fix: "vaccines are very safe"
    if (
        "vaccine" in combined_text
        and "safe" in combined_text
        and "not safe" not in combined_text
        and "safe not" not in combined_text
        and "unsafe" not in combined_text
        and "safe nahi" not in combined_text
        and "nahi safe" not in combined_text
    ):
        if not any(pattern in combined_text for pattern in NEGATIVE_PATTERNS):
            results["sentiment"]["prediction"] = "positive"
            results["sentiment"]["confidence"] = _safe_confidence(
                results["sentiment"]["confidence"],
                0.95
            )
            results["emotion"]["prediction"] = "Trust"
            results["sentiment"]["confidence"] = max(
                results["sentiment"]["confidence"] or 0, 0.95
            )
            
            results["misinformation"]["prediction"] = "Genuine"
            results["misinformation"]["confidence"] = _safe_confidence(
                results["misinformation"]["confidence"],
                0.90
            )

    # Misinformation correction
    if any(pattern in combined_text for pattern in MISINFO_FAKE_PATTERNS):
        results["misinformation"]["prediction"] = "Fake"
        results["misinformation"]["confidence"] = _safe_confidence(
            results["misinformation"]["confidence"],
            0.90
        )

    elif any(pattern in combined_text for pattern in MISINFO_MISLEADING_PATTERNS):
        results["misinformation"]["prediction"] = "Misleading"
        results["misinformation"]["confidence"] = _safe_confidence(
            results["misinformation"]["confidence"],
            0.85
        )

    # Topic correction
    if "side effect" in combined_text or "side effects" in combined_text:
        results["topic"]["prediction"] = "Side effects"
        results["topic"]["confidence"] = _safe_confidence(
            results["topic"]["confidence"],
            0.88
        )

    elif "booster" in combined_text:
        results["topic"]["prediction"] = "Booster dose"
        results["topic"]["confidence"] = _safe_confidence(
            results["topic"]["confidence"],
            0.88
        )

    elif "campaign" in combined_text or "awareness" in combined_text:
        results["topic"]["prediction"] = "Awareness campaigns"
        results["topic"]["confidence"] = _safe_confidence(
            results["topic"]["confidence"],
            0.88
        )

    elif "doctor" in combined_text or "hospital" in combined_text:
        results["topic"]["prediction"] = "Healthcare trust"
        results["topic"]["confidence"] = _safe_confidence(
            results["topic"]["confidence"],
            0.85
        )
    
    return results

def predict_all(text):
    # Use improved preprocessing that preserves negations
    cleaned_text = preprocess_text(text)
    language = detect_language(text)
    sarcasm_result = detect_sarcasm(text)
    
    results = {}
    
    # Check for sentiments in different languages
    code_mixed_sentiment, code_mixed_conf = handle_code_mixed_sentiment(text)
    tamil_sentiment, tamil_conf = handle_tamil_sentiment(text)
    kannada_sentiment, kannada_conf = handle_kannada_sentiment(text)
    
    for task in MODEL_PATHS.keys():
        if task == "sentiment":
            if kannada_sentiment:
                results[task] = {
                    "prediction": kannada_sentiment,
                    "confidence": kannada_conf
                }
            elif tamil_sentiment:
                results[task] = {
                    "prediction": tamil_sentiment,
                    "confidence": tamil_conf
                }
            elif code_mixed_sentiment:
                results[task] = {
                    "prediction": code_mixed_sentiment,
                    "confidence": code_mixed_conf
                }
            else:
                prediction, confidence = _predict_with_model(task, cleaned_text)
                results[task] = {
                    "prediction": prediction,
                    "confidence": confidence
                }
        else:
            prediction, confidence = _predict_with_model(task, cleaned_text)
            results[task] = {
                "prediction": prediction,
                "confidence": confidence
            }
    
    # Apply rule corrections
    results = apply_rule_corrections(text, cleaned_text, results)
    
    # Override emotion for Kannada text
    if 'kannada' in language.lower():
        kannada_emotion, kannada_emotion_conf = handle_kannada_emotion(text, results["sentiment"]["prediction"])
        if kannada_emotion:
            results["emotion"]["prediction"] = kannada_emotion
            results["emotion"]["confidence"] = kannada_emotion_conf
    elif 'tamil' in language.lower() or 'code-mixed' in language.lower():
        tamil_emotion, tamil_emotion_conf = handle_tamil_emotion(text, results["sentiment"]["prediction"])
        if tamil_emotion:
            results["emotion"]["prediction"] = tamil_emotion
            results["emotion"]["confidence"] = tamil_emotion_conf
    elif 'code-mixed' in language.lower():
        code_mixed_emotion, code_mixed_emotion_conf = handle_code_mixed_emotion(text, results["sentiment"]["prediction"])
        if code_mixed_emotion:
            results["emotion"]["prediction"] = code_mixed_emotion
            results["emotion"]["confidence"] = code_mixed_emotion_conf
    
    results["language"] = {
        "prediction": language,
        "confidence": 1.0
    }

    results["sarcasm"] = {
        "prediction": sarcasm_result.get("sarcasm", "Not Detected"),
        "confidence": sarcasm_result.get("confidence", 0.60)
    }

    results["cleaned_text"] = {
        "prediction": cleaned_text,
        "confidence": 1.0
    }

    return results

if __name__ == "__main__":
    sample_text = input("Enter vaccine-related text: ")
    
    output = predict_all(sample_text)
    
    print("\nPrediction Results")
    print("-" * 40)
    
    for task, result in output.items():
        conf = result["confidence"]
        conf_text = f"{conf:.2f}" if conf is not None else "N/A"
        print(f"{task.capitalize()}: {result['prediction']} | Confidence: {conf_text}")