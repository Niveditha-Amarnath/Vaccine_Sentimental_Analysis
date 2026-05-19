def detect_sarcasm(text):
    text = text.lower()

    sarcasm_patterns = [
        "wow great",
        "yeah right",
        "as if",
        "totally safe",
        "what a joke",
        "amazing side effects",
        "wonderful side effects",
        "so trustworthy",
        "great job hiding",
        "sure, vaccines are perfect"
    ]

    for pattern in sarcasm_patterns:
        if pattern in text:
            return {
                "sarcasm": "Detected",
                "confidence": 0.85
            }

    return {
        "sarcasm": "Not Detected",
        "confidence": 0.60
    }