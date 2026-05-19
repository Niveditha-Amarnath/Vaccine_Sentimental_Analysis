import os
import pandas as pd
from datetime import datetime

LOG_FILE = "outputs/prediction_history.csv"

def save_prediction(text, results):
    os.makedirs("outputs", exist_ok=True)

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_text": text,
        "cleaned_text": results["cleaned_text"]["prediction"],
        "language": results["language"]["prediction"],
        "sentiment": results["sentiment"]["prediction"],
        "sentiment_confidence": results["sentiment"]["confidence"],
        "emotion": results["emotion"]["prediction"],
        "emotion_confidence": results["emotion"]["confidence"],
        "misinformation": results["misinformation"]["prediction"],
        "misinformation_confidence": results["misinformation"]["confidence"],
        "topic": results["topic"]["prediction"],
        "topic_confidence": results["topic"]["confidence"],
        "sarcasm": results["sarcasm"]["prediction"],
        "sarcasm_confidence": results["sarcasm"]["confidence"]
    }

    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(LOG_FILE, index=False)