import pickle
from utils.text_preprocessor import preprocess_text, detect_language
from utils.sarcasm_detector import detect_sarcasm

MODEL_PATHS = {
    "sentiment": "models/sentiment_model.pkl",
    "emotion": "models/emotion_model.pkl",
    "misinformation": "models/misinformation_model.pkl",
    "topic": "models/topic_model.pkl"
}

models = {}

for task, path in MODEL_PATHS.items():
    with open(path, "rb") as f:
        models[task] = pickle.load(f)

def predict_all(text):
    cleaned_text = preprocess_text(text)
    language = detect_language(text)
    sarcasm_result = detect_sarcasm(text)

    results = {}

    for task, model in models.items():
        prediction = model.predict([cleaned_text])[0]

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([cleaned_text])[0]
            confidence = max(probabilities)
        else:
            confidence = None

        results[task] = {
            "prediction": prediction,
            "confidence": confidence
        }

    results["language"] = {
        "prediction": language,
        "confidence": 1.0
    }

    results["cleaned_text"] = {
        "prediction": cleaned_text,
        "confidence": 1.0
    }
    results["sarcasm"] = {
        "prediction": sarcasm_result["sarcasm"],
        "confidence": sarcasm_result["confidence"]
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