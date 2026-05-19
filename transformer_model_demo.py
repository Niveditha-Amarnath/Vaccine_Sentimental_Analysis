from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def transformer_predict(text):
    result = classifier(text)[0]

    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        sentiment = "Positive"
    else:
        sentiment = "Negative"

    return {
        "sentiment": sentiment,
        "confidence": score
    }

if __name__ == "__main__":
    text = input("Enter vaccine-related text: ")
    result = transformer_predict(text)

    print("\nTransformer Prediction")
    print("-" * 40)
    print("Sentiment:", result["sentiment"])
    print("Confidence:", round(result["confidence"], 2))