import pickle
from lime.lime_text import LimeTextExplainer

SENTIMENT_LABELS = ["negative", "neutral", "positive"]

with open("models/sentiment_model.pkl", "rb") as f:
    sentiment_model = pickle.load(f)

def explain_sentiment(text):
    explainer = LimeTextExplainer(class_names=SENTIMENT_LABELS)

    explanation = explainer.explain_instance(
        text,
        sentiment_model.predict_proba,
        num_features=8
    )

    return explanation.as_list()