# M-VaxSentXAI

## Multilingual Vaccine Sentiment & Misinformation Analysis using Explainable AI

M-VaxSentXAI is an NLP-based healthcare analytics system that analyzes vaccine-related social media text and predicts sentiment, emotion, misinformation risk, discussion topic, language type, and sarcasm presence.

---

## Project Objective

The objective of this project is to overcome limitations in existing vaccine sentiment analysis studies by building a multi-task NLP system that supports:

- Sentiment Analysis
- Emotion Detection
- Misinformation Detection
- Topic Classification
- Language Detection
- Code-Mixed Text Handling
- Sarcasm Detection
- Explainable AI using LIME
- Dashboard Analytics
- Real-Time Monitoring Simulation

---

## Problem Statement

Existing vaccine sentiment analysis studies mainly focus on simple positive, negative, and neutral classification. Most systems are limited to English-only data, single-platform datasets, static analysis, and lack misinformation detection or explainability.

This project proposes M-VaxSentXAI to analyze vaccine-related Indian social media discussions more effectively.

---

## Features

- Multi-task prediction system
- Vaccine sentiment classification
- Emotion recognition
- Misinformation risk prediction
- Topic detection
- Language detection
- Code-mixed preprocessing
- Sarcasm detection
- LIME-based explainability
- Confidence score visualization
- Dataset dashboard
- Live monitoring simulation
- Prediction history storage

---

## Model Used

### Baseline Working Model

The current working implementation uses:

- TF-IDF Vectorization
- Logistic Regression
- Four separate classifiers:
  - Sentiment classifier
  - Emotion classifier
  - Misinformation classifier
  - Topic classifier

### Proposed Advanced Model

The proposed future model architecture includes:

- MuRIL / IndicBERT
- BERTweet
- BiLSTM
- Attention Mechanism
- Multi-task classification heads
- SHAP / LIME explainability

---

## Dataset Format

The enhanced dataset contains:

```csv
tweet,label,emotion,misinformation,topic,language
```

Example:

"Vaccine safe hai but side effects scary",negative,Anxiety,Misleading,Side effects,Code-mixed
Project Structure
vaccine_sentiment/
│
├── app.py
├── train_multitask.py
├── predict_multitask.py
├── upgrade_dataset.py
├── prediction_logger.py
├── xai_explainer.py
├── enhanced_dataset.csv
├── dataset.csv
├── requirements.txt
│
├── models/
│ ├── sentiment_model.pkl
│ ├── emotion_model.pkl
│ ├── misinformation_model.pkl
│ └── topic_model.pkl
│
├── utils/
│ ├── text_preprocessor.py
│ └── sarcasm_detector.py
│
└── outputs/
├── results.txt
└── prediction_history.csv
Installation

Create virtual environment:

python -m venv venv

Activate virtual environment:

.\venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Install extra dependencies:

pip install streamlit plotly lime emoji
How to Run

Train models:

python train_multitask.py

Run app:

python -m streamlit run app.py
Sample Input
Vaccine safe hai but side effects scary
Sample Output
Sentiment: negative
Emotion: Anxiety
Misinformation: Misleading
Topic: Side effects
Language: Code-mixed
Sarcasm: Not Detected
Research Gaps Addressed
Research Gap Project Solution
English-only analysis Code-mixed preprocessing
Sentiment-only classification Multi-task prediction
No misinformation detection Dedicated misinformation classifier
No explainability LIME explanation
Static analysis Live monitoring simulation
Poor context handling Sarcasm detector
No prediction history CSV logging system
Future Scope
Add MuRIL / IndicBERT transformer model
Add SHAP explanations
Add real-time API streaming
Add Twitter/Reddit/YouTube scraping
Add Indian regional language datasets
Add geospatial sentiment analysis
Add FastAPI backend
Add React dashboard
Conclusion

M-VaxSentXAI provides a practical and extensible vaccine sentiment analysis system that improves upon traditional sentiment analysis by integrating emotion detection, misinformation prediction, topic classification, explainability, and live monitoring simulation.

---

# STEP 36 — Final Check Commands

Run these:

```bash
python train_multitask.py
python -m streamlit run app.py

Test:

Vaccine safe hai but side effects scary
Wow great, another vaccine side effect
Vaccines contain tracking chips
```
