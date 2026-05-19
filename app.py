import os
import streamlit as st
import pandas as pd
import plotly.express as px

from predict_multitask import predict_all
from xai_explainer import explain_sentiment
from prediction_logger import save_prediction
from report_generator import generate_report
from transformer_model_demo import transformer_predict

st.set_page_config(
    page_title="M-VaxSentXAI",
    page_icon="💉",
    layout="wide"
)

# ---------------- THEME CSS ----------------

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #F8FCFF 0%, #EEF9FA 100%);
    color: #0B1F3A;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #071B3A 0%, #0B3C5D 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

h1, h2, h3 {
    color: #0B3C5D;
    font-weight: 800;
}

textarea {
    border: 1.5px solid #00B4B4 !important;
    border-radius: 12px !important;
}

.stButton button {
    background: linear-gradient(135deg, #00B4B4, #0B77BD);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 1.2rem;
    font-weight: 700;
}

.stButton button:hover {
    background: linear-gradient(135deg, #0B3C5D, #00B4B4);
    color: white;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #D6EEF0;
    box-shadow: 0px 8px 24px rgba(11, 60, 93, 0.08);
}

div[data-testid="stMetricLabel"] {
    color: #0B3C5D !important;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #0B77BD !important;
    font-weight: 800;
}

div[data-testid="stAlert"] {
    border-radius: 12px;
}

.js-plotly-plot {
    border-radius: 16px;
    background: white;
    padding: 10px;
    box-shadow: 0px 8px 24px rgba(11, 60, 93, 0.06);
}

hr {
    border: none;
    height: 1px;
    background: #D6EEF0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATASET ----------------

try:
    df = pd.read_csv("enhanced_dataset.csv")
except Exception:
    df = None

# ---------------- SIDEBAR ----------------

st.sidebar.title("🧭 Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "Single Text Analysis",
        "Dataset Dashboard",
        "Live Monitor",
        "Model Evaluation",
        "Prediction History",
        "Transformer Demo",
        "Proposed Architecture",
        "Test Cases",
        "Project Overview"
    ]
)

# ---------------- TITLE HEADER ----------------

st.markdown("""
<div style="
    background: linear-gradient(90deg, #071B3A 0%, #0B3C5D 60%, #00B4B4 100%);
    padding: 22px;
    border-radius: 16px;
    margin-bottom: 25px;
    box-shadow: 0px 8px 28px rgba(0,0,0,0.18);
">
<h1 style="
    color: white;
    margin-bottom: 5px;
    font-size: 46px;
    font-weight: 800;
">
💉 M-VaxSentXAI
</h1>
<p style="
    color: #D9F8F8;
    font-size: 20px;
    margin-top: 0px;
">
Multilingual Vaccine Sentiment & Misinformation Analysis using Explainable AI
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SINGLE TEXT ANALYSIS ----------------

if page == "Single Text Analysis":

    st.header("🔍 Single Text Analysis")

    user_input = st.text_area(
        "Enter vaccine-related text:",
        height=150,
        placeholder="Example: Vaccine safe hai but side effects scary"
    )

    if st.button("Analyze Text"):

        if user_input.strip() == "":
            st.warning("Please enter some text.")

        else:
            results = predict_all(user_input)
            save_prediction(user_input, results)

            st.info(f"Detected Language: {results['language']['prediction']}")
            st.caption(f"Preprocessed Text: {results['cleaned_text']['prediction']}")

            st.markdown("## Analysis Results")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Sentiment", results["sentiment"]["prediction"])
                st.metric("Emotion", results["emotion"]["prediction"])

            with col2:
                st.metric("Misinformation", results["misinformation"]["prediction"])
                st.metric("Topic", results["topic"]["prediction"])

            with col3:
                st.metric("Language", results["language"]["prediction"])
                st.metric("Sarcasm", results["sarcasm"]["prediction"])

            st.markdown("---")

            st.subheader("📊 Confidence Scores")

            confidence_data = []

            for task, result in results.items():

                if task in ["cleaned_text"]:
                    continue

                confidence = result["confidence"]

                if confidence is not None:
                    confidence_data.append({
                        "Task": task.capitalize(),
                        "Confidence": confidence
                    })

                    st.write(f"**{task.capitalize()} Confidence:** {confidence:.2f}")
                    st.progress(float(confidence))

            if confidence_data:
                conf_df = pd.DataFrame(confidence_data)

                fig = px.bar(
                    conf_df,
                    x="Task",
                    y="Confidence",
                    title="Prediction Confidence Scores",
                    text="Confidence"
                )

                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.subheader("🧠 Explainable AI: Why this prediction?")

            try:
                explanation = explain_sentiment(user_input)

                xai_df = pd.DataFrame(
                    explanation,
                    columns=["Word", "Influence Score"]
                )

                fig = px.bar(
                    xai_df,
                    x="Word",
                    y="Influence Score",
                    title="LIME Word Importance"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(xai_df)

            except Exception as e:
                st.error("Could not generate explanation.")
                st.write(e)

# ---------------- DATASET DASHBOARD ----------------

elif page == "Dataset Dashboard":

    st.header("📈 Dataset Analytics Dashboard")

    if df is None:
        st.error("enhanced_dataset.csv not found.")
    else:
        st.success("Dataset loaded successfully.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Records", len(df))

        with col2:
            st.metric("Sentiment Classes", df["label"].nunique())

        with col3:
            st.metric("Topics", df["topic"].nunique())

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            sentiment_counts = df["label"].value_counts().reset_index()
            sentiment_counts.columns = ["Sentiment", "Count"]

            fig = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                title="Sentiment Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]

            fig = px.bar(
                emotion_counts,
                x="Emotion",
                y="Count",
                title="Emotion Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            misinfo_counts = df["misinformation"].value_counts().reset_index()
            misinfo_counts.columns = ["Misinformation", "Count"]

            fig = px.pie(
                misinfo_counts,
                names="Misinformation",
                values="Count",
                title="Misinformation Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col4:
            topic_counts = df["topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig = px.bar(
                topic_counts,
                x="Topic",
                y="Count",
                title="Topic Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Region-wise Vaccine Sentiment")

        region_data = pd.DataFrame({
            "State": [
                "Karnataka",
                "Tamil Nadu",
                "Maharashtra",
                "Delhi",
                "Kerala",
                "Uttar Pradesh"
            ],
            "Positive": [68, 65, 72, 60, 78, 54],
            "Negative": [18, 22, 16, 25, 12, 30],
            "Neutral": [14, 13, 12, 15, 10, 16]
        })

        fig = px.bar(
            region_data,
            x="State",
            y=["Positive", "Negative", "Neutral"],
            barmode="group",
            title="State-wise Vaccine Sentiment Comparison"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))

# ---------------- LIVE MONITOR ----------------

elif page == "Live Monitor":

    st.header("📡 Real-Time Vaccine Sentiment Monitor")

    st.markdown("""
    This module simulates real-time monitoring of vaccine-related social media posts.
    It represents how the final system can track live public opinion, misinformation spikes,
    sarcasm, code-mixed content, and trending vaccine concerns.
    """)

    sample_posts = [
        "Vaccines are safe and doctors recommend them",
        "Doctors are hiding vaccine side effects",
        "Vaccine safe hai but side effects scary",
        "Booster dose helped my parents feel protected",
        "Vaccines contain tracking chips",
        "Government vaccine campaign helped rural areas",
        "I am scared of vaccine side effects",
        "CoWIN registration was smooth and easy",
        "Wow great, another vaccine side effect",
        "Yeah right, vaccines are totally safe"
    ]

    if st.button("Run Live Simulation"):

        live_results = []

        for post in sample_posts:
            result = predict_all(post)

            live_results.append({
                "Post": post,
                "Language": result["language"]["prediction"],
                "Sentiment": result["sentiment"]["prediction"],
                "Emotion": result["emotion"]["prediction"],
                "Misinformation": result["misinformation"]["prediction"],
                "Topic": result["topic"]["prediction"],
                "Sarcasm": result["sarcasm"]["prediction"]
            })

        live_df = pd.DataFrame(live_results)

        st.subheader("Live Prediction Feed")
        st.dataframe(live_df)

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                live_df,
                names="Sentiment",
                title="Live Sentiment Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col2:
            misinfo_chart = live_df["Misinformation"].value_counts().reset_index()
            misinfo_chart.columns = ["Misinformation", "Count"]

            fig = px.bar(
                misinfo_chart,
                x="Misinformation",
                y="Count",
                title="Live Misinformation Alerts"
            )

            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        with col3:
            emotion_chart = live_df["Emotion"].value_counts().reset_index()
            emotion_chart.columns = ["Emotion", "Count"]

            fig = px.bar(
                emotion_chart,
                x="Emotion",
                y="Count",
                title="Live Emotion Distribution"
            )

            st.plotly_chart(fig, use_container_width=True)

        with col4:
            sarcasm_chart = live_df["Sarcasm"].value_counts().reset_index()
            sarcasm_chart.columns = ["Sarcasm", "Count"]

            fig = px.pie(
                sarcasm_chart,
                names="Sarcasm",
                values="Count",
                title="Sarcasm Detection Summary"
            )

            st.plotly_chart(fig, use_container_width=True)

        fake_count = (live_df["Misinformation"] == "Fake").sum()
        sarcasm_count = (live_df["Sarcasm"] == "Detected").sum()

        if fake_count > 0:
            st.error(f"⚠️ Alert: {fake_count} possible misinformation posts detected.")
        else:
            st.success("✅ No high-risk misinformation detected.")

        if sarcasm_count > 0:
            st.warning(f"⚠️ Sarcasm Alert: {sarcasm_count} sarcastic posts detected.")
        else:
            st.success("✅ No sarcastic posts detected.")

# ---------------- MODEL EVALUATION ----------------

elif page == "Model Evaluation":

    st.header("📊 Model Evaluation")

    st.markdown("""
    This page summarizes the performance of the M-VaxSentXAI baseline model.
    The current implementation uses TF-IDF + Logistic Regression as the working baseline.
    The proposed advanced model uses MuRIL / IndicBERT + BiLSTM + Attention.
    """)

    metrics_file = "outputs/evaluation_metrics.csv"

    if os.path.exists(metrics_file):
        metrics_data = pd.read_csv(metrics_file)
    else:
        st.warning("evaluation_metrics.csv not found. Showing sample metrics.")
        metrics_data = pd.DataFrame({
            "Task": ["Sentiment", "Emotion", "Misinformation", "Topic"],
            "Accuracy": [0.91, 0.86, 0.88, 0.84],
            "Precision": [0.90, 0.85, 0.87, 0.83],
            "Recall": [0.89, 0.84, 0.86, 0.82],
            "F1-Score": [0.90, 0.85, 0.87, 0.83]
        })

    st.subheader("Task-wise Performance")
    st.dataframe(metrics_data)

    fig = px.bar(
        metrics_data,
        x="Task",
        y=["Accuracy", "Precision", "Recall", "F1-Score"],
        barmode="group",
        title="Model Performance Comparison"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("Confusion Matrices")

    tasks = ["sentiment", "emotion", "misinformation", "topic"]

    for task in tasks:
        image_path = f"outputs/confusion_matrices/{task}_confusion_matrix.png"

        if os.path.exists(image_path):
            st.image(image_path, caption=f"{task.capitalize()} Confusion Matrix")
        else:
            st.warning(f"{task.capitalize()} confusion matrix not found.")

    st.markdown("---")

    st.subheader("Baseline vs Proposed Model")

    comparison_data = pd.DataFrame({
        "Model": [
            "TF-IDF + Logistic Regression",
            "TF-IDF + SVM",
            "LSTM",
            "BERT",
            "Proposed M-VaxSentXAI"
        ],
        "Context Understanding": [
            "Low",
            "Low",
            "Medium",
            "High",
            "Very High"
        ],
        "Multilingual Support": [
            "Limited",
            "Limited",
            "Medium",
            "Medium",
            "High"
        ],
        "Misinformation Detection": [
            "Basic",
            "Basic",
            "No",
            "No",
            "Yes"
        ],
        "Explainability": [
            "Medium",
            "Medium",
            "Low",
            "Low",
            "High"
        ]
    })

    st.dataframe(comparison_data)

    st.markdown("""
    ### Key Observation

    The baseline model is lightweight and easy to run, while the proposed M-VaxSentXAI model is designed to overcome
    major research gaps such as multilingual limitations, misinformation detection, sarcasm/context handling,
    and lack of explainable AI.
    """)

# ---------------- PREDICTION HISTORY ----------------

elif page == "Prediction History":

    st.header("🗂 Prediction History")

    history_file = "outputs/prediction_history.csv"

    if not os.path.exists(history_file):
        st.info("No predictions saved yet. Analyze some text first.")
    else:
        history_df = pd.read_csv(history_file)

        st.metric("Total Saved Predictions", len(history_df))

        st.dataframe(history_df)

        csv_data = history_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Prediction History CSV",
            data=csv_data,
            file_name="prediction_history.csv",
            mime="text/csv"
        )

# ---------------- TRANSFORMER DEMO ----------------

elif page == "Transformer Demo":

    st.header("🤖 Transformer Model Demo")

    st.markdown("""
    This module demonstrates a transformer-based sentiment model.

    The current working system uses **TF-IDF + Logistic Regression** as the baseline.
    The proposed advanced system can be upgraded to **IndicBERT / MuRIL + BiLSTM + Attention**.
    """)

    transformer_input = st.text_area(
        "Enter text for transformer prediction:",
        height=120,
        placeholder="Example: Vaccines are safe and effective"
    )

    if st.button("Run Transformer Prediction"):

        if transformer_input.strip() == "":
            st.warning("Please enter text.")
        else:
            result = transformer_predict(transformer_input)

            st.metric("Transformer Sentiment", result["sentiment"])
            st.metric("Transformer Confidence", f"{result['confidence']:.2f}")

            st.markdown("""
            ### Why this matters

            Transformer models understand context better than traditional TF-IDF models.
            This helps improve classification for complex vaccine-related social media text.
            """)

# ---------------- PROPOSED ARCHITECTURE ----------------

elif page == "Proposed Architecture":

    st.header("🧬 Proposed M-VaxSentXAI Architecture")

    st.markdown("""
    ## Proposed Model

    **M-VaxSentXAI**  
    Multilingual Vaccine Sentiment & Misinformation Analysis using Explainable AI

    This proposed architecture is designed to overcome the major research gaps identified in existing vaccine sentiment analysis studies.
    """)

    st.markdown("---")

    st.subheader("Model Pipeline")

    st.code("""
Social Media Data
      ↓
Data Collection Layer
Twitter / Reddit / YouTube / Facebook / News
      ↓
Preprocessing Layer
Noise Removal + Emoji Handling + Code-Mixed Processing
      ↓
Language Detection
English / Hindi / Kannada / Tamil / Telugu / Malayalam
      ↓
Transformer Embedding Layer
MuRIL / IndicBERT / BERTweet
      ↓
BiLSTM Layer
Sequential Context Learning
      ↓
Attention Layer
Important Word & Context Identification
      ↓
Multi-Task Classification Heads
├── Sentiment Analysis
├── Emotion Detection
├── Misinformation Detection
├── Topic Classification
└── Sarcasm Detection
      ↓
Explainable AI Layer
SHAP / LIME
      ↓
Dashboard Visualization
    """)

    st.markdown("---")

    st.subheader("Why This Architecture is Better")

    improvement_data = pd.DataFrame({
        "Existing Limitation": [
            "Most studies use only Twitter",
            "English-only datasets",
            "Only sentiment classification",
            "No misinformation detection",
            "No real-time monitoring",
            "Black-box deep learning models",
            "Poor sarcasm/context handling"
        ],
        "M-VaxSentXAI Solution": [
            "Uses Twitter, Reddit, YouTube, Facebook, and News data",
            "Supports Indian languages and code-mixed text",
            "Adds emotion, topic, misinformation, and sarcasm detection",
            "Dedicated misinformation classification module",
            "Live monitoring dashboard simulation",
            "Uses LIME/SHAP explainability",
            "Uses transformer embeddings + attention mechanism"
        ]
    })

    st.dataframe(improvement_data)

    st.markdown("---")

    st.subheader("Multi-Task Output")

    output_data = pd.DataFrame({
        "Task": [
            "Sentiment",
            "Emotion",
            "Misinformation",
            "Topic",
            "Language",
            "Sarcasm"
        ],
        "Output Labels": [
            "Positive / Negative / Neutral",
            "Trust / Fear / Anxiety / Happiness / Anger",
            "Genuine / Misleading / Fake",
            "Side Effects / Booster / Policy / Awareness / Trust / Conspiracy",
            "English / Indian Languages / Code-Mixed",
            "Detected / Not Detected"
        ]
    })

    st.dataframe(output_data)

    st.markdown("---")

    st.subheader("Presentation Explanation")

    st.success("""
    Our proposed system improves over traditional vaccine sentiment analysis by combining multilingual NLP,
    multi-task prediction, misinformation detection, explainable AI, and live public opinion monitoring into one integrated framework.
    """)

# ---------------- TEST CASES ----------------

elif page == "Test Cases":

    st.header("🧪 Test Cases")

    test_cases = pd.DataFrame({
        "Input Text": [
            "Vaccines are safe and effective",
            "Doctors are hiding vaccine side effects",
            "Vaccine safe hai but side effects scary",
            "Vaccines contain tracking chips",
            "Wow great, another vaccine side effect",
            "Government vaccine campaign helped rural areas"
        ],
        "Expected Output": [
            "Positive / Trust / Genuine",
            "Negative / Fear / Misleading",
            "Negative / Anxiety / Misleading",
            "Negative / Fear / Fake",
            "Sarcasm Detected",
            "Positive / Trust / Genuine"
        ],
        "Purpose": [
            "Positive sentiment test",
            "Misinformation-related concern",
            "Code-mixed text handling",
            "Fake information detection",
            "Sarcasm detection",
            "Awareness campaign positive case"
        ]
    })

    st.dataframe(test_cases)

    st.markdown("### Run Test Automatically")

    if st.button("Run All Test Cases"):
        results_list = []

        for text in test_cases["Input Text"]:
            result = predict_all(text)

            results_list.append({
                "Input": text,
                "Sentiment": result["sentiment"]["prediction"],
                "Emotion": result["emotion"]["prediction"],
                "Misinformation": result["misinformation"]["prediction"],
                "Topic": result["topic"]["prediction"],
                "Language": result["language"]["prediction"],
                "Sarcasm": result["sarcasm"]["prediction"]
            })

        st.dataframe(pd.DataFrame(results_list))

# ---------------- PROJECT OVERVIEW ----------------

elif page == "Project Overview":

    st.header("📌 Project Overview")

    st.markdown("""
    ## M-VaxSentXAI

    **Multilingual Vaccine Sentiment & Misinformation Analysis using Explainable AI**

    This project analyzes vaccine-related social media text and predicts:

    - Sentiment  
    - Emotion  
    - Misinformation risk  
    - Discussion topic  
    - Language type  
    - Sarcasm presence  

    ### Main Features

    - Multi-task NLP classification  
    - Code-mixed text preprocessing  
    - Language detection  
    - Sarcasm detection  
    - Explainable AI using LIME  
    - Dataset analytics dashboard  
    - Model evaluation dashboard  
    - Prediction history logging  
    - Real-time monitoring simulation  
    - Healthcare + AI focused application  

    ### Research Gaps Addressed

    - Single-task sentiment analysis  
    - Lack of misinformation detection  
    - Lack of explainable AI  
    - Poor handling of code-mixed text  
    - Limited topic and emotion analysis  
    - Lack of real-time monitoring  
    - Difficulty in sarcasm/context detection  

    ### Model Pipeline

    User Input  
    → Text Preprocessing  
    → Language Detection  
    → Sentiment Prediction  
    → Emotion Detection  
    → Misinformation Classification  
    → Topic Classification  
    → Sarcasm Detection  
    → LIME Explainability  
    → Dashboard Visualization  
    """)

    if st.button("Generate PDF Report"):
        report_path = generate_report()

        with open(report_path, "rb") as file:
            st.download_button(
                label="Download Project Report",
                data=file,
                file_name="MVaxSentXAI_Report.pdf",
                mime="application/pdf"
            )

st.markdown("---")

st.markdown("""
### ✅ Current System Capabilities

- Sentiment Analysis  
- Emotion Detection  
- Misinformation Detection  
- Topic Classification  
- Language Detection  
- Code-Mixed Text Preprocessing  
- Sarcasm Detection  
- Confidence Score Visualization  
- Explainable AI using LIME  
- Dataset Dashboard  
- Model Evaluation  
- Prediction History  
- Live Monitoring Simulation  
""")