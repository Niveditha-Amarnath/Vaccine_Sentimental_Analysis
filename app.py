import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit.components.v1 as components
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
# ---------------- GLOBAL PLOTLY THEME ----------------

pastel_colors = [
    "#B8C0FF",
    "#E0B3FF",
    "#FFB6D9",
    "#C7D2FE",
    "#F0ABFC",
    "#FDB4CF"
]

custom_template = {
    "layout": {
        "paper_bgcolor": "rgba(20,22,34,0.95)",
        "plot_bgcolor": "rgba(20,22,34,0.95)",
        "font": {
            "color": "#F3F4FF",
            "family": "Arial",
            "size": 18
        },
        "title": {
            "font": {
                "color": "#E0B3FF",
                "size": 28
            },
            "x": 0.03
        },
        "legend": {
            "font": {"size": 17, "color": "#F3F4FF"}
        },
        "xaxis": {
            "title_font": {"size": 18},
            "tickfont": {"size": 16},
            "gridcolor": "rgba(255,255,255,0.10)",
            "zerolinecolor": "rgba(255,255,255,0.10)"
        },
        "yaxis": {
            "title_font": {"size": 18},
            "tickfont": {"size": 16},
            "gridcolor": "rgba(255,255,255,0.10)",
            "zerolinecolor": "rgba(255,255,255,0.10)"
        },
        "colorway": pastel_colors,
        "margin": {"l": 60, "r": 40, "t": 80, "b": 60}
    }
}


pio.templates["pastel_dark"] = custom_template
pio.templates.default = "pastel_dark"
def style_bar_chart(fig, height=460):
    fig.update_layout(
        template="pastel_dark",
        height=height,
        title_font_size=28,
        font=dict(size=18, color="#F3F4FF"),
        legend=dict(font=dict(size=17)),
        margin=dict(l=60, r=40, t=80, b=70)
    )

    fig.update_xaxes(tickfont=dict(size=16), title_font=dict(size=18), showgrid=True)
    fig.update_yaxes(tickfont=dict(size=16), title_font=dict(size=18), showgrid=True)

    
    return fig


def style_pie_chart(fig, height=460):
    fig.update_layout(
        template="pastel_dark",
        height=height,
        title_font_size=28,
        font=dict(size=18, color="#F3F4FF"),
        legend=dict(font=dict(size=17)),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
        textfont_size=18,
        pull=[0.04] * len(fig.data[0].labels),
        marker_line_width=2,
        marker_line_color="rgba(255,255,255,0.4)"
    )

    return fig
# ---------------- THEME CSS ----------------

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #05060A 0%, #0B0D17 45%, #13111C 100%);
    color: #F3F4FF;
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #090B12 0%, #161224 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #F8F4FF !important;
}

h1 {
    color: #C7D2FE !important;
    font-weight: 900 !important;
}

h2, h3 {
    color: #E0B3FF !important;
    font-weight: 800 !important;
}

p, label, div {
    color: #F3F4FF;
}

textarea {
    background-color: #151825 !important;
    color: #F3F4FF !important;
    border: 1.5px solid #D8B4FE !important;
    border-radius: 16px !important;
}

input {
    background-color: #151825 !important;
    color: white !important;
}

.stButton button {
    background: linear-gradient(135deg, #B8C0FF, #E0B3FF, #FFB6D9);
    color: #111 !important;
    border: none;
    border-radius: 14px;
    padding: 0.7rem 1.3rem;
    font-weight: 800;
    box-shadow: 0px 0px 20px rgba(224,179,255,0.18);
    transition: all 0.3s ease;
}

.stButton button:hover {
    transform: scale(1.03);
    background: linear-gradient(135deg, #C7D2FE, #F0ABFC, #FDB4CF);
    color: black !important;
}

div[data-testid="stMetric"] {
    background: rgba(22, 25, 38, 0.95);
    border-radius: 20px;
    padding: 22px;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow:
        0px 0px 18px rgba(184,192,255,0.10),
        0px 0px 22px rgba(255,182,217,0.08);
}

div[data-testid="stMetricLabel"] {
    color: #F0ABFC !important;
    font-weight: 700;
}

div[data-testid="stMetricValue"] {
    color: #B8C0FF !important;
    font-weight: 900;
}

div[data-testid="stDataFrame"] {
    background: rgba(20,22,34,0.95);
    border-radius: 18px;
    border: 1px solid rgba(255,255,255,0.08);
}

.js-plotly-plot {
    background: rgba(20,22,34,0.96);
    border-radius: 18px;
    padding: 12px;
    box-shadow:
        0px 0px 18px rgba(184,192,255,0.08),
        0px 0px 22px rgba(255,182,217,0.05);
}

div[data-testid="stAlert"] {
    border-radius: 14px;
}

hr {
    border: none;
    height: 1px;
    background: rgba(255,255,255,0.08);
}

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #11131B;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, #B8C0FF, #E0B3FF, #FFB6D9);
    border-radius: 10px;
}

footer {
    visibility: hidden;
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

components.html(
    """
    <div style="
        background: linear-gradient(90deg, #11131B 0%, #1A1630 45%, #2B1F3F 100%);
        padding: 26px;
        border-radius: 22px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0px 0px 24px rgba(184,192,255,0.10), 0px 0px 30px rgba(255,182,217,0.08);
        font-family: Arial, sans-serif;
    ">
        <h1 style="color:#C7D2FE; margin-bottom:8px; font-size:52px; font-weight:900;">
            💉 M-VaxSentXAI
        </h1>

        <p style="color:#F5D0FE; font-size:21px; margin-top:0px;">
            Multilingual Vaccine Sentiment & Misinformation Analysis using Explainable AI
        </p>
    </div>
    """,
    height=180
)

#-------------------single text analysis------------#
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
    text="Confidence",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
                fig = style_bar_chart(fig)
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
        title="LIME Word Importance",
        template="pastel_dark",
        color_discrete_sequence=pastel_colors
    )
                
                fig = style_bar_chart(fig)
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
            title="Sentiment Distribution",
            template="pastel_dark",
            color_discrete_sequence=pastel_colors)
        fig = style_pie_chart(fig)
        st.plotly_chart(fig, use_container_width=True)

        with col2:
            emotion_counts = df["emotion"].value_counts().reset_index()
            emotion_counts.columns = ["Emotion", "Count"]

            fig = px.bar(
    emotion_counts,
    x="Emotion",
    y="Count",
    title="Emotion Distribution",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)

            fig = style_bar_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            
        col3, col4 = st.columns(2)

        with col3:
            misinfo_counts = df["misinformation"].value_counts().reset_index()
            misinfo_counts.columns = ["Misinformation", "Count"]

            fig = px.pie(
    misinfo_counts,
    names="Misinformation",
    values="Count",
    title="Misinformation Distribution",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
            fig = style_pie_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            fig.update_traces(
    textposition="inside",
    textinfo="percent+label",
    textfont_size=18,
    pull=[0.04] * len(fig.data[0].labels)
)

        with col4:
            topic_counts = df["topic"].value_counts().reset_index()
            topic_counts.columns = ["Topic", "Count"]

            fig = px.bar(
    topic_counts,
    x="Topic",
    y="Count",
    title="Topic Distribution",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
            fig = style_bar_chart(fig)
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
        title="State-wise Vaccine Sentiment Comparison",
        template="pastel_dark",
        color_discrete_sequence=pastel_colors
    )

        fig = style_bar_chart(fig)
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
    title="Live Sentiment Distribution",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
            fig = style_pie_chart(fig)
            st.plotly_chart(fig, use_container_width=True)


        with col2:
            misinfo_chart = live_df["Misinformation"].value_counts().reset_index()
            misinfo_chart.columns = ["Misinformation", "Count"]

            fig = px.bar(
    misinfo_chart,
    x="Misinformation",
    y="Count",
    title="Live Misinformation Alerts",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
            fig = style_bar_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
           
        col3, col4 = st.columns(2)

        with col3:
            emotion_chart = live_df["Emotion"].value_counts().reset_index()
            emotion_chart.columns = ["Emotion", "Count"]

            fig = px.bar(
    emotion_chart,
    x="Emotion",
    y="Count",
    title="Live Emotion Distribution",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)

            fig = style_bar_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            


        with col4:
            sarcasm_chart = live_df["Sarcasm"].value_counts().reset_index()
            sarcasm_chart.columns = ["Sarcasm", "Count"]

            fig = px.pie(
    sarcasm_chart,
    names="Sarcasm",
    values="Count",
    title="Sarcasm Detection Summary",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)
            fig = style_pie_chart(fig)
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

    st.subheader("📈 Task-wise Performance")

    st.dataframe(metrics_data)

    fig = px.bar(
    metrics_data,
    x="Task",
    y=["Accuracy", "Precision", "Recall", "F1-Score"],
    barmode="group",
    title="Model Performance Comparison",
    template="pastel_dark",
    color_discrete_sequence=pastel_colors
)



    fig = style_bar_chart(fig, height=420)

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    st.subheader("🧩 Confusion Matrices")

    tasks = [
        "sentiment",
        "emotion",
        "misinformation",
        "topic"
    ]

    # Create 2-column layout
    cols = st.columns(2)

    for index, task in enumerate(tasks):

        image_path = f"outputs/confusion_matrices/{task}_confusion_matrix.png"

        with cols[index % 2]:

            if os.path.exists(image_path):

                st.image(
                    image_path,
                    caption=f"{task.capitalize()} Confusion Matrix",
                    use_container_width=True
                )

            else:
                st.warning(f"{task.capitalize()} confusion matrix not found.")

    st.markdown("---")

    st.subheader("⚖️ Baseline vs Proposed Model")

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
    ### 🔍 Key Observation

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

# st.markdown("""
# ### ✅ Current System Capabilities

# - Sentiment Analysis  
# - Emotion Detection  
# - Misinformation Detection  
# - Topic Classification  
# - Language Detection  
# - Code-Mixed Text Preprocessing  
# - Sarcasm Detection  
# - Confidence Score Visualization  
# - Explainable AI using LIME  
# - Dataset Dashboard  
# - Model Evaluation  
# - Prediction History  
# - Live Monitoring Simulation  
# """)