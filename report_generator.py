from fpdf import FPDF
from datetime import datetime
import os

def generate_report():
    os.makedirs("outputs", exist_ok=True)

    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "M-VaxSentXAI Analysis Report", ln=True, align="C")

    pdf.ln(10)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Project Summary", ln=True)

    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8,
        "M-VaxSentXAI is a vaccine sentiment and misinformation analysis system. "
        "It performs sentiment analysis, emotion detection, misinformation detection, "
        "topic classification, sarcasm detection, language detection, and explainable AI."
    )

    pdf.ln(5)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Research Gaps Addressed", ln=True)

    gaps = [
        "Sentiment-only classification",
        "Lack of misinformation detection",
        "Lack of explainable AI",
        "Poor handling of code-mixed text",
        "No real-time monitoring",
        "Difficulty detecting sarcasm"
    ]

    pdf.set_font("Arial", "", 12)

    for gap in gaps:
        pdf.cell(0, 8, f"- {gap}", ln=True)

    output_path = "outputs/MVaxSentXAI_Report.pdf"
    pdf.output(output_path)

    return output_path