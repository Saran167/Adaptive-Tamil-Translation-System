import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from langdetect import detect
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import pandas as pd
import os
import uuid

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Adaptive Tamil Translation System",
    page_icon="🧠",
    layout="wide"
)

# Register Tamil font for PDF
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# ---------------- UTILS ----------------
def translate_to_tamil(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    generated = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id["tam_Taml"],
        max_length=512
    )
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def simplify_tamil(text):
    # simple rule-based simplification (research-friendly, explainable)
    replacements = {
        "செயல்படுத்தப்படுகிறது": "செய்கிறது",
        "மாற்றத்தை ஏற்படுத்துகிறது": "மாற்றுகிறது",
        "நடந்து கொள்ள மாட்டாள்": "நல்லபடியாக நடக்க மாட்டாள்"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def create_pdf(tamil_text):
    file_name = f"translation_{uuid.uuid4().hex}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "HeiseiMin-W3"
    doc.build([Paragraph(tamil_text, styles["Normal"])])
    return file_name

def create_audio(tamil_text):
    audio_file = f"audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=tamil_text, lang="ta")
    tts.save(audio_file)
    return audio_file

def save_feedback(text, feedback, ces):
    df = pd.DataFrame([{
        "input_text": text,
        "feedback": feedback,
        "CES": ces
    }])
    if os.path.exists("feedback.csv"):
        df.to_csv("feedback.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("feedback.csv", index=False)

# ---------------- UI ----------------
st.markdown(
    """
    <style>
    .main {
        background-color: #f4f6ff;
    }
    .block {
        background: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 2px 2px 12px #ccc;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🧠 Adaptive Tamil Translation System")
st.subheader("Research-Oriented | User-Adaptive | Comprehension-Driven")

input_text = st.text_area("✍️ Enter text in ANY language", height=150)

if "ces" not in st.session_state:
    st.session_state.ces = 0

if st.button("🌍 Translate to Tamil"):
    if input_text.strip() == "":
        st.warning("Please enter text")
    else:
        tamil_output = translate_to_tamil(input_text)
        st.session_state.tamil = tamil_output
        st.session_state.ces = 0

if "tamil" in st.session_state:
    st.markdown("### 📘 Tamil Translation")
    st.success(st.session_state.tamil)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Not Clear – Simplify Again"):
            st.session_state.tamil = simplify_tamil(st.session_state.tamil)
            st.session_state.ces += 1

    with col2:
        if st.button("👍 Clear & Good"):
            save_feedback(input_text, "Good", st.session_state.ces)
            st.success("Feedback recorded ✅")

    with col3:
        if st.button("👎 Not Clear"):
            save_feedback(input_text, "Not Clear", st.session_state.ces)
            st.warning("Feedback recorded")

    st.markdown(f"**🧮 Comprehension Effort Score (CES):** `{st.session_state.ces}`")

    # Downloads
    pdf_file = create_pdf(st.session_state.tamil)
    audio_file = create_audio(st.session_state.tamil)

    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Tamil PDF", f, file_name="Tamil_Translation.pdf")

    with open(audio_file, "rb") as f:
        st.download_button("🔊 Download Tamil Audio", f, file_name="Tamil_Audio.mp3")
