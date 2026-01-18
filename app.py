import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from gtts import gTTS
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
import pandas as pd
import uuid
import os

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Adaptive Tamil Translation System",
    page_icon="🧠",
    layout="wide"
)

# --------------------------------------------------
# REGISTER TAMIL FONT FOR PDF
# --------------------------------------------------
pdfmetrics.registerFont(UnicodeCIDFont("HeiseiMin-W3"))

# --------------------------------------------------
# LOAD TRANSLATION MODEL (LIGHTWEIGHT & SAFE)
# --------------------------------------------------
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-mul-ta"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# --------------------------------------------------
# CORE FUNCTIONS
# --------------------------------------------------
def translate_to_tamil(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    output = model.generate(**inputs, max_length=512)
    return tokenizer.decode(output[0], skip_special_tokens=True)

def simplify_tamil(text):
    rules = {
        "மாற்றத்தை ஏற்படுத்துகிறது": "மாற்றுகிறது",
        "செயல்படுத்தப்படுகிறது": "செய்கிறது",
        "நடந்து கொள்ள மாட்டாள்": "நன்றாக நடந்து கொள்ள மாட்டாள்"
    }
    for k, v in rules.items():
        text = text.replace(k, v)
    return text

def create_pdf(tamil_text):
    file_name = f"tamil_translation_{uuid.uuid4().hex}.pdf"
    doc = SimpleDocTemplate(file_name)
    styles = getSampleStyleSheet()
    styles["Normal"].fontName = "HeiseiMin-W3"
    doc.build([Paragraph(tamil_text, styles["Normal"])])
    return file_name

def create_audio(tamil_text):
    audio_file = f"tamil_audio_{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=tamil_text, lang="ta")
    tts.save(audio_file)
    return audio_file

def save_feedback(text, status, ces):
    row = pd.DataFrame([{
        "Input Text": text,
        "Feedback": status,
        "CES": ces
    }])
    if os.path.exists("feedback.csv"):
        row.to_csv("feedback.csv", mode="a", header=False, index=False)
    else:
        row.to_csv("feedback.csv", index=False)

# --------------------------------------------------
# UI DESIGN
# --------------------------------------------------
st.markdown("""
<style>
body {
    background-color: #eef2ff;
}
.block {
    background: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

st.title("🧠 Adaptive Tamil Translation System")
st.caption("Human-in-the-Loop | Comprehension-Driven | Research-Oriented")

st.markdown("---")

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
input_text = st.text_area(
    "✍️ Enter text in ANY language",
    height=150,
    placeholder="Type English, Hindi, Malayalam, etc..."
)

if "tamil" not in st.session_state:
    st.session_state.tamil = ""
if "ces" not in st.session_state:
    st.session_state.ces = 0

# --------------------------------------------------
# TRANSLATE BUTTON
# --------------------------------------------------
if st.button("🌍 Translate to Tamil"):
    if input_text.strip() == "":
        st.warning("Please enter some text")
    else:
        st.session_state.tamil = translate_to_tamil(input_text)
        st.session_state.ces = 0

# --------------------------------------------------
# OUTPUT SECTION
# --------------------------------------------------
if st.session_state.tamil:
    st.markdown("### 📘 Tamil Translation")
    st.success(st.session_state.tamil)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Not Clear – Simplify"):
            st.session_state.tamil = simplify_tamil(st.session_state.tamil)
            st.session_state.ces += 1

    with col2:
        if st.button("👍 Clear & Good"):
            save_feedback(input_text, "Good", st.session_state.ces)
            st.success("Feedback saved")

    with col3:
        if st.button("👎 Not Clear"):
            save_feedback(input_text, "Not Clear", st.session_state.ces)
            st.warning("Feedback saved")

    st.markdown(f"**🧮 Comprehension Effort Score (CES):** `{st.session_state.ces}`")

    # --------------------------------------------------
    # DOWNLOADS
    # --------------------------------------------------
    pdf_file = create_pdf(st.session_state.tamil)
    audio_file = create_audio(st.session_state.tamil)

    with open(pdf_file, "rb") as f:
        st.download_button("📄 Download Tamil PDF", f, file_name="Tamil_Translation.pdf")

    with open(audio_file, "rb") as f:
        st.download_button("🔊 Download Tamil Audio", f, file_name="Tamil_Audio.mp3")

