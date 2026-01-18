import streamlit as st
import torch
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect
from gtts import gTTS
import tempfile
import os

# -------------------------------
# Streamlit Page Config
# -------------------------------
st.set_page_config(
    page_title="Adaptive Tamil Translation System",
    layout="centered"
)

st.title("📝 Adaptive Tamil Translation System")
st.write("Translate English text into **Tamil** using AI")

# -------------------------------
# Device (CPU ONLY)
# -------------------------------
device = torch.device("cpu")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    model_name = "Helsinki-NLP/opus-mt-en-ta"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# -------------------------------
# User Input
# -------------------------------
input_text = st.text_area(
    "Enter English Text:",
    height=150,
    placeholder="Type something in English..."
)

# -------------------------------
# Translation Function
# -------------------------------
def translate_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    translated = model.generate(**inputs)
    tamil_text = tokenizer.decode(translated[0], skip_special_tokens=True)
    return tamil_text

# -------------------------------
# Translate Button
# -------------------------------
if st.button("Translate to Tamil"):
    if input_text.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            lang = detect(input_text)
            if lang != "en":
                st.warning("Please enter **English** text only.")
            else:
                with st.spinner("Translating..."):
                    tamil_output = translate_text(input_text)

                st.success("Translation Completed ✅")
                st.text_area(
                    "Tamil Translation:",
                    value=tamil_output,
                    height=150
                )

                # -------------------------------
                # Text to Speech (Tamil Audio)
                # -------------------------------
                tts = gTTS(text=tamil_output, lang="ta")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    audio_file = fp.name

                st.audio(audio_file, format="audio/mp3")

        except Exception as e:
            st.error(f"Error: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & HuggingFace")



