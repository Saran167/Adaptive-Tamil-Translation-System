import streamlit as st
from transformers import MarianMTModel, MarianTokenizer
from langdetect import detect

st.set_page_config(page_title="Adaptive Tamil Translation", layout="centered")

st.title("📝 Adaptive Tamil Translation System")
st.write("English → Tamil Translation (CPU Safe)")

MODEL_NAME = "Helsinki-NLP/opus-mt-en-ta"

@st.cache_resource(show_spinner=True)
def load_model():
    st.write("🔄 Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    st.write("🔄 Loading model (CPU)...")
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    model.to("cpu")

    st.write("✅ Model loaded successfully")
    return tokenizer, model

tokenizer, model = load_model()

text = st.text_area("Enter English text:")

if st.button("Translate"):
    if text.strip() == "":
        st.warning("Please enter text")
    else:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        translated = model.generate(**inputs, max_length=200)
        tamil = tokenizer.decode(translated[0], skip_special_tokens=True)

        st.success("Translated Text:")
        st.write(tamil)


