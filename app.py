import os; os.system('apt-get install tesseract-ocr libgl1 -qq')
import streamlit as st
import os
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pdf2image import convert_from_bytes
import openai

# --- Configuration ---
st.set_page_config(layout="wide")
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# --- Session State ---
if "textbooks" not in st.session_state:
    st.session_state.textbooks = {}

# --- OCR Functions ---
def preprocess_image(img):
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def extract_text(uploaded_file):
    images = convert_from_bytes(uploaded_file.read())
    text = ""
    for img in images:
        processed = preprocess_image(img)
        text += pytesseract.image_to_string(processed) + "\n\n"  # ← 4 spaces before
    return text

# --- Q&A Function ---
def query_textbook(question, text_content, model="gpt-4-turbo"):
    prompt = f'''Answer ONLY using this text. Say "Not found" if unsure.
    TEXTBOOK CONTENT:
    {text_content[:50000]}
    QUESTION: {question}
    ANSWER:'''
    response = openai.ChatCompletion.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

# --- Streamlit UI ---
st.title("📖 AI Textbook Tutor")
api_key = st.text_input("Enter OpenAI API Key:", type="password")
openai.api_key = api_key

# Textbook Uploader
uploaded_file = st.file_uploader("Upload Textbook (PDF)", type="pdf")
if uploaded_file and uploaded_file.name not in st.session_state.textbooks:
    with st.spinner("Extracting text..."):
        text = extract_text(uploaded_file)
        st.session_state.textbooks[uploaded_file.name] = text
    st.success(f"✅ {uploaded_file.name} loaded!")

# Q&A Interface
if st.session_state.textbooks:
    selected_book = st.selectbox("Select Textbook", list(st.session_state.textbooks.keys()))
    question = st.text_input("Ask a question about the textbook:")
    if question:
        answer = query_textbook(question, st.session_state.textbooks[selected_book])
        st.markdown(f"**Answer:** {answer}")

# Exam Prep Generator
if st.button("Generate Flashcards"):
    flashcards = query_textbook("Create 5 flashcards from this text", 
                              st.session_state.textbooks[selected_book])
    st.code(flashcards)
    
