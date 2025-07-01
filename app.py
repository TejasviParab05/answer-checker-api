import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os

# Load your fine-tuned model
model = SentenceTransformer("fine-tuned-similarity-model")

# Extract correct and student answers from PDF
def extract_answers_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    lines = text.strip().split("\n")
    correct_answer = ""
    student_answers = []

    for line in lines:
        if line.lower().startswith("correct answer:"):
            correct_answer = line.split(":", 1)[1].strip()
        elif line.lower().startswith("student"):
            student_answers.append(line.split(":", 1)[1].strip())

    return correct_answer, student_answers

# UI Setup
st.set_page_config(page_title="Answer Similarity Checker", layout="centered")
st.title(" Answer Similarity Checker")
st.markdown("Upload a PDF:")

# Upload PDF
uploaded_file = st.file_uploader(" Upload PDF", type="pdf")

if uploaded_file:
    correct_answer, student_answers = extract_answers_from_pdf(uploaded_file)

    if not correct_answer or len(student_answers) == 0:
        st.error(" PDF format is incorrect. Please check the formatting.")
    else:
        #st.success(f" Found 1 correct answer and {len(student_answers)} student answers.")

        correct_emb = model.encode(correct_answer, convert_to_tensor=True)

        st.markdown(f"###  Correct Answer")
        st.info(correct_answer)

        st.markdown("###  Similarity Scores")
        for idx, student in enumerate(student_answers):
            student_emb = model.encode(student, convert_to_tensor=True)
            score = util.cos_sim(correct_emb, student_emb).item() * 100
            st.write(f"**Student {idx + 1}** - Similarity: `{score:.2f}%`")
            st.caption(student)
