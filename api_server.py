from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF
import os

app = FastAPI()

# Load fine-tuned model from HuggingFace
model = SentenceTransformer("tejuparab551/fine-tuned-similarity-model")


def extract_answers_from_pdf(file_path):
    doc = fitz.open(file_path)
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


@app.post("/check_similarity")
async def check_similarity(file: UploadFile = File(...)):
    file_location = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)

    with open(file_location, "wb") as f:
        f.write(await file.read())

    correct_answer, student_answers = extract_answers_from_pdf(file_location)
    os.remove(file_location)  # clean up temp file

    if not correct_answer or len(student_answers) == 0:
        return JSONResponse(content={"error": "PDF format is incorrect"}, status_code=400)

    results = []
    correct_embedding = model.encode(correct_answer, convert_to_tensor=True)

    for idx, answer in enumerate(student_answers):
        student_embedding = model.encode(answer, convert_to_tensor=True)
        score = util.cos_sim(correct_embedding, student_embedding).item() * 100
        results.append({
            "student": f"Student {idx+1}",
            "answer": answer,
            "similarity": round(score, 2)
        })

    return JSONResponse(content={"results": results})
