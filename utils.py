from PyPDF2 import PdfReader

def extract_answers(file_bytes):
    # Your logic to extract answers from PDF
    return ["answer1", "answer2", ...]

def compute_similarity(model, answers):
    # Your logic to compare answers with ideal answer
    results = []
    ideal_answer = "Correct reference answer here"
    for i, ans in enumerate(answers):
        sim = model.similarity(ideal_answer, ans)  # or cosine sim
        results.append({
            "student": f"Student {i+1}",
            "answer": ans,
            "similarity": round(sim * 100, 2)
        })
    return results
