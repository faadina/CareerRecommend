from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import fitz
import spacy
import re
import traceback
import os
from scipy.sparse import load_npz, csr_matrix

app = Flask(__name__, template_folder='.')

# ---------- Load Models and Data ---------- #
try:
    print("[INFO] Loading BERT classifier...")
    model = BertForSequenceClassification.from_pretrained("./bert_finetuned_ori_90")
    tokenizer = BertTokenizer.from_pretrained("./bert_finetuned_ori_90")
    model.eval()

    print("[INFO] Loading SentenceTransformer...")
    bert_embedder = joblib.load("text/bert_embedder.joblib")

    print("[INFO] Loading job embeddings...")
    career_embeddings = np.load("text/career_embeddings.npy")

    print("[INFO] Normalizing embeddings...")
    scaler = MinMaxScaler()
    career_embeddings = scaler.fit_transform(career_embeddings)

    print("[INFO] Loading career data...")
    career_df = pd.read_csv("dataset/datacleanJobstreet.csv")
    nlp = spacy.load("en_core_web_sm")

except Exception as e:
    print("[ERROR] Model loading failed:", str(e))
    exit(1)

matched_jobs = []

# ---------- Resume Extraction ---------- #
def extract_resume_info(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = " ".join([page.get_text() for page in doc])
    doc.close()

    name = ""
    lines = text.strip().split('\n')
    for line in lines:
        if re.match(r"^[A-Z\s]{10,}$", line) and "BINTI" in line:
            name = line.title()
            break
    if not name:
        email = next((line for line in text.split() if "@" in line), "")
        name = email.split("@")[0].replace(".", " ").title() if email else "Unknown"

    email = next((line for line in text.split() if "@" in line), "")
    phone = re.findall(r"(01[0-9]-?\d{7,8})", text)
    phone = phone[0].replace("-", "") if phone else "Not found"

    job_titles = extract_job_titles_dynamic(text)
    skills = extract_skills_from_text(text)

    soft_skills = []
    for chunk in nlp(text).noun_chunks:
        if any(kw in chunk.text.lower() for kw in ["skill", "ability", "trait"]):
            soft_skills.append(chunk.text.strip())
    soft_skills = ", ".join(set(soft_skills))

    languages = [ent.text.strip() for ent in nlp(text).ents if ent.label_ == "LANGUAGE"]
    languages = ", ".join(set(languages))

    education_match = re.findall(
        r"(Bachelor|Master|PhD|Diploma|Certificate|SPM|STPM|Matrikulasi)\s+(?:of|in)?\s*([A-Za-z &\-]+)?",
        text, re.IGNORECASE)
    education = [f"{deg.title()} in {fld.strip().title()}" if fld else deg.title() for deg, fld in education_match]

    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": "Malaysia",
        "jobTitle": job_titles[0] if job_titles else "",
        "skills": skills,
        "softSkills": soft_skills,
        "languages": languages,
        "workExperience": "\n".join(job_titles) if job_titles else "No job titles found",
        "education": ", ".join(set(education)),
        "job_level": predict_job_level(text)
    }

def extract_job_titles_dynamic(text):
    doc = nlp(text)
    job_titles = set()
    for sent in doc.sents:
        if any(keyword in sent.text.lower() for keyword in ["worked as", "role as", "position", "job title", "experience as"]):
            for chunk in sent.noun_chunks:
                if 1 <= len(chunk.text.split()) <= 4:
                    job_titles.add(chunk.text.title())
    return list(job_titles)

def extract_skills_from_text(text):
    doc = nlp(text)
    skill_keywords = ["skills", "proficient in", "expert in", "knowledge of"]
    candidate_skills = set()
    for line in text.splitlines():
        if any(kw in line.lower() for kw in skill_keywords):
            parts = line.split(":")
            possible_skills = parts[1] if len(parts) > 1 else line
            candidate_skills.update(re.split(r"[,;&]", possible_skills))
    candidate_skills.update(chunk.text.strip() for chunk in doc.noun_chunks)
    return sorted({s.strip() for s in candidate_skills if is_valid_skill(s)})

def is_valid_skill(skill):
    return skill and 2 <= len(skill) <= 40 and len(skill.split()) <= 4 and not any(char.isdigit() for char in skill)

def extract_years_of_experience(text):
    # Look for patterns like "3 years of experience", "5+ yrs", etc.
    matches = re.findall(r'(\d+)\s*(?:\+)?\s*(years?|yrs?)\s+(?:of\s+)?experience', text, re.I)
    years = [int(m[0]) for m in matches if m[0].isdigit()]
    return max(years) if years else 0


def predict_job_level(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return {0: "Entry", 1: "Mid", 2: "Senior"}.get(predicted_class, "Entry")

# ---------- API Routes ---------- #
@app.route('/extract', methods=['POST'])
def extract():
    if 'resume' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    file = request.files['resume']
    return jsonify(extract_resume_info(file))

@app.route('/recommend', methods=['POST'])
def recommend():
    global matched_jobs
    try:
        data = request.get_json() if request.is_json else request.form

        skills = data.get("skills", "")
        soft_skills = data.get("softSkills", "")
        education = data.get("education", "")
        job_level = data.get("job_level", "")
        query = " ".join([skills, soft_skills, education, job_level]).strip()

        if not query:
            return jsonify({"error": "Empty query"}), 400

        # === Load TF-IDF vectorizer and correct matrix ===
        tfidf_vectorizer = joblib.load("text/tfidf_vectorizer.joblib")
        job_tfidf_matrix = load_npz("text/career_features.npz")
        user_tfidf_vector = tfidf_vectorizer.transform([query])
        tfidf_sim = cosine_similarity(user_tfidf_vector, job_tfidf_matrix)[0]

        # === BERT embedding similarity ===
        user_embedding = bert_embedder.encode([query], convert_to_numpy=True)
        user_embedding = scaler.transform(user_embedding)
        bert_sim = cosine_similarity(user_embedding, career_embeddings)[0]

        # === Combine similarities ===
        hybrid_sim = 0.75 * bert_sim + 0.25 * tfidf_sim
        sorted_indices = np.argsort(hybrid_sim)[::-1][:30]

        title_col = "title" if "title" in career_df.columns else "job_title"
        desc_col = "descriptions" if "descriptions" in career_df.columns else "description"

        matched_jobs = []
        for idx in sorted_indices:
            matched_jobs.append({
                "job_title": str(career_df.iloc[idx].get(title_col, "Unknown Title")),
                "description": str(career_df.iloc[idx].get(desc_col, "No description provided.")),
                "match_percent": int(round(hybrid_sim[idx] * 100)),
                "company": str(career_df.iloc[idx].get("company", "Not specified")),
                "location": str(career_df.iloc[idx].get("location", "Not specified")),
                "skills": str(career_df.iloc[idx].get("skills", ""))
            })

        return jsonify(matched_jobs)

    except Exception as e:
        print("[ERROR] Recommendation failed:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Recommendation failed"}), 500


@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def css():
    return send_from_directory('.', 'styles.css')

if __name__ == '__main__':
    app.run(debug=True)
