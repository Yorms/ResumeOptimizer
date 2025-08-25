# ----------------------------
# Resume Optimization Agent - Full Version
# ----------------------------
import re
import json
import requests
from bs4 import BeautifulSoup
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for PDF parsing
import docx
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# 1. Resume Parsing
# ----------------------------
def parse_pdf_resume(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        text += page.get_text()
    return text

def parse_docx_resume(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([p.text for p in doc.paragraphs])
    return text

def parse_resume_text(text):
    skills_match = re.findall(r"Skills:\s*(.*)", text, re.IGNORECASE)
    skills = skills_match[0].split(", ") if skills_match else []

    experience_matches = re.findall(r"- (.*) at (.*) \((\d+) years\): (.*)", text)
    experience = [{"title": m[0], "company": m[1], "years": int(m[2]), "projects":[m[3]]} 
                  for m in experience_matches]

    education_match = re.findall(r"Education:\s*(.*)", text)
    education = education_match[0] if education_match else ""

    cert_match = re.findall(r"Certifications:\s*(.*)", text)
    certifications = cert_match[0].split(", ") if cert_match else []

    return {"skills": skills, "experience": experience, "education": education, "certifications": certifications}

# ----------------------------
# 2. Semantic Job Parsing
# ----------------------------
def parse_job_semantic(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator="\n")
    sentences = sent_tokenize(text)

    required_keywords = ["must have", "required", "experience with", "proficient in"]
    desired_keywords = ["preferred", "nice to have", "bonus", "familiar with"]

    required_sents = [s for s in sentences if any(k in s.lower() for k in required_keywords)]
    desired_sents = [s for s in sentences if any(k in s.lower() for k in desired_keywords)]

    def extract_skills(sent_list):
        skills = []
        for s in sent_list:
            found = re.findall(r'\b[A-Z][A-Za-z0-9+#]*\b', s)
            skills.extend(found)
        return list(set(skills))

    required_skills = extract_skills(required_sents)
    desired_skills = extract_skills(desired_sents)

    degree_patterns = ["Bachelor", "Master", "PhD", "B.Sc", "M.Sc"]
    qualifications = []
    for s in sentences:
        for d in degree_patterns:
            if d.lower() in s.lower():
                qualifications.append(s)
    qualifications = list(set(qualifications))

    job_title = soup.find('h1').get_text(strip=True) if soup.find('h1') else sentences[0]

    return {"job_title": job_title, "required_skills": required_skills, "desired_skills": desired_skills,
            "qualifications": qualifications}

# ----------------------------
# 3. Company Research
# ----------------------------
def get_company_info(company_name):
    search_url = f"https://en.wikipedia.org/wiki/{company_name.replace(' ','_')}"
    response = requests.get(search_url)
    if response.status_code != 200:
        return {"name": company_name, "industry":"N/A", "size":"N/A", "culture":"N/A", "recent_news":[]}

    soup = BeautifulSoup(response.text, 'html.parser')
    description = soup.find('p').get_text() if soup.find('p') else ""
    size_match = re.findall(r'\b(\d{2,5}) employees\b', description)
    industry_match = re.findall(r'\b(software|technology|finance|healthcare|SaaS|telecom)\b', description, re.I)

    return {"name": company_name,
            "industry": industry_match[0] if industry_match else "N/A",
            "size": size_match[0] if size_match else "N/A",
            "culture": "N/A",
            "recent_news": []}

# ----------------------------
# 4. Semantic Skill Matching
# ----------------------------
def semantic_skill_match(resume_skills, job_skills, threshold=0.7):
    matches = []
    if not resume_skills or not job_skills:
        return matches
    resume_emb = model.encode(resume_skills, convert_to_tensor=True)
    job_emb = model.encode(job_skills, convert_to_tensor=True)
    for i, r in enumerate(resume_emb):
        for j, j_emb in enumerate(job_emb):
            sim = util.pytorch_cos_sim(r, j_emb).item()
            if sim >= threshold:
                matches.append(job_skills[j])
    return list(set(matches))

# ----------------------------
# 5. Resume Scoring
# ----------------------------
def calculate_scores(resume, job):
    matched_skills = semantic_skill_match(resume["skills"], job["required_skills"] + job["desired_skills"])
    skill_score = len(matched_skills) / max(len(job["required_skills"] + job["desired_skills"]), 1)

    total_required_years = 2
    relevant_years = sum([exp["years"] for exp in resume["experience"]])
    experience_score = min(relevant_years / total_required_years, 1.0)

    matched_qual = len([q for q in job["qualifications"] if q in resume["education"] or q in resume["certifications"]])
    qualification_score = matched_qual / max(len(job["qualifications"]), 1)

    overall_score = 0.5 * skill_score + 0.3 * experience_score + 0.2 * qualification_score

    return {"skill_score": round(skill_score,2), "experience_score": round(experience_score,2),
            "qualification_score": round(qualification_score,2), "overall_score": round(overall_score,2)}

# ----------------------------
# 6. Optimization Suggestions
# ----------------------------
def optimization_suggestions(resume, job):
    all_job_skills = job["required_skills"] + job["desired_skills"]
    missing_skills = list(set(all_job_skills) - set(resume["skills"]))
    experience_tips = ["Highlight relevant projects with metrics"]
    qualification_tips = ["Include missing certifications if applicable"]
    ats_keywords = all_job_skills
    return {"missing_skills": missing_skills, "experience_tips": experience_tips,
            "qualification_tips": qualification_tips, "ATS_keywords": ats_keywords}

# ----------------------------
# 7. Streamlit Interface
# ----------------------------
st.title("Resume Optimization Agent - Full Version")

uploaded_file = st.file_uploader("Upload your resume (PDF or DOCX)", type=['pdf','docx'])
job_url = st.text_input("Enter Job Posting URL")
company_name = st.text_input("Enter Company Name")

if uploaded_file and job_url and company_name:
    # Resume
    if uploaded_file.type == "application/pdf":
        resume_text = parse_pdf_resume(uploaded_file)
    else:
        resume_text = parse_docx_resume(uploaded_file)
    resume_data = parse_resume_text(resume_text)

    # Job parsing
    job_data = parse_job_semantic(job_url)

    # Company info
    company_info = get_company_info(company_name)

    # Scores & Optimization
    scores = calculate_scores(resume_data, job_data)
    optimizations = optimization_suggestions(resume_data, job_data)

    # Final report
    report = {"resume": resume_data, "job": job_data, "company": company_info,
              "scores": scores, "optimization": optimizations}

    st.header("Overall Match Score")
    st.metric("Overall Score", report["scores"]["overall_score"])

    st.header("Score Breakdown")
    st.json(report["scores"])

    st.header("Optimization Suggestions")
    st.json(report["optimization"])

    st.header("Company Info")
    st.json(report["company"])

    # Save JSON report
    with open("resume_report_full.json","w") as f:
        json.dump(report, f, indent=4)
