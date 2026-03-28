"""
Resume Screening AI — Core NLP Engine
======================================
Techniques : spaCy lemmatisation · TF-IDF vectorisation · Cosine similarity
Author     : NeuralHire

Install:
    pip install spacy scikit-learn pandas numpy flask flask-cors
    python -m spacy download en_core_web_sm

Run CLI:
    python resume_screener.py

Run API:
    python api.py   →  http://127.0.0.1:5000
"""

import re, json
import numpy  as np
import pandas as pd
from pathlib    import Path
from dataclasses import dataclass, field
from typing      import Optional

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise        import cosine_similarity

# ── spaCy ──────────────────────────────────────────────────────────────────────
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise SystemExit(
        "\n[ERROR] spaCy model missing.\n"
        "Fix   : python -m spacy download en_core_web_sm\n"
    )


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class Resume:
    name:             str
    raw_text:         str
    email:            str        = ""
    phone:            str        = ""
    skills:           list[str]  = field(default_factory=list)
    experience_years: float      = 0.0
    cleaned_text:     str        = ""
    score:            float      = 0.0
    rank:             int        = 0
    shortlisted:      bool       = False


# ── NLP helpers ────────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    """Lower-case → strip noise → lemmatise → remove stop-words."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+",       " ", text).strip()
    doc  = nlp(text)
    return " ".join(
        t.lemma_ for t in doc
        if not t.is_stop and not t.is_punct and len(t.text) > 2
    )


_EMAIL = re.compile(r"[\w.\-]+@[\w.\-]+\.\w+")
_PHONE = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_YEARS = re.compile(r"(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience", re.I)

SKILLS_DB = [
    # languages
    "python","java","javascript","typescript","c++","c#","ruby","go","rust",
    "swift","kotlin","scala","r","matlab","julia",
    # web / mobile
    "react","angular","vue","nodejs","django","flask","fastapi","spring",
    "flutter","android","ios",
    # ML / AI / DL
    "machine learning","deep learning","nlp","computer vision",
    "tensorflow","pytorch","keras","jax","scikit-learn","xgboost","lightgbm",
    "pandas","numpy","scipy","matplotlib","seaborn",
    # NLP / LLM
    "spacy","nltk","hugging face","transformers","bert","gpt","llm","rag","langchain",
    # CV
    "opencv","yolo","cnn","image processing",
    # MLOps / Cloud
    "mlops","mlflow","dvc","airflow","docker","kubernetes","terraform","jenkins",
    "aws","azure","gcp","sagemaker","spark","hadoop","databricks",
    # Databases
    "sql","mysql","postgresql","mongodb","redis","elasticsearch","cassandra","dynamodb",
    # Practices
    "git","ci/cd","agile","scrum","linux","communication","teamwork",
    "leadership","project management","research","kaggle",
]
_SKILL_RES = [(re.compile(r"\b" + re.escape(s) + r"\b", re.I), s) for s in SKILLS_DB]


def extract_email(t):  m = _EMAIL.search(t); return m.group() if m else ""
def extract_phone(t):  m = _PHONE.search(t); return m.group().strip() if m else ""
def extract_years(t):  return max((float(y) for y in _YEARS.findall(t)), default=0.0)
def extract_skills(t): return list({s for p, s in _SKILL_RES if p.search(t)})


# ── Core screener ──────────────────────────────────────────────────────────────
class ResumeScreener:
    """
    Screen resumes against a job description.

    Parameters
    ----------
    threshold : float   min cosine similarity to shortlist (default 0.15)
    top_n     : int     if > 0, shortlist best N regardless of threshold
    """

    def __init__(self, threshold: float = 0.15, top_n: int = 0):
        self.threshold   = threshold
        self.top_n       = top_n
        self.vectorizer  = TfidfVectorizer(
            ngram_range=(1, 2), sublinear_tf=True,
            min_df=1, max_features=10_000,
        )
        self.resumes: list[Resume] = []
        self.job_description       = ""

    def load_jd(self, text: str):
        self.job_description = text

    def add_resume(self, name: str, text: str):
        r = Resume(name=name, raw_text=text)
        r.email            = extract_email(text)
        r.phone            = extract_phone(text)
        r.experience_years = extract_years(text)
        r.skills           = extract_skills(text)
        r.cleaned_text     = clean_text(text)
        self.resumes.append(r)

    def screen(self) -> list[Resume]:
        corpus  = [clean_text(self.job_description)] + [r.cleaned_text for r in self.resumes]
        matrix  = self.vectorizer.fit_transform(corpus)
        scores  = cosine_similarity(matrix[0], matrix[1:]).flatten()

        for r, s in zip(self.resumes, scores):
            r.score = round(float(s), 4)

        self.resumes.sort(key=lambda r: r.score, reverse=True)
        for i, r in enumerate(self.resumes, 1):
            r.rank = i

        if self.top_n > 0:
            for r in self.resumes[:self.top_n]: r.shortlisted = True
        else:
            for r in self.resumes: r.shortlisted = r.score >= self.threshold

        return self.resumes

    def report(self) -> pd.DataFrame:
        return pd.DataFrame([{
            "Rank":             r.rank,
            "Name":             r.name,
            "Score":            r.score,
            "Shortlisted":      "✅ Yes" if r.shortlisted else "❌ No",
            "Experience (yrs)": r.experience_years,
            "Skills Found":     ", ".join(sorted(r.skills)),
            "Email":            r.email,
        } for r in self.resumes])

    def to_json(self, path: Optional[str] = None) -> str:
        data = [{
            "rank": r.rank, "name": r.name, "score": r.score,
            "shortlisted": r.shortlisted, "experience_years": r.experience_years,
            "skills": sorted(r.skills), "email": r.email, "phone": r.phone,
        } for r in self.resumes]
        out = json.dumps(data, indent=2)
        if path: Path(path).write_text(out)
        return out


# ── Sample data ────────────────────────────────────────────────────────────────
JD = """
Senior Machine Learning Engineer — 4+ years required.

Requirements:
- Strong Python (pandas, numpy, scikit-learn)
- Deep Learning: TensorFlow or PyTorch
- NLP: spaCy, NLTK, Hugging Face Transformers, LLM, RAG
- Computer Vision: OpenCV, YOLO, CNNs
- MLOps: Docker, Kubernetes, CI/CD, MLflow
- Cloud: AWS SageMaker, GCP Vertex AI, or Azure ML
- SQL and NoSQL databases
- Git, Agile/Scrum
- Bonus: published research, Kaggle, open-source contributions
"""

RESUMES = [
    {"name": "Alice Chen",
     "text": """Alice Chen | alice@email.com | +1-555-0101
Senior ML Engineer · 6 years experience
Python TensorFlow PyTorch spaCy NLP Hugging Face LLM RAG langchain
scikit-learn pandas numpy AWS SageMaker Docker Kubernetes CI/CD MLflow
PostgreSQL MongoDB Git Agile computer vision OpenCV YOLO
Led NLP RAG pipeline with Hugging Face + LangChain. Deployed on AWS SageMaker.
YOLO-based computer vision system. Kaggle master. 2 NLP papers published."""},

    {"name": "Bob Martinez",
     "text": """Bob Martinez | bob@mail.com
Front-End Developer · 4 years
JavaScript TypeScript React Vue HTML CSS Node.js Git REST APIs
Built SPAs. No ML or data science experience."""},

    {"name": "Carol Lee",
     "text": """Carol Lee | carol@example.com
AI Research Scientist · 5 years NLP and Computer Vision
Python PyTorch TensorFlow NLP BERT GPT transformers Hugging Face
computer vision CNN OpenCV YOLO pandas numpy scikit-learn
Spark GCP Vertex AI Docker Kubernetes SQL Agile research Kaggle
PhD CS MIT. Top-1% Kaggle. 4 published NLP papers on transformers."""},

    {"name": "David Kim",
     "text": """David Kim | david@email.com
Data Analyst · 2 years
Python pandas numpy SQL Excel Tableau Power BI statistics
Dashboards in Tableau. Basic SQL reporting. No ML experience."""},

    {"name": "Eva Patel",
     "text": """Eva Patel | eva@techmail.com | +1-555-0404
ML Engineer · 4 years experience
Python scikit-learn TensorFlow spaCy NLP pandas numpy
AWS Docker MySQL Agile CI/CD Git MLflow xgboost lightgbm
NLP classification with spaCy + TF-IDF on AWS Docker. MLflow tracking."""},
]


# ── CLI entry point ────────────────────────────────────────────────────────────
def main():
    print("=" * 62)
    print("  Resume Screening AI — NLP + TF-IDF + Cosine Similarity")
    print("=" * 62)

    s = ResumeScreener(threshold=0.15)
    s.load_jd(JD)
    for r in RESUMES:
        s.add_resume(r["name"], r["text"])
        print(f"  ✔  Loaded: {r['name']}")

    print(f"\n  Screening {len(RESUMES)} resumes…\n")
    s.screen()
    print(s.report().to_string(index=False))

    s.to_json("screening_results.json")
    s.report().to_csv("screening_results.csv", index=False)
    print("\n  Saved → screening_results.json / .csv")

    hit = [r for r in s.resumes if r.shortlisted]
    print(f"\n  Shortlisted {len(hit)}/{len(s.resumes)}:")
    for r in hit:
        print(f"    #{r.rank}  {r.name}  ({r.score:.4f})")
    print("=" * 62)


if __name__ == "__main__":
    main()
