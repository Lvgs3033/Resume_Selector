# 🧠 NeuralHire — AI/ML Resume Screener

> **Automatically shortlist the best candidates for AI/ML roles using NLP + TF-IDF + Cosine Similarity**

---

## 📌 What It Does

NeuralHire reads every uploaded resume, compares it against your job description using mathematical text-similarity, assigns a match score (0–100%) to each candidate, ranks them from best to worst, and marks top performers as **Shortlisted** — all in seconds.

```
Input  →  Job Description + Resumes (PDF / TXT / paste)
Output →  Ranked candidates + Match scores + Skills matrix + Radar chart
```

---

## 🗂 Project Files

| File | Description |
|------|-------------|
| `index.html` | Full frontend — drag-drop PDF upload, results dashboard, radar chart |
| `resume_screener.py` | Core NLP engine: spaCy + TF-IDF + Cosine Similarity + CLI |
| `api.py` | Flask REST API wrapping the engine |
| `requirements.txt` | Python dependencies |
| `NOTES.txt` | Full technical explanation of every component |
| `README.md` | This file |

---

## ⚡ Quick Start

### Option A — Browser Only (Zero Setup)

```bash
1. Download  index.html
2. Open it in Chrome / Firefox / Edge
3. Drag & drop your PDF resumes onto the drop zone
4. Click  RUN ANALYSIS
```

> No Python, no server, no installation. Everything runs in your browser.  
> Your resumes **never leave your device**.

---

### Option B — Python CLI

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the spaCy English model
python -m spacy download en_core_web_sm

# 3. Run the screener with sample data
python resume_screener.py
```

**Output:**
```
================================================================
  Resume Screening AI — NLP + TF-IDF + Cosine Similarity
================================================================
  ✔  Loaded: Alice Chen
  ✔  Loaded: Bob Martinez
  ✔  Loaded: Carol Lee
  ...

  Rank  Name          Score   Shortlisted   Skills Found
     1  Alice Chen    0.4821  ✅ Yes         python, tensorflow, nlp...
     2  Carol Lee     0.3956  ✅ Yes         pytorch, bert, gpt...
     3  Eva Patel     0.2834  ✅ Yes         scikit-learn, spacy...
     4  David Kim     0.0889  ❌ No          pandas, sql...
     5  Bob Martinez  0.0312  ❌ No          javascript, react...

  Saved → screening_results.json / .csv
```

---

### Option C — Flask REST API

```bash
# Start the API server
python api.py

# Server runs at:  http://127.0.0.1:5000
```

Then open in your browser:

| URL | Description |
|-----|-------------|
| `http://127.0.0.1:5000/` | Welcome page + endpoint list |
| `http://127.0.0.1:5000/health` | `{"status": "ok"}` |
| `http://127.0.0.1:5000/demo` | Run built-in sample data |

**POST /screen — Screen your own resumes:**

```bash
curl -X POST http://127.0.0.1:5000/screen \
  -H "Content-Type: application/json" \
  -d '{
    "job_description": "Senior ML Engineer with Python, TensorFlow, NLP...",
    "resumes": [
      {"name": "Alice", "text": "Python TensorFlow spaCy NLP AWS Docker..."},
      {"name": "Bob",   "text": "JavaScript React HTML CSS Node.js..."}
    ],
    "threshold": 0.15
  }'
```

---

## 🔬 How It Works

### 1. Text Cleaning (spaCy)
Every resume and the job description go through:
- **Lowercasing** → `Python` → `python`
- **Noise removal** → strip punctuation, numbers, special chars
- **Lemmatisation** → `building` → `build`, `engineers` → `engineer`
- **Stop-word removal** → delete `the`, `and`, `is`, `of`, etc.

### 2. TF-IDF Vectorisation (scikit-learn)
Converts cleaned text into numerical vectors. Each word gets a weight based on:
- **TF** (Term Frequency) — how often it appears in this document
- **IDF** (Inverse Document Frequency) — how rare it is across all documents

Rare, specific words (`langchain`, `sagemaker`) get higher weights than common words (`python`, `experience`).

```
ngram_range = (1, 2)   → captures "machine learning" as one feature
sublinear_tf = True    → log-normalises TF to prevent word flooding
max_features = 10,000  → top 10,000 most meaningful features
```

### 3. Cosine Similarity Scoring (scikit-learn)
Measures the angle between the Job Description vector and each Resume vector:

```
Score = 1.0  →  Perfect match  (identical topic distribution)
Score = 0.0  →  No match at all  (completely different topics)
```

Typical real-world scores: `0.05` (unrelated) to `0.60` (excellent match).

### 4. Shortlisting Decision
**Threshold Mode** (default): `score >= 0.15` → Shortlisted  
**Top-N Mode**: Always pick the best N candidates regardless of score

---

## 🎯 Score Interpretation

| Score | Label | Meaning |
|-------|-------|---------|
| 0.00 – 0.05 | Very Poor | Completely different field |
| 0.05 – 0.15 | Poor–Weak | Some overlap, missing key skills |
| 0.15 – 0.30 | Moderate | Partial match |
| 0.30 – 0.50 | Good–Strong | Most key skills present |
| 0.50 – 1.00 | Excellent | Deep vocabulary alignment |

---

## 📊 Frontend Features

### Upload Mode
- 🗂 **Drag & drop** multiple PDF / TXT / DOCX files at once
- 📄 **PDF.js** extracts text from every page (runs in browser)
- ✅ **Live skill detection** — tags appear as files are parsed
- ✏️ **Editable candidate names** (defaults to filename)
- 📊 **Progress bar** per file

### Results Dashboard
| Section | Description |
|---------|-------------|
| **Winner Banner** | Best candidate with large score display |
| **Ranked Cards** | All candidates, animated score bars, gold/silver/bronze |
| **Radar Chart** | 8-axis skill comparison for top 4 candidates |
| **Skills Matrix** | ● / ○ grid for 40+ skills across all candidates |
| **Summary Panel** | Shortlist count, avg score, winner margin |

### 8 Radar Axes
`ML/DL` · `NLP/LLM` · `Vision` · `MLOps` · `Cloud` · `Python` · `Databases` · `Practices`

---

## 📦 Dependencies

```txt
spacy>=3.7.0
scikit-learn>=1.4.0
pandas>=2.0.0
numpy>=1.26.0
flask>=3.0.0
flask-cors>=4.0.0
```

**Frontend only** (loaded from CDN, no install):
- PDF.js v3.11.174 — PDF text extraction
- Bebas Neue + DM Mono — Google Fonts

---

## 🗂 Supported File Types

| Format | Parsing Method |
|--------|---------------|
| `.pdf` | PDF.js — full text extraction from all pages |
| `.txt` | Direct text read |
| `.md` | Direct text read |
| `.rtf` | Read as text |
| `.doc` / `.docx` | Text layer extraction (convert binary .doc first) |

---

## 🛠 Skills Tracked (60+)

**Languages:** Python, Java, JavaScript, Scala, R, MATLAB, Julia  
**ML/DL:** TensorFlow, PyTorch, Keras, JAX, XGBoost, LightGBM  
**NLP/LLM:** spaCy, NLTK, Transformers, BERT, GPT, LLM, RAG, LangChain  
**Vision:** OpenCV, YOLO, CNN, Computer Vision  
**ML Libraries:** scikit-learn, Pandas, NumPy, SciPy  
**MLOps:** Docker, Kubernetes, MLflow, DVC, Airflow, CI/CD  
**Cloud:** AWS, GCP, Azure, SageMaker, Spark, Databricks  
**Databases:** SQL, PostgreSQL, MongoDB, Redis, Elasticsearch  
**Practices:** Git, Agile, Scrum, Kaggle, Research  

---

## 📁 Output Files (CLI mode)

| File | Format | Contents |
|------|--------|---------|
| `screening_results.json` | JSON | Full results with all fields |
| `screening_results.csv` | CSV | Spreadsheet-ready ranked table |

---

## 🔐 Privacy

- **Browser frontend:** 100% local. Files never leave your device.
- **Python CLI:** 100% local. No internet connection needed.
- **API:** Runs on localhost by default (`127.0.0.1:5000`).

---

## 📖 Documentation

For complete technical documentation — including detailed explanations of
TF-IDF, cosine similarity, how shortlisting works, how rejection works,
formula derivations, and worked examples — see **`NOTES.txt`**.

---

## 🧪 Technology Stack

```
Backend  : Python 3.10+  ·  spaCy  ·  scikit-learn  ·  Flask
Frontend : HTML5  ·  CSS3  ·  Vanilla JavaScript  ·  PDF.js
NLP      : Lemmatisation  ·  TF-IDF  ·  Cosine Similarity
```

---

## 💡 Usage Tips

1. **Adjust threshold** — lower to 0.10 for specialized roles, raise to 0.25 for broad roles
2. **Use Top-N mode** when you have exactly N interview slots
3. **Edit candidate names** after upload if filenames are not meaningful
4. **Check the Skills Matrix** to see why a candidate scored high or low
5. **Radar chart** shows where top candidates excel and where they have gaps
6. **The score is relative** — always look at the ranking, not absolute values

---

*NeuralHire · Resume Screening AI · Built with spaCy + TF-IDF + Cosine Similarity*
