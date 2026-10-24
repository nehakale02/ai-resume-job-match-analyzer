# 🎯 AI Resume–Job Match Analyzer & ATS Optimizer

An interview-ready, full-stack NLP & Generative AI web application designed to evaluate resume alignment with target job descriptions, compute ATS match scores, perform resume formatting & structural audits, and generate Google XYZ bullet statements using Gemini 1.5 Flash.

Designed with a clean, modern **light pastel aesthetic** and built to showcase software engineering rigor, deterministic NLP algorithms, and resilient LLM system integration.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| **🎨 Light Pastel Theme** | Clean, accessible, modern UI with soft pastel accents and slate typography. |
| **⚡ 1-Click Demo Presets** | Instant pre-loaded profiles (Full-Stack Engineer, ML/Data Science Intern, Frontend Developer) for instant interview demonstration. |
| **📊 Composite ATS Score** | Combines 60% domain keyword coverage with 40% sublinear TF-IDF cosine similarity for balanced real-world scoring. |
| **🛡️ ATS Resume Health Audit** | Evaluates structural section headers, contact info, quantifiable metrics density, and active power verbs. |
| **🏷️ Categorized Skills Matrix** | Auto-segments matched and missing skills into Languages, Frameworks, Cloud/DevOps, and Concepts. |
| **✍️ Google XYZ Bullet Optimizer** | Rewrites weak resume bullet points to naturally incorporate target skills following Google's XYZ formula. |
| **🤖 Gemini 1.5 Flash Guidance** | Provides personalized executive summaries, gap analysis, and tailored portfolio project ideas. |
| **📄 Multi-Format Export** | Exports comprehensive analysis reports in both formatted Markdown (`.md`) and plain text (`.txt`). |
| **🛡️ Resilient Offline Fallback** | 100% of NLP matching, ATS audits, and templates work locally without requiring an API key. |

---

## 🗂️ Project Structure

```
resume-matcher/
├── app.py              ← Modern Streamlit application with pastel design system
├── matcher.py          ← TF-IDF, cosine similarity, keyword categorizer & ATS audit
├── gemini_service.py   ← Gemini 1.5 API client, schema validator & XYZ bullet rewriter
├── sample_data.py      ← Realistic pre-loaded student & early-career presets
├── utils.py            ← PDF and TXT text extraction pipelines
├── test_local.py       ← Comprehensive test suite verifying NLP, audit & API handling
├── requirements.txt    ← Python dependencies
├── .env.example        ← Environment template for GEMINI_API_KEY
├── .gitignore          ← Git exclusion rules
└── README.md           ← Architecture documentation and interview notes
```

---

## 🚀 Quick Start (Windows PowerShell)

### 1. Clone / Navigate to the folder

```powershell
cd c:\Users\neha\Downloads\resume-matcher
```

### 2. Activate virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

*(If script execution is restricted on your system, run: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`)*

### 3. Install dependencies (if not already installed)

```powershell
pip install -r requirements.txt
```

### 4. (Optional) Configure Gemini API Key

Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```
Add your free Google Gemini API key:
```env
GEMINI_API_KEY=AIzaSy...
```
*(Get a free key in 30 seconds at [Google AI Studio](https://aistudio.google.com/app/apikey). If omitted, the app automatically runs in local offline mode.)*

### 5. Launch the Application

```powershell
streamlit run app.py
```
Open **http://localhost:8501** in your browser.

---

## 🧠 System Architecture & Engineering Decisions

```
 Candidate Resume (PDF/TXT)        Target Job Description
             │                               │
             ▼                               ▼
     ┌───────────────┐               ┌───────────────┐
     │ Text Extractor│               │ Text Cleaner  │
     └───────┬───────┘               └───────┬───────┘
             │                               │
             ├───────────────────────────────┤
             ▼                               ▼
  ┌──────────────────────┐        ┌──────────────────────┐
  │  ATS Health Auditor  │        │ TF-IDF Vectorizer    │
  │  - Action Verbs      │        │ - (1,2) N-Grams      │
  │  - Metrics ($/%)     │        │ - Sublinear TF       │
  │  - Core Sections     │        │ - Stopword Filtering │
  └──────────────────────┘        └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │  Cosine Similarity   │
                                  │  + Keyword Coverage  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ Composite ATS Score  │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                                  ┌──────────────────────┐
                                  │ Gemini 1.5 Flash AI  │
                                  │ (Structured JSON     │
                                  │  & XYZ Rewriter)     │
                                  └──────────┬───────────┘
                                             │
                                             ▼
                             Streamlit Pastel Web Interface
```

### 1. Why Composite Scoring instead of pure Cosine Similarity?
Raw TF-IDF cosine similarity between full documents rarely exceeds 35-45% even for great candidates due to resume boilerplate (headers, addresses, graduation dates). Our composite metric factors:
- **60% Keyword & Skill Coverage**: Did the candidate match critical technical competencies?
- **40% TF-IDF Cosine Similarity**: How semantically aligned is the contextual vocabulary?

### 2. Google's XYZ Formula Bullet Rewriter
Recruiters look for the formula:
> *"Accomplished [X] as measured by [Y] by doing [Z]"*
The built-in bullet rewriter takes any weak candidate bullet point and injects missing target skills with quantified metric placeholders.

---

## 🎓 Common Interview Questions & Answers

### Q: How do you extract multi-word technical skills like "Machine Learning" or "CI/CD"?
**A:** We use a two-pronged strategy:
1. A curated regex dictionary of high-signal bigrams/trigrams (`TECH_BIGRAMS`) scanned before tokenization.
2. An N-gram range `(1, 2)` inside `TfidfVectorizer` to capture contextual bigram frequencies automatically.

### Q: Why use `sublinear_tf=True` in scikit-learn?
**A:** Sublinear TF replaces raw term frequency $tf$ with $1 + \log(tf)$. This dampens the impact of terms repeated dozens of times in long job descriptions, preventing keyword stuffing from skewing the cosine similarity.

### Q: How does the system handle API failures or rate limits?
**A:** `gemini_service.py` features exponential retry backoff, multi-stage JSON regex extraction (handling raw JSON, markdown-wrapped JSON, and regex block matching), and complete offline fallbacks. The core NLP matcher and ATS audit never crash even if no internet connection is present.

---

## 🧪 Testing

Run the local test suite:
```powershell
.\venv\Scripts\python test_local.py
```
All unit tests should pass with status exit code 0.

---

## 📄 License

MIT License — Feel free to use and customize for your own job applications and portfolio!
