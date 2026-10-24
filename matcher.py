"""
matcher.py
----------
Handles TF-IDF vectorization, cosine similarity calculation,
keyword / skill extraction, categorical skill breakdown, and ATS resume health auditing.
"""

import math
import re
import string
from typing import Dict, List, Optional, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Stopwords (lightweight, no NLTK download required)
# ---------------------------------------------------------------------------
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "dare",
    "ought", "used", "it", "its", "this", "that", "these", "those", "i",
    "me", "my", "we", "our", "you", "your", "he", "his", "she", "her",
    "they", "their", "what", "which", "who", "whom", "how", "when", "where",
    "why", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "same", "so", "than", "too",
    "very", "just", "also", "then", "than", "well", "as", "up", "out",
    "about", "if", "into", "through", "during", "including", "until",
    "against", "among", "throughout", "despite", "towards", "upon", "whether",
    "while", "since", "without", "within", "along", "following", "across",
    "behind", "beyond", "plus", "except", "around", "however", "we", "us",
    "any", "between", "after", "before", "under", "over", "per", "nor",
    "yet", "s", "t", "re", "ve", "ll", "d", "m", "o", "don", "doesn",
    "didn", "won", "shouldn", "couldn", "wouldn", "isn", "aren", "wasn",
    "weren", "hasn", "haven", "hadn", "mustn", "let", "get", "got", "go",
    "make", "made", "way", "work", "working", "worked", "use", "using",
    "must", "like", "job", "role", "position", "candidate", "applicant",
    "company", "team", "strong", "good", "excellent", "ability", "skills",
    "skill", "experience", "knowledge", "understanding", "including", "related",
    "relevant", "responsible", "responsibilities", "requirements", "required",
    "preferred", "plus", "degree", "years", "year", "equivalent", "e.g",
    "i.e", "etc", "new", "high", "large", "small", "based", "proven",
    "solid", "deep", "broad", "key", "core", "primary", "main",
}

# Known tech / skill multi-word phrases to detect as a single token
TECH_BIGRAMS = {
    "machine learning", "deep learning", "natural language processing",
    "natural language", "computer vision", "data science", "data engineering",
    "data analysis", "data analytics", "software engineering",
    "software development", "web development", "full stack", "back end",
    "front end", "backend development", "frontend development",
    "cloud computing", "continuous integration", "continuous deployment",
    "ci cd", "ci/cd", "version control", "object oriented", "object-oriented",
    "test driven", "restful api", "restful apis", "rest api", "rest apis",
    "api development", "agile methodology", "agile development", "scrum master",
    "product management", "project management", "system design",
    "distributed systems", "microservices architecture", "large language models",
    "llm", "llms", "generative ai", "reinforcement learning",
    "transfer learning", "neural network", "neural networks",
    "convolutional neural", "transformer model", "transformer models",
    "time series", "feature engineering", "model deployment", "mlops",
    "data pipeline", "etl pipeline", "stream processing", "batch processing",
    "real time", "real-time", "sql server", "google cloud", "amazon web services",
    "azure machine learning", "big data", "data warehouse", "data lake",
    "statistical analysis", "a/b testing", "ab testing", "unit testing",
    "integration testing", "code review", "code reviews", "user experience",
    "user interface", "web vitals", "core web vitals", "responsive design",
}

# Meaningful single-word tech/domain terms that should NOT be filtered as stopwords
IMPORTANT_TERMS = {
    "python", "java", "javascript", "typescript", "kotlin", "scala",
    "golang", "go", "rust", "c++", "c#", "ruby", "php", "swift",
    "r", "matlab", "bash", "shell", "powershell", "html", "html5", "css", "css3",
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn", "pandas",
    "numpy", "scipy", "matplotlib", "seaborn", "plotly", "streamlit",
    "huggingface", "transformers", "spacy", "nltk", "gensim",
    "langchain", "openai", "gemini", "anthropic", "llama", "bert",
    "gpt", "t5", "fastapi", "flask", "django", "spring", "node", "nodejs", "react",
    "angular", "vue", "nextjs", "docker", "kubernetes", "terraform",
    "ansible", "jenkins", "github", "gitlab", "bitbucket", "jira",
    "confluence", "airflow", "spark", "hadoop", "kafka", "flink",
    "elasticsearch", "redis", "postgresql", "mysql", "mongodb", "cassandra",
    "dynamodb", "snowflake", "databricks", "dbt", "looker", "tableau",
    "powerbi", "grafana", "prometheus", "aws", "gcp", "azure", "lambda",
    "ec2", "s3", "sagemaker", "bigquery", "pubsub", "dataflow",
    "classification", "regression", "clustering", "nlp", "cv",
    "recommendation", "forecasting", "optimization", "statistics",
    "probability", "linear", "algebra", "calculus", "algorithms",
    "sql", "nosql", "api", "rest", "graphql", "grpc", "microservices",
    "devops", "mlops", "agile", "scrum", "kanban", "ci", "cd",
    "testing", "debugging", "profiling", "deployment", "monitoring",
    "scalability", "performance", "security", "authentication", "authorization",
    "encryption", "networking", "linux", "unix", "git", "oop",
    "functional", "concurrency", "multithreading", "async", "distributed",
    "cloud", "serverless", "container", "containerization", "virtualization",
    "xgboost", "lightgbm", "catboost", "gradient", "boosting",
    "random", "forest", "svm", "pca", "embeddings", "vectors", "similarity",
    "recsys", "etl", "pipeline", "orchestration", "datalake", "warehouse",
    "dashboard", "visualization", "reporting", "analytics",
    "communication", "leadership", "collaboration", "problem-solving",
    "critical", "thinking", "analytical", "attention", "detail",
    "presentation", "stakeholder", "cross-functional", "mentoring",
    "research", "publications", "papers", "patents", "open-source",
    "figma", "storybook", "jest", "cypress", "tailwind", "redux", "vite", "webpack",
}

# Categorization mapping for visual grouping in results
SKILL_CATEGORIES: Dict[str, Set[str]] = {
    "Languages": {
        "python", "java", "javascript", "typescript", "c++", "c#", "golang", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "bash", "shell",
        "powershell", "sql", "html", "html5", "css", "css3", "matlab"
    },
    "Frameworks & Libraries": {
        "react", "nextjs", "angular", "vue", "node", "nodejs", "express", "fastapi",
        "flask", "django", "spring", "pytorch", "tensorflow", "keras", "scikit-learn",
        "sklearn", "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
        "streamlit", "huggingface", "transformers", "xgboost", "lightgbm", "redux",
        "tailwind", "jest", "cypress", "spacy", "nltk", "langchain"
    },
    "Cloud, Data & DevOps": {
        "aws", "amazon web services", "gcp", "google cloud", "azure", "docker",
        "kubernetes", "terraform", "ci/cd", "ci cd", "continuous integration",
        "continuous deployment", "jenkins", "github", "gitlab", "postgresql",
        "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "dynamodb",
        "snowflake", "databricks", "spark", "hadoop", "kafka", "airflow",
        "linux", "serverless", "microservices", "microservices architecture",
        "ec2", "s3", "bigquery", "data pipeline", "etl pipeline", "mlops", "devops"
    },
    "Architecture, Concepts & Soft Skills": {
        "machine learning", "deep learning", "natural language processing", "nlp",
        "computer vision", "cv", "data science", "data analysis", "system design",
        "distributed systems", "object oriented", "restful api", "rest api",
        "agile", "scrum", "kanban", "a/b testing", "unit testing", "code review",
        "problem-solving", "communication", "leadership", "collaboration",
        "cross-functional", "presentation", "stakeholder", "responsive design",
        "web vitals", "generative ai", "llm", "large language models"
    }
}

# Action verbs commonly looked for by ATS and recruiters
STRONG_ACTION_VERBS = {
    "accelerated", "achieved", "analyzed", "architected", "automated",
    "built", "collaborated", "configured", "constructed", "created",
    "decreased", "delivered", "deployed", "designed", "developed",
    "engineered", "enhanced", "established", "executed", "formulated",
    "generated", "implemented", "improved", "increased", "initiated",
    "integrated", "launched", "led", "managed", "maximized", "mentored",
    "minimized", "modeled", "optimized", "orchestrated", "partnered",
    "pioneered", "reduced", "refactored", "resolved", "restructured",
    "scaled", "spearheaded", "streamlined", "supervised", "trained", "validated",
}


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """
    Lowercase, remove punctuation and extra whitespace.
    Keeps hyphens inside words (e.g. 'full-stack') intact.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split preprocessed text into tokens, filtering pure stopwords."""
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def extract_keywords(text: str, top_n: int = 50) -> List[str]:
    """
    Extract key job-related skills and concepts from text.
    Combines bigram detection, domain term dictionary matching, and TF-IDF scoring.
    """
    processed = preprocess_text(text)
    found_keywords: Set[str] = set()

    # Step 1: Detect known tech bigrams / trigrams
    for phrase in TECH_BIGRAMS:
        if phrase in processed:
            found_keywords.add(phrase)

    # Step 2: Detect curated important terms present in the text
    tokens = tokenize(processed)
    for token in tokens:
        if token in IMPORTANT_TERMS:
            found_keywords.add(token)

    # Step 3: TF-IDF on the whole text for statistically significant terms
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=250,
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform([processed])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        scored_terms = sorted(
            zip(feature_names, scores), key=lambda x: x[1], reverse=True
        )
        for term, score in scored_terms[:top_n]:
            if score > 0.05 and term not in STOPWORDS and len(term) > 2:
                found_keywords.add(term)
    except Exception:
        pass

    return sorted(found_keywords)


def categorize_keywords(keywords: List[str]) -> Dict[str, List[str]]:
    """Group keywords into functional categories for clear UI presentation."""
    result: Dict[str, List[str]] = {cat: [] for cat in SKILL_CATEGORIES}
    result["Other Relevant Skills"] = []

    for kw in keywords:
        kw_clean = kw.lower().strip()
        matched = False
        for cat, terms in SKILL_CATEGORIES.items():
            if kw_clean in terms or any(t in kw_clean for t in terms if len(t) > 3):
                result[cat].append(kw)
                matched = True
                break
        if not matched:
            result["Other Relevant Skills"].append(kw)

    # Filter out empty categories
    return {k: sorted(v) for k, v in result.items() if v}


# ---------------------------------------------------------------------------
# Similarity calculation
# ---------------------------------------------------------------------------

def compute_similarity(resume_text: str, job_text: str) -> float:
    """
    Compute cosine similarity between resume and job description
    using TF-IDF vectorisation. Returns a float in [0.0, 1.0].
    """
    if not resume_text.strip() or not job_text.strip():
        return 0.0

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    try:
        tfidf_matrix = vectorizer.fit_transform(
            [preprocess_text(resume_text), preprocess_text(job_text)]
        )
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return float(round(score, 4))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Keyword comparison
# ---------------------------------------------------------------------------

def compare_keywords(
    resume_text: str, job_text: str
) -> Tuple[List[str], List[str], List[str]]:
    """
    Compare keywords extracted from job description against the resume.
    """
    resume_processed = preprocess_text(resume_text)
    job_keywords = extract_keywords(job_text, top_n=60)

    matched: List[str] = []
    missing: List[str] = []

    for kw in job_keywords:
        # Check substring match with boundary consideration
        pattern = r"\b" + re.escape(kw) + r"\b"
        if re.search(pattern, resume_processed) or kw.lower() in resume_processed:
            matched.append(kw)
        else:
            missing.append(kw)

    # Suggested = missing skills that are high-signal tech or domain terms
    suggested = [
        kw for kw in missing
        if any(term == kw or term in kw for term in IMPORTANT_TERMS) or kw in TECH_BIGRAMS
    ]

    return sorted(matched), sorted(missing), sorted(set(suggested))


# ---------------------------------------------------------------------------
# ATS Resume Health Audit
# ---------------------------------------------------------------------------

def audit_resume(resume_text: str) -> Dict:
    """
    Performs an ATS structure, readability, and impact audit on the resume text.
    Returns quantitative health metrics, detected sections, and actionable feedback.
    """
    if not resume_text.strip():
        return {
            "score": 0,
            "word_count": 0,
            "reading_time_min": 0,
            "contact_info": {"email": False, "phone": False, "links": False},
            "sections": {"Education": False, "Experience": False, "Skills": False, "Projects": False},
            "action_verbs": [],
            "action_verbs_count": 0,
            "metrics": [],
            "metrics_count": 0,
            "checklist": [],
            "tips": ["Paste your resume text to generate an ATS health audit."],
        }

    words = resume_text.split()
    word_count = len(words)
    reading_time = round(word_count / 220, 1)

    # 1. Contact Info Detection
    has_email = bool(re.search(r"[\w\.-]+@[\w\.-]+\.\w+", resume_text))
    has_phone = bool(re.search(r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", resume_text))
    has_links = bool(re.search(r"(linkedin\.com|github\.com|gitlab\.com|portfolio|\bhttps?://)", resume_text, re.IGNORECASE))

    # 2. Section Detection
    lower_text = resume_text.lower()
    sections = {
        "Education": bool(re.search(r"\b(education|university|college|bachelor|master|degree|gpa)\b", lower_text)),
        "Experience": bool(re.search(r"\b(experience|work history|employment|intern|internship)\b", lower_text)),
        "Skills": bool(re.search(r"\b(skills|technical skills|proficiencies|technologies|tools)\b", lower_text)),
        "Projects": bool(re.search(r"\b(projects|capstone|portfolio|open source|hackathon)\b", lower_text)),
    }

    # 3. Measurable Impact Metrics Detection (e.g. 40%, $5k, 15,000+, 2.5x, sub-50ms)
    metrics_patterns = re.findall(
        r"\b\d+(?:\.\d+)?(?:%|\+|x|k|m|ms|s)?\b|\$\d+(?:,\d{3})*(?:\.\d+)?",
        resume_text,
        re.IGNORECASE
    )
    # Filter out pure small numbers like 1, 2, or single years (2023, 2024)
    significant_metrics = []
    for m in metrics_patterns:
        m_str = m.strip()
        if any(c in m_str for c in ["%", "$", "+", "x", "k", "ms", ","]):
            significant_metrics.append(m_str)
        elif m_str.isdigit() and len(m_str) <= 3 and int(m_str) > 10:
            significant_metrics.append(m_str)
    
    unique_metrics = sorted(list(set(significant_metrics)))[:15]

    # 4. Action Verbs Detection
    found_verbs = set()
    tokens_clean = [re.sub(r"[^\w]", "", w.lower()) for w in words]
    for w in tokens_clean:
        if w in STRONG_ACTION_VERBS:
            found_verbs.add(w)

    action_verbs_list = sorted(list(found_verbs))

    # 5. Calculate Health Score (0 - 100)
    score = 0

    # Contact info: 15 points
    if has_email: score += 5
    if has_phone: score += 5
    if has_links: score += 5

    # Core sections: 35 points
    for sec, found in sections.items():
        if found:
            score += 8
    if all(sections.values()):
        score += 3  # Bonus for complete core layout

    # Action verbs richness: 25 points
    verb_points = min(25, len(action_verbs_list) * 5)
    score += verb_points

    # Measurable metrics: 25 points
    metric_points = min(25, len(unique_metrics) * 5)
    score += metric_points

    # Word count calibration
    tips = []
    if word_count < 250:
        tips.append("Resume is quite brief (< 250 words). Aim for 350-700 words to sufficiently showcase skills and accomplishments.")
    elif word_count > 900:
        tips.append("Resume is somewhat lengthy (> 900 words). If you are a student or early career, a focused 1-page resume (~500 words) is preferred.")

    if not has_links:
        tips.append("Add a clickable GitHub or LinkedIn link to increase recruiter response rates.")
    if len(action_verbs_list) < 4:
        tips.append("Incorporate more active power verbs (e.g., 'Architected', 'Optimized', 'Spearheaded') at the start of bullet points.")
    if len(unique_metrics) < 3:
        tips.append("Add quantified metrics (e.g. '% latency reduction', 'users served', 'data volume') to demonstrate concrete business impact.")
    if not sections["Projects"]:
        tips.append("Consider adding a dedicated 'Projects' section to highlight technical capstones and side projects.")

    if not tips:
        tips.append("Strong ATS fundamentals! Your resume format, metrics, and core sections are well-balanced.")

    checklist = [
        {"item": "Contact Email Present", "status": has_email},
        {"item": "Phone Number Present", "status": has_phone},
        {"item": "LinkedIn / GitHub Link", "status": has_links},
        {"item": "Education Section", "status": sections["Education"]},
        {"item": "Work / Internship Experience", "status": sections["Experience"]},
        {"item": "Technical Skills Section", "status": sections["Skills"]},
        {"item": "Projects / Capstones Section", "status": sections["Projects"]},
        {"item": f"Action Verbs Richness ({len(action_verbs_list)} detected)", "status": len(action_verbs_list) >= 4},
        {"item": f"Quantified Impact Metrics ({len(unique_metrics)} detected)", "status": len(unique_metrics) >= 3},
    ]

    return {
        "score": min(100, score),
        "word_count": word_count,
        "reading_time_min": reading_time,
        "contact_info": {"email": has_email, "phone": has_phone, "links": has_links},
        "sections": sections,
        "action_verbs": action_verbs_list,
        "action_verbs_count": len(action_verbs_list),
        "metrics": unique_metrics,
        "metrics_count": len(unique_metrics),
        "checklist": checklist,
        "tips": tips,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(resume_text: str, job_text: str) -> dict:
    """
    Run the full analysis:
      - Raw TF-IDF Cosine Similarity
      - Composite ATS Match Score
      - Matched, missing, suggested keywords
      - Categorized skill breakdown
      - ATS resume health audit
    """
    similarity = compute_similarity(resume_text, job_text)
    matched, missing, suggested = compare_keywords(resume_text, job_text)

    total_keywords = len(matched) + len(missing)
    keyword_coverage = (len(matched) / total_keywords) if total_keywords > 0 else 0.0

    # Composite ATS score: 60% keyword coverage + 40% TF-IDF semantic overlap
    # This reflects real-world ATS behavior better than a pure raw text cosine metric
    if total_keywords > 0 and (resume_text.strip() and job_text.strip()):
        composite_score = round((keyword_coverage * 0.60) + (similarity * 0.40), 4)
    else:
        composite_score = 0.0

    categorized_matched = categorize_keywords(matched)
    categorized_missing = categorize_keywords(missing)
    resume_audit_result = audit_resume(resume_text)

    return {
        "similarity": similarity,
        "composite_score": composite_score,
        "keyword_coverage": round(keyword_coverage, 4),
        "matched": matched,
        "missing": missing,
        "suggested": suggested,
        "categorized_matched": categorized_matched,
        "categorized_missing": categorized_missing,
        "resume_audit": resume_audit_result,
    }
