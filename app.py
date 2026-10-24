"""
app.py
------
AI Resume–Job Match Analyzer
A modern, recruiter-ready resume analysis platform built with Streamlit,
scikit-learn TF-IDF, and Google Gemini AI.

Light pastel design theme tailored for portfolio excellence.
"""

import streamlit as st

from gemini_service import check_api_key_status, get_ai_suggestions, rewrite_bullet_point
from matcher import analyze
from utils import extract_text_from_file

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="AI Resume Matcher & ATS Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Clean, Soft Light Pastel Aesthetics
# ============================================================
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Fira+Code:wght@400;500&display=swap');

/* ---------- Global Reset & Fonts ---------- */
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background-color: #f8fafc !important;
    color: #1e293b !important;
}

/* Background subtle ambient pastel glow */
.stApp {
    background: radial-gradient(circle at 10% 10%, rgba(238, 242, 255, 0.7) 0%, rgba(248, 250, 252, 1) 40%),
                radial-gradient(circle at 90% 90%, rgba(236, 253, 245, 0.6) 0%, rgba(248, 250, 252, 1) 40%) !important;
}

/* Hide default Streamlit chrome & Deploy button */
#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }
[data-testid="stToolbar"] { display: none !important; }
.stDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }
div[data-testid="stStatusWidget"] { display: none !important; }

/* ---------- Sidebar Styling ---------- */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    border-right: 1px solid #e2e8f0 !important;
    box-shadow: 2px 0 10px rgba(0, 0, 0, 0.02);
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1e293b !important;
    font-weight: 700 !important;
}

/* ---------- Hero Banner ---------- */
.hero-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 2.2rem 2.5rem;
    margin-bottom: 1.8rem;
    box-shadow: 0 4px 20px -2px rgba(99, 102, 241, 0.06);
    position: relative;
    overflow: hidden;
}
.hero-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #818cf8 0%, #34d399 50%, #f472b6 100%);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background-color: #eef2ff;
    color: #4f46e5;
    border: 1px solid #c7d2fe;
    padding: 0.25rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.03em;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.03em;
    margin: 0 0 0.5rem 0;
    line-height: 1.25;
}
.hero-subtitle {
    font-size: 1.0rem;
    color: #64748b;
    font-weight: 400;
    max-width: 800px;
    line-height: 1.6;
    margin: 0;
}

/* ---------- Section Cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 1.3rem 1.5rem !important;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03) !important;
    transition: all 0.25s ease !important;
    margin-bottom: 1rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #cbd5e1 !important;
    box-shadow: 0 6px 16px rgba(15, 23, 42, 0.06) !important;
}
.card-header-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #1e293b;
    letter-spacing: -0.01em;
    margin: 0 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ---------- Stat Metric Tiles ---------- */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 1rem;
    margin: 1.2rem 0;
}
.metric-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1.1rem 1rem;
    text-align: center;
    box-shadow: 0 2px 6px rgba(0,0,0,0.02);
}
.metric-val {
    font-size: 1.9rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.25rem;
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ---------- Match Score Banner Card ---------- */
.score-hero-box {
    background: linear-gradient(135deg, #ffffff 0%, #f8faff 100%);
    border: 1px solid #e0e7ff;
    border-radius: 18px;
    padding: 1.8rem;
    text-align: center;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.07);
    margin-bottom: 1.2rem;
}
.score-large-number {
    font-size: 4.2rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.04em;
    margin-bottom: 0.4rem;
}
.score-badge-pill {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    font-size: 0.9rem;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.score-note {
    font-size: 0.8rem;
    color: #64748b;
    line-height: 1.5;
    max-width: 450px;
    margin: 0 auto;
}

/* ---------- Skill Chips (Pastel) ---------- */
.chip-container {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    margin-top: 0.5rem;
}
.chip-item {
    display: inline-flex;
    align-items: center;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'Fira Code', monospace;
    transition: transform 0.15s ease;
}
.chip-item:hover {
    transform: translateY(-1px);
}
.chip-matched {
    background-color: #ecfdf5;
    color: #047857;
    border: 1px solid #a7f3d0;
}
.chip-missing {
    background-color: #fff1f2;
    color: #be123c;
    border: 1px solid #fecdd3;
}
.chip-suggested {
    background-color: #fffbeb;
    color: #b45309;
    border: 1px solid #fde68a;
}
.chip-ai {
    background-color: #f5f3ff;
    color: #6d28d9;
    border: 1px solid #ddd6fe;
}

/* ---------- AI & Coach Cards ---------- */
.ai-summary-card {
    background: #f8faff;
    border: 1px solid #c7d2fe;
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    color: #1e293b;
    font-size: 0.94rem;
    line-height: 1.65;
    margin-bottom: 1.2rem;
}
.suggestion-card {
    background: #ffffff;
    border-left: 4px solid #818cf8;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    color: #334155;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    line-height: 1.55;
}
.project-card {
    background: #ffffff;
    border-left: 4px solid #34d399;
    border-radius: 0 12px 12px 0;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.7rem;
    font-size: 0.9rem;
    color: #334155;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    line-height: 1.55;
}

/* ---------- ATS Checklist Rows ---------- */
.checklist-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.85rem;
}
.checklist-row:last-child {
    border-bottom: none;
}
.status-pill-pass {
    background-color: #ecfdf5;
    color: #065f46;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
}
.status-pill-fail {
    background-color: #fff1f2;
    color: #9f1239;
    padding: 0.15rem 0.6rem;
    border-radius: 9999px;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ---------- Streamlit Widgets Customization ---------- */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.6rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.35) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Light pastel secondary buttons */
div[data-testid="stHorizontalBlock"] .stButton > button {
    border-radius: 10px !important;
}

/* Text area fields */
.stTextArea textarea {
    background-color: #ffffff !important;
    color: #1e293b !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 12px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    transition: all 0.2s ease !important;
}
.stTextArea textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #ffffff !important;
    border: 1px dashed #cbd5e1 !important;
    border-radius: 12px !important;
    padding: 0.8rem !important;
}

/* Streamlit Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background-color: #f1f5f9;
    padding: 6px;
    border-radius: 12px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    color: #64748b;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 16px;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background-color: #ffffff !important;
    color: #4f46e5 !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

/* Progress bar */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1 0%, #34d399 100%) !important;
    border-radius: 9999px;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ============================================================
# UI Helpers: Score Palette & Chips
# ============================================================
def get_score_palette(score: float):
    if score >= 0.65:
        return {
            "color": "#059669",
            "bg": "#ecfdf5",
            "border": "#a7f3d0",
            "label": "Strong Match",
            "badge": "🟢 High Relevance",
        }
    elif score >= 0.40:
        return {
            "color": "#d97706",
            "bg": "#fffbeb",
            "border": "#fde68a",
            "label": "Moderate Match",
            "badge": "🟡 Potential Match",
        }
    else:
        return {
            "color": "#e11d48",
            "bg": "#fff1f2",
            "border": "#fecdd3",
            "label": "Low Match",
            "badge": "🔴 Skill Gaps Present",
        }


def render_pastel_chips(items: list[str], chip_class: str) -> str:
    if not items:
        return "<p style='color:#94a3b8;font-style:italic;font-size:0.85rem;'>None detected.</p>"
    chips = "".join(f'<span class="chip-item {chip_class}">{kw}</span>' for kw in items)
    return f'<div class="chip-container">{chips}</div>'


# ============================================================
# Session State Initialization
# ============================================================
if "resume_content" not in st.session_state:
    st.session_state["resume_content"] = ""
if "job_content" not in st.session_state:
    st.session_state["job_content"] = ""
if "last_analysis" not in st.session_state:
    st.session_state["last_analysis"] = None
if "ai_analysis" not in st.session_state:
    st.session_state["ai_analysis"] = None
if "ai_error" not in st.session_state:
    st.session_state["ai_error"] = None


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### 🎯 Resume Matcher")
    st.markdown(
        "<p style='color:#64748b;font-size:0.85rem;margin-top:-0.5rem;'>"
        "Portfolio-Grade ATS & NLP Resume Analyzer</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # API Key Status
    key_ok, key_msg = check_api_key_status()
    if key_ok:
        st.success(f"✅ {key_msg}", icon=None)
    else:
        st.info(f"ℹ️ {key_msg}\nLocal NLP & ATS metrics are active. Add Gemini key in `.env` for AI coaching.", icon=None)

    st.markdown("---")
    st.markdown("#### 🧠 Architecture")
    st.markdown("""
- **NLP Engine**: Scikit-Learn TF-IDF vectorizer (sublinear TF, unigrams + bigrams)
- **Matching Metric**: Cosine similarity & weighted keyword coverage
- **ATS Health**: Section hierarchy, quantifiable metric extraction, action verbs audit
- **AI Coach**: Google Gemini 1.5 Flash structured synthesis
""")

    st.markdown("---")
    if st.button("🗑️ Reset All", use_container_width=True):
        st.session_state.clear()
        st.rerun()


# ============================================================
# Hero Section
# ============================================================
st.markdown("""
<div class="hero-card">
  <div class="hero-badge">✨ AI-Powered &bull; NLP & Google Gemini</div>
  <h1 class="hero-title">AI Resume–Job Match Analyzer</h1>
  <p class="hero-subtitle">
    Evaluate job alignment with precision. Get an instant ATS similarity score,
    identify critical missing skills, audit resume formatting health, and generate
    tailored resume bullet points.
  </p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# Input Section: Two Columns (Resume vs Job Description)
# ============================================================
col_resume, col_job = st.columns(2, gap="large")

with col_resume:
    with st.container(border=True):
        st.markdown('<div class="card-header-title">📄 Candidate Resume</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Resume (.pdf or .txt)",
            type=["pdf", "txt"],
            key="resume_uploader",
            help="Upload your resume file or paste text below.",
        )

        if uploaded_file is not None:
            extracted, err = extract_text_from_file(uploaded_file)
            if err:
                st.error(f"Upload error: {err}")
            else:
                st.session_state["resume_content"] = extracted
                st.success(f"Extracted {len(extracted):,} characters from `{uploaded_file.name}`")

        resume_text = st.text_area(
            "Resume Text",
            value=st.session_state.get("resume_content", ""),
            height=300,
            placeholder="Paste your full resume text here...\n(Education, experience, skills, projects)",
            key="txt_resume",
            label_visibility="collapsed",
        )
        st.session_state["resume_content"] = resume_text

        word_count_res = len(resume_text.split()) if resume_text.strip() else 0
        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.75rem;text-align:right;margin-top:0.3rem;'>"
            f"Words: {word_count_res} | Characters: {len(resume_text):,}</p>",
            unsafe_allow_html=True,
        )


with col_job:
    with st.container(border=True):
        st.markdown('<div class="card-header-title">💼 Target Job Description</div>', unsafe_allow_html=True)

        job_text = st.text_area(
            "Job Description Text",
            value=st.session_state.get("job_content", ""),
            height=358,
            placeholder="Paste the target job description here...\n(Responsibilities, required skills, qualifications)",
            key="txt_job",
            label_visibility="collapsed",
        )
        st.session_state["job_content"] = job_text

        word_count_job = len(job_text.split()) if job_text.strip() else 0
        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.75rem;text-align:right;margin-top:0.3rem;'>"
            f"Words: {word_count_job} | Characters: {len(job_text):,}</p>",
            unsafe_allow_html=True,
        )


# ============================================================
# Action Buttons
# ============================================================
btn_col1, btn_col2, _ = st.columns([1.2, 0.8, 3])
with btn_col1:
    analyze_clicked = st.button("🚀 Analyze Alignment", use_container_width=True)
with btn_col2:
    if st.button("🗑️ Clear Inputs", use_container_width=True):
        st.session_state["resume_content"] = ""
        st.session_state["job_content"] = ""
        st.session_state["last_analysis"] = None
        st.session_state["ai_analysis"] = None
        st.rerun()


# ============================================================
# Execute Analysis
# ============================================================
if analyze_clicked:
    if not resume_text.strip():
        st.error("Please provide your resume (paste text or upload a PDF/TXT file).")
        st.stop()
    if not job_text.strip():
        st.error("Please paste the target job description.")
        st.stop()

    with st.spinner("Extracting n-grams, computing TF-IDF similarity, and running ATS audit..."):
        analysis_data = analyze(resume_text, job_text)
        st.session_state["last_analysis"] = analysis_data

    # Optional Gemini AI Analysis
    with st.spinner("Synthesizing personalized AI feedback with Gemini 1.5..."):
        ai_res, ai_err = get_ai_suggestions(
            resume_text,
            job_text,
            analysis_data["composite_score"],
            analysis_data["missing"],
        )
        st.session_state["ai_analysis"] = ai_res
        st.session_state["ai_error"] = ai_err


# ============================================================
# Display Results
# ============================================================
results = st.session_state.get("last_analysis")

if results:
    similarity = results["similarity"]
    composite = results["composite_score"]
    coverage = results["keyword_coverage"]
    matched = results["matched"]
    missing = results["missing"]
    suggested = results["suggested"]
    cat_matched = results["categorized_matched"]
    cat_missing = results["categorized_missing"]
    audit = results["resume_audit"]

    ai_result = st.session_state.get("ai_analysis")
    ai_error = st.session_state.get("ai_error")

    palette = get_score_palette(composite)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("## 📊 Analysis & Optimization Results")

    # ---- Top Summary Banner ----
    col_score_hero, col_metrics_hero = st.columns([1.1, 2], gap="large")

    with col_score_hero:
        pct_display = f"{composite * 100:.1f}%"
        st.markdown(f"""
<div class="score-hero-box">
  <div class="score-large-number" style="color: {palette['color']};">{pct_display}</div>
  <div class="score-badge-pill" style="background-color: {palette['bg']}; color: {palette['color']}; border: 1px solid {palette['border']};">
    {palette['badge']}
  </div>
  <p class="score-note">
    <strong>Composite ATS Match Score</strong>: Combines core technical keyword coverage (60%) with TF-IDF semantic overlap (40%).
  </p>
</div>
""", unsafe_allow_html=True)

    with col_metrics_hero:
        total_kw = len(matched) + len(missing)
        st.markdown(f"""
<div class="metric-grid">
  <div class="metric-box">
    <div class="metric-val" style="color: #059669;">{len(matched)}</div>
    <div class="metric-label">Matched Skills</div>
  </div>
  <div class="metric-box">
    <div class="metric-val" style="color: #e11d48;">{len(missing)}</div>
    <div class="metric-label">Missing Skills</div>
  </div>
  <div class="metric-box">
    <div class="metric-val" style="color: #d97706;">{len(suggested)}</div>
    <div class="metric-label">High-Impact Gaps</div>
  </div>
  <div class="metric-box">
    <div class="metric-val" style="color: #4f46e5;">{audit['score']}/100</div>
    <div class="metric-label">ATS Health Score</div>
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<p style='font-size:0.85rem;color:#475569;margin-bottom:0.3rem;'><strong>Overall Match Progress</strong></p>", unsafe_allow_html=True)
        st.progress(composite, text=f"Composite Match: {pct_display} | TF-IDF Raw: {similarity * 100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- Tabbed Deep Dive ----
    tab_matrix, tab_audit, tab_ai, tab_rewriter, tab_export = st.tabs([
        "🏷️ Skills & Keyword Matrix",
        "📋 ATS Resume Health",
        "🤖 AI Career Coach",
        "✍️ Bullet Point Rewriter",
        "📄 Export & Raw Data",
    ])

    # ============================================================
    # Tab 1: Skills & Keyword Matrix
    # ============================================================
    with tab_matrix:
        st.markdown("<br>", unsafe_allow_html=True)
        col_m_left, col_m_right = st.columns(2, gap="large")

        with col_m_left:
            with st.container(border=True):
                st.markdown(f'<div class="card-header-title">✅ Matched Keywords ({len(matched)})</div>', unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:0.8rem;'>Skills and competencies detected in both your resume and the job posting.</p>", unsafe_allow_html=True)

                if cat_matched:
                    for cat, items in cat_matched.items():
                        st.markdown(f"<p style='font-size:0.82rem;font-weight:700;color:#334155;margin:0.8rem 0 0.3rem 0;'>{cat} ({len(items)})</p>", unsafe_allow_html=True)
                        st.markdown(render_pastel_chips(items, "chip-matched"), unsafe_allow_html=True)
                else:
                    st.markdown(render_pastel_chips(matched, "chip-matched"), unsafe_allow_html=True)

        with col_m_right:
            with st.container(border=True):
                st.markdown(f'<div class="card-header-title">❌ Missing Keywords ({len(missing)})</div>', unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:0.8rem;'>Keywords requested by the employer that were not identified in your resume.</p>", unsafe_allow_html=True)

                if cat_missing:
                    for cat, items in cat_missing.items():
                        st.markdown(f"<p style='font-size:0.82rem;font-weight:700;color:#334155;margin:0.8rem 0 0.3rem 0;'>{cat} ({len(items)})</p>", unsafe_allow_html=True)
                        st.markdown(render_pastel_chips(items, "chip-missing"), unsafe_allow_html=True)
                else:
                    st.markdown(render_pastel_chips(missing, "chip-missing"), unsafe_allow_html=True)

                if suggested:
                    st.markdown("<hr style='margin:1.2rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:0.85rem;font-weight:700;color:#b45309;'>⭐ High Priority Technical Skills to Add ({len(suggested)})</p>", unsafe_allow_html=True)
                    st.markdown("<p style='font-size:0.78rem;color:#94a3b8;margin-bottom:0.5rem;'>Add these if you have coursework, projects, or self-taught experience in them.</p>", unsafe_allow_html=True)
                    st.markdown(render_pastel_chips(suggested, "chip-suggested"), unsafe_allow_html=True)

    # ============================================================
    # Tab 2: ATS Resume Health Audit
    # ============================================================
    with tab_audit:
        st.markdown("<br>", unsafe_allow_html=True)
        col_aud_left, col_aud_right = st.columns(2, gap="large")

        with col_aud_left:
            with st.container(border=True):
                st.markdown(f'<div class="card-header-title">🛡️ ATS Structural Checklist (Score: {audit["score"]}/100)</div>', unsafe_allow_html=True)

                for check in audit["checklist"]:
                    pill = '<span class="status-pill-pass">PASS</span>' if check["status"] else '<span class="status-pill-fail">MISSING</span>'
                    st.markdown(f"""
<div class="checklist-row">
  <span style="color:#334155;font-weight:500;">{check['item']}</span>
  {pill}
</div>
""", unsafe_allow_html=True)

                st.markdown("<hr style='margin:1.2rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#1e293b;margin-bottom:0.5rem;'>💡 Tailored ATS Feedback</p>", unsafe_allow_html=True)
                for tip in audit["tips"]:
                    st.markdown(f"<p style='font-size:0.82rem;color:#475569;margin-bottom:0.4rem;'>&bull; {tip}</p>", unsafe_allow_html=True)

        with col_aud_right:
            with st.container(border=True):
                st.markdown(f'<div class="card-header-title">⚡ Action Verbs Detected ({audit["action_verbs_count"]})</div>', unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:0.6rem;'>Recruiters and ATS favor strong past-tense action verbs at the beginning of experience bullets.</p>", unsafe_allow_html=True)
                st.markdown(render_pastel_chips(audit["action_verbs"], "chip-matched"), unsafe_allow_html=True)

                st.markdown("<hr style='margin:1.2rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
                st.markdown(f'<div class="card-header-title">📈 Quantifiable Metrics Found ({audit["metrics_count"]})</div>', unsafe_allow_html=True)
                st.markdown("<p style='font-size:0.8rem;color:#64748b;margin-bottom:0.6rem;'>Quantified achievements (percentages, counts, performance improvements) dramatically increase interview callbacks.</p>", unsafe_allow_html=True)
                st.markdown(render_pastel_chips(audit["metrics"], "chip-ai"), unsafe_allow_html=True)

                st.markdown("<hr style='margin:1.2rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
                st.markdown(f"""
<div style="font-size:0.82rem;color:#64748b;line-height:1.6;">
  &bull; Total Word Count: <strong>{audit['word_count']} words</strong><br>
  &bull; Estimated Recruiter Reading Time: <strong>{audit['reading_time_min']} minutes</strong>
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Tab 3: AI Career Coach (Gemini)
    # ============================================================
    with tab_ai:
        st.markdown("<br>", unsafe_allow_html=True)
        if ai_error:
            st.info(f"ℹ️ {ai_error}")

        if ai_result:
            if ai_result.get("summary"):
                st.markdown(f"""
<div class="ai-summary-card">
  <strong style="color: #4f46e5;">🤖 Executive AI Summary:</strong><br>
  {ai_result["summary"]}
</div>
""", unsafe_allow_html=True)

            col_ai_left, col_ai_right = st.columns(2, gap="large")

            with col_ai_left:
                with st.container(border=True):
                    st.markdown('<div class="card-header-title">💡 Actionable Improvement Suggestions</div>', unsafe_allow_html=True)
                    for i, sug in enumerate(ai_result.get("improvement_suggestions", []), 1):
                        st.markdown(f"""
<div class="suggestion-card">
  <strong style="color:#6366f1;">#{i}</strong> {sug}
</div>
""", unsafe_allow_html=True)

            with col_ai_right:
                with st.container(border=True):
                    st.markdown('<div class="card-header-title">🚀 Portfolio Project Ideas to Close Skill Gaps</div>', unsafe_allow_html=True)
                    for proj in ai_result.get("project_suggestions", []):
                        st.markdown(f"""
<div class="project-card">
  <strong>Project Concept:</strong> {proj}
</div>
""", unsafe_allow_html=True)

                    if ai_result.get("resume_keywords"):
                        st.markdown("<hr style='margin:1.2rem 0;border-color:#f1f5f9;'>", unsafe_allow_html=True)
                        st.markdown('<div class="card-header-title">💎 Your Standout Strengths</div>', unsafe_allow_html=True)
                        st.markdown(render_pastel_chips(ai_result["resume_keywords"], "chip-ai"), unsafe_allow_html=True)

    # ============================================================
    # Tab 4: Interactive Bullet Point Rewriter
    # ============================================================
    with tab_rewriter:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<div class="card-header-title">✍️ Google XYZ Resume Bullet Optimizer</div>', unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.85rem;color:#64748b;margin-bottom:1.2rem;'>"
                "Transform standard resume bullet points into high-impact, ATS-optimized statements following Google's "
                "formula: <em>'Accomplished [X] as measured by [Y] by doing [Z]'</em>.</p>",
                unsafe_allow_html=True,
            )

            sample_bullet = "Worked on web app and added user login and backend APIs."
            col_rw1, col_rw2 = st.columns([1, 1], gap="large")

            with col_rw1:
                input_bullet = st.text_area(
                    "Original Resume Bullet Point",
                    value=sample_bullet,
                    height=110,
                    key="rewriter_input_bullet",
                    help="Paste any existing bullet point from your resume.",
                )

                # Dropdown of missing or suggested skills
                skill_options = suggested if suggested else (missing[:15] if missing else ["Docker", "PostgreSQL", "CI/CD"])
                target_skill = st.selectbox(
                    "Target Skill to Incorporate (from Job Description)",
                    options=skill_options,
                    key="rewriter_target_skill",
                )

                rewrite_clicked = st.button("✨ Optimize Bullet Point", key="btn_rewrite")

            with col_rw2:
                st.markdown("<p style='font-size:0.85rem;font-weight:700;color:#1e293b;margin-top:0.3rem;'>Optimized ATS Statement:</p>", unsafe_allow_html=True)

                if rewrite_clicked or "last_rewritten_bullet" in st.session_state:
                    if rewrite_clicked:
                        with st.spinner("Crafting XYZ impact statement..."):
                            optimized, rw_err = rewrite_bullet_point(
                                input_bullet, target_skill, job_text[:200]
                            )
                            st.session_state["last_rewritten_bullet"] = optimized
                            st.session_state["last_rewritten_err"] = rw_err

                    opt_text = st.session_state.get("last_rewritten_bullet", "")
                    st.markdown(f"""
<div style="background:#f8faff; border:1px solid #c7d2fe; border-radius:12px; padding:1.2rem; font-size:0.95rem; color:#1e293b; line-height:1.6;">
  &bull; {opt_text}
</div>
""", unsafe_allow_html=True)
                    st.caption("Copy and paste directly into your resume under your relevant project or work experience.")
                else:
                    st.markdown("""
<div style="background:#f8fafc; border:1px dashed #cbd5e1; border-radius:12px; padding:2rem 1.2rem; text-align:center; color:#94a3b8; font-size:0.88rem;">
  Click <strong>Optimize Bullet Point</strong> to generate an ATS-ready XYZ bullet statement.
</div>
""", unsafe_allow_html=True)

    # ============================================================
    # Tab 5: Export & Raw Data
    # ============================================================
    with tab_export:
        st.markdown("<br>", unsafe_allow_html=True)
        col_exp1, col_exp2 = st.columns(2, gap="large")

        with col_exp1:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">📥 Download Comprehensive Match Report</div>', unsafe_allow_html=True)

                report_lines = [
                    "# AI Resume Match & ATS Optimization Report",
                    "",
                    f"**Overall Composite Match Score**: {composite * 100:.1f}%",
                    f"**TF-IDF Semantic Cosine Similarity**: {similarity * 100:.1f}%",
                    f"**Keyword Coverage**: {coverage * 100:.1f}%",
                    f"**ATS Resume Health Score**: {audit['score']}/100",
                    "",
                    "## 1. Matched Competencies (" + str(len(matched)) + ")",
                    ", ".join(matched) if matched else "None",
                    "",
                    "## 2. Missing Skills (" + str(len(missing)) + ")",
                    ", ".join(missing) if missing else "None",
                    "",
                    "## 3. High-Priority Gaps to Address",
                    ", ".join(suggested) if suggested else "None",
                    "",
                    "## 4. ATS Formatting & Structure Audit",
                    f"- Word count: {audit['word_count']} words",
                    f"- Action verbs detected: {', '.join(audit['action_verbs']) if audit['action_verbs'] else 'None'}",
                    f"- Quantified metrics detected: {', '.join(audit['metrics']) if audit['metrics'] else 'None'}",
                    "",
                    "### Recommendations:",
                ]
                for tip in audit["tips"]:
                    report_lines.append(f"- {tip}")

                if ai_result:
                    report_lines += [
                        "",
                        "## 5. AI Career Coach Guidance",
                        ai_result.get("summary", ""),
                        "",
                        "### Actionable Suggestions:",
                    ]
                    for sug in ai_result.get("improvement_suggestions", []):
                        report_lines.append(f"- {sug}")
                    report_lines += ["", "### Portfolio Project Recommendations:"]
                    for proj in ai_result.get("project_suggestions", []):
                        report_lines.append(f"- {proj}")

                report_content = "\n".join(report_lines)

                st.download_button(
                    label="⬇️ Download Markdown Report (.md)",
                    data=report_content,
                    file_name="resume_match_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

                st.download_button(
                    label="⬇️ Download Plain Text (.txt)",
                    data=report_content,
                    file_name="resume_match_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

        with col_exp2:
            with st.container(border=True):
                st.markdown('<div class="card-header-title">🔬 Raw JSON Telemetry</div>', unsafe_allow_html=True)
                st.json({
                    "composite_score": composite,
                    "tfidf_similarity": similarity,
                    "keyword_coverage": coverage,
                    "matched_count": len(matched),
                    "missing_count": len(missing),
                    "ats_health_score": audit["score"],
                    "action_verbs_count": audit["action_verbs_count"],
                    "metrics_count": audit["metrics_count"],
                })


st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.78rem; padding:1.5rem 0;">
  🎯 AI Resume–Job Match Analyzer &bull; Powered by scikit-learn & Google Gemini
</div>
""", unsafe_allow_html=True)
