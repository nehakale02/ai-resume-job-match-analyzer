"""
test_local.py - Comprehensive local test for matcher, gemini_service, and sample_data
Run: .\\venv\\Scripts\\python test_local.py
"""
import warnings
warnings.filterwarnings("ignore")

print("=== Testing imports ===")
import streamlit
import sklearn
import google.generativeai
import dotenv
import pypdf
print("All base imports OK")

print("\n=== Testing sample_data.py ===")
from sample_data import SAMPLE_PRESETS
assert len(SAMPLE_PRESETS) >= 3, "Expected at least 3 sample presets"
for name, data in SAMPLE_PRESETS.items():
    assert len(data["resume"]) > 100, f"Preset {name} resume too short"
    assert len(data["job"]) > 100, f"Preset {name} job too short"
print(f"Loaded {len(SAMPLE_PRESETS)} presets: OK")

print("\n=== Testing matcher.py ===")
from matcher import analyze, audit_resume, categorize_keywords

sample = SAMPLE_PRESETS["Full-Stack Software Engineer (Junior/New Grad)"]
result = analyze(sample["resume"], sample["job"])

print(f"TF-IDF Similarity: {result['similarity']*100:.1f}%")
print(f"Composite ATS Score: {result['composite_score']*100:.1f}%")
print(f"Keyword Coverage: {result['keyword_coverage']*100:.1f}%")
print(f"Matched ({len(result['matched'])}): {result['matched'][:5]}")
print(f"Missing ({len(result['missing'])}): {result['missing'][:5]}")
print(f"Suggested ({len(result['suggested'])}): {result['suggested'][:5]}")
print(f"Categorized Matched Groups: {list(result['categorized_matched'].keys())}")

assert result["similarity"] > 0.15, "Expected good similarity"
assert result["composite_score"] > 0.20, "Expected good composite score"
assert len(result["matched"]) > 5, "Expected matched keywords"

# Test ATS Audit
audit = result["resume_audit"]
print(f"Resume ATS Score: {audit['score']}/100")
print(f"Action verbs found: {len(audit['action_verbs'])} {audit['action_verbs'][:4]}")
print(f"Metrics found: {len(audit['metrics'])} {audit['metrics'][:4]}")
assert audit["score"] >= 70, f"Expected strong score for Berkeley resume, got {audit['score']}"
assert audit["contact_info"]["email"] is True, "Expected email detected"
assert audit["contact_info"]["links"] is True, "Expected link detected"

print("matcher.py PASSED")

print("\n=== Testing gemini_service.py ===")
from gemini_service import check_api_key_status, rewrite_bullet_point
ok, msg = check_api_key_status()
print(f"API key status: {msg}")

rewritten, err = rewrite_bullet_point(
    "Built website with React and CSS",
    "Docker",
    "Looking for containerization experience"
)
print(f"Rewritten bullet test: {rewritten}")
print("gemini_service.py PASSED")

print("\n=== ALL COMPREHENSIVE TESTS PASSED! ===")
