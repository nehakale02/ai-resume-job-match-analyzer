"""
gemini_service.py
-----------------
Handles all communication with the Google Gemini API.
Uses google-generativeai SDK (gemini-1.5-flash model).
Reads the API key from a .env file, builds the prompt,
parses structured JSON responses, and handles all errors gracefully.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple
import warnings

# Suppress SDK deprecation FutureWarning before import
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import google.generativeai as genai  # noqa: E402
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------------------------
load_dotenv()  # Reads .env file in the current working directory

# ---------------------------------------------------------------------------
# Gemini model config
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-3.8-flash"   # Latest Gemini Flash model
MAX_RETRIES = 2
RETRY_DELAY = 3  # seconds

# ---------------------------------------------------------------------------
# Expected JSON schema (used for validation)
# ---------------------------------------------------------------------------
EXPECTED_KEYS = {
    "summary": str,
    "missing_skills": list,
    "improvement_suggestions": list,
    "resume_keywords": list,
    "project_suggestions": list,
}

DEFAULT_RESPONSE: dict = {
    "summary": "AI analysis unavailable. See keyword analysis below.",
    "missing_skills": [],
    "improvement_suggestions": [],
    "resume_keywords": [],
    "project_suggestions": [],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """
    Retrieve the Gemini API key from environment variables or .env file.
    Returns None if not found or placeholder.
    """
    load_dotenv(override=True)
    key = os.getenv("GEMINI_API_KEY")
    if not key or "your_" in key.lower():
        return None
    return key.strip()


def _configure_genai(api_key: str) -> None:
    """Configure the google-generativeai SDK with the given key using REST transport."""
    genai.configure(api_key=api_key, transport="rest")


def _build_prompt(
    resume_text: str,
    job_text: str,
    similarity_score: float,
    missing_keywords: list,
) -> str:
    """
    Build the structured prompt sent to Gemini.
    Truncates very long inputs to avoid token limits.
    """
    max_chars = 3000
    resume_snippet = resume_text[:max_chars] + ("..." if len(resume_text) > max_chars else "")
    job_snippet = job_text[:max_chars] + ("..." if len(job_text) > max_chars else "")
    missing_str = ", ".join(missing_keywords[:30]) if missing_keywords else "None identified"
    similarity_pct = f"{similarity_score * 100:.1f}%"

    prompt = f"""You are an expert career coach and resume consultant. Analyze this resume against the job description and provide actionable, personalized advice.

=== RESUME ===
{resume_snippet}

=== JOB DESCRIPTION ===
{job_snippet}

=== ANALYSIS DATA ===
Text Similarity Score: {similarity_pct}
Keywords missing from resume: {missing_str}

=== YOUR TASK ===
Provide a structured JSON response with EXACTLY these fields and no markdown code blocks:

{{
  "summary": "2-3 sentence analysis of how well the resume matches this job, noting key strengths and gaps",
  "missing_skills": ["List of important skills/technologies from the JD genuinely absent from resume, max 10"],
  "improvement_suggestions": [
    "Specific, actionable suggestion 1 (e.g., Add a bullet under your X role demonstrating Y impact)",
    "Specific, actionable suggestion 2",
    "Specific, actionable suggestion 3",
    "Specific, actionable suggestion 4",
    "Specific, actionable suggestion 5"
  ],
  "resume_keywords": ["Strong keywords/skills already present in resume worth highlighting, max 10"],
  "project_suggestions": ["Concrete project idea that would demonstrate missing skills, max 3"]
}}

Rules:
- Return ONLY valid JSON, no markdown, no explanation outside the JSON
- Be specific to THIS resume and THIS job, not generic advice
- Only list skills in missing_skills if they are genuinely important for the role and absent from the resume
- Improvement suggestions must reference actual content from the resume or job description
- Keep each suggestion concise (1-2 sentences max)"""

    return prompt


def _extract_json_from_text(text: str) -> Optional[dict]:
    """
    Try multiple strategies to extract a JSON object from the model's response.
    """
    # Strategy 1: Direct parse
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Strategy 2: Extract from markdown code block
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # Strategy 3: Find first {...} block
    match = re.search(r"\{[\s\S]+\}", text)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def _validate_response(data: dict) -> dict:
    """
    Ensure all expected keys are present and of the correct type.
    Fills in defaults for missing or malformed fields.
    """
    validated = {}
    for key, expected_type in EXPECTED_KEYS.items():
        value = data.get(key)
        if isinstance(value, expected_type) and value:
            validated[key] = value
        else:
            validated[key] = DEFAULT_RESPONSE[key]
    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ai_suggestions(
    resume_text: str,
    job_text: str,
    similarity_score: float,
    missing_keywords: list,
) -> tuple:
    """
    Send resume and job description to Gemini and return structured suggestions.

    Returns:
        (result_dict, error_message)
        - result_dict: validated response dict (always safe to display)
        - error_message: human-readable error string, or None if successful
    """
    # --- Check API key ---
    api_key = _get_api_key()
    if not api_key:
        return DEFAULT_RESPONSE.copy(), (
            "GEMINI_API_KEY not found. Please add it to your .env file. "
            "Showing local keyword analysis only."
        )

    # --- Configure SDK ---
    try:
        _configure_genai(api_key)
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        return DEFAULT_RESPONSE.copy(), f"Failed to initialise Gemini SDK: {e}"

    # --- Build prompt ---
    prompt = _build_prompt(resume_text, job_text, similarity_score, missing_keywords)

    # --- Call API with retries ---
    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            generation_config = genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            )
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            raw_text = response.text.strip()

            # --- Parse JSON ---
            parsed = _extract_json_from_text(raw_text)
            if parsed is None:
                last_error = "Gemini returned a response that could not be parsed as JSON."
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_DELAY)
                    continue
                return DEFAULT_RESPONSE.copy(), last_error

            # --- Validate ---
            validated = _validate_response(parsed)
            return validated, None  # Success

        except Exception as e:
            err_str = str(e)
            if "quota" in err_str.lower() or "429" in err_str or "resource_exhausted" in err_str.lower() or "rate" in err_str.lower():
                last_error = "Gemini API rate limit reached (Free Tier limit: 5 requests/min). Please wait a few seconds and try again."
            elif "api_key_invalid" in err_str.lower() or "api key not valid" in err_str.lower():
                last_error = "Invalid Gemini API key. Please verify your key at Google AI Studio."
                return DEFAULT_RESPONSE.copy(), last_error  # No point retrying
            elif "network" in err_str.lower() or "connection" in err_str.lower():
                last_error = "Network error connecting to Gemini API. Check your internet connection."
            else:
                last_error = f"Gemini API note: {err_str}"

            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    return DEFAULT_RESPONSE.copy(), last_error


def check_api_key_status() -> tuple:
    """
    Quick check to see if an API key is configured (does not make an API call).

    Returns:
        (is_configured: bool, status_message: str)
    """
    key = _get_api_key()
    if not key:
        return False, "No API key configured."
    if len(key) < 10:
        return False, "API key looks too short -- please verify it."
    # Mask the key for display: show first 6 chars then asterisks
    masked = key[:6] + "*" * (len(key) - 6)
    return True, f"API key configured ({masked})."


def rewrite_bullet_point(
    bullet_text: str,
    target_skill: str,
    job_context: str = "",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Rewrites a single resume bullet point to naturally integrate a missing skill
    using the Google XYZ formula (Accomplished [X] as measured by [Y] by doing [Z]).

    Returns:
        (rewritten_bullet, error_message)
    """
    api_key = _get_api_key()
    if not api_key:
        # High quality local fallback template if API key is not set
        action_verb = "Engineered and deployed"
        fallback = f"{action_verb} a core module utilizing {target_skill}, enhancing system reliability and reducing processing latency by 35%."
        return fallback, "Using local XYZ formula template (Gemini API key not configured)."

    try:
        _configure_genai(api_key)
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""You are an elite tech resume coach. Rewrite this single resume bullet point to strongly incorporate the target skill: "{target_skill}".
Follow Google's XYZ formula: "Accomplished [X], as measured by [Y], by doing [Z]".

Original bullet point:
"{bullet_text.strip()}"

Target skill to weave in:
"{target_skill.strip()}"

Optional Job Context:
"{job_context[:300].strip()}"

Rules:
- Output ONLY the single rewritten bullet point. No preamble, no quotes, no markdown, no explanation.
- Start with a powerful past-tense action verb (e.g., Architected, Optimized, Spearheaded, Implemented).
- Keep it under 35 words.
- Include a realistic, quantifiable metric (%, ms, $, or users) where appropriate.
"""
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=150,
            ),
        )
        rewritten = response.text.strip().strip('"').strip("'").strip("- ")
        return rewritten, None
    except Exception as e:
        # Fallback template
        fallback = f"Architected high-impact solutions with {target_skill}, accelerating deliverable throughput by 25% across the pipeline."
        return fallback, f"Gemini error ({e}); provided fallback template."

