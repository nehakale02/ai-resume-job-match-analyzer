"""
utils.py
--------
File upload helpers — extract text from PDF and TXT files.
"""

import io
from typing import Optional


def extract_text_from_file(uploaded_file) -> tuple[str, Optional[str]]:
    """
    Extract plain text from a Streamlit UploadedFile object.
    Supports .txt and .pdf files.

    Returns:
        (text, error_message)
        text is the extracted text (may be empty string on failure)
        error_message is None on success, otherwise a human-readable error
    """
    filename: str = uploaded_file.name.lower()
    raw_bytes: bytes = uploaded_file.read()

    # ---- TXT ----
    if filename.endswith(".txt"):
        try:
            text = raw_bytes.decode("utf-8", errors="replace")
            return text.strip(), None
        except Exception as e:
            return "", f"Could not read TXT file: {e}"

    # ---- PDF ----
    if filename.endswith(".pdf"):
        try:
            import pypdf  # pypdf is the maintained fork of PyPDF2

            reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
            pages_text = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    pages_text.append(page_text)
            full_text = "\n".join(pages_text).strip()
            if not full_text:
                return "", "PDF parsed but no text could be extracted. The file may be scanned / image-only."
            return full_text, None
        except ImportError:
            return "", "pypdf is not installed. Run: pip install pypdf"
        except Exception as e:
            return "", f"Could not read PDF file: {e}"

    return "", f"Unsupported file type: '{uploaded_file.name}'. Please upload a .pdf or .txt file."
