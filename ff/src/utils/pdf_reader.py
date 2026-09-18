"""
PDF Reader Utility
===================
Extracts text from uploaded PDF files using PyPDF2.
"""

from __future__ import annotations

from io import BytesIO

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Read all pages from a PDF and return the concatenated text."""
    reader = PdfReader(BytesIO(file_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text.strip())
    return "\n\n".join(pages_text)
