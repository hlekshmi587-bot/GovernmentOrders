
"""
Classify Government Order PDF pages based on text extraction quality.
"""

from enum import Enum
from pathlib import Path

import pymupdf

from app.services.pdf_service import PDFService


class PDFType(str, Enum):
    """Possible PDF page classifications."""

    NATIVE = "NATIVE"
    SCANNED = "SCANNED"
    CORRUPTED = "CORRUPTED"


class PDFClassifier:
    """Classify PDF documents using embedded text quality."""

    def __init__(self) -> None:
        self.pdf_service = PDFService()

    def classify_page(self, page: pymupdf.Page) -> PDFType:
        """Classify one page of a PDF."""

        embedded_text = page.get_text("text").strip()

        if not embedded_text:
            return PDFType.SCANNED

        if self.pdf_service._is_extraction_poor(embedded_text):
            return PDFType.CORRUPTED

        return PDFType.NATIVE

    def classify_pdf(self, pdf_path: Path) -> dict:
        """Classify a PDF using the classification of its pages."""

        page_results = []

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                page_type = self.classify_page(page)

                page_results.append({
                    "page": page_number,
                    "type": page_type.value,
                })

        page_types = {
            result["type"]
            for result in page_results
        }

        if page_types == {PDFType.NATIVE.value}:
            overall_type = "NATIVE"
        elif page_types == {PDFType.SCANNED.value}:
            overall_type = "SCANNED"
        elif page_types == {PDFType.CORRUPTED.value}:
            overall_type = "CORRUPTED"
        else:
            overall_type = "MIXED"

        return {
            "filename": pdf_path.name,
            "overall_type": overall_type,
            "pages": page_results,
        }
