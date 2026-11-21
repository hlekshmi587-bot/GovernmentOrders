
"""
PDF text extraction and OCR service for Government Orders.
"""

from pathlib import Path

import pymupdf
import pytesseract
from PIL import Image

from app.config import TESSERACT_PATH, OCR_LANGUAGES


class PDFService:
    """Extract embedded text and OCR from Government Order PDFs."""

    def __init__(self) -> None:
        """Configure the Tesseract OCR executable."""
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    @staticmethod
    def _is_extraction_poor(text: str) -> bool:
        """Detect missing or corrupted embedded text."""

        if not text.strip():
            return True

        mojibake_patterns = [
            "à´",
            "àµ",
            "Ã",
            "â€",
            "ï¿½",
        ]

        mojibake_count = sum(
            text.count(pattern)
            for pattern in mojibake_patterns
        )

        control_characters = sum(
            1
            for char in text
            if ord(char) < 32 and char not in "\n\r\t\f"
        )

        return mojibake_count >= 3 or control_characters >= 2

    @staticmethod
    def _perform_ocr(page: pymupdf.Page) -> str:
        """Run OCR on one PDF page."""

        pixmap = page.get_pixmap(
            matrix=pymupdf.Matrix(2, 2),
            alpha=False,
        )

        image = Image.frombytes(
            "RGB",
            [pixmap.width, pixmap.height],
            pixmap.samples,
        )

        return pytesseract.image_to_string(
            image,
            lang=OCR_LANGUAGES,
        ).strip()

    def extract_text(self, pdf_path: Path) -> str:
        """
        Extract embedded text and OCR when necessary.

        Both sources are preserved when embedded text is corrupted.
        """

        extracted_pages: list[str] = []

        with pymupdf.open(pdf_path) as document:
            for page_number, page in enumerate(document, start=1):
                embedded_text = page.get_text("text").strip()

                if self._is_extraction_poor(embedded_text):
                    print(
                        f"Embedded text may be corrupted. "
                        f"Running OCR for page {page_number} "
                        f"of {pdf_path.name}"
                    )

                    ocr_text = self._perform_ocr(page)

                    page_text = (
                        f"--- Page {page_number} ---\n"
                        f"--- Embedded Text ---\n"
                        f"{embedded_text}\n\n"
                        f"--- OCR Text ---\n"
                        f"{ocr_text}"
                    )
                else:
                    page_text = (
                        f"--- Page {page_number} ---\n"
                        f"{embedded_text}"
                    )

                extracted_pages.append(page_text)

        return "\n\n".join(extracted_pages)
