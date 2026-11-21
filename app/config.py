"""
Central configuration for the Government Orders Search System.
"""

from pathlib import Path


# Main project directory
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Dataset directory containing original Government Order PDFs
DATASET_DIR: Path = PROJECT_ROOT / "dataset"

# Storage directories
STORAGE_DIR: Path = PROJECT_ROOT / "storage"
EXTRACTED_DIR: Path = STORAGE_DIR / "extracted"

# Tesseract OCR executable
TESSERACT_PATH: str = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

# Supported OCR languages
OCR_LANGUAGES: str = "eng+mal"

# Qdrant Vector Storage Configuration (Local Persistent Mode)
QDRANT_STORAGE_PATH: Path = STORAGE_DIR / "qdrant"
QDRANT_COLLECTION_NAME: str = "kerala_government_orders"
EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSION: int = 384
