"""
Document chunking service for Government Orders.

Splits document text into semantic chunks respecting paragraph, clause,
and page boundaries for vector embedding and retrieval.
"""

import re
from typing import Any, Dict, List, Optional


def chunk_document_text(
    full_text: str,
    target_chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Dict[str, Any]]:
    """
    Split Government Order text into meaningful chunks respecting paragraph,
    numbered clause, and page boundaries.

    Args:
        full_text: Complete extracted document text.
        target_chunk_size: Ideal character size for semantic embedding.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        List of dicts: [{'chunk_index': int, 'chunk_text': str, 'page_number': Optional[int]}]
    """
    if not full_text or not full_text.strip():
        return []

    # Detect page markers if present (e.g. "--- Page 1 ---" or form feed)
    page_splits = re.split(r"(--- Page \d+ ---|\f)", full_text)

    raw_segments: List[Dict[str, Any]] = []
    current_page: Optional[int] = None

    for part in page_splits:
        part_clean = part.strip()
        if not part_clean:
            continue

        page_match = re.search(r"Page\s+(\d+)", part_clean, re.IGNORECASE)
        if page_match:
            try:
                current_page = int(page_match.group(1))
            except ValueError:
                pass
            continue

        # Split part into paragraphs / numbered clauses
        paragraphs = re.split(r"\n\s*\n|\n(?=\d+[\).]\s+)", part_clean)
        for p in paragraphs:
            p_text = re.sub(r"\s+", " ", p).strip()
            if p_text:
                raw_segments.append({"text": p_text, "page": current_page})

    # Combine small paragraphs into target_chunk_size blocks
    chunks: List[Dict[str, Any]] = []
    chunk_idx = 1
    current_chunk_text = ""
    current_chunk_page: Optional[int] = None

    for seg in raw_segments:
        seg_text = seg["text"]
        seg_page = seg["page"]

        if not current_chunk_text:
            current_chunk_text = seg_text
            current_chunk_page = seg_page
        elif seg_page is not None and current_chunk_page is not None and seg_page != current_chunk_page:
            # Force chunk boundary when crossing between different document pages
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_text": current_chunk_text,
                    "page_number": current_chunk_page,
                }
            )
            chunk_idx += 1
            current_chunk_text = seg_text
            current_chunk_page = seg_page
        elif len(current_chunk_text) + len(seg_text) + 1 <= target_chunk_size:
            current_chunk_text += " " + seg_text
        else:
            # Chunk is full
            chunks.append(
                {
                    "chunk_index": chunk_idx,
                    "chunk_text": current_chunk_text,
                    "page_number": current_chunk_page,
                }
            )
            chunk_idx += 1
            # Retain overlap from end of previous chunk if possible
            if len(current_chunk_text) > chunk_overlap and current_chunk_page == seg_page:
                overlap_text = current_chunk_text[-chunk_overlap:]
                # Cut at first space in overlap to avoid word truncation
                space_idx = overlap_text.find(" ")
                if space_idx != -1:
                    overlap_text = overlap_text[space_idx + 1:]
                current_chunk_text = overlap_text + " " + seg_text
            else:
                current_chunk_text = seg_text
            current_chunk_page = seg_page

    # Add remaining text
    if current_chunk_text.strip():
        chunks.append(
            {
                "chunk_index": chunk_idx,
                "chunk_text": current_chunk_text.strip(),
                "page_number": current_chunk_page,
            }
        )

    return chunks
