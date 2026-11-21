
"""
Rule-based relationship and reference extraction service
for Government Orders.
"""

import re

from app.models.identifier_extraction import (
    ExtractedReference,
    ExtractedRelationship,
    ExtractionConfidence,
    RelationshipType,
)


class RelationshipExtractionService:
    """Extract relationship signals and GO references."""

    RELATIONSHIP_PATTERNS = {
        RelationshipType.PARTIALLY_AMENDS: [
            r"\bpartially\s+amended\b",
            r"\bpartly\s+amended\b",
            r"\bpartially\s+amends?\b",
        ],
        RelationshipType.AMENDS: [
            r"\bfurther\s+to\s+amend\b",
            r"\bamendment\s+to\b",
            r"\bamended\s+by\b",
            r"\bamended\b",
            r"\bamends?\b",
            r"\bamending\b",
            r"ഭേദഗതി\s+ചെയ്ത",
        ],
        RelationshipType.MODIFIES: [
            r"\bmodified\s+by\b",
            r"\bhereby\s+modified\b",
            r"\bmodifies\b",
            r"\bmodified\b",
            r"\bmodification\s+of\b",
            r"മാറ്റം\s+വരുത്ത",
        ],
        RelationshipType.EXTENDS: [
            r"\bextended\s+by\b",
            r"\bperiod\s+is\s+extended\b",
            r"\bextension\s+of\b",
            r"\bextends\b",
            r"ദീർഘിപ്പി",
        ],
        RelationshipType.SUPERSEDES: [
            r"\bsuperseded\s+by\b",
            r"\bin\s+partial\s+supersession\s+of\b",
            r"\bin\s+supersession\s+of\b",
            r"\bsupersedes\b",
        ],
        RelationshipType.CANCELS: [
            r"\bstands?\s+cancelled\b",
            r"\bstands?\s+canceled\b",
            r"\bhereby\s+cancelled\b",
            r"\bhereby\s+canceled\b",
            r"\bcancel(?:led|ed|s)?\b",
            r"റദ്ദാക്ക",
            r"റദ്ദ്\s+ചെയ്ത",
        ],
        RelationshipType.REVOKES: [
            r"\brevoked\s+by\b",
            r"\bhereby\s+revoked\b",
            r"\brevokes?\b",
            r"\bwithdrawn\b",
            r"\bwithdraws?\b",
        ],
        RelationshipType.READ_AS: [
            r"\bshall\s+be\s+read\s+as\b",
            r"\bshould\s+be\s+read\s+as\b",
            r"\bto\s+be\s+read\s+as\b",
            r"\bread\s+as\s+follows\b",
        ],
    }

    @staticmethod
    def _split_pages(text: str) -> list[tuple[int, str]]:
        """Separate extracted text using page markers."""

        page_pattern = re.compile(
            r"---\s*Page\s+(\d+)\s*---",
            re.IGNORECASE,
        )

        matches = list(page_pattern.finditer(text))

        if not matches:
            return [(1, text)]

        pages = []

        for index, match in enumerate(matches):
            start = match.end()

            if index + 1 < len(matches):
                end = matches[index + 1].start()
            else:
                end = len(text)

            pages.append(
                (
                    int(match.group(1)),
                    text[start:end].strip(),
                )
            )

        return pages

    @staticmethod
    def _get_evidence_sentences(text: str) -> list[str]:
        """Split page text into possible evidence sentences."""

        cleaned_text = re.sub(
            r"---\s*(?:Embedded Text|OCR Text)\s*---",
            " ",
            text,
            flags=re.IGNORECASE,
        )

        cleaned_text = re.sub(
            r"\s+",
            " ",
            cleaned_text,
        ).strip()

        if not cleaned_text:
            return []

        sentences = re.split(
            r"(?<=[.!?])\s+|(?<=।)\s*",
            cleaned_text,
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _extract_reference_ordinals(
        sentence: str,
    ) -> list[int]:
        """Extract ordinal references such as 1st, 2nd and 4th."""

        ordinal_pattern = r"\b(\d+)(?:st|nd|rd|th)\b"

        matches = re.findall(
            ordinal_pattern,
            sentence,
            re.IGNORECASE,
        )

        ordinals = []

        for value in matches:
            number = int(value)

            if number not in ordinals:
                ordinals.append(number)

        return ordinals

    def _find_relationships_in_sentence(
        self,
        sentence: str,
        page_number: int,
    ) -> list[ExtractedRelationship]:
        """Find relationship phrases in one sentence."""

        extracted = []

        reference_ordinals = (
            self._extract_reference_ordinals(sentence)
        )

        for relationship_type, patterns in (
            self.RELATIONSHIP_PATTERNS.items()
        ):
            for pattern in patterns:
                match = re.search(
                    pattern,
                    sentence,
                    re.IGNORECASE,
                )

                if match:
                    extracted.append(
                        ExtractedRelationship(
                            relationship_type=relationship_type,
                            matched_phrase=match.group(0),
                            evidence_text=sentence,
                            page_number=page_number,
                            reference_ordinals=reference_ordinals,
                            confidence=ExtractionConfidence.MEDIUM,
                            extraction_method="RULE_BASED",
                            review_required=True,
                        )
                    )
                    break

        return extracted

    def extract_relationships(
        self,
        text: str,
    ) -> list[ExtractedRelationship]:
        """Extract relationship signals from Government Order text."""

        relationships = []

        for page_number, page_text in self._split_pages(text):
            sentences = self._get_evidence_sentences(page_text)

            for sentence in sentences:
                relationships.extend(
                    self._find_relationships_in_sentence(
                        sentence,
                        page_number,
                    )
                )

        unique_relationships = []
        seen = set()

        for relationship in relationships:
            key = (
                relationship.relationship_type,
                relationship.matched_phrase.lower(),
                relationship.evidence_text,
                relationship.page_number,
            )

            if key not in seen:
                seen.add(key)
                unique_relationships.append(relationship)

        return unique_relationships

    def extract_references(
        self,
        text: str,
    ) -> list[ExtractedReference]:
        """
        Extract numbered Government Order references
        from the Read section on the first page.
        """

        pages = self._split_pages(text)

        if not pages:
            return []

        page_number, first_page = pages[0]

        # Locate the Read section.
        read_match = re.search(
            r"\bRead\b([\s\S]*?)(?=\bGOVERNMENT\s+OF\s+KERALA\b)",
            first_page,
            re.IGNORECASE,
        )

        if not read_match:
            return []

        read_section = read_match.group(1)

        # Match numbered GO references, excluding letters.
        reference_pattern = re.compile(
            r"(?m)^\s*(\d+)\s+"
            r"(G\.?\s*O\.?\s*\(\s*(?:Rt|P|Ms)\s*\)"
            r"\s*No\.?\s*\d+\s*/\s*\d{4}\s*/\s*[A-Z&]+)"
            r"(?:\s+dated\s+(\d{1,2}[/-]\d{1,2}[/-]\d{4}))?",
            re.IGNORECASE,
        )

        references = []

        for match in reference_pattern.finditer(read_section):
            ordinal = int(match.group(1))

            raw_go_number = match.group(2)

            go_number = re.sub(
                r"\s+",
                " ",
                raw_go_number,
            ).strip()

            issue_date = match.group(3)

            references.append(
                ExtractedReference(
                    ordinal=ordinal,
                    go_number=go_number,
                    page_number=page_number,
                    evidence_text=match.group(0).strip(),
                    issue_date=issue_date,
                )
            )

        return references

    @staticmethod
    def resolve_referenced_go_numbers(
        relationships: list[ExtractedRelationship],
        references: list[ExtractedReference],
    ) -> list[ExtractedRelationship]:
        """Resolve relationship ordinals to actual GO numbers."""

        reference_map = {
            reference.ordinal: reference.go_number
            for reference in references
        }

        for relationship in relationships:
            relationship.referenced_go_numbers = [
                reference_map[ordinal]
                for ordinal in relationship.reference_ordinals
                if ordinal in reference_map
            ]

            if (
                relationship.reference_ordinals
                and len(relationship.referenced_go_numbers)
                != len(relationship.reference_ordinals)
            ):
                relationship.review_required = True

        return relationships
