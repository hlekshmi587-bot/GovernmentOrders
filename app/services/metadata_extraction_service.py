
"""
Metadata extraction service for Government Orders.
Supports English and Malayalam GO numbers and department names.
"""

import re
from datetime import datetime


class MetadataService:
    """Extract basic metadata from English and Malayalam GOs."""

    DEPARTMENT_CODES = {
        "P&ARD": "Personnel and Administrative Reforms Department",
        "ഉഭപവ": "Personnel and Administrative Reforms Department",
        "ഉ.ഭ.പ.വ": "Personnel and Administrative Reforms Department",
        "LSGD": "Local Self Government Department",
        "SJD": "Social Justice Department",
        "DMD": "Disaster Management Department",
        "FIN": "Finance Department",
        "ITD": "Electronics and Information Technology Department",
        "TRANS": "Transport Department",
    }

    @staticmethod
    def _get_first_page(text: str) -> str:
        """Return the content of the first page."""
        return re.split(r"--- Page 2 ---", text, maxsplit=1)[0]

    @staticmethod
    def _get_text_sources(first_page: str) -> tuple[str, str]:
        """Separate embedded text and OCR text when both exist."""
        embedded_marker = "--- Embedded Text ---"
        ocr_marker = "--- OCR Text ---"

        if embedded_marker in first_page and ocr_marker in first_page:
            sections = first_page.split(embedded_marker, 1)[1]
            embedded_text, ocr_text = sections.split(ocr_marker, 1)
            return embedded_text.strip(), ocr_text.strip()

        return first_page.strip(), ""

    @staticmethod
    def _get_lines(text: str) -> list[str]:
        """Convert text into cleaned non-empty lines."""
        return [
            re.sub(r"\s+", " ", line).strip()
            for line in text.splitlines()
            if line.strip()
        ]

    @staticmethod
    def _normalise_date(date_text: str) -> str:
        """Convert supported dates into DD-MM-YYYY."""
        date_text = re.sub(r"\s+", " ", date_text.strip())

        numeric_match = re.search(
            r"(?<![\w/])"
            r"(\d{1,2})\s*[./-]\s*"
            r"(\d{1,2})\s*[./-]\s*(\d{4})"
            r"(?![\w/])",
            date_text,
        )

        if numeric_match:
            day, month, year = numeric_match.groups()

            try:
                parsed_date = datetime(
                    int(year), int(month), int(day)
                )
                return parsed_date.strftime("%d-%m-%Y")
            except ValueError:
                return "Unknown"

        written_match = re.search(
            r"(\d{1,2})\s*(?:st|nd|rd|th)?\s+"
            r"(January|February|March|April|May|June|"
            r"July|August|September|October|November|December)"
            r",?\s+(\d{4})",
            date_text,
            re.IGNORECASE,
        )

        if written_match:
            day, month, year = written_match.groups()

            try:
                parsed_date = datetime.strptime(
                    f"{day} {month} {year}",
                    "%d %B %Y",
                )
                return parsed_date.strftime("%d-%m-%Y")
            except ValueError:
                return "Unknown"

        return "Unknown"

    @staticmethod
    def _normalise_go_number(value: str) -> str:
        """Clean spacing and punctuation in a GO number."""
        value = re.sub(r"\s*/\s*", "/", value)
        value = re.sub(r"\s+", " ", value)
        return value.strip(" ,:;.")

    def _extract_go_number(self, lines: list[str]) -> str:
        """Extract complete English or Malayalam GO numbers."""

        patterns = [
            (
                r"G\s*\.?\s*O\s*\.?\s*"
                r"\(\s*(?:P|Rt|Ms)\s*\)\s*"
                r"No\s*\.?\s*"
                r"\d+\s*/\s*\d{4}\s*/\s*"
                r"[A-Za-z0-9&.-]+"
            ),
            (
                r"സ\s*\.?\s*ഉ\s*\.?\s*"
                r"\([^)]*\)\s*"
                r"(?:നം|നമ്പർ)\s*\.?\s*"
                r"\d+\s*/\s*\d{4}\s*/\s*"
                r"[\w&.-]+"
            ),
        ]

        for line in lines[:30]:
            for pattern in patterns:
                match = re.search(
                    pattern,
                    line,
                    re.IGNORECASE,
                )

                if match:
                    candidate = self._normalise_go_number(
                        match.group(0)
                    )

                    if re.search(r"\d+/\d{4}/", candidate):
                        return candidate

        return "Unknown"

    def _extract_date(self, lines: list[str]) -> str:
        """Extract the issue date associated with the GO heading."""

        go_heading_index = None

        for index, line in enumerate(lines[:30]):
            if (
                re.search(r"G\s*\.?\s*O|സ\s*\.?\s*ഉ", line, re.IGNORECASE)
                and re.search(r"\d", line)
            ):
                go_heading_index = index
                break

        if go_heading_index is None:
            return "Unknown"

        heading_context = " ".join(
            lines[go_heading_index:go_heading_index + 3]
        )

        date = self._normalise_date(heading_context)

        if date != "Unknown":
            return date

        previous_context = " ".join(
            lines[max(0, go_heading_index - 2):go_heading_index]
        )

        return self._normalise_date(previous_context)

    def _extract_department(self, lines: list[str]) -> str:
        """Extract a department name from the document heading."""

        department_patterns = [
            r"DISASTER MANAGEMENT(?:\s*\([^)]+\))?\s*DEPARTMENT",
            r"PERSONNEL AND ADMINISTRATIVE REFORMS"
            r"(?:\s*\([^)]+\))?\s*DEPARTMENT",
            r"FINANCE(?:\s*\([^)]+\))?\s*DEPARTMENT",
            r"Electronics\s*(?:&|AND)\s*Information"
            r".{0,60}Department",
            r"ഗതാഗത\s*\([^)]*\)\s*വകുപ്പ്",
            r"ഉദ്യോഗസ്ഥ.{0,60}വകുപ്പ്",
        ]

        for line in lines[:30]:
            candidate = line.strip()

            if not candidate or len(candidate) > 150:
                continue

            if re.search(
                r"Read|Letter|Reference|പരാമർശം|"
                r"സംഗ്രഹം|Abstract|ORDER",
                candidate,
                re.IGNORECASE,
            ):
                continue

            for pattern in department_patterns:
                match = re.search(
                    pattern,
                    candidate,
                    re.IGNORECASE,
                )

                if match:
                    return match.group(0).strip()

        return "Unknown"

    def _extract_department_from_go_number(
        self,
        go_number: str,
    ) -> str:
        """Use the department code in the GO number as a fallback."""

        if go_number == "Unknown":
            return "Unknown"

        parts = go_number.split("/")

        if len(parts) < 3:
            return "Unknown"

        code = parts[-1].strip()
        normalised_code = re.sub(r"[\s.]", "", code).upper()

        for department_code, department_name in self.DEPARTMENT_CODES.items():
            key = re.sub(
                r"[\s.]",
                "",
                department_code,
            ).upper()

            if normalised_code == key:
                return department_name

        return "Unknown"

    @staticmethod
    def _filename_go_candidate(filename: str) -> str:
        """Extract a possible GO number from the filename."""

        patterns = [
            (
                r"G\.?\s*O\.?\s*\(\s*(P|Rt|Ms)\s*\)"
                r"\s*No\.?\s*([0-9]+)"
                r"\s*[/_-]\s*([0-9]{4})"
            ),
            (
                r"GOPNo\s*([0-9]+)"
                r"[-_]\s*([0-9]{4})"
            ),
        ]

        for pattern in patterns:
            match = re.search(pattern, filename, re.IGNORECASE)

            if match:
                if len(match.groups()) == 3:
                    order_type, number, year = match.groups()
                    return f"G.O.({order_type})No.{number}/{year}"

                number, year = match.groups()
                return f"G.O.(P)No.{number}/{year}"

        return "Unknown"

    @staticmethod
    def _is_complete_go_number(go_number: str) -> bool:
        """Check whether a GO number contains number, year and suffix."""

        if go_number == "Unknown":
            return False

        return bool(
            re.search(
                r"\d+/\d{4}/[\w&.-]+$",
                go_number,
            )
        )

    def extract_metadata(
        self,
        text: str,
        filename: str,
    ) -> dict[str, str]:
        """Extract metadata from embedded text and OCR text."""

        first_page = self._get_first_page(text)

        embedded_text, ocr_text = self._get_text_sources(
            first_page
        )

        embedded_lines = self._get_lines(embedded_text)
        ocr_lines = self._get_lines(ocr_text)

        # Extract GO number from embedded text, OCR, or filename.
        go_number = self._extract_go_number(embedded_lines)

        if go_number == "Unknown":
            go_number = self._extract_go_number(ocr_lines)

        if go_number == "Unknown":
            go_number = self._filename_go_candidate(filename)

        # Extract the issue date.
        date = self._extract_date(embedded_lines)

        if date == "Unknown":
            date = self._extract_date(ocr_lines)

        # Extract department from heading first.
        department = self._extract_department(ocr_lines)

        if department == "Unknown":
            department = self._extract_department(embedded_lines)

        # Use the GO number's department code as a fallback.
        if department == "Unknown":
            department = self._extract_department_from_go_number(
                go_number
            )

        metadata = {
            "filename": filename,
            "go_number": go_number,
            "date": date,
            "department": department,
            "verification_status": "Needs Review",
        }

        if (
            self._is_complete_go_number(go_number)
            and date != "Unknown"
            and department != "Unknown"
        ):
            metadata["verification_status"] = "Extracted - Verify"

        return metadata
