# Copyright (C) 2023-2025 Cognizant Digital Business, Evolutionary AI.
# All Rights Reserved.
# Issued under the Academic Public License.
#
# You can be released from the terms, and requirements of the Academic Public
# License by purchasing a commercial license.
# Purchase of a commercial license is mandatory for any use of the
# neuro-san SDK Software in commercial settings.
#
# END COPYRIGHT
"""
UNFCCC Climate Document Parser.

Pure text processing for UNFCCC climate documents: splitting, metadata
extraction, reference detection, paragraph indexing, and episode construction.
No database or framework dependencies.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional


class DocumentParser:
    """Parses UNFCCC climate documents into structured episodes with metadata.

    Handles all text processing responsibilities: document splitting by
    decision/resolution headings, metadata extraction from filenames and
    document headers, cross-document reference detection, paragraph indexing,
    and episode body construction.

    This class contains no database or framework dependencies. It is designed
    to be inherited by ingestion tools that add pipeline and connectivity logic.
    """

    logger: logging.Logger
    config: Dict[str, Any]
    decision_registry: Dict[str, str]
    reference_map: Dict[str, List[Dict[str, Any]]]

    CONFERENCE_NAMES = {
        "CMA": (
            "Conference of the Parties serving as the meeting of the Parties "
            "to the Paris Agreement"
        ),
        "CMP": (
            "Conference of the Parties serving as the meeting of the Parties "
            "to the Kyoto Protocol"
        ),
        "COP": "Conference of the Parties",
        "SBI": "Subsidiary Body for Implementation",
        "SBSTA": "Subsidiary Body for Scientific and Technological Advice",
    }

    FRONT_MATTER_MARKERS = [
        "Contents",
        "Decisions adopted by the Conference",
        "Conference of the Parties serving as the meeting",
        "Addendum",
        "Part two:",
    ]

    FALLBACK_CHARS = 6000
    CHUNK_OVERLAP_CHARS = 200
    INCLUDE_HEADING_IN_BODY = True
    MAX_EPISODE_NAME_LEN = 250
    MIN_SECTION_LEN = 40
    PROCESS_SUBDIRECTORIES = True

    YEAR_RE = re.compile(r"(20\d{2})")
    CONFERENCE_TYPE_RE = re.compile(
        r"(?P<type>CMA|CMP|COP|SBI|SBSTA)(?P<year>\d{4})_(?P<session>\d+)",
        re.IGNORECASE,
    )
    DECISION_RESOLUTION_HEADING_RE = re.compile(
        r"(?m)^(?P<h>(?:Decision|Resolution)\s+(?:No\.?\s*)?[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+|"
        r"(?:\s*/\s*[A-Za-z]+(?:\.\d+)?))?[^\n]*)$"
    )
    OTHER_HEADING_RE = re.compile(
        r"(?mi)^(?P<h>(Chapter|Section|Agenda item|Article|Part)\s+[^\n]+)$",
        re.IGNORECASE,
    )
    ANNEX_HEADING_RE = re.compile(
        r"(?mi)^(?P<h>Annex(?:\s+[IVXLC]+|\s+\d+)?[^\n]*)$", re.IGNORECASE
    )
    LOCATION_DATE_RE = re.compile(
        r"held in (?P<location>[^,\n]+?)(?:\s+from\s+(?P<date>[^\n]+?))?(?:\n|$)",
        re.IGNORECASE,
    )
    FCCC_DOC_RE = re.compile(r"(?P<fccc>FCCC/[A-Z/]+/\d{4}/[\d/A-Za-z\.]+)")

    DECISION_REFERENCE_RE = re.compile(
        r"(?i)(?:decision|resolution)s?\s+(?:No\.?\s*)?"
        r"(?P<number>[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+))"
        r"(?:,\s*paragraph(?:s)?\s+(?P<paragraphs>[\d\-,\s]+))?"
    )
    ARTICLE_REFERENCE_RE = re.compile(
        r"(?i)Article\s+(?P<number>[\dIVXLC]+)"
        r"(?:,\s*paragraph(?:s)?\s+(?P<paragraphs>[\d\-,\s]+))?"
        r"(?:\s+of\s+the\s+(?P<agreement>[^\n,;\.]+))?"
    )
    PARAGRAPH_REFERENCE_RE = re.compile(
        r"(?i)paragraphs?\s+(?P<paragraphs>[\d\-–,\s]+)"
        r"(?:\s+of\s+(?:decision|resolution)\s+(?P<number>[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+)))?"
    )
    ANNEX_REFERENCE_RE = re.compile(
        r"(?i)(?:the\s+)?annex(?:\s+(?P<annex_id>[IVXLC]+|\d+))?"
        r"(?:\s+to\s+(?:decision|resolution)\s+(?P<decision_number>[\dIVXLC]+"
        r"(?:\s*/\s*[A-Za-z]+\.\d+)))?"
    )

    def build_episodes(self, dir_path: Path) -> List[Dict[str, Any]]:
        """Converts source documents into structured episodes with metadata and references.

        Processes all text files in the specified directory, splitting them into
        decision/resolution sections, extracting metadata, identifying cross-document
        references, and building paragraph indices for granular access.

        Args:
            dir_path: Path to directory containing UNFCCC document text files.

        Returns:
            List of episode dictionaries, each containing:
                - name: Unique episode identifier
                - episode_body: Formatted text with metadata headers
                - source_description: Human-readable source description
                - reference_time: Datetime for temporal indexing
                - metadata: Extracted metadata (conference type, year, location, etc.)
                - decision_id: Decision/resolution identifier if applicable
                - annex_id: Annex identifier if applicable
                - references: List of cross-document references found
                - document_name: Source filename stem

        Raises:
            FileNotFoundError: If dir_path does not exist.
            NotADirectoryError: If dir_path is not a directory.
        """
        if not dir_path.exists():
            raise FileNotFoundError(f"DATA_DIR does not exist: {dir_path}")
        if not dir_path.is_dir():
            raise NotADirectoryError(f"DATA_DIR is not a directory: {dir_path}")

        documents = self._collect_documents(dir_path)
        episodes: List[Dict[str, Any]] = []

        if not documents:
            self.logger.warning("No .txt files found in %s", dir_path)
            return episodes

        self.logger.info("Found %d document files to process", len(documents))

        for path in documents:
            text = self._read_text(path)
            if not text:
                continue

            year = self._extract_year_from_filename(path)
            reference_time = datetime(year, 1, 1, tzinfo=timezone.utc)

            file_metadata = self._extract_metadata_from_filename(path)
            doc_metadata = self._extract_document_metadata(text)
            combined_metadata = {**file_metadata, **doc_metadata}

            sections = self._split_document(text)
            if not sections:
                self.logger.warning("No sections extracted from %s", path.name)
                continue

            self.logger.info("Processing %s: %d sections", path.name, len(sections))
            source_description = self._build_source_description(
                path, file_metadata, year
            )

            for idx, section in enumerate(sections, 1):
                decision_id = self._extract_decision_id(section.get("title", ""))
                annex_id = self._extract_annex_id(section.get("title", ""))

                base_episode_name = (
                    f"{path.stem}::{section.get('title') or f'Part {idx}'}"
                )
                body = self._build_episode_body(section["body"], combined_metadata)

                references = self._extract_references(section["body"], decision_id)
                paragraph_index = self._extract_paragraph_index(section["body"])

                enriched_metadata = combined_metadata.copy()
                if decision_id:
                    enriched_metadata["decision_id"] = decision_id
                    self.decision_registry[decision_id] = base_episode_name
                    self._post_extract_decision_metadata(
                        enriched_metadata, section["body"]
                    )

                if annex_id:
                    enriched_metadata["annex_id"] = annex_id

                # Chunk oversized sections into ~FALLBACK_CHARS pieces
                chunk_list = self._chunk_section(body)

                for chunk_info in chunk_list:
                    chunk_idx = chunk_info["chunk_index"]
                    total_chunks = chunk_info["total_chunks"]

                    if total_chunks > 1:
                        episode_name = (
                            f"{base_episode_name} "
                            f"[{chunk_idx + 1}/{total_chunks}]"
                        )
                    else:
                        episode_name = base_episode_name

                    chunk_metadata = enriched_metadata.copy()
                    if total_chunks > 1:
                        chunk_metadata["chunk_index"] = chunk_idx
                        chunk_metadata["total_chunks"] = total_chunks

                    # Paragraph index and references only on first chunk
                    is_first_chunk = chunk_idx == 0
                    if is_first_chunk and references:
                        chunk_metadata["references"] = references
                        self.reference_map[base_episode_name] = references
                    if is_first_chunk and paragraph_index:
                        chunk_metadata["paragraph_index"] = paragraph_index
                        if self.config["verbose_logging"]:
                            self.logger.debug(
                                "Extracted %d paragraphs from %s",
                                len(paragraph_index),
                                base_episode_name[:50],
                            )

                    episodes.append(
                        {
                            "name": episode_name[: self.MAX_EPISODE_NAME_LEN],
                            "episode_body": chunk_info["body"],
                            "source_description": source_description,
                            "reference_time": reference_time,
                            "metadata": chunk_metadata,
                            "decision_id": decision_id,
                            "annex_id": annex_id,
                            "references": references if is_first_chunk else [],
                            "document_name": path.stem,
                        }
                    )
                    if self.config["verbose_logging"]:
                        chunk_label = (
                            f" [{chunk_idx + 1}/{total_chunks}]"
                            if total_chunks > 1
                            else ""
                        )
                        self.logger.info(
                            "Prepared episode: %s%s",
                            base_episode_name[:100],
                            chunk_label,
                        )
                        if decision_id:
                            self.logger.info("  Decision ID: %s", decision_id)
                        if annex_id:
                            self.logger.info("  Annex ID: %s", annex_id)
                        if is_first_chunk and references:
                            self.logger.info(
                                "  Found %d references", len(references)
                            )

        return episodes

    def _post_extract_decision_metadata(
        self, enriched_metadata: Dict[str, Any], section_body: str
    ) -> None:
        """Hook for subclasses to enrich metadata after decision ID extraction.

        Called during build_episodes when a decision_id is found. Subclasses can
        override to add additional metadata such as decision action classification.

        Args:
            enriched_metadata: Mutable metadata dictionary to enrich.
            section_body: Full text of the decision section.
        """
        del enriched_metadata, section_body

    def _extract_decision_id(self, title: str) -> Optional[str]:
        """Extracts normalized decision or resolution identifier from section title.

        Parses decision/resolution references (e.g., 'Decision 1/CP.21', 'Resolution 5/CMA.3')
        and returns a normalized identifier. Excludes annex sections which are handled separately.

        Args:
            title: Section heading text to parse.

        Returns:
            Normalized decision/resolution ID (e.g., '1/CP.21'), or None if not a
            decision/resolution.
        """
        if not title:
            return None

        if re.match(r"(?i)^\s*Annex", title):
            return None

        match = re.search(
            r"(?i)(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+))",
            title,
        )
        if match:
            return match.group(1).strip()
        return None

    def _extract_annex_id(self, title: str) -> Optional[str]:
        """Extracts annex identifier from section title.

        Detects annex sections and extracts their Roman numeral or Arabic number identifiers.
        Unlabeled annexes are marked as 'unnumbered'.

        Args:
            title: Section heading text to parse.

        Returns:
            Annex identifier ('I', 'II', '1', '2', 'unnumbered'), or None if not an annex.
        """
        if not title:
            return None

        match = re.match(r"(?i)^\s*Annex(?:\s+([IVXLC]+|\d+))?", title)
        if match:
            annex_id = match.group(1)
            return annex_id.strip() if annex_id else "unnumbered"
        return None

    def _extract_references(
        self, text: str, source_decision_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extracts all cross-document references from text.

        Identifies and extracts references to other decisions, resolutions, articles,
        paragraphs, and annexes using regex patterns. Captures context around each
        reference for semantic understanding.

        Args:
            text: Text to search for references.
            source_decision_id: Optional decision ID of the source document for tracking.

        Returns:
            List of reference dictionaries, each containing:
                - type: Reference type ('decision', 'article', 'paragraph', 'annex')
                - target: Referenced decision/article identifier
                - paragraphs: Referenced paragraph numbers if specified
                - raw_text: Original matched text
                - context: Surrounding text context
                - source_decision: Source decision ID if provided
        """
        references: List[Dict[str, Any]] = []

        for match in self.DECISION_REFERENCE_RE.finditer(text):
            ref = {
                "type": "decision",
                "target": match.group("number").strip(),
                "paragraphs": (
                    match.group("paragraphs").strip()
                    if match.group("paragraphs")
                    else None
                ),
                "raw_text": match.group(0),
                "context": self._extract_context(text, match.start(), match.end()),
            }
            references.append(ref)

        for match in self.ARTICLE_REFERENCE_RE.finditer(text):
            ref = {
                "type": "article",
                "target": match.group("number").strip(),
                "paragraphs": (
                    match.group("paragraphs").strip()
                    if match.group("paragraphs")
                    else None
                ),
                "agreement": (
                    match.group("agreement").strip()
                    if match.group("agreement")
                    else None
                ),
                "raw_text": match.group(0),
                "context": self._extract_context(text, match.start(), match.end()),
            }
            references.append(ref)

        for match in self.ANNEX_REFERENCE_RE.finditer(text):
            annex_id = match.group("annex_id")
            decision_number = match.group("decision_number")

            ref = {
                "type": "annex",
                "annex_id": annex_id.strip() if annex_id else None,
                "target": (decision_number.strip() if decision_number else None),
                "raw_text": match.group(0),
                "context": self._extract_context(text, match.start(), match.end()),
            }
            references.append(ref)

        for match in self.PARAGRAPH_REFERENCE_RE.finditer(text):
            if match.group("number"):
                continue

            ref = {
                "type": "paragraph",
                "paragraphs": match.group("paragraphs").strip(),
                "raw_text": match.group(0),
                "context": self._extract_context(text, match.start(), match.end()),
            }
            references.append(ref)

        for ref in references:
            ref["source_decision"] = source_decision_id

        return references

    def _extract_context(
        self, text: str, start: int, end: int, window: int = 100
    ) -> str:
        """Extracts surrounding context for a reference match.

        Captures text before and after the matched reference to provide semantic context,
        useful for understanding how the reference is being used.

        Args:
            text: Full text containing the reference.
            start: Start index of the reference match.
            end: End index of the reference match.
            window: Number of characters to include before and after (default: 100).

        Returns:
            Context string with ellipsis markers if truncated.
        """
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end]
        context = " ".join(context.split())
        if context_start > 0:
            context = "..." + context
        if context_end < len(text):
            context = context + "..."
        return context

    def _extract_paragraph_index(self, text: str) -> Dict[str, str]:
        """Extracts numbered paragraphs for granular citation resolution.

        Builds an index mapping paragraph numbers to their text content, enabling
        direct access to specific paragraphs (e.g., 'paragraph 69 of decision 1/CP.21')
        without searching the full document.

        Args:
            text: Decision or resolution text containing numbered paragraphs.

        Returns:
            Dictionary mapping paragraph numbers (strings) to paragraph text (truncated
            to 1000 chars).
        """
        paragraphs: Dict[str, str] = {}

        numbered_pattern = r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s|\Z)"
        matches = re.finditer(numbered_pattern, text, re.MULTILINE | re.DOTALL)

        for match in matches:
            para_num = match.group(1).strip()
            para_text = match.group(2).strip()
            para_text = " ".join(para_text.split())
            paragraphs[para_num] = para_text[:1000]

        if not paragraphs:
            roman_pattern = r"^\s*([IVXLC]+)\.\s+(.+?)(?=^\s*[IVXLC]+\.\s|\Z)"
            matches = re.finditer(roman_pattern, text, re.MULTILINE | re.DOTALL)

            for match in matches:
                para_num = match.group(1).strip()
                para_text = match.group(2).strip()
                para_text = " ".join(para_text.split())
                paragraphs[para_num] = para_text[:1000]

        return paragraphs

    def _collect_documents(self, base_dir: Path) -> List[Path]:
        """Collects all text document files from the specified directory.

        Recursively searches for .txt files if PROCESS_SUBDIRECTORIES is enabled,
        otherwise searches only the top-level directory. Excludes hidden files
        (those starting with '.').

        Args:
            base_dir: Directory path to search for documents.

        Returns:
            Sorted list of Path objects for all discovered text files.
        """
        if self.PROCESS_SUBDIRECTORIES:
            txt_files = sorted(base_dir.rglob("*.txt"))
        else:
            txt_files = sorted(base_dir.glob("*.txt"))
        return [path for path in txt_files if not path.name.startswith(".")]

    def _read_text(self, path: Path) -> Optional[str]:
        """Reads text content from a file with error handling.

        Uses UTF-8 encoding and ignores decoding errors to handle potentially
        malformed text files gracefully.

        Args:
            path: Path to the text file to read.

        Returns:
            File contents as string, or None if reading fails.
        """
        try:
            return path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            self.logger.error("Error reading %s: %s", path, exc)
            return None

    def _extract_year_from_filename(self, path: Path) -> int:
        """Extracts the year from a filename for temporal indexing.

        Searches for a 4-digit year pattern (2000-2025) in the filename.
        Falls back to current year if no valid year is found.

        Args:
            path: Path to the document file.

        Returns:
            Extracted year as integer, or current year as fallback.
        """
        match = self.YEAR_RE.search(path.stem)
        if match:
            year = int(match.group(1))
            if 2000 <= year <= 2030:
                return year
        return datetime.now().year

    def _extract_metadata_from_filename(self, path: Path) -> Dict[str, str]:
        """Extracts conference metadata from standardized filename patterns.

        Parses filenames like 'COP2015_21_Decisions.txt' to extract conference type,
        year, and session number. Maps conference types to full names.

        Args:
            path: Path to the document file.

        Returns:
            Dictionary with extracted metadata (conference_type, year, session_number,
            conference_name), or empty dict if pattern doesn't match.
        """
        metadata: Dict[str, str] = {}
        match = self.CONFERENCE_TYPE_RE.search(path.stem)
        if match:
            conf_type = match.group("type").upper()
            metadata["conference_type"] = conf_type
            metadata["year"] = match.group("year")
            metadata["session_number"] = match.group("session")
            metadata["conference_name"] = self.CONFERENCE_NAMES.get(
                conf_type, conf_type
            )
        return metadata

    def _extract_document_metadata(self, text: str) -> Dict[str, str]:
        """Extracts metadata from document header content.

        Searches the document header for location, date, and FCCC document reference
        information using regex patterns.

        Args:
            text: Full document text (header portion is searched).

        Returns:
            Dictionary with extracted metadata (location, date, fccc_reference).
        """
        header = text[:2000]
        metadata: Dict[str, str] = {}

        location_match = self.LOCATION_DATE_RE.search(header)
        if location_match:
            metadata["location"] = location_match.group("location").strip()
            if location_match.group("date"):
                metadata["date"] = location_match.group("date").strip()

        fccc_match = self.FCCC_DOC_RE.search(header)
        if fccc_match:
            metadata["fccc_reference"] = fccc_match.group("fccc")
        return metadata

    @staticmethod
    def _normalize_pipe_table_lines(text: str) -> str:
        """Strip pipe-table cell delimiters from single-column pipe-table rows.

        Some UNFCCC PDF-to-text conversions wrap entire paragraphs and headings
        inside pipe-table cells, for example::

            | Decision 3/CMA.1 |
            | 4.  Decides, having considered the draft decisions... |

        These lines are indistinguishable from plain text by the heading regex
        because they start with ``|`` instead of ``Decision``.  Stripping the
        leading and trailing ``|`` normalises them so the heading regex can
        recognise ``Decision 3/CMA.1`` as a section heading.

        Lines that contain inner pipes (multi-column tables such as
        ``| col1 | col2 | col3 |``) are also normalised: the outer delimiters
        are removed and the inner content is preserved as-is.  This is
        acceptable for RAG text extraction purposes.

        Args:
            text: Raw document text (line endings already normalised to ``\\n``).

        Returns:
            Text with outer pipe delimiters stripped from pipe-table rows.
        """
        lines = text.split("\n")
        result = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|") and len(stripped) > 2:
                result.append(stripped[1:-1].strip())
            else:
                result.append(line)
        return "\n".join(result)

    def _split_document(self, raw_text: str) -> List[Dict[str, str]]:
        """Splits a document into sections based on decision/resolution headings.

        Attempts to split by decision/resolution headings first, then other heading
        types. Separates annexes into distinct sections linked to their parent decisions.
        Falls back to length-based splitting if no headings are found.

        Args:
            raw_text: Raw document text with potential line ending variations.

        Returns:
            List of section dictionaries with 'title' and 'body' keys.
        """
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._normalize_pipe_table_lines(text)
        text = self._strip_front_matter(text)

        for regex in (self.DECISION_RESOLUTION_HEADING_RE, self.OTHER_HEADING_RE):
            sections = self._split_by_headings(text, regex)
            if sections:
                final_sections = []
                for section in sections:
                    annex_matches = list(
                        self.ANNEX_HEADING_RE.finditer(section["body"])
                    )
                    if annex_matches:
                        first_annex_pos = annex_matches[0].start()

                        if first_annex_pos > self.MIN_SECTION_LEN:
                            decision_part = section["body"][:first_annex_pos].strip()
                            final_sections.append(
                                {"title": section["title"], "body": decision_part}
                            )

                        annex_text = section["body"][first_annex_pos:]
                        annex_subsections = self._split_by_headings(
                            annex_text, self.ANNEX_HEADING_RE
                        )
                        if annex_subsections:
                            for annex_subsection in annex_subsections:
                                annex_subsection["title"] = (
                                    f"{annex_subsection['title']} to {section['title']}"
                                )
                            final_sections.extend(annex_subsections)
                    else:
                        final_sections.append(section)
                return final_sections
        return self._split_by_length(text)

    def _strip_front_matter(self, text: str) -> str:
        """Removes front matter (table of contents, headers) from document text.

        Identifies the first substantive heading (decision/resolution) or known
        front matter markers and returns text starting from that point.

        Args:
            text: Full document text including front matter.

        Returns:
            Document text with front matter removed.
        """
        first_idx = len(text)
        for regex in (self.DECISION_RESOLUTION_HEADING_RE, self.OTHER_HEADING_RE):
            match = regex.search(text)
            if match:
                first_idx = min(first_idx, match.start())
        if first_idx < len(text):
            return text[first_idx:]
        for marker in self.FRONT_MATTER_MARKERS:
            position = text.find(marker)
            if 0 <= position < len(text):
                return text[position:]
        return text

    def _split_by_headings(
        self, text: str, heading_re: re.Pattern
    ) -> List[Dict[str, str]]:
        """Splits text into sections based on a heading pattern.

        Finds all matches of the heading pattern and extracts the text between
        consecutive headings as section bodies. Filters out sections that are
        too short (below MIN_SECTION_LEN).

        Args:
            text: Text to split into sections.
            heading_re: Compiled regex pattern for identifying headings.

        Returns:
            List of section dictionaries with 'title' and 'body' keys, or empty
            list if no headings match.
        """
        matches = list(heading_re.finditer(text))
        if not matches:
            return []

        sections: List[Dict[str, str]] = []
        starts = [match.start() for match in matches] + [len(text)]
        titles = [match.group("h").strip() for match in matches]

        for idx, title in enumerate(titles):
            start = matches[idx].start()
            end = starts[idx + 1]
            block = text[start:end].strip()
            if not block:
                continue

            lines = block.splitlines()
            first_line = lines[0].strip() if lines else ""
            body = (
                "\n".join(lines[1:]).strip()
                if first_line == title and len(lines) > 1
                else block
            )
            if len(body) < self.MIN_SECTION_LEN:
                continue

            if self.INCLUDE_HEADING_IN_BODY:
                body = f"{title}\n\n{body}".strip()
            sections.append({"title": title, "body": body})
        return sections

    def _split_by_length(self, text: str) -> List[Dict[str, str]]:
        """Splits text into fixed-length chunks as a fallback strategy.

        Used when no recognizable headings are found. Creates sections of
        FALLBACK_CHARS length, filtering out sections that are too short.

        Args:
            text: Text to split into chunks.

        Returns:
            List of section dictionaries with auto-generated 'title' (e.g., 'Part 1')
            and 'body' keys.
        """
        sections: List[Dict[str, str]] = []
        for idx in range(0, len(text), self.FALLBACK_CHARS):
            chunk = text[idx : idx + self.FALLBACK_CHARS].strip()  # noqa: E203
            if len(chunk) < self.MIN_SECTION_LEN:
                continue
            sections.append(
                {"title": f"Part {idx // self.FALLBACK_CHARS + 1}", "body": chunk}
            )
        return sections

    def _chunk_section(self, body: str) -> List[Dict[str, Any]]:
        """Split an episode body into ~FALLBACK_CHARS chunks with overlap.

        Attempts to split at paragraph boundaries (lines starting with a digit
        followed by a period, e.g. ``1. ``, ``42. ``).  Falls back to splitting
        at the nearest newline before the character limit, then to a hard
        character split.

        Args:
            body: Full episode body text (including metadata headers).

        Returns:
            List of dicts with ``body`` (chunk text), ``chunk_index`` (0-based),
            and ``total_chunks``.  If body <= FALLBACK_CHARS a single-element
            list is returned (no chunking).
        """
        if len(body) <= self.FALLBACK_CHARS:
            return [{"body": body, "chunk_index": 0, "total_chunks": 1}]

        chunks: List[str] = []
        start = 0
        while start < len(body):
            end = start + self.FALLBACK_CHARS
            if end >= len(body):
                chunks.append(body[start:])
                break

            search_region = body[start:end]

            # Try to break at last paragraph boundary in the region
            para_boundary = None
            for m in re.finditer(r"\n(?=\d+\.\s)", search_region):
                if m.start() > self.FALLBACK_CHARS // 2:
                    para_boundary = m.start()

            if para_boundary is not None:
                split_at = start + para_boundary
            else:
                # Fallback: split at last newline in second half of region
                last_newline = search_region.rfind("\n", self.FALLBACK_CHARS // 2)
                split_at = (start + last_newline) if last_newline > 0 else end

            chunks.append(body[start:split_at])
            # Next chunk starts with CHUNK_OVERLAP_CHARS overlap
            overlap_start = max(start, split_at - self.CHUNK_OVERLAP_CHARS)
            start = overlap_start if overlap_start < split_at else split_at

        total = len(chunks)
        return [
            {"body": chunk, "chunk_index": idx, "total_chunks": total}
            for idx, chunk in enumerate(chunks)
        ]

    def _build_source_description(
        self, path: Path, metadata: Dict[str, str], year: int
    ) -> str:
        """Builds a human-readable source description for episodes.

        Creates a formatted description like 'Conference of the Parties Session 21 (2015)'
        if metadata is available, otherwise uses the filename.

        Args:
            path: Path to source document file.
            metadata: Extracted metadata dictionary.
            year: Document year.

        Returns:
            Formatted source description string.
        """
        if metadata.get("conference_name"):
            return (
                f"{metadata['conference_name']} "
                f"Session {metadata.get('session_number', 'N/A')} "
                f"({metadata.get('year', year)})"
            )
        return path.stem

    def _build_episode_body(self, section_body: str, metadata: Dict[str, str]) -> str:
        """Builds episode body text with metadata headers.

        Prepends conference, session, year, location, and document reference
        information to the section body for context. Subclasses can add additional
        headers via _extra_header_lines().

        Args:
            section_body: The main text content of the section.
            metadata: Metadata dictionary with conference information.

        Returns:
            Formatted episode body with metadata headers and section content.
        """
        header_lines: List[str] = []
        if metadata.get("conference_name"):
            header_lines.append(f"Conference: {metadata['conference_name']}")
        if metadata.get("session_number"):
            header_lines.append(f"Session: {metadata['session_number']}")
        if metadata.get("year"):
            header_lines.append(f"Year: {metadata['year']}")
        if metadata.get("location"):
            header_lines.append(f"Location: {metadata['location']}")
        if metadata.get("fccc_reference"):
            header_lines.append(f"Document: {metadata['fccc_reference']}")
        header_lines.extend(self._extra_header_lines(metadata))

        if not header_lines:
            return section_body
        return "\n".join(header_lines) + "\n\n" + section_body

    def _extra_header_lines(self, metadata: Dict[str, str]) -> List[str]:
        """Hook for subclasses to add additional header lines to episode body.

        Called by _build_episode_body after the standard headers. Override in
        subclasses to append database-specific metadata headers.

        Args:
            metadata: Metadata dictionary with conference information.

        Returns:
            List of additional header line strings (empty by default).
        """
        del metadata
        return []
