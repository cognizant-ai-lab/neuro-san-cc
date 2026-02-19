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
# pylint: disable=too-many-lines,wrong-import-position
"""
UNFCCC Climate Document Ingestion for Neo4j Knowledge Graph.

This module ingests United Nations Framework Convention on Climate Change (UNFCCC)
documents into a Neo4j-backed knowledge graph using the Graphiti framework.

Key Features:
    - Parses UNFCCC COP, CMA, CMP, SBI, and SBSTA decision documents
    - Extracts structured information: decisions, resolutions, annexes, and paragraphs
    - Identifies and extracts cross-document references between decisions
    - Supports incremental processing with checkpoint-based resume functionality
    - Includes performance optimizations for large-scale graph operations

Document Structure:
    - Splits documents by decisions/resolutions as primary sections
    - Separates annexes into distinct episodes
    - Extracts numbered paragraphs for granular reference access
    - Preserves metadata: conference type, year, session, location, FCCC references

Reference Extraction:
    - Decision references (e.g., "decision 1/CP.21, paragraph 69")
    - Article references (e.g., "Article 6 of the Paris Agreement")
    - Annex references (e.g., "Annex I to decision 4/CMA.1")
    - Paragraph references with optional decision context

Note:
    Reference linking, annex linking, and conference linking features have been
    disabled to reduce episode count and improve processing performance. References
    are still extracted and stored in episode metadata for potential future use.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from datetime import timezone
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from neuro_san.interfaces.coded_tool import CodedTool

from dotenv import load_dotenv

# Load environment variables before importing graphiti modules.
# The graphiti_core.embedder.client reads EMBEDDING_DIM at import time.
_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


class Neo4jIngestionEnhanced(CodedTool):  # pylint: disable=too-many-instance-attributes
    """Ingests UNFCCC climate documents into Neo4j knowledge graph.

    This class processes United Nations Framework Convention on Climate Change (UNFCCC)
    documents, extracting structured information about decisions, resolutions, annexes,
    and cross-document references. Creates a queryable knowledge graph enabling semantic
    search, temporal analysis, and citation resolution across climate conference proceedings.

    Features:
        - Document parsing for COP, CMA, CMP, SBI, and SBSTA sessions
        - Decision and resolution extraction with metadata
        - Cross-document reference detection (stored in metadata)
        - Annex identification and separation into distinct episodes
        - Paragraph-level indexing for granular access
        - Checkpoint-based resume capability for interrupted ingestion
        - Performance optimizations for large-scale graph operations

    Note:
        Reference linking, annex linking, and conference hierarchy features are currently
        disabled to optimize processing performance and reduce episode count.
    """

    # Configure console-only logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[logging.StreamHandler()]
    )

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
    INCLUDE_HEADING_IN_BODY = True
    MAX_EPISODE_NAME_LEN = 250
    MIN_SECTION_LEN = 40
    PROCESS_SUBDIRECTORIES = True

    # Regex patterns for document parsing
    YEAR_RE = re.compile(r"(20\d{2})")
    CONFERENCE_TYPE_RE = re.compile(
        r"(?P<type>CMA|CMP|COP|SBI|SBSTA)(?P<year>\d{4})_(?P<session>\d+)",
        re.IGNORECASE,
    )
    # Combined pattern for both Decisions and Resolutions
    DECISION_RESOLUTION_HEADING_RE = re.compile(
        r"(?m)^(?P<h>(?:Decision|Resolution)\s+(?:No\.?\s*)?[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+|"
        r"(?:\s*/\s*[A-Za-z]+(?:\.\d+)?))?[^\n]*)$"
    )
    OTHER_HEADING_RE = re.compile(
        r"(?mi)^(?P<h>(Chapter|Section|Agenda item|Article|Part)\s+[^\n]+)$",
        re.IGNORECASE,
    )
    # Annex pattern for metadata extraction only (not for primary splitting)
    ANNEX_HEADING_RE = re.compile(
        r"(?mi)^(?P<h>Annex(?:\s+[IVXLC]+|\s+\d+)?[^\n]*)$", re.IGNORECASE
    )
    LOCATION_DATE_RE = re.compile(
        r"held in (?P<location>[^,\n]+?)(?:\s+from\s+(?P<date>[^\n]+?))?(?:\n|$)",
        re.IGNORECASE,
    )
    FCCC_DOC_RE = re.compile(r"(?P<fccc>FCCC/[A-Z/]+/\d{4}/[\d/A-Za-z\.]+)")

    # Reference extraction patterns
    # Matches decision/resolution references with optional paragraph numbers
    DECISION_REFERENCE_RE = re.compile(
        r"(?i)(?:decision|resolution)s?\s+(?:No\.?\s*)?"
        r"(?P<number>[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+))"
        r"(?:,\s*paragraph(?:s)?\s+(?P<paragraphs>[\d\-,\s]+))?"
    )

    # Matches article references with optional paragraphs and agreements
    ARTICLE_REFERENCE_RE = re.compile(
        r"(?i)Article\s+(?P<number>[\dIVXLC]+)"
        r"(?:,\s*paragraph(?:s)?\s+(?P<paragraphs>[\d\-,\s]+))?"
        r"(?:\s+of\s+the\s+(?P<agreement>[^\n,;\.]+))?"
    )

    # Matches standalone paragraph references
    PARAGRAPH_REFERENCE_RE = re.compile(
        r"(?i)paragraphs?\s+(?P<paragraphs>[\d\-–,\s]+)"
        r"(?:\s+of\s+(?:decision|resolution)\s+(?P<number>[\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+)))?"
    )

    # Matches annex references with optional decision context
    ANNEX_REFERENCE_RE = re.compile(
        r"(?i)(?:the\s+)?annex(?:\s+(?P<annex_id>[IVXLC]+|\d+))?"
        r"(?:\s+to\s+(?:decision|resolution)\s+(?P<decision_number>[\dIVXLC]+"
        r"(?:\s*/\s*[A-Za-z]+\.\d+)))?"
    )

    # Decision action classification patterns
    CREATION_VERBS_RE = re.compile(
        r'(?i)\b(establishes?|creates?|decides\s+to\s+establish|'
        r'decides\s+to\s+create|launches?|inaugurates?|sets?\s+up)\b'
    )
    FOLLOWUP_VERBS_RE = re.compile(
        r'(?i)\b(further\s+develops?|also\s+recalling|'
        r'builds?\s+on|welcomes?\s+the\s+continued|reaffirms?|'
        r'decides\s+that\s+.{5,40}shall\s+have)\b'
    )

    CLIMATE_QUERIES = [
        "What is the Paris Agreement?",
        "What are the decisions on climate finance?",
        "What is the global goal on adaptation?",
        "What are the mechanisms for Article 6?",
        "What is the new collective quantified goal on climate finance?",
    ]

    NODE_QUERIES = [
        "climate finance",
        "adaptation",
        "mitigation",
        "Paris Agreement",
        "developed country Parties",
    ]

    def __init__(self) -> None:
        """Initializes the Neo4j ingestion tool with default configuration.

        Sets up logging, loads configuration from environment variables, and
        initializes data structures for tracking decisions and references during ingestion.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict[str, Any] = self._load_config()
        # Maps decision IDs to episode names (only actual decisions, not references)
        self.decision_registry: Dict[str, str] = {}
        # Tracks all references found during parsing
        self.reference_map: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """Entry point for Neuro-San CodedTool interface.

        Accepts configuration overrides via args parameter and executes the full
        ingestion pipeline, returning a summary of processing results.

        Args:
            args: Optional configuration overrides (data_dir, batch_size, etc.).
            sly_data: Neuro-San system data (unused in this implementation).

        Returns:
            Summary string with counts of episodes processed, references extracted,
            and relationships created.
        """
        del sly_data  # unused in this tool
        args = args or {}
        if args:
            self.config = self._load_config(args)
        summary = await self.run()
        return (
            "Neo4j ingestion completed: "
            f"{summary['success']} succeeded, {summary['failed']} failed "
            f"out of {summary['prepared']} prepared episodes. "
            f"{summary['references_extracted']} references extracted and stored in metadata."
        )

    async def run(self) -> Dict[str, int]:
        """Executes complete document ingestion pipeline.

        Connects to Neo4j, processes all documents in the configured data directory,
        extracts structured information and references, optionally runs demonstration
        queries, and returns processing statistics.

        Returns:
            Dictionary with processing statistics:
                - prepared: Total episodes created from documents
                - success: Episodes successfully added to graph
                - failed: Episodes that failed to process
                - references_extracted: Total cross-document references found
        """
        self.logger.info(
            "Connecting to Neo4j database at %s",
            self.config["neo4j_uri"],
        )
        graphiti = self._create_graphiti_client()
        self.logger.info("Successfully connected to Neo4j")

        summary: Dict[str, int] = {
            "prepared": 0,
            "success": 0,
            "failed": 0,
            "references_extracted": 0,
        }
        success_count = 0
        failed: List[Tuple[str, str]] = []

        try:
            episodes = self.build_episodes(self.config["data_dir"])
            episodes = self._limit_episodes(episodes)
            self._log_episode_stats(episodes)

            success_count, failed = await self._add_episodes(graphiti, episodes)

            # Extract and count references
            total_refs = sum(len(refs) for refs in self.reference_map.values())
            summary["references_extracted"] = total_refs
            self.logger.info(
                "Extracted %d references from %d episodes",
                total_refs,
                len(self.reference_map),
            )

            # Reference linking, annex linking, and conference linking removed to reduce
            # episode count and improve processing performance

            if self.config["enable_demo_searches"] and success_count > 0:
                await self._run_demo_searches(graphiti)
            else:
                self._log_demo_skipped_reason(success_count)

            summary["prepared"] = len(episodes)
            summary["success"] = success_count
            summary["failed"] = len(failed)

            if summary["failed"] > 0:
                self.logger.warning("Failed episodes (first 10 shown):")
                for name, error in failed[:10]:
                    self.logger.warning("  - %s: %s", name[:100], error[:200])
        finally:
            await graphiti.close()
            self.logger.info("\nConnection closed")

        return summary

    def build_episodes(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
        self, dir_path: Path
    ) -> List[Dict[str, Any]]:
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

                episode_name = f"{path.stem}::{section.get('title') or f'Part {idx}'}"
                body = self._build_episode_body(section["body"], combined_metadata)

                references = self._extract_references(section["body"], decision_id)
                paragraph_index = self._extract_paragraph_index(section["body"])

                # Store in combined metadata
                enriched_metadata = combined_metadata.copy()
                if decision_id:
                    enriched_metadata["decision_id"] = decision_id
                    self.decision_registry[decision_id] = episode_name
                    enriched_metadata["decision_action_type"] = self._classify_decision_action(section["body"])

                if annex_id:
                    enriched_metadata["annex_id"] = annex_id

                if references:
                    enriched_metadata["references"] = references
                    self.reference_map[episode_name] = references

                if paragraph_index:
                    enriched_metadata["paragraph_index"] = paragraph_index
                    if self.config["verbose_logging"]:
                        self.logger.debug(
                            "Extracted %d paragraphs from %s",
                            len(paragraph_index),
                            episode_name[:50],
                        )

                episodes.append(
                    {
                        "name": episode_name[: self.MAX_EPISODE_NAME_LEN],
                        "episode_body": body,
                        "source_description": source_description,
                        "reference_time": reference_time,
                        "metadata": enriched_metadata,
                        "decision_id": decision_id,
                        "annex_id": annex_id,
                        "references": references,
                        "document_name": path.stem,
                    }
                )
                if self.config["verbose_logging"]:
                    self.logger.info("Prepared episode: %s", episode_name[:100])
                    if decision_id:
                        self.logger.info("  Decision ID: %s", decision_id)
                    if annex_id:
                        self.logger.info("  Annex ID: %s", annex_id)
                    if references:
                        self.logger.info("  Found %d references", len(references))

        return episodes

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

        # Match only decisions and resolutions
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

        # Match annex headings
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

        # Extract decision/resolution references
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

        # Extract article references
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

        # Extract annex references
        for match in self.ANNEX_REFERENCE_RE.finditer(text):
            annex_id = match.group("annex_id")
            decision_number = match.group("decision_number")

            ref = {
                "type": "annex",
                "annex_id": annex_id.strip() if annex_id else None,
                "target": (
                    decision_number.strip() if decision_number else None
                ),
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

        # Standard numbered paragraphs (e.g., "1.", "67.")
        numbered_pattern = r"^\s*(\d+)\.\s+(.+?)(?=^\s*\d+\.\s|\Z)"
        matches = re.finditer(numbered_pattern, text, re.MULTILINE | re.DOTALL)

        for match in matches:
            para_num = match.group(1).strip()
            para_text = match.group(2).strip()
            para_text = " ".join(para_text.split())
            paragraphs[para_num] = para_text[:1000]

        # If no numbered paragraphs found, try Roman numerals (e.g., "I.", "IV.")
        if not paragraphs:
            roman_pattern = r"^\s*([IVXLC]+)\.\s+(.+?)(?=^\s*[IVXLC]+\.\s|\Z)"
            matches = re.finditer(roman_pattern, text, re.MULTILINE | re.DOTALL)

            for match in matches:
                para_num = match.group(1).strip()
                para_text = match.group(2).strip()
                para_text = " ".join(para_text.split())
                paragraphs[para_num] = para_text[:1000]

        return paragraphs

    def _classify_decision_action(self, text: str) -> str:
        """Classify whether a decision creates something new or follows up on prior work.

        Analyzes the first 2000 characters of the decision text for creation verbs
        (establishes, creates, launches) vs follow-up verbs (further develops, recalling,
        reaffirms) to determine the decision's action type.

        Args:
            text: Decision text to classify.

        Returns:
            One of 'founding', 'follow_up', or 'neutral'.
        """
        check_text = text[:2000]
        creation_matches = len(self.CREATION_VERBS_RE.findall(check_text))
        followup_matches = len(self.FOLLOWUP_VERBS_RE.findall(check_text))
        if creation_matches > 0 and creation_matches >= followup_matches:
            return "founding"
        if followup_matches > creation_matches:
            return "follow_up"
        return "neutral"

    def _load_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Loads configuration from environment variables with optional overrides.

        Reads connection settings, processing parameters, and feature flags from
        environment variables. Accepts runtime overrides for flexible configuration.

        Args:
            overrides: Optional dictionary of configuration overrides.

        Returns:
            Complete configuration dictionary with keys:
                - neo4j_uri, neo4j_user, neo4j_password: Database connection settings
                - data_dir: Path to source documents
                - batch_size, max_episodes, search_limit: Processing parameters
                - enable_demo_searches, verbose_logging: Feature flags
        """
        overrides = overrides or {}

        config: Dict[str, Any] = {
            # Connection settings from environment only (no overrides allowed)
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
            "neo4j_password": os.environ.get("NEO4J_PASSWORD", "password"),
            # Data source (overrideable)
            "data_dir": Path(os.environ.get("DATA_DIR", "documents")),
            # Processing settings (overrideable)
            "batch_size": int(os.environ.get("BATCH_SIZE", "10")),
            "max_episodes": int(os.environ.get("MAX_EPISODES", "0")),
            "search_limit": int(os.environ.get("SEARCH_LIMIT", "5")),
            # Feature toggles (overrideable)
            "enable_demo_searches": os.environ.get(
                "ENABLE_DEMO_SEARCHES", "true"
            ).lower()
            == "true",
            "verbose_logging": os.environ.get("VERBOSE_LOGGING", "false").lower()
            == "true",
        }

        if "data_dir" in overrides and overrides["data_dir"]:
            config["data_dir"] = Path(str(overrides["data_dir"]))

        int_keys = ("batch_size", "max_episodes", "search_limit")
        for key in int_keys:
            if key in overrides and overrides[key] is not None:
                config[key] = int(overrides[key])

        bool_keys = ("enable_demo_searches", "verbose_logging")
        for key in bool_keys:
            if key in overrides and overrides[key] is not None:
                config[key] = self._to_bool(overrides[key])

        return config

    @staticmethod
    def _to_bool(value: Any) -> bool:
        """Converts various value types to boolean.

        Handles string representations of booleans (e.g., "true", "1", "yes")
        in addition to native boolean and other types.

        Args:
            value: Value to convert to boolean.

        Returns:
            Boolean representation of the input value.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    def _create_graphiti_client(self) -> Graphiti:  # pylint: disable=too-many-locals
        """Creates and configures a Graphiti client for Neo4j.

        Initializes the Graphiti framework with Neo4j driver and applies
        constrained prompts to prevent entity hallucinations during graph construction.

        Returns:
            Configured Graphiti client instance ready for use.
        """
        # Apply constrained prompts to prevent entity hallucinations
        try:
            try:
                from . import constrained_prompts  # pylint: disable=import-outside-toplevel
            except ImportError:
                import sys  # pylint: disable=import-outside-toplevel

                graph_rag_dir = Path(__file__).parent
                if str(graph_rag_dir) not in sys.path:
                    sys.path.insert(0, str(graph_rag_dir))
                import constrained_prompts  # pylint: disable=import-error,import-outside-toplevel

            import graphiti_core.prompts.extract_edges as extract_edges_module  # pylint: disable=import-outside-toplevel
            import graphiti_core.prompts.extract_nodes as extract_nodes_module  # pylint: disable=import-outside-toplevel

            extract_nodes_module.extract_text = (
                constrained_prompts.extract_text_constrained
            )
            extract_edges_module.edge = constrained_prompts.extract_edges_constrained

            self.logger.info("Applied constrained prompts to prevent hallucinations")
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self.logger.warning("Failed to apply constrained prompts: %s", exc)
            self.logger.warning(
                "Continuing with default Graphiti prompts (may cause hallucinations)"
            )

        return Graphiti(
            self.config["neo4j_uri"],
            self.config["neo4j_user"],
            self.config["neo4j_password"]
        )

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
        except Exception as exc:  # pylint: disable=broad-exception-caught
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
            if 2000 <= year <= 2025:
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

    def _split_document(  # pylint: disable=too-many-nested-blocks
        self, raw_text: str
    ) -> List[Dict[str, str]]:
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
        text = self._strip_front_matter(text)

        # Split by decisions/resolutions first, then extract annexes as separate sections
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

                        # Decision content (before annex)
                        if first_annex_pos > self.MIN_SECTION_LEN:
                            decision_part = section["body"][:first_annex_pos].strip()
                            final_sections.append(
                                {"title": section["title"], "body": decision_part}
                            )

                        # Annex sections
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
            chunk = text[idx : idx + self.FALLBACK_CHARS].strip()
            if len(chunk) < self.MIN_SECTION_LEN:
                continue
            sections.append(
                {"title": f"Part {idx // self.FALLBACK_CHARS + 1}", "body": chunk}
            )
        return sections

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
        information to the section body for context.

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
        if metadata.get("decision_action_type"):
            header_lines.append(f"Decision Type: {metadata['decision_action_type']}")

        if not header_lines:
            return section_body
        return "\n".join(header_lines) + "\n\n" + section_body

    def _limit_episodes(self, episodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Limits the number of episodes to process based on MAX_EPISODES configuration.

        Used for testing or partial processing. Returns all episodes if MAX_EPISODES
        is 0 or exceeds the episode count.

        Args:
            episodes: Full list of prepared episodes.

        Returns:
            Truncated or full episode list based on MAX_EPISODES setting.
        """
        max_episodes = self.config["max_episodes"]
        if 0 < max_episodes < len(episodes):
            self.logger.info(
                "Limiting to first %d episodes (MAX_EPISODES is set)", max_episodes
            )
            return episodes[:max_episodes]
        return episodes

    def _log_episode_stats(self, episodes: List[Dict[str, Any]]) -> None:
        """Logs statistical summary of prepared episodes.

        Reports total episode count, distribution by conference type and year,
        episode types (decisions vs annexes), and paragraph indexing statistics.

        Args:
            episodes: List of prepared episode dictionaries.
        """
        self.logger.info("Successfully prepared %d episodes", len(episodes))

        conf_types: Dict[str, int] = {}
        years: Dict[str, int] = {}
        total_paragraphs = 0
        episodes_with_paragraphs = 0
        decision_count = 0
        annex_count = 0

        for episode in episodes:
            metadata = episode.get("metadata", {})
            if "conference_type" in metadata:
                conf_type = metadata["conference_type"]
                conf_types[conf_type] = conf_types.get(conf_type, 0) + 1
                year = metadata.get("year", "Unknown")
                years[year] = years.get(year, 0) + 1

            paragraph_index = metadata.get("paragraph_index", {})
            if paragraph_index:
                episodes_with_paragraphs += 1
                total_paragraphs += len(paragraph_index)

            if episode.get("decision_id"):
                decision_count += 1
            if episode.get("annex_id"):
                annex_count += 1

        if conf_types:
            self.logger.info("Episode distribution by conference: %s", conf_types)
        if years:
            ordered_years = dict(sorted(years.items(), key=lambda entry: entry[0]))
            self.logger.info("Episode distribution by year: %s", ordered_years)

        self.logger.info(
            "Episode types: %d decisions, %d annexes, %d other",
            decision_count,
            annex_count,
            len(episodes) - decision_count - annex_count,
        )

        if episodes_with_paragraphs > 0:
            avg_paragraphs = total_paragraphs / episodes_with_paragraphs
            self.logger.info(
                "Paragraph indexing: %d episodes with %d total paragraphs (avg: %.1f per episode)",
                episodes_with_paragraphs,
                total_paragraphs,
                avg_paragraphs,
            )

    async def _add_episodes(  # pylint: disable=too-many-locals
        self, graphiti: Graphiti, episodes: List[Dict[str, Any]]
    ) -> Tuple[int, List[Tuple[str, str]]]:
        """Adds episodes to the knowledge graph with checkpoint-based resume.

        Processes episodes sequentially, writing checkpoints after each success to
        enable resumption after interruptions. Skips already-processed episodes.

        Args:
            graphiti: The Graphiti client instance for adding episodes.
            episodes: List of episode dictionaries to process.

        Returns:
            Tuple of (success_count, failures_list) where failures is a list of
            (episode_name, error_message) tuples.
        """
        success_count = 0
        failures: List[Tuple[str, str]] = []
        start_time = datetime.now()

        checkpoint_file = Path(__file__).parent / ".ingestion_checkpoint.txt"
        processed_episodes = set()
        if checkpoint_file.exists():
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                for line in f:
                    episode_name = line.rstrip("\n")
                    if episode_name:
                        processed_episodes.add(episode_name)
            self.logger.info(
                "\nCheckpoint found: %d episodes already processed",
                len(processed_episodes)
            )
            self.logger.info("Will skip these and continue from where you left off\n")

        skipped_count = 0
        for idx, episode in enumerate(episodes, 1):
            episode_name = episode["name"]

            if episode_name in processed_episodes:
                skipped_count += 1
                if skipped_count % 50 == 0:
                    self.logger.info("Skipped %d already-processed episodes...", skipped_count)
                continue

            try:
                self.logger.info("\n[%d/%d] Processing episode: %s", idx, len(episodes), episode_name)
                self.logger.info("    Source: %s", episode['source_description'])

                await graphiti.add_episode(
                    name=episode_name,
                    episode_body=episode["episode_body"],
                    source=EpisodeType.text,
                    source_description=episode["source_description"],
                    reference_time=episode["reference_time"],
                )
                success_count += 1
                self.logger.info("    ✓ Successfully added")

                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    f.write(f"{episode_name}\n")

                if idx % self.config["batch_size"] == 0:
                    self._log_progress(idx, len(episodes), start_time)
                elif self.config["verbose_logging"]:
                    self.logger.info("Added episode %d/%d: %s", idx, len(episodes), episode_name[:100])
            except Exception as exc:  # pylint: disable=broad-exception-caught
                failure = (episode_name, str(exc))
                failures.append(failure)
                self.logger.error("Failed to add episode %s: %s", episode_name, exc)

        self._log_summary(len(episodes), success_count, failures, start_time)
        return success_count, failures

    def _log_progress(self, completed: int, total: int, start_time: datetime) -> None:
        """Logs processing progress with rate and ETA calculations.

        Args:
            completed: Number of episodes processed so far.
            total: Total number of episodes to process.
            start_time: Processing start timestamp for rate calculations.
        """
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = total - completed
        eta = remaining / rate if rate > 0 else 0
        percentage = completed / total * 100 if total else 0
        self.logger.info(
            "Progress: %d/%d episodes added (%.1f%%) - Rate: %.1f eps/sec - ETA: %.1f min",
            completed,
            total,
            percentage,
            rate,
            eta / 60,
        )

    def _log_summary(
        self,
        total: int,
        success_count: int,
        failures: List[Tuple[str, str]],
        start_time: datetime,
    ) -> None:
        """Logs final processing summary with statistics.

        Args:
            total: Total number of episodes attempted.
            success_count: Number of episodes successfully processed.
            failures: List of (episode_name, error_message) tuples for failures.
            start_time: Processing start timestamp for duration calculation.
        """
        elapsed_total = (datetime.now() - start_time).total_seconds()
        success_rate = success_count / total * 100 if total else 0
        average_rate = success_count / elapsed_total if elapsed_total else 0
        self.logger.info("\n%s", "=" * 80)
        self.logger.info("EPISODE LOADING SUMMARY")
        self.logger.info("%s", "=" * 80)
        self.logger.info("Total episodes processed: %d", total)
        self.logger.info("Successfully added: %d", success_count)
        self.logger.info("Failed: %d", len(failures))
        self.logger.info("Success rate: %.1f%%", success_rate)
        self.logger.info(
            "Total time: %.1f minutes", elapsed_total / 60 if elapsed_total else 0
        )
        self.logger.info("Average rate: %.2f episodes/second", average_rate)

    def _print_section(self, title: str, double_space: bool = False) -> None:
        """Prints a formatted section header with consistent styling.

        Args:
            title: The section title to display.
            double_space: If True, adds extra newline before the section header.
        """
        prefix = "\n\n" if double_space else "\n"
        self.logger.info("%s%s", prefix, "=" * 80)
        self.logger.info("%s", title)
        self.logger.info("%s", "=" * 80)

    async def _run_demo_searches(self, graphiti: Graphiti) -> None:
        """Executes all demonstration search queries to showcase knowledge graph capabilities.

        Runs climate queries, node searches, reference traversal demonstrations,
        temporal queries, and paragraph-level access examples.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("CLIMATE CONFERENCE KNOWLEDGE GRAPH SEARCH RESULTS")
        await self._run_climate_queries(graphiti)
        await self._run_node_queries(graphiti)
        await self._demo_reference_traversal(graphiti)
        await self._demo_temporal_queries(graphiti)
        await self._demo_paragraph_access(graphiti)

    async def _run_climate_queries(self, graphiti: Graphiti) -> None:
        """Executes predefined climate-related search queries.

        Demonstrates semantic search capabilities on climate conference decisions,
        including queries about the Paris Agreement, climate finance, adaptation,
        mitigation, and Article 6 mechanisms.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        for query in self.CLIMATE_QUERIES:
            self._print_section(f"QUERY: {query}", double_space=True)
            try:
                results = await graphiti.search(query)
                results = (results or [])[: self.config["search_limit"]]
                if results:
                    self.logger.info("\nFound %d results:\n", len(results))
                    for idx, result in enumerate(results, 1):
                        self.logger.info("\n--- Result %d ---", idx)
                        self.logger.info("Fact: %s", result.fact)
                        if getattr(result, "valid_at", None):
                            self.logger.info("Valid from: %s", result.valid_at)
                        if getattr(result, "invalid_at", None):
                            self.logger.info("Valid until: %s", result.invalid_at)
                else:
                    self.logger.info("\nNo results found for this query.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.logger.error('Search failed for query "%s": %s', query, exc)

    async def _run_node_queries(self, graphiti: Graphiti) -> None:
        """Performs node-based searches for key climate topics.

        Uses hybrid search with reciprocal rank fusion to find the most relevant
        entities in the knowledge graph for topics like climate finance, adaptation,
        mitigation, and the Paris Agreement.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("NODE SEARCH: Key Climate Topics", double_space=True)

        for query in self.NODE_QUERIES:
            self.logger.info("\n--- Searching nodes for: %s ---", query)
            try:
                node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
                node_search_config.limit = 3
                node_search_results = (
                    await graphiti._search(  # pylint: disable=protected-access
                        query=query,
                        config=node_search_config,
                    )
                )
                if node_search_results.nodes:
                    for node in node_search_results.nodes:
                        self.logger.info("\n  Node: %s", node.name)
                        summary = (node.summary or "").strip()
                        summary = (
                            summary[:150] + "..." if len(summary) > 150 else summary
                        )
                        self.logger.info("  Summary: %s", summary)
                        self.logger.info("  Labels: %s", ', '.join(node.labels))
                else:
                    self.logger.info("  No nodes found.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.logger.error('Node search failed for "%s": %s', query, exc)

    async def _demo_reference_traversal(self, graphiti: Graphiti) -> None:
        """Demonstrates cross-document reference traversal capabilities.

        Shows how decisions reference other decisions, articles, and paragraphs,
        enabling navigation through the interconnected web of climate conference
        decisions and legal instruments.

        Args:
            graphiti: The Graphiti client instance (not used in this demo).
        """
        del graphiti  # Not needed for this demo method
        self._print_section("REFERENCE TRAVERSAL DEMO", double_space=True)

        # Show some sample references
        sample_count = 0
        for episode_name, refs in list(self.reference_map.items())[:3]:
            self.logger.info("\n--- Episode: %s ---", episode_name[:80])
            self.logger.info("References %d other decisions/articles:", len(refs))
            for ref in refs[:5]:  # Show first 5 references
                if ref["type"] == "decision":
                    ref_text = f"  → Decision {ref['target']}"
                    if ref.get("paragraphs"):
                        ref_text += f", paragraph(s) {ref['paragraphs']}"
                    self.logger.info("%s", ref_text)
                elif ref["type"] == "article":
                    ref_text = f"  → Article {ref['target']}"
                    if ref.get("agreement"):
                        ref_text += f" of {ref['agreement']}"
                    if ref.get("paragraphs"):
                        ref_text += f", paragraph(s) {ref['paragraphs']}"
                    self.logger.info("%s", ref_text)
            sample_count += 1
            if sample_count >= 3:
                break

    async def _demo_temporal_queries(  # pylint: disable=too-many-locals,too-many-branches,too-many-nested-blocks
        self, graphiti: Graphiti
    ) -> None:
        """Demonstrates temporal query capabilities for tracking decision evolution.

        Shows how to query decisions by year, analyze temporal ranges, visualize
        decision timelines, and track how climate policy approaches evolved over time.
        Includes examples of querying decision trends from specific time periods.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("TEMPORAL QUERY DEMO", double_space=True)

        # Build temporal index from decision registry
        decisions_by_year: Dict[str, List[str]] = defaultdict(list)
        for decision_id, episode_name in self.decision_registry.items():
            # Extract year from metadata if available
            for episode_name_iter, refs in self.reference_map.items():
                if episode_name_iter == episode_name and refs:
                    # Get year from first reference's context or episode metadata
                    break
            # Extract year from decision ID (e.g., "15/CMA.1" -> session 1 -> typically 2018)
            # Or from episode name which contains year
            if "::" in episode_name:
                filename_part = episode_name.split("::")[0]
                # Extract year from filename (e.g., "CMA2018_1.3_Decisions")
                year_match = re.search(r"(\d{4})", filename_part)
                if year_match:
                    year = year_match.group(1)
                    decisions_by_year[year].append(decision_id)

        if not decisions_by_year:
            self.logger.info("\nNo temporal data available for demo.")
            return

        self.logger.info("\nDecisions by year:")
        sorted_years = sorted(decisions_by_year.keys())
        for year in sorted_years:
            self.logger.info("%s: %d decisions", year, len(decisions_by_year[year]))

        # Demo: Show decisions from a specific time period
        self._print_section("TEMPORAL RANGE QUERY DEMO")

        # Example: Show decisions from 2020-2022
        start_year = "2020"
        end_year = "2022"
        decisions_in_range = []
        for year in sorted_years:
            if start_year <= year <= end_year:
                decisions_in_range.extend(decisions_by_year[year])

        if decisions_in_range:
            self.logger.info(
                "\nDecisions from %s to %s: %d total",
                start_year, end_year, len(decisions_in_range)
            )
            self.logger.info("\nSample decisions from this period:")
            for decision_id in decisions_in_range[:10]:
                self.logger.info("  • %s", decision_id)
            if len(decisions_in_range) > 10:
                self.logger.info("  ... and %d more", len(decisions_in_range) - 10)

        # Demo: Timeline evolution query
        self._print_section("TEMPORAL EVOLUTION QUERY")

        # Check if we have decisions across multiple years
        if len(sorted_years) >= 2:
            self.logger.info("\nTimeline: %s to %s", sorted_years[0], sorted_years[-1])
            self.logger.info(
                "Total span: %d years",
                int(sorted_years[-1]) - int(sorted_years[0]) + 1
            )
            self.logger.info("\nDecisions per year:")
            for year in sorted_years:
                count = len(decisions_by_year[year])
                bar_chart = "█" * min(count, 50)
                self.logger.info("  %s: %s (%d)", year, bar_chart, count)

        # Demo: Query with temporal context
        self._print_section("QUERYING WITH TEMPORAL CONTEXT")

        temporal_queries = [
            (
                f"What decisions were made about climate finance "
                f"between {sorted_years[0]} and {sorted_years[-1]}?"
            ),
            (
                f"How did the approach to adaptation evolve "
                f"from {sorted_years[0]} to {sorted_years[-1]}?"
            ),
        ]

        for query in temporal_queries[:1]:  # Just run one to avoid too much output
            self.logger.info("\nQuery: %s", query)
            try:
                results = await graphiti.search(query)
                results = (results or [])[:3]
                if results:
                    self.logger.info("Found %d results (showing top 3):", len(results))
                    for idx, result in enumerate(results, 1):
                        fact_preview = (
                            result.fact[:150] + "..."
                            if len(result.fact) > 150
                            else result.fact
                        )
                        self.logger.info("  %d. %s", idx, fact_preview)
                else:
                    self.logger.info("No results found.")
            except Exception as exc:  # pylint: disable=broad-exception-caught
                self.logger.error("Temporal query failed: %s", exc)

    async def _demo_paragraph_access(  # pylint: disable=too-many-locals
        self, graphiti: Graphiti
    ) -> None:
        """Demonstrates paragraph-level granular access and citation resolution.

        Shows how the system provides direct access to specific paragraphs within
        decisions, enabling precise citation resolution (e.g., 'paragraph 69 of
        decision 1/CP.21') without searching full text. Demonstrates the paragraph
        indexing functionality that enables efficient granular retrieval.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("PARAGRAPH-LEVEL ACCESS DEMO", double_space=True)

        # Find episodes with paragraph indices
        episodes_with_paragraphs = []
        for decision_id, episode_name in self.decision_registry.items():
            # Check if this episode has references with paragraph numbers
            refs = self.reference_map.get(episode_name, [])
            has_para_refs = any(ref.get("paragraphs") for ref in refs)
            if has_para_refs:
                episodes_with_paragraphs.append((decision_id, episode_name, refs))

        if not episodes_with_paragraphs:
            self.logger.info("\nNo paragraph-level references found in demo.")
            return

        self.logger.info(
            "\nFound %d decisions with paragraph-level references",
            len(episodes_with_paragraphs)
        )

        # Show a few examples
        self._print_section("EXAMPLE: Decisions Referencing Specific Paragraphs")

        for decision_id, episode_name, refs in episodes_with_paragraphs[:3]:
            self.logger.info("\n--- Decision: %s ---", decision_id)

            # Show paragraph references this decision makes
            para_refs = [ref for ref in refs if ref.get("paragraphs")]
            if para_refs:
                self.logger.info("This decision references specific paragraphs in:")
                for ref in para_refs[:5]:
                    target = ref.get("target", "unknown")
                    paras = ref.get("paragraphs", "")
                    self.logger.info("  • %s, paragraph(s) %s", target, paras)

        # Example query demonstration
        self._print_section("EXAMPLE PARAGRAPH-LEVEL QUERY")

        # Pick a decision with paragraph references to demonstrate
        if episodes_with_paragraphs:
            decision_id, episode_name, refs = episodes_with_paragraphs[0]
            para_ref = next((ref for ref in refs if ref.get("paragraphs")), None)

            if para_ref:
                target = para_ref.get("target")
                paras = para_ref.get("paragraphs")
                query = f"What does paragraph {paras} of decision {target} say?"

                self.logger.info("\nQuery: %s", query)
                try:
                    # Actually run the query
                    results = await graphiti.search(query)
                    results = (results or [])[:2]

                    if results:
                        self.logger.info("\nResults found: %d", len(results))
                        for idx, result in enumerate(results, 1):
                            fact_preview = (
                                result.fact[:200] + "..."
                                if len(result.fact) > 200
                                else result.fact
                            )
                            self.logger.info("\n  Result %d:", idx)
                            self.logger.info("  %s", fact_preview)
                    else:
                        self.logger.info(
                            "\nNo results found (decision may not be in current batch)"
                        )
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    self.logger.error("Paragraph query failed: %s", exc)

    def _log_demo_skipped_reason(self, success_count: int) -> None:
        """Logs the reason why demo searches were skipped.

        Args:
            success_count: Number of episodes successfully added to the graph.
        """
        if not self.config["enable_demo_searches"]:
            self.logger.info("Demo searches disabled (ENABLE_DEMO_SEARCHES=false)")
        elif success_count == 0:
            self.logger.info(
                "Skipping demo searches (no episodes were successfully added)"
            )


async def main() -> None:
    """Standalone entry point for manual execution.

    Creates a Neo4jIngestionEnhanced instance with default configuration
    from environment variables and runs the complete ingestion pipeline.
    """
    runner = Neo4jIngestionEnhanced()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
