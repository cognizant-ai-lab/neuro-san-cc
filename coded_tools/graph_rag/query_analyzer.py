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

import logging
import re
from typing import Any
from typing import Dict
from typing import Optional

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """
    Query analysis component for GraphSearchTool.

    Extracts intent, UNFCCC concepts, temporal markers, conference filters,
    structural markers, decision/paragraph references, timeline detection,
    and complexity scoring from user queries.
    """

    UNFCCC_CONCEPTS = [
        "mitigation",
        "adaptation",
        "co-benefits",
        "NDC",
        "nationally determined contribution",
        "transparency",
        "reporting",
        "review",
        "compliance",
        "finance",
        "technology transfer",
        "capacity building",
        "loss and damage",
        "market mechanism",
        "Article 6",
        "global stocktake",
        "enhanced transparency framework",
        "ETF",
        "Parties",
        "biennial transparency report",
        "BTR",
        "nationally appropriate mitigation action",
        "NAMA",
        "adaptation communication",
        "economic diversification",
        "just transition",
        "common timeframes",
        "IPCC",
        "developed country",
        "developing country",
        "least developed country",
        "LDC",
        "small island developing state",
        "SIDS",
    ]

    CONFERENCE_TYPES = ["COP", "CMA", "CMP", "SBI", "SBSTA"]

    CONF_CONSTRAINT_PATTERNS = [
        (r"\bcma\s+(?:decision|session)", "CMA"),
        (r"\bcop\s+(?:decision|session)", "COP"),
        (r"\bcp\s+(?:decision|session)", "COP"),
        (r"\bcmp\s+(?:decision|session)", "CMP"),
        (r"\bsbi\s+(?:decision|session|report)", "SBI"),
        (r"\bsbsta\s+(?:decision|session|report)", "SBSTA"),
        (r"\bacross\s+(?:\w+\s+)?cma\b", "CMA"),
        (r"\bacross\s+(?:\w+\s+)?cop\b", "COP"),
        (r"\bacross\s+(?:\w+\s+)?cmp\b", "CMP"),
        (r"\ba\s+(?:later\s+)?cma\b", "CMA"),
        (r"\ba\s+(?:later\s+)?cop\b", "COP"),
        (r"\ba\s+(?:later\s+)?cmp\b", "CMP"),
        (r"\bwhich\s+cma\b", "CMA"),
        (r"\bwhich\s+cop\b", "COP"),
        (r"\bwhich\s+cmp\b", "CMP"),
    ]

    STRUCTURAL_TERMS = {
        "annex": r"\bannex(?:es)?\s*[IVX\d]*\b",
        "decision": r"\bdecision\s*\d+\s*[/]?\s*(?:cma|cop|cp|cmp|sbi|sbsta)\s*[./]?\s*\d+\b",
        "article": r"\barticle\s*\d+\b",
        "paragraph": r"\bparagraph\s*\d+\b",
        "section": r"\bsection\s*[IVX\d]+\b",
    }

    CHRONOLOGICAL_EARLIEST_MARKERS = [
        "first",
        "originally",
        "initially",
        "created",
        "creates",
        "established",
        "establishes",
        "founded",
        "founds",
        "inception",
        "origin",
        "earliest",
        "when was",
        "who created",
        "who established",
        "which decision created",
        "which decision established",
        "which decision creates",
        "set up",
        "sets up",
        "launched",
        "launches",
    ]

    CHRONOLOGICAL_LATEST_MARKERS = [
        "latest",
        "most recent",
        "current",
        "last updated",
        "newest",
    ]

    TIMELINE_MARKERS = [
        "all decisions",
        "every decision",
        "timeline",
        "evolution",
        "history of",
        "how did",
        "what decisions were made",
        "what decisions address",
        "complete history",
        "chronolog",
        "over time",
        "across sessions",
        "track",
        "follow-up",
        "follow up",
        "review timing",
        "review cycle",
        "reporting cycle",
        "subsequent",
        "later sessions",
        "multiple sessions",
        "across multiple",
        "governance",
    ]

    IDENTIFICATION_MARKERS = [
        "a decision",
        "a later decision",
        "a cma decision",
        "a cop decision",
        "a cmp decision",
        "the decision that",
        "which decision",
        "which cma decision",
        "which cop decision",
        "which cmp decision",
        "identify the decision",
        "find the decision",
        "a later cma",
        "a later cop",
        "a later cmp",
    ]

    GOVERNANCE_PHASE_MARKERS = [
        "establishes",
        "creates",
        "set up",
        "sets up",
        "follow-up",
        "follow up",
        "operationalize",
        "review",
        "reporting",
        "integration",
        "mandate",
        "recommendations",
        "workplan",
        "work plan",
        "annual report",
        "midterm review",
    ]

    # --- Query analysis thresholds ---
    LONG_QUERY_WORD_THRESHOLD = 15
    MULTI_PHASE_THRESHOLD = 2
    MULTI_ACTION_AND_THRESHOLD = 2
    MULTI_ACTION_ALSO_THRESHOLD = 1
    HIGH_COMPLEXITY_SCORE = 5
    LOW_COMPLEXITY_SCORE = 2

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract key information and determine optimal search strategy.

        Delegates to focused sub-methods for each analysis dimension, then combines
        results into a single analysis dictionary.

        :param query: User's search query
        :return: Dictionary with query analysis results
        """
        query_lower = query.lower()
        analysis: Dict[str, Any] = {
            "intent": self._detect_intent(query_lower),
            "key_concepts": [c for c in self.UNFCCC_CONCEPTS if c in query_lower],
            "temporal_markers": [],
            "structural_markers": [],
            "complexity": "medium",
            "recommended_search_type": "general",
            "query_type": "medium_context",
        }

        self._extract_temporal_markers(query, query_lower, analysis)
        analysis["conference_filter"] = self._detect_conference_filter(query_lower)
        self._extract_structural_markers(query_lower, analysis)
        self._extract_decision_and_paragraph_refs(query_lower, analysis)
        self._detect_query_mode(query, query_lower, analysis)
        self._determine_complexity_and_type(query, analysis)

        return analysis

    def _detect_intent(self, query_lower: str) -> str:
        """
        Classify query intent as factual, requirement, definition, or relationship.

        :param query_lower: Lowercased query string
        :return: Intent classification string
        """
        requirement_words = [
            "what condition",
            "must",
            "require",
            "obligation",
            "should",
            "shall",
        ]
        if any(word in query_lower for word in requirement_words):
            return "requirement"
        if any(
            word in query_lower
            for word in ["what is", "define", "definition", "meaning of"]
        ):
            return "definition"
        relationship_words = [
            "relationship",
            "connect",
            "relate",
            "between",
            "link",
        ]
        if any(word in query_lower for word in relationship_words):
            return "relationship"
        return "factual"

    def _extract_temporal_markers(
        self, query: str, query_lower: str, analysis: Dict[str, Any]
    ) -> None:
        """
        Extract year and conference-type temporal markers from the query.

        :param query: Original query string (for year regex)
        :param query_lower: Lowercased query string
        :param analysis: Analysis dict to populate
        """
        years = re.findall(r"\b(19\d{2}|20\d{2})\b", query)
        analysis["temporal_markers"].extend(years)
        for conf in self.CONFERENCE_TYPES:
            if conf.lower() in query_lower:
                analysis["temporal_markers"].append(conf)

    def _detect_conference_filter(self, query_lower: str) -> Optional[str]:
        """
        Detect conference type constraint for filtering results.

        Checks explicit patterns first (e.g., "CMA decision"), then falls back to
        single-mention detection. Handles "CP" as alias for "COP".

        :param query_lower: Lowercased query string
        :return: Conference type string or None
        """
        for pattern, conf_type in self.CONF_CONSTRAINT_PATTERNS:
            if re.search(pattern, query_lower):
                return conf_type
        mentioned_confs = [c for c in self.CONFERENCE_TYPES if c.lower() in query_lower]
        if "cp" in query_lower.split() and "COP" not in mentioned_confs:
            mentioned_confs.append("COP")
        if len(mentioned_confs) == 1:
            return mentioned_confs[0]
        return None

    def _extract_structural_markers(
        self, query_lower: str, analysis: Dict[str, Any]
    ) -> None:
        """
        Extract document structure references (annex, decision, article, etc.).

        :param query_lower: Lowercased query string
        :param analysis: Analysis dict to populate
        """
        for term, pattern in self.STRUCTURAL_TERMS.items():
            if re.search(pattern, query_lower):
                analysis["structural_markers"].append(term)

    def _extract_decision_and_paragraph_refs(
        self, query_lower: str, analysis: Dict[str, Any]
    ) -> None:
        """
        Extract specific decision IDs and paragraph references from the query.

        :param query_lower: Lowercased query string
        :param analysis: Analysis dict to populate
        """
        # Flexible pattern: accepts slash or space as separator before conference type,
        # and dot or space as separator before session number.
        # Constrained to known conference abbreviations to avoid false positives.
        # Handles: "10/CMA.1", "10 CMA.1", "10/CMA 1", "10 CMA 1", "10/cma.1"
        decision_pattern = (
            r"\bdecision\s+(\d+)\s*[/]?\s*"
            r"(cma|cop|cp|cmp|sbi|sbsta)"
            r"\s*[./]?\s*(\d+)\b"
        )
        decision_match = re.search(decision_pattern, query_lower, re.IGNORECASE)
        if decision_match:
            num = decision_match.group(1)
            conf = decision_match.group(2).upper()
            session = decision_match.group(3)
            analysis["decision_id"] = f"{num}/{conf}.{session}"
        else:
            analysis["decision_id"] = None

        paragraph_patterns = [
            r"\b(?:para\.?|paragraph)\s+(\d+)(?:\s*\(([a-z])\))?(?:\s*\(([ivxlc]+)\))?",
            r"\(para\.?\s*(\d+)(?:\s*\(([a-z])\))?(?:\s*\(([ivxlc]+)\))?\)",
        ]
        analysis["paragraph_refs"] = []
        for pattern in paragraph_patterns:
            matches = re.finditer(pattern, query_lower, re.IGNORECASE)
            for match in matches:
                para_ref = {
                    "number": match.group(1),
                    "subsection": match.group(2) if match.lastindex >= 2 else None,
                    "subsubsection": match.group(3) if match.lastindex >= 3 else None,
                }
                analysis["paragraph_refs"].append(para_ref)

    def _detect_query_mode(
        self, query: str, query_lower: str, analysis: Dict[str, Any]
    ) -> None:
        """
        Detect identification, temporal direction, and timeline query modes.

        :param query: Original query string
        :param query_lower: Lowercased query string
        :param analysis: Analysis dict to populate
        """
        is_identification = any(
            marker in query_lower for marker in self.IDENTIFICATION_MARKERS
        )
        if not is_identification and re.search(
            r"\b(?:cma|cop|cmp|cp)\s+decision\s+\w+s\b", query_lower
        ):
            is_identification = True
        if (
            not is_identification
            and not analysis.get("decision_id")
            and len(query.split()) > self.LONG_QUERY_WORD_THRESHOLD
            and any(conf.lower() in query_lower for conf in self.CONFERENCE_TYPES)
        ):
            is_identification = True
        analysis["is_identification_query"] = is_identification

        if any(marker in query_lower for marker in self.CHRONOLOGICAL_EARLIEST_MARKERS):
            analysis["temporal_direction"] = "earliest"
            analysis["follow_references"] = True
        elif any(marker in query_lower for marker in self.CHRONOLOGICAL_LATEST_MARKERS):
            analysis["temporal_direction"] = "latest"
            analysis["follow_references"] = False
        else:
            analysis["temporal_direction"] = None
            analysis["follow_references"] = False

        analysis["is_timeline_query"] = any(
            marker in query_lower for marker in self.TIMELINE_MARKERS
        )

        phase_count = sum(1 for m in self.GOVERNANCE_PHASE_MARKERS if m in query_lower)
        if (
            phase_count >= self.MULTI_PHASE_THRESHOLD
            and not analysis["is_timeline_query"]
        ):
            analysis["is_timeline_query"] = True
            logger.debug(
                "Multi-stage governance query detected (%d phases mentioned)",
                phase_count,
            )

        has_multiple_actions = (
            query_lower.count(" and ") >= self.MULTI_ACTION_AND_THRESHOLD
            or query_lower.count("also") >= self.MULTI_ACTION_ALSO_THRESHOLD
        )
        if (
            has_multiple_actions
            and analysis.get("is_identification_query")
            and not analysis["is_timeline_query"]
        ):
            analysis["is_timeline_query"] = True
            logger.debug(
                "Multi-action identification query detected, enabling timeline search"
            )

    def _determine_complexity_and_type(
        self, query: str, analysis: Dict[str, Any]
    ) -> None:
        """
        Compute query complexity score and recommend search type.

        :param query: Original query string
        :param analysis: Analysis dict to populate
        """
        complexity_score = 0
        complexity_score += len(analysis["key_concepts"])
        complexity_score += 2 if analysis["structural_markers"] else 0
        complexity_score += 2 if analysis["intent"] == "requirement" else 0
        complexity_score += (
            1 if len(query.split()) > self.LONG_QUERY_WORD_THRESHOLD else 0
        )

        if complexity_score >= self.HIGH_COMPLEXITY_SCORE:
            analysis["complexity"] = "high"
        elif complexity_score <= self.LOW_COMPLEXITY_SCORE:
            analysis["complexity"] = "low"

        if (
            analysis.get("temporal_direction")
            or analysis.get("is_timeline_query")
            or analysis.get("is_identification_query")
        ):
            analysis["complexity"] = "high"

        if analysis["intent"] == "requirement" and analysis["structural_markers"]:
            analysis["recommended_search_type"] = "episode"
        elif analysis["intent"] == "relationship":
            analysis["recommended_search_type"] = "relationship"
        elif analysis["intent"] == "definition" and len(analysis["key_concepts"]) == 1:
            analysis["recommended_search_type"] = "entity"
        else:
            analysis["recommended_search_type"] = "general"
