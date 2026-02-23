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
from typing import List
from typing import Optional

logger = logging.getLogger(__name__)


class SearchPipeline:
    """
    Multi-stage search orchestration component for GraphSearchTool.

    Contains search pipeline stages, post-processing utilities (conference
    filtering, temporal reranking, founding classification), and query
    term expansion. All search limit constants are defined here.

    """

    # --- Classification regexes ---
    CREATION_RE = re.compile(
        r"(?i)\b(establishes?\b|creates?\b|decides\s+to\s+establish\b|"
        r"launches?\b|sets?\s+up\b|inaugurates?\b)"
    )
    FOLLOWUP_RE = re.compile(
        r"(?i)\b(further\s+develops?\b|also\s+recalling\b|"
        r"builds?\s+on\b|welcomes?\s+the\s+continued\b|reaffirms?\b|"
        r"operationaliz\w+\b|decides\s+that\s+.{5,40}shall\s+have\b)"
    )

    # --- Search result limits ---
    CANDIDATE_SEARCH_LIMIT = 10
    TIMELINE_SEARCH_LIMIT = 20
    TIMELINE_EPISODE_BOOST = 15

    # --- Per-stage limit calculations ---
    MIN_ENTITY_LIMIT_PARAGRAPH = 2
    ENTITY_LIMIT_DIVISOR_PARAGRAPH = 4
    MIN_ENTITY_LIMIT_DEFAULT = 3
    ENTITY_LIMIT_DIVISOR_DEFAULT = 3
    MIN_EPISODE_LIMIT_DEFAULT = 5
    EPISODE_LIMIT_DIVISOR_DEFAULT = 2
    RELATIONSHIP_LIMIT_MIN = 2
    RELATIONSHIP_LIMIT_DIVISOR = 4

    # --- Complexity-based search limits ---
    COMPLEXITY_LIMIT_DIRECT = 1
    COMPLEXITY_LIMIT_MEDIUM = 5
    COMPLEXITY_LIMIT_EXTENSIVE = 15

    # --- Content analysis character limits ---
    FOUNDING_CONTENT_CHECK_CHARS = 3000
    TIMELINE_CONTENT_CHECK_CHARS = 2000
    FIRST_PARAGRAPH_CHARS = 500
    FALLBACK_SEARCH_QUERY_CHARS = 100

    # --- Result slice limits ---
    MAX_SYNONYM_EXPANSIONS = 2
    MAX_BACKWARD_REFERENCES = 5

    # --- Year validation ---
    YEAR_RANGE_MIN = 2000
    YEAR_RANGE_MAX = 2030
    TEMPORAL_SORT_EARLIEST_DEFAULT = 9999
    TEMPORAL_SORT_LATEST_DEFAULT = 0

    # --- Summary thresholds ---
    MIN_SUMMARY_LINE_LENGTH = 30

    # --- Log/display preview character limits ---
    LOG_QUERY_PREVIEW_CHARS = 80
    LOG_QUERY_SHORT_PREVIEW_CHARS = 50
    LOG_EXPANDED_QUERY_PREVIEW_CHARS = 70
    LOG_DECISION_NAME_CHARS = 60
    LOG_SUMMARY_LINE_CHARS = 200
    LOG_RESULT_PREVIEW_CHARS = 300

    QUERY_TERM_MAPPINGS = {
        "mitigation co-benefits": [
            "co-benefits from mitigation",
            "mitigation benefits",
            "ancillary benefits",
        ],
        "adaptation action": [
            "adaptation activities",
            "adaptation measures",
            "adaptation efforts",
        ],
        "economic diversification": [
            "diversification plans",
            "economic transformation",
            "structural transformation",
        ],
        "additional information": [
            "further information",
            "supplementary information",
            "more detailed information",
        ],
        "Party": ["country", "nation", "State Party", "Parties"],
        "must provide": [
            "shall provide",
            "required to provide",
            "obligation to provide",
            "provide",
        ],
        "transparency": [
            "transparency framework",
            "ETF",
            "enhanced transparency framework",
        ],
        "reporting": [
            "report",
            "BTR",
            "biennial transparency report",
            "national communication",
        ],
        "NDC": [
            "nationally determined contribution",
            "nationally determined contributions",
        ],
        "annex": ["annexes", "appendix", "attachment"],
        "decision": ["decisions", "decision document"],
    }

    CONF_ALIASES = {
        "COP": ["COP", "CP"],
        "CMA": ["CMA"],
        "CMP": ["CMP"],
        "SBI": ["SBI"],
        "SBSTA": ["SBSTA"],
    }

    async def _multi_stage_search(
        self, query: str, query_analysis: Dict[str, Any], limit: int
    ) -> tuple:
        """
        Perform multi-stage search for complex queries requiring comprehensive answers.

        Orchestrates priority searches (decision, paragraph), then entity/episode/
        relationship stages, post-processing (conference filter, backward refs,
        timeline, temporal rerank, founding classification), and output assembly.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param limit: Maximum results per stage
        :return: Tuple of (combined_results, formatted_output)
        """
        logger.debug(
            "Executing multi-stage search for complex query: %s...",
            query[: self.LOG_QUERY_PREVIEW_CHARS],
        )

        decision_results = await self._run_decision_search(query_analysis, query)
        paragraph_results = await self._run_paragraph_search(query, query_analysis)

        entity_limit, episode_limit = self._compute_search_limits(
            query_analysis, limit, paragraph_results
        )

        logger.debug("Stage 1: Searching entities (limit=%d)", entity_limit)
        entity_results = await self._search_entities(query, entity_limit)

        episode_results = await self._run_episode_search(
            query, query_analysis, episode_limit, paragraph_results
        )

        referenced_decisions = await self._run_backward_references(
            query_analysis, episode_results
        )

        timeline_results = await self._run_timeline_search(query, query_analysis)

        if query_analysis.get("temporal_direction") and episode_results:
            episode_results = self._temporal_rerank(
                episode_results, query_analysis["temporal_direction"]
            )
            logger.debug(
                "Re-ranked episodes by %s year", query_analysis["temporal_direction"]
            )

        founding_analysis, is_creation_query = self._run_founding_classification(
            query, query_analysis, episode_results
        )

        relationship_results = await self._run_relationship_search(
            query, query_analysis, entity_results, limit
        )

        formatted_output = await self._assemble_multi_stage_output(
            query,
            query_analysis,
            decision_results,
            paragraph_results,
            founding_analysis,
            is_creation_query,
            referenced_decisions,
            timeline_results,
            episode_results,
            entity_results,
            relationship_results,
        )

        combined_results = {
            "decision": decision_results,
            "paragraphs": paragraph_results,
            "referenced_decisions": referenced_decisions,
            "timeline": timeline_results,
            "episodes": episode_results,
            "entities": entity_results,
            "relationships": relationship_results,
        }

        return combined_results, formatted_output

    async def _run_decision_search(
        self, query_analysis: Dict[str, Any], query: str
    ) -> str:
        """
        PRIORITY 0: Search for a specific decision if referenced in the query.

        :param query_analysis: Analysis results from _analyze_query
        :param query: Original search query
        :return: Formatted decision text, or empty string if none found
        """
        if query_analysis.get("decision_id"):
            logger.debug(
                "Detected specific decision reference: %s, searching directly...",
                query_analysis["decision_id"],
            )
            return await self._search_by_decision(query_analysis["decision_id"])
        return ""

    async def _run_paragraph_search(
        self, query: str, query_analysis: Dict[str, Any]
    ) -> List[Any]:
        """
        PRIORITY 1: Search for specific paragraphs if referenced in the query.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :return: Paragraph search results (may be empty list)
        """
        if query_analysis.get("paragraph_refs"):
            para_count = len(query_analysis["paragraph_refs"])
            logger.debug(
                "Detected %d paragraph reference(s), attempting metadata search...",
                para_count,
            )
            results = await self._search_by_paragraph(query, query_analysis)
            if results:
                logger.debug("Found %d paragraphs via metadata search", len(results))
            else:
                logger.debug(
                    "Paragraph metadata search found nothing, falling back to standard search"
                )
            return results
        return []

    def _compute_search_limits(
        self,
        query_analysis: Dict[str, Any],
        limit: int,
        paragraph_results: List[Any],
    ) -> tuple:
        """
        Compute per-stage search limits based on query analysis and prior results.

        :param query_analysis: Analysis results from _analyze_query
        :param limit: Base limit from caller
        :param paragraph_results: Results from paragraph search (affects limits)
        :return: Tuple of (entity_limit, episode_limit)
        """
        if paragraph_results:
            entity_limit = max(
                self.MIN_ENTITY_LIMIT_PARAGRAPH,
                limit // self.ENTITY_LIMIT_DIVISOR_PARAGRAPH,
            )
            episode_limit = 0
        else:
            entity_limit = max(
                self.MIN_ENTITY_LIMIT_DEFAULT,
                limit // self.ENTITY_LIMIT_DIVISOR_DEFAULT,
            )
            episode_limit = max(
                self.MIN_EPISODE_LIMIT_DEFAULT,
                limit // self.EPISODE_LIMIT_DIVISOR_DEFAULT,
            )

        if query_analysis.get("is_timeline_query") or (
            query_analysis.get("is_identification_query")
            and query_analysis.get("temporal_direction")
        ):
            episode_limit = max(episode_limit, self.TIMELINE_EPISODE_BOOST)
            logger.debug(
                "Boosted episode limit to %d for timeline/evolution query",
                episode_limit,
            )

        return entity_limit, episode_limit

    async def _run_episode_search(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        episode_limit: int,
        paragraph_results: List[Any],
    ) -> List[Any]:
        """
        Stage 2: Search primary source documents (episodes) with conference filtering.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param episode_limit: Maximum episode results
        :param paragraph_results: Prior paragraph results (for skip logging)
        :return: Filtered episode results
        """
        episode_results: List[Any] = []
        if episode_limit > 0:
            logger.debug(
                "Stage 2: Searching episodes/documents (limit=%d)", episode_limit
            )
            episode_query = query
            if query_analysis["structural_markers"]:
                structural_terms = " ".join(query_analysis["structural_markers"])
                episode_query = f"{query} {structural_terms}"
            episode_results = await self._search_episodes(episode_query, episode_limit)

        conf_filter = query_analysis.get("conference_filter")
        if conf_filter and episode_results:
            pre_filter_count = len(episode_results)
            episode_results = self._filter_by_conference(episode_results, conf_filter)
            if len(episode_results) < pre_filter_count:
                logger.debug(
                    "Conference filter '%s': %d → %d episodes",
                    conf_filter,
                    pre_filter_count,
                    len(episode_results),
                )

        if not episode_results and paragraph_results:
            logger.debug(
                "Stage 2: Skipping general episode search (using paragraph results instead)"
            )

        return episode_results

    async def _run_backward_references(
        self, query_analysis: Dict[str, Any], episode_results: List[Any]
    ) -> List[Any]:
        """
        Stage 2B: Follow backward references for chronological/creation queries.

        :param query_analysis: Analysis results from _analyze_query
        :param episode_results: Episode results to trace references from
        :return: Referenced earlier decisions (may be empty list)
        """
        should_follow_refs = query_analysis.get("follow_references") or (
            query_analysis.get("is_identification_query")
            and query_analysis.get("temporal_direction") == "earliest"
        )
        if should_follow_refs and episode_results:
            logger.debug(
                "Stage 2B: Following backward references for chronological/creation query..."
            )
            referenced_decisions = await self._follow_backward_references(
                episode_results
            )
            if referenced_decisions:
                logger.debug(
                    "Found %d referenced earlier decision(s)", len(referenced_decisions)
                )
            return referenced_decisions
        return []

    async def _run_timeline_search(
        self, query: str, query_analysis: Dict[str, Any]
    ) -> List[Any]:
        """
        Stage 2C: Search broadly for all decisions on the topic (timeline queries).

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :return: Timeline results sorted chronologically (may be empty list)
        """
        if query_analysis.get("is_timeline_query"):
            logger.debug("Stage 2C: Searching for complete decision timeline...")
            timeline_results = await self._search_topic_timeline(
                query, query_analysis, limit=self.TIMELINE_SEARCH_LIMIT
            )
            if timeline_results:
                logger.debug("Found %d decisions for timeline", len(timeline_results))
            return timeline_results
        return []

    def _run_founding_classification(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        episode_results: List[Any],
    ) -> tuple:
        """
        Stage 2D: Classify episodes as founding/follow-up for creation queries.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param episode_results: Episodes to classify
        :return: Tuple of (founding_analysis dict or None, is_creation_query bool)
        """
        creation_words = [
            "creates",
            "created",
            "establishes",
            "established",
            "set up",
            "sets up",
            "launches",
            "launched",
            "founded",
        ]
        is_creation_query = query_analysis.get("temporal_direction") == "earliest" or (
            query_analysis.get("is_identification_query")
            and any(w in query.lower() for w in creation_words)
        )
        founding_analysis = None
        if is_creation_query and episode_results:
            founding_analysis = self._classify_episodes_founding(episode_results)
            logger.debug("Founding analysis: %s", founding_analysis["summary"])
        return founding_analysis, is_creation_query

    async def _run_relationship_search(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        entity_results: List[Any],
        limit: int,
    ) -> List[Any]:
        """
        Stage 3: Search relationships if the query intent or entity results warrant it.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param entity_results: Entity results from stage 1
        :param limit: Base limit from caller
        :return: Relationship search results (may be empty list)
        """
        if query_analysis["intent"] == "requirement" or len(entity_results) > 0:
            rel_limit = max(
                self.RELATIONSHIP_LIMIT_MIN,
                limit // self.RELATIONSHIP_LIMIT_DIVISOR,
            )
            logger.debug("Stage 3: Searching relationships (limit=%d)", rel_limit)
            return await self._search_relationships(query, rel_limit)
        return []

    async def _assemble_multi_stage_output(
        self,
        query: str,
        query_analysis: Dict[str, Any],
        decision_results: str,
        paragraph_results: List[Any],
        founding_analysis: Optional[Dict[str, Any]],
        is_creation_query: bool,
        referenced_decisions: List[Any],
        timeline_results: List[Any],
        episode_results: List[Any],
        entity_results: List[Any],
        relationship_results: List[Any],
    ) -> str:
        """
        Assemble prioritized output from all search stages into formatted text.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param decision_results: Specific decision results
        :param paragraph_results: Specific paragraph results
        :param founding_analysis: Founding/follow-up classification (or None)
        :param is_creation_query: Whether this is a creation/founding query
        :param referenced_decisions: Backward reference results
        :param timeline_results: Topic timeline results
        :param episode_results: Primary source episode results
        :param entity_results: Entity search results
        :param relationship_results: Relationship search results
        :return: Formatted output string
        """
        logger.debug("Synthesizing multi-stage results...")
        conf_filter = query_analysis.get("conference_filter")
        key_concepts_preview = (
            ", ".join(query_analysis["key_concepts"][: self.MAX_KEY_CONCEPTS_DISPLAY])
            if query_analysis["key_concepts"]
            else "None detected"
        )
        output_parts = [
            f"MULTI-STAGE SEARCH RESULTS FOR: '{query}'",
            f"Query Complexity: {query_analysis['complexity'].upper()}",
            f"Query Intent: {query_analysis['intent'].upper()}",
            f"Key Concepts: {key_concepts_preview}",
            f"Conference Filter: {conf_filter if conf_filter else 'None (all conferences)'}",
            "",
            "=" * 80,
            "",
        ]

        if decision_results:
            output_parts.append("=" * 80)
            output_parts.append("SPECIFIC DECISION (EXACT MATCH)")
            output_parts.append(
                "Direct result for the decision referenced in your query"
            )
            output_parts.append("=" * 80)
            output_parts.append("")
            output_parts.append(decision_results)
            output_parts.append("")

        if paragraph_results:
            output_parts.append("=" * 80)
            output_parts.append("SPECIFIC PARAGRAPHS (EXTRACTED FROM METADATA)")
            output_parts.append(
                "These are the exact paragraphs referenced in your query"
            )
            output_parts.append("=" * 80)
            output_parts.append("")
            for para_result in paragraph_results:
                output_parts.append(para_result)
                output_parts.append("")

        if founding_analysis and founding_analysis.get("founding_episodes"):
            output_parts.append("=" * 80)
            output_parts.append(">>> FOUNDING DECISION(S) IDENTIFIED <<<")
            output_parts.append(
                "These decisions contain CREATION language"
                " (establishes, creates, sets up)."
            )
            output_parts.append(
                "Use these to answer 'which decision CREATED/ESTABLISHED X' questions."
            )
            output_parts.append("=" * 80)
            output_parts.append("")
            founding_text = await self._format_episodes(
                query, founding_analysis["founding_episodes"]
            )
            output_parts.append(founding_text)
            output_parts.append("")

        if founding_analysis and founding_analysis.get("followup_episodes"):
            output_parts.append("=" * 80)
            output_parts.append(
                "FOLLOW-UP DECISIONS (these further develop/operationalize, NOT create)"
            )
            output_parts.append(
                "Do NOT cite these as the 'founding' or 'creation' decision."
            )
            output_parts.append("=" * 80)
            output_parts.append("")
            followup_text = await self._format_episodes(
                query, founding_analysis["followup_episodes"]
            )
            output_parts.append(followup_text)
            output_parts.append("")

        if referenced_decisions:
            output_parts.append("=" * 80)
            if is_creation_query:
                output_parts.append(
                    "EARLIER REFERENCED DECISIONS (POTENTIAL FOUNDING DECISIONS)"
                )
                output_parts.append(
                    "These were recalled/referenced by later decisions."
                )
                output_parts.append(
                    "CHECK THESE FIRST — founding decision is often referenced by later follow-ups."
                )
            else:
                output_parts.append(
                    "EARLIER REFERENCED DECISIONS (TRACED VIA BACKWARD REFERENCES)"
                )
                output_parts.append(
                    "These were recalled/referenced by the results above"
                )
            output_parts.append("=" * 80)
            output_parts.append("")
            for ref_decision in referenced_decisions:
                output_parts.append(ref_decision)
                output_parts.append("")

        if timeline_results:
            output_parts.append("=" * 80)
            output_parts.append("COMPLETE DECISION TIMELINE ON THIS TOPIC")
            output_parts.append(
                "All decisions found, sorted chronologically (earliest first)"
            )
            output_parts.append("=" * 80)
            output_parts.append("")
            for tl_result in timeline_results:
                output_parts.append(tl_result)
                output_parts.append("")

        if episode_results and not founding_analysis:
            output_parts.append("=" * 80)
            output_parts.append("PRIMARY SOURCE DOCUMENTS (HIGHEST PRIORITY)")
            output_parts.append("These contain the EXACT TEXT from UNFCCC documents")
            output_parts.append("=" * 80)
            output_parts.append("")
            episode_text = await self._format_episodes(query, episode_results)
            output_parts.append(episode_text)
            output_parts.append("")

        if entity_results:
            output_parts.append("=" * 80)
            output_parts.append("RELEVANT ENTITIES AND CONCEPTS")
            output_parts.append("=" * 80)
            output_parts.append("")
            entity_text = await self._format_entities(query, entity_results)
            output_parts.append(entity_text)
            output_parts.append("")

        if relationship_results:
            output_parts.append("=" * 80)
            output_parts.append("RELATIONSHIPS AND CONNECTIONS")
            output_parts.append("=" * 80)
            output_parts.append("")
            rel_text = await self._format_relationships(query, relationship_results)
            output_parts.append(rel_text)
            output_parts.append("")

        if not any(
            [
                decision_results,
                paragraph_results,
                referenced_decisions,
                timeline_results,
                episode_results,
                entity_results,
                relationship_results,
            ]
        ):
            output_parts.append("No results found across all search stages.")
            output_parts.append("")
            output_parts.append("Suggestions:")
            output_parts.append("- Try broader search terms")
            output_parts.append("- Check spelling of technical terms")
            output_parts.append("- Verify the concept exists in UNFCCC documents")

        formatted_output = "\n".join(output_parts)

        if len(formatted_output) > self.MAX_OUTPUT_CHARS:
            logger.debug(
                "Multi-stage output too large (%d chars). Truncating.",
                len(formatted_output),
            )
            formatted_output = self._truncate_content(
                formatted_output, self.MAX_OUTPUT_CHARS
            )

        return formatted_output

    def _expand_query_terms(self, query: str) -> str:
        """
        Expand query with UNFCCC synonyms and related terms for better document retrieval.

        Maps common terms to official UNFCCC terminology to improve matching.

        :param query: Original query string
        :return: Expanded query string with additional terms
        """
        query_lower = query.lower()
        expansions = []

        for key_phrase, synonyms in self.QUERY_TERM_MAPPINGS.items():
            if key_phrase in query_lower:
                expansions.extend(synonyms[: self.MAX_SYNONYM_EXPANSIONS])

        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            return expanded
        return query

    def _filter_by_conference(
        self, results: List[Any], conference_type: str
    ) -> List[Any]:
        """Filter episode results to only include episodes from the specified conference type.

        Checks episode metadata 'conference_type' and episode name/decision_id for
        conference type markers (CMA, COP/CP, CMP, SBI, SBSTA).

        :param results: List of episode results to filter
        :param conference_type: Conference type to filter by (e.g., "CMA", "COP", "CMP")
        :return: Filtered list of results matching the conference type
        """
        target_aliases = self.CONF_ALIASES.get(conference_type, [conference_type])

        filtered = []
        for episode in results:
            metadata = getattr(episode, "metadata", None) or {}

            meta_conf = metadata.get("conference_type", "").upper()
            if meta_conf in target_aliases:
                filtered.append(episode)
                continue

            decision_id = metadata.get("decision_id", "").upper()
            if any(alias in decision_id for alias in target_aliases):
                filtered.append(episode)
                continue

            name = (getattr(episode, "name", "") or "").upper()
            if any(alias in name for alias in target_aliases):
                filtered.append(episode)
                continue

        if not filtered and results:
            logger.warning(
                "Conference filter '%s' removed all results. Keeping originals.",
                conference_type,
            )
            return results

        return filtered

    def _temporal_rerank(self, results: List[Any], direction: str) -> List[Any]:
        """Re-rank episode results by year for chronological queries.

        :param results: List of episode results to re-rank
        :param direction: 'earliest' to sort ascending, 'latest' to sort descending
        :return: Re-ranked list of results
        """

        def get_year(episode) -> int:
            metadata = getattr(episode, "metadata", None) or {}
            year_str = metadata.get("year", "")
            if year_str:
                try:
                    return int(year_str)
                except (ValueError, TypeError):
                    pass
            name = getattr(episode, "name", "") or ""
            year_match = re.search(r"(\d{4})", name)
            if year_match:
                y = int(year_match.group(1))
                if self.YEAR_RANGE_MIN <= y <= self.YEAR_RANGE_MAX:
                    return y
            return (
                self.TEMPORAL_SORT_EARLIEST_DEFAULT
                if direction == "earliest"
                else self.TEMPORAL_SORT_LATEST_DEFAULT
            )

        reverse = direction != "earliest"
        return sorted(results, key=get_year, reverse=reverse)

    def _classify_episodes_founding(self, episode_results: List[Any]) -> Dict[str, Any]:
        """Classify episodes as founding or follow-up based on creation vs operational language.

        Scans episode content for creation verbs (establishes, creates, sets up) and
        follow-up verbs (further develops, also recalling, reaffirms) to separate
        founding decisions from follow-up decisions.

        :param episode_results: List of episode results to classify
        :return: Dict with 'founding_episodes', 'followup_episodes', and 'summary'
        """
        creation_re = self.CREATION_RE
        followup_re = self.FOLLOWUP_RE

        founding = []
        followup = []

        for episode in episode_results:
            content = getattr(episode, "content", "") or ""
            check_text = content[: self.FOUNDING_CONTENT_CHECK_CHARS]

            creation_count = len(creation_re.findall(check_text))
            followup_count = len(followup_re.findall(check_text))

            metadata = getattr(episode, "metadata", None) or {}
            decision_id = metadata.get("decision_id", "")
            year = metadata.get("year", "?")
            name = getattr(episode, "name", "") or ""

            if creation_count > 0 and creation_count >= followup_count:
                founding.append(episode)
                name_preview = name[: self.LOG_DECISION_NAME_CHARS]
                logger.debug(
                    "  FOUNDING: Decision %s (%s) - %s [creation=%d, followup=%d]",
                    decision_id,
                    year,
                    name_preview,
                    creation_count,
                    followup_count,
                )
            else:
                followup.append(episode)
                name_preview = name[: self.LOG_DECISION_NAME_CHARS]
                logger.debug(
                    "  FOLLOW-UP: Decision %s (%s) - %s [creation=%d, followup=%d]",
                    decision_id,
                    year,
                    name_preview,
                    creation_count,
                    followup_count,
                )

        founding = self._temporal_rerank(founding, "earliest") if founding else []
        followup = self._temporal_rerank(followup, "earliest") if followup else []

        founding_ids = [
            getattr(e, "metadata", {}).get("decision_id", "?")
            for e in founding
            if hasattr(e, "metadata")
        ]
        followup_ids = [
            getattr(e, "metadata", {}).get("decision_id", "?")
            for e in followup
            if hasattr(e, "metadata")
        ]

        summary = (
            f"Found {len(founding)} founding decision(s): {founding_ids} "
            f"and {len(followup)} follow-up decision(s): {followup_ids}"
        )

        return {
            "founding_episodes": founding,
            "followup_episodes": followup,
            "summary": summary,
        }

    async def _follow_backward_references(
        self, episode_results: List[Any]
    ) -> List[str]:
        """Follow 'recalling' references in episode content to find earlier founding decisions.

        Parses episode content for backward references like "recalling decision X/Y.Z"
        and fetches those earlier decisions to help identify founding decisions.

        :param episode_results: List of episode results to scan for backward references
        :return: List of formatted decision result strings
        """
        referenced_decision_ids = set()
        for episode in episode_results:
            content = getattr(episode, "content", "") or ""
            recall_patterns = [
                r"(?:recalling|recalls|also recalling|further recalling)\s+"
                r"(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)",
                r"(?:pursuant to|in accordance with)\s+"
                r"(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)",
            ]
            for pattern in recall_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    ref_id = match.group(1).upper()
                    referenced_decision_ids.add(ref_id)

        results = []
        for decision_id in list(referenced_decision_ids)[
            : self.MAX_BACKWARD_REFERENCES
        ]:
            logger.debug(
                "  Following backward reference to Decision %s...", decision_id
            )
            result = await self._search_by_decision(decision_id)
            if result:
                results.append(result)

        return results

    async def _search_topic_timeline(
        self, query: str, query_analysis: Dict[str, Any], limit: int = 20
    ) -> List[str]:
        """Search for ALL decisions on a topic and present them chronologically.

        Performs a broad episode search, classifies each decision as founding or
        follow-up based on creation vs operational language, and returns a sorted
        timeline of all decisions found.

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param limit: Maximum number of episodes to search
        :return: List of formatted timeline entry strings, sorted chronologically
        """
        episode_results = await self._search_episodes(query, limit)
        if not episode_results:
            return []

        conf_filter = query_analysis.get("conference_filter")
        if conf_filter:
            episode_results = self._filter_by_conference(episode_results, conf_filter)

        decisions = []
        seen_ids = set()
        for episode in episode_results:
            metadata = getattr(episode, "metadata", None) or {}
            decision_id = metadata.get("decision_id", "")
            if not decision_id or decision_id in seen_ids:
                name = getattr(episode, "name", "") or ""
                name_match = re.search(
                    r"(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+))",
                    name,
                    re.IGNORECASE,
                )
                if name_match:
                    decision_id = name_match.group(1).strip()
                else:
                    continue
            if decision_id in seen_ids:
                continue
            seen_ids.add(decision_id)

            year_str = metadata.get("year", "")
            year = int(year_str) if year_str else self.TEMPORAL_SORT_EARLIEST_DEFAULT
            content = getattr(episode, "content", "") or ""

            creation_count = len(
                self.CREATION_RE.findall(content[: self.TIMELINE_CONTENT_CHECK_CHARS])
            )
            followup_count = len(
                self.FOLLOWUP_RE.findall(content[: self.TIMELINE_CONTENT_CHECK_CHARS])
            )
            if creation_count > 0 and creation_count >= followup_count:
                action_type = "FOUNDING"
            elif followup_count > creation_count:
                action_type = "FOLLOW-UP"
            else:
                action_type = "RELATED"

            first_para = content[: self.FIRST_PARAGRAPH_CHARS].split("\n")
            summary_line = ""
            for line in first_para:
                line = line.strip()
                if len(line) > self.MIN_SUMMARY_LINE_LENGTH and not line.startswith(
                    "Conference"
                ):
                    summary_line = line[: self.LOG_SUMMARY_LINE_CHARS]
                    break

            conference = metadata.get("conference_type", "")
            session = metadata.get("session_number", "")
            location = metadata.get("location", "")
            formatted = (
                f"[{year}] Decision {decision_id} ({conference} Session {session}"
                f"{', ' + location if location else ''}) "
                f"— {action_type}: {summary_line}"
            )
            decisions.append((year, formatted))

        decisions.sort(key=lambda x: x[0])

        backward_ids = set()
        for episode in episode_results:
            content = getattr(episode, "content", "") or ""
            for match in re.finditer(
                r"(?:recalling|also recalling)\s+(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)",
                content,
                re.IGNORECASE,
            ):
                ref_id = match.group(1).upper()
                if ref_id not in seen_ids:
                    backward_ids.add(ref_id)

        for ref_id in list(backward_ids)[: self.MAX_BACKWARD_REFERENCES]:
            result = await self._search_by_decision(ref_id)
            if result:
                earlier_text = (
                    f"[EARLIER] Decision {ref_id} — Referenced by later "
                    f"decisions (search for full text)\n"
                    f"{result[: self.LOG_RESULT_PREVIEW_CHARS]}"
                )
                decisions.insert(0, (0, earlier_text))

        return [d[1] for d in decisions]
