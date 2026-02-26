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


class ResultFormatter:
    """
    Result formatting and citation component for GraphSearchTool.

    Contains formatters for each result type (facts, entities, relationships,
    episodes), citation builders, and claim validation. All formatting
    constants are defined here.

    """

    # --- Content truncation ---
    DEFAULT_TRUNCATE_CHARS = 8000

    # --- Display limits ---
    MAX_KEY_CONCEPTS_DISPLAY = 5
    MAX_CITATION_KEY_TERMS = 5
    MAX_CITATION_EXCERPTS = 3
    MAX_CITATION_EPISODES = 3

    # --- Citation validation thresholds ---
    HIGH_CONFIDENCE_THRESHOLD = 0.6
    SUPPORT_CONFIDENCE_THRESHOLD = 0.3
    MIN_CLAIM_WORD_LENGTH = 3
    MIN_CLAIM_LENGTH = 20

    COMMON_STOP_WORDS = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "shall",
        "can",
    }

    @staticmethod
    def _truncate_content(content: str, max_chars: int = DEFAULT_TRUNCATE_CHARS) -> str:
        """Truncate content to fit within context limits."""
        if len(content) <= max_chars:
            return content
        truncation_msg = (
            "\n\n[... content truncated. Use a more specific query for full text ...]"
        )
        return content[:max_chars] + truncation_msg

    def _format_decision_result(self, decision_id: str, episode) -> str:
        """Format a found decision episode into a readable result."""
        episode_content = (
            episode.content if hasattr(episode, "content") else str(episode)
        )
        metadata = (
            episode.metadata
            if hasattr(episode, "metadata") and episode.metadata
            else {}
        )

        conference = metadata.get("conference_name", "Unknown")
        year = metadata.get("year", "Unknown")
        location = metadata.get("location", "Unknown")

        action_type = metadata.get("decision_action_type", "")
        action_label = f" [{action_type.upper()}]" if action_type else ""

        parts = [
            f">>> DECISION {decision_id} ({year}){action_label} <<<",
            f"Conference: {conference}",
            f"Year: {year}, Location: {location}",
            "",
            "EXACT TEXT FROM SOURCE:",
            "-" * 80,
            episode_content,
        ]

        return "\n".join(parts)

    async def _format_entities(self, query: str, results: List[Any]) -> str:
        """
        Format entity nodes with connected relationships.

        :param query: Original search query
        :param results: List of entity node results
        :return: Formatted string representation
        """
        if not results:
            return f"No entities found matching query: {query}"

        output_parts: List[str] = [
            f"Found {len(results)} relevant entities for query: '{query}'\n",
            "=" * 80,
            "",
        ]

        for i, node in enumerate(results, 1):
            name: str = node.name
            summary: str = getattr(node, "summary", "")
            labels: List[str] = getattr(node, "labels", [])
            entity_type: str = getattr(node, "entity_type", "")
            uuid: str = getattr(node, "uuid", "")

            output_parts.append(f"ENTITY {i}: {name}")

            if entity_type:
                output_parts.append(f"Type: {entity_type}")
            elif labels:
                output_parts.append(f"Type: {', '.join(labels)}")

            if summary:
                output_parts.append(f"\nDescription: {summary}")

            if uuid:
                connections: List[Dict[str, Any]] = await self._get_entity_connections(
                    uuid
                )
                if connections:
                    output_parts.append(
                        f"\nConnected to ({len(connections)} relationships):"
                    )
                    for conn in connections:
                        output_parts.append(
                            f"  - {conn['relationship']}: {conn['target_name']}"
                        )

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    async def _format_relationships(self, query: str, results: List[Any]) -> str:
        """
        Format relationship edges with connected entity details.

        :param query: Original search query
        :param results: List of relationship edge results
        :return: Formatted string representation
        """
        if not results:
            return f"No relationships found matching query: {query}"

        output_parts: List[str] = [
            f"Found {len(results)} relevant relationships for query: '{query}'\n",
            "=" * 80,
            "",
        ]

        for i, edge in enumerate(results, 1):
            fact: str = edge.fact
            name: str = getattr(edge, "name", "")
            source_uuid: Optional[str] = getattr(edge, "source_node_uuid", None)
            target_uuid: Optional[str] = getattr(edge, "target_node_uuid", None)

            output_parts.append(f"RELATIONSHIP {i}")
            if name:
                output_parts.append(f"Type: {name}")

            output_parts.append(f"Description: {fact}")

            if source_uuid and target_uuid:
                source_node: Dict[str, Any] = await self._get_node_by_uuid(source_uuid)
                target_node: Dict[str, Any] = await self._get_node_by_uuid(target_uuid)

                if source_node or target_node:
                    output_parts.append("\nConnected Entities:")
                    if source_node:
                        output_parts.append(
                            f"  From: {source_node.get('name', 'Unknown')}"
                        )
                        if source_node.get("summary"):
                            output_parts.append(f"        {source_node['summary']}")
                    if target_node:
                        output_parts.append(
                            f"  To: {target_node.get('name', 'Unknown')}"
                        )
                        if target_node.get("summary"):
                            output_parts.append(f"      {target_node['summary']}")

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    @staticmethod
    def _group_chunks(results: List[Any]) -> List[List[Any]]:
        """Group chunked episodes by their base name (without ``[N/M]`` suffix).

        Unchunked episodes form single-element groups.  Chunked episodes
        sharing the same base name are grouped together and sorted by
        ``chunk_index``.

        :param results: List of episode result objects.
        :return: List of groups (each group is a list of episodes).
        """
        groups: Dict[str, List[Any]] = {}
        order: List[str] = []
        for ep in results:
            name = getattr(ep, "name", "") or ""
            base_name = re.sub(r"\s*\[\d+/\d+\]$", "", name)
            if base_name not in groups:
                groups[base_name] = []
                order.append(base_name)
            groups[base_name].append(ep)

        for base_name in groups:
            groups[base_name].sort(
                key=lambda ep: (getattr(ep, "metadata", None) or {}).get(
                    "chunk_index", 0
                )
            )

        return [groups[base_name] for base_name in order]

    async def _format_episodes(self, query: str, results: List[Any]) -> str:
        """
        Format episode nodes with primary source content and metadata.

        If total content exceeds MAX_OUTPUT_CHARS, each episode's content
        is truncated to fit within context limits.

        :param query: Original search query
        :param results: List of episode node results
        :return: Formatted string representation with STRICT CITATIONS
        """
        if not results:
            return f"No primary source documents found matching query: {query}"

        # Group chunked episodes so they appear as a single logical result
        grouped = self._group_chunks(results)

        total_content_chars = sum(
            len(getattr(ep, "content", "") or "") for ep in results
        )
        needs_truncation = total_content_chars > self.MAX_OUTPUT_CHARS

        if needs_truncation and len(grouped) > 0:
            per_group_limit = self.MAX_OUTPUT_CHARS // len(grouped)
            logger.debug(
                "Total content: %d chars > %d limit. Truncating each group to ~%d chars.",
                total_content_chars,
                self.MAX_OUTPUT_CHARS,
                per_group_limit,
            )
        else:
            per_group_limit = 0

        output_parts: List[str] = [
            f"SEARCH RESULTS FOR: '{query}'",
            f"Found {len(grouped)} source document(s)\n",
            "=" * 80,
            "",
        ]

        for i, group in enumerate(grouped, 1):
            # Use first chunk for metadata / citation
            episode = group[0]
            name = getattr(episode, "name", "Unnamed Episode")
            # Strip chunk suffix from display name
            display_name = re.sub(r"\s*\[\d+/\d+\]$", "", name)
            source_description = getattr(episode, "source_description", "")

            ep_metadata = getattr(episode, "metadata", None) or {}
            db_metadata: Dict[str, Any] = await self._get_episode_metadata(
                getattr(episode, "uuid", "")
            )
            metadata = {**db_metadata, **{k: v for k, v in ep_metadata.items() if v}}

            citation = self._build_citation(display_name, metadata, source_description)

            decision_id = metadata.get("decision_id", "")
            annex_id = metadata.get("annex_id", "")
            year = metadata.get("year", "")
            conf_type = metadata.get("conference_type", "")
            action_type = metadata.get("decision_action_type", "")

            output_parts.append(f"[RESULT {i}]")
            if decision_id:
                header = f">>> DECISION {decision_id}"
                if year:
                    header += f" ({year})"
                if action_type:
                    header += f" [{action_type.upper()}]"
                header += " <<<"
                output_parts.append(header)
            if annex_id:
                output_parts.append(
                    f">>> ANNEX {annex_id} to Decision {decision_id} <<<"
                )
            output_parts.append(f"CITATION: {citation}")
            if conf_type:
                output_parts.append(f"Conference: {conf_type} {year}")
            output_parts.append("")
            output_parts.append("EXACT TEXT FROM SOURCE:")
            output_parts.append("-" * 80)

            # Concatenate content from all chunks in the group
            content = "\n".join(
                (getattr(ep, "content", "") or "").strip()
                for ep in group
                if (getattr(ep, "content", "") or "").strip()
            )

            if content:
                if needs_truncation:
                    output_parts.append(
                        self._truncate_content(content, per_group_limit)
                    )
                else:
                    output_parts.append(content)
            else:
                output_parts.append("[No content available in database]")

            output_parts.append("")
            output_parts.append("=" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    def _build_citation(
        self, episode_name: str, metadata: Dict[str, Any], source_description: str
    ) -> str:
        """
        Build a formal citation string for an episode.

        :param episode_name: The episode name from the database
        :param metadata: Episode metadata dictionary
        :param source_description: Human-readable source description
        :return: Formatted citation string
        """
        citation_parts = []

        decision_id = metadata.get("decision_id")
        if not decision_id and "::" in episode_name:
            title_part = episode_name.split("::", 1)[1] if "::" in episode_name else ""
            if "Decision" in title_part or "Resolution" in title_part:
                match = re.search(
                    r"(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:/[A-Za-z]+\.\d+)?)",
                    title_part,
                )
                if match:
                    decision_id = match.group(1)

        if decision_id:
            citation_parts.append(f"Decision {decision_id}")

        annex_id = metadata.get("annex_id")
        if annex_id:
            citation_parts.append(f"Annex {annex_id}")

        conf_type = metadata.get("conference_type")
        session = metadata.get("session_number")
        year = metadata.get("year")

        if conf_type and session and year:
            citation_parts.append(f"{conf_type} Session {session} ({year})")
        elif source_description:
            citation_parts.append(source_description)

        if citation_parts:
            return ", ".join(citation_parts)
        return f"Source: {episode_name}"
