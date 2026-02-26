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

from __future__ import annotations

import asyncio
import os
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

from dotenv import load_dotenv

_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

from graphiti_core import Graphiti

from .base_ingestion import BaseIngestionTool


class Neo4jIngestionEnhanced(BaseIngestionTool):
    """Ingests UNFCCC climate documents into Neo4j knowledge graph.

    Extends BaseIngestionTool with Neo4j-specific driver configuration and
    decision action classification. Classifies each decision as 'founding'
    (establishes new mechanisms), 'follow_up' (builds on prior work), or
    'neutral' using verb pattern analysis.

    Features:
        - All base ingestion capabilities (document parsing, reference extraction, etc.)
        - Neo4j driver connection via URI/user/password
        - Decision action type classification (founding/follow_up/neutral)
        - Decision type metadata in episode headers
    """

    DB_NAME = "Neo4j"

    CREATION_VERBS_RE = re.compile(
        r"(?i)\b(establishes?|creates?|decides\s+to\s+establish|"
        r"decides\s+to\s+create|launches?|inaugurates?|sets?\s+up)\b"
    )
    FOLLOWUP_VERBS_RE = re.compile(
        r"(?i)\b(further\s+develops?|also\s+recalling|"
        r"builds?\s+on|welcomes?\s+the\s+continued|reaffirms?|"
        r"decides\s+that\s+.{5,40}shall\s+have)\b"
    )

    def _connection_config(self) -> Dict[str, Any]:
        """Returns Neo4j-specific connection configuration.

        Reads URI, user, and password from environment variables.

        Returns:
            Dictionary with Neo4j connection settings.
        """
        return {
            "neo4j_uri": os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
            "neo4j_user": os.environ.get("NEO4J_USER", "neo4j"),
            "neo4j_password": os.environ.get("NEO4J_PASSWORD", "password"),
        }

    def _log_connection_info(self) -> None:
        """Logs Neo4j connection details (URI)."""
        self.logger.info(
            "Connecting to Neo4j database at %s",
            self.config["neo4j_uri"],
        )

    def _create_graphiti_client(self) -> Graphiti:
        """Creates and configures a Graphiti client for Neo4j.

        Initializes the Graphiti framework with Neo4j connection details and applies
        constrained prompts to prevent entity hallucinations during graph construction.

        Returns:
            Configured Graphiti client instance ready for use.
        """
        self._apply_constrained_prompts()

        return Graphiti(
            self.config["neo4j_uri"],
            self.config["neo4j_user"],
            self.config["neo4j_password"],
        )

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

    def _post_extract_decision_metadata(
        self, enriched_metadata: Dict[str, Any], section_body: str
    ) -> None:
        """Enriches decision metadata with action type classification.

        Classifies the decision as founding, follow-up, or neutral and adds
        the classification to the metadata dictionary.

        Args:
            enriched_metadata: Mutable metadata dictionary to enrich.
            section_body: Full text of the decision section.
        """
        enriched_metadata["decision_action_type"] = self._classify_decision_action(
            section_body
        )

    def _extra_header_lines(self, metadata: Dict[str, str]) -> List[str]:
        """Adds decision action type to episode body headers.

        Appends the decision type (founding/follow_up/neutral) header line
        when the metadata includes decision action classification.

        Args:
            metadata: Metadata dictionary with conference information.

        Returns:
            List containing the decision type header line, or empty list.
        """
        lines: List[str] = []
        if metadata.get("decision_action_type"):
            lines.append(f"Decision Type: {metadata['decision_action_type']}")
        return lines

    async def _post_add_episode_metadata(
        self, graphiti: Graphiti, episode_name: str, metadata: Dict[str, Any]
    ) -> None:
        """Persists UNFCCC metadata as properties on the Neo4j Episodic node.

        Runs a Cypher SET query to store structured metadata (conference_type,
        year, decision_id, etc.) directly on the Episodic node so that
        graph_search_tool._get_episode_metadata() can retrieve them later.

        Args:
            graphiti: Graphiti client (provides .driver for direct Cypher access).
            episode_name: Name of the episode node to update.
            metadata: Metadata dict from the episode (may contain None values).
        """
        props = {
            "decision_id": metadata.get("decision_id"),
            "conference_type": metadata.get("conference_type"),
            "year": metadata.get("year"),
            "session_number": metadata.get("session_number"),
            "annex_id": metadata.get("annex_id"),
            "decision_action_type": metadata.get("decision_action_type"),
            "chunk_index": metadata.get("chunk_index"),
            "total_chunks": metadata.get("total_chunks"),
        }
        props = {k: v for k, v in props.items() if v is not None}
        if not props:
            return
        try:
            await graphiti.driver.execute_query(
                "MATCH (e:Episodic {name: $name}) SET e += $props",
                name=episode_name,
                props=props,
            )
        except Exception as exc:
            self.logger.warning(
                "Failed to set metadata for episode '%s': %s",
                episode_name[:80],
                exc,
            )

    async def _delete_from_neo4j(self, graphiti: Graphiti, doc_name: str) -> int:
        """Delete all Episodic nodes for a document from Neo4j.

        Uses DETACH DELETE so all edges attached to each Episodic node are also
        removed. Entity nodes are intentionally left in place — they may be shared
        across multiple documents.

        Args:
            graphiti: Graphiti client (provides .driver for direct Cypher access).
            doc_name: Document stem (filename without .txt extension).

        Returns:
            Number of Episodic nodes deleted.
        """
        prefix = doc_name + "::"
        try:
            result = await graphiti.driver.execute_query(
                """
                MATCH (e:Episodic)
                WHERE e.name STARTS WITH $prefix
                WITH count(e) AS total, collect(e) AS nodes
                FOREACH (e IN nodes | DETACH DELETE e)
                RETURN total
                """,
                prefix=prefix,
            )
            count = result.records[0]["total"] if result.records else 0
            return count
        except Exception as exc:
            self.logger.error(
                "Failed to delete Neo4j nodes for '%s': %s", doc_name, exc
            )
            raise

    async def delete_document(self, doc_name: str) -> None:
        """Delete a document's data from Neo4j and the ingestion checkpoint.

        Removes all Episodic nodes whose name starts with ``doc_name::`` and
        purges the corresponding checkpoint entries so the document can be
        re-ingested if needed.

        Args:
            doc_name: Document stem (filename without .txt extension).
        """
        graphiti = self._create_graphiti_client()
        try:
            deleted = await self._delete_from_neo4j(graphiti, doc_name)
            removed = self._remove_from_checkpoint(doc_name)
            self.logger.info(
                "Deleted %d Episodic node(s) and %d checkpoint entry/entries for '%s'",
                deleted,
                removed,
                doc_name,
            )
        finally:
            await graphiti.close()

    async def edit_document(self, doc_name: str) -> None:
        """Delete a document's existing data and re-ingest the updated file from DATA_DIR.

        Equivalent to running delete_document() followed by a targeted ingestion
        of the single file whose stem matches doc_name.

        Args:
            doc_name: Document stem (filename without .txt extension). The corresponding
                .txt file must exist in the configured DATA_DIR.
        """
        self.logger.info("Deleting existing data for '%s'...", doc_name)
        graphiti = self._create_graphiti_client()
        try:
            deleted = await self._delete_from_neo4j(graphiti, doc_name)
            removed = self._remove_from_checkpoint(doc_name)
            self.logger.info(
                "Removed %d Episodic node(s) and %d checkpoint entry/entries",
                deleted,
                removed,
            )
            self.logger.info("Re-ingesting '%s'...", doc_name)
            episodes = self.build_episodes(self.config["data_dir"], file_filter=doc_name)
            episodes = self._limit_episodes(episodes)
            self._log_episode_stats(episodes)
            await self._add_episodes(graphiti, episodes)
            self.logger.info("Edit complete for '%s'", doc_name)
        finally:
            await graphiti.close()

    async def migrate_metadata(self) -> None:
        """Backfills metadata properties onto existing Neo4j Episodic nodes.

        Use this once to update nodes that were ingested before this fix was
        applied. Re-parses all documents and runs SET for each matching node.
        Nodes that never existed in the graph are silently skipped (MATCH finds
        no rows).
        """
        graphiti = self._create_graphiti_client()
        episodes = self.build_episodes(self.config["data_dir"])
        self.logger.info(
            "migrate_metadata: updating %d episodes in Neo4j...", len(episodes)
        )
        updated = 0
        for episode in episodes:
            episode_name = episode["name"][: self.MAX_EPISODE_NAME_LEN]
            await self._post_add_episode_metadata(
                graphiti, episode_name, episode.get("metadata", {})
            )
            updated += 1
            if updated % 100 == 0:
                self.logger.info(
                    "  Migrated %d / %d episodes...", updated, len(episodes)
                )
        self.logger.info("migrate_metadata: done (%d episodes processed)", updated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Neo4j GraphRAG document manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python neo4j_ingestion.py
      Ingest all new documents from DATA_DIR (default behaviour).

  python neo4j_ingestion.py --delete CMA2016_1.1_Decisions_1_to_2
      Remove all Episodic nodes for that document from Neo4j and purge
      the corresponding checkpoint entries.

  python neo4j_ingestion.py --edit CMA2016_1.1_Decisions_1_to_2
      Delete existing data for the document, then re-ingest the updated
      .txt file from DATA_DIR.
        """,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--delete",
        metavar="DOC_NAME",
        help="Delete document nodes from Neo4j and remove from checkpoint",
    )
    group.add_argument(
        "--edit",
        metavar="DOC_NAME",
        help="Delete existing data and re-ingest updated document from DATA_DIR",
    )
    cli_args = parser.parse_args()

    tool = Neo4jIngestionEnhanced()
    if cli_args.delete:
        asyncio.run(tool.delete_document(cli_args.delete))
    elif cli_args.edit:
        asyncio.run(tool.edit_document(cli_args.edit))
    else:
        asyncio.run(tool.run())
