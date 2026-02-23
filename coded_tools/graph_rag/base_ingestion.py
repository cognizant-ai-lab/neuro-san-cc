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
Base class for UNFCCC Climate Document Ingestion.

Provides the ingestion pipeline: database connectivity, episode loading,
checkpoint-based resume, progress logging, and demonstration query logic.
Document parsing is inherited from DocumentParser.

Subclasses must implement:
    - DB_NAME: Class attribute with the database display name
    - _connection_config(): Returns database-specific config keys
    - _log_connection_info(): Logs database-specific connection details
    - _create_graphiti_client(): Creates and returns a configured Graphiti instance
"""

from __future__ import annotations

import logging
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import graphiti_core.prompts.extract_edges as _extract_edges_module
import graphiti_core.prompts.extract_nodes as _extract_nodes_module
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from neuro_san.interfaces.coded_tool import CodedTool

from .document_parser import DocumentParser

try:
    from . import constrained_prompts as _constrained_prompts
except ImportError:
    _constrained_prompts = None  # type: ignore[assignment]

_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")


class BaseIngestionTool(DocumentParser, CodedTool):
    """Base class for UNFCCC climate document ingestion into a knowledge graph.

    Inherits document parsing from DocumentParser and provides the ingestion
    pipeline: database connectivity, episode loading with checkpoint-based resume,
    progress logging, and demonstration queries. Subclasses provide database-specific
    connection and client logic.

    Features:
        - All document parsing capabilities (via DocumentParser)
        - Checkpoint-based resume capability for interrupted ingestion
        - Demonstration queries for verifying graph contents
        - Progress logging and statistics

    Note:
        Reference linking, annex linking, and conference hierarchy features are currently
        disabled to optimize processing performance and reduce episode count.
    """

    DB_NAME = ""

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
        """Initializes the ingestion tool with default configuration.

        Sets up logging, loads configuration from environment variables, and
        initializes data structures for tracking decisions and references during ingestion.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler()]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config: Dict[str, Any] = self._load_config()
        self.decision_registry: Dict[str, str] = {}
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
        del sly_data
        args = args or {}
        if args:
            self.config = self._load_config(args)
        summary = await self.run()
        return (
            f"{self.DB_NAME} ingestion completed: "
            f"{summary['success']} succeeded, {summary['failed']} failed "
            f"out of {summary['prepared']} prepared episodes. "
            f"{summary['references_extracted']} references extracted and stored in metadata."
        )

    async def run(self) -> Dict[str, int]:
        """Executes complete document ingestion pipeline.

        Connects to the configured database, processes all documents in the configured
        data directory, extracts structured information and references, optionally runs
        demonstration queries, and returns processing statistics.

        Returns:
            Dictionary with processing statistics:
                - prepared: Total episodes created from documents
                - success: Episodes successfully added to graph
                - failed: Episodes that failed to process
                - references_extracted: Total cross-document references found
        """
        self._log_connection_info()
        graphiti = self._create_graphiti_client()
        self.logger.info("Successfully connected to %s", self.DB_NAME)

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

    def _log_connection_info(self) -> None:
        """Logs database-specific connection information.

        Subclasses must override to log their connection details
        (host, port, URI, etc.).
        """
        raise NotImplementedError

    def _connection_config(self) -> Dict[str, Any]:
        """Returns database-specific configuration keys.

        Subclasses must override to provide connection settings
        (e.g., host/port for FalkorDB, URI/user/password for Neo4j).

        Returns:
            Dictionary of database connection configuration.
        """
        raise NotImplementedError

    def _create_graphiti_client(self) -> Graphiti:
        """Creates and configures a Graphiti client for the target database.

        Subclasses must override to create the appropriate Graphiti client
        with database-specific driver configuration.

        Returns:
            Configured Graphiti client instance ready for use.
        """
        raise NotImplementedError

    def _apply_constrained_prompts(self) -> None:
        """Applies constrained prompts to prevent entity hallucinations.

        Replaces default Graphiti entity and edge extraction prompts with
        UNFCCC-domain-specific versions that only extract explicitly mentioned
        entities and relationships.
        """
        if _constrained_prompts is None:
            self.logger.warning("constrained_prompts module not available")
            self.logger.warning(
                "Continuing with default Graphiti prompts (may cause hallucinations)"
            )
            return
        try:
            _extract_nodes_module.extract_text = (
                _constrained_prompts.extract_text_constrained
            )
            _extract_edges_module.edge = _constrained_prompts.extract_edges_constrained
            self.logger.info("Applied constrained prompts to prevent hallucinations")
        except AttributeError as exc:
            self.logger.warning("Failed to apply constrained prompts: %s", exc)
            self.logger.warning(
                "Continuing with default Graphiti prompts (may cause hallucinations)"
            )

    def _load_config(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Loads configuration from environment variables with optional overrides.

        Merges database-specific connection settings from _connection_config() with
        shared processing parameters. Accepts runtime overrides for flexible configuration.

        Args:
            overrides: Optional dictionary of configuration overrides.

        Returns:
            Complete configuration dictionary with connection and processing settings.
        """
        overrides = overrides or {}

        config: Dict[str, Any] = {
            **self._connection_config(),
            "data_dir": Path(os.environ.get("DATA_DIR", "documents")),
            "batch_size": int(os.environ.get("BATCH_SIZE", "10")),
            "max_episodes": int(os.environ.get("MAX_EPISODES", "0")),
            "search_limit": int(os.environ.get("SEARCH_LIMIT", "5")),
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

    async def _add_episodes(
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
                len(processed_episodes),
            )
            self.logger.info("Will skip these and continue from where you left off\n")

        skipped_count = 0
        for idx, episode in enumerate(episodes, 1):
            episode_name = episode["name"]

            if episode_name in processed_episodes:
                skipped_count += 1
                if skipped_count % 50 == 0:
                    self.logger.info(
                        "Skipped %d already-processed episodes...", skipped_count
                    )
                continue

            try:
                self.logger.info(
                    "\n[%d/%d] Processing episode: %s",
                    idx,
                    len(episodes),
                    episode_name,
                )
                self.logger.info("    Source: %s", episode["source_description"])

                await graphiti.add_episode(
                    name=episode_name,
                    episode_body=episode["episode_body"],
                    source=EpisodeType.text,
                    source_description=episode["source_description"],
                    reference_time=episode["reference_time"],
                )
                success_count += 1
                self.logger.info("    ✓ Successfully added")
                await self._post_add_episode_metadata(
                    graphiti, episode_name, episode.get("metadata", {})
                )

                with open(checkpoint_file, "a", encoding="utf-8") as f:
                    f.write(f"{episode_name}\n")

                if idx % self.config["batch_size"] == 0:
                    self._log_progress(idx, len(episodes), start_time)
                elif self.config["verbose_logging"]:
                    self.logger.info(
                        "Added episode %d/%d: %s",
                        idx,
                        len(episodes),
                        episode_name[:100],
                    )
            except Exception as exc:  # pylint: disable=broad-exception-caught
                failure = (episode_name, str(exc))
                failures.append(failure)
                self.logger.error("Failed to add episode %s: %s", episode_name, exc)

        self._log_summary(len(episodes), success_count, failures, start_time)
        return success_count, failures

    async def _post_add_episode_metadata(
        self, graphiti: Graphiti, episode_name: str, metadata: Dict[str, Any]
    ) -> None:
        """Hook called after each episode is successfully added to the graph.

        Override in subclasses to persist custom metadata properties onto the
        newly created graph node.

        Args:
            graphiti: The Graphiti client instance (provides database driver).
            episode_name: The name of the episode that was just added.
            metadata: Episode metadata dict (conference_type, year, decision_id, etc.).
        """

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
                node_search_results = await graphiti._search(  # pylint: disable=protected-access
                    query=query,
                    config=node_search_config,
                )
                if node_search_results.nodes:
                    for node in node_search_results.nodes:
                        self.logger.info("\n  Node: %s", node.name)
                        summary = (node.summary or "").strip()
                        summary = (
                            summary[:150] + "..." if len(summary) > 150 else summary
                        )
                        self.logger.info("  Summary: %s", summary)
                        self.logger.info("  Labels: %s", ", ".join(node.labels))
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
        del graphiti
        self._print_section("REFERENCE TRAVERSAL DEMO", double_space=True)

        sample_count = 0
        for episode_name, refs in list(self.reference_map.items())[:3]:
            self.logger.info("\n--- Episode: %s ---", episode_name[:80])
            self.logger.info("References %d other decisions/articles:", len(refs))
            for ref in refs[:5]:
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

    async def _demo_temporal_queries(self, graphiti: Graphiti) -> None:
        """Demonstrates temporal query capabilities for tracking decision evolution.

        Shows how to query decisions by year, analyze temporal ranges, visualize
        decision timelines, and track how climate policy approaches evolved over time.
        Includes examples of querying decision trends from specific time periods.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("TEMPORAL QUERY DEMO", double_space=True)

        decisions_by_year: Dict[str, List[str]] = defaultdict(list)
        for decision_id, episode_name in self.decision_registry.items():
            if "::" in episode_name:
                filename_part = episode_name.split("::")[0]
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

        self._print_section("TEMPORAL RANGE QUERY DEMO")

        start_year = "2020"
        end_year = "2022"
        decisions_in_range = []
        for year in sorted_years:
            if start_year <= year <= end_year:
                decisions_in_range.extend(decisions_by_year[year])

        if decisions_in_range:
            self.logger.info(
                "\nDecisions from %s to %s: %d total",
                start_year,
                end_year,
                len(decisions_in_range),
            )
            self.logger.info("\nSample decisions from this period:")
            for decision_id in decisions_in_range[:10]:
                self.logger.info("  • %s", decision_id)
            if len(decisions_in_range) > 10:
                self.logger.info("  ... and %d more", len(decisions_in_range) - 10)

        self._print_section("TEMPORAL EVOLUTION QUERY")

        if len(sorted_years) >= 2:
            self.logger.info("\nTimeline: %s to %s", sorted_years[0], sorted_years[-1])
            self.logger.info(
                "Total span: %d years", int(sorted_years[-1]) - int(sorted_years[0]) + 1
            )
            self.logger.info("\nDecisions per year:")
            for year in sorted_years:
                count = len(decisions_by_year[year])
                bar_chart = "█" * min(count, 50)
                self.logger.info("  %s: %s (%d)", year, bar_chart, count)

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

        for query in temporal_queries[:1]:
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

    async def _demo_paragraph_access(self, graphiti: Graphiti) -> None:
        """Demonstrates paragraph-level granular access and citation resolution.

        Shows how the system provides direct access to specific paragraphs within
        decisions, enabling precise citation resolution (e.g., 'paragraph 69 of
        decision 1/CP.21') without searching full text. Demonstrates the paragraph
        indexing functionality that enables efficient granular retrieval.

        Args:
            graphiti: The Graphiti client instance for executing queries.
        """
        self._print_section("PARAGRAPH-LEVEL ACCESS DEMO", double_space=True)

        episodes_with_paragraphs = []
        for decision_id, episode_name in self.decision_registry.items():
            refs = self.reference_map.get(episode_name, [])
            has_para_refs = any(ref.get("paragraphs") for ref in refs)
            if has_para_refs:
                episodes_with_paragraphs.append((decision_id, episode_name, refs))

        if not episodes_with_paragraphs:
            self.logger.info("\nNo paragraph-level references found in demo.")
            return

        self.logger.info(
            "\nFound %d decisions with paragraph-level references",
            len(episodes_with_paragraphs),
        )

        self._print_section("EXAMPLE: Decisions Referencing Specific Paragraphs")

        for decision_id, episode_name, refs in episodes_with_paragraphs[:3]:
            self.logger.info("\n--- Decision: %s ---", decision_id)

            para_refs = [ref for ref in refs if ref.get("paragraphs")]
            if para_refs:
                self.logger.info("This decision references specific paragraphs in:")
                for ref in para_refs[:5]:
                    target = ref.get("target", "unknown")
                    paras = ref.get("paragraphs", "")
                    self.logger.info("  • %s, paragraph(s) %s", target, paras)

        self._print_section("EXAMPLE PARAGRAPH-LEVEL QUERY")

        if episodes_with_paragraphs:
            decision_id, episode_name, refs = episodes_with_paragraphs[0]
            para_ref = next((ref for ref in refs if ref.get("paragraphs")), None)

            if para_ref:
                target = para_ref.get("target")
                paras = para_ref.get("paragraphs")
                query = f"What does paragraph {paras} of decision {target} say?"

                self.logger.info("\nQuery: %s", query)
                try:
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
