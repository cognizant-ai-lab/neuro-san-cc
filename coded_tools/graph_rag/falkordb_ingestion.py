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
# pylint: disable=wrong-import-position

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any
from typing import Dict

from dotenv import load_dotenv

# Load environment variables before importing graphiti modules.
# The graphiti_core.embedder.client reads EMBEDDING_DIM at import time.
_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

# Monkey patch FalkorDB search functions to prevent query timeout issues.
import graphiti_core.search.search_filters as search_filters_module
import graphiti_core.search.search_utils as search_utils_module
from graphiti_core import Graphiti
from graphiti_core.driver.driver import GraphDriver
from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.edges import EntityEdge

from .base_ingestion import BaseIngestionTool

_original_edge_search_filter = (
    search_filters_module.edge_search_filter_query_constructor
)
_original_edge_fulltext_search = search_utils_module.edge_fulltext_search


def patched_edge_search_filter_query_constructor(
    filters, provider: GraphProvider
) -> tuple[list[str], dict[str, Any]]:
    """Prevents FalkorDB query timeouts by normalizing empty edge UUID lists.

    Converts empty edge_uuids lists to None to avoid inefficient query patterns
    that cause timeouts in FalkorDB when filtering with empty collections.

    Args:
        filters: Search filter object potentially containing edge_uuids list.
        provider: Graph database provider instance.

    Returns:
        Tuple of query parts and parameters for edge search filtering.
    """
    if (
        hasattr(filters, "edge_uuids")
        and filters.edge_uuids is not None
        and len(filters.edge_uuids) == 0
    ):
        filters.edge_uuids = None
    return _original_edge_search_filter(filters, provider)


async def patched_edge_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter,
    group_ids: list[str] | None = None,
    limit: int = 10,
) -> list[EntityEdge]:
    """Optimizes edge search performance by using direct lookups instead of full-text search.

    For large knowledge graphs, full-text search becomes prohibitively slow. This
    patched version uses direct UUID-based lookups when edge_uuids are provided,
    falling back to empty results for unrestricted searches to maintain performance.

    Args:
        driver: Graph database driver instance.
        query: Search query string (used only in fallback).
        search_filter: Filter object with optional edge_uuids constraint.
        group_ids: Optional list of group IDs to filter results.
        limit: Maximum number of results to return.

    Returns:
        List of EntityEdge objects matching the search criteria.
    """

    # Treat empty edge_uuids list as None
    if (
        hasattr(search_filter, "edge_uuids")
        and search_filter.edge_uuids is not None
        and len(search_filter.edge_uuids) == 0
    ):
        search_filter.edge_uuids = None

    # For large graphs, full-text search is too slow. If no specific edge_uuids
    # are provided, skip full-text search and rely on vector similarity instead.
    if not hasattr(search_filter, "edge_uuids") or search_filter.edge_uuids is None:
        return []

    # For small UUID lists, direct lookup is much faster than full-text search
    if (
        hasattr(search_filter, "edge_uuids")
        and search_filter.edge_uuids is not None
        and len(search_filter.edge_uuids) > 0
        and len(search_filter.edge_uuids) <= 20
    ):
        match_query = """
        MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
        WHERE e.uuid IN $edge_uuids
        """
        if group_ids is not None:
            match_query += " AND e.group_id IN $group_ids"

        match_query += """
        RETURN
            e.uuid AS uuid,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.group_id AS group_id,
            e.created_at AS created_at,
            e.name AS name,
            e.fact AS fact,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            properties(e) AS attributes
        LIMIT $limit
        """

        try:
            records, _, _ = await driver.execute_query(
                match_query,
                edge_uuids=search_filter.edge_uuids,
                group_ids=group_ids if group_ids else [],
                limit=limit,
                routing_="r",
            )

            from graphiti_core.search.search_utils import get_entity_edge_from_record  # pylint: disable=import-outside-toplevel

            edges = [
                get_entity_edge_from_record(record, driver.provider)
                for record in records
            ]
            return edges
        except Exception:
            # Fallback to original full-text search if direct lookup fails
            pass

    # Fallback to original implementation for other cases
    return await _original_edge_fulltext_search(
        driver, query, search_filter, group_ids, limit
    )


search_filters_module.edge_search_filter_query_constructor = (
    patched_edge_search_filter_query_constructor
)
search_utils_module.edge_fulltext_search = patched_edge_fulltext_search

# Patch FalkorDriver.execute_query to sanitize empty edge_uuids at the driver level.
_original_falkor_execute_query = FalkorDriver.execute_query


async def patched_falkor_execute_query(self, cypher_query_, **kwargs):
    """Sanitizes query parameters to prevent FalkorDB execution errors.

    Removes empty edge_uuids parameters and adjusts Cypher query syntax accordingly
    to avoid malformed queries that would cause database errors or timeouts.

    Args:
        self: FalkorDriver instance.
        cypher_query_: The Cypher query string to execute.
        **kwargs: Query parameters including potential edge_uuids list.

    Returns:
        Query execution result from the original FalkorDB driver method.
    """
    if (
        "edge_uuids" in kwargs
        and isinstance(kwargs["edge_uuids"], list)
        and len(kwargs["edge_uuids"]) == 0
    ):
        kwargs.pop("edge_uuids", None)
        if "WHERE e.uuid in $edge_uuids" in cypher_query_:
            cypher_query_ = cypher_query_.replace(
                " WHERE e.uuid in $edge_uuids AND ", " WHERE "
            )
            cypher_query_ = cypher_query_.replace(" AND e.uuid in $edge_uuids", "")
            cypher_query_ = cypher_query_.replace(" WHERE e.uuid in $edge_uuids", "")
    return await _original_falkor_execute_query(self, cypher_query_, **kwargs)


FalkorDriver.execute_query = patched_falkor_execute_query


class FalkorDBIngestionEnhanced(BaseIngestionTool):
    """Ingests UNFCCC climate documents into FalkorDB knowledge graph.

    Extends BaseIngestionTool with FalkorDB-specific driver configuration,
    performance monkey patches, and annex sub-section splitting for granular
    ingestion of large annexes.

    Features:
        - All base ingestion capabilities (document parsing, reference extraction, etc.)
        - FalkorDB driver connection via host/port/username/password
        - Annex sub-section splitting for granular entity/relationship extraction
        - Performance optimizations for large-scale graph operations
    """

    DB_NAME = "FalkorDB"

    ANNEX_SPLIT_THRESHOLD = 1000

    def _connection_config(self) -> Dict[str, Any]:
        """Returns FalkorDB-specific connection configuration.

        Reads host, port, username, and password from environment variables.

        Returns:
            Dictionary with FalkorDB connection settings.
        """
        return {
            "host": os.environ.get("FALKORDB_HOST", "localhost"),
            "port": int(os.environ.get("FALKORDB_PORT", "6379")),
            "username": os.environ.get("FALKORDB_USERNAME"),
            "password": os.environ.get("FALKORDB_PASSWORD"),
            "graph_name": os.environ.get(
                "FALKORDB_GRAPH_NAME", "unfccc_knowledge_graph"
            ),
        }

    def _log_connection_info(self) -> None:
        """Logs FalkorDB connection details (host, port, database)."""
        self.logger.info(
            "Connecting to FalkorDB at %s:%s, database: %s",
            self.config["host"],
            self.config["port"],
            self.config["graph_name"],
        )

    def _create_graphiti_client(self) -> Graphiti:
        """Creates and configures a Graphiti client for FalkorDB.

        Initializes the Graphiti framework with FalkorDB driver and applies
        constrained prompts to prevent entity hallucinations during graph construction.

        Returns:
            Configured Graphiti client instance ready for use.
        """
        driver = FalkorDriver(
            host=self.config["host"],
            port=self.config["port"],
            username=self.config["username"],
            password=self.config["password"],
            database=self.config["graph_name"],
        )

        self._apply_constrained_prompts()

        return Graphiti(graph_driver=driver)


async def main() -> None:
    """Standalone entry point for manual execution.

    Creates a FalkorDBIngestionEnhanced instance with default configuration
    from environment variables and runs the complete ingestion pipeline.
    """
    runner = FalkorDBIngestionEnhanced()
    await runner.run()


if __name__ == "__main__":
    asyncio.run(main())
