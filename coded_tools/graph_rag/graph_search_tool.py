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
import os
import re
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from dotenv import load_dotenv

# Load environment variables before importing graphiti modules.
# graphiti_core.embedder.client reads EMBEDDING_DIM at import time.
_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

from graphiti_core import Graphiti  # noqa: E402
from graphiti_core.driver.falkordb_driver import FalkorDriver  # noqa: E402
from graphiti_core.driver.neo4j_driver import Neo4jDriver  # noqa: E402
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF  # noqa: E402
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF  # noqa: E402
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF  # noqa: E402
from neuro_san.interfaces.coded_tool import CodedTool  # noqa: E402

from .formatters import ResultFormatter  # noqa: E402
from .query_analyzer import QueryAnalyzer  # noqa: E402
from .search_stages import SearchPipeline  # noqa: E402

logger = logging.getLogger(__name__)


class GraphSearchTool(QueryAnalyzer, SearchPipeline, ResultFormatter, CodedTool):
    """
    Graph-based RAG tool for semantic search over UNFCCC climate documents.

    Performs hybrid search (vector + keyword) over knowledge graph stored in Neo4j or FalkorDB.
    Supports four search modes: general facts, entities, relationships, and episodes.
    Automatically detects which database to use based on environment variables.
    """

    # Class-level instances shared across all invocations
    _graphiti_instance: Optional[Graphiti] = None
    _driver_instance: Optional[Any] = None  # Can be Neo4jDriver or FalkorDriver
    _db_type: Optional[str] = None  # Track which database we're using
    # Max output size in characters (~10K tokens) to stay within GPT-4o context
    MAX_OUTPUT_CHARS = 40000
    _NO_CONNECTION_MSG = (
        "Error: Failed to initialize graph connection.\n"
        "\n"
        "This tool supports both Neo4j and FalkorDB. Please configure ONE of:\n"
        "\n"
        "Option 1 - Neo4j (Recommended):\n"
        "  Set these in your .env file:\n"
        "  - NEO4J_URI (e.g., bolt://localhost:7687 or "
        "neo4j+s://xxx.databases.neo4j.io)\n"
        "  - NEO4J_USER (e.g., neo4j)\n"
        "  - NEO4J_PASSWORD\n"
        "\n"
        "Option 2 - FalkorDB:\n"
        "  Set these in your .env file:\n"
        "  - FALKORDB_HOST (default: localhost)\n"
        "  - FALKORDB_PORT (default: 6379)\n"
        "  - GRAPH_NAME (default: unfccc_knowledge_graph)\n"
        "\n"
        "Make sure your chosen database is running and accessible."
    )

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """
        Execute graph relationship search and return relevant facts.

        :param args: Dictionary with query and optional parameters
        :param sly_data: Additional context data (not used)
        :return: Formatted string with retrieved graph facts
        """
        del sly_data
        try:
            # Initialize graph if needed
            if GraphSearchTool._graphiti_instance is None:
                await self._initialize_graph()

            # Validate connection
            if (
                GraphSearchTool._graphiti_instance is None
                or GraphSearchTool._driver_instance is None
            ):
                return self._NO_CONNECTION_MSG

            try:
                query, limit, search_type = self._parse_args(args)
            except ValueError as parse_error:
                return str(parse_error)

            query_analysis = self._analyze_query(query)
            logger.debug(
                "Query analysis: intent=%s, key_concepts=%s, decision_id=%s, "
                "paragraph_refs=%s, temporal_direction=%s, is_timeline=%s, "
                "is_identification=%s, conference_filter=%s",
                query_analysis["intent"],
                query_analysis["key_concepts"],
                query_analysis.get("decision_id"),
                query_analysis.get("paragraph_refs", []),
                query_analysis.get("temporal_direction"),
                query_analysis.get("is_timeline_query"),
                query_analysis.get("is_identification_query"),
                query_analysis.get("conference_filter"),
            )

            # If user didn't specify search_type, auto-select based on query intent
            if args.get("search_type") is None:
                search_type = query_analysis["recommended_search_type"]
                logger.debug("Auto-selected search_type: %s", search_type)

            _, formatted_output = await self._dispatch_search(
                search_type, query, query_analysis, limit
            )
            return formatted_output

        except ConnectionError as conn_error:
            db_name = GraphSearchTool._db_type or "graph database"
            error_msg: str = (
                f"Connection error: {str(conn_error)}\n"
                f"Please verify {db_name} is running and connection settings are correct."
            )
            logger.exception("%s", error_msg)
            return error_msg
        except TimeoutError as timeout_error:
            error_msg = (
                f"Timeout error: {str(timeout_error)}\n"
                "The search query took too long. Try a more specific query or reduce the limit."
            )
            logger.exception("%s", error_msg)
            return error_msg
        except Exception as exception:
            error_msg = (
                f"Unexpected error during graph search:\n"
                f"  Error type: {type(exception).__name__}\n"
                f"  Error message: {str(exception)}\n"
                f"  Query: {args.get('query', 'N/A')}\n"
                f"  Search type: {args.get('search_type', 'general')}\n"
                "\nPlease check the traceback above for more details."
            )
            logger.exception("%s", error_msg)
            return error_msg

    def _parse_args(self, args: Dict[str, Any]) -> Tuple[str, int, str]:
        """
        Parse and validate search arguments from async_invoke.

        :param args: Raw args dict
        :return: Tuple of (query, limit, search_type)
        :raises ValueError: If any argument is invalid
        """
        query: str = args.get("query")
        if not query:
            raise ValueError("Error: query parameter is required")

        complexity_limits = {"direct": 1, "medium": 5, "extensive": 15}
        query_complexity: str = args.get("query_complexity", "medium")
        if query_complexity not in complexity_limits:
            query_complexity = "medium"

        try:
            limit: int = int(args.get("limit", complexity_limits[query_complexity]))
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Error: limit must be a valid integer, got: {args.get('limit')}"
            ) from exc

        search_type: str = args.get("search_type", "general")
        valid_types = ["general", "entity", "relationship", "episode"]
        if search_type not in valid_types:
            raise ValueError(
                f"Error: search_type must be one of {valid_types}, got: {search_type}"
            )

        logger.debug("Query complexity: %s, Limit: %d", query_complexity, limit)
        return query, limit, search_type

    async def _dispatch_search(
        self,
        search_type: str,
        query: str,
        query_analysis: Dict[str, Any],
        limit: int,
    ) -> Tuple[List[Any], str]:
        """
        Dispatch to the appropriate search method based on search_type.

        "general" uses multi-stage search (episodes + entities + relationships).
        All other types use their dedicated single-stage search + formatter pair.

        :param search_type: One of "general", "entity", "relationship", "episode"
        :param query: Search query text
        :param query_analysis: Query analysis results from _analyze_query
        :param limit: Maximum number of results
        :return: Tuple of (raw results, formatted output string)
        """
        if search_type == "general":
            return await self._multi_stage_search(query, query_analysis, limit)

        dispatch: Dict[str, Any] = {
            "entity": (self._search_entities, self._format_entities),
            "relationship": (self._search_relationships, self._format_relationships),
            "episode": (self._search_episodes, self._format_episodes),
        }
        search_fn, format_fn = dispatch[search_type]
        results = await search_fn(query, limit)
        formatted_output = await format_fn(query, results)

        # When a specific decision_id is known, prepend a direct lookup so the
        # decision content is always present regardless of semantic search ranking.
        decision_id = query_analysis.get("decision_id")
        if decision_id:
            decision_text = await self._search_by_decision(decision_id)
            if decision_text and decision_text not in formatted_output:
                logger.debug("Prepending direct decision lookup for %s", decision_id)
                formatted_output = decision_text + "\n\n" + formatted_output

        return results, formatted_output

    async def _initialize_graph(self) -> None:
        """
        Initialize Neo4j or FalkorDB connection and Graphiti instance.

        Automatically detects which database to use based on environment variables:
        - If NEO4J_URI is set, uses Neo4j
        - Otherwise, if FALKORDB_HOST is set, uses FalkorDB
        - Falls back to FalkorDB defaults if neither is set

        :return: None
        :raises ConnectionError: If connection fails
        """
        try:
            # Check for Neo4j configuration first (priority)
            neo4j_uri = os.getenv("NEO4J_URI")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")

            if neo4j_uri and neo4j_user and neo4j_password:
                # Use Neo4j
                GraphSearchTool._db_type = "neo4j"
                logger.info("Detected Neo4j configuration")
                logger.info("Initializing Neo4j connection to %s", neo4j_uri)

                GraphSearchTool._driver_instance = Neo4jDriver(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                )

                GraphSearchTool._graphiti_instance = Graphiti(
                    graph_driver=GraphSearchTool._driver_instance
                )

                logger.info("Successfully initialized Neo4j graph connection")

            else:
                # Use FalkorDB
                GraphSearchTool._db_type = "falkordb"
                host = os.getenv("FALKORDB_HOST", "localhost")
                port_str = os.getenv("FALKORDB_PORT", "6379")

                # Validate port is a valid integer
                try:
                    port = int(port_str)
                except (ValueError, TypeError) as exc:
                    raise ValueError(
                        f"FALKORDB_PORT must be a valid integer, got: {port_str}"
                    ) from exc

                username = os.getenv("FALKORDB_USERNAME")
                password = os.getenv("FALKORDB_PASSWORD")
                database = os.getenv("GRAPH_NAME", "unfccc_knowledge_graph")

                logger.info("Detected FalkorDB configuration (or using defaults)")
                logger.info(
                    "Initializing FalkorDB connection to %s:%s, database: %s",
                    host,
                    port,
                    database,
                )

                GraphSearchTool._driver_instance = FalkorDriver(
                    host=host,
                    port=port,
                    username=username,
                    password=password,
                    database=database,
                )

                GraphSearchTool._graphiti_instance = Graphiti(
                    graph_driver=GraphSearchTool._driver_instance
                )

                logger.info("Successfully initialized FalkorDB graph connection")

        except Exception as e:
            # Provide detailed error message based on which DB we tried to connect to
            if GraphSearchTool._db_type == "neo4j":
                error_msg = (
                    f"Failed to initialize Neo4j connection: {str(e)}\n"
                    f"Connection details:\n"
                    f"  URI: {os.getenv('NEO4J_URI', 'NOT SET')}\n"
                    f"  User: {os.getenv('NEO4J_USER', 'NOT SET')}\n"
                    "Please verify:\n"
                    "  1. Neo4j is running and accessible\n"
                    "  2. NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD are correctly set in .env\n"
                    "  3. Network connectivity to Neo4j server"
                )
            else:
                error_msg = (
                    f"Failed to initialize FalkorDB connection: {str(e)}\n"
                    f"Connection details:\n"
                    f"  Host: {os.getenv('FALKORDB_HOST', 'localhost')}\n"
                    f"  Port: {os.getenv('FALKORDB_PORT', '6379')}\n"
                    f"  Database: {os.getenv('GRAPH_NAME', 'unfccc_knowledge_graph')}\n"
                    "Please verify:\n"
                    "  1. FalkorDB is running and accessible\n"
                    "  2. FALKORDB_HOST, FALKORDB_PORT are correctly set in .env\n"
                    "  3. Network connectivity to FalkorDB server"
                )

            logger.exception("%s", error_msg)
            raise ConnectionError(error_msg) from e

    async def _run_search(
        self, query: str, limit: int, config, result_attr: str
    ) -> List[Any]:
        """
        Shared implementation for entity, relationship, and episode searches.

        :param query: Search query text (pre-expanded if needed)
        :param limit: Maximum number of results
        :param config: Search config recipe (NODE_, EDGE_, or COMBINED_HYBRID_SEARCH_RRF)
        :param result_attr: Attribute on SearchResults to return ("nodes", "edges", "episodes")
        :return: List of results
        :raises: Exception if search fails
        """
        try:
            cfg = config.model_copy(deep=True)
            cfg.limit = limit
            results = await GraphSearchTool._graphiti_instance._search(  # pylint: disable=protected-access
                query=query, config=cfg
            )
            items = (results and getattr(results, result_attr)) or []
            logger.debug("%s search returned %d results", result_attr, len(items))
            return items
        except Exception as e:
            logger.error("Error during %s search: %s", result_attr, e)
            raise

    async def _search_entities(self, query: str, limit: int) -> List[Any]:
        return await self._run_search(query, limit, NODE_HYBRID_SEARCH_RRF, "nodes")

    async def _search_relationships(self, query: str, limit: int) -> List[Any]:
        return await self._run_search(query, limit, EDGE_HYBRID_SEARCH_RRF, "edges")

    async def _search_episodes(self, query: str, limit: int) -> List[Any]:
        expanded = self._expand_query_terms(query)
        logger.debug(
            "Episode query expansion: '%s...' -> '%s...'", query[:50], expanded[:70]
        )
        return await self._run_search(
            expanded, limit, COMBINED_HYBRID_SEARCH_RRF, "episodes"
        )

    async def _search_by_decision(self, decision_id: str) -> str:
        """
        Search for a specific decision by its ID in episode metadata.

        Uses two strategies:
        1. Semantic search + metadata matching (fast but may miss some)
        2. Direct Cypher query on episode names/content (fallback, more reliable)

        :param decision_id: Decision ID (e.g., "3/CMA.1", "17/CP.22")
        :return: Formatted decision text or empty string if not found
        """
        try:
            # Strategy 1: Semantic search + metadata matching
            search_query = f"decision {decision_id}"
            logger.debug("Searching for decision: %s", decision_id)

            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 10

            search_results = await GraphSearchTool._graphiti_instance._search(  # pylint: disable=protected-access
                query=search_query, config=config
            )

            episodes = (search_results and search_results.episodes) or []
            logger.debug(
                "Found %d candidate episodes via semantic search", len(episodes)
            )

            # Check metadata for exact decision_id match
            for episode in episodes:
                if not hasattr(episode, "metadata") or not episode.metadata:
                    continue

                episode_decision_id = episode.metadata.get("decision_id", "")
                if (
                    episode_decision_id
                    and episode_decision_id.strip().upper()
                    == decision_id.strip().upper()
                ):
                    episode_name = (
                        episode.name if hasattr(episode, "name") else "Unknown"
                    )
                    logger.debug(
                        "Found exact match via semantic search: %s", episode_name
                    )
                    return self._format_decision_result(decision_id, episode)

            # Strategy 2: Direct Cypher query as fallback
            logger.debug(
                "Semantic search didn't find Decision %s, trying direct database query...",
                decision_id,
            )
            direct_result = await self._search_decision_by_cypher(decision_id)
            if direct_result:
                return direct_result

            logger.warning("Decision %s not found by any method", decision_id)
            return ""

        except Exception as e:
            logger.exception("Error during decision search: %s", e)
            return ""

    async def _search_decision_by_cypher(self, decision_id: str) -> str:
        """
        Search for a decision directly in the database using Cypher query.

        Uses two precision strategies to avoid false positives:
        1. Exact metadata property match (e.decision_id = X) — most reliable
        2. Episode name match using 'DECISION X/Y.Z' prefix — avoids content false
           positives from episodes that merely *mention* the decision (e.g., preambles
           of other decisions that say "recalling Decision 3/CMA.1").

        Returns ALL main-body sections/chunks of the decision (up to 20), concatenated.

        :param decision_id: Decision ID (e.g., "3/CMA.1", "17/CP.22")
        :return: Formatted result concatenating all sections, or empty string
        """
        if not GraphSearchTool._driver_instance:
            return ""

        try:
            decision_id_clean = decision_id.strip().upper()

            # Strategy A: Exact metadata property match (requires migrate_metadata run)
            cypher_a = """
            MATCH (e:Episodic)
            WHERE toUpper(e.decision_id) = $decision_id
              AND e.annex_id IS NULL
            RETURN e.name AS name, e.content AS content,
                   e.source_description AS source_description
            ORDER BY coalesce(e.chunk_index, 0), e.name
            LIMIT 20
            """
            logger.debug(
                "Cypher strategy A: exact decision_id = '%s'", decision_id_clean
            )
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                cypher_a,
                decision_id=decision_id_clean,
            )

            if not records:
                # Strategy B: Name-only match using 'DECISION X/Y.Z' prefix.
                # Searching the full content is intentionally excluded here because
                # other decisions' preambles often mention "recalling Decision X/Y.Z",
                # which would create false positives. The name field is set during
                # ingestion as "<file>::Decision X/Y.Z" and is unique to the episode.
                search_name = f"DECISION {decision_id_clean}"
                cypher_b = """
                MATCH (e:Episodic)
                WHERE toUpper(e.name) CONTAINS $search_name
                  AND NOT toLower(e.name) CONTAINS 'annex'
                RETURN e.name AS name, e.content AS content,
                       e.source_description AS source_description
                ORDER BY coalesce(e.chunk_index, 0), e.name
                LIMIT 20
                """
                logger.debug("Cypher strategy B: name contains '%s'", search_name)
                records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                    cypher_b,
                    search_name=search_name,
                )

            if not records:
                logger.debug(
                    "Direct Cypher query found no results for Decision %s", decision_id
                )
                return ""

            # Build output from all returned sections
            source_desc = dict(records[0]).get("source_description", "")
            parts = [
                f">>> DECISION {decision_id} <<<",
                f"Source: {source_desc}",
                "",
            ]
            for record in records:
                r = dict(record)
                name = r.get("name", "Unknown")
                content = r.get("content", "")
                parts.append(f"EXACT TEXT FROM SOURCE: {name}")
                parts.append("-" * 80)
                parts.append(content)
                parts.append("")

            logger.debug(
                "Found Decision %s via Cypher (%d section(s))",
                decision_id,
                len(records),
            )
            return "\n".join(parts)

        except Exception as e:
            logger.exception("Error during direct Cypher search: %s", e)
            return ""

    async def _search_by_paragraph(
        self, query: str, query_analysis: Dict[str, Any]
    ) -> List[str]:
        """
        Search for specific paragraphs using metadata paragraph_index.

        This method extracts paragraph references from the query, searches for
        episodes containing those paragraphs, and returns the exact paragraph text
        from the metadata.

        :param query: Original search query
        :param query_analysis: Analysis results containing paragraph_refs
        :return: List of formatted paragraph results
        """

        paragraph_refs = query_analysis.get("paragraph_refs", [])
        if not paragraph_refs:
            return []

        results = []

        # Extract decision reference from query (e.g., "Decision 4/CMA.1")
        decision_pattern = r"\b(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)\b"
        decision_match = re.search(decision_pattern, query, re.IGNORECASE)
        decision_id = decision_match.group(1) if decision_match else None

        # Try to find episodes that might contain these paragraphs
        # Strategy: search for decision ID or use broader search
        if decision_id:
            search_query = f"decision {decision_id}"
        else:
            # Extract key terms from query for broader search
            search_query = query[:100]

        logger.debug(
            "Searching for episodes containing decision: %s",
            decision_id or "unknown (using query)",
        )

        try:
            # Use combined search to find episodes
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 10  # Get more candidates to find the right decision

            search_results = await GraphSearchTool._graphiti_instance._search(  # pylint: disable=protected-access
                query=search_query, config=config
            )

            episodes = (search_results and search_results.episodes) or []
            logger.debug(
                "Found %d candidate episodes to search for paragraphs", len(episodes)
            )

            # Extract paragraphs from episode metadata
            for para_ref in paragraph_refs:
                para_num = para_ref["number"]
                subsection = para_ref.get("subsection")
                subsubsection = para_ref.get("subsubsection")

                # Build paragraph identifier
                para_id = para_num
                if subsection:
                    para_id += f"({subsection})"
                if subsubsection:
                    para_id += f"({subsubsection})"

                found = False
                for episode in episodes:
                    # Check if episode has paragraph_index in metadata
                    if not hasattr(episode, "metadata") or not episode.metadata:
                        continue

                    paragraph_index = episode.metadata.get("paragraph_index", {})
                    if not paragraph_index:
                        continue

                    # Try to find the paragraph in the index
                    para_text = None
                    if para_num in paragraph_index:
                        para_text = paragraph_index[para_num]
                    elif str(para_num) in paragraph_index:
                        para_text = paragraph_index[str(para_num)]

                    if para_text:
                        # Format the result
                        episode_name = (
                            episode.name if hasattr(episode, "name") else "Unknown"
                        )
                        decision_from_metadata = episode.metadata.get(
                            "decision_id", "Unknown"
                        )

                        formatted_result = (
                            f"**Paragraph {para_id}** from {decision_from_metadata}\n"
                        )
                        formatted_result += f"Source: {episode_name}\n"
                        formatted_result += f"\n{para_text}\n"

                        results.append(formatted_result)
                        found = True
                        logger.debug(
                            "Found paragraph %s in episode: %s", para_id, episode_name
                        )
                        break

                if not found:
                    logger.warning(
                        "Could not find paragraph %s in any episode metadata", para_id
                    )

        except Exception as e:
            logger.exception("Error during paragraph metadata search: %s", e)

        return results

    async def _get_episode_metadata(self, uuid: str) -> Dict[str, Any]:
        """
        Fetch episode metadata by UUID.

        :param uuid: Episode UUID to look up
        :return: Dictionary with episode metadata or empty dict if not found
        """
        if not GraphSearchTool._driver_instance or not uuid:
            return {}

        try:
            query: str = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.decision_id as decision_id,
                   e.conference_type as conference_type,
                   e.year as year,
                   e.session_number as session_number,
                   e.annex_id as annex_id
            LIMIT 1
            """
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                query, uuid=uuid
            )
            if records:
                return dict(records[0])
            return {}
        except Exception as e:
            logger.warning("Failed to fetch episode metadata %s: %s", uuid, e)
            return {}

    async def _get_episode_citation_data(self, episode_uuid: str) -> Dict[str, Any]:
        """
        Fetch episode data for building citations.

        Queries Episodic nodes for name, source_description, content, and source
        fields used by _build_citation.

        :param episode_uuid: Episode UUID to look up
        :return: Dictionary with episode citation data or empty dict if not found
        """
        if not GraphSearchTool._driver_instance:
            return {}

        try:
            query: str = """
            MATCH (e:Episodic {uuid: $uuid})
            RETURN e.name as name, e.source_description as source_description,
                   e.content as content, e.source as source
            LIMIT 1
            """
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                query, uuid=episode_uuid
            )
            if records:
                record_dict = dict(records[0])
                metadata = {}
                source = record_dict.get("source", "")
                if source:
                    decision_match = re.search(
                        r"decision[s]?\s*(\d+/[A-Z]+\.\d+)", source, re.IGNORECASE
                    )
                    if decision_match:
                        metadata["decision_id"] = decision_match.group(1)

                record_dict["metadata"] = metadata
                return record_dict
            return {}
        except Exception as e:
            logger.warning("Failed to fetch episode %s: %s", episode_uuid, e)
            return {}

    async def _get_node_by_uuid(self, uuid: str) -> Dict[str, Any]:
        """
        Fetch node details by UUID.

        :param uuid: Node UUID to look up
        :return: Dictionary with node details or empty dict if not found
        """
        if not GraphSearchTool._driver_instance:
            return {}

        try:
            query: str = """
            MATCH (n:Entity {uuid: $uuid})
            RETURN n.name as name, n.summary as summary, n.entity_type as entity_type, labels(n) as labels
            LIMIT 1
            """
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                query, uuid=uuid
            )
            if records:
                return dict(records[0])
            return {}
        except Exception as e:
            logger.warning("Failed to fetch node %s: %s", uuid, e)
            return {}

    async def _get_entity_connections(self, uuid: str) -> List[Dict[str, Any]]:
        """
        Get relationships connected to an entity.

        :param uuid: Entity UUID to find connections for
        :return: List of connection dictionaries
        """
        if not GraphSearchTool._driver_instance:
            return []

        try:
            query: str = """
            MATCH (source:Entity {uuid: $uuid})-[r:RELATES_TO]->(target:Entity)
            RETURN r.name as relationship, target.name as target_name, r.fact as fact
            LIMIT 5
            """
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                query, uuid=uuid
            )
            return [dict(record) for record in records]
        except Exception as e:
            logger.warning("Failed to fetch connections for entity %s: %s", uuid, e)
            return []
