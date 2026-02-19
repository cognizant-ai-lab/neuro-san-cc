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
Graph search tool for UNFCCC climate documents using Neo4j or FalkorDB knowledge graph.

This module implements a Graph-based Retrieval-Augmented Generation (RAG) tool that provides
semantic search capabilities over a knowledge graph of climate conference documents stored in
Neo4j or FalkorDB. It leverages the Graphiti library to perform hybrid search (combining vector
similarity and keyword matching) over entities and relationships extracted from UNFCCC documents
including COP, CMA, CMP, SBI, and SBSTA proceedings.

Architecture:
    - Uses Graphiti Core for graph operations and hybrid search
    - Automatically detects and connects to Neo4j or FalkorDB based on environment variables
    - Implements singleton pattern for graph connections (shared across invocations)
    - Supports four search modes: general facts, entities, relationships, and episodes
    - Automatically enriches results with connected nodes and relationships

Database Detection:
    - If NEO4J_URI is set, uses Neo4j
    - If FALKORDB_HOST is set (and NEO4J_URI is not), uses FalkorDB
    - Priority: Neo4j > FalkorDB

Search Types:
    1. "general" (default): Searches for general facts and relationships in the graph.
       Returns facts with connected entity details.

    2. "entity": Searches for entity nodes (e.g., countries, organizations, concepts).
       Returns entities with their summaries and connected relationships.

    3. "relationship": Searches for relationship edges between entities.
       Returns relationships with source and target entity details.

    4. "episode": Searches for primary source document content (episodes).
       Returns original document text with metadata (conference, year, decision IDs).
       Best for getting exact wording from source documents.
"""

# pylint: disable=too-many-lines
# pylint: disable=wrong-import-position
# pylint: disable=line-too-long
# pylint: disable=too-many-locals
# pylint: disable=too-many-return-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-statements
# pylint: disable=too-many-boolean-expressions
# pylint: disable=protected-access
# pylint: disable=broad-exception-caught

import os
import re
import traceback
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from dotenv import load_dotenv
from neuro_san.interfaces.coded_tool import CodedTool

# Load environment variables BEFORE importing graphiti
_current_dir = Path(__file__).parent
load_dotenv(dotenv_path=_current_dir / ".env")

from graphiti_core import Graphiti
from graphiti_core.driver.falkordb_driver import FalkorDriver
from graphiti_core.driver.neo4j_driver import Neo4jDriver
from graphiti_core.search.search_config_recipes import COMBINED_HYBRID_SEARCH_RRF
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_RRF
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


class GraphSearchTool(CodedTool):
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

    async def async_invoke(self, args: Dict[str, Any], sly_data: Dict[str, Any]) -> str:
        """
        Execute graph relationship search and return relevant facts.

        :param args: Dictionary with query and optional parameters
        :param sly_data: Additional context data (not used)
        :return: Formatted string with retrieved graph facts
        """
        try:
            # Initialize graph if needed
            if GraphSearchTool._graphiti_instance is None:
                await self._initialize_graph()

            # Validate connection
            if GraphSearchTool._graphiti_instance is None or GraphSearchTool._driver_instance is None:
                return (
                    "Error: Failed to initialize graph connection.\n"
                    "\n"
                    "This tool supports both Neo4j and FalkorDB. Please configure ONE of:\n"
                    "\n"
                    "Option 1 - Neo4j (Recommended):\n"
                    "  Set these in your .env file:\n"
                    "  - NEO4J_URI (e.g., bolt://localhost:7687 or neo4j+s://xxx.databases.neo4j.io)\n"
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

            # Extract and validate arguments
            query: str = args.get("query")
            if not query:
                return "Error: query parameter is required"

            # Get query complexity to determine search depth
            query_complexity: str = args.get("query_complexity", "medium")
            valid_complexities = ["direct", "medium", "extensive"]
            if query_complexity not in valid_complexities:
                query_complexity = "medium"  # Default to medium if invalid

            # Adjust limits based on query complexity
            complexity_limits = {
                "direct": 1,      # Minimal: 1 result for specific questions
                "medium": 5,      # Balanced: 5 results for moderate topics
                "extensive": 15   # Comprehensive: 15 results for broad exploration
            }

            # Convert string inputs to appropriate types
            try:
                # Use complexity-based default if limit not specified
                default_limit = complexity_limits.get(query_complexity, 5)
                limit: int = int(args.get("limit", default_limit))
            except (ValueError, TypeError):
                return f"Error: limit must be a valid integer, got: {args.get('limit')}"

            search_type: str = args.get("search_type", "general")
            valid_types = ["general", "entity", "relationship", "episode"]
            if search_type not in valid_types:
                return f"Error: search_type must be one of {valid_types}, got: {search_type}"

            print(f"Query complexity: {query_complexity}, Limit: {limit}")

            # **ENHANCED SEARCH QUALITY**: Analyze query and determine optimal search strategy
            query_analysis = self._analyze_query(query)
            print(f"Query analysis: intent={query_analysis['intent']}, key_concepts={query_analysis['key_concepts']}, decision_id={query_analysis.get('decision_id')}, paragraph_refs={query_analysis.get('paragraph_refs', [])}, temporal_direction={query_analysis.get('temporal_direction')}, is_timeline={query_analysis.get('is_timeline_query')}, is_identification={query_analysis.get('is_identification_query')}, conference_filter={query_analysis.get('conference_filter')}")

            # If user didn't specify search_type, auto-select based on query intent
            if args.get("search_type") is None:
                search_type = query_analysis["recommended_search_type"]
                print(f"Auto-selected search_type: {search_type}")

            # **ENHANCED SEARCH QUALITY**: Use multi-stage search for ALL general queries.
            # Multi-stage search produces strictly better results by combining episodes
            # (primary source documents), entities, AND relationships — whereas single-stage
            # only returns abstract facts without source document context.
            # Only use single-stage for explicitly requested non-general search types.
            if search_type == "general":
                results, formatted_output = await self._multi_stage_search(query, query_analysis, limit)
            elif search_type == "entity":
                results: List[Any] = await self._search_entities(query, limit)
                formatted_output: str = await self._format_entities(query, results)
            elif search_type == "relationship":
                results = await self._search_relationships(query, limit)
                formatted_output = await self._format_relationships(query, results)
            elif search_type == "episode":
                results = await self._search_episodes(query, limit)
                formatted_output = await self._format_episodes(query, results)
            else:
                results = await self._search_graph(query, limit)
                formatted_output = await self._format_facts(query, results)

            return formatted_output

        except ConnectionError as conn_error:
            db_name = GraphSearchTool._db_type or "graph database"
            error_msg: str = (
                f"Connection error: {str(conn_error)}\n"
                f"Please verify {db_name} is running and connection settings are correct."
            )
            traceback.print_exc()
            return error_msg
        except TimeoutError as timeout_error:
            error_msg: str = (
                f"Timeout error: {str(timeout_error)}\n"
                "The search query took too long. Try a more specific query or reduce the limit."
            )
            traceback.print_exc()
            return error_msg
        # pylint: disable=broad-exception-caught
        except Exception as exception:
            error_msg: str = (
                "Unexpected error during graph search:\n"
                f"  Error type: {type(exception).__name__}\n"
                f"  Error message: {str(exception)}\n"
                f"  Query: {args.get('query', 'N/A')}\n"
                f"  Search type: {args.get('search_type', 'general')}\n"
                "\nPlease check the traceback above for more details."
            )
            traceback.print_exc()
            return error_msg

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to extract key information and determine optimal search strategy.

        Identifies:
        - Query intent (factual, requirement, definition, relationship)
        - Key concepts and entities
        - Temporal markers (years, conferences)
        - Structural markers (annex, decision, article)
        - Complexity level
        - Recommended search approach

        :param query: User's search query
        :return: Dictionary with query analysis results
        """
        query_lower = query.lower()
        analysis = {
            "intent": "factual",  # factual, requirement, definition, relationship
            "key_concepts": [],
            "temporal_markers": [],
            "structural_markers": [],
            "complexity": "medium",  # low, medium, high
            "recommended_search_type": "general",
            "query_type": "medium_context",  # direct, medium_context, extensive_context
        }

        # Detect intent signals
        if any(word in query_lower for word in ["what condition", "must", "require", "obligation", "shall"]):
            analysis["intent"] = "requirement"
        elif any(word in query_lower for word in ["what is", "define", "definition", "meaning of"]):
            analysis["intent"] = "definition"
        elif any(word in query_lower for word in ["relationship", "connect", "relate", "between", "link"]):
            analysis["intent"] = "relationship"

        # Extract UNFCCC-specific concepts
        unfccc_concepts = [
            "mitigation", "adaptation", "co-benefits", "NDC", "nationally determined contribution",
            "transparency", "reporting", "review", "compliance", "finance", "technology transfer",
            "capacity building", "loss and damage", "market mechanism", "Article 6",
            "global stocktake", "enhanced transparency framework", "ETF", "biennial transparency report",
            "BTR", "nationally appropriate mitigation action", "NAMA", "adaptation communication",
            "economic diversification", "just transition", "common timeframes", "IPCC",
            "developed country", "developing country", "least developed country", "LDC",
            "small island developing state", "SIDS", "Party", "Parties"
        ]
        for concept in unfccc_concepts:
            if concept in query_lower:
                analysis["key_concepts"].append(concept)

        # Extract temporal markers
        years = re.findall(r'\b(19\d{2}|20\d{2})\b', query)
        analysis["temporal_markers"].extend(years)

        conference_types = ["COP", "CMA", "CMP", "SBI", "SBSTA"]
        for conf in conference_types:
            if conf.lower() in query_lower:
                analysis["temporal_markers"].append(conf)

        # Extract conference type constraint for filtering results
        # If user says "CMA decision" or "across CMA sessions", only return CMA results
        # Also handle "CP" as alias for "COP"
        conference_filter = None
        # Check for explicit conference constraints (order matters: check longer patterns first)
        conf_constraint_patterns = [
            (r'\bcma\s+(?:decision|session)', "CMA"),
            (r'\bcop\s+(?:decision|session)', "COP"),
            (r'\bcp\s+(?:decision|session)', "COP"),
            (r'\bcmp\s+(?:decision|session)', "CMP"),
            (r'\bsbi\s+(?:decision|session|report)', "SBI"),
            (r'\bsbsta\s+(?:decision|session|report)', "SBSTA"),
            (r'\bacross\s+(?:\w+\s+)?cma\b', "CMA"),
            (r'\bacross\s+(?:\w+\s+)?cop\b', "COP"),
            (r'\bacross\s+(?:\w+\s+)?cmp\b', "CMP"),
            (r'\ba\s+(?:later\s+)?cma\b', "CMA"),
            (r'\ba\s+(?:later\s+)?cop\b', "COP"),
            (r'\ba\s+(?:later\s+)?cmp\b', "CMP"),
            (r'\bwhich\s+cma\b', "CMA"),
            (r'\bwhich\s+cop\b', "COP"),
            (r'\bwhich\s+cmp\b', "CMP"),
        ]
        for pattern, conf_type in conf_constraint_patterns:
            if re.search(pattern, query_lower):
                conference_filter = conf_type
                break
        # Fallback: if only one conference type is mentioned in the query, use it as filter
        if not conference_filter:
            mentioned_confs = [c for c in conference_types if c.lower() in query_lower]
            # Also check for "cp" as alias for COP
            if "cp" in query_lower.split() and "COP" not in mentioned_confs:
                mentioned_confs.append("COP")
            if len(mentioned_confs) == 1:
                conference_filter = mentioned_confs[0]
        analysis["conference_filter"] = conference_filter

        # Extract structural markers (document structure references)
        structural_terms = {
            "annex": r'\bannex(?:es)?\s*[IVX\d]*\b',
            "decision": r'\bdecision\s*\d+/[A-Z]+\.\d+\b',
            "article": r'\barticle\s*\d+\b',
            "paragraph": r'\bparagraph\s*\d+\b',
            "section": r'\bsection\s*[IVX\d]+\b'
        }
        for term, pattern in structural_terms.items():
            if re.search(pattern, query_lower):
                analysis["structural_markers"].append(term)

        # Extract specific decision references (e.g., "Decision 3/CMA.1")
        decision_pattern = r'\bdecision\s+(\d+/[A-Z]+\.\d+)\b'
        decision_match = re.search(decision_pattern, query_lower, re.IGNORECASE)
        if decision_match:
            analysis["decision_id"] = decision_match.group(1).upper()
        else:
            analysis["decision_id"] = None

        # Extract specific paragraph references (e.g., "para 123", "paragraph 121(m)(i-iv)")
        paragraph_patterns = [
            r'\b(?:para\.?|paragraph)\s+(\d+)(?:\s*\(([a-z])\))?(?:\s*\(([ivxlc]+)\))?',
            r'\(para\.?\s*(\d+)(?:\s*\(([a-z])\))?(?:\s*\(([ivxlc]+)\))?\)',
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

        # Detect chronological/temporal direction queries
        chronological_earliest = [
            "first", "originally", "initially", "created", "creates",
            "established", "establishes", "founded", "founds",
            "inception", "origin", "earliest", "when was",
            "who created", "who established", "which decision created",
            "which decision established", "which decision creates",
            "set up", "sets up", "launched", "launches",
        ]
        chronological_latest = [
            "latest", "most recent", "current", "last updated", "newest",
        ]
        # Detect timeline/evolution queries (all decisions on a topic)
        timeline_markers = [
            "all decisions", "every decision", "timeline", "evolution",
            "history of", "how did", "what decisions were made",
            "what decisions address", "complete history",
            "chronolog", "over time", "across sessions", "track",
            "follow-up", "follow up", "review timing", "review cycle",
            "reporting cycle", "subsequent", "later sessions",
            "multiple sessions", "across multiple", "governance",
        ]
        # Detect identification queries (user describes a decision by content, not by ID)
        # e.g., "A later CMA decision creates a new network to support developing countries"
        identification_markers = [
            "a decision", "a later decision", "a cma decision", "a cop decision",
            "a cmp decision", "the decision that", "which decision",
            "which cma decision", "which cop decision", "which cmp decision",
            "identify the decision", "find the decision",
            "a later cma", "a later cop", "a later cmp",
        ]
        is_identification = any(
            marker in query_lower for marker in identification_markers
        )
        # Also detect descriptive patterns: "[conference] decision [verb]s..."
        if not is_identification and re.search(
            r'\b(?:cma|cop|cmp|cp)\s+decision\s+\w+s\b', query_lower
        ):
            is_identification = True
        # Long descriptive queries about what a decision does are likely identification
        if not is_identification and len(query.split()) > 15 and any(
            conf.lower() in query_lower for conf in conference_types
        ):
            is_identification = True

        analysis["is_identification_query"] = is_identification

        if any(marker in query_lower for marker in chronological_earliest):
            analysis["temporal_direction"] = "earliest"
            analysis["follow_references"] = True
        elif any(marker in query_lower for marker in chronological_latest):
            analysis["temporal_direction"] = "latest"
            analysis["follow_references"] = False
        else:
            analysis["temporal_direction"] = None
            analysis["follow_references"] = False

        analysis["is_timeline_query"] = any(
            marker in query_lower for marker in timeline_markers
        )

        # Detect multi-stage governance evolution queries
        # When a query describes multiple governance phases, treat as timeline query
        # e.g., "creates a body... and also sets out follow-up actions... and future review timing"
        governance_phase_markers = [
            "establishes", "creates", "set up", "sets up",  # Phase 1: creation
            "follow-up", "follow up", "operationalize",     # Phase 2: operationalization
            "review", "reporting", "integration", "mandate", # Phase 3: review/integration
            "recommendations", "workplan", "work plan",
            "annual report", "midterm review",
        ]
        phase_count = sum(1 for m in governance_phase_markers if m in query_lower)
        if phase_count >= 2 and not analysis["is_timeline_query"]:
            analysis["is_timeline_query"] = True
            print(f"Multi-stage governance query detected ({phase_count} phases mentioned)")

        # Also trigger timeline for identification queries that mention multiple actions
        # e.g., "creates a network AND sets out follow-up actions AND future review timing"
        has_multiple_actions = query_lower.count(" and ") >= 2 or query_lower.count("also") >= 1
        if has_multiple_actions and analysis.get("is_identification_query") and not analysis["is_timeline_query"]:
            analysis["is_timeline_query"] = True
            print("Multi-action identification query detected, enabling timeline search")

        # Determine complexity
        complexity_score = 0
        complexity_score += len(analysis["key_concepts"])
        complexity_score += 2 if analysis["structural_markers"] else 0
        complexity_score += 2 if analysis["intent"] == "requirement" else 0
        complexity_score += 1 if len(query.split()) > 15 else 0

        if complexity_score >= 5:
            analysis["complexity"] = "high"
        elif complexity_score <= 2:
            analysis["complexity"] = "low"

        # Force high complexity for temporal/timeline/identification queries
        if analysis.get("temporal_direction") or analysis.get("is_timeline_query") or analysis.get("is_identification_query"):
            analysis["complexity"] = "high"

        # Recommend search type based on analysis
        if analysis["intent"] == "requirement" and analysis["structural_markers"]:
            # Questions about specific requirements in document structure need primary sources
            analysis["recommended_search_type"] = "episode"
        elif analysis["intent"] == "relationship":
            analysis["recommended_search_type"] = "relationship"
        elif analysis["intent"] == "definition" and len(analysis["key_concepts"]) == 1:
            analysis["recommended_search_type"] = "entity"
        else:
            analysis["recommended_search_type"] = "general"

        return analysis

    async def _multi_stage_search(
        self, query: str, query_analysis: Dict[str, Any], limit: int
    ) -> tuple:
        """
        Perform multi-stage search for complex queries requiring comprehensive answers.

        Strategy:
        1. Search for relevant entities to understand key concepts
        2. Search episodes (primary sources) for exact document text
        3. Search relationships to understand connections
        4. Merge and synthesize results

        :param query: Original search query
        :param query_analysis: Analysis results from _analyze_query
        :param limit: Maximum results per stage
        :return: Tuple of (combined_results, formatted_output)
        """
        print(f"Executing multi-stage search for complex query: {query[:80]}...")

        # PRIORITY 0: If specific decision is referenced, search for it directly
        decision_results = []
        if query_analysis.get("decision_id"):
            print(f"Detected specific decision reference: {query_analysis['decision_id']}, searching directly...")
            decision_results = await self._search_by_decision(query_analysis["decision_id"], query)

        # PRIORITY 1: If specific paragraphs are referenced, try paragraph-based search
        paragraph_results = []
        if query_analysis.get("paragraph_refs"):
            print(f"Detected {len(query_analysis['paragraph_refs'])} paragraph reference(s), attempting metadata search...")
            paragraph_results = await self._search_by_paragraph(query, query_analysis)

            # If we found specific paragraphs, prioritize those and do lighter follow-up searches
            if paragraph_results:
                print(f"Found {len(paragraph_results)} paragraphs via metadata search")
                # Still do entity/relationship search but with lower limits
                entity_limit = max(2, limit // 4)
                episode_limit = 0  # Skip general episode search since we have specific paragraphs
            else:
                print("Paragraph metadata search found nothing, falling back to standard search")
                entity_limit = max(3, limit // 3)
                episode_limit = max(5, limit // 2)
        else:
            # Normal search limits
            entity_limit = max(3, limit // 3)
            episode_limit = max(5, limit // 2)

        # Boost episode limit for timeline/evolution queries (need comprehensive results)
        if query_analysis.get("is_timeline_query") or (
            query_analysis.get("is_identification_query") and query_analysis.get("temporal_direction")
        ):
            episode_limit = max(episode_limit, 15)
            print(f"Boosted episode limit to {episode_limit} for timeline/evolution query")

        # Stage 1: Find relevant entities (reduced limit for speed)
        print(f"Stage 1: Searching entities (limit={entity_limit})")
        entity_results = await self._search_entities(query, entity_limit)

        # Stage 2: Search primary source documents (episodes) - THIS IS CRITICAL
        # (May be skipped if we already found specific paragraphs)
        episode_results = []
        if episode_limit > 0:
            print(f"Stage 2: Searching episodes/documents (limit={episode_limit})")

            # Enhanced query for episode search: add structural markers if detected
            episode_query = query
            if query_analysis["structural_markers"]:
                # Questions about annexes/articles need exact document text
                structural_terms = " ".join(query_analysis["structural_markers"])
                episode_query = f"{query} {structural_terms}"

            episode_results = await self._search_episodes(episode_query, episode_limit)

        # Apply conference type filter if specified (e.g., only CMA results when user asks about CMA)
        conf_filter = query_analysis.get("conference_filter")
        if conf_filter and episode_results:
            pre_filter_count = len(episode_results)
            episode_results = self._filter_by_conference(episode_results, conf_filter)
            if len(episode_results) < pre_filter_count:
                print(f"Conference filter '{conf_filter}': {pre_filter_count} → {len(episode_results)} episodes")

        if not episode_results and paragraph_results:
            print("Stage 2: Skipping general episode search (using paragraph results instead)")

        # Stage 2B: Follow backward references for chronological/creation queries
        # Run backward refs for BOTH follow_references AND identification queries with creation language
        referenced_decisions = []
        should_follow_refs = (
            query_analysis.get("follow_references")
            or (query_analysis.get("is_identification_query") and query_analysis.get("temporal_direction") == "earliest")
        )
        if should_follow_refs and episode_results:
            print("Stage 2B: Following backward references for chronological/creation query...")
            referenced_decisions = await self._follow_backward_references(episode_results)
            if referenced_decisions:
                print(f"Found {len(referenced_decisions)} referenced earlier decision(s)")

        # Stage 2C: For timeline queries, search broadly for all decisions on the topic
        timeline_results = []
        if query_analysis.get("is_timeline_query"):
            print("Stage 2C: Searching for complete decision timeline...")
            timeline_results = await self._search_topic_timeline(query, query_analysis, limit=20)
            if timeline_results:
                print(f"Found {len(timeline_results)} decisions for timeline")

        # Temporal re-ranking for chronological queries
        if query_analysis.get("temporal_direction") and episode_results:
            episode_results = self._temporal_rerank(episode_results, query_analysis["temporal_direction"])
            print(f"Re-ranked episodes by {query_analysis['temporal_direction']} year")

        # Stage 2D: Classify episodes as founding/follow-up for creation queries
        founding_analysis = None
        is_creation_query = query_analysis.get("temporal_direction") == "earliest" or (
            query_analysis.get("is_identification_query") and any(
                w in query.lower() for w in ["creates", "created", "establishes", "established", "set up", "sets up", "launches", "launched", "founded"]
            )
        )
        if is_creation_query and episode_results:
            founding_analysis = self._classify_episodes_founding(episode_results)
            print(f"Founding analysis: {founding_analysis['summary']}")

        # Stage 3: Search relationships if relevant
        relationship_results = []
        if query_analysis["intent"] == "requirement" or len(entity_results) > 0:
            rel_limit = max(2, limit // 4)
            print(f"Stage 3: Searching relationships (limit={rel_limit})")
            relationship_results = await self._search_relationships(query, rel_limit)

        # Synthesize results with priority on episodes (primary sources)
        print("Synthesizing multi-stage results...")
        output_parts = [
            f"MULTI-STAGE SEARCH RESULTS FOR: '{query}'",
            f"Query Complexity: {query_analysis['complexity'].upper()}",
            f"Query Intent: {query_analysis['intent'].upper()}",
            f"Key Concepts: {', '.join(query_analysis['key_concepts'][:5]) if query_analysis['key_concepts'] else 'None detected'}",
            f"Conference Filter: {conf_filter if conf_filter else 'None (all conferences)'}",
            "",
            "=" * 80,
            "",
        ]

        # PRIORITY 0A: Specific decision (HIGHEST PRIORITY for decision queries)
        if decision_results:
            output_parts.append("=" * 80)
            output_parts.append("SPECIFIC DECISION (EXACT MATCH)")
            output_parts.append("Direct result for the decision referenced in your query")
            output_parts.append("=" * 80)
            output_parts.append("")
            output_parts.append(decision_results)
            output_parts.append("")

        # PRIORITY 0B: Specific paragraphs from metadata (HIGHEST PRIORITY for paragraph queries)
        if paragraph_results:
            output_parts.append("=" * 80)
            output_parts.append("SPECIFIC PARAGRAPHS (EXTRACTED FROM METADATA)")
            output_parts.append("These are the exact paragraphs referenced in your query")
            output_parts.append("=" * 80)
            output_parts.append("")
            for para_result in paragraph_results:
                output_parts.append(para_result)
                output_parts.append("")

        # PRIORITY 0C: FOUNDING DECISION ANALYSIS (for creation queries)
        # When the query asks about creation/establishment, put founding decisions FIRST
        if founding_analysis and founding_analysis.get("founding_episodes"):
            output_parts.append("=" * 80)
            output_parts.append(">>> FOUNDING DECISION(S) IDENTIFIED <<<")
            output_parts.append("These decisions contain CREATION language (establishes, creates, sets up).")
            output_parts.append("Use these to answer 'which decision CREATED/ESTABLISHED X' questions.")
            output_parts.append("=" * 80)
            output_parts.append("")
            founding_text = await self._format_episodes(query, founding_analysis["founding_episodes"])
            output_parts.append(founding_text)
            output_parts.append("")

        if founding_analysis and founding_analysis.get("followup_episodes"):
            output_parts.append("=" * 80)
            output_parts.append("FOLLOW-UP DECISIONS (these further develop/operationalize, NOT create)")
            output_parts.append("Do NOT cite these as the 'founding' or 'creation' decision.")
            output_parts.append("=" * 80)
            output_parts.append("")
            followup_text = await self._format_episodes(query, founding_analysis["followup_episodes"])
            output_parts.append(followup_text)
            output_parts.append("")

        # PRIORITY 0D: Referenced earlier decisions (backward ref traversal)
        if referenced_decisions:
            output_parts.append("=" * 80)
            if is_creation_query:
                output_parts.append(">>> EARLIER REFERENCED DECISIONS (POTENTIAL FOUNDING DECISIONS) <<<")
                output_parts.append("These were recalled/referenced by later decisions.")
                output_parts.append("CHECK THESE FIRST — the founding decision is often referenced by later follow-ups.")
            else:
                output_parts.append("EARLIER REFERENCED DECISIONS (TRACED VIA BACKWARD REFERENCES)")
                output_parts.append("These were recalled/referenced by the results above")
            output_parts.append("=" * 80)
            output_parts.append("")
            for ref_decision in referenced_decisions:
                output_parts.append(ref_decision)
                output_parts.append("")

        # PRIORITY 0E: Complete topic timeline
        if timeline_results:
            output_parts.append("=" * 80)
            output_parts.append("COMPLETE DECISION TIMELINE ON THIS TOPIC")
            output_parts.append("All decisions found, sorted chronologically (earliest first)")
            output_parts.append("=" * 80)
            output_parts.append("")
            for tl_result in timeline_results:
                output_parts.append(tl_result)
                output_parts.append("")

        # PRIORITY 1: Episodes (primary sources with exact text)
        # Skip if we already showed them in founding analysis
        if episode_results and not founding_analysis:
            output_parts.append("=" * 80)
            output_parts.append("PRIMARY SOURCE DOCUMENTS (HIGHEST PRIORITY)")
            output_parts.append("These contain the EXACT TEXT from UNFCCC documents")
            output_parts.append("=" * 80)
            output_parts.append("")
            episode_text = await self._format_episodes(query, episode_results)
            output_parts.append(episode_text)
            output_parts.append("")

        # PRIORITY 2: Entities (concepts and definitions)
        if entity_results:
            output_parts.append("=" * 80)
            output_parts.append("RELEVANT ENTITIES AND CONCEPTS")
            output_parts.append("=" * 80)
            output_parts.append("")
            entity_text = await self._format_entities(query, entity_results)
            output_parts.append(entity_text)
            output_parts.append("")

        # PRIORITY 3: Relationships (connections between concepts)
        if relationship_results:
            output_parts.append("=" * 80)
            output_parts.append("RELATIONSHIPS AND CONNECTIONS")
            output_parts.append("=" * 80)
            output_parts.append("")
            rel_text = await self._format_relationships(query, relationship_results)
            output_parts.append(rel_text)
            output_parts.append("")

        if not decision_results and not paragraph_results and not referenced_decisions and not timeline_results and not episode_results and not entity_results and not relationship_results:
            output_parts.append("No results found across all search stages.")
            output_parts.append("")
            output_parts.append("Suggestions:")
            output_parts.append("- Try broader search terms")
            output_parts.append("- Check spelling of technical terms")
            output_parts.append("- Verify the concept exists in UNFCCC documents")

        formatted_output = "\n".join(output_parts)

        # Safety net: if total output is still too large, truncate it
        if len(formatted_output) > self.MAX_OUTPUT_CHARS:
            print(f"Multi-stage output too large ({len(formatted_output)} chars). Truncating.")
            formatted_output = self._truncate_content(formatted_output, self.MAX_OUTPUT_CHARS)

        # Return combined results and formatted output
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
                print("Detected Neo4j configuration")
                print(f"Initializing Neo4j connection to {neo4j_uri}")

                GraphSearchTool._driver_instance = Neo4jDriver(
                    uri=neo4j_uri,
                    user=neo4j_user,
                    password=neo4j_password,
                )

                GraphSearchTool._graphiti_instance = Graphiti(
                    graph_driver=GraphSearchTool._driver_instance
                )

                print("Successfully initialized Neo4j graph connection")

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

                print("Detected FalkorDB configuration (or using defaults)")
                print(f"Initializing FalkorDB connection to {host}:{port}, database: {database}")

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

                print("Successfully initialized FalkorDB graph connection")

        except Exception as e:
            # Provide detailed error message based on which DB we tried to connect to
            if GraphSearchTool._db_type == "neo4j":
                error_msg = (
                    f"Failed to initialize Neo4j connection: {str(e)}\n"
                    "Connection details:\n"
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
                    "Connection details:\n"
                    f"  Host: {os.getenv('FALKORDB_HOST', 'localhost')}\n"
                    f"  Port: {os.getenv('FALKORDB_PORT', '6379')}\n"
                    f"  Database: {os.getenv('GRAPH_NAME', 'unfccc_knowledge_graph')}\n"
                    "Please verify:\n"
                    "  1. FalkorDB is running and accessible\n"
                    "  2. FALKORDB_HOST, FALKORDB_PORT are correctly set in .env\n"
                    "  3. Network connectivity to FalkorDB server"
                )

            print(error_msg)
            raise ConnectionError(error_msg) from e

    async def _search_graph(self, query: str, limit: int) -> List[Any]:
        """
        Search graph for general facts and relationships.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of search results
        :raises: Exception if search fails
        """
        try:
            print(f"Searching graph: query='{query[:50]}...', limit={limit}")
            results = await GraphSearchTool._graphiti_instance.search(
                query=query, num_results=limit
            )
            result_count = len(results) if results else 0
            print(f"Graph search returned {result_count} results")
            return results or []
        except Exception as e:
            print(f"Error during graph search: {e}")
            raise

    async def _search_entities(self, query: str, limit: int) -> List[Any]:
        """
        Search for entity nodes in the graph.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of entity nodes
        :raises: Exception if search fails
        """
        try:
            print(f"Searching entities: query='{query[:50]}...', limit={limit}")
            config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = limit

            results = await GraphSearchTool._graphiti_instance._search(
                query=query, config=config
            )

            node_count = len(results.nodes) if results and results.nodes else 0
            print(f"Entity search returned {node_count} nodes")
            return results.nodes if results and results.nodes else []
        except Exception as e:
            print(f"Error during entity search: {e}")
            raise

    async def _search_relationships(self, query: str, limit: int) -> List[Any]:
        """
        Search for relationship edges in the graph.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of relationship edges
        :raises: Exception if search fails
        """
        try:
            print(f"Searching relationships: query='{query[:50]}...', limit={limit}")
            config = EDGE_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = limit

            results = await GraphSearchTool._graphiti_instance._search(
                query=query, config=config
            )

            edge_count = len(results.edges) if results and results.edges else 0
            print(f"Relationship search returned {edge_count} edges")
            return results.edges if results and results.edges else []
        except Exception as e:
            print(f"Error during relationship search: {e}")
            raise

    async def _search_episodes(self, query: str, limit: int) -> List[Any]:
        """
        Search for episode nodes (primary source documents) in the graph.

        Episodes contain the original document text and metadata, providing
        direct access to primary information from UNFCCC documents.

        **ENHANCED**: Expands query with UNFCCC terminology and synonyms for better retrieval.

        :param query: Search query text
        :param limit: Maximum number of results
        :return: List of episode nodes
        :raises: Exception if search fails
        """
        try:
            # **ENHANCED SEARCH QUALITY**: Expand query with UNFCCC-specific terminology
            expanded_query = self._expand_query_terms(query)
            print(f"Searching episodes: original='{query[:50]}...', expanded='{expanded_query[:70]}...'")

            # Use combined search config (searches all types, including episodes)
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = limit

            results = await GraphSearchTool._graphiti_instance._search(
                query=expanded_query, config=config
            )

            episode_count = len(results.episodes) if results and results.episodes else 0
            print(f"Episode search returned {episode_count} episodes")
            return results.episodes if results and results.episodes else []
        except Exception as e:
            print(f"Error during episode search: {e}")
            raise

    async def _search_by_decision(self, decision_id: str, query: str) -> str:
        """
        Search for a specific decision by its ID in episode metadata.

        Uses two strategies:
        1. Semantic search + metadata matching (fast but may miss some)
        2. Direct Cypher query on episode names/content (fallback, more reliable)

        :param decision_id: Decision ID (e.g., "3/CMA.1", "17/CP.22")
        :param query: Original query for context
        :return: Formatted decision text or empty string if not found
        """
        _ = query

        try:
            # Strategy 1: Semantic search + metadata matching
            search_query = f"decision {decision_id}"
            print(f"Searching for decision: {decision_id}")

            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 10

            search_results = await GraphSearchTool._graphiti_instance._search(
                query=search_query, config=config
            )

            episodes = search_results.episodes if search_results and search_results.episodes else []
            print(f"Found {len(episodes)} candidate episodes via semantic search")

            # Check metadata for exact decision_id match
            for episode in episodes:
                if not hasattr(episode, 'metadata') or not episode.metadata:
                    continue

                episode_decision_id = episode.metadata.get("decision_id", "")
                if episode_decision_id and episode_decision_id.strip().upper() == decision_id.strip().upper():
                    print(f"Found exact match via semantic search: {episode.name if hasattr(episode, 'name') else 'Unknown'}")
                    return self._format_decision_result(decision_id, episode)

            # Strategy 2: Direct Cypher query as fallback
            print(f"Semantic search didn't find Decision {decision_id}, trying direct database query...")
            direct_result = await self._search_decision_by_cypher(decision_id)
            if direct_result:
                return direct_result

            print(f"WARNING: Decision {decision_id} not found by any method")
            return ""

        except Exception as e:
            print(f"Error during decision search: {e}")
            traceback.print_exc()
            return ""

    def _format_decision_result(self, decision_id: str, episode) -> str:
        """Format a found decision episode into a readable result."""
        episode_content = episode.content if hasattr(episode, 'content') else str(episode)
        metadata = episode.metadata if hasattr(episode, 'metadata') and episode.metadata else {}

        conference = metadata.get("conference_name", "Unknown")
        year = metadata.get("year", "Unknown")
        location = metadata.get("location", "Unknown")

        action_type = metadata.get("decision_action_type", "")
        action_label = f" [{action_type.upper()}]" if action_type else ""

        formatted_result = f">>> DECISION {decision_id} ({year}){action_label} <<<\n"
        formatted_result += f"Conference: {conference}\n"
        formatted_result += f"Year: {year}, Location: {location}\n"
        formatted_result += "\nEXACT TEXT FROM SOURCE:\n"
        formatted_result += "-" * 80 + "\n"
        formatted_result += f"{episode_content}\n"

        return formatted_result

    async def _search_decision_by_cypher(self, decision_id: str) -> str:
        """
        Search for a decision directly in the database using Cypher query.

        Searches episode names and content for the decision ID string.
        This is a reliable fallback when semantic search misses the episode.

        :param decision_id: Decision ID (e.g., "17/CP.22")
        :return: Formatted result or empty string
        """
        if not GraphSearchTool._driver_instance:
            return ""

        try:
            # Normalize: try both with and without spaces around /
            decision_id_clean = decision_id.strip().upper()
            # Search by episode name containing the decision ID
            cypher_query = """
            MATCH (e:Episodic)
            WHERE toUpper(e.name) CONTAINS $decision_id
               OR toUpper(e.content) CONTAINS $search_term
            RETURN e.name as name, e.content as content, e.source_description as source_description
            LIMIT 1
            """
            search_term = f"DECISION {decision_id_clean}"
            print(f"Direct Cypher query for: {search_term}")

            records, _, _ = await GraphSearchTool._driver_instance.execute_query(
                cypher_query,
                decision_id=decision_id_clean,
                search_term=search_term,
            )

            if records:
                record = dict(records[0])
                content = record.get("content", "")
                name = record.get("name", "Unknown")
                source_desc = record.get("source_description", "")

                print(f"Found decision via direct Cypher: {name}")

                formatted_result = f">>> DECISION {decision_id} <<<\n"
                formatted_result += f"Source: {source_desc or name}\n"
                formatted_result += "\nEXACT TEXT FROM SOURCE:\n"
                formatted_result += "-" * 80 + "\n"
                formatted_result += f"{content}\n"

                return formatted_result

            print(f"Direct Cypher query found no results for Decision {decision_id}")
            return ""

        except Exception as e:
            print(f"Error during direct Cypher search: {e}")
            traceback.print_exc()
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
        decision_pattern = r'\b(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)\b'
        decision_match = re.search(decision_pattern, query, re.IGNORECASE)
        decision_id = decision_match.group(1) if decision_match else None

        # Try to find episodes that might contain these paragraphs
        # Strategy: search for decision ID or use broader search
        if decision_id:
            search_query = f"decision {decision_id}"
        else:
            # Extract key terms from query for broader search
            search_query = query[:100]

        print(f"Searching for episodes containing decision: {decision_id or 'unknown (using query)'}")

        try:
            # Use combined search to find episodes
            config = COMBINED_HYBRID_SEARCH_RRF.model_copy(deep=True)
            config.limit = 10  # Get more candidates to find the right decision

            search_results = await GraphSearchTool._graphiti_instance._search(
                query=search_query, config=config
            )

            episodes = search_results.episodes if search_results and search_results.episodes else []
            print(f"Found {len(episodes)} candidate episodes to search for paragraphs")

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
                    if not hasattr(episode, 'metadata') or not episode.metadata:
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
                        episode_name = episode.name if hasattr(episode, 'name') else 'Unknown'
                        decision_from_metadata = episode.metadata.get("decision_id", "Unknown")

                        formatted_result = f"**Paragraph {para_id}** from {decision_from_metadata}\n"
                        formatted_result += f"Source: {episode_name}\n"
                        formatted_result += f"\n{para_text}\n"

                        results.append(formatted_result)
                        found = True
                        print(f"Found paragraph {para_id} in episode: {episode_name}")
                        break

                if not found:
                    print(f"WARNING: Could not find paragraph {para_id} in any episode metadata")

        except Exception as e:
            print(f"Error during paragraph metadata search: {e}")
            traceback.print_exc()

        return results

    @staticmethod
    def _truncate_content(content: str, max_chars: int = 8000) -> str:
        """Truncate content to fit within context limits."""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n\n[... content truncated. Use a more specific query for full text ...]"

    def _expand_query_terms(self, query: str) -> str:
        """
        Expand query with UNFCCC synonyms and related terms for better document retrieval.

        Maps common terms to official UNFCCC terminology to improve matching.

        :param query: Original query string
        :return: Expanded query string with additional terms
        """
        query_lower = query.lower()
        expansions = []

        # UNFCCC terminology mappings
        term_mappings = {
            "mitigation co-benefits": ["co-benefits from mitigation", "mitigation benefits", "ancillary benefits"],
            "adaptation action": ["adaptation activities", "adaptation measures", "adaptation efforts"],
            "economic diversification": ["diversification plans", "economic transformation", "structural transformation"],
            "additional information": ["further information", "supplementary information", "more detailed information"],
            "Party": ["country", "nation", "State Party", "Parties"],
            "must provide": ["shall provide", "required to provide", "obligation to provide", "provide"],
            "transparency": ["transparency framework", "ETF", "enhanced transparency framework"],
            "reporting": ["report", "BTR", "biennial transparency report", "national communication"],
            "NDC": ["nationally determined contribution", "nationally determined contributions"],
            "annex": ["annexes", "appendix", "attachment"],
            "decision": ["decisions", "decision document"],
        }

        # Add expansion terms if key phrases are found
        for key_phrase, synonyms in term_mappings.items():
            if key_phrase in query_lower:
                # Add the most relevant synonyms (limit to avoid over-expansion)
                expansions.extend(synonyms[:2])

        # Combine original query with expansions
        if expansions:
            expanded = f"{query} {' '.join(expansions)}"
            return expanded
        return query

    def _filter_by_conference(self, results: List[Any], conference_type: str) -> List[Any]:
        """Filter episode results to only include episodes from the specified conference type.

        Checks episode metadata 'conference_type' and episode name/decision_id for
        conference type markers (CMA, COP/CP, CMP, SBI, SBSTA).

        :param results: List of episode results to filter
        :param conference_type: Conference type to filter by (e.g., "CMA", "COP", "CMP")
        :return: Filtered list of results matching the conference type
        """
        # Normalize conference type aliases
        conf_aliases = {
            "COP": ["COP", "CP"],
            "CMA": ["CMA"],
            "CMP": ["CMP"],
            "SBI": ["SBI"],
            "SBSTA": ["SBSTA"],
        }
        target_aliases = conf_aliases.get(conference_type, [conference_type])

        filtered = []
        for episode in results:
            metadata = getattr(episode, 'metadata', None) or {}

            # Check metadata conference_type field
            meta_conf = metadata.get('conference_type', '').upper()
            if meta_conf in target_aliases:
                filtered.append(episode)
                continue

            # Check decision_id pattern (e.g., "2/CMA.2" contains "CMA")
            decision_id = metadata.get('decision_id', '').upper()
            if any(alias in decision_id for alias in target_aliases):
                filtered.append(episode)
                continue

            # Check episode name for conference type
            name = (getattr(episode, 'name', '') or '').upper()
            if any(alias in name for alias in target_aliases):
                filtered.append(episode)
                continue

        # If filtering removed ALL results, return original (don't lose everything)
        if not filtered and results:
            print(f"WARNING: Conference filter '{conference_type}' removed all results. Keeping originals.")
            return results

        return filtered

    def _temporal_rerank(self, results: List[Any], direction: str) -> List[Any]:
        """Re-rank episode results by year for chronological queries.

        :param results: List of episode results to re-rank
        :param direction: 'earliest' to sort ascending, 'latest' to sort descending
        :return: Re-ranked list of results
        """
        def get_year(episode) -> int:
            metadata = getattr(episode, 'metadata', None) or {}
            year_str = metadata.get('year', '')
            if year_str:
                try:
                    return int(year_str)
                except (ValueError, TypeError):
                    pass
            name = getattr(episode, 'name', '') or ''
            year_match = re.search(r'(\d{4})', name)
            if year_match:
                y = int(year_match.group(1))
                if 2000 <= y <= 2030:
                    return y
            return 9999 if direction == "earliest" else 0

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
        creation_re = re.compile(
            r'(?i)\b(establishes?\b|creates?\b|decides\s+to\s+establish\b|'
            r'launches?\b|sets?\s+up\b|inaugurates?\b)'
        )
        followup_re = re.compile(
            r'(?i)\b(further\s+develops?\b|also\s+recalling\b|'
            r'builds?\s+on\b|welcomes?\s+the\s+continued\b|reaffirms?\b|'
            r'operationaliz\w+\b|decides\s+that\s+.{5,40}shall\s+have\b)'
        )

        founding = []
        followup = []

        for episode in episode_results:
            content = getattr(episode, 'content', '') or ''
            check_text = content[:3000]

            creation_count = len(creation_re.findall(check_text))
            followup_count = len(followup_re.findall(check_text))

            metadata = getattr(episode, 'metadata', None) or {}
            decision_id = metadata.get('decision_id', '')
            year = metadata.get('year', '?')
            name = getattr(episode, 'name', '') or ''

            if creation_count > 0 and creation_count >= followup_count:
                founding.append(episode)
                print(f"  FOUNDING: Decision {decision_id} ({year}) - {name[:60]} [creation={creation_count}, followup={followup_count}]")
            else:
                followup.append(episode)
                print(f"  FOLLOW-UP: Decision {decision_id} ({year}) - {name[:60]} [creation={creation_count}, followup={followup_count}]")

        # Sort founding by year ascending (earliest first)
        founding = self._temporal_rerank(founding, "earliest") if founding else []
        followup = self._temporal_rerank(followup, "earliest") if followup else []

        founding_ids = [getattr(e, 'metadata', {}).get('decision_id', '?') for e in founding if hasattr(e, 'metadata')]
        followup_ids = [getattr(e, 'metadata', {}).get('decision_id', '?') for e in followup if hasattr(e, 'metadata')]

        summary = f"Found {len(founding)} founding decision(s): {founding_ids} and {len(followup)} follow-up decision(s): {followup_ids}"

        return {
            "founding_episodes": founding,
            "followup_episodes": followup,
            "summary": summary,
        }

    async def _follow_backward_references(self, episode_results: List[Any]) -> List[str]:
        """Follow 'recalling' references in episode content to find earlier founding decisions.

        Parses episode content for backward references like "recalling decision X/Y.Z"
        and fetches those earlier decisions to help identify founding decisions.

        :param episode_results: List of episode results to scan for backward references
        :return: List of formatted decision result strings
        """
        referenced_decision_ids = set()
        for episode in episode_results:
            content = getattr(episode, 'content', '') or ''
            recall_patterns = [
                r'(?:recalling|recalls|also recalling|further recalling)\s+'
                r'(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)',
                r'(?:pursuant to|in accordance with)\s+'
                r'(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)',
            ]
            for pattern in recall_patterns:
                for match in re.finditer(pattern, content, re.IGNORECASE):
                    ref_id = match.group(1).upper()
                    referenced_decision_ids.add(ref_id)

        results = []
        for decision_id in list(referenced_decision_ids)[:5]:
            print(f"  Following backward reference to Decision {decision_id}...")
            result = await self._search_by_decision(decision_id, f"decision {decision_id}")
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
        # Search broadly for episodes on this topic
        episode_results = await self._search_episodes(query, limit)
        if not episode_results:
            return []

        # Apply conference filter if specified
        conf_filter = query_analysis.get("conference_filter")
        if conf_filter:
            episode_results = self._filter_by_conference(episode_results, conf_filter)

        # Collect decision info with year for sorting
        decisions = []
        seen_ids = set()
        for episode in episode_results:
            metadata = getattr(episode, 'metadata', None) or {}
            decision_id = metadata.get('decision_id', '')
            if not decision_id or decision_id in seen_ids:
                # Fall back to extracting from episode name
                name = getattr(episode, 'name', '') or ''
                name_match = re.search(
                    r'(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:\s*/\s*[A-Za-z]+\.\d+))',
                    name, re.IGNORECASE
                )
                if name_match:
                    decision_id = name_match.group(1).strip()
                else:
                    continue
            if decision_id in seen_ids:
                continue
            seen_ids.add(decision_id)

            year_str = metadata.get('year', '')
            year = int(year_str) if year_str else 9999
            content = getattr(episode, 'content', '') or ''

            # Classify as founding or follow-up
            creation_re = re.compile(
                r'(?i)\b(establishes?|creates?|decides\s+to\s+establish|'
                r'launches?|sets?\s+up)\b'
            )
            followup_re = re.compile(
                r'(?i)\b(further\s+develops?|also\s+recalling|'
                r'welcomes?\s+the\s+continued|reaffirms?)\b'
            )
            creation_count = len(creation_re.findall(content[:2000]))
            followup_count = len(followup_re.findall(content[:2000]))
            if creation_count > 0 and creation_count >= followup_count:
                action_type = "FOUNDING"
            elif followup_count > creation_count:
                action_type = "FOLLOW-UP"
            else:
                action_type = "RELATED"

            # Extract first substantive sentence as summary
            first_para = content[:500].split('\n')
            summary_line = ""
            for line in first_para:
                line = line.strip()
                if len(line) > 30 and not line.startswith("Conference"):
                    summary_line = line[:200]
                    break

            conference = metadata.get('conference_type', '')
            session = metadata.get('session_number', '')
            location = metadata.get('location', '')
            formatted = (
                f"[{year}] Decision {decision_id} ({conference} Session {session}"
                f"{', ' + location if location else ''}) "
                f"— {action_type}: {summary_line}"
            )
            decisions.append((year, formatted))

        # Sort by year ascending
        decisions.sort(key=lambda x: x[0])

        # Also follow backward references to find any missing earlier decisions
        backward_ids = set()
        for episode in episode_results:
            content = getattr(episode, 'content', '') or ''
            for match in re.finditer(
                r'(?:recalling|also recalling)\s+(?:decision|resolution)\s+(\d+/[A-Z]+\.\d+)',
                content, re.IGNORECASE
            ):
                ref_id = match.group(1).upper()
                if ref_id not in seen_ids:
                    backward_ids.add(ref_id)

        # Fetch missing backward-referenced decisions
        for ref_id in list(backward_ids)[:5]:
            result = await self._search_by_decision(ref_id, f"decision {ref_id}")
            if result:
                decisions.insert(0, (0, f"[EARLIER] Decision {ref_id} — Referenced by later decisions (search for full text)\n{result[:300]}"))

        return [d[1] for d in decisions]

    def _validate_citation(
        self, claim: str, source_text: str, metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that a claim can be supported by the source text.

        Checks if key terms from the claim appear in the source text and
        returns confidence score and matching excerpts.

        :param claim: The statement/claim to validate
        :param source_text: The source document text
        :param metadata: Document metadata (decision_id, annex, etc.)
        :return: Validation result with confidence score and evidence
        """
        claim_lower = claim.lower()
        source_lower = source_text.lower()

        # Extract key terms from claim (nouns, verbs, adjectives)
        # Remove common words
        common_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "has", "have", "had", "do", "does", "did", "will", "would", "should",
            "could", "may", "might", "must", "shall", "can"
        }

        # Extract words, filter common words and short words
        claim_words = [
            word for word in re.findall(r'\b\w+\b', claim_lower)
            if word not in common_words and len(word) > 3
        ]

        # Check how many key terms appear in source
        matches = sum(1 for word in claim_words if word in source_lower)
        match_ratio = matches / len(claim_words) if claim_words else 0

        # Find excerpts containing key terms
        excerpts = []
        for word in claim_words[:5]:  # Check top 5 key terms
            pattern = re.compile(r'.{0,100}\b' + re.escape(word) + r'\b.{0,100}', re.IGNORECASE | re.DOTALL)
            found = pattern.search(source_text)
            if found:
                excerpt = found.group(0).strip()
                # Clean up the excerpt
                excerpt = ' '.join(excerpt.split())  # Normalize whitespace
                if excerpt not in excerpts:
                    excerpts.append(excerpt)

        # Calculate confidence score
        confidence = "high" if match_ratio >= 0.6 else "medium" if match_ratio >= 0.3 else "low"

        validation_result = {
            "is_supported": match_ratio >= 0.3,
            "confidence": confidence,
            "match_ratio": round(match_ratio, 2),
            "matched_terms": matches,
            "total_terms": len(claim_words),
            "excerpts": excerpts[:3],  # Limit to top 3 excerpts
            "metadata": metadata,
        }

        return validation_result

    def _extract_claims_from_fact(self, fact: str) -> List[str]:
        """
        Extract individual verifiable claims from a compound fact statement.

        Splits facts on conjunctions and punctuation to isolate individual claims.

        :param fact: The fact statement to break down
        :return: List of individual claims
        """
        # Split on common conjunctions and semicolons
        claims = re.split(r'[;]|\band\b|\bor\b|\balso\b|\bfurthermore\b|\badditionally\b', fact, flags=re.IGNORECASE)

        # Clean and filter
        claims = [claim.strip() for claim in claims if claim.strip() and len(claim.strip()) > 20]

        # If no splits, return original fact
        if not claims:
            claims = [fact]

        return claims

    async def _format_facts(self, query: str, results: List[Any]) -> str:
        """
        Format general graph facts with SOURCE CITATIONS and connected node details.

        **ENHANCED**: Includes episode sources, metadata, and citation guidance.

        :param query: Original search query
        :param results: List of graph fact results
        :return: Formatted string representation
        """
        if not results:
            return f"No graph facts found matching query: {query}"

        output_parts: List[str] = [
            f"SEARCH RESULTS FOR: '{query}'",
            f"Found {len(results)} knowledge graph fact(s)\n",
            "=" * 80,
            "",
            "NOTE: These facts are extracted from UNFCCC documents.",
            "For exact text and formal citations, search type='episode' is recommended.",
            "",
            "=" * 80,
            "",
        ]

        for i, result in enumerate(results, 1):
            fact: str = result.fact
            name: str = getattr(result, "name", "")
            source_uuid: Optional[str] = getattr(result, "source_node_uuid", None)
            target_uuid: Optional[str] = getattr(result, "target_node_uuid", None)
            episode_uuids: List[str] = getattr(result, "episode_uuids", [])
            created_at = getattr(result, "created_at", None)
            valid_at = getattr(result, "valid_at", None)

            output_parts.append(f"[FACT {i}]")
            if name:
                output_parts.append(f"Relationship Type: {name}")
            output_parts.append(f"Statement: {fact}")

            # **ENHANCED**: Show source document episodes if available
            if episode_uuids:
                output_parts.append(f"\nSource Documents: {len(episode_uuids)} episode(s)")
                # Fetch episode metadata for proper citations
                for ep_uuid in episode_uuids[:3]:  # Limit to first 3 for brevity
                    ep_metadata = await self._get_episode_metadata(ep_uuid)
                    if ep_metadata:
                        citation = self._build_citation(
                            ep_metadata.get("name", ""),
                            ep_metadata.get("metadata", {}),
                            ep_metadata.get("source_description", "")
                        )
                        if citation:
                            output_parts.append(f"  • {citation}")

            # Temporal context if available
            if valid_at:
                output_parts.append(f"Valid as of: {valid_at}")
            elif created_at:
                output_parts.append(f"Recorded: {created_at}")

            # Get connected node details
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
                            output_parts.append(f"       {source_node['summary']}")
                    if target_node:
                        output_parts.append(
                            f"  To: {target_node.get('name', 'Unknown')}"
                        )
                        if target_node.get("summary"):
                            output_parts.append(f"      {target_node['summary']}")

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        # Add usage guidance
        output_parts.append("")
        output_parts.append("HOW TO USE THESE RESULTS:")
        output_parts.append("- These facts summarize relationships extracted from documents")
        output_parts.append("- For exact wording and formal citations, use search_type='episode'")
        output_parts.append("- Verify critical claims by checking the source documents listed above")
        output_parts.append("- Temporal context (dates) indicates when the information was valid")

        return "\n".join(output_parts)

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
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(query, uuid=uuid)
            if records:
                return dict(records[0])
            return {}
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch node {uuid}: {e}")
            return {}

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

            # Get connected relationships
            if uuid:
                connections: List[Dict[str, Any]] = await self._get_entity_connections(
                    uuid
                )
                if connections:
                    output_parts.append(
                        f"\nConnected to ({len(connections)} relationships):"
                    )
                    for (
                        conn
                    ) in connections:  # Show all connections (limited to 5 in query)
                        output_parts.append(
                            f"  - {conn['relationship']}: {conn['target_name']}"
                        )

            output_parts.append("")
            output_parts.append("-" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

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
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(query, uuid=uuid)
            return [dict(record) for record in records]
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch connections for entity {uuid}: {e}")
            return []

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

            # Get connected entity details
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

        # Calculate total content size to determine per-episode budget
        total_content_chars = sum(len(getattr(ep, "content", "") or "") for ep in results)
        needs_truncation = total_content_chars > self.MAX_OUTPUT_CHARS

        # If truncating, divide budget evenly across episodes
        if needs_truncation and len(results) > 0:
            per_episode_limit = self.MAX_OUTPUT_CHARS // len(results)
            print(f"Total content: {total_content_chars} chars > {self.MAX_OUTPUT_CHARS} limit. Truncating each episode to ~{per_episode_limit} chars.")
        else:
            per_episode_limit = 0  # No limit

        output_parts: List[str] = [
            f"SEARCH RESULTS FOR: '{query}'",
            f"Found {len(results)} source document(s)\n",
            "=" * 80,
            "",
        ]

        for i, episode in enumerate(results, 1):
            name = getattr(episode, "name", "Unnamed Episode")
            content = getattr(episode, "content", "") or ""
            source_description = getattr(episode, "source_description", "")

            # Get metadata from episode object first (most reliable), then fall back to DB query
            ep_metadata = getattr(episode, 'metadata', None) or {}
            db_metadata: Dict[str, Any] = await self._get_episode_metadata(
                getattr(episode, "uuid", "")
            )
            # Merge: episode object metadata takes priority
            metadata = {**db_metadata, **{k: v for k, v in ep_metadata.items() if v}}

            # Build formal citation
            citation = self._build_citation(name, metadata, source_description)

            # Extract key identifiers for prominent header
            decision_id = metadata.get("decision_id", "")
            annex_id = metadata.get("annex_id", "")
            year = metadata.get("year", "")
            conf_type = metadata.get("conference_type", "")
            action_type = metadata.get("decision_action_type", "")

            # Prominent header so agent can't miss decision IDs
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
                output_parts.append(f">>> ANNEX {annex_id} to Decision {decision_id} <<<")
            output_parts.append(f"CITATION: {citation}")
            if conf_type:
                output_parts.append(f"Conference: {conf_type} {year}")
            output_parts.append("")
            output_parts.append("EXACT TEXT FROM SOURCE:")
            output_parts.append("-" * 80)

            if content:
                if needs_truncation:
                    output_parts.append(self._truncate_content(content.strip(), per_episode_limit))
                else:
                    output_parts.append(content.strip())
            else:
                output_parts.append("[No content available in database]")

            output_parts.append("")
            output_parts.append("=" * 80)
            output_parts.append("")

        return "\n".join(output_parts)

    def _build_citation(self, episode_name: str, metadata: Dict[str, Any],
                       source_description: str) -> str:
        """
        Build a formal citation string for an episode.

        :param episode_name: The episode name from the database
        :param metadata: Episode metadata dictionary
        :param source_description: Human-readable source description
        :return: Formatted citation string
        """
        citation_parts = []

        # Extract decision/resolution from name if not in metadata
        decision_id = metadata.get("decision_id")
        if not decision_id and "::" in episode_name:
            # Try to extract from episode name
            title_part = episode_name.split("::", 1)[1] if "::" in episode_name else ""
            if "Decision" in title_part or "Resolution" in title_part:
                # Extract decision number from title
                match = re.search(
                    r"(?:Decision|Resolution)\s+(?:No\.?\s*)?([\dIVXLC]+(?:/[A-Za-z]+\.\d+)?)",
                    title_part
                )
                if match:
                    decision_id = match.group(1)

        if decision_id:
            citation_parts.append(f"Decision {decision_id}")

        # Add annex if present
        annex_id = metadata.get("annex_id")
        if annex_id:
            citation_parts.append(f"Annex {annex_id}")

        # Add conference info
        conf_type = metadata.get("conference_type")
        session = metadata.get("session_number")
        year = metadata.get("year")

        if conf_type and session and year:
            citation_parts.append(f"{conf_type} Session {session} ({year})")
        elif source_description:
            citation_parts.append(source_description)

        if citation_parts:
            return ", ".join(citation_parts)
        # Fallback to episode name
        return f"Source: {episode_name}"

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
            MATCH (e:Episode {uuid: $uuid})
            RETURN e.decision_id as decision_id,
                   e.conference_type as conference_type,
                   e.year as year,
                   e.session_number as session_number,
                   e.annex_id as annex_id
            LIMIT 1
            """
            records, _, _ = await GraphSearchTool._driver_instance.execute_query(query, uuid=uuid)
            if records:
                return dict(records[0])
            return {}
        # pylint: disable=broad-exception-caught
        except Exception as e:
            print(f"Warning: Failed to fetch episode metadata {uuid}: {e}")
            return {}
