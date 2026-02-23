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
Modified versions of Graphiti's entity and fact extraction prompts.
"""

from typing import Any

from graphiti_core.prompts.models import Message


def extract_text_constrained(context: dict[str, Any]) -> list[Message]:
    """
    Modified version of Graphiti's entity extraction prompt.
    This prompt is designed to extract only entities explicitly mentioned
    in the text.
    """

    sys_prompt = """
You are an AI assistant that extracts entity nodes from climate conference
documents. Your ONLY task is to extract entities that are EXPLICITLY MENTIONED
in the provided text.

CRITICAL RULES:
1. DO NOT use your general knowledge about climate, politics, or world events.
2. DO NOT infer entities that are not explicitly written in the text.
3. ONLY extract entities that appear as specific words or phrases in the \
document.
4. If an entity is not mentioned by name in the text, DO NOT extract it.

PROHIBITIONS (NEVER EXTRACT THESE):
- "Conference of the Parties" or any variation (e.g., "Conference of the \
Parties serving as the meeting of the Parties to the Paris Agreement or \
Conference of the Parties serving as the meeting of the Parties to the Kyoto \
Protocol") or "CP" or "CMP" or "CMA" when they refer to the body itself.
- "Subsidiary Body for Implementation" or "SBI" when it refers to the body \
itself.
- "Subsidiary Body for Scientific and Technological Advice" or "SBSTA" when \
it refers to the body itself.
- Session numbers, meeting descriptors, or procedural language.
- These are document metadata, NOT entities to extract.
"""

    user_prompt = f"""
<ENTITY TYPES>
{context["entity_types"]}
</ENTITY TYPES>

<TEXT>
{context["episode_content"]}
</TEXT>

Given the above text, extract entities that are "EXPLICITLY MENTIONED" \
in the TEXT. You must ONLY extract entities whose names appear as actual \
words or phrases in the provided text.

For each entity extracted, also determine its entity type based on the \
provided ENTITY TYPES and their descriptions. Indicate the classified \
entity type by providing its entity_type_id.

{context["custom_prompt"]}

PURPOSE OF EXTRACTION:
You are extracting entities for question-answering. When a user asks a \
question, the graph will find relevant relationships, then use the \
descriptions present in the entities/concepts to answer questions.

CRITICAL: Extract GENEROUSLY. The more entities you extract, the easier \
it is to map questions to the right information. It's better to extract \
too many entities than to miss important concepts.

WHAT TO EXTRACT (text explicitly mentioned in the documents):

1. Structural References: Decision/Resolution IDs, Article numbers, \
Annex references. Examples: "Decision 1/CP.21", "decision 1/CP.21" \
(lowercase too), "Article 6", "Annex I".

2. Named Entities: Specific agreements, organizations, funds, countries, \
bodies, committees. Examples: "Paris Agreement", "Green Climate Fund", \
"Morocco", "Standing Committee on Finance".

3. ALL Important Concepts: Extract EVERY substantive topic, mechanism, or \
term in the text:
- All substantive topics and themes: Every topic the text discusses
- All named entities: Every country, organization, agreement, mechanism, \
fund, body, committee, people, groups and more.
- All references: Every decision, article, annex, or document referenced
- Climate topics: mitigation, adaptation, finance, transparency, technology, \
capacity-building
- Policy mechanisms and governance concepts: NDCs, NAPs, work programmes, \
frameworks, guidelines, modalities
- Sectors: energy, forestry, oceans, agriculture, water, transport, buildings
- Groups: developing countries, vulnerable communities, country groups, \
indigenous people, LDCs, SIDS, VNS, G77, G20, important people, etc.
- Processes: reporting, review, stocktake, compliance, implementation, \
monitoring, treaties, guidelines, methodologies, discussions
- Financial: funds, mobilization, flows, support, resources, contributions
- Technical: emissions, renewable energy, technology transfer, assessments
- ANY concept, term, or topic that appears - if in doubt, please extract it!

4. EXTRACTION STRATEGY - BE COMPREHENSIVE:
- Extract both general concepts AND specific instances - e.g., extract \
"climate finance" AND "Green Climate Fund" AND "financial resources"
- Extract multi-word phrases when they form meaningful concepts - e.g., \
"technology needs assessments", "capacity-building initiatives"
- Extract both full names and common abbreviations if both appear - e.g., \
"nationally determined contributions" AND "NDCs"

5. GUIDELINES:
- If someone could ask "What decisions relate to X?", extract X as an entity
- Extract LIBERALLY and ABUNDANTLY - more entities = better retrieval = \
better answers
- Every paragraph should yield multiple entities
- Aim for comprehensive coverage so no information is missed

6. CRITICAL CONSTRAINTS:
- DO NOT extract generic role titles, institutional references, or group \
categories (e.g., "president", "government", "countries", "ministers", \
"delegates") unless they include specific identifying context (e.g., \
"President of France", "Government of Canada", "developing countries").
- DO NOT extract implied entities based on the document topic.
- DO NOT use your knowledge of climate politics to add entities.
- If a country, person, or organization is mentioned, extract its EXACT \
name from the text.
- Avoid creating nodes for relationships or actions.
- Avoid creating nodes for temporal information like dates, times or years \
(these will be added to edges later)
- Be as explicit as possible in your node names, using full names and \
avoiding abbreviations

7. ABSOLUTE PROHIBITIONS - DO NOT EXTRACT THESE AS ENTITIES:
- "Conference of the Parties" (COP) - This is a procedural body name, NOT \
an entity
- "Conference of the Parties serving as the meeting of the Parties to the \
Paris Agreement" (CMA) - This is just the full name of CMA, NOT an entity
- "Conference of the Parties serving as the meeting of the Parties to the \
Kyoto Protocol" (CMP) - This is just the full name of CMP, NOT an entity
- "Subsidiary Body for Implementation" - This is a procedural body, NOT an \
entity
- "Subsidiary Body for Scientific and Technological Advice" - This is a \
procedural body, NOT an entity
- Document headers/metadata (session numbers, report numbers, dates)
- Decision/Resolution IDs themselves (e.g., "Decision 15/CMA.1" is metadata, \
not an entity)
  * However, DO extract what the decision is ABOUT if explicitly stated
- Annex headers (e.g., "Annex to Decision 15/CMA.1" is metadata)

IF YOU SEE THESE PHRASES IN THE TEXT, SKIP THEM COMPLETELY. DO NOT CREATE \
ENTITY NODES FOR THEM.

GOAL: Create a DENSE entity graph where:
1. Every important concept is captured as an entity.
2. Questions can easily map to relevant concepts and comprehensively answer \
all questions about the text independently.
3. No information is lost due to missing entities.
4. No entities are missed due to assumptions or omissions.
5. No entities are created that are not explicitly mentioned in the text.

CRITICAL: ANNEX CONTENT IS SEPARATE FROM DECISION CONTENT:
If you are processing an annex episode (episode name contains "Annex to \
Decision X"):
- Extract entities from the annex content
- These entities belong to the ANNEX, NOT to the parent decision
- Do NOT create relationships between annex entities and the decision ID
- The annex EXPLAINS / SUPPLEMENTS the decision, but its entities are separate
Example: If "Financial Mechanism" appears in "Annex to Decision 15/CMA.1":
  - Extract "Financial Mechanism" as an entity
  - This entity appears in the ANNEX, not in the decision text itself
  - The relationship is:
  Financial Mechanism → mentioned in → ANNEX → connected to → Decision
  - The entities are indirectly connected through the ANNEX
  - CORRECT: "The ANNEX to Decision 15/CMA.1 mentions Financial Mechanism"
  - INCORRECT: "Decision 15/CMA.1 mentions Financial Mechanism"
"""
    return [
        Message(role="system", content=sys_prompt),
        Message(role="user", content=user_prompt),
    ]


def extract_edges_constrained(context: dict[str, Any]) -> list[Message]:
    """
    Modified version of Graphiti's fact extraction prompt.
    This prompt is designed to extract only relationships explicitly mentioned
    in the text.
    """
    return [
        Message(
            role="system",
            content=(
                """
You are an expert fact extractor that extracts fact triples ONLY from \
explicitly stated relationships in text. You must ONLY extract relationships \
that are directly stated in the text.
DO NOT infer relationships from your general knowledge.
DO NOT make assumptions about what entities might be related based on typical \
patterns.
1. Extracted fact triples should also be extracted with relevant date \
information when explicitly provided.
2. Treat CURRENT TIME as the time CURRENT MESSAGE was sent.
All temporal information should be extracted relative to this time.
"""
            ),
        ),
        Message(
            role="user",
            content=f"""
<FACT TYPES>
{context["edge_types"]}
</FACT TYPES>

<PREVIOUS_MESSAGES>
{list(context["previous_episodes"])}
</PREVIOUS_MESSAGES>

<CURRENT_MESSAGE>
{context["episode_content"]}
</CURRENT_MESSAGE>

<ENTITIES>
{context["nodes"]}
</ENTITIES>

<REFERENCE_TIME>
{context["reference_time"]}  # ISO 8601 (UTC); used to resolve relative \
time mentions
</REFERENCE_TIME>

# TASK
Extract ONLY factual relationships between the given ENTITIES that are \
EXPLICITLY STATED in the CURRENT MESSAGE.

## CRITICAL CONSTRAINTS:
- Only extract facts where the relationship is DIRECTLY STATED in the text.
- DO NOT infer relationships from your general knowledge.
- DO NOT assume connections because entities are typically related (e.g., \
don't connect "climate decision" to "president" just because presidents are \
involved in climate policy).
- Only extract facts that:
  * involve two DISTINCT ENTITIES from the ENTITIES list,
  * are EXPLICITLY STATED in the CURRENT MESSAGE (not implied, not inferred),
  * can be represented as edges in a knowledge graph.
- Facts should include entity names rather than pronouns whenever possible.
- The FACT TYPES provide a list of the most important types of facts, make \
sure to extract facts of these types.
- The FACT TYPES are not an exhaustive list, extract all facts from message \
even if they do not fit into one of the FACT TYPES.
- The FACT TYPES each contain their fact_type_signature which represents \
the source and target entity types.

You may use information from the PREVIOUS MESSAGES only to disambiguate \
references or support continuity.

{context["custom_prompt"]}

## EXTRACTION RULES

1. Entity ID Validation: `source_entity_id` and `target_entity_id` must use \
only the `id` values from the ENTITIES list provided above.
   - CRITICAL: Using IDs not in the list will cause the edge to be rejected.
2. Each fact must involve two distinct entities.
3. Use a SCREAMING_SNAKE_CASE string as the `relation_type` (e.g., \
REFERENCES, CITES, SUPPLEMENTS, HAS_ANNEX).
4. Do not emit duplicate or semantically redundant facts.
5. The `fact` should closely paraphrase the original source sentence(s).
Do not verbatim quote the original text.
6. Use `REFERENCE_TIME` to resolve vague or relative temporal expressions \
(e.g., "last week").
7. Do **not** hallucinate or infer temporal bounds from unrelated events.

## RELATIONSHIP TYPES FOR CLIMATE DOCUMENTS

### PURPOSE: Extract relationships that connect concepts to enable Q&A.
When a user asks "What decisions relate to climate finance?", the graph
should show which decisions explicitly mention/address/establish/relate \
to climate finance.

### CRITICAL: Extract MANY relationships. Every connection between entities \
should captured. Dense relationship mapping ensures comprehensive information \
retrieval.

Common explicit relationships to extract (when stated in text):

1. Structural Relationships:
  - Decision A REFERENCES Decision B (e.g., "Recalling decision 1/CP.21")
  - Decision A CITES Article X (e.g., "pursuant to Article 6")
  - Annex SUPPLEMENTS Decision (structural relationship)
  - Decision ADOPTS framework/guidance/mechanism

2. Topical Relationships (extract when explicitly stated):
  - Decision ADDRESSES concept (e.g., "decision on climate finance addresses \
mobilization")
  - Decision ESTABLISHES mechanism (e.g., "establishes the transparency \
  framework")
  - Decision RELATES_TO topic (when text explicitly discusses the topic)
  - Article DEFINES concept (e.g., "Article 6 defines cooperative approaches")
  - Decision INVITES/REQUESTS/URGES body (when explicitly stated)

3. Financial & Support Relationships:
  - Fund PROVIDES_SUPPORT_FOR concept (e.g., "Adaptation Fund provides \
support for adaptation")
  - Decision ESTABLISHES_FUND fund_name
  - Country/Group RECEIVES_SUPPORT (when explicitly mentioned)

4. Governance Relationships:
  - Body CONDUCTS process (e.g., "Standing Committee on Finance conducts \
review")
  - Decision MANDATES work_programme
  - Body REPORTS_TO parent_body

### EXAMPLES (EXTRACT RELATIONSHIPS COMPREHENSIVELY):
1. "Decision 15/CMA.1 addresses climate finance and technology transfer"
Extract ALL relationships:
- Decision 15/CMA.1 ADDRESSES climate finance
- Decision 15/CMA.1 ADDRESSES technology transfer

2. "The Standing Committee on Finance will review financial flows from \
developed to developing countries"
Extract ALL relationships:
- Standing Committee on Finance CONDUCTS review
- Standing Committee on Finance REVIEWS financial flows
- financial flows FLOW_FROM developed countries
- financial flows FLOW_TO developing countries

3. "Decision 1/CP.21 requests the IPCC to provide special reports on 1.5°C"
Extract ALL relationships:
- Decision 1/CP.21 REQUESTS IPCC
- IPCC PROVIDES special reports
- special reports RELATES_TO 1.5°C goal

## CRITICAL: FOUNDING vs FOLLOW-UP DISTINCTION
When extracting relationships, distinguish between CREATION and FOLLOW-UP actions:

FOUNDING relationships (use these relation_types):
- ESTABLISHES: "Establishes the Santiago network", "Decides to establish a committee"
- CREATES: "Creates a new fund", "Sets up a mechanism"
- LAUNCHES: "Launches the Koronivia joint work"
→ Set valid_at to the decision's reference_time (marks creation date)

FOLLOW-UP relationships (use these relation_types):
- FURTHER_DEVELOPS: "Further develops the institutional arrangements"
- EXPANDS: "Decides that [X] shall have the following functions"
- OPERATIONALIZES: "Operationalizes the network by defining its structure"
- REVIEWS: "Reviews the mechanism and recommends changes"
→ Reference the original founding decision in the fact text when possible

Example:
- Text: "Establishes the Santiago network for averting loss and damage"
  → relation_type: ESTABLISHES (FOUNDING action)
- Text: "Decides that the Santiago network shall have the following functions"
  → relation_type: FURTHER_DEVELOPS (FOLLOW-UP action, not creation)
- Text: "Also recalling decision 2/CMA.2"
  → relation_type: REFERENCES (points backward to founding decision)

This distinction is CRITICAL for answering questions like
"Which decision first created X?" vs "Which decision expanded X?"

## EXTRACTION MANDATE:
- Extract ALL explicit relationships, not just the primary one
- If a decision discusses multiple topics, create multiple ADDRESSES / \
RELATES_TO edges
- If text mentions actions, create edges for those actions
- Extract liberally - dense relationships = better question answering

## CRITICAL: KEEP ANNEX ENTITIES SEPARATE FROM DECISION ENTITIES:
- If processing "Annex to Decision X/Y.Z" episode:
  * Entities mentioned in the annex belong to the ANNEX episode
  * Do NOT create edges like "Decision X/Y.Z MENTIONS entity" for entities \
only in the annex
  * Only the annex episode itself relates to the decision (structural \
relationship)
- Example: "Financial Mechanism" in "Annex to Decision 15/CMA.1"
  * CORRECT: Extract "Financial Mechanism" entity (belongs to annex episode)
  * WRONG: Create edge "Decision 15/CMA.1 MENTIONS Financial Mechanism"

## DATETIME RULES
- Use ISO 8601 with "Z" suffix (UTC) (e.g., 2025-04-30T00:00:00Z)
- If the fact is ongoing (present tense), set `valid_at` to REFERENCE_TIME
- If a change/termination is expressed, set `invalid_at` to the relevant \
timestamp
- Leave both fields `null` if no explicit or resolvable time is stated
- If only a date is mentioned (no time), assume 00:00:00
- If only a year is mentioned, use January 1st at 00:00:00
        """,
        ),
    ]
