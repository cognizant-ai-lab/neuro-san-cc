# Graph RAG

The **Graph RAG** agent network lets you ask plain-English questions about UNFCCC climate change decisions and get precise, cited answers. It searches a knowledge graph built from official UNFCCC documents — covering CMA, COP, CMP, SBI, and SBSTA decisions from 2014 to 2024 — and returns answers grounded in the actual text, with decision IDs and paragraph references.

---

## File

[graph_rag.hocon](../registries/graph_rag.hocon)

---

## Description

Think of this system as a research assistant that has read UNFCCC climate decision and can find connections across them instantly.

When you ask a question, the **UNFCCC_Research** agent figures out what you're looking for a specific decision, a concept, a timeline of how something evolved, and calls the `knowledge_graph_search` tool to search the knowledge graph. The tool doesn't just do a keyword match; it understands relationships between documents, knows which decision *created* something versus which one *expanded* it later, and can trace backward references (e.g., "Recalling Decision X...") to pinpoint the original source.

The agent then assembles the retrieved evidence into a clear, structured answer with exact citations like decision ID, paragraph number and will never invent a fact or mix up content between two different decisions.

---

## Prerequisites

This agent network requires the following setup:

### Python Dependencies

```bash
pip install graphiti-core neo4j
```

### Environment Variables

Fill in `coded_tools/graph_rag/.env` with your Neo4j credentials and OpenAI API key. A template is provided in `coded_tools/graph_rag/.env.example`.

### System Requirements

- A running **Neo4j Aura** instance with the UNFCCC knowledge graph already ingested
- The ingestion step must be completed once before the agent network can answer queries

> **FalkorDB support**: The system also supports FalkorDB as an alternative graph database backend. However, you will need to ingest the documents into FalkorDB yourself. See `coded_tools/graph_rag/falkordb_ingestion.py` for the ingestion tool.

For more information on database setup, see:

- [Graphiti Documentation](https://help.getzep.com/graphiti)

- [Neo4j Getting Started](https://neo4j.com/docs/getting-started/)

---

## Example Conversation

### Human

```text
Under the transparency and compliance architecture:
Who reviews the information used to track NDC progress?
What are they explicitly prohibited from assessing?
Where is NDC adequacy shielded?
How does flexibility remain self-determined?
Why does this structure prevent transformation into a compliance regime?
```

### AI (UNFCCC_Research)

```text
### Who reviews that information?
- A **technical expert review** reviews Parties’ information submitted under Article 13 (Decision **18/CMA.1**, Annex, para. **146(a)**).
- Its scope includes “**a review of the consistency of the information submitted by the Party under Article 13, paragraphs 7 and 9, of the Paris Agreement with these MPGs, taking into account the flexibility accorded to the Party under Article 13, paragraph 2**” (Decision **18/CMA.1**, Annex, para. **146(a)**).
- Its scope also includes “**consideration of the Party’s implementation and achievement of its NDC under Article 4 of the Paris Agreement**” (Decision **18/CMA.1**, Annex, para. **146(b)**).

### What are they explicitly prohibited from assessing?
**Article 15 Committee (implementation and compliance) — explicit content-limit**
- When the Committee initiates consideration in cases where a Party has not done certain things (including not having “communicated or maintained” an NDC; not having submitted mandatory reports; not having participated in the facilitative, multilateral consideration of progress; or not having submitted a mandatory Article 9.5 communication) (Decision **20/CMA.1**, Annex, para. **22(a)(i–iv)**), then:
- “**The consideration of the issues referred to in paragraph 22(a) above will not address the content of the contributions, communications, information and reports referred to in paragraph 22(a)(i–iv) above.**” (Decision **20/CMA.1**, Annex, para. **23**).

**Technical expert review teams — explicit prohibition**
- **Not in the provided excerpt.** (Decision **18/CMA.1**, Annex excerpt shown does not include an explicit “shall not assess” clause.)

### Where is NDC adequacy shielded?
**Not in the provided excerpt.**

(Decision **20/CMA.1**, Annex, para. **23** limits the Article 15 Committee from addressing “the content” of certain items in that specific consideration pathway, but the excerpt does not expressly use “adequacy” language.)

### How does flexibility remain self-determined?
- The technical expert review is defined to occur “**taking into account the flexibility accorded to the Party under Article 13, paragraph 2**” (Decision **18/CMA.1**, Annex, para. **146(a)**).
- The Article 15 Committee, when engaging in facilitative consideration (with the Party’s consent) for “significant and persistent inconsistencies,” “**shall take into account**” “**the flexibility provided in the provisions of the modalities, procedures and guidelines under Article 13 of the Paris Agreement for those developing country Parties that need it in the light of their capacities**” (Decision **20/CMA.1**, Annex, para. **22(b)**).

**Text explicitly stating that the Party itself determines (“self-determined”) its need for flexibility is not in the provided excerpt.**

### Why does this prevent transformation into a compliance regime? (limited to what is in the excerpt)
- **No sanctioning clause identified** in the excerpted scope description of technical expert review (Decision **18/CMA.1**, Annex, para. **146**).
- The Article 15 Committee has an express limitation that, for the para. 22(a) pathway, its consideration “**will not address the content**” of the NDC/reports/communications it references (Decision **20/CMA.1**, Annex, para. **23**).
```

### Human

```text
How did decisions adopted at different CMA sessions gradually operationalize the Paris Agreement from a political framework into a functioning implementation system?
```

### AI (UNFCCC_Research)

```text
1) **CMA 1 (2018) — decisions “adopt” implementation rules and set core system deadlines**
- **Decision 3/CMA.1 (2018) (decides/adopts)**: The CMA **decides** “to adopt” a package of draft decisions across Paris Agreement Articles (Decision 3/CMA.1, para. 4).
- **Decision 18/CMA.1 (2018) (adopts/requests/decides)**: The CMA **adopts** the “modalities, procedures and guidelines for the transparency framework for action and support” in an annex (Decision 18/CMA.1, para. 1); **decides** that Parties “shall submit their first biennial transparency report and national inventory report … at the latest by 31 December 2024” (Decision 18/CMA.1, para. 3); and **requests** SBSTA to undertake a first review and update of those MPGs “no later than 2028” (Decision 18/CMA.1, para. 2).
- **Decision 15/CMA.1 (2018) (adopts/decides/requests)**: The CMA **adopts** the “technology framework … as elaborated in the annex” (Decision 15/CMA.1, para. 1), **decides** that the TEC and CTCN “shall implement the technology framework … under the guidance of” the CMA (Decision 15/CMA.1, para. 2), and **requests** them to incorporate the framework into workplans and report on incorporation in a “joint annual report for 2019” (Decision 15/CMA.1, para. 3(a)–(b)).
- **Decision 4/CMA.1 (2018) (reaffirms/encourages/invites/decides)**: The CMA **reaffirms** that support “shall be provided to developing country Parties” for implementation of Article 4, including capacity to prepare/communicate/account for NDCs (Decision 4/CMA.1, para. 1), and **encourages** relevant operating entities/constituted bodies to provide such support and **invites** other organizations to do so (Decision 4/CMA.1, paras. 2–3).

2) **CMA 2 (2019) — decisions keep mechanisms on a governance/review track (loss and damage)**
- **Decision 2/CMA.2 (2019) (review-related framing in excerpt)**: The decision is on the Warsaw International Mechanism “and its 2019 review” (Decision 2/CMA.2, title), and the preamble **recalls** that a next review was recommended “to be held in 2019” (Decision 2/CMA.2, preambular text).

3) **CMA 3 (2021) — decisions add detailed operational tools, bodies, and scheduled work programmes**
- **Decision 5/CMA.3 (2021) (adopts/decides/requests/invites)**: The CMA **adopts** multiple standardized transparency tools: “common reporting tables,” “common tabular formats,” “outlines for the biennial transparency report … and technical expert review report,” and a “training programme for technical experts” (Decision 5/CMA.3, para. 1(a)–(e)); **decides** that review teams “will follow” the review report outline (Decision 5/CMA.3, para. 3); and **requests** the secretariat to develop electronic reporting tools with a test version “by June 2023” and a final version “by June 2024,” “subject to the timely availability of sufficient financial resources” (Decision 5/CMA.3, para. 8).
- **Decision 3/CMA.3 (2021) (adopts/designates/decides/requests)**: The CMA **adopts** “rules, modalities and procedures” for the Article 6.4 mechanism in an annex (Decision 3/CMA.3, para. 1), **designates** a supervising body and names it the “Supervisory Body” (Decision 3/CMA.3, para. 2), and **decides** the Supervisory Body shall hold “at least two meetings … in 2022” (Decision 3/CMA.3, para. 4). It also **requests** the Supervisory Body to develop provisions for multiple operational processes (including “methodologies,” “validation,” “registration,” “monitoring,” “verification and certification,” and “issuance”) (Decision 3/CMA.3, para. 5(a)).
- **Decision 1/CMA.3 (2021) (decides/requests; implementation dialogues and funding arrangements in excerpt)**: The CMA **decides to establish** the “Glasgow Dialogue” with an annual schedule “until it is concluded at [SBI’s] sixtieth session (June 2024)” (Decision 1/CMA.3, para. 73), and **requests** SBI to organize it “in cooperation with the Executive Committee of the Warsaw International Mechanism” (Decision 1/CMA.3, para. 74). The CMA also **decides** the Santiago network “will be provided with funds to support technical assistance” (Decision 1/CMA.3, para. 67) and includes a resource condition for secretariat actions: “subject to the availability of financial resources” (Decision 1/CMA.3, para. 97).
- **Decision 12/CMA.3 (2021) (calls upon/requests; support delivery mechanisms)**: The CMA **calls upon** developed country Parties to contribute to a robust eighth replenishment of the GEF to support developing countries implementing the Paris Agreement (Decision 12/CMA.3, para. 2) and **requests** the GEF to facilitate improved access to the Capacity-building Initiative for Transparency (Decision 12/CMA.3, para. 5).
- **Decision 23/CMA.3 (2021) (adopts; internal operating rules for a constituted expert body)**: The CMA **adopts** “the revised rules of procedure” of the Katowice Committee of Experts on the Impacts of the Implementation of Response Measures in an annex (Decision 23/CMA.3, para. 6).

4) **CMA 4 (2022) — decisions further institutionalize specific components and set near-term operationalization targets**
- **Decision 12/CMA.4 (2022) (decides/adopts/establishes)**: The CMA **decides** the Santiago network “will have” a defined structure including “a hosted secretariat,” “an Advisory Board,” and “a network of member organizations” (Decision 12/CMA.4, para. 3(a)–(c)); **adopts** the Santiago network terms of reference in “annex I” (Decision 12/CMA.4, para. 8); and **decides to establish** the Advisory Board (Decision 12/CMA.4, para. 9).
- **Decision 1/CMA.4 (2022) (welcome/affirms determination; operationalization timing in excerpt)**: The CMA **welcomes** adoption of decisions “establishing the institutional arrangements of the Santiago network … to enable its full operationalization” and **affirms its determination** “to select the host of the secretariat of the Santiago network by 2023” (Decision 1/CMA.4, para. 47).
- **Decision 3/CMA.4 (2022) (welcomes/notes)**: The CMA **welcomes** workshops held under the “Glasgow–Sharm el-Sheikh work programme on the global goal on adaptation” (Decision 3/CMA.4, para. 1) and **notes** that a “single annual report” will serve as input to further considerations (Decision 3/CMA.4, para. 4).
- **Decision 22/CMA.4 (2022) (adopt/requests; implementation planning + M&E)**: The CMA **adopts** a four-year action plan under the Glasgow work programme on Action for Climate Empowerment in an annex (Decision 22/CMA.4, para. 1) and **requests** the secretariat to include specified information in annual summary reports (Decision 22/CMA.4, para. 7).

**Dependency chain explicitly stated in the retrieved text**
- Secretarial actions under Decision 5/CMA.3 are conditional:
  - Decision 5/CMA.3, para. 8 (“requests the secretariat” to develop reporting tools) CONDITIONALLY activates  
    ↓ subject to  
  - “the timely availability of sufficient financial resources” (Decision 5/CMA.3, para. 8)  
    ↓ if condition is met: test version by June 2023; final version by June 2024 (Decision 5/CMA.3, para. 8)  
    ↓ if unfulfilled: the text does not specify the consequence.  
- Secretarial actions under Decision 1/CMA.3 are conditional:
  - Actions of the secretariat called for in the decision CONDITIONALLY activate  
    ↓ subject to  
  - “the availability of financial resources” (Decision 1/CMA.3, para. 97)  
    ↓ if condition is met: secretariat undertakes the actions (Decision 1/CMA.3, para. 97)  
    ↓ if unfulfilled: the text does not specify the consequence.
```

### Human

```text
Explain Decision 10/CMA.1.
```

### AI (UNFCCC_Research)

```text
**Decision 10/CMA.1 (2018, Katowice)** is titled: **“Modalities and procedures for the operation and use of a public registry referred to in Article 7, paragraph 12, of the Paris Agreement”** (Document: **FCCC/PA/CMA/2018/3/Add.1**).

Operative provisions in the excerpt:
1. The CMA **decides to establish** a **public registry for adaptation communications** to contain adaptation communications submitted by Parties (Decision 10/CMA.1, para. 1).
2. The CMA **decides to adopt** the **modalities and procedures** for that registry **as contained in the annex** (Decision 10/CMA.1, para. 2).
3. The CMA **decides** the registry **shall be made available by the secretariat**, together with the Article 4(12) registry, through a **registry portal with two parts**: adaptation communications and nationally determined contributions (Decision 10/CMA.1, para. 3).
4. The CMA **requests the secretariat** to (a) **develop a prototype by June 2019** and present it at an event alongside the **fiftieth sessions of the subsidiary bodies (June 2019)**, and (b) **operate the registry** and provide assistance on its use (Decision 10/CMA.1, para. 4(a)–(b)).
5. The CMA **decides to consider and conclude at its second session (December 2019)** whether the prototype shall serve as the public registry under Article 7(12) (Decision 10/CMA.1, para. 5).
6. The CMA **takes note** of estimated budgetary implications for the secretariat’s activities (Decision 10/CMA.1, para. 6).
```
---

## Architecture Overview

### Frontman Agent: UNFCCC_Research

- Primary entry point for user questions about UNFCCC climate documents.

- Calls `knowledge_graph_search` to search the knowledge graph before forming any answer.

- Synthesizes results into structured responses with per-decision attribution and paragraph-level citations.

- Enforces strict anti-hallucination rules: never invents decision numbers, never mixes content across decisions.

### Tools: knowledge_graph_search

This agent is a coded tool that can be called from the toolbox with the name `knowledge_graph_search`, which executes multi-stage hybrid searches over the UNFCCC knowledge graph using Graphiti.

#### Tool Arguments and Parameters

- `query`: The user's full natural language question (built-in query analysis extracts intent, conference type, decision IDs, and temporal direction automatically)

- `limit`: Maximum number of results to return (default: `10`)

- `search_type`: Type of search — `general`, `entity`, `relationship`, or `episode`

- `query_complexity`: Search depth — `direct` (1 result), `medium` (5 results), or `extensive` (15 results)

#### Multi-Stage Search Pipeline

- **Stage 1 — Entity Search**: Finds concepts, countries, funds, and mechanisms matching the query

- **Stage 2 — Episode Search**: Retrieves primary source document content with conference-type filtering; follows backward references to trace founding decisions and builds a topic timeline for governance evolution queries

- **Stage 3 — Relationship Search**: Finds edges connecting entities (e.g., ESTABLISHES, REFERENCES, ADDRESSES)

- **Stage 4 — Output Assembly**: Assembles results in priority order: exact decision match → founding decisions → follow-up decisions → earlier referenced decisions → full timeline → source documents → entities → relationships

---

## Debugging Hints

When developing or debugging the Graph RAG agent network, keep the following in mind:

- **Database Connection**: Verify that the Neo4j instance is running and reachable with the configured credentials.

- **Ingestion Required**: The knowledge graph must be populated before querying. Run the ingestion tool once using the `neo4j_ingestion` coded tool.

- **First Query Latency**: The first invocation initializes a singleton database connection, which may take several seconds. Subsequent queries reuse the same connection.

- **API Key Validation**: Ensure `OPENAI_API_KEY` is set and valid — it is used by both the coordinator agent and Graphiti's embedding and search operations.

- **Conference Type Filtering**: CMA, COP/CP, CMP, SBI, and SBSTA are completely separate decision tracks. Specify the conference type in your query for more precise results.

- **Tool Registration**: Confirm that `knowledge_graph_search` is correctly mapped to the `GraphSearchTool` coded tool in the registry.

### Common Issues

- **Empty Results**: The knowledge graph may not be ingested yet, or the query conference type does not match the indexed data

- **Connection Errors**: Check database host, port, and credentials in `coded_tools/graph_rag/.env`

- **Slow Initialization**: Normal on first query — the Graphiti client and database driver are initialized once per process

- **Wrong Decision Track**: If results are missing, try specifying "CMA", "COP", or "CMP" explicitly in your question

- **Import Errors**: Ensure `graphiti-core` and `neo4j` are installed

---

## Resources

- [Graphiti Documentation](https://help.getzep.com/graphiti)
  Complete guide to Graphiti's hybrid search, entity extraction, and knowledge graph capabilities.

- [Neo4j Documentation](https://neo4j.com/docs/)
  Setup, configuration, and querying for the Neo4j graph database backend.

- [UNFCCC Climate Documents](https://unfccc.int/documents)
  Official source for UNFCCC decision documents indexed in the knowledge graph.
