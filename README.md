# 🚗 AutoDiag Pro — Multi-Agent Automotive Diagnostic Platform

A production-style multi-agent system: report symptoms and fault codes, get a
complete diagnosis, safety verdict, cost estimate, and correct repair order —
produced by four specialist AI agents that reason, critique, and coordinate.

This README is written as a **learning path** — it walks through the project
in the order you'd actually build it: storage first, then agent
infrastructure, then the agents themselves, then reasoning and graph
algorithms, then the API, then deployment. Read top to bottom to understand
*why* each layer exists before the next one is added.

---

## 1. Storage Layer — SQLite (`db.py`)

Everything the system remembers lives in one SQLite file. No JSON files —
SQLite from the start, so reads/writes are atomic and queryable.

**Tables:**
| Table | Purpose |
|---|---|
| `feedback` | User 👍/👎 ratings per diagnosis |
| `episodic_memory` | Past diagnoses (only positively-rated ones get saved) |
| `semantic_memory` | Learned facts — vehicle make, user preferences |
| `procedural_memory` | Successful tool call sequences, reusable per fault code |
| `consolidation_log` | Audit trail of memory cleanup runs |

**Key patterns used:**
- `contextmanager` connection wrapper — every call opens, commits/rolls back, closes
- `PRAGMA journal_mode=WAL` — safe concurrent reads while writing
- `INSERT ... ON CONFLICT DO UPDATE` — upsert pattern for facts and procedures
- Write policies enforced at the DB layer: `upsert_fact()` rejects confidence < 0.5,
  `upsert_procedure()` rejects success_rate < 0.7 — bad data never gets in

Run `init_db()` once at import time — tables are created if they don't exist.

---

## 2. Agent Infrastructure — before any agent runs

These four files are the scaffolding every specialist agent is built on top of.

### `schemas.py` — structured output enforcement
Every request, result, and internal message is a Pydantic model:
`DiagnosticRequest`, `AgentHandoff`, `AgentResult`, `BlackboardEntry`,
`TraceStep`, `EvaluationResult`. Validators catch bad fault-code formats and
empty symptoms before an agent ever runs. Enums (`SeverityLevel`, `TaskType`,
`AgentStatus`) constrain values to fixed sets instead of free strings.

### `blackboard.py` — shared agent memory
Agents never call each other directly. They read and write to one shared
store keyed by `TaskType`. `threading.Lock()` makes writes safe if agents
ever run in parallel. Lower-confidence writes to an already-final task are
blocked — this is the redundancy guard: if the diagnostic agent already
wrote a high-confidence result, a duplicate low-confidence write is rejected
and logged.

### `agent_protocol.py` — structured handoffs
Instead of passing raw strings between orchestrator and agents, every task
is a typed `AgentHandoff` (task_id, task_type, payload, priority) and every
response is a typed `AgentResult` (status, output, confidence, tools_used).
Factory functions (`create_diagnostic_task`, `create_safety_task`, etc.)
keep task construction consistent.

### `tracer.py` — full observability
Logs every thought, tool call, and final answer with latency and simulated
cost. `detect_redundant_work()` scans the trace and flags when two agents
call the same tool with the same input — proof the blackboard pattern is
actually working. `flush_to_mlflow()` uploads the whole trace as a JSON
artifact at the end of a run.

### `base_agent.py` — the shared agent contract
Abstract base class (`abc.ABC`) all four specialist agents inherit from.
Provides:
- `run_with_retry()` — exponential backoff, 3 attempts, checks the
  blackboard first (skip work if another agent already solved this task)
- `check_stopping_criteria()` — hard iteration cap, prevents infinite ReAct loops
- `invoke_llm()` — wraps every LLM call with latency/cost logging
- `get_cost_report()` — per-agent cost tracking

Agents are constructed with their tool list already built —
`DiagnosticAgent(blackboard, tracer, tools)` — so each agent has no direct
dependency on Pinecone, SQLite, or anything outside its own reasoning loop.
Each specialist agent only has to implement one method: `run(task)`.

---

## 3. Guardrails — validate before anything runs

`guardrails.py` sits between the API and the agents.

**Input validation** (`validate_input`): rejects symptoms under 5 characters
or over 2000, blocks prompt-injection patterns ("ignore previous
instructions", "you are now"), and enforces OBD-II format
(`P/C/B/U` + 4 digits) before an agent ever sees the request.

**Output validation** (`validate_agent_output`): flags empty output or
suspiciously low confidence scores.

**Human-in-the-loop trigger** (`needs_human_review`): scans the diagnosis
and safety result for critical keywords or safety-category fault codes
(`C0xxx`, `B0xxx`) and forces escalation — this is the single most important
guardrail in the system, because a wrong "safe to drive" call is the
highest-cost failure mode.

**Tool-call validation** (`validate_tool_call`): blocks a fixed denylist —
this exists specifically because `run_diagnostic_code` executes real Python
in a sandbox, and no agent-generated tool call should ever touch the
filesystem or network.

---

## 4. Context Engineering — keeping agents fast and cheap

`context_manager.py` manages what actually goes into each LLM call.

- **Sliding window** — keep last N conversation turns, not the whole history
- **Truncation** — tool output capped at 500 chars before it enters a prompt
- **Stale observation filtering** — if a tool was called twice, only the
  latest observation survives in context; earlier ones are dropped
- **Summarization fallback** — if context still exceeds the token budget,
  an LLM call compresses history into 2-3 sentences before continuing

This is what keeps a 6-iteration ReAct loop from ballooning into a massive,
slow, expensive prompt by iteration 5.

---

## 5. Memory System — 4 types, persisted to SQLite

`memory_manager.py` wraps `db.py` into four distinct memory types, each with
a different write policy:

| Type | What it stores | Write policy |
|---|---|---|
| Short-term | Last 5 conversation turns | In-memory only, no persistence |
| Episodic | Past diagnoses | Only saved if the session was rated positively |
| Semantic | Vehicle facts, preferences | Only saved if confidence ≥ 0.5 |
| Procedural | Successful tool sequences | Only saved if success_rate ≥ 0.7 |

**Consolidation** runs every 10 episodes: removes near-duplicate episodes
(same symptoms/diagnosis prefix), decays anything untouched for 30+ days,
and logs the run to `consolidation_log`.

**Memory vs RAG** — this is a deliberate distinction in the design.
Memory (this layer) is *this system's own experience* — what it diagnosed
before, for this user. RAG (next section) is *external knowledge* — the
uploaded vehicle manuals. They're separate retrieval paths that both feed
into agent context.

---

## 6. Tools — what agents can actually do

`tools.py` — 6 original tools, upgraded to `StructuredTool` with typed
Pydantic input schemas (so the LLM can't pass malformed arguments):

| Tool | What it does |
|---|---|
| `fault_code_lookup` | 18-code OBD-II database — severity, causes, urgency, suggested fix |
| `safety_assessment` | 3-tier severity check (CRITICAL/HIGH/MEDIUM) against keyword lists |
| `repair_estimate` | Cost table lookup across 12 common components |
| `search_vehicle_manual` | Vector similarity search over uploaded manuals |
| `maintenance_schedule` | Make-specific service intervals (Toyota ≠ BMW ≠ default) |
| `run_diagnostic_code` | Sandboxed subprocess execution — 5s timeout, blocklist on os/sys/socket |

Plus graph-powered tools layered on top (see section 8) and:
`online_search` (DuckDuckGo fallback when a code isn't in the local DB) and
`ask_human` (queues a question for human-in-the-loop review).

---

## 7. LangChain — how the agents actually reason

Each specialist agent is built the same way:

```python
agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=prompt)
executor = AgentExecutor(agent=agent, tools=self.tools, max_iterations=6,
    handle_parsing_errors=True, return_intermediate_steps=True)
result = executor.invoke({"input": query})
```

- `create_react_agent` builds the Thought → Action → Observation loop from
  an LLM, a tool list, and a `PromptTemplate`
- `AgentExecutor` actually runs it, capping iterations and catching
  malformed LLM output instead of crashing
- `return_intermediate_steps=True` is what lets `tracer.py` log every step,
  not just the final answer

The four agents:

| Agent | Tools | Role |
|---|---|---|
| `DiagnosticAgent` | fault lookup, manual search, graph tools | Finds primary issue + likely causes |
| `SafetyAgent` | safety assessment, fault lookup, ask_human | Independent safety verdict — overrides everything on CRITICAL |
| `CostAgent` | repair estimate, repair order, Dijkstra path | Cost + correct repair sequencing |
| `ReflectionAgent` | none — direct LLM calls | Draft → Critique → Revise over all three outputs |

---

## 8. Graph Algorithms — where this goes beyond a normal RAG chatbot

Fault codes have real structure: some *cause* others, some *worsen*
existing issues, some *co-occur*. Modeling that as a graph instead of a
flat lookup table is what makes the diagnostic agent capable of real
reasoning instead of keyword matching. Three algorithms, three different
questions, over the *same* graph (`fault_graph.py`):

| Algorithm | Question it answers | Function |
|---|---|---|
| **BFS** | "What else is nearby?" — all codes within N hops | `bfs_related_codes()` |
| **DFS** | "What's the deepest root cause?" — follow `caused_by` all the way down | `get_root_cause_path()` |
| **Dijkstra** | "What's the *cheapest* path?" — weighted by diagnostic effort or real repair dollars | `dijkstra_min_cost_path()`, `dijkstra_cheapest_repair_path()` |

Dijkstra is the one worth explaining carefully: BFS gives you *nearest*,
DFS gives you *deepest*, neither optimizes for cost. Dijkstra runs over the
same edges but with weights — `EDGE_WEIGHTS` for diagnostic complexity, or
real `$` pulled from the repair cost table — and finds the minimum-cost
route between two reported codes.

Beyond the fault graph:

- **`vehicle_hierarchy.py`** — nested system hierarchy (Engine → Fuel
  System → Fuel Injectors), plus a repair *dependency* graph with
  **topological sort** — you can't fix a catalytic converter before fixing
  the O2 sensor that's fouling it; topo sort enforces that ordering
  automatically.
- **`chunk_graph.py`** — graph over document *chunks*, not fault codes.
  Chunks above a cosine-similarity threshold are connected. Retrieval
  returns a matched chunk *plus its graph neighbors*, giving richer context
  than top-k similarity alone. **Deduplication** uses connected components —
  if three manual chunks say nearly the same thing, only one representative
  per cluster survives.
- **`reasoning_engine.py`** — Tree of Thoughts extended into **Graph of
  Thoughts**: independent reasoning branches that reach the same conclusion
  get *merged* instead of staying separate, and a merged node's score is
  boosted — convergence is treated as evidence. Also: a **World Model** that
  simulates the real-world consequence of a recommendation and
  auto-revises it if the simulated outcome looks bad.

---

## 9. Orchestration — LangGraph state machine

`graph.py` replaces manual if/else control flow with a real state machine.

```python
graph = StateGraph(DiagnosticState)
graph.add_node("diagnose", diagnose_node)
graph.add_node("safety", safety_node)
graph.add_conditional_edges("safety", route_after_safety,
    {"human_review": "human_review", "cost": "cost"})
```

- **State** — a `TypedDict` passed between every node; nodes read it, do
  work, return an updated copy
- **Conditional edges** — after the safety node runs, a router function
  decides: CRITICAL → `human_review` node (interrupt + pause), anything
  else → continue to `cost`
- **Checkpointing** — persists graph state so a failed run resumes from the
  last completed node instead of starting over
- **Streaming** — `stream_graph()` yields state after each node completes,
  so a client sees the diagnostic result appear, then safety, then cost,
  instead of waiting for the whole run

`orchestrator.py` is the non-graph alternative path — a master agent that
runs the same four agents sequentially, resolves conflicts (safety always
wins on CRITICAL), and optionally runs a Graph-of-Thoughts pass plus a
World Model validation before returning the final report.

`agent_architectures.py` layers 8 *switchable* coordination patterns on top
of the same 4 agents — master agent, sequential pipeline, collaborative,
parallel swarm, hierarchical, blackboard-only, peer-to-peer, and
self-improving — selectable per request via `/diagnose/architecture/{kind}`.

---

## 10. Vector Store — Pinecone + LangChain

Base RAG: `ExtendedPineconeStore` (in `pinecone_manager.py`) wraps
`PineconeVectorStore` + `HuggingFaceEmbeddings` (all-MiniLM-L6-v2, 384
dimensions) over uploaded PDF/TXT manuals, chunked with
`RecursiveCharacterTextSplitter`.

Production features layered on top:
- **Namespaces** — isolate vectors per vehicle make (`toyota`, `honda`, ...)
- **Native hybrid search** — dense (semantic) + sparse (BM25 via
  `pinecone-text`) combined server-side
- **MMR search** — `max_marginal_relevance_search()` for diverse (not just
  similar) results
- **Score threshold filtering** — discard low-similarity chunks instead of
  always returning top-k
- **`get_stats()`** — surfaced in `/health` for index visibility
- **Backup/restore** via Pinecone collections

`ExtendedPineconeStore` implements `similarity_search(query, k)` directly,
so it can be passed as `vector_store` straight into agents and tools with
no adapter layer.

---

## 11. Evaluation — how good is any given agent run?

`evaluator.py` scores a completed run across 3 independent dimensions:

1. **Task success rate** — did the diagnosis mention the expected fault
   causes? (string-matched against a test's expected list)
2. **Trajectory evaluation** — did the agent use the right tools, avoid
   redundant calls, and stay within the iteration budget?
3. **LLM-as-judge** — a second LLM call scores correctness, completeness,
   safety, and clarity 0–10 against the transcript

All three combine into one `overall_score`, logged to MLflow per run.

---

## 12. Observability — MLflow + Prometheus + custom callbacks

- **MLflow** — every diagnostic run, every feedback event, every memory
  consolidation logs params/metrics via `mlflow.start_run()`
- **`callbacks.py`** — `TracerCallback` hooks LangChain's own callback
  system (`on_llm_start`, `on_tool_end`, `on_agent_action`, ...) directly
  into `tracer.py`, so you get step-level tracing without manually
  instrumenting every agent call. `AutoMLflowCallback` auto-logs LLM/tool
  call counts per chain run.
- **Prometheus** — request counters, latency histograms, safety-alert
  counters, graph-algorithm-call counters, scraped at `/metrics`
- **Grafana** — dashboards over the Prometheus data (via docker-compose)

---

## 13. FastAPI — tying it all together

`main.py` wires every layer above into a REST API. Startup initializes
Pinecone, embeddings, and the vector store once; every request reuses them.

Key endpoint groups:

| Group | Example | What it exercises |
|---|---|---|
| Core diagnosis | `POST /diagnose` | Full LangGraph run |
| Architecture testing | `POST /diagnose/architecture/{kind}`, `POST /diagnose/compare` | 8 coordination patterns |
| Graph tools | `GET /graph/fault/related/{code}`, `GET /graph/fault/cheapest-path/{start}/{end}`, `POST /graph/repair-order` | BFS/DFS/Dijkstra/topo-sort directly |
| Memory | `GET /memory/{session_id}` | Per-session memory stats |
| Feedback | `POST /feedback`, `GET /feedback/stats` | SQLite-backed rating loop |
| Admin | `GET /admin/db-stats` (API-key protected) | DB health |
| Ops | `GET /health`, `GET /metrics`, `GET /graph-structure` | Service + graph introspection |

Guardrails run before any agent call; a rejected request never reaches an
LLM.

---

## 14. Docker + Compose — from one container to six services

**`Dockerfile`** — multi-stage build, non-root `USER`, `HEALTHCHECK` so an
orchestrator knows when the container is actually ready, BuildKit cache
mounts to keep rebuilds fast.

**`docker-compose.yml`** — six services on two isolated networks
(`frontend_net`, `backend_net`):

| Service | Role |
|---|---|
| `nginx` | Reverse proxy, only public-facing entrypoint |
| `backend` | FastAPI app — no direct port exposure, only reachable via nginx |
| `mlflow` | Tracking server, SQLite backend store |
| `redis` | Cache + rate-limit storage |
| `prometheus` | Metrics scraping |
| `grafana` | Dashboards |

`depends_on` uses `condition: service_healthy` — the backend won't start
accepting traffic until MLflow and Redis pass their own healthchecks. Bind
mounts (`./data:/app/data`) persist the SQLite database and uploaded
manuals across container restarts.

---

## 15. Testing

| File | Covers |
|---|---|
| `test_db.py` | Every SQLite table — CRUD, write policies, consolidation |
| `test_agents.py` | Blackboard, tracer, agent protocol, memory manager — no LLM calls |
| `test_tools.py` | All tools, all guardrail rules, graph tool wrappers, Dijkstra tool |
| `test_graph.py` | BFS, DFS, Dijkstra, topological sort, chunk graph clustering |

No LLM calls in the infrastructure tests — they test pure logic (blackboard
redundancy rules, SQL upserts, graph traversal correctness), so they run in
milliseconds and need no API keys.

---

## Quick Start

```bash
git clone <repo>
cd autodiag-pro
cp backend/.env.example backend/.env   # add your HF + Pinecone keys

cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# in another terminal
mlflow ui --port 5000

# Swagger UI
open http://localhost:8000/docs
```

Or full stack:
```bash
docker compose up --build
```

---

## Tech Stack

Python · LangChain · LangGraph · FastAPI · Pinecone · HuggingFace ·
SQLite · MLflow · Prometheus · Grafana · Docker · Docker Compose ·
custom graph algorithms (BFS, DFS, Dijkstra, topological sort) ·
AWS EC2/ECR · GitHub Actions

---

## Folder Structure

```
autodiag-pro/
├── backend/
│   ├── main.py
│   ├── graph.py
│   ├── orchestrator.py
│   ├── agent_architectures.py
│   ├── base_agent.py
│   ├── blackboard.py
│   ├── agent_protocol.py
│   ├── tracer.py
│   ├── context_manager.py
│   ├── memory_manager.py
│   ├── reasoning_engine.py
│   ├── guardrails.py
│   ├── callbacks.py
│   ├── evaluator.py
│   ├── feedback.py
│   ├── db.py
│   ├── schemas.py
│   ├── tools.py
│   ├── fault_graph.py
│   ├── vehicle_hierarchy.py
│   ├── chunk_graph.py
│   ├── pinecone_manager.py
│   ├── run_mission.py
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── diagnostic_agent.py
│   │   ├── safety_agent.py
│   │   ├── cost_agent.py
│   │   └── reflection_agent.py
│   └── tests/
│       ├── test_agents.py
│       ├── test_tools.py
│       ├── test_graph.py
│       └── test_db.py
├── docker-compose.yml
├── monitoring/
│   └── prometheus.yml
├── nginx/
│   └── nginx.conf
└── README.md
```