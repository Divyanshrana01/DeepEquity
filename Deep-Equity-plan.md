# Equity Research Desk

A multi-agent equity research system where AI agents argue both sides of a stock thesis before producing a cited, balanced research note.

---

## 1. Project Description

### The idea in one line
Give the system a ticker. Two agents research it with opposite mandates, a third agent judges the debate, and you get a research note where every claim traces back to a real filing.

### Why this project exists
Most portfolio projects are single-shot RAG chatbots: ask a question, retrieve chunks, generate an answer. They demonstrate retrieval, nothing more. This project demonstrates orchestration, structured disagreement, evaluation, memory, and production hygiene, which is the actual gap UK employers cite when rejecting junior candidates.

The bull/bear structure is not a gimmick. It solves a real failure mode: a single LLM asked "is this stock a good buy" produces confidently one-sided output shaped by whatever it retrieved first. Forcing two adversarial passes over the same evidence base, then reconciling them, is a genuine reasoning architecture, and it is defensible under interview questioning.

### Market alignment
Built by reverse-engineering recurring requirements from real UK AI Engineer job descriptions and 2026 UK vacancy data:

| Requirement seen in UK JDs | How this project covers it |
|---|---|
| Multi-agent orchestration, supervisor pattern | LangGraph supervisor routing Bull, Bear, Synthesis |
| RAG with hybrid search and reranking | BM25 + dense, RRF fusion, cross-encoder rerank |
| MCP / tool-calling | Custom MCP server exposing filings and market data tools |
| Vector database | pgvector (one of the two most-cited vector DBs in UK listings) |
| LLMOps, evaluation, observability | RAGAS + golden dataset + LangSmith or Langfuse tracing |
| Production API | FastAPI, async, JWT auth, rate limiting |
| Caching and cost control | Redis semantic cache, model routing, cost per request logged |
| Responsible AI | Prompt injection defense, PII masking, guardrails, disclaimers |
| Containerisation and CI/CD | Docker Compose, GitHub Actions |

### Explicitly out of scope
- No brokerage or trading integration (regulatory exposure, zero interview value, large scope cost)
- No live streaming market data (own engineering problem, adds nothing to the demonstration)
- No fine-tuning (RAG and agents are what entry-level UK roles actually ask for)
- No Kubernetes (Docker Compose is sufficient at this scale)

These are deliberate cuts, and each one is a good interview answer when asked "what would you add next and why haven't you."

---

## 2. Architecture

### Agents

**Supervisor**
Owns the graph. Decides which agent runs next, whether the debate needs another round, and when to terminate. Enforces a max-rounds ceiling so the loop cannot run away.

**Research Planner Agent**
Runs before Bull and Bear touch any evidence. Decides which filings to pull, how many quarters back to look, whether earnings calls or news matter more for this ticker, and what retrieval strategy to use. This mirrors how a real analyst scopes a research task before diving in, and it is what separates a scripted pipeline from a system that reasons about its own research process. Promoted into the core build rather than left as a future enhancement, because it is the single highest-leverage addition to the agent story.

**Bull Agent**
Mandate: build the strongest evidence-backed case for the stock. Calls MCP tools to pull filings, transcripts, and price history, following the Planner's scope. Retrieves supporting passages. Outputs a structured thesis with claims, each attached to a source reference.

**Bear Agent**
Mandate: build the strongest evidence-backed case against. Same tools, same corpus, opposite objective. Outputs a structured counter-thesis in the same schema.

**Synthesis Agent**
Reads both theses. Identifies where they agree, where they genuinely conflict, and where one side has weaker evidence. Produces the final note with citations and a multi-dimensional confidence breakdown rather than one number:

| Dimension | What it captures |
|---|---|
| Evidence strength | Amount and quality of supporting evidence per claim |
| Reasoning consistency | Internal consistency of the synthesis logic |
| Source diversity | Spread across filings, transcripts, and news, not one document doing all the work |
| Data freshness | How recent the underlying evidence is |
| Overall confidence | Weighted combination of the above |

A single confidence score hides exactly the information a real analyst would want to see before trusting a report, this breakdown is cheap to produce since it is a formatting decision on output the Synthesis agent already has, not new infrastructure.

### State
Typed state object flowing through the graph. Holds the ticker, retrieved evidence, both theses, round count, and cost accumulated so far. Checkpointed to Redis so a run can pause and resume.

### Memory
- **Short-term**: LangGraph checkpointing backed by Redis, per-run conversation state
- **Long-term**: past research notes embedded into pgvector, so a repeat query on a ticker recalls prior findings and builds on them rather than starting cold

This distinction is worth stating explicitly in the README. It is a common interview probe.

### Retrieval
- Corpus: SEC filings, earnings call transcripts, recent news
- Chunking: parent-child (index small chunks for precision, pass parent chunks to the LLM for context). Filings are legal-style text, exactly the case where parent-child wins.
- Search: BM25 + dense vector, fused with Reciprocal Rank Fusion
- Rerank: cross-encoder over the fused candidate set
- Storage: pgvector

### Prompt strategy
Each agent has a different job, so each needs a different prompting approach. These are starting hypotheses to be tested, not settled decisions.

| Agent | Starting approach | Why |
|---|---|---|
| Bull / Bear | ReAct plus few-shot | They interleave reasoning with tool calls; few-shot examples anchor the output schema and the required evidence-per-claim discipline |
| Supervisor | Zero-shot with a tight decision rubric | Routing is a narrow decision; a long prompt adds cost and drift for no gain |
| Synthesis | Chain of Thought plus self-consistency | Reconciling two adversarial theses is the reasoning-heavy step, and it is where an unforced error is most expensive |

Every agent returns a structured schema, not prose. Free-text output is unparseable, unevaluable, and untestable.

**Prompts are versioned artefacts, not strings in the code.** Store them separately, version them, and never change one without re-running the eval suite. A prompt change is a deployment.

### MCP server
Custom MCP server exposing tools:
- `fetch_filing(ticker, form_type)`
- `fetch_transcript(ticker, quarter)`
- `fetch_price_history(ticker, period)`
- `fetch_news(ticker, days)`

Agents call these through MCP rather than hardcoded functions. Worth doing properly: MCP appeared in zero UK vacancies in 2024 and 82 by 2026, and very few candidates can say they have built a server rather than just consumed one.

---

## 3. Workflow

### Ingestion (async, fault-tolerant)
```
Client requests ticker
  -> Idempotency check (already indexed this filing?)
  -> If new: fetch via MCP tool
  -> Store raw document
  -> Publish ingestion event
  -> Return 202 Accepted immediately
  -> Worker consumes event
  -> Second idempotency check (duplicate message guard)
  -> Parse -> chunk (parent-child) -> embed -> store in pgvector
  -> Mark complete
  -> On transient failure: retry with exponential backoff
  -> On permanent failure: dead letter queue
```

### Research run
```
POST /research {ticker}
  -> JWT auth
  -> Rate limit check (Redis sliding window)
  -> Semantic cache check (near-duplicate query answered recently?)
       -> HIT: return cached note
       -> MISS: continue
  -> Supervisor initialises state
  -> Bull Agent: MCP tool calls -> hybrid retrieval -> thesis
  -> Bear Agent: MCP tool calls -> hybrid retrieval -> counter-thesis
  -> Supervisor: enough evidence? -> another round OR proceed
  -> Synthesis Agent: reconcile -> note + confidence + citations
  -> Long-term memory write (embed note into pgvector)
  -> Cache write
  -> Cost + trace logged
  -> Return note
```

### Guardrails at every boundary
- External text (filings, news) treated as untrusted, prompt injection scanning before it reaches an agent
- PII masking on any user-supplied document
- Circuit breaker around external data sources
- Token budget ceiling per run, hard stop if exceeded
- "Not financial advice" disclaimer on every output

---

## 4. Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph | 76 UK vacancies, up from 0 in 2024; supervisor pattern is directly asked about in interviews |
| Tool protocol | MCP (custom server) | 82 UK vacancies, up from 0 in 2024; rare skill, strong differentiator |
| Vector DB | pgvector | Top-2 UK vector DB; sensible default when already on Postgres. Qdrant was evaluated (native hybrid retrieval, named vectors) and rejected: hybrid is already handled at the application layer via RRF, and pgvector keeps stronger UK market alignment plus portfolio diversity from KubePilot, which already uses Qdrant |
| Retrieval | BM25 + dense, RRF, cross-encoder rerank | Hybrid search and reranking are standard interview probes |
| API | FastAPI (async) | 185 UK vacancies, the default for production AI APIs |
| Database | PostgreSQL | Primary store plus pgvector extension |
| Cache / memory / limits | Redis | Semantic cache, sliding-window rate limiter, checkpoint store |
| Auth | JWT | "How do you secure an enterprise GenAI app" is a live interview question |
| Evaluation | RAGAS + golden dataset | Evaluation is the new system design round |
| Prompt management | Versioned prompt files + LangSmith or Langfuse prompt tracking | Prompt engineering is named explicitly in UK JDs; versioning turns it from guesswork into a measurable experiment |
| Observability | LangSmith or Langfuse | LangSmith 29 UK vacancies; Langfuse if self-hosting matters |
| Experiment tracking | MLflow | 67 UK vacancies |
| Container | Docker + Compose | Baseline expectation |
| CI/CD | GitHub Actions | Baseline expectation |
| Cloud | Deferred | Deploy somewhere with a public URL before applying; the platform matters less than the fact that it is live |

---

## 5. Build Phases

### Phase 1: Skeleton and infrastructure
- Docker Compose: api, worker, postgres+pgvector, redis, mcp-server
- FastAPI app with health and readiness endpoints on every container
- JWT auth on the API
- Redis sliding-window rate limiter, per API key, proper 429 with Retry-After
- Structured logging with request tracing IDs
- GitHub Actions: lint, test, build

Exit criteria: `docker compose up` gives a running, authenticated, rate-limited API that does nothing useful yet. This is the boring phase that most portfolios skip and most interviews probe.

### Phase 2: Ingestion and retrieval
- MCP server with the four tools
- Ingestion pipeline: idempotency, event publish, worker consume, retry, DLQ
- Parent-child chunking over filings
- pgvector storage, BM25 index
- Hybrid search with RRF, cross-encoder reranker
- **Measure the baseline here.** Retrieval precision and recall on a small hand-built query set, before any tuning.

Exit criteria: you can ask a retrieval question about a filing and get correct passages back, with a recorded baseline number.

### Phase 3: Agents
- LangGraph state schema
- Research Planner Agent: decides scope (which filings, how many quarters, transcripts vs news weighting) before evidence collection starts
- Bull and Bear agents with structured output schemas, operating within the Planner's scope
- Supervisor with routing logic and a max-rounds ceiling
- Synthesis agent producing the note with the multi-dimensional confidence breakdown and citations
- Redis checkpointing so runs resume

Exit criteria: `POST /research {ticker}` returns a real, cited note, and the Planner's scoping decision is visible in the trace, not just the final output.

### Phase 4: Memory and cost
- Long-term memory: embed past notes into pgvector, recall on repeat ticker
- Semantic cache in Redis, measure hit rate
- Model routing: cheap model for extraction, stronger model for synthesis
- Cost tracking per request and per agent, logged and exposed

Exit criteria: second query on the same ticker is measurably faster and cheaper, with numbers to prove it.

### Phase 5: Guardrails, evaluation, and failure testing
- Prompt injection scanning on ingested external text
- PII masking
- Circuit breaker on external sources
- Token budget ceiling
- Golden dataset: hand-label 20-30 ticker theses with known correct claims
- RAGAS: faithfulness, context precision, answer relevancy
- Regression suite wired into CI
- **Failure simulation**: deliberately kill Redis, kill Postgres, time out the MCP server, feed it an empty retrieval, feed it a malformed filing, attempt a prompt injection. Confirm the system degrades gracefully (clear error, fallback, or retry) rather than crashing outright. Cheap to run since it mostly means breaking infrastructure you already built, and it is one of the strongest "what did you test for" interview answers available.

Exit criteria: published before/after metrics, baseline first then delta, never invented. Plus a short written note on what happens under each simulated failure.

### Phase 6: Prompt iteration

This phase comes *after* evaluation on purpose. Tuning prompts without a golden dataset is vibes-driven development: you change wording, the output "feels better," and you have learned nothing. With the eval harness in place, every prompt change becomes a measurable experiment.

**The loop**
```
Freeze the golden dataset
  -> Run the eval suite on prompt v1 -> record the baseline
  -> Change ONE thing in ONE prompt
  -> Re-run the eval suite -> record the delta
  -> Keep it or revert it, based on the number, not the vibe
  -> Log the run in MLflow: prompt version, scores, cost, latency
```

**What to actually try, per agent**

Bull and Bear:
- Zero-shot vs few-shot: do worked examples improve claim-to-source discipline, or just burn tokens
- Vary the number of few-shot examples (1, 3, 5), find where the gain flattens
- Explicit adversarial framing ("your job is to find the strongest case against, not a balanced view") vs neutral framing: does hard framing sharpen the debate or produce hallucinated pessimism
- Does forcing a citation on every claim in the prompt reduce unsupported claims, and by how much

Supervisor:
- Zero-shot vs a short rubric: does a rubric reduce unnecessary extra debate rounds
- Measure rounds-per-run, because every extra round is real money

Synthesis:
- Plain vs Chain of Thought: does the reasoning trace improve faithfulness enough to justify the tokens
- Self-consistency (sample n times, take the majority view) vs single pass: measure the faithfulness gain against the n-times cost multiplier
- Does asking for the confidence score *before* the note change the note, versus asking after

Cross-cutting:
- Prompt compression: strip the prompt down until scores drop, then step back one. Directly answers the "inference cost is exploding" interview question.
- Model routing: does the cheap model hold its score on extraction, and where exactly does it break

**Rules**
- One variable per run. Two changes and one score means you learned nothing.
- Same golden dataset every run, or the numbers are not comparable.
- Record cost and latency alongside quality. A prompt that wins on faithfulness but doubles the token spend is a tradeoff decision, not a win.
- Keep the losers in the log. "I tried self-consistency, it cost 3x for a 2% faithfulness gain, so I dropped it" is a stronger interview answer than only ever mentioning what worked.

Exit criteria: a prompt experiment log showing what you tried, what the numbers did, and what you kept. This log is a portfolio asset in its own right.

### Phase 7: Ship
- Deploy with a public URL
- README: architecture diagram, decision log (why LangGraph, why pgvector, why hybrid search), measured metrics, failure modes, cost/latency notes, "not financial advice" disclaimer
- Two-minute readability test: can a reviewer understand it without running it

Exit criteria: a live link you can put on a CV.

---

## 6. Success Metrics

Measure, never invent. Baseline first, then delta.

- **Retrieval**: precision and recall on the hand-built query set, before and after hybrid search plus reranking
- **Faithfulness**: RAGAS faithfulness and context precision on the golden dataset, before and after tuning
- **Citations**: proportion of claims in the final note that trace to a real source passage vs unsupported
- **Cache**: hit rate, and its measured effect on p50 latency and cost per request
- **Latency**: p50 and p95 under load
- **Cost**: average dollars per research run, before and after model routing
- **Prompt experiments**: faithfulness delta per prompt version, with the token cost of each change recorded next to it

Every one of these becomes a CV bullet only after it has been measured.

---

## 7. Interview Answers This Project Buys You

- "Explain multi-agent and supervisor architecture" -> your own graph
- "What is MCP and why does it matter" -> you built a server
- "How do you choose chunk size, when do you use parent-child" -> your filings pipeline
- "Hybrid search vs pure vector, when does keyword win" -> your RRF implementation
- "How do you evaluate an LLM app before shipping" -> golden dataset, RAGAS, regression suite in CI
- "Explain zero-shot, few-shot, CoT, self-consistency, ReAct" -> you ran all of them as controlled experiments and have the numbers
- "How do you know a prompt change actually improved anything" -> frozen golden set, one variable per run, logged delta, kept the losers
- "Your inference cost is exploding, give me five levers" -> caching, routing, budgets, all measured
- "How do you secure an enterprise GenAI app" -> JWT, rate limiting, injection scanning, PII masking
- "Design a fault-tolerant ingestion pipeline" -> idempotency, events, retry, DLQ
- "How do you stop an agent looping forever" -> max rounds, token ceiling, circuit breaker
- "Walk me through what happens when the user hits enter" -> the workflow section above
- "What's next for this project, what would you add" -> the Future Enhancements appendix below, and you can explain the reasoning behind each cut

---

## 8. Future Enhancements (post-MVP)

Kept separate deliberately so the MVP stays achievable. Build the core exactly as planned above, ship it, then pick these up one at a time.

**Human-in-the-Loop (HITL)**
Analyst approval gate before a report publishes: approve, reject, or edit, with feedback stored. Demonstrates enterprise AI workflow patterns and creates data that could later support DPO or fine-tuning.

**Online Evaluation Agent**
Distinct from the Phase 5 RAGAS suite, which runs offline against a frozen golden dataset during development. This agent runs online, checking each live output (citation validity, unsupported claims, hallucinations, missing evidence, confidence consistency) before it reaches the user. Keep the two clearly separated in the README so they don't read as the same thing twice.

**Portfolio Analysis Agent**
Extend from single-ticker to a full portfolio upload: diversification, sector exposure, risk concentration, a bull/bear summary across the whole portfolio, and suggested areas for further research.

**Temporal Reasoning**
Compare filings across reporting periods automatically: revenue growth, margin changes, guidance revisions, new risk factors, shifts in executive commentary quarter over quarter.

**Retrieval Benchmark Dashboard**
A dashboard comparing dense-only, BM25-only, hybrid, hybrid plus RRF, and hybrid plus RRF plus cross-encoder reranking, tracked on precision, recall, context precision, faithfulness, latency, and cost. Good if the goal shifts toward a retrieval-research narrative rather than just shipping the product.

**Failure Simulation Suite**
Already pulled into the core Phase 5 build above, since it's cheap and high-value. Not treated as a deferred item.

**Multi-Model Routing (deeper version)**
Phase 4 already includes basic cheap/strong model routing. A future version assigns a distinct model tier per agent role (small model for extraction and evaluation, medium for Bull/Bear/Supervisor, strong for Synthesis) rather than a single cheap/expensive split, for finer-grained cost control.

**User Feedback Loop**
Collect ticker, report, rating, reason, and timestamp from real users. Feeds back into prompt improvement, retrieval tuning, and future evaluation datasets.

**Fine-Tuning Roadmap**
The MVP intentionally avoids fine-tuning, since RAG and agents are what UK entry-level roles actually test for. Future path if pursued: DPO using analyst preference data from the feedback loop, LoRA adaptation for financial-domain language, company-specific summarisation style. Having this roadmap ready is itself a good answer to "where would fine-tuning fit here."

**Dropped, not deferred: standalone Agent Observability Dashboard**
Considered and rejected rather than deferred. LangSmith or Langfuse, already in the core stack, already surfaces tokens, latency, cost, tool calls, and retries per agent. A custom dashboard on top would duplicate that data without adding a new capability, unless the actual goal became a dashboard-building exercise in its own right, which is not the point of this project.

**Considered and rejected: Qdrant migration**
See the Tech Stack table above. pgvector kept for UK market alignment and portfolio diversity against KubePilot.
