-- Tracks every document we've ingested and where it is in the pipeline.
-- Runs on startup and is safe to re-run, everything is IF NOT EXISTS.

-- pgvector gives us the vector column type and the similarity operators. The postgres
-- image already ships the extension, this just switches it on for our database.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id              BIGSERIAL PRIMARY KEY,
    ticker          TEXT        NOT NULL,
    doc_type        TEXT        NOT NULL,
    -- source_ref is whatever uniquely names this doc at the source. For SEC filings
    -- that's the accession number, which is the thing that makes ingestion idempotent.
    source_ref      TEXT        NOT NULL,
    source_url      TEXT,
    -- Raw text lives right here. Filings come back capped at 20k chars from the MCP
    -- tool, so a text column is plenty, no need for object storage yet.
    raw_text        TEXT,
    -- pending -> processing -> complete, or -> failed once retries run out.
    status          TEXT        NOT NULL DEFAULT 'pending',
    attempts        INTEGER     NOT NULL DEFAULT 0,
    last_error      TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- This is the idempotency guarantee. Same filing twice = same key = insert is rejected,
-- so we can never end up with two rows for one document even if two requests race.
CREATE UNIQUE INDEX IF NOT EXISTS documents_source_key
    ON documents (ticker, doc_type, source_ref);

-- Worker queries by status when checking what's outstanding, so index it.
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);


-- Parent chunks: the bigger pieces of text we hand to the LLM once a search hits. These
-- aren't searched over directly, they're what gives a matched child enough surrounding
-- context to be useful.
CREATE TABLE IF NOT EXISTS parent_chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INTEGER     NOT NULL,
    text        TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

-- Child chunks: small pieces we embed and search. Small means a match points at the
-- exact sentence that matters rather than a whole page, which is the precision half of
-- parent-child retrieval.
CREATE TABLE IF NOT EXISTS child_chunks (
    id              BIGSERIAL PRIMARY KEY,
    document_id     BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    parent_chunk_id BIGINT      NOT NULL REFERENCES parent_chunks (id) ON DELETE CASCADE,
    chunk_index     INTEGER     NOT NULL,
    text            TEXT        NOT NULL,
    -- 384 dimensions, matching BAAI/bge-small-en-v1.5. Changing the model means
    -- changing this, and the column type can't be altered in place with data in it.
    embedding       vector(384),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS child_chunks_document_idx ON child_chunks (document_id);
CREATE INDEX IF NOT EXISTS parent_chunks_document_idx ON parent_chunks (document_id);

-- HNSW index for nearest-neighbour search. Without it postgres scans every row, which
-- is fine for a handful of chunks and hopeless once there are hundreds of thousands.
-- vector_cosine_ops because we normalise embeddings and compare by cosine distance.
CREATE INDEX IF NOT EXISTS child_chunks_embedding_idx
    ON child_chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text index over the child chunks, ready for the BM25 half of hybrid search on
-- Day 4. Cheap to create now while the table is being defined.
CREATE INDEX IF NOT EXISTS child_chunks_fts_idx
    ON child_chunks USING gin (to_tsvector('english', text));


-- Long-term memory: every finished note, embedded, so a later run on the same ticker
-- starts from what we already worked out instead of from nothing.
--
-- Separate from the chunks table on purpose. Chunks are things other people wrote and we
-- retrieved. These are things this system concluded. Mixing them would let the agents
-- cite their own past opinion as if it were a source document, which is exactly the kind
-- of quiet circularity that makes a research tool untrustworthy.
CREATE TABLE IF NOT EXISTS research_memory (
    id          BIGSERIAL PRIMARY KEY,
    ticker      TEXT        NOT NULL,
    run_id      TEXT        NOT NULL,
    -- what the planner decided to look at that time, so a later run can see whether it is
    -- covering the same ground or genuinely new ground
    focus       TEXT        NOT NULL,
    summary     TEXT        NOT NULL,
    key_risks   TEXT,
    -- the overall confidence of that note. a past conclusion carried low confidence is
    -- worth much less as a starting point, and the recall should be able to say so.
    confidence  REAL,
    embedding   vector(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One memory per run. Re-running the write for a run that already stored its note is a
-- no-op rather than a duplicate, which matters because a resumed run can reach synthesis
-- twice.
CREATE UNIQUE INDEX IF NOT EXISTS research_memory_run_idx ON research_memory (run_id);

-- Recall is always scoped to one ticker and usually wants the recent ones first.
CREATE INDEX IF NOT EXISTS research_memory_ticker_idx
    ON research_memory (ticker, created_at DESC);

CREATE INDEX IF NOT EXISTS research_memory_embedding_idx
    ON research_memory USING hnsw (embedding vector_cosine_ops);
