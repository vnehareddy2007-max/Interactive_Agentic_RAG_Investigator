# Solve the Case: Interactive Agentic RAG Investigation

An interactive detective-style AI system where the user investigates a case
with the help of two AI agents — an **Investigator** that forms and
self-corrects theories, and a **Fact-Checker** that adversarially searches
for evidence against those theories.

**Case:** *The Skyline Retreat Poisoning* — a startup CEO is found dead at a
company retreat, poisoned. Four candidates, 19 interconnected documents
(witness statements, interrogation transcripts, forensic reports, financial
records, security logs), including one deliberately unverified/misleading
anonymous tip.

## Live Demo

- **Frontend (Streamlit):** https://interactiveagenticraginvestigator-lhcm4ntsazd4udbyedcfic.streamlit.app/
- **Backend API (FastAPI, Render):** https://rag-investigation-backend.onrender.com
- **API docs:** https://rag-investigation-backend.onrender.com/docs

> Note: the backend runs on a free-tier instance that spins down after ~15
> minutes of inactivity. The first request after idle time may take up to a
> minute to respond while it wakes up.

## Architecture

```
corpus/ (19 authored documents)
      │
      ▼
ingestion/preprocess.py  ──▶  LLM (Groq) extracts candidates, entities,
                               relationships, evidence per document
      │
      ▼
graph/evidence_graph.py  ──▶  networkx graph (candidates + entities as
                               nodes, relationships as edges)
      │
retrieval/hybrid_search.py ─▶ BM25 (keyword) + fastembed (semantic)
      │                        combined ranking
      ▼
agents/investigator.py   ──▶  forms theory → retrieves evidence → self-
                               evaluates → retries (max 2) → grounded
                               citations, flags unverified evidence
      │
agents/fact_checker.py   ──▶  adversarially searches for contradictions,
                               alternative suspects, weaknesses in the
                               Investigator's theory
      │
      ▼
main.py (FastAPI)        ──▶  wires all of the above into REST endpoints
      │
      ▼
frontend/app.py (Streamlit) ─▶ interrogation, evidence search, graph view,
                                agent results, verdict submission
```

## Tech Stack

- **Backend:** FastAPI, Pydantic
- **LLM:** Groq API (`openai/gpt-oss-20b`)
- **Retrieval:** `rank-bm25` (keyword) + `fastembed` (semantic, ONNX-based —
  chosen over `sentence-transformers`/`torch` specifically to fit within
  low-memory free-tier hosting)
- **Graph:** `networkx`
- **Frontend:** Streamlit
- **Deployment:** Render (backend, Docker), Streamlit Community Cloud (frontend)

## Running Locally

**1. Clone and set up environment**
```bash
git clone https://github.com/vnehareddy2007-max/Interactive_Agentic_RAG_Investigator.git
cd Interactive_Agentic_RAG_Investigator
python -m venv venv
venv\Scripts\Activate.ps1      # Windows PowerShell
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_key_here
```

**3. Run the backend**
```bash
uvicorn main:app --reload
```

**4. Run the frontend** (in a separate terminal)
```bash
streamlit run frontend/app.py
```

The frontend defaults to `http://127.0.0.1:8000` for local development.

## API Endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/ingest` | Processes the corpus into candidates/entities/relationships/evidence |
| GET | `/graph` | Returns the evidence graph |
| POST | `/search_evidence` | Hybrid keyword + semantic search |
| POST | `/investigate` | Runs the Investigator agent |
| POST | `/fact_check` | Runs the Fact-Checker agent against a theory |
| POST | `/interrogate` | Chat with a candidate, grounded in their documents |
| POST | `/submit_verdict` | Scores the user's final suspect choice |

## Dataset

The corpus (`corpus/`) is a synthetic, self-authored case rather than an
existing public dataset. This was a deliberate choice — it guarantees the
corpus has exactly the structure the task requires (interconnected
documents, multiple candidates with both supporting and contradicting
evidence, a clear ground truth, and at least one deliberately unverified/
misleading document) without licensing, scraping, or PII concerns. All
preprocessing code that extracts structure from this corpus is fully
general (LLM-based extraction, not hardcoded to this specific case) and
would work the same way against a different document corpus.

## Bonus Features Implemented

- **Visual evidence graph** — rendered inline in the Streamlit frontend
  using networkx + matplotlib (red nodes = candidates, blue = other entities)

## AI Tools Used

Claude (Anthropic) was used throughout development to design the system
architecture, generate and debug code across all modules, and troubleshoot
deployment issues (including diagnosing and fixing a memory-limit crash on
the free-tier host by switching the embedding backend from
`sentence-transformers`/`torch` to the lighter `fastembed` library).

