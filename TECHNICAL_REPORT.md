# Technical Report: Interactive Agentic RAG Investigation

## Overview

This project implements a detective-style investigation system where a user
works alongside two AI agents to solve a fictional poisoning case. The
system ingests a 19-document case file, builds an evidence graph, supports
hybrid retrieval over the corpus, and runs an Investigator agent (which
forms and self-corrects a theory) and a Fact-Checker agent (which
adversarially searches for evidence against that theory).

## Dataset Choice

Rather than sourcing an existing public dataset, I authored a synthetic
case (*The Skyline Retreat Poisoning*): 19 documents including witness
statements, interrogation transcripts, a forensic report, financial and
phone records, a security/CCTV log, and a deliberately unverified anonymous
tip. This was a deliberate tradeoff. Real-world corpora (news archives,
court records) often need substantial cleaning and don't guarantee the
specific structural properties the task requires — multiple candidates with
both supporting and conflicting evidence, a verifiable ground truth, and at
least one document that should be flagged as unreliable. Authoring the
corpus let me guarantee all of these properties and made it possible to
verify agent output against a known-correct answer during development. The
ingestion and extraction pipeline itself is fully general (LLM-based, not
hardcoded to this case) and would work against any similarly-structured
document set.

## Architecture

The pipeline is intentionally linear and modular: `ingestion` extracts
structured data from raw text via an LLM, `graph` turns that structure into
a queryable networkx graph, `retrieval` indexes the raw documents for
search, and the two `agents` build on top of retrieval. `main.py` exposes
all of this through FastAPI routes; `frontend/app.py` is a thin Streamlit
client that only talks to the API over HTTP, keeping the ML/agent logic
entirely server-side.

## Retrieval Design

Hybrid retrieval combines BM25 (`rank-bm25`) for exact-term matching with
embedding-based semantic search, min-max normalized and blended (default
50/50 weighting). Each of the 19 documents is treated as a single chunk
rather than being split further, since individual documents are short and
already topically coherent (a single witness statement or report) —
splitting them further would have fragmented context without a clear
retrieval benefit at this corpus size.

## Agent Design

**Investigator:** runs a bounded loop (max 2 retries) — retrieve evidence
for the current query, ask the LLM to form/refine a theory with citations
and a confidence score, and check whether the model itself reports
`needs_more_evidence`. If so, the model proposes its own reformulated query
for the next iteration, rather than the system reformulating on its behalf
— this keeps the "reasoning about what's missing" step inside the agent's
own judgment rather than a fixed heuristic. Testing confirmed this
genuinely varies behavior run to run (confirmed via `retries_used` in
output), rather than always converging identically.

**Fact-Checker:** deliberately does *not* reuse the Investigator's
retrieval. It issues five separate adversarial queries (alibi/clearing
evidence, contradictions, alternative suspects, unverified evidence,
timeline conflicts) to actively search for evidence the Investigator's own
retrieval might not have surfaced. In testing, this agent has genuinely
disagreed with and corrected the Investigator in cases where the
Investigator was overly cautious — evidence of real adversarial behavior
rather than rubber-stamping.

## Grounding and Hallucination Resistance

Every extracted evidence item is tagged `verified` / `unverified` /
`contested` at ingestion time (based on the source document's own framing —
e.g. an anonymous, uncorroborated tip is tagged `unverified`). Both agents
receive an explicit list of which document IDs are unverified and are
instructed never to treat those as established fact. During development, I
found and fixed a real failure mode where the Fact-Checker occasionally
fabricated a claim about what the Investigator's theory relied on (rather
than critiquing only what the theory text actually said) — fixed by adding
an explicit instruction constraining the model to the literal theory text,
plus a code-level filter that strips malformed/leaked JSON fragments from
its output.

## Deployment

The backend is deployed on Render (Docker) and the frontend on Streamlit
Community Cloud. A significant engineering constraint surfaced during
deployment: the original embedding backend (`sentence-transformers` +
`torch`) exceeded Render's free-tier 512MB memory limit and crashed the
container. This was resolved by switching to `fastembed`, an ONNX-based
embedding library with no PyTorch dependency, which produces comparable
retrieval quality (verified against the same test queries before and after
the switch) at a fraction of the memory footprint.

## Limitations and Possible Extensions

Given time constraints, retrieval treats whole documents as chunks rather
than supporting finer-grained passage retrieval, which would matter more at
larger corpus scale. The Fact-Checker's adversarial queries are fixed
templates rather than dynamically generated per-theory. Both would be
natural next steps, along with the bonus features not yet implemented
(reranking, temporal reasoning, persistent agent memory across sessions).
