"""
FastAPI backend for the Interactive Agentic RAG Investigation app. Wires
together ingestion, the evidence graph, hybrid retrieval, the Investigator
agent, and the Fact-Checker agent into a set of API routes the frontend
calls.

"""

import os
import re
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from groq import Groq

from schemas import (
    IngestResponse, SearchRequest, SearchResponse,
    InvestigateRequest, InvestigationResult,
    FactCheckRequest, FactCheckResult,
    InterrogateRequest, InterrogateResponse,
    VerdictRequest, VerdictResult,
    EvidenceGraphResponse,
)
from ingestion.preprocess import ingest_corpus
from graph.evidence_graph import build_graph, graph_to_response
from retrieval.hybrid_search import get_retriever
from agents.investigator import investigate
from agents.fact_checker import fact_check

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


CORRECT_SUSPECT_NAME = "Rohan Desai"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[STARTUP] Loading ingestion, graph, and retriever...")
    state["ingest_result"] = ingest_corpus(use_cache=True)
    state["graph"] = build_graph(state["ingest_result"])
    state["retriever"] = get_retriever()
    print("[STARTUP] Ready.")
    yield


app = FastAPI(title="Interactive Agentic RAG Investigation", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """returns cachedcorpus ingestion"""
    result = ingest_corpus(use_cache=True)
    state["ingest_result"] = result
    state["graph"] = build_graph(result)
    return result


@app.get("/graph", response_model=EvidenceGraphResponse)
def get_graph():
    """Returns the evidence graph for frontend visualization."""
    return graph_to_response(state["graph"])


@app.post("/search_evidence", response_model=SearchResponse)
def search_evidence(request: SearchRequest):
    """Runs hybrid keyword+semantic search over the corpus."""
    retriever = state["retriever"]
    results = retriever.hybrid_search(request.query, top_k=request.top_k)
    return SearchResponse(query=request.query, results=results)


@app.post("/investigate", response_model=InvestigationResult)
def run_investigate(request: InvestigateRequest):
    """Runs the Investigator Agent's self-correcting loop."""
    return investigate(focus_candidate_id=request.focus_candidate_id)


@app.post("/fact_check", response_model=FactCheckResult)
def run_fact_check(request: FactCheckRequest):
    """Runs the Fact-Checker Agent against a given theory."""
    return fact_check(theory=request.theory, suspect_id=request.suspect_id)


INTERROGATE_PROMPT = """You are roleplaying as {candidate_name} ({candidate_role}) being
questioned by a detective in a criminal investigation. Stay strictly in character.

Only say things consistent with the documents below about you and this case. If asked
about something not covered by these documents, respond naturally as this person would
(deflect, say you don't know, or give a vague plausible answer) - do NOT invent specific
facts not present in the documents. Do not break character or mention you are an AI.

DOCUMENTS ABOUT YOU AND THIS CASE:
{evidence_text}

DETECTIVE'S QUESTION: {question}

Return ONLY valid JSON in exactly this shape:
{{
  "answer": "your in-character response to the question",
  "grounded_document_ids": ["doc_id1", "doc_id2"]
}}
"""


@app.post("/interrogate", response_model=InterrogateResponse)
def interrogate(request: InterrogateRequest):
    """Lets the user chat with a candidate, grounded in that candidate's
    own statements/documents from the corpus."""
    ingest_result = state["ingest_result"]
    candidate = next((c for c in ingest_result.candidates if c.id == request.candidate_id), None)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    retriever = state["retriever"]
    query = f"{candidate.name} {request.question}"
    chunks = retriever.hybrid_search(query, top_k=5)
    evidence_text = "\n\n".join(f"[{c.document_id}]\n{c.text}" for c in chunks)

    prompt = INTERROGATE_PROMPT.format(
        candidate_name=candidate.name,
        candidate_role=candidate.role or "person of interest",
        evidence_text=evidence_text,
        question=request.question,
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_completion_tokens=1024,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {
            "answer": "I... I'm not sure how to answer that.",
            "grounded_document_ids": [],
        }

    return InterrogateResponse(
        candidate_id=candidate.id,
        answer=parsed.get("answer", ""),
        grounded_document_ids=parsed.get("grounded_document_ids", []),
    )


@app.post("/submit_verdict", response_model=VerdictResult)
def submit_verdict(request: VerdictRequest):
    """Evaluates the user's final suspect choice against the case's answer."""
    ingest_result = state["ingest_result"]
    id_to_name = {c.id: c.name for c in ingest_result.candidates}

    chosen_name = id_to_name.get(request.suspect_id, "unknown")
    correct = chosen_name == CORRECT_SUSPECT_NAME

    actual_suspect_id = next(
        (c.id for c in ingest_result.candidates if c.name == CORRECT_SUSPECT_NAME),
        None,
    )

    if correct:
        feedback = (
            f"Correct! {CORRECT_SUSPECT_NAME} was responsible. Your reasoning and "
            f"supporting evidence were reviewed against the case file."
        )
    else:
        feedback = (
            f"Not quite - the evidence points most strongly to {CORRECT_SUSPECT_NAME}, "
            f"not {chosen_name}. Review the badge log, financial records, and IT "
            f"incident report for the strongest leads."
        )

    return VerdictResult(
        correct=correct,
        actual_suspect_id=actual_suspect_id or "",
        feedback=feedback,
        score=1.0 if correct else 0.0,
    )


@app.get("/")
def root():
    return {"status": "ok", "message": "Interactive Agentic RAG Investigation API"}