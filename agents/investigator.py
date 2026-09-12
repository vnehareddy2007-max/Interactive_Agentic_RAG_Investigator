"""
The Investigator Agent: forms an initial theory, retrieves supporting
evidence via hybrid search, evaluates whether that evidence is sufficient,
and if not, reformulates its query and retries.
Returns a grounded InvestigationResult with citations.

"""

import os
import re
import sys
import json

from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import InvestigationResult, Citation, EvidenceStatus
from retrieval.hybrid_search import get_retriever
from ingestion.preprocess import ingest_corpus

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"
MAX_RETRIES = 2


THEORY_PROMPT = """You are a detective's Investigator Agent analyzing evidence in a case.

KNOWN CANDIDATES/SUSPECTS: {candidates}

DOCUMENTS FLAGGED AS UNVERIFIED OR CONTESTED (do not treat these as established
fact — only cite them as "unverified" and do not let them override corroborated
evidence): {unverified_docs}

CURRENT SEARCH QUERY: {query}

RETRIEVED EVIDENCE:
{evidence_text}

Based ONLY on the evidence above, form or refine your theory of the case.
Return ONLY valid JSON in exactly this shape:
{{
  "theory": "a clear paragraph explaining who did it, motive, means, and opportunity",
  "suspect_name": "the name of your primary suspect, must be one of the known candidates",
  "confidence": 0.0 to 1.0,
  "citations": [{{"document_id": "...", "claim": "...", "status": "verified|unverified|contested"}}],
  "needs_more_evidence": true or false,
  "next_query": "if needs_more_evidence is true, a reformulated search query to fill the gap; otherwise empty string"
}}

Rules:
- Ground every claim in a specific document_id from the retrieved evidence.
- If evidence is thin, contradictory, or you can't confidently name a suspect, set
  needs_more_evidence to true and confidence low, and suggest a next_query that would
  help (e.g. asking about a specific candidate's alibi, motive, or the camera outage).
- Never cite an unverified/contested document as if it were established fact.
"""


def _build_evidence_text(chunks) -> str:
    return "\n\n".join(f"[{c.document_id}]\n{c.text}" for c in chunks)


def _call_llm_for_theory(query: str, candidates: list, unverified_docs: list, chunks) -> dict:
    prompt = THEORY_PROMPT.format(
        candidates=", ".join(candidates),
        unverified_docs=", ".join(unverified_docs) if unverified_docs else "none",
        query=query,
        evidence_text=_build_evidence_text(chunks),
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_completion_tokens=4096,
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return {
            "theory": "Unable to form a theory due to a parsing error.",
            "suspect_name": "",
            "confidence": 0.0,
            "citations": [],
            "needs_more_evidence": True,
            "next_query": query,
        }


def investigate(initial_query: str = None, focus_candidate_id: str = None) -> InvestigationResult:
    ingest_result = ingest_corpus(use_cache=True)
    retriever = get_retriever()

    candidate_names = [c.name for c in ingest_result.candidates]
    name_to_id = {c.name: c.id for c in ingest_result.candidates}

    unverified_docs = [
        ev.document_id for ev in ingest_result.evidence_items
        if ev.status != EvidenceStatus.VERIFIED
    ]
    unverified_docs = sorted(set(unverified_docs))

    if focus_candidate_id:
        focus_name = next((c.name for c in ingest_result.candidates if c.id == focus_candidate_id), "")
        query = initial_query or f"evidence, motive, means, and opportunity for {focus_name}"
    else:
        query = initial_query or "who murdered Aditya Rao, what was the motive, means, and opportunity"

    retries_used = 0
    result_json = None

    while retries_used <= MAX_RETRIES:
        chunks = retriever.hybrid_search(query, top_k=6)
        result_json = _call_llm_for_theory(query, candidate_names, unverified_docs, chunks)

        if not result_json.get("needs_more_evidence") or retries_used == MAX_RETRIES:
            break

        next_query = result_json.get("next_query", "").strip()
        if not next_query:
            break

        query = next_query
        retries_used += 1

    citations = [
        Citation(
            document_id=c.get("document_id", "unknown"),
            claim=c.get("claim", ""),
            status=EvidenceStatus(c.get("status", "unverified")) if c.get("status") in
                   ("verified", "unverified", "contested") else EvidenceStatus.UNVERIFIED,
        )
        for c in result_json.get("citations", [])
    ]

    suspect_name = result_json.get("suspect_name", "")
    suspect_id = name_to_id.get(suspect_name)

    return InvestigationResult(
        theory=result_json.get("theory", ""),
        suspect_id=suspect_id,
        confidence=float(result_json.get("confidence", 0.0)),
        citations=citations,
        needs_more_evidence=bool(result_json.get("needs_more_evidence", False)),
        retries_used=retries_used,
    )


if __name__ == "__main__":
    result = investigate()

    print(f"THEORY:\n{result.theory}\n")
    print(f"SUSPECT ID: {result.suspect_id}")
    print(f"CONFIDENCE: {result.confidence}")
    print(f"RETRIES USED: {result.retries_used}")
    print(f"NEEDS MORE EVIDENCE: {result.needs_more_evidence}")
    print(f"\nCITATIONS ({len(result.citations)}):")
    for c in result.citations:
        print(f"  [{c.document_id}] ({c.status.value}) {c.claim}")