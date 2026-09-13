"""
The Fact-Checker Agent: takes the Investigator's theory and actively
searches for evidence AGAINST it — contradictions, alternative suspects,
conflicting timelines, and weaknesses in the reasoning. This is the
adversarial counterpart to the Investigator's self-correction loop.

"""

import os
import re
import sys
import json

from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import FactCheckResult, Contradiction, EvidenceStatus
from retrieval.hybrid_search import get_retriever
from ingestion.preprocess import ingest_corpus

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"


FACT_CHECK_PROMPT = """You are a Fact-Checker Agent. Your job is to be skeptical and
adversarial toward the theory below — actively look for reasons it might be WRONG.

THEORY UNDER REVIEW:
{theory}

PRIMARY SUSPECT: {suspect_name}

KNOWN CANDIDATES: {candidates}

DOCUMENTS FLAGGED AS UNVERIFIED OR CONTESTED (any theory relying on these as
established fact is a weakness you must flag): {unverified_docs}

EVIDENCE GATHERED FOR ADVERSARIAL REVIEW:
{evidence_text}

Look specifically for:
- Contradictions: evidence that conflicts with the theory's claims
- Alternative explanations: could someone else have done it instead?
- Conflicting timelines: do the times/sequence of events actually line up?
- Weaknesses: circumstantial or weak links in the reasoning (e.g. old/indirect evidence
  treated as if it were strong proof)
- Whether the theory improperly treats any unverified/contested evidence as fact

CRITICAL: Only critique claims that actually appear in the THEORY UNDER REVIEW text above.
Do NOT invent or assume the theory relies on a document, person, or claim that is not
explicitly present in that theory text. If the theory does not mention an unverified
document, do not claim it does.

Return ONLY valid JSON in exactly this shape:
{{
  "contradictions": [{{"claim": "...", "conflicting_document_id": "...", "explanation": "..."}}],
  "alternative_candidates": ["name1", "name2"],
  "weaknesses": ["weakness 1", "weakness 2"],
  "overall_assessment": "REQUIRED, never leave empty - a short paragraph giving your honest adversarial verdict on how well the theory holds up overall"
}}

If you genuinely find no contradictions after looking, say so honestly in
overall_assessment rather than inventing weak ones — but always still list any
real weaknesses (even strong theories usually have at least one), and always fill
in overall_assessment even if it's brief.
"""


def _build_evidence_text(chunks) -> str:
    return "\n\n".join(f"[{c.document_id}]\n{c.text}" for c in chunks)


def _gather_adversarial_evidence(suspect_name: str, retriever, top_k_per_query: int = 4):
    queries = [
        f"alibi or evidence clearing {suspect_name}",
        f"contradictions or inconsistencies in {suspect_name} account",
        f"evidence pointing to a suspect other than {suspect_name}",
        "unverified anonymous or uncorroborated evidence in the case",
        "conflicting timeline or unexplained gap in events",
    ]

    seen_doc_ids = set()
    combined_chunks = []
    for q in queries:
        for chunk in retriever.hybrid_search(q, top_k=top_k_per_query):
            if chunk.document_id not in seen_doc_ids:
                seen_doc_ids.add(chunk.document_id)
                combined_chunks.append(chunk)

    return combined_chunks


def fact_check(theory: str, suspect_id: str = None) -> FactCheckResult:
    ingest_result = ingest_corpus(use_cache=True)
    retriever = get_retriever()

    candidate_names = [c.name for c in ingest_result.candidates]
    name_to_id = {c.name: c.id for c in ingest_result.candidates}
    id_to_name = {c.id: c.name for c in ingest_result.candidates}

    suspect_name = id_to_name.get(suspect_id, "unknown")

    unverified_docs = sorted(set(
        ev.document_id for ev in ingest_result.evidence_items
        if ev.status != EvidenceStatus.VERIFIED
    ))

    chunks = _gather_adversarial_evidence(suspect_name, retriever)

    prompt = FACT_CHECK_PROMPT.format(
        theory=theory,
        suspect_name=suspect_name,
        candidates=", ".join(candidate_names),
        unverified_docs=", ".join(unverified_docs) if unverified_docs else "none",
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
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        parsed = json.loads(match.group(0)) if match else {
            "contradictions": [],
            "alternative_candidates": [],
            "weaknesses": ["Fact-check parsing failed."],
            "overall_assessment": "Unable to complete fact-check due to a parsing error.",
        }

    contradictions = [
        Contradiction(
            claim=c.get("claim", ""),
            conflicting_document_id=c.get("conflicting_document_id", "unknown"),
            explanation=c.get("explanation", ""),
        )
        for c in parsed.get("contradictions", []) if isinstance(c, dict)
    ]

    alt_ids = [
        name_to_id[name] for name in parsed.get("alternative_candidates", [])
        if name in name_to_id
    ]

    junk_values = {"overall_assessment", ":", "contradictions", "alternative_candidates", "weaknesses"}
    raw_weaknesses = parsed.get("weaknesses", [])
    weaknesses = [
        w.strip() for w in raw_weaknesses
        if isinstance(w, str) and len(w.strip()) > 15 and w.strip().rstrip(":").strip() not in junk_values
    ]

    overall_assessment = parsed.get("overall_assessment", "").strip()
    if not overall_assessment:
        overall_assessment = (
            "The Fact-Checker did not produce a summary verdict; see contradictions "
            "and weaknesses above for its findings."
        )

    return FactCheckResult(
        contradictions=contradictions,
        alternative_candidate_ids=alt_ids,
        weaknesses=weaknesses,
        overall_assessment=overall_assessment,
    )


if __name__ == "__main__":
    # Reuses the theory we already confirmed works, instead of re-running
    # the Investigator (which costs extra LLM calls) every time we test
    # just the Fact-Checker.
    sample_theory = (
        "Rohan Desai was the person who physically disconnected CCTV camera 3 "
        "behind the bar counter during the 11:12-11:18 PM window, thereby "
        "creating a blind spot that allowed him to slip a lethal dose of "
        "industrial-grade cyanide into the victim's whiskey glass. He had a "
        "clear motive - he was aware of a confidential Q3 audit that could "
        "jeopardize his position, as indicated by his text to an unknown "
        "contact at 11:00 PM (doc_13). He had the means - cyanide is a "
        "compound used in industrial or laboratory settings, and the "
        "forensic report confirms the poison's purity is consistent with "
        "such sources (doc_03). He had the opportunity - badge scans show "
        "he was present in the bar area at 11:10 PM, the exact time the "
        "camera lost signal, and the IT incident report attributes the "
        "disconnection to a physical cut at the junction box behind the "
        "bar, an area only accessible by someone who was physically "
        "present there (doc_12, doc_16)."
    )
    sample_suspect_id = "cand_04"

    print("Running Fact-Checker against a known theory...\n")
    fc_result = fact_check(sample_theory, sample_suspect_id)

    print(f"OVERALL ASSESSMENT:\n{fc_result.overall_assessment}\n")

    print(f"CONTRADICTIONS ({len(fc_result.contradictions)}):")
    for c in fc_result.contradictions:
        print(f"  [{c.conflicting_document_id}] {c.claim} -> {c.explanation}")

    print(f"\nALTERNATIVE CANDIDATES: {fc_result.alternative_candidate_ids}")

    print(f"\nWEAKNESSES:")
    for w in fc_result.weaknesses:
        print(f"  - {w}")