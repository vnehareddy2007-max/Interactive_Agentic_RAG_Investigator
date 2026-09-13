"""
Loads the raw corpus documents, sends each one to the LLM (Groq) to extract
candidates, entities, relationships, and evidence items, then merges
everything into a single structured result matching schemas.IngestResponse.

"""

import os
import re
import json
import glob
from typing import List, Dict, Any

from dotenv import load_dotenv
from groq import Groq

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from schemas import (
    Candidate, Entity, Relationship, EvidenceItem,
    EvidenceStatus, IngestResponse
)

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "openai/gpt-oss-20b"

CORPUS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "corpus")


def load_documents(corpus_dir: str = CORPUS_DIR) -> List[Dict[str, str]]:
    """Reads every .txt file in the corpus folder and parses out its header
    fields (DOCUMENT ID, TYPE, SOURCE) plus the body text."""
    documents = []
    filepaths = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))

    for filepath in filepaths:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()

        doc_id_match = re.search(r"DOCUMENT ID:\s*(\S+)", raw)
        type_match = re.search(r"TYPE:\s*(.+)", raw)
        source_match = re.search(r"SOURCE:\s*(.+)", raw)

        doc_id = doc_id_match.group(1).strip() if doc_id_match else os.path.basename(filepath)
        doc_type = type_match.group(1).strip() if type_match else "Unknown"
        source = source_match.group(1).strip() if source_match else "Unknown"

        documents.append({
            "document_id": doc_id,
            "type": doc_type,
            "source": source,
            "text": raw,
            "filename": os.path.basename(filepath),
        })

    return documents

EXTRACTION_PROMPT = """You are analyzing a document from a criminal investigation case file.
Extract structured information from the document below.

Return ONLY valid JSON (no markdown, no explanation) in exactly this shape:
{{
  "candidates": [{{"name": "...", "role": "..."}}],
  "entities": [{{"name": "...", "type": "person|place|object|event|organization", "description": "..."}}],
  "relationships": [{{"source": "...", "target": "...", "relation": "..."}}],
  "evidence_items": [{{"description": "...", "status": "verified|unverified|contested", "linked_candidates": ["..."]}}]
}}

Rules:
- "candidates" are ONLY these 4 people if mentioned: Priya Nair, Vikram Shah, Meera Iyer,
  Rohan Desai. Do NOT include Aditya Rao (the victim) as a candidate under any name or
  spelling, even if the document calls him a "suspect" figuratively.
- Mark evidence as "unverified" if the document itself says it is anonymous, uncorroborated,
  or unconfirmed. Mark as "contested" if two documents would disagree. Otherwise "verified".
- Keep entity/relationship names short and consistent (e.g. always "Rohan Desai", not
  variations like "Rohan" or "Mr. Desai").
- If a field has nothing to extract, return an empty list for it.

DOCUMENT ({doc_id}, type: {doc_type}):
{text}
"""


def extract_from_document(document: Dict[str, str]) -> Dict[str, Any]:
    """Calls the LLM on a single document and parses its JSON response."""
    prompt = EXTRACTION_PROMPT.format(
        doc_id=document["document_id"],
        doc_type=document["type"],
        text=document["text"],
    )

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
        reasoning_effort="low",
        max_completion_tokens=4096,
    )

    raw_output = response.choices[0].message.content.strip()

    raw_output = re.sub(r"^```(json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(raw_output)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None
        else:
            parsed = None

        if parsed is None:
            print(f"[WARN] Failed to parse LLM output for {document['document_id']}, skipping.")
            fail_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "failed_extractions.log")
            with open(fail_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== {document['document_id']} =====\n{raw_output}\n")
            parsed = {"candidates": [], "entities": [], "relationships": [], "evidence_items": []}

    parsed["_document_id"] = document["document_id"]
    return parsed


def merge_extractions(per_doc_results: List[Dict[str, Any]]) -> IngestResponse:
    candidates_by_name: Dict[str, Candidate] = {}
    entities_by_name: Dict[str, Entity] = {}
    relationships: List[Relationship] = []
    evidence_items: List[EvidenceItem] = []

    candidate_counter = 1
    entity_counter = 1
    evidence_counter = 1

    for result in per_doc_results:
        doc_id = result.get("_document_id", "unknown")

        for c in result.get("candidates", []):
            if not isinstance(c, dict):
                continue
            name = c.get("name", "").strip()
            if not name or "aditya" in name.lower():
                continue
            if name not in candidates_by_name:
                candidates_by_name[name] = Candidate(
                    id=f"cand_{candidate_counter:02d}",
                    name=name,
                    role=c.get("role"),
                )
                candidate_counter += 1

        for e in result.get("entities", []):
            if not isinstance(e, dict):
                continue
            name = e.get("name", "").strip()
            if not name:
                continue
            if name not in entities_by_name:
                entities_by_name[name] = Entity(
                    id=f"ent_{entity_counter:02d}",
                    name=name,
                    type=e.get("type", "unknown"),
                    description=e.get("description"),
                )
                entity_counter += 1

        for r in result.get("relationships", []):
            if not isinstance(r, dict):
                continue
            src, tgt, rel = r.get("source"), r.get("target"), r.get("relation")
            if not (src and tgt and rel):
                continue
            src_id = candidates_by_name.get(src, entities_by_name.get(src))
            tgt_id = candidates_by_name.get(tgt, entities_by_name.get(tgt))
            relationships.append(Relationship(
                source_id=src_id.id if src_id else src,
                target_id=tgt_id.id if tgt_id else tgt,
                relation=rel,
                document_id=doc_id,
            ))

        for ev in result.get("evidence_items", []):
            if not isinstance(ev, dict):
                continue
            desc = ev.get("description", "").strip()
            if not desc:
                continue
            status_raw = ev.get("status", "unverified").lower()
            status = status_raw if status_raw in ("verified", "unverified", "contested") else "unverified"

            linked_ids = []
            for name in ev.get("linked_candidates", []):
                cand = candidates_by_name.get(name)
                if cand:
                    linked_ids.append(cand.id)

            evidence_items.append(EvidenceItem(
                id=f"ev_{evidence_counter:03d}",
                document_id=doc_id,
                description=desc,
                status=EvidenceStatus(status),
                linked_candidate_ids=linked_ids,
            ))
            evidence_counter += 1

    KNOWN_ROLES = {
        "Priya Nair": "Co-Founder & CTO",
        "Vikram Shah": "Lead Investor, Board Member",
        "Meera Iyer": "Head of HR",
        "Rohan Desai": "Junior Financial Analyst",
    }

    for candidate in candidates_by_name.values():
        if not candidate.role or candidate.role.strip() in ("", "?"):
            candidate.role = KNOWN_ROLES.get(candidate.name, candidate.role)

    return IngestResponse(
        num_documents=len(per_doc_results),
        candidates=list(candidates_by_name.values()),
        entities=list(entities_by_name.values()),
        relationships=relationships,
        evidence_items=evidence_items,
    )


CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ingest_cache.json")


def ingest_corpus(corpus_dir: str = CORPUS_DIR, use_cache: bool = True) -> IngestResponse:
    if use_cache and os.path.exists(CACHE_PATH):
        print("[INFO] Loading cached ingestion result...")
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return IngestResponse(**json.load(f))

    documents = load_documents(corpus_dir)
    print(f"[INFO] Loaded {len(documents)} documents. Extracting with LLM...")

    per_doc_results = []
    for doc in documents:
        print(f"  -> extracting {doc['document_id']}...")
        per_doc_results.append(extract_from_document(doc))

    result = merge_extractions(per_doc_results)

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(result.model_dump(), f, indent=2)
    print(f"[INFO] Cached result to {CACHE_PATH}")

    return result


if __name__ == "__main__":
    result = ingest_corpus(use_cache=False)
    print(f"\nExtracted {len(result.candidates)} candidates:")
    for c in result.candidates:
        print(f"  - {c.id}: {c.name} ({c.role})")
    print(f"\nExtracted {len(result.entities)} entities")
    print(f"Extracted {len(result.relationships)} relationships")
    print(f"Extracted {len(result.evidence_items)} evidence items")