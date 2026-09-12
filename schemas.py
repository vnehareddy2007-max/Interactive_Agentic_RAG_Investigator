from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum



class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    CONTESTED = "contested"


class Candidate(BaseModel):
    id: str
    name: str
    role: Optional[str] = None
    summary: Optional[str] = None


class Entity(BaseModel):
    id: str
    name: str
    type: str  
    description: Optional[str] = None


class Relationship(BaseModel):
    source_id: str
    target_id: str
    relation: str  
    document_id: Optional[str] = None  


class EvidenceItem(BaseModel):
    id: str
    document_id: str
    description: str
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    linked_candidate_ids: List[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    document_id: str
    text: str
    keyword_score: Optional[float] = None
    semantic_score: Optional[float] = None
    combined_score: Optional[float] = None


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


class SearchResponse(BaseModel):
    query: str
    results: List[RetrievedChunk]



class Citation(BaseModel):
    document_id: str
    claim: str
    status: EvidenceStatus = EvidenceStatus.VERIFIED



class InvestigateRequest(BaseModel):
    case_id: str = "default"
    focus_candidate_id: Optional[str] = None 


class InvestigationResult(BaseModel):
    theory: str
    suspect_id: Optional[str] = None
    confidence: float
    citations: List[Citation]
    needs_more_evidence: bool
    retries_used: int = 0


class FactCheckRequest(BaseModel):
    theory: str
    suspect_id: Optional[str] = None


class Contradiction(BaseModel):
    claim: str
    conflicting_document_id: str
    explanation: str


class FactCheckResult(BaseModel):
    contradictions: List[Contradiction]
    alternative_candidate_ids: List[str] = Field(default_factory=list)
    weaknesses: List[str]
    overall_assessment: str



class InterrogateRequest(BaseModel):
    candidate_id: str
    question: str


class InterrogateResponse(BaseModel):
    candidate_id: str
    answer: str
    grounded_document_ids: List[str] = Field(default_factory=list)



class VerdictRequest(BaseModel):
    suspect_id: str
    reasoning: str
    supporting_document_ids: List[str] = Field(default_factory=list)


class VerdictResult(BaseModel):
    correct: bool
    actual_suspect_id: str
    feedback: str
    score: Optional[float] = None



class GraphNode(BaseModel):
    id: str
    label: str
    type: str 

class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str


class EvidenceGraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]



class IngestResponse(BaseModel):
    num_documents: int
    candidates: List[Candidate]
    entities: List[Entity]
    relationships: List[Relationship]
    evidence_items: List[EvidenceItem]