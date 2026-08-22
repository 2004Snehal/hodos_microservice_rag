from typing import List, Optional
from pydantic import BaseModel, Field


class PolicyUpdateRequest(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for this policy document, e.g. 'travel-policy-2026'")
    text: str = Field(..., description="Full raw text of the policy document")


class PolicyUpdateResponse(BaseModel):
    doc_id: str
    chunks_created: int
    chunk_size: int
    overlap: int
    store_stats: dict


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the ingested policy")
    top_k: Optional[int] = Field(4, description="Number of chunks to retrieve")


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[Citation]


class CategoryVerdictResponse(BaseModel):
    category: str
    claimed_total: float
    rule_type: str
    retrieved_limit: Optional[float]
    verdict: str
    citation_chunk_id: Optional[str]
    citation_text: Optional[str]
    citation_score: Optional[float]


class AuditResponse(BaseModel):
    source_file: str
    extraction_methods_used: List[str]
    vendor_guess: Optional[str]
    date_guess: Optional[str]
    raw_extracted_text: str
    category_verdicts: List[CategoryVerdictResponse]
