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


class PageRoute(BaseModel):
    page: int
    route: str
    method: str
    p_handwritten: Optional[float]
    ocr_confidence: Optional[float]
    notes: List[str]


class AuditResponse(BaseModel):
    source_file: str
    extraction_methods_used: List[str]
    page_routes: List[PageRoute] = []
    vendor_guess: Optional[str]
    date_guess: Optional[str]
    raw_extracted_text: str
    category_verdicts: List[CategoryVerdictResponse]


# ---------- /v2: probabilistic RAPTOR + multi-hop pipeline ----------

class V2PolicyRequest(BaseModel):
    doc_id: str = Field(..., description="Unique identifier for this policy document")
    text: str = Field(..., description="Full raw text of the policy document")


class V2AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the policy")
    doc_id: str = Field(..., description="Which ingested policy to ask")
    category: str = Field("any", description="Expense category, e.g. meal / lodging / transport (cache scope)")
    grade: str = Field("any", description="Employee grade, e.g. L3 (cache scope)")
    location: str = Field("any", description="Location, e.g. Mumbai (cache scope)")
    version: str = Field("v1", description="Policy version label (cache scope)")
    use_cache: bool = Field(True, description="Set false to force fresh retrieval")
    use_structured: bool = Field(True, description="Set false to skip the exact table-row lookup")
