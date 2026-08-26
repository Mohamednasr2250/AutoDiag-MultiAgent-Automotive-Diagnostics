from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from enum import Enum

class SeverityLevel(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"; CRITICAL = "CRITICAL"

class RiskLevel(str, Enum):
    LOW = "LOW"; MEDIUM = "MEDIUM"; HIGH = "HIGH"

class AgentStatus(str, Enum):
    PENDING = "PENDING"; RUNNING = "RUNNING"; COMPLETED = "COMPLETED"; FAILED = "FAILED"; RETRYING = "RETRYING"

class TaskType(str, Enum):
    DIAGNOSE = "DIAGNOSE"; SAFETY = "SAFETY"; COST = "COST"; REFLECT = "REFLECT"

class SearchType(str, Enum):
    SIMILARITY = "similarity"; MMR = "mmr"; SCORE_THRESHOLD = "similarity_score_threshold"

class DiagnosticRequest(BaseModel):
    symptoms: str = Field(..., min_length=5)
    fault_codes: Optional[str] = Field(default="")
    vehicle_info: Optional[str] = Field(default="")
    session_id: Optional[str] = Field(default=None)
    use_tot: Optional[bool] = Field(default=False)

    @validator("symptoms")
    def symptoms_not_empty(cls, v):
        if not v.strip(): raise ValueError("symptoms cannot be empty")
        return v.strip()

    @validator("fault_codes")
    def validate_fault_codes(cls, v):
        if not v: return v
        codes = v.upper().replace(",", " ").split()
        for code in codes:
            code = code.strip()
            if code and not (len(code) == 5 and code[0] in ("P","C","B","U") and code[1:].isdigit()):
                raise ValueError(f"Invalid fault code format: {code}")
        return v.strip().upper()

class FeedbackRequest(BaseModel):
    query: str; answer: str; rating: bool
    comment: Optional[str] = ""; session_id: Optional[str] = ""

class EvaluationRequest(BaseModel):
    session_id: str; expected_issues: List[str] = []
    expected_tools: List[str] = []; ground_truth: Optional[str] = ""

class BatchDiagnosticRequest(BaseModel):
    requests: List[DiagnosticRequest]

class ResumeRequest(BaseModel):
    session_id: str; human_decision: Optional[str] = "approved"
    corrections: Optional[Dict] = None

class AgentHandoff(BaseModel):
    task_id: str; task_type: TaskType; payload: Dict[str, Any]
    from_agent: str; to_agent: str; priority: int = Field(default=1, ge=1, le=5)
    context_summary: Optional[str] = ""

class AgentResult(BaseModel):
    task_id: str; agent_name: str; status: AgentStatus
    output: Dict[str, Any]; confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    steps_taken: int = Field(default=0, ge=0); tools_used: List[str] = []
    error: Optional[str] = None; needs_human: bool = False

class BlackboardEntry(BaseModel):
    agent_name: str; task_type: TaskType; content: Dict[str, Any]
    confidence: float = 0.5; timestamp: str; is_final: bool = False

class TraceStep(BaseModel):
    step_number: int; agent_name: str; action: str
    tool: Optional[str] = None; tool_input: Optional[str] = None
    observation: Optional[str] = None; thought: Optional[str] = None
    latency_ms: float = 0.0; cost_usd: float = 0.0; timestamp: str

class EvaluationResult(BaseModel):
    task_success: bool; success_score: float; trajectory_score: float
    tool_accuracy: float; judge_score: float; judge_feedback: str; overall_score: float

class PineconeQueryConfig(BaseModel):
    k: int = Field(default=5, ge=1, le=20); search_type: SearchType = SearchType.SIMILARITY
    score_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    fetch_k: int = Field(default=20, ge=1, le=100); lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    namespace: Optional[str] = None; filter: Optional[Dict] = None; use_mmr: bool = False