from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class DecisionType(str, Enum):
    PERMIT = "PERMIT"
    ABSTAIN = "ABSTAIN"
    FORBID = "FORBID"


class RiskLevel(str, Enum):
    CLEAR = "CLEAR"
    ELEVATED = "ELEVATED"
    CRITICAL = "CRITICAL"


class EvaluationDimension(str, Enum):
    FACTUAL_ACCURACY = "factual_accuracy"
    DATA_LEAKAGE = "data_leakage"
    TONE_APPROPRIATENESS = "tone_appropriateness"
    POLICY_COMPLIANCE = "policy_compliance"
    RECIPIENT_AUTHORIZATION = "recipient_authorization"


class OutboundEmail(BaseModel):
    """
    The message Tex is evaluating before it is allowed to go out.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    subject: str = Field(..., min_length=1, max_length=300)
    recipient: EmailStr
    sender: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=20000)

    @field_validator("subject", "sender", "body")
    @classmethod
    def must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Field cannot be blank.")
        return value


class DeterministicFinding(BaseModel):
    """
    A hard-rule or pattern-based issue Tex found without using a model.
    Example: SSN pattern, multiple prices, internal URL, named customer leakage.
    """

    type: str = Field(..., min_length=1, max_length=100)
    severity: RiskLevel
    matches: List[str] = Field(default_factory=list)
    message: str = Field(..., min_length=1, max_length=500)


class DimensionResult(BaseModel):
    """
    One semantic evaluation category returned by the analyzer.
    score:
        0.0 = no risk
        1.0 = critical risk
    """

    score: float = Field(..., ge=0.0, le=1.0)
    finding: str = Field(..., min_length=1, max_length=1000)
    evidence: str = Field(..., min_length=1, max_length=1000)


class SemanticEvaluation(BaseModel):
    """
    The model-assisted evaluation payload.
    This is analysis only.
    The model does not have final authority.
    """

    dimensions: Dict[EvaluationDimension, DimensionResult]
    summary: str = Field(..., min_length=1, max_length=1000)

    @field_validator("dimensions")
    @classmethod
    def validate_all_dimensions_present(
        cls,
        value: Dict[EvaluationDimension, DimensionResult],
    ) -> Dict[EvaluationDimension, DimensionResult]:
        required_dimensions = set(EvaluationDimension)
        provided_dimensions = set(value.keys())

        missing = required_dimensions - provided_dimensions
        if missing:
            missing_names = ", ".join(d.value for d in sorted(missing, key=lambda x: x.value))
            raise ValueError(f"Missing required dimensions: {missing_names}")

        return value


class DecisionReason(BaseModel):
    """
    A normalized reason Tex can return to the UI or audit log.
    """

    source: Literal["deterministic", "semantic", "policy"]
    dimension: Optional[EvaluationDimension] = None
    message: str = Field(..., min_length=1, max_length=1000)
    evidence: Optional[str] = Field(default=None, max_length=1000)


class TexDecision(BaseModel):
    """
    Tex's final decision.
    This is the only authority that matters to downstream systems.
    """

    decision: DecisionType
    reasons: List[DecisionReason] = Field(default_factory=list)
    summary: str = Field(..., min_length=1, max_length=1000)


class EvaluationResponse(BaseModel):
    """
    Full response returned after Tex evaluates an outbound message.
    """

    email: OutboundEmail
    deterministic_findings: List[DeterministicFinding] = Field(default_factory=list)
    semantic_evaluation: Optional[SemanticEvaluation] = None
    final_decision: TexDecision