from __future__ import annotations

from typing import Set

from pydantic import BaseModel, ConfigDict, Field, field_validator

from models import DecisionType


class PolicySnapshot(BaseModel):
    """
    Tex policy configuration for outbound email review.

    This file defines how Tex interprets findings and semantic scores.
    It does not perform detection itself.
    It does not perform semantic analysis itself.
    It only decides what those results mean under policy.
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    version: str = Field(..., min_length=1, max_length=50)

    # Semantic score thresholds
    forbid_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    abstain_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

    # Deterministic finding rules
    auto_forbid_finding_types: Set[str] = Field(
        default_factory=lambda: {
            "SSN",
            "Credit Card",
            "API Key Pattern",
            "Internal URL",
            "Sensitive Customer Reference",
            "Unauthorized Pricing Disclosure",
        }
    )

    force_abstain_finding_types: Set[str] = Field(
        default_factory=lambda: {
            "Multiple Price Points",
            "Unverified Compliance Claim",
            "Unverified Accuracy Claim",
            "Aggressive Tone Phrase",
            "Weekend / Personal Contact Language",
            "Recipient Authorization Unclear",
        }
    )

    # Optional business controls
    allow_weekend_send_language: bool = False
    allow_named_customer_reference: bool = False
    allow_discount_language_without_approval: bool = False

    # Safety rule: if semantic analysis fails completely, what should Tex do?
    semantic_failure_default: DecisionType = DecisionType.ABSTAIN

    @field_validator("semantic_failure_default")
    @classmethod
    def validate_semantic_failure_default(cls, value: DecisionType) -> DecisionType:
        if value not in {DecisionType.ABSTAIN, DecisionType.FORBID}:
            raise ValueError("semantic_failure_default must be ABSTAIN or FORBID")
        return value

    def model_post_init(self, __context) -> None:
        if self.abstain_threshold >= self.forbid_threshold:
            raise ValueError(
                "abstain_threshold must be lower than forbid_threshold"
            )


def get_default_policy() -> PolicySnapshot:
    """
    Returns the default Tex v1 policy.
    Keep this explicit for now.
    Later, this can load from JSON, YAML, or a database.
    """
    return PolicySnapshot(version="tex-v1")