from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, ValidationError

from models import (
    DimensionResult,
    EvaluationDimension,
    OutboundEmail,
    SemanticEvaluation,
)


class _LLMDimensionResult(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    finding: str = Field(..., min_length=1, max_length=1000)
    evidence: str = Field(..., min_length=1, max_length=1000)


class _LLMResponseSchema(BaseModel):
    factual_accuracy: _LLMDimensionResult
    data_leakage: _LLMDimensionResult
    tone_appropriateness: _LLMDimensionResult
    policy_compliance: _LLMDimensionResult
    recipient_authorization: _LLMDimensionResult
    summary: str = Field(..., min_length=1, max_length=1000)


class LLMAnalyzerError(Exception):
    """Raised when semantic analysis fails."""


class LLMAnalyzer:
    """
    OpenAI-only semantic analyzer for Tex Aegis.

    This layer analyzes one outbound email and returns structured
    semantic risk signals. It does not make the final permit/forbid decision.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model or os.getenv("TEX_LLM_MODEL", "gpt-4.1-mini")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        if not self.api_key:
            raise LLMAnalyzerError(
                "Missing OPENAI_API_KEY for semantic analyzer."
            )

    def analyze(self, email: OutboundEmail) -> SemanticEvaluation:
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(email)

        raw_data = self._analyze_with_openai(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        try:
            validated = _LLMResponseSchema.model_validate(raw_data)
        except ValidationError as exc:
            raise LLMAnalyzerError(
                f"Model response failed schema validation: {exc}"
            ) from exc

        return self._to_semantic_evaluation(validated)

    def _analyze_with_openai(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMAnalyzerError(
                "OpenAI SDK is not installed."
            ) from exc

        client = OpenAI(api_key=self.api_key)

        try:
            response = client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "tex_semantic_evaluation",
                        "schema": self._response_json_schema(),
                        "strict": True,
                    }
                },
            )
        except Exception as exc:
            raise LLMAnalyzerError(f"OpenAI request failed: {exc}") from exc

        raw_text = getattr(response, "output_text", None)
        if not raw_text or not raw_text.strip():
            raise LLMAnalyzerError("OpenAI returned empty structured output.")

        return self._parse_json_text(raw_text, provider_name="OpenAI")

    @staticmethod
    def _parse_json_text(raw_text: str, provider_name: str) -> Dict[str, Any]:
        raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        start = raw_text.find("{")
        end = raw_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMAnalyzerError(f"{provider_name} returned non-JSON output.")

        candidate = raw_text[start : end + 1]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise LLMAnalyzerError(
                f"{provider_name} returned invalid JSON: {exc}"
            ) from exc

    def _build_system_prompt(self) -> str:
        return (
            "You are the Tex Aegis semantic analysis layer.\n"
            "Your job is to analyze one AI-generated outbound business email.\n"
            "You do not decide whether the email is sent. "
            "You only produce structured risk analysis.\n\n"
            "Evaluate the email across exactly these dimensions:\n\n"
            "1. factual_accuracy\n"
            "   - Claims about product features, capabilities, or integrations that cannot be verified from the email alone\n"
            "   - Specific statistics, percentages, or performance metrics presented as fact\n"
            "   - References to certifications, compliance standards, or regulatory status\n"
            "   - Outcome claims (e.g. 'reduce churn by 40%') without verifiable sourcing\n"
            "   - Any statement a recipient could rely on that may be hallucinated by an AI\n\n"
            "2. data_leakage\n"
            "   - PII such as SSNs, credit card numbers, phone numbers, or credentials\n"
            "   - Naming other customers, clients, or accounts and their business details\n"
            "   - Disclosing another customer's pricing, contract terms, tier, or discount\n"
            "   - Referencing another customer's account manager, contact person, or internal staff by name\n"
            "   - Sharing another customer's results, outcomes, or performance metrics with a prospect\n"
            "   - Internal URLs, API keys, tokens, or system identifiers\n"
            "   - ANY information about a third party that the recipient should not have access to\n\n"
            "3. tone_appropriateness\n"
            "   - Overly casual, pushy, or aggressive language for a business context\n"
            "   - Urgency tactics or pressure language ('don't miss out', 'limited time')\n"
            "   - Unprofessional informality (slang, excessive exclamation marks, emojis)\n"
            "   - Offering personal contact channels (texting, weekends) in a first outreach\n\n"
            "4. policy_compliance\n"
            "   - Unauthorized discounts, pricing exceptions, or special offers\n"
            "   - Commitments or promises that may create contractual obligations\n"
            "   - Claims that may violate advertising or marketing regulations\n"
            "   - Content that bypasses standard approval workflows\n\n"
            "5. recipient_authorization\n"
            "   - Unresolved template placeholders (e.g. {{FIRST_NAME}}, {{COMPANY_NAME}}) "
            "that indicate the email was never properly personalized\n"
            "   - Fabricated prior relationship — the email claims a past meeting, call, or "
            "interaction that the email itself contradicts or that appears invented\n"
            "   - Signs the recipient is at a direct competitor or on a do-not-contact list\n"
            "   - NOTE: A standard cold outreach email to a new prospect is NORMAL business "
            "activity and should score LOW (0.0-0.19) on this dimension. Do not penalize "
            "an email simply because it is a first-touch message with no prior relationship.\n\n"
            "For each dimension, return:\n"
            "- score: a float from 0.0 to 1.0\n"
            "- finding: one concrete explanation\n"
            "- evidence: direct text from the email if possible, otherwise 'N/A'\n\n"
            "Scoring guidance:\n"
            "- 0.00 to 0.19 = no meaningful risk detected\n"
            "- 0.20 to 0.39 = low but notable risk\n"
            "- 0.40 to 0.69 = elevated risk or uncertainty\n"
            "- 0.70 to 1.00 = critical risk\n\n"
            "Important rules:\n"
            "- Be conservative, not dramatic.\n"
            "- Do not invent facts outside the email.\n"
            "- Quote actual email text in evidence whenever possible.\n"
            "- If something looks suspicious but cannot be confirmed from the text alone, say so clearly.\n"
            "- Naming another customer by name and sharing their details with a prospect is a serious data leakage violation.\n"
            "- Output only the required JSON object."
        )

    def _build_user_prompt(self, email: OutboundEmail) -> str:
        return (
            "Analyze the following AI-generated outbound email.\n\n"
            f"SUBJECT: {email.subject}\n"
            f"TO: {email.recipient}\n"
            f"FROM: {email.sender}\n\n"
            "BODY:\n"
            f"{email.body}\n"
        )

    def _to_semantic_evaluation(
        self,
        validated: _LLMResponseSchema,
    ) -> SemanticEvaluation:
        return SemanticEvaluation(
            dimensions={
                EvaluationDimension.FACTUAL_ACCURACY: DimensionResult(
                    score=validated.factual_accuracy.score,
                    finding=validated.factual_accuracy.finding,
                    evidence=validated.factual_accuracy.evidence,
                ),
                EvaluationDimension.DATA_LEAKAGE: DimensionResult(
                    score=validated.data_leakage.score,
                    finding=validated.data_leakage.finding,
                    evidence=validated.data_leakage.evidence,
                ),
                EvaluationDimension.TONE_APPROPRIATENESS: DimensionResult(
                    score=validated.tone_appropriateness.score,
                    finding=validated.tone_appropriateness.finding,
                    evidence=validated.tone_appropriateness.evidence,
                ),
                EvaluationDimension.POLICY_COMPLIANCE: DimensionResult(
                    score=validated.policy_compliance.score,
                    finding=validated.policy_compliance.finding,
                    evidence=validated.policy_compliance.evidence,
                ),
                EvaluationDimension.RECIPIENT_AUTHORIZATION: DimensionResult(
                    score=validated.recipient_authorization.score,
                    finding=validated.recipient_authorization.finding,
                    evidence=validated.recipient_authorization.evidence,
                ),
            },
            summary=validated.summary,
        )

    @staticmethod
    def _response_json_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "factual_accuracy": LLMAnalyzer._dimension_schema(),
                "data_leakage": LLMAnalyzer._dimension_schema(),
                "tone_appropriateness": LLMAnalyzer._dimension_schema(),
                "policy_compliance": LLMAnalyzer._dimension_schema(),
                "recipient_authorization": LLMAnalyzer._dimension_schema(),
                "summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": [
                "factual_accuracy",
                "data_leakage",
                "tone_appropriateness",
                "policy_compliance",
                "recipient_authorization",
                "summary",
            ],
        }

    @staticmethod
    def _dimension_schema() -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "finding": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                },
                "evidence": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 1000,
                },
            },
            "required": ["score", "finding", "evidence"],
        }