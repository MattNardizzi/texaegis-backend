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
    Provider-agnostic semantic analyzer for Tex Aegis.

    The rest of Tex should not care which model provider is used.
    This class selects a provider at runtime and normalizes the result
    into a SemanticEvaluation.

    Supported providers in this file:
    - openai
    - anthropic

    Configuration:
    - TEX_LLM_PROVIDER: "openai" or "anthropic"
    - TEX_LLM_MODEL: provider-specific model name

    Provider credentials:
    - OPENAI_API_KEY
    - ANTHROPIC_API_KEY

    Notes:
    - If initialization fails, main.py should fall back to deterministic-only mode.
    - The model never has final decision authority. It only returns analysis.
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.provider = (provider or os.getenv("TEX_LLM_PROVIDER", "openai")).strip().lower()
        self.model = model or os.getenv("TEX_LLM_MODEL") or self._default_model_for_provider(self.provider)
        self.api_key = api_key or self._api_key_from_env(self.provider)

        if self.provider not in {"openai", "anthropic"}:
            raise LLMAnalyzerError(
                f"Unsupported TEX_LLM_PROVIDER '{self.provider}'. "
                "Supported providers: openai, anthropic."
            )

        if not self.api_key:
            raise LLMAnalyzerError(
                f"Missing API key for provider '{self.provider}'. "
                f"Expected env var: {self._api_key_env_name(self.provider)}"
            )

    def analyze(self, email: OutboundEmail) -> SemanticEvaluation:
        """
        Run semantic analysis on one outbound email and return normalized results.
        """
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(email)

        if self.provider == "openai":
            raw_data = self._analyze_with_openai(system_prompt=system_prompt, user_prompt=user_prompt)
        elif self.provider == "anthropic":
            raw_data = self._analyze_with_anthropic(system_prompt=system_prompt, user_prompt=user_prompt)
        else:
            raise LLMAnalyzerError(f"Unsupported provider '{self.provider}'.")

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
        """
        Use OpenAI Responses API with structured JSON schema output.
        """
        try:
            from openai import OpenAI
        except Exception as exc:
            raise LLMAnalyzerError(
                "OpenAI SDK is not installed, but TEX_LLM_PROVIDER is set to 'openai'."
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

    def _analyze_with_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        Use Anthropic Messages API and require JSON-only output.
        """
        try:
            from anthropic import Anthropic
        except Exception as exc:
            raise LLMAnalyzerError(
                "Anthropic SDK is not installed, but TEX_LLM_PROVIDER is set to 'anthropic'."
            ) from exc

        client = Anthropic(api_key=self.api_key)

        json_only_user_prompt = (
            f"{user_prompt}\n\n"
            "Return ONLY valid JSON matching this exact schema:\n"
            f"{json.dumps(self._response_json_schema(), ensure_ascii=False)}"
        )

        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=1500,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": json_only_user_prompt,
                    }
                ],
            )
        except Exception as exc:
            raise LLMAnalyzerError(f"Anthropic request failed: {exc}") from exc

        text_blocks: list[str] = []
        for block in getattr(response, "content", []):
            if getattr(block, "type", None) == "text":
                block_text = getattr(block, "text", "")
                if block_text:
                    text_blocks.append(block_text)

        raw_text = "\n".join(text_blocks).strip()
        if not raw_text:
            raise LLMAnalyzerError("Anthropic returned empty output.")

        return self._parse_json_text(raw_text, provider_name="Anthropic")

    @staticmethod
    def _default_model_for_provider(provider: str) -> str:
        defaults = {
            "openai": "gpt-4.1-mini",
            "anthropic": "claude-sonnet-4-5",
        }
        if provider not in defaults:
            raise LLMAnalyzerError(f"No default model configured for provider '{provider}'.")
        return defaults[provider]

    @staticmethod
    def _api_key_env_name(provider: str) -> str:
        env_names = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        if provider not in env_names:
            raise LLMAnalyzerError(f"No API key env mapping configured for provider '{provider}'.")
        return env_names[provider]

    def _api_key_from_env(self, provider: str) -> Optional[str]:
        return os.getenv(self._api_key_env_name(provider))

    @staticmethod
    def _parse_json_text(raw_text: str, provider_name: str) -> Dict[str, Any]:
        """
        Parse JSON from a provider response. Tries strict parse first.
        If the model wrapped JSON in extra text, extract the outermost object.
        """
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
            "Evaluate the email across exactly these dimensions:\n"
            "1. factual_accuracy\n"
            "2. data_leakage\n"
            "3. tone_appropriateness\n"
            "4. policy_compliance\n"
            "5. recipient_authorization\n\n"
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