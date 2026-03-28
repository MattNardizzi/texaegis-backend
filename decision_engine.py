from __future__ import annotations

from typing import List, Optional

from models import (
    DecisionReason,
    DecisionType,
    DeterministicFinding,
    EvaluationDimension,
    RiskLevel,
    SemanticEvaluation,
    TexDecision,
)
from policy import PolicySnapshot


class DecisionEngine:
    """
    Tex's final authority layer.

    Inputs:
    - deterministic findings
    - semantic evaluation
    - policy snapshot

    Output:
    - one final TexDecision

    Rules:
    1. Hard deterministic forbids take priority.
    2. Elevated deterministic findings can force at least ABSTAIN.
    3. Semantic scores are then evaluated against policy thresholds.
    4. If semantic analysis is unavailable, fallback to policy.semantic_failure_default.
    """

    def decide(
        self,
        deterministic_findings: List[DeterministicFinding],
        semantic_evaluation: Optional[SemanticEvaluation],
        policy: PolicySnapshot,
    ) -> TexDecision:
        reasons: List[DecisionReason] = []

        hard_forbid_reasons = self._build_hard_forbid_reasons(
            deterministic_findings=deterministic_findings,
            policy=policy,
        )
        if hard_forbid_reasons:
            reasons.extend(hard_forbid_reasons)
            summary = self._build_summary(
                decision=DecisionType.FORBID,
                deterministic_findings=deterministic_findings,
                semantic_evaluation=semantic_evaluation,
                policy=policy,
            )
            return TexDecision(
                decision=DecisionType.FORBID,
                reasons=reasons,
                summary=summary,
            )

        force_abstain_reasons = self._build_force_abstain_reasons(
            deterministic_findings=deterministic_findings,
            policy=policy,
        )
        reasons.extend(force_abstain_reasons)

        if semantic_evaluation is None:
            reasons.append(
                DecisionReason(
                    source="policy",
                    message=(
                        "Semantic analysis was unavailable. "
                        f"Applying policy fallback: {policy.semantic_failure_default.value}."
                    ),
                    evidence=None,
                )
            )

            summary = self._build_summary(
                decision=policy.semantic_failure_default,
                deterministic_findings=deterministic_findings,
                semantic_evaluation=None,
                policy=policy,
            )
            return TexDecision(
                decision=policy.semantic_failure_default,
                reasons=reasons,
                summary=summary,
            )

        semantic_forbid_reasons = self._build_semantic_forbid_reasons(
            semantic_evaluation=semantic_evaluation,
            policy=policy,
        )
        if semantic_forbid_reasons:
            reasons.extend(semantic_forbid_reasons)
            summary = self._build_summary(
                decision=DecisionType.FORBID,
                deterministic_findings=deterministic_findings,
                semantic_evaluation=semantic_evaluation,
                policy=policy,
            )
            return TexDecision(
                decision=DecisionType.FORBID,
                reasons=reasons,
                summary=summary,
            )

        semantic_abstain_reasons = self._build_semantic_abstain_reasons(
            semantic_evaluation=semantic_evaluation,
            policy=policy,
        )
        reasons.extend(semantic_abstain_reasons)

        if force_abstain_reasons or semantic_abstain_reasons:
            summary = self._build_summary(
                decision=DecisionType.ABSTAIN,
                deterministic_findings=deterministic_findings,
                semantic_evaluation=semantic_evaluation,
                policy=policy,
            )
            return TexDecision(
                decision=DecisionType.ABSTAIN,
                reasons=reasons,
                summary=summary,
            )

        reasons.append(
            DecisionReason(
                source="policy",
                message=(
                    "No deterministic auto-blocks were triggered and all semantic "
                    "dimension scores remained below the abstain threshold."
                ),
                evidence=None,
            )
        )

        summary = self._build_summary(
            decision=DecisionType.PERMIT,
            deterministic_findings=deterministic_findings,
            semantic_evaluation=semantic_evaluation,
            policy=policy,
        )
        return TexDecision(
            decision=DecisionType.PERMIT,
            reasons=reasons,
            summary=summary,
        )

    def _build_hard_forbid_reasons(
        self,
        deterministic_findings: List[DeterministicFinding],
        policy: PolicySnapshot,
    ) -> List[DecisionReason]:
        reasons: List[DecisionReason] = []

        for finding in deterministic_findings:
            if (
                finding.type in policy.auto_forbid_finding_types
                or finding.severity == RiskLevel.CRITICAL
            ):
                reasons.append(
                    DecisionReason(
                        source="deterministic",
                        message=(
                            f"Hard-rule violation detected: {finding.type}. "
                            f"{finding.message}"
                        ),
                        evidence=self._join_matches(finding.matches),
                    )
                )

        return reasons

    def _build_force_abstain_reasons(
        self,
        deterministic_findings: List[DeterministicFinding],
        policy: PolicySnapshot,
    ) -> List[DecisionReason]:
        reasons: List[DecisionReason] = []

        for finding in deterministic_findings:
            if finding.type in policy.force_abstain_finding_types:
                reasons.append(
                    DecisionReason(
                        source="deterministic",
                        message=(
                            f"Deterministic review flag detected: {finding.type}. "
                            f"{finding.message}"
                        ),
                        evidence=self._join_matches(finding.matches),
                    )
                )

        return reasons

    def _build_semantic_forbid_reasons(
        self,
        semantic_evaluation: SemanticEvaluation,
        policy: PolicySnapshot,
    ) -> List[DecisionReason]:
        reasons: List[DecisionReason] = []

        for dimension, result in semantic_evaluation.dimensions.items():
            if result.score >= policy.forbid_threshold:
                reasons.append(
                    DecisionReason(
                        source="semantic",
                        dimension=dimension,
                        message=(
                            f"{self._pretty_dimension_name(dimension)} scored "
                            f"{result.score:.2f}, which meets or exceeds the forbid "
                            f"threshold of {policy.forbid_threshold:.2f}. "
                            f"{result.finding}"
                        ),
                        evidence=result.evidence,
                    )
                )

        return reasons

    def _build_semantic_abstain_reasons(
        self,
        semantic_evaluation: SemanticEvaluation,
        policy: PolicySnapshot,
    ) -> List[DecisionReason]:
        reasons: List[DecisionReason] = []

        for dimension, result in semantic_evaluation.dimensions.items():
            if policy.abstain_threshold <= result.score < policy.forbid_threshold:
                reasons.append(
                    DecisionReason(
                        source="semantic",
                        dimension=dimension,
                        message=(
                            f"{self._pretty_dimension_name(dimension)} scored "
                            f"{result.score:.2f}, which meets or exceeds the abstain "
                            f"threshold of {policy.abstain_threshold:.2f}. "
                            f"{result.finding}"
                        ),
                        evidence=result.evidence,
                    )
                )

        return reasons

    @staticmethod
    def _pretty_dimension_name(dimension: EvaluationDimension) -> str:
        mapping = {
            EvaluationDimension.FACTUAL_ACCURACY: "Factual accuracy",
            EvaluationDimension.DATA_LEAKAGE: "Data leakage",
            EvaluationDimension.TONE_APPROPRIATENESS: "Tone appropriateness",
            EvaluationDimension.POLICY_COMPLIANCE: "Policy compliance",
            EvaluationDimension.RECIPIENT_AUTHORIZATION: "Recipient authorization",
        }
        return mapping.get(dimension, dimension.value)

    @staticmethod
    def _join_matches(matches: List[str]) -> Optional[str]:
        if not matches:
            return None
        return " | ".join(matches)

    def _build_summary(
        self,
        decision: DecisionType,
        deterministic_findings: List[DeterministicFinding],
        semantic_evaluation: Optional[SemanticEvaluation],
        policy: PolicySnapshot,
    ) -> str:
        if decision == DecisionType.FORBID:
            if deterministic_findings:
                hard_forbid_types = [
                    finding.type
                    for finding in deterministic_findings
                    if (
                        finding.type in policy.auto_forbid_finding_types
                        or finding.severity == RiskLevel.CRITICAL
                    )
                ]
                if hard_forbid_types:
                    unique_types = self._unique_preserve_order(hard_forbid_types)
                    return (
                        "Tex blocked this message because hard-rule violations were "
                        f"detected: {', '.join(unique_types)}."
                    )

            if semantic_evaluation:
                highest = self._highest_semantic_dimension(semantic_evaluation)
                if highest is not None:
                    dimension, score = highest
                    return (
                        "Tex blocked this message because "
                        f"{self._pretty_dimension_name(dimension).lower()} reached "
                        f"a critical risk score ({score:.2f})."
                    )

            return "Tex blocked this message because it presents unacceptable outbound risk."

        if decision == DecisionType.ABSTAIN:
            if deterministic_findings:
                elevated_types = [
                    finding.type
                    for finding in deterministic_findings
                    if finding.severity == RiskLevel.ELEVATED
                ]
                if elevated_types:
                    unique_types = self._unique_preserve_order(elevated_types)
                    return (
                        "Tex withheld this message for human review because it triggered "
                        f"review flags: {', '.join(unique_types)}."
                    )

            if semantic_evaluation:
                highest = self._highest_semantic_dimension(semantic_evaluation)
                if highest is not None:
                    dimension, score = highest
                    return (
                        "Tex withheld this message for human review because "
                        f"{self._pretty_dimension_name(dimension).lower()} showed "
                        f"elevated risk ({score:.2f})."
                    )

            return "Tex withheld this message for human review due to uncertainty."

        return (
            "Tex permitted this message because no hard-rule violations were found "
            "and the evaluated risk remained below review thresholds."
        )

    @staticmethod
    def _highest_semantic_dimension(
        semantic_evaluation: SemanticEvaluation,
    ) -> Optional[tuple[EvaluationDimension, float]]:
        if not semantic_evaluation.dimensions:
            return None

        return max(
            (
                (dimension, result.score)
                for dimension, result in semantic_evaluation.dimensions.items()
            ),
            key=lambda item: item[1],
            default=None,
        )

    @staticmethod
    def _unique_preserve_order(items: List[str]) -> List[str]:
        seen = set()
        result: List[str] = []

        for item in items:
            if item in seen:
                continue
            seen.add(item)
            result.append(item)

        return result