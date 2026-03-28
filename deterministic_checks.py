from __future__ import annotations

import re
from typing import List, Pattern

from models import DeterministicFinding, OutboundEmail, RiskLevel
from policy import PolicySnapshot


class DeterministicChecker:
    """
    Runs hard-rule checks against an outbound email before any model-based analysis.

    These checks are intentionally explicit and explainable.
    They do not attempt semantic reasoning.
    They only catch concrete patterns and policy-triggering phrases.
    """

    SSN_PATTERN: Pattern[str] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
    CREDIT_CARD_PATTERN: Pattern[str] = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")
    PHONE_PATTERN: Pattern[str] = re.compile(r"\b(?:\(\d{3}\)\s?|\d{3}[-. ])\d{3}[-. ]\d{4}\b")
    API_KEY_PATTERN: Pattern[str] = re.compile(
    r"\b(?:sk|pk|api)(?:[-_][a-zA-Z0-9]+){2,}\b",
    re.IGNORECASE,
    )
    INTERNAL_URL_PATTERN: Pattern[str] = re.compile(
        r"https?://(?:internal|staging|dev|admin)\.[^\s]+",
        re.IGNORECASE,
    )
    PRICE_PATTERN: Pattern[str] = re.compile(
        r"\$[\d,]+(?:\.\d{2})?(?:/(?:year|month|yr|mo))?",
        re.IGNORECASE,
    )
    PERCENT_PATTERN: Pattern[str] = re.compile(
        r"\b\d+(?:\.\d+)?%",
        re.IGNORECASE,
    )

    # These are simple v1 phrase checks.
    # Keep them explicit and easy to understand.
    AGGRESSIVE_TONE_PHRASES = [
        "don't want you to miss out",
        "let's make this happen",
        "shoot me a text",
        "even weekends work for me",
        "super excited",
        "awesome team",
        "jump in",
        "what we've got cooking",
        "lol",
    ]

    COMPLIANCE_CLAIM_PHRASES = [
        "soc 2 compliant",
        "soc 2 compliance monitoring",
        "gdpr compliant",
        "gdpr data residency",
        "hipaa compliant",
        "iso 27001 certified",
        "fully compliant",
        "certified",
    ]

    ACCURACY_CLAIM_PHRASES = [
        "accuracy",
        "forecast",
        "predictive analytics",
        "guarantee",
        "guaranteed",
        "will cut that in half",
        "real-time",
        "automatically route customer data",
    ]

    CUSTOMER_REFERENCE_HINTS = [
        "they're on the same tier",
        "their account manager",
        "can vouch for the value",
        "another customer",
        "similar account",
        "meridian health",
    ]

    DISCOUNT_HINTS = [
        "discount",
        "same package at",
        "offer you",
        "special rate",
        "reduced price",
        "same tier",
    ]

    WEEKEND_SEND_HINTS = [
        "weekends work for me",
        "text me anytime",
        "shoot me a text",
    ]

    def run(self, email: OutboundEmail, policy: PolicySnapshot) -> List[DeterministicFinding]:
        """
        Run all deterministic checks and return structured findings.
        """
        findings: List[DeterministicFinding] = []
        body = email.body
        lowered_body = body.lower()

        findings.extend(self._check_ssn(body))
        findings.extend(self._check_credit_cards(body))
        findings.extend(self._check_api_keys(body))
        findings.extend(self._check_internal_urls(body))
        findings.extend(self._check_multiple_prices(body))
        findings.extend(self._check_unverified_compliance_claims(body, lowered_body))
        findings.extend(self._check_unverified_accuracy_claims(body, lowered_body))
        findings.extend(self._check_aggressive_tone(lowered_body))
        findings.extend(self._check_customer_references(body, lowered_body, policy))
        findings.extend(self._check_discount_language(body, lowered_body, policy))
        findings.extend(self._check_weekend_language(lowered_body, policy))

        return findings

    def _check_ssn(self, body: str) -> List[DeterministicFinding]:
        matches = self.SSN_PATTERN.findall(body)
        if not matches:
            return []

        return [
            DeterministicFinding(
                type="SSN",
                severity=RiskLevel.CRITICAL,
                matches=self._unique(matches),
                message="Detected Social Security number pattern in outbound content.",
            )
        ]

    def _check_credit_cards(self, body: str) -> List[DeterministicFinding]:
        matches = self.CREDIT_CARD_PATTERN.findall(body)
        if not matches:
            return []

        return [
            DeterministicFinding(
                type="Credit Card",
                severity=RiskLevel.CRITICAL,
                matches=self._unique(matches),
                message="Detected credit card number pattern in outbound content.",
            )
        ]

    def _check_api_keys(self, body: str) -> List[DeterministicFinding]:
        matches = self.API_KEY_PATTERN.findall(body)
        if not matches:
            return []

        return [
            DeterministicFinding(
                type="API Key Pattern",
                severity=RiskLevel.CRITICAL,
                matches=self._unique(matches),
                message="Detected potential API key or secret token in outbound content.",
            )
        ]

    def _check_internal_urls(self, body: str) -> List[DeterministicFinding]:
        matches = self.INTERNAL_URL_PATTERN.findall(body)
        if not matches:
            return []

        return [
            DeterministicFinding(
                type="Internal URL",
                severity=RiskLevel.CRITICAL,
                matches=self._unique(matches),
                message="Detected internal or non-public URL in outbound content.",
            )
        ]

    def _check_multiple_prices(self, body: str) -> List[DeterministicFinding]:
        matches = self.PRICE_PATTERN.findall(body)
        unique_prices = self._unique(matches)

        if len(unique_prices) < 2:
            return []

        return [
            DeterministicFinding(
                type="Multiple Price Points",
                severity=RiskLevel.ELEVATED,
                matches=unique_prices,
                message="Detected multiple price points in one message, which may indicate pricing leakage or unauthorized discounting.",
            )
        ]

    def _check_unverified_compliance_claims(
        self,
        body: str,
        lowered_body: str,
    ) -> List[DeterministicFinding]:
        matched_phrases = [
            phrase for phrase in self.COMPLIANCE_CLAIM_PHRASES if phrase in lowered_body
        ]
        if not matched_phrases:
            return []

        evidence = self._extract_snippets(body, matched_phrases)
        return [
            DeterministicFinding(
                type="Unverified Compliance Claim",
                severity=RiskLevel.ELEVATED,
                matches=evidence,
                message="Detected compliance or certification language that should be verified before sending.",
            )
        ]

    def _check_unverified_accuracy_claims(
        self,
        body: str,
        lowered_body: str,
    ) -> List[DeterministicFinding]:
        matched_phrases = [
            phrase for phrase in self.ACCURACY_CLAIM_PHRASES if phrase in lowered_body
        ]

        percent_matches = self.PERCENT_PATTERN.findall(body)
        evidence = self._extract_snippets(body, matched_phrases)
        evidence.extend(percent_matches)

        if not evidence:
            return []

        return [
            DeterministicFinding(
                type="Unverified Accuracy Claim",
                severity=RiskLevel.ELEVATED,
                matches=self._unique(evidence),
                message="Detected performance, accuracy, or outcome claims that may require validation.",
            )
        ]

    def _check_aggressive_tone(self, lowered_body: str) -> List[DeterministicFinding]:
        matched_phrases = [
            phrase for phrase in self.AGGRESSIVE_TONE_PHRASES if phrase in lowered_body
        ]
        if not matched_phrases:
            return []

        return [
            DeterministicFinding(
                type="Aggressive Tone Phrase",
                severity=RiskLevel.ELEVATED,
                matches=self._unique(matched_phrases),
                message="Detected phrases that may be too informal, pushy, or inappropriate for professional outbound communication.",
            )
        ]

    def _check_customer_references(
        self,
        body: str,
        lowered_body: str,
        policy: PolicySnapshot,
    ) -> List[DeterministicFinding]:
        if policy.allow_named_customer_reference:
            return []

        matched_phrases = [
            phrase for phrase in self.CUSTOMER_REFERENCE_HINTS if phrase in lowered_body
        ]
        if not matched_phrases:
            return []

        evidence = self._extract_snippets(body, matched_phrases)
        return [
            DeterministicFinding(
                type="Sensitive Customer Reference",
                severity=RiskLevel.CRITICAL,
                matches=evidence,
                message="Detected possible disclosure of another customer's identity, relationship, or commercial terms.",
            )
        ]

    def _check_discount_language(
        self,
        body: str,
        lowered_body: str,
        policy: PolicySnapshot,
    ) -> List[DeterministicFinding]:
        if policy.allow_discount_language_without_approval:
            return []

        matched_phrases = [
            phrase for phrase in self.DISCOUNT_HINTS if phrase in lowered_body
        ]
        if not matched_phrases:
            return []

        evidence = self._extract_snippets(body, matched_phrases)
        return [
            DeterministicFinding(
                type="Unauthorized Pricing Disclosure",
                severity=RiskLevel.CRITICAL,
                matches=evidence,
                message="Detected discounting or pricing exception language that may require approval before sending.",
            )
        ]

    def _check_weekend_language(
        self,
        lowered_body: str,
        policy: PolicySnapshot,
    ) -> List[DeterministicFinding]:
        if policy.allow_weekend_send_language:
            return []

        matched_phrases = [
            phrase for phrase in self.WEEKEND_SEND_HINTS if phrase in lowered_body
        ]
        if not matched_phrases:
            return []

        return [
            DeterministicFinding(
                type="Weekend / Personal Contact Language",
                severity=RiskLevel.ELEVATED,
                matches=self._unique(matched_phrases),
                message="Detected personal-contact or weekend-availability language that may be inappropriate for outbound messaging policy.",
            )
        ]

    @staticmethod
    def _unique(items: List[str]) -> List[str]:
        """
        Preserve order while removing duplicates and blank values.
        """
        seen = set()
        result: List[str] = []

        for item in items:
            cleaned = item.strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            result.append(cleaned)

        return result

    @staticmethod
    def _extract_snippets(body: str, phrases: List[str], radius: int = 45) -> List[str]:
        """
        Pull small body snippets around matched phrases for better evidence display.
        """
        snippets: List[str] = []
        lowered_body = body.lower()

        for phrase in phrases:
            start_index = lowered_body.find(phrase.lower())
            if start_index == -1:
                continue

            snippet_start = max(0, start_index - radius)
            snippet_end = min(len(body), start_index + len(phrase) + radius)
            snippet = body[snippet_start:snippet_end].strip()

            if snippet_start > 0:
                snippet = "..." + snippet
            if snippet_end < len(body):
                snippet = snippet + "..."

            snippets.append(snippet)

        return DeterministicChecker._unique(snippets)